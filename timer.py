"""Latency profiler for the plan-execute workflow.

Measures wall-clock time at each phase:
  1. Discovery  — spawning MCP servers and listing tools
  2. Planning   — LLM call to decompose the question into steps
  3. Execution  — per step: MCP tool call + optional LLM arg resolution
  4. Summary    — LLM call to synthesise the final answer

Usage:
    PYTHONPATH=src uv run python timer.py "What assets are at site MAIN?"
    PYTHONPATH=src uv run python timer.py --runs 3 "What assets are at site MAIN?"
    PYTHONPATH=src uv run python timer.py --model-id watsonx/ibm/granite-3-3-8b-instruct "..."

Profiling modes (pick one):
  Baseline (no cache):  timer.py --skip-summary "..."
  Query-level semantic cache (pre-planner short-circuit):
    Lightweight (difflib, no heavy deps):
                          timer.py --skip-summary --query-cache "..."
    Full Asteria paper stack (Qwen embeddings + reranker judger + Sine):
                          timer.py --skip-summary --full-asteria "..."

Tool-call-level caching was removed on purpose: tool args are already
structured JSON, so a semantic cache there adds no value over exact match
and the MCP call itself. Semantic reuse only matters at the question level.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class StepTiming:
    step_number: int
    server: str
    task: str
    tool: str
    llm_resolve_s: float = 0.0   # time spent resolving tool args via LLM
    tool_call_s: float = 0.0     # time spent in the MCP tool call
    total_s: float = 0.0
    success: bool = True


@dataclass
class RunTiming:
    question: str
    discovery_s: float = 0.0
    planning_s: float = 0.0
    steps: list[StepTiming] = field(default_factory=list)
    summarization_s: float = 0.0
    total_s: float = 0.0
    query_cache_summary: dict[str, object] | None = None
    query_cache_hit: bool = False
    query_cache_mode: str = ""
    asteria_query_hit: bool = False
    asteria_summary: dict[str, object] | None = None

    @property
    def execution_s(self) -> float:
        return sum(s.total_s for s in self.steps)


# ── instrumented runner ───────────────────────────────────────────────────────

class ProfiledRunner:
    """Wraps PlanExecuteRunner and injects timing at each phase boundary."""

    def __init__(
        self,
        model_id: str,
        server_paths: dict | None = None,
        summarize: bool = True,
        summary_max_chars: int = 12000,
        step_response_max_chars: int = 3000,
        query_cache_enabled: bool = False,
        query_cache_threshold: float = 0.92,
        query_cache_ttl_seconds: float = 1800.0,
        full_asteria: bool = False,
    ) -> None:
        from llm.litellm import LiteLLMBackend
        from agent.plan_execute.executor import (
            Executor,
            DEFAULT_SERVER_PATHS,
            _resolve_args_with_llm,
            _call_tool,
            _list_tools,
        )
        from agent.plan_execute.planner import Planner

        self._model_id = model_id
        self._llm = LiteLLMBackend(model_id=model_id)
        self._server_paths = server_paths or DEFAULT_SERVER_PATHS
        self._planner = Planner(self._llm)
        self._executor = Executor(self._llm, self._server_paths)
        self._query_cache = None
        self._asteria_cache = None
        self._summarize = summarize
        self._summary_max_chars = summary_max_chars
        self._step_response_max_chars = step_response_max_chars

        # stash references to internal helpers for timed calls
        self._resolve_args_with_llm = _resolve_args_with_llm
        self._call_tool = _call_tool
        self._list_tools = _list_tools

        if full_asteria:
            from asteria.integrations.assetops.full_asteria_adapter import (
                build_asteria_cache_stack,
            )

            self._asteria_cache = build_asteria_cache_stack()
        elif query_cache_enabled:
            from asteria.integrations.assetops import QueryIntentCache

            self._query_cache = QueryIntentCache(
                semantic_threshold=query_cache_threshold,
                ttl_seconds=query_cache_ttl_seconds,
            )

    async def run(self, question: str) -> RunTiming:
        from agent.plan_execute.models import StepResult

        timing = RunTiming(question=question)
        run_start = time.perf_counter()

        # ── 0a. Full Asteria query-level lookup (Sine + judger) ───────────────
        if self._asteria_cache is not None:
            from asteria.integrations.assetops.query_intent_cache import classify_query
            _tclass, _ = classify_query(question)
            if _tclass != "VOLATILE":
                cached_ans, _dbg = self._asteria_cache.lookup(question)
                timing.asteria_summary = self._asteria_cache.stats_summary()
                if cached_ans is not None:
                    timing.asteria_query_hit = True
                    timing.discovery_s = 0.0
                    timing.planning_s = 0.0
                    timing.summarization_s = 0.0
                    timing.steps = []
                    timing.total_s = time.perf_counter() - run_start
                    return timing

        # ── 0b. Query-intent cache (before planner) ───────────────────────────
        if self._query_cache is not None:
            hit, cached_answer = self._query_cache.lookup(question)
            timing.query_cache_hit = hit
            timing.query_cache_mode = str(self._query_cache.last_event.get("mode", ""))
            timing.query_cache_summary = self._query_cache.summary()
            if hit and cached_answer is not None:
                timing.discovery_s = 0.0
                timing.planning_s = 0.0
                timing.summarization_s = 0.0
                timing.steps = []
                timing.total_s = time.perf_counter() - run_start
                return timing

        # ── 1. Discovery ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        server_descriptions = await self._executor.get_server_descriptions()
        timing.discovery_s = time.perf_counter() - t0

        # ── 2. Planning ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self._planner.generate_plan(question, server_descriptions)
        timing.planning_s = time.perf_counter() - t0

        # ── 3. Execution (step by step) ───────────────────────────────────────
        ordered = plan.resolved_order()
        context: dict[int, StepResult] = {}
        tool_schemas: dict[str, dict[str, str]] = {}

        # Match executor behavior: fetch tool schemas once for arg resolution.
        server_names = {step.server for step in ordered}
        for name in server_names:
            path = self._server_paths.get(name)
            if path is None:
                continue
            try:
                tools = await self._list_tools(path)
                tool_schemas[name] = {
                    t["name"]: ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    for t in tools
                }
            except Exception:  # noqa: BLE001
                tool_schemas[name] = {}

        for step in ordered:
            step_start = time.perf_counter()
            st = StepTiming(
                step_number=step.step_number,
                server=step.server,
                task=step.task,
                tool=step.tool or "none",
            )

            server_path = self._server_paths.get(step.server)
            if server_path is None or not step.tool or step.tool.lower() in ("none", "null"):
                # no tool call — record zero times
                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response=step.expected_output,
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
                st.total_s = time.perf_counter() - step_start
                context[step.step_number] = result
                timing.steps.append(st)
                continue

            try:
                tool_schema = tool_schemas.get(step.server, {}).get(step.tool, "")
                t_llm = time.perf_counter()
                resolved_args = await self._resolve_args_with_llm(
                    question,
                    step.task,
                    step.tool,
                    tool_schema,
                    context,
                    self._llm,
                )
                st.llm_resolve_s = time.perf_counter() - t_llm

                t_tool = time.perf_counter()
                response = await self._call_tool(server_path, step.tool, resolved_args)
                st.tool_call_s = time.perf_counter() - t_tool

                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response=response,
                    tool=step.tool,
                    tool_args=resolved_args,
                )
            except Exception as exc:  # noqa: BLE001
                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response="",
                    error=str(exc),
                    tool=step.tool,
                    tool_args=step.tool_args,
                )
                st.success = False

            st.total_s = time.perf_counter() - step_start
            context[step.step_number] = result
            timing.steps.append(st)

        # ── 4. Summarization ──────────────────────────────────────────────────
        from agent.plan_execute.runner import _SUMMARIZE_PROMPT

        if self._summarize:
            parts = []
            for r in context.values():
                detail = r.response if r.success else f"ERROR: {r.error}"
                if len(detail) > self._step_response_max_chars:
                    detail = detail[: self._step_response_max_chars] + "\n...[truncated]..."
                parts.append(
                    f"Step {r.step_number} — {r.task} (server: {r.server}):\n{detail}"
                )
            results_text = "\n\n".join(parts)
            if len(results_text) > self._summary_max_chars:
                results_text = (
                    results_text[: self._summary_max_chars] + "\n\n...[summary input truncated]..."
                )

            t0 = time.perf_counter()
            self._llm.generate(
                _SUMMARIZE_PROMPT.format(question=question, results=results_text)
            )
            timing.summarization_s = time.perf_counter() - t0
        else:
            timing.summarization_s = 0.0
        if self._query_cache is not None:
            from asteria.integrations.assetops.full_asteria_adapter import compose_stored_answer_from_steps
            answer = compose_stored_answer_from_steps(ordered, context)
            self._query_cache.store(question, answer)
            timing.query_cache_summary = self._query_cache.summary()

        timing.total_s = time.perf_counter() - run_start

        if self._asteria_cache is not None:
            from asteria.config import DEFAULT_CONFIG
            from asteria.integrations.assetops.full_asteria_adapter import compose_stored_answer_from_steps
            from asteria.integrations.assetops.query_intent_cache import classify_query as _cq
            _insert_tclass, _ = _cq(question)
            if not timing.asteria_query_hit and _insert_tclass != "VOLATILE":
                body = compose_stored_answer_from_steps(ordered, context)
                if body.strip():
                    self._asteria_cache.insert(
                        question,
                        body,
                        cost=DEFAULT_CONFIG.remote_cost_per_call,
                        latency_ms=timing.total_s * 1000.0,
                    )
            timing.asteria_summary = self._asteria_cache.stats_summary()

        return timing


# ── reporting ─────────────────────────────────────────────────────────────────

def _bar(value: float, total: float, width: int = 20) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_run(timing: RunTiming, run_index: int | None = None) -> None:
    label = f"Run {run_index}" if run_index is not None else "Result"
    print(f"\n{'═' * 62}")
    print(f"  {label}: {timing.question[:55]}")
    print(f"{'═' * 62}")

    rows = [
        ("Discovery",     timing.discovery_s),
        ("Planning (LLM)", timing.planning_s),
    ]
    for st in timing.steps:
        tag = f"Step {st.step_number} [{st.server}] {st.tool}"
        rows.append((tag, st.total_s))
        if st.llm_resolve_s > 0:
            rows.append((f"  └─ LLM resolve", st.llm_resolve_s))
        if st.tool_call_s > 0:
            rows.append((f"  └─ tool call",   st.tool_call_s))
    rows.append(("Summarization (LLM)", timing.summarization_s))

    col = 32
    for label, t in rows:
        bar = _bar(t, timing.total_s)
        print(f"  {label:<{col}} {t:6.3f}s  {bar}")

    print(f"  {'─' * (col + 30)}")
    print(f"  {'TOTAL':<{col}} {timing.total_s:6.3f}s")
    if timing.query_cache_summary:
        s = timing.query_cache_summary
        print(
            "  "
            + f"QueryCache: hit={timing.query_cache_hit} mode={timing.query_cache_mode or 'n/a'} "
            + f"hits={s.get('hits', 0)} misses={s.get('misses', 0)} "
            + f"rate={float(s.get('hit_rate', 0.0)):.1%}"
        )
    if timing.asteria_summary:
        s = timing.asteria_summary
        print(
            "  "
            + f"Asteria (full): query_hit={timing.asteria_query_hit} "
            + f"cache_hits={s.get('cache_hits', 0)} misses={s.get('cache_misses', 0)} "
            + f"hit_rate={float(s.get('hit_rate_%', 0.0)):.1f}% "
            + f"ses={s.get('ses_in_cache', 0)}"
        )


def print_summary(timings: list[RunTiming]) -> None:
    if len(timings) < 2:
        return

    print(f"\n{'═' * 62}")
    print(f"  Summary across {len(timings)} runs")
    print(f"{'═' * 62}")

    def stats(values: list[float]) -> str:
        mn, mx, avg = min(values), max(values), sum(values) / len(values)
        return f"avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s"

    col = 28
    print(f"  {'Phase':<{col}}  avg      min      max")
    print(f"  {'─' * 56}")

    phases = [
        ("Discovery",          [t.discovery_s for t in timings]),
        ("Planning (LLM)",     [t.planning_s for t in timings]),
        ("Execution (total)",  [t.execution_s for t in timings]),
        ("Summarization (LLM)",[t.summarization_s for t in timings]),
        ("TOTAL",              [t.total_s for t in timings]),
    ]
    for name, values in phases:
        print(f"  {name:<{col}}  {stats(values)}")
    q_cache_runs = [t.query_cache_summary for t in timings if t.query_cache_summary is not None]
    if q_cache_runs:
        hits = sum(int(s.get("hits", 0)) for s in q_cache_runs)
        misses = sum(int(s.get("misses", 0)) for s in q_cache_runs)
        total = hits + misses
        rate = hits / total if total else 0.0
        print(f"  {'QueryCache (aggregate)':<{col}}  hits={hits} misses={misses} rate={rate:.1%}")
    asteria_last = next((t.asteria_summary for t in reversed(timings) if t.asteria_summary), None)
    if asteria_last is not None:
        s = asteria_last
        print(
            f"  {'Asteria (full, final state)':<{col}}  "
            f"hits={s.get('cache_hits', 0)} misses={s.get('cache_misses', 0)} "
            f"hit_rate={float(s.get('hit_rate_%', 0.0)):.1f}% ses={s.get('ses_in_cache', 0)}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler",
        description="Phase-level latency profiler for the plan-execute workflow.",
    )
    parser.add_argument("question", help="The question to profile.")
    parser.add_argument(
        "--model-id",
        default="watsonx/meta-llama/llama-3-3-70b-instruct",
        metavar="MODEL_ID",
        help="LiteLLM model string (default: watsonx/meta-llama/llama-3-3-70b-instruct).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to run the query (default: 1). Use 3+ for stable averages.",
    )
    parser.add_argument(
        "--query-cache",
        action="store_true",
        help="Enable query-intent semantic cache before planning.",
    )
    parser.add_argument(
        "--query-cache-threshold",
        type=float,
        default=0.92,
        metavar="RATIO",
        help="Query-intent semantic threshold (default: 0.92).",
    )
    parser.add_argument(
        "--query-cache-ttl-seconds",
        type=float,
        default=1800.0,
        metavar="SECONDS",
        help="TTL for query-intent cache entries (default: 1800).",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip final LLM summarization (recommended for large history payloads).",
    )
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=12000,
        metavar="N",
        help="Maximum chars passed to summary prompt (default: 12000).",
    )
    parser.add_argument(
        "--step-response-max-chars",
        type=int,
        default=3000,
        metavar="N",
        help="Max chars per step response in summary prompt (default: 3000).",
    )
    parser.add_argument(
        "--full-asteria",
        action="store_true",
        help=(
            "Use paper Asteria stack (Qwen embeddings + reranker judger + Sine / AsteriaCache) "
            "for query-level caching. Incompatible with --query-cache. "
            "Requires torch, sentence-transformers, faiss-cpu, transformers."
        ),
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    if args.full_asteria and args.query_cache:
        raise SystemExit(
            "Use either --full-asteria or --query-cache, not both."
        )
    runner = ProfiledRunner(
        model_id=args.model_id,
        query_cache_enabled=args.query_cache,
        query_cache_threshold=args.query_cache_threshold,
        query_cache_ttl_seconds=args.query_cache_ttl_seconds,
        summarize=not args.skip_summary,
        summary_max_chars=args.summary_max_chars,
        step_response_max_chars=args.step_response_max_chars,
        full_asteria=args.full_asteria,
    )
    timings: list[RunTiming] = []

    for i in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\nRun {i}/{args.runs}...", flush=True)
        t = await runner.run(args.question)
        timings.append(t)
        print_run(t, run_index=i if args.runs > 1 else None)

    print_summary(timings)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
