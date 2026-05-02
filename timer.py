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

Profiling modes:
  Baseline (no cache):  timer.py --skip-summary "..."
  Asteria query cache (pre-orchestrator ANN + judger + temporal logic):
                        timer.py --skip-summary --asteria "..."

Tool-call-level caching was removed on purpose: tool args are already
structured JSON, so a semantic cache there adds no value over exact match
and the MCP call itself. Semantic reuse only matters at the question level.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sample_queries import filter_rows, load_rows, sample_rows




@dataclass(frozen=True)
class QueryCase:
    question: str
    metadata: dict[str, str] | None = None

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
    question_metadata: dict[str, str] | None = None
    asteria_lookup_s: float = 0.0
    discovery_s: float = 0.0
    planning_s: float = 0.0
    steps: list[StepTiming] = field(default_factory=list)
    summarization_s: float = 0.0
    asteria_insert_s: float = 0.0
    total_s: float = 0.0
    temporal_debug: dict[str, object] | None = None
    asteria_lookup_debug: dict[str, object] | None = None
    asteria_insert_debug: dict[str, object] | None = None
    asteria_hit: bool = False
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
        asteria_enabled: bool = False,
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
        self._asteria_cache = None
        self._summarize = summarize
        self._summary_max_chars = summary_max_chars
        self._step_response_max_chars = step_response_max_chars

        # stash references to internal helpers for timed calls
        self._resolve_args_with_llm = _resolve_args_with_llm
        self._call_tool = _call_tool
        self._list_tools = _list_tools

        if asteria_enabled:
            from asteria.integrations.assetops.full_asteria_adapter import (
                build_asteria_cache_stack,
            )

            self._asteria_cache = build_asteria_cache_stack()

            # Pre-pay PyTorch / Transformers JIT warmup on the judger now,
            # not during the first user query.  Without this, the first row
            # that produces ANN candidates is artificially charged ~30-50 s
            # of kernel warmup and skews bench latencies dramatically.
            try:
                self._asteria_cache.judger.score("warmup query", "warmup answer")
            except Exception:  # noqa: BLE001
                pass

    async def run(self, question: str, now=None) -> RunTiming:
        """
        Execute the plan-execute pipeline for `question`.

        Parameters
        ----------
        question : str
            The user query.
        now : datetime.datetime | None
            Simulated wall clock used by the temporal classifier when
            resolving relative phrases.  Forwarded to AsteriaCache.lookup
            and AsteriaCache.insert so seed and test runs over the same
            CSV row see identical resolved windows.  None → real wall clock.
        """
        from agent.plan_execute.models import StepResult

        timing = RunTiming(question=question)
        timing.temporal_debug = _describe_temporal_policy(question, now=now)
        run_start = time.perf_counter()

        # ── 0. Asteria query-level lookup (Sine + judger + temporal logic) ───
        if self._asteria_cache is not None:
            t0 = time.perf_counter()
            cached_answer, debug = self._asteria_cache.lookup(question, now=now)
            timing.asteria_lookup_s = time.perf_counter() - t0
            timing.asteria_lookup_debug = _annotate_lookup_debug(debug, timing.temporal_debug)
            timing.asteria_summary = self._asteria_cache.stats_summary()
            if cached_answer is not None:
                timing.asteria_hit = True
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
        if self._asteria_cache is not None:
            from asteria.config import DEFAULT_CONFIG
            from asteria.integrations.assetops.full_asteria_adapter import compose_stored_answer_from_steps
            if not timing.asteria_hit:
                body = compose_stored_answer_from_steps(ordered, context)
                if body.strip():
                    t0 = time.perf_counter()
                    timing.asteria_insert_debug = self._asteria_cache.insert(
                        question,
                        body,
                        cost=DEFAULT_CONFIG.remote_cost_per_call,
                        latency_ms=(time.perf_counter() - run_start) * 1000.0,
                        now=now,
                    )
                    timing.asteria_insert_s = time.perf_counter() - t0
            timing.asteria_summary = self._asteria_cache.stats_summary()
        if timing.asteria_lookup_debug is not None:
            timing.asteria_lookup_debug = _annotate_lookup_debug(
                timing.asteria_lookup_debug,
                timing.temporal_debug,
            )

        timing.total_s = time.perf_counter() - run_start
        return timing


# ── reporting ─────────────────────────────────────────────────────────────────

def _bar(value: float, total: float, width: int = 20) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _format_metadata(metadata: dict[str, str] | None) -> str | None:
    if not metadata:
        return None
    parts = []
    if metadata.get("id"):
        parts.append(f"id={metadata['id']}")
    if metadata.get("type"):
        parts.append(f"type={metadata['type']}")
    if metadata.get("category"):
        parts.append(f"category={metadata['category']}")
    if metadata.get("entity"):
        parts.append(f"entity={metadata['entity']}")
    if metadata.get("group"):
        parts.append(f"group={metadata['group']}")
    return " ".join(parts) if parts else None


def _describe_temporal_policy(question: str, now=None) -> dict[str, Any]:
    from asteria.temporal_classifier import classify as temporal_classify

    tag = temporal_classify(question, now=now)
    display_tag = tag.bucket.value  # VOLATILE / ANCHORED / STATIC

    policy = {
        "VOLATILE": "lookup bypassed; insert skipped",
        "ANCHORED": "lookup allowed; exact window match required; long TTL",
        "STATIC":   "lookup allowed; insert allowed; TTL from staticity",
        # legacy: classifier no longer returns RELATIVE, but stub caches may
        "RELATIVE": "(legacy) treated as ANCHORED",
    }[display_tag]

    out: dict[str, Any] = {
        "display_tag": display_tag,
        "asteria_bucket": tag.bucket.value,
        "asteria_bucket_name": tag.bucket.name,
        "cache_policy": policy,
    }
    if tag.time_window is not None:
        out["time_window"] = f"{tag.time_window.start} -> {tag.time_window.end}"
    return out


def _annotate_lookup_debug(
    debug: dict[str, object] | None,
    temporal_debug: dict[str, object] | None,
) -> dict[str, object] | None:
    if debug is None:
        return None
    merged = dict(debug)
    if temporal_debug is not None:
        merged.setdefault("temporal_display_tag", temporal_debug.get("display_tag"))
        merged.setdefault("temporal_policy", temporal_debug.get("cache_policy"))
        merged.setdefault("temporal_bucket_name", temporal_debug.get("asteria_bucket_name"))
        if temporal_debug.get("time_window") is not None:
            merged.setdefault("time_window", temporal_debug.get("time_window"))
    if merged.get("temporal_bypass"):
        merged["lookup_decision"] = "bypass"
    elif merged.get("hit"):
        merged["lookup_decision"] = "hit"
    else:
        merged["lookup_decision"] = "miss"
    return merged


def print_run(timing: RunTiming, run_index: int | None = None) -> None:
    label = f"Run {run_index}" if run_index is not None else "Result"
    print(f"\n{'═' * 62}")
    print(f"  {label}: {timing.question[:55]}")
    print(f"{'═' * 62}")
    meta = _format_metadata(timing.question_metadata)
    if meta:
        print(f"  Query Meta: {meta}")
    if timing.temporal_debug:
        td = timing.temporal_debug
        window = f" window={td['time_window']}" if td.get("time_window") else ""
        print(
            "  "
            + f"Temporal: tag={td.get('display_tag')} "
            + f"asteria_bucket={td.get('asteria_bucket')} "
            + f"policy={td.get('cache_policy')}{window}"
        )

    rows = [
        ("Asteria Lookup", timing.asteria_lookup_s),
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
    rows.append(("Asteria Insert", timing.asteria_insert_s))

    col = 32
    for label, t in rows:
        bar = _bar(t, timing.total_s)
        print(f"  {label:<{col}} {t:6.3f}s  {bar}")

    print(f"  {'─' * (col + 30)}")
    print(f"  {'TOTAL':<{col}} {timing.total_s:6.3f}s")
    if timing.asteria_summary:
        s = timing.asteria_summary
        dbg = timing.asteria_lookup_debug or {}
        print(
            "  "
            + f"Asteria: hit={timing.asteria_hit} "
            + f"display_tag={dbg.get('temporal_display_tag', 'n/a')} "
            + f"temporal_bucket={dbg.get('temporal_bucket', 'n/a')} "
            + f"temporal_bypass={dbg.get('temporal_bypass', False)} "
            + f"cache_hits={s.get('cache_hits', 0)} misses={s.get('cache_misses', 0)} "
            + f"hit_rate={float(s.get('hit_rate_%', 0.0)):.1f}% "
            + f"ses={s.get('ses_in_cache', 0)}"
        )
        print(
            "  "
            + f"Lookup Detail: decision={dbg.get('lookup_decision', 'n/a')} "
            + f"source={dbg.get('source', 'n/a')} "
            + f"ann_candidates={dbg.get('ann_candidates', 'n/a')} "
            + f"judger_scores={dbg.get('judger_scores', [])}"
        )
        if timing.asteria_insert_debug:
            ins = timing.asteria_insert_debug
            print(
                "  "
                + f"Insert Detail: inserted={ins.get('inserted', False)} "
                + f"skip_reason={ins.get('skip_reason', 'none')} "
                + f"staticity={ins.get('staticity', 'n/a')} "
                + f"ttl_hours={ins.get('ttl_hours', 'n/a')} "
                + f"temporal_bucket={ins.get('temporal_bucket', 'n/a')}"
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
        ("Asteria Lookup",     [t.asteria_lookup_s for t in timings]),
        ("Discovery",          [t.discovery_s for t in timings]),
        ("Planning (LLM)",     [t.planning_s for t in timings]),
        ("Execution (total)",  [t.execution_s for t in timings]),
        ("Summarization (LLM)",[t.summarization_s for t in timings]),
        ("Asteria Insert",     [t.asteria_insert_s for t in timings]),
        ("TOTAL",              [t.total_s for t in timings]),
    ]
    for name, values in phases:
        print(f"  {name:<{col}}  {stats(values)}")
    asteria_last = next((t.asteria_summary for t in reversed(timings) if t.asteria_summary), None)
    if asteria_last is not None:
        s = asteria_last
        print(
            f"  {'Asteria (final state)':<{col}}  "
            f"hits={s.get('cache_hits', 0)} misses={s.get('cache_misses', 0)} "
            f"hit_rate={float(s.get('hit_rate_%', 0.0)):.1f}% ses={s.get('ses_in_cache', 0)}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler",
        description="Phase-level latency profiler for the plan-execute workflow.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to profile. Omit this when using CSV sampling.",
    )
    parser.add_argument(
        "--now",
        default=None,
        metavar="ISO",
        help=(
            "Simulated wall clock for the temporal classifier "
            "(ISO 8601, e.g. 2024-08-15T02:01:51).  Drives "
            "RELATIVE→ANCHORED resolution in a reproducible way."
        ),
    )
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
        "--asteria",
        "--full-asteria",
        dest="asteria",
        action="store_true",
        help=(
            "Enable Asteria query-level caching before the plan-execute workflow "
            "(Qwen embeddings + reranker judger + Sine + temporal logic). "
            "Requires torch, sentence-transformers, faiss-cpu, transformers."
        ),
    )
    parser.add_argument(
        "--compare-asteria",
        action="store_true",
        help="Run the same query set twice: baseline first, then Asteria.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to a query CSV like all_utterance.csv.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        metavar="N",
        help="How many random queries to draw from --csv (default: 1).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        metavar="N",
        help="Optional seed for repeatable CSV sampling.",
    )
    parser.add_argument("--sample-type", dest="sample_type", help="CSV filter: type.")
    parser.add_argument(
        "--sample-category",
        dest="sample_category",
        help="CSV filter: category.",
    )
    parser.add_argument(
        "--sample-entity",
        dest="sample_entity",
        help="CSV filter: entity.",
    )
    parser.add_argument(
        "--sample-group",
        dest="sample_group",
        help="CSV filter: group.",
    )
    return parser


def _load_query_cases(args: argparse.Namespace) -> list[QueryCase]:
    if args.csv is not None:
        rows = load_rows(args.csv)
        matches = filter_rows(
            rows,
            type_filter=args.sample_type,
            category_filter=args.sample_category,
            entity_filter=args.sample_entity,
            group_filter=args.sample_group,
        )
        if not matches:
            raise SystemExit("No matching CSV queries found.")
        chosen = sample_rows(matches, count=args.sample_count, seed=args.sample_seed)
        return [
            QueryCase(
                question=row["text"],
                metadata={
                    "id": row.get("id", ""),
                    "type": row.get("type", ""),
                    "category": row.get("category", ""),
                    "entity": row.get("entity", ""),
                    "group": row.get("group", ""),
                },
            )
            for row in chosen
        ]
    return [QueryCase(question=args.question)]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.question and args.csv is not None:
        parser.error("Pass either a single question or --csv sampling, not both.")
    if not args.question and args.csv is None:
        parser.error("Pass a question or provide --csv for sampled runs.")
    if args.sample_count < 1:
        parser.error("--sample-count must be >= 1.")


async def _run_mode(
    args: argparse.Namespace,
    query_cases: list[QueryCase],
    *,
    mode_name: str,
    asteria_enabled: bool,
) -> None:
    runner = ProfiledRunner(
        model_id=args.model_id,
        summarize=not args.skip_summary,
        summary_max_chars=args.summary_max_chars,
        step_response_max_chars=args.step_response_max_chars,
        asteria_enabled=asteria_enabled,
    )
    timings: list[RunTiming] = []

    print(f"\n{'#' * 62}")
    print(f"  Mode: {mode_name}")
    print(f"{'#' * 62}")

    for case_index, query_case in enumerate(query_cases, start=1):
        if len(query_cases) > 1:
            print(f"\n{'─' * 62}")
            print(f"  Query {case_index}/{len(query_cases)}: {query_case.question}")
            meta = _format_metadata(query_case.metadata)
            if meta:
                print(f"  {meta}")
            print(f"{'─' * 62}")
        for i in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"\nRun {i}/{args.runs}...", flush=True)
            t = await runner.run(query_case.question, now=getattr(args, "_parsed_now", None))
            t.question_metadata = query_case.metadata
            timings.append(t)
            print_run(t, run_index=i if args.runs > 1 else None)

    print_summary(timings)


async def _main(args: argparse.Namespace) -> None:
    query_cases = _load_query_cases(args)
    modes = [("Baseline", False), ("Asteria", True)] if args.compare_asteria else [
        ("Asteria", True) if args.asteria else ("Baseline", False)
    ]
    for mode_name, asteria_enabled in modes:
        await _run_mode(
            args,
            query_cases,
            mode_name=mode_name,
            asteria_enabled=asteria_enabled,
        )


def main() -> None:
    import datetime as _dt
    from dotenv import load_dotenv
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    args._parsed_now = (
        _dt.datetime.fromisoformat(args.now) if args.now else None
    )
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
