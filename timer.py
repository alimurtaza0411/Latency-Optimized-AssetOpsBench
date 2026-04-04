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
    cache_hit: bool | None = None
    cache_mode: str = ""


@dataclass
class RunTiming:
    question: str
    discovery_s: float = 0.0
    planning_s: float = 0.0
    steps: list[StepTiming] = field(default_factory=list)
    summarization_s: float = 0.0
    total_s: float = 0.0
    cache_summary: dict[str, object] | None = None

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
        cache_enabled: bool = False,
        cache_semantic: bool = True,
        cache_semantic_threshold: float = 0.94,
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
        self._cache = None

        # stash references to internal helpers for timed calls
        self._resolve_args_with_llm = _resolve_args_with_llm
        self._call_tool = _call_tool
        self._list_tools = _list_tools

        if cache_enabled:
            from asteria.integrations.assetops import IoTToolCache, build_cached_call_tool

            self._cache = IoTToolCache(
                enable_semantic=cache_semantic,
                semantic_threshold=cache_semantic_threshold,
            )
            self._call_tool = build_cached_call_tool(self._call_tool, self._cache)

    async def run(self, question: str) -> RunTiming:
        from agent.plan_execute.models import StepResult

        timing = RunTiming(question=question)
        run_start = time.perf_counter()

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
                if self._cache is not None:
                    st.cache_hit = bool(self._cache.last_event.get("hit", False))
                    st.cache_mode = str(self._cache.last_event.get("mode", ""))

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

        results_text = "\n\n".join(
            f"Step {r.step_number} — {r.task} (server: {r.server}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in context.values()
        )

        t0 = time.perf_counter()
        self._llm.generate(
            _SUMMARIZE_PROMPT.format(question=question, results=results_text)
        )
        timing.summarization_s = time.perf_counter() - t0
        if self._cache is not None:
            timing.cache_summary = self._cache.summary()

        timing.total_s = time.perf_counter() - run_start
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
        cache_tag = ""
        if st.cache_hit is True:
            cache_tag = " [cache HIT]"
        elif st.cache_hit is False:
            cache_tag = " [cache MISS]"
        tag = f"Step {st.step_number} [{st.server}] {st.tool}{cache_tag}"
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
    if timing.cache_summary:
        s = timing.cache_summary
        print(
            "  "
            + f"Cache: hits={s.get('hits', 0)} misses={s.get('misses', 0)} "
            + f"hit_rate={float(s.get('hit_rate', 0.0)):.1%} entries={s.get('entries', 0)}"
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
    cache_runs = [t.cache_summary for t in timings if t.cache_summary is not None]
    if cache_runs:
        hits = sum(int(s.get("hits", 0)) for s in cache_runs)
        misses = sum(int(s.get("misses", 0)) for s in cache_runs)
        total = hits + misses
        rate = hits / total if total else 0.0
        print(f"  {'Cache (aggregate)':<{col}}  hits={hits} misses={misses} rate={rate:.1%}")


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
        "--cache",
        action="store_true",
        help="Enable Asteria IoT tool cache wrapper during profiling.",
    )
    parser.add_argument(
        "--cache-no-semantic",
        action="store_true",
        help="Disable semantic fallback in cache (exact-match only).",
    )
    parser.add_argument(
        "--cache-semantic-threshold",
        type=float,
        default=0.94,
        metavar="RATIO",
        help="Semantic similarity threshold (default: 0.94).",
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    runner = ProfiledRunner(
        model_id=args.model_id,
        cache_enabled=args.cache,
        cache_semantic=not args.cache_no_semantic,
        cache_semantic_threshold=args.cache_semantic_threshold,
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
