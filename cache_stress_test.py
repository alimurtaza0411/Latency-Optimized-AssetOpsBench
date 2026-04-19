"""Stress test semantic cache behavior with realistic query workloads.

This script is designed for fast iteration:
1) Edit QUERY_SCENARIOS below (add your own similar/paraphrased prompts).
2) Generate a Zipfian workload over those scenarios.
3) Run baseline (no cache) and/or cache-enabled profiling.
4) Compare latency and cache hit metrics.

Run examples:
  PYTHONPATH=src uv run python cache_stress_test.py --help
  PYTHONPATH=src uv run python cache_stress_test.py --mode both --requests 30
  PYTHONPATH=src uv run python cache_stress_test.py --mode cached --requests 50 --zipf-s 1.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv

from timer import ProfiledRunner, RunTiming


# ---------------------------------------------------------------------------
# EDIT THIS BLOCK: scenarios + paraphrases
# ---------------------------------------------------------------------------
#
# - name: identifier for reporting
# - variants: paraphrases that should map to similar tool behavior
# - rank: lower rank => more frequent under Zipf workload
#
QUERY_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "assets_main",
        "rank": 1,
        "variants": [
            "What assets are available at site MAIN?",
            "List all assets at the MAIN site.",
            "Show me the asset inventory for MAIN.",
            "Which assets exist in site MAIN?",
            "Give me every asset in MAIN.",
            "Can you enumerate assets at MAIN?",
            "Return the list of assets for site MAIN.",
        ],
    },
    {
        "name": "assets_main_rephrase2",
        "rank": 2,
        "variants": [
            "I need all equipment IDs at MAIN.",
            "Provide available asset IDs for MAIN site.",
            "What equipment is present in MAIN?",
        ],
    },
    {
        "name": "sensors_chiller6",
        "rank": 3,
        "variants": [
            "List sensors for Chiller 6 at site MAIN.",
            "What sensors does Chiller 6 have in MAIN?",
            "Show sensor names for Chiller 6 in site MAIN.",
            "Give sensor list for asset Chiller 6 at MAIN.",
            "Which telemetry fields are available for Chiller 6 in MAIN?",
        ],
    },
    {
        "name": "sensors_chiller6_rephrase2",
        "rank": 4,
        "variants": [
            "Fetch all sensor tags on Chiller 6 from site MAIN.",
            "What are the measured signals for Chiller 6 in MAIN?",
            "Show instrumentation variables for Chiller 6 at MAIN.",
        ],
    },
    {
        "name": "sensors_chiller6_typoish",
        "rank": 5,
        "variants": [
            "list sensor for chiller 6 main site",
            "chiller 6 sensor names at MAIN please",
            "MAIN chiller6 what sensors",
        ],
    },
    {
        "name": "assets_plus_sensors_composite",
        "rank": 6,
        "variants": [
            "First list assets at MAIN, then list sensors for Chiller 6.",
            "What assets are at MAIN and what sensors are on Chiller 6?",
            "Give me MAIN assets and Chiller 6 sensor names.",
        ],
    },
    {
        "name": "history_chiller6_window1",
        "rank": 7,
        "variants": [
            "Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN.",
            "Show Chiller 6 observations at MAIN between 2020-06-01T00:00:00 and 2020-06-01T01:00:00.",
            "Fetch historical readings for Chiller 6 in MAIN from 2020-06-01T00:00:00 to 2020-06-01T01:00:00.",
            "Return time-series data for Chiller 6 at MAIN for 2020-06-01 00:00 to 01:00.",
        ],
    },
    {
        "name": "history_chiller6_window2",
        "rank": 8,
        "variants": [
            "Get history for Chiller 6 from 2020-06-01T01:00:00 to 2020-06-01T02:00:00 at MAIN.",
            "Show Chiller 6 observations at MAIN between 2020-06-01T01:00:00 and 2020-06-01T02:00:00.",
            "Fetch historical readings for Chiller 6 in MAIN from 2020-06-01T01:00:00 to 2020-06-01T02:00:00.",
            "Return time-series data for Chiller 6 at MAIN for 2020-06-01 01:00 to 02:00.",
        ],
    },
    {
        "name": "history_chiller6_window3",
        "rank": 9,
        "variants": [
            "Get history for Chiller 6 from 2020-06-01T02:00:00 to 2020-06-01T03:00:00 at MAIN.",
            "Show Chiller 6 observations at MAIN between 2020-06-01T02:00:00 and 2020-06-01T03:00:00.",
            "Historical sensor data for Chiller 6 MAIN from 2am to 3am on 2020-06-01.",
        ],
    },
    {
        "name": "history_chiller6_open_ended",
        "rank": 10,
        "variants": [
            "Get history for Chiller 6 from 2020-06-01T00:00:00 at MAIN.",
            "Show Chiller 6 observations in MAIN starting 2020-06-01T00:00:00.",
            "Return all Chiller 6 records from 2020-06-01T00:00:00 onward in MAIN.",
        ],
    },
    {
        "name": "history_chiller6_typoish",
        "rank": 11,
        "variants": [
            "history chiller 6 main 2020-06-01T00:00:00 to 2020-06-01T01:00:00",
            "show data chiller6 MAIN between 00 and 01 2020-06-01",
            "need chiller 6 obs MAIN 2020-06-01T00:00:00 2020-06-01T01:00:00",
        ],
    },
    {
        "name": "history_chiller6_natural_date",
        "rank": 12,
        "variants": [
            "For Chiller 6 at MAIN, give readings from June 1, 2020 00:00 to 01:00.",
            "Chiller 6 MAIN data for 1 Jun 2020 between midnight and 1am.",
            "Pull Chiller 6 history in MAIN for 2020/06/01 00:00-01:00.",
        ],
    },
    {
        "name": "history_unknown_asset_control",
        "rank": 13,
        "variants": [
            "Get history for asset UNKNOWN_ASSET from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN.",
            "Show MAIN observations for UNKNOWN_ASSET between 2020-06-01T00:00:00 and 2020-06-01T01:00:00.",
        ],
    },
    {
        "name": "sensors_unknown_asset_control",
        "rank": 14,
        "variants": [
            "List sensors for UNKNOWN_ASSET at site MAIN.",
            "What sensors exist for UNKNOWN_ASSET in MAIN?",
        ],
    },
    {
        "name": "assets_invalid_site_control",
        "rank": 15,
        "variants": [
            "What assets are available at site INVALID?",
            "List all assets at INVALID site.",
            "Show asset inventory for site INVALID.",
        ],
    },
    {
        "name": "multi_turn_style_query",
        "rank": 16,
        "variants": [
            "Need a quick check: list MAIN assets. Also include sensors for Chiller 6.",
            "Please do two things: assets at MAIN and sensors on Chiller 6.",
            "Can you first list MAIN assets then tell me Chiller 6 sensors?",
        ],
    },
    {
        "name": "assets_main_short",
        "rank": 17,
        "variants": [
            "assets MAIN",
            "MAIN assets?",
            "list MAIN equipment",
        ],
    },
    {
        "name": "sensors_main_short",
        "rank": 18,
        "variants": [
            "sensors Chiller 6 MAIN",
            "Chiller6 MAIN sensors?",
            "MAIN Chiller 6 telemetry fields",
        ],
    },
    {
        "name": "history_main_short",
        "rank": 19,
        "variants": [
            "history Chiller 6 MAIN 2020-06-01T00:00:00 2020-06-01T01:00:00",
            "MAIN Chiller 6 data 00:00 to 01:00 on 2020-06-01",
            "Chiller6 MAIN obs 2020-06-01 00-01",
        ],
    },
    {
        "name": "history_chiller6_window4",
        "rank": 20,
        "variants": [
            "Get history for Chiller 6 from 2020-06-01T03:00:00 to 2020-06-01T04:00:00 at MAIN.",
            "Show Chiller 6 observations at MAIN between 2020-06-01T03:00:00 and 2020-06-01T04:00:00.",
            "Historical sensor data for Chiller 6 MAIN from 3am to 4am on 2020-06-01.",
        ],
    },
]


@dataclass
class WorkloadItem:
    scenario_name: str
    query: str


@dataclass
class RequestResult:
    scenario_name: str
    query: str
    total_s: float | None
    discovery_s: float | None
    planning_s: float | None
    execution_s: float | None
    summarization_s: float | None
    query_cache_hit: bool = False
    query_cache_mode: str = ""
    error: str | None = None


def _validate_scenarios(scenarios: list[dict[str, Any]]) -> None:
    if not scenarios:
        raise ValueError("QUERY_SCENARIOS is empty.")
    for s in scenarios:
        if "name" not in s or "variants" not in s or "rank" not in s:
            raise ValueError("Each scenario needs name, variants, rank.")
        if not s["variants"]:
            raise ValueError(f"Scenario {s['name']} has no variants.")


def _zipf_probabilities(ranks: list[int], zipf_s: float) -> list[float]:
    # P(i) proportional to 1 / rank^s
    weights = [1.0 / (float(r) ** zipf_s) for r in ranks]
    total = sum(weights)
    return [w / total for w in weights]


def build_zipf_workload(
    scenarios: list[dict[str, Any]],
    requests: int,
    zipf_s: float,
    seed: int,
) -> list[WorkloadItem]:
    rnd = random.Random(seed)
    ordered = sorted(scenarios, key=lambda x: int(x["rank"]))
    probs = _zipf_probabilities([int(s["rank"]) for s in ordered], zipf_s)

    out: list[WorkloadItem] = []
    for _ in range(requests):
        picked = rnd.choices(ordered, weights=probs, k=1)[0]
        query = rnd.choice(picked["variants"])
        out.append(WorkloadItem(scenario_name=picked["name"], query=query))
    return out


async def run_workload(
    workload: list[WorkloadItem],
    model_id: str,
    include_summary: bool,
    summary_max_chars: int,
    step_response_max_chars: int,
    server_paths: dict[str, str] | None,
    query_cache_enabled: bool,
    query_cache_threshold: float,
    query_cache_ttl_seconds: float,
    llm_retries: int,
    retry_delay_s: float,
    continue_on_error: bool,
) -> list[RequestResult]:
    runner = ProfiledRunner(
        model_id=model_id,
        server_paths=server_paths,
        query_cache_enabled=query_cache_enabled,
        query_cache_threshold=query_cache_threshold,
        query_cache_ttl_seconds=query_cache_ttl_seconds,
        summarize=include_summary,
        summary_max_chars=summary_max_chars,
        step_response_max_chars=step_response_max_chars,
    )
    results: list[RequestResult] = []

    for item in workload:
        attempt = 0
        last_error: str | None = None
        while attempt <= llm_retries:
            try:
                timing = await runner.run(item.query)
                results.append(
                    RequestResult(
                        scenario_name=item.scenario_name,
                        query=item.query,
                        total_s=timing.total_s,
                        discovery_s=timing.discovery_s,
                        planning_s=timing.planning_s,
                        execution_s=timing.execution_s,
                        summarization_s=timing.summarization_s,
                        query_cache_hit=timing.query_cache_hit,
                        query_cache_mode=timing.query_cache_mode,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                attempt += 1
                if attempt <= llm_retries:
                    await asyncio.sleep(retry_delay_s)
                    continue
                if continue_on_error:
                    results.append(
                        RequestResult(
                            scenario_name=item.scenario_name,
                            query=item.query,
                            total_s=None,
                            discovery_s=None,
                            planning_s=None,
                            execution_s=None,
                            summarization_s=None,
                            error=last_error,
                        )
                    )
                    break
                raise
    return results


def summarize(label: str, results: list[RequestResult]) -> dict[str, Any]:
    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    totals = [r.total_s for r in successes if r.total_s is not None]
    execs = [r.execution_s for r in successes if r.execution_s is not None]
    query_hits = sum(1 for r in results if r.query_cache_hit)
    return {
        "label": label,
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "total_avg_s": mean(totals) if totals else 0.0,
        "total_median_s": median(totals) if totals else 0.0,
        "total_p95_s": sorted(totals)[int(0.95 * (len(totals) - 1))] if totals else 0.0,
        "execution_avg_s": mean(execs) if execs else 0.0,
        "query_cache_hits": query_hits,
        "query_cache_hit_rate": (query_hits / len(results)) if results else 0.0,
    }


def print_summary_table(summary: dict[str, Any]) -> None:
    print(f"\n[{summary['label']}]")
    print(f"  requests       : {summary['requests']}")
    print(f"  successful     : {summary['successful_requests']}")
    print(f"  failed         : {summary['failed_requests']}")
    print(f"  total avg (s)  : {summary['total_avg_s']:.3f}")
    print(f"  total median(s): {summary['total_median_s']:.3f}")
    print(f"  total p95 (s)  : {summary['total_p95_s']:.3f}")
    print(f"  exec avg (s)   : {summary['execution_avg_s']:.3f}")
    print(f"  query hits     : {summary['query_cache_hits']}")
    print(f"  query hit rate : {summary['query_cache_hit_rate']:.1%}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zipfian stress test for Asteria cache.")
    p.add_argument(
        "--model-id",
        default="watsonx/meta-llama/llama-3-3-70b-instruct",
        help="LLM model id for planner/executor/summarizer.",
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "cached", "both"],
        default="both",
        help="Run baseline only, query-cache only, or both for comparison.",
    )
    p.add_argument("--requests", type=int, default=30, help="Number of requests in workload.")
    p.add_argument("--zipf-s", type=float, default=1.1, help="Zipf exponent.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--query-cache-threshold",
        type=float,
        default=0.92,
        help="Similarity threshold for query-intent cache.",
    )
    p.add_argument(
        "--query-cache-ttl-seconds",
        type=float,
        default=1800.0,
        help="TTL for query-intent cache entries.",
    )
    p.add_argument(
        "--out-json",
        default="tmp/cache_stress_report.json",
        help="Write full report JSON to this path.",
    )
    p.add_argument(
        "--include-summary",
        action="store_true",
        help="Include final LLM summarization in each profiled request (off by default).",
    )
    p.add_argument(
        "--summary-max-chars",
        type=int,
        default=12000,
        help="Maximum chars fed to summary prompt if --include-summary is set.",
    )
    p.add_argument(
        "--step-response-max-chars",
        type=int,
        default=3000,
        help="Max chars per step response for summary prompt if enabled.",
    )
    p.add_argument(
        "--iot-only",
        action="store_true",
        help="Use only IoT server during planning/execution (recommended for IoT cache tests).",
    )
    p.add_argument(
        "--llm-retries",
        type=int,
        default=1,
        help="Retries per request when planner/tool LLM call fails (default: 1).",
    )
    p.add_argument(
        "--retry-delay-s",
        type=float,
        default=1.5,
        help="Delay between retries in seconds (default: 1.5).",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue workload on request failure and record failed request stats.",
    )
    return p


async def _main(args: argparse.Namespace) -> None:
    _validate_scenarios(QUERY_SCENARIOS)
    workload = build_zipf_workload(
        QUERY_SCENARIOS,
        requests=args.requests,
        zipf_s=args.zipf_s,
        seed=args.seed,
    )

    print("\nGenerated workload sample (first 10):")
    for i, w in enumerate(workload[:10], start=1):
        print(f"  {i:02d}. [{w.scenario_name}] {w.query}")

    payload: dict[str, Any] = {
        "config": {
            "mode": args.mode,
            "requests": args.requests,
            "zipf_s": args.zipf_s,
            "seed": args.seed,
            "model_id": args.model_id,
            "query_cache_threshold": args.query_cache_threshold,
            "query_cache_ttl_seconds": args.query_cache_ttl_seconds,
            "include_summary": args.include_summary,
            "summary_max_chars": args.summary_max_chars,
            "step_response_max_chars": args.step_response_max_chars,
            "iot_only": args.iot_only,
            "llm_retries": args.llm_retries,
            "retry_delay_s": args.retry_delay_s,
            "continue_on_error": args.continue_on_error,
        },
        "workload": [w.__dict__ for w in workload],
        "results": {},
        "comparison": {},
    }

    baseline_summary = None
    cached_summary = None
    server_paths = {"iot": "iot-mcp-server"} if args.iot_only else None

    if args.mode in {"baseline", "both"}:
        baseline_results = await run_workload(
            workload=workload,
            model_id=args.model_id,
            query_cache_enabled=False,
            query_cache_threshold=args.query_cache_threshold,
            query_cache_ttl_seconds=args.query_cache_ttl_seconds,
            include_summary=args.include_summary,
            summary_max_chars=args.summary_max_chars,
            step_response_max_chars=args.step_response_max_chars,
            server_paths=server_paths,
            llm_retries=args.llm_retries,
            retry_delay_s=args.retry_delay_s,
            continue_on_error=args.continue_on_error,
        )
        baseline_summary = summarize("baseline", baseline_results)
        payload["results"]["baseline"] = {
            "summary": baseline_summary,
            "requests": [r.__dict__ for r in baseline_results],
        }
        print_summary_table(baseline_summary)

    if args.mode in {"cached", "both"}:
        cached_results = await run_workload(
            workload=workload,
            model_id=args.model_id,
            query_cache_enabled=True,
            query_cache_threshold=args.query_cache_threshold,
            query_cache_ttl_seconds=args.query_cache_ttl_seconds,
            include_summary=args.include_summary,
            summary_max_chars=args.summary_max_chars,
            step_response_max_chars=args.step_response_max_chars,
            server_paths=server_paths,
            llm_retries=args.llm_retries,
            retry_delay_s=args.retry_delay_s,
            continue_on_error=args.continue_on_error,
        )
        cached_summary = summarize("cached", cached_results)
        payload["results"]["cached"] = {
            "summary": cached_summary,
            "requests": [r.__dict__ for r in cached_results],
        }
        print_summary_table(cached_summary)

    if baseline_summary and cached_summary:
        base = float(baseline_summary["total_avg_s"])
        cached = float(cached_summary["total_avg_s"])
        speedup = (base / cached) if cached > 0 else 0.0
        reduction = ((base - cached) / base) if base > 0 else 0.0
        comp = {
            "avg_speedup_x": speedup,
            "avg_latency_reduction": reduction,
        }
        payload["comparison"] = comp
        print("\n[comparison]")
        print(f"  avg speedup (x)       : {speedup:.2f}")
        print(f"  avg latency reduction : {reduction:.1%}")

    write_json(Path(args.out_json), payload)
    print(f"\nWrote report: {args.out_json}")


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
