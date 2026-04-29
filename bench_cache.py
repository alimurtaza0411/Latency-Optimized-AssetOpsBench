"""
Cache benchmark — baseline vs Asteria over a two-stage paraphrase corpus.

Workflow:
  1. Seed pass: run each row of cache_seed.csv through a ProfiledRunner with
     Asteria enabled.  Cache is populated with real pipeline answers (subject
     to staticity gate / temporal-bucket policy).
  2. Sample pass: draw N rows from cache_test.csv via sample_queries helpers.
  3. Baseline pass: run sampled rows through a SECOND ProfiledRunner with
     Asteria disabled.  Records pure end-to-end latency.
  4. Cached pass: run the same sampled rows through the WARMED ProfiledRunner
     from step 1.  Records latency + hit rate.
  5. Print side-by-side summary.

The seed and test CSVs are expected to come from generate_scenarios.py with
DIFFERENT --seed values.  Shifted-anchored windows are deterministic per
parent_id, so seed-and-test pairs of shifted_anchored rows align by window.

Usage:
  PYTHONPATH=src:. uv run python bench_cache.py \\
      --seed-csv cache_seed.csv \\
      --test-csv cache_test.csv \\
      --sample-count 5

Common options:
  --skip-summary             Skip the LLM summarization step (faster).
  --seed-types IoT,Workorder Restrict seed pass to these types.
  --test-types IoT,Workorder Restrict sampling to these types.
  --sample-seed N            RNG seed for test sampling.
  --max-seed-rows N          Cap seed pass for fast smoke runs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv

from sample_queries import filter_rows, sample_rows
from timer import ProfiledRunner, RunTiming


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    row_id:        str
    parent_id:     str
    similarity:    str
    asteria_hit:   bool
    total_s:       float
    discovery_s:   float
    planning_s:    float
    execution_s:   float
    summarize_s:   float
    asteria_lookup_s: float
    asteria_insert_s: float


@dataclass
class PassSummary:
    label:    str
    runs:     int = 0
    hits:     int = 0
    totals:   list[float] = field(default_factory=list)
    by_id:    dict[str, SampleResult] = field(default_factory=dict)


# ── CSV helpers ──────────────────────────────────────────────────────────────

def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _filter_by_types(rows: list[dict], types_csv: str | None) -> list[dict]:
    if not types_csv:
        return rows
    keep = {t.strip() for t in types_csv.split(",")}
    return [r for r in rows if r.get("type") in keep]


# ── seed + sample passes ─────────────────────────────────────────────────────

async def _run_seed(
    runner: ProfiledRunner,
    rows: list[dict],
    *,
    label: str,
) -> None:
    print(f"\n[{label}] seeding {len(rows)} rows …")
    for i, row in enumerate(rows, start=1):
        text = row.get("text", "").strip()
        if not text:
            continue
        t = await runner.run(text)
        print(
            f"  {i:>3}/{len(rows)}  "
            f"id={row.get('id','?')}  parent_id={row.get('parent_id','-')}  "
            f"tier={row.get('similarity_tier','-')}  "
            f"hit={t.asteria_hit}  total={t.total_s:.2f}s",
            flush=True,
        )


async def _run_sample_pass(
    runner: ProfiledRunner,
    sampled: list[dict],
    *,
    label: str,
) -> PassSummary:
    out = PassSummary(label=label)
    print(f"\n[{label}] running {len(sampled)} sampled rows …")
    for i, row in enumerate(sampled, start=1):
        text = row.get("text", "").strip()
        if not text:
            continue
        t = await runner.run(text)
        sr = SampleResult(
            row_id=row.get("id", "?"),
            parent_id=row.get("parent_id", "-"),
            similarity=row.get("similarity_tier", "-"),
            asteria_hit=t.asteria_hit,
            total_s=t.total_s,
            discovery_s=t.discovery_s,
            planning_s=t.planning_s,
            execution_s=t.execution_s,
            summarize_s=t.summarization_s,
            asteria_lookup_s=t.asteria_lookup_s,
            asteria_insert_s=t.asteria_insert_s,
        )
        out.runs += 1
        out.totals.append(t.total_s)
        if t.asteria_hit:
            out.hits += 1
        out.by_id[sr.row_id] = sr
        print(
            f"  {i:>3}/{len(sampled)}  "
            f"id={sr.row_id}  parent_id={sr.parent_id}  tier={sr.similarity}  "
            f"hit={sr.asteria_hit}  total={t.total_s:.2f}s",
            flush=True,
        )
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"avg={mean(values):.2f}s  med={median(values):.2f}s  "
        f"min={min(values):.2f}s  max={max(values):.2f}s"
    )


def _print_compare(baseline: PassSummary, cached: PassSummary) -> None:
    print("\n" + "═" * 70)
    print("  Benchmark Summary")
    print("═" * 70)
    print(f"  baseline runs   : {baseline.runs}    {_stats(baseline.totals)}")
    print(f"  cached   runs   : {cached.runs}    {_stats(cached.totals)}")
    print(f"  cache hits      : {cached.hits}/{cached.runs}  "
          f"({100.0 * cached.hits / max(1, cached.runs):.1f}%)")

    if baseline.totals and cached.totals:
        b_avg = mean(baseline.totals)
        c_avg = mean(cached.totals)
        if c_avg > 0:
            speedup = b_avg / c_avg
            reduction = (b_avg - c_avg) / b_avg if b_avg > 0 else 0.0
            print(f"  speedup         : {speedup:.2f}x")
            print(f"  latency reduce  : {reduction * 100:.1f}%")

    # Per-row diff
    common = sorted(set(baseline.by_id) & set(cached.by_id))
    if common:
        print("\n  Per-row comparison (baseline → cached):")
        print(f"  {'id':<6s}  {'parent':<6s}  {'tier':<18s}  "
              f"{'baseline':>10s}  {'cached':>10s}  {'hit':>4s}")
        for rid in common:
            b = baseline.by_id[rid]
            c = cached.by_id[rid]
            print(
                f"  {rid:<6s}  {b.parent_id:<6s}  {b.similarity:<18s}  "
                f"{b.total_s:>9.2f}s  {c.total_s:>9.2f}s  "
                f"{('YES' if c.asteria_hit else 'no'):>4s}"
            )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_cache",
        description="Cache benchmark using paraphrase seed + test CSVs.",
    )
    p.add_argument("--seed-csv", type=Path, required=True,
                   help="CSV produced by generate_scenarios.py to pre-warm cache.")
    p.add_argument("--test-csv", type=Path, required=True,
                   help="CSV with paraphrases to sample for measurement.")
    p.add_argument("--sample-count", type=int, default=5,
                   help="Rows sampled from test CSV (default: 5).")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="RNG seed for sampling (default: 42).")
    p.add_argument("--seed-types", default=None,
                   help="Comma-separated types to include from seed CSV.")
    p.add_argument("--test-types", default=None,
                   help="Comma-separated types to include from test CSV.")
    p.add_argument("--max-seed-rows", type=int, default=None,
                   help="Cap seed pass for fast smoke runs.")
    p.add_argument("--model-id",
                   default="watsonx/meta-llama/llama-3-3-70b-instruct",
                   help="LiteLLM model id.")
    p.add_argument("--skip-summary", action="store_true",
                   help="Skip final LLM summarization in each request.")
    p.add_argument("--summary-max-chars", type=int, default=12000)
    p.add_argument("--step-response-max-chars", type=int, default=3000)
    return p


async def _main(args: argparse.Namespace) -> None:
    seed_rows = _filter_by_types(_load_rows(args.seed_csv), args.seed_types)
    test_rows = _filter_by_types(_load_rows(args.test_csv), args.test_types)

    if args.max_seed_rows:
        seed_rows = seed_rows[: args.max_seed_rows]

    if not seed_rows:
        sys.exit("No seed rows after filtering.")
    if not test_rows:
        sys.exit("No test rows after filtering.")

    sampled = sample_rows(test_rows, count=args.sample_count, seed=args.sample_seed)
    print(f"Seed rows         : {len(seed_rows)}")
    print(f"Test rows total   : {len(test_rows)}")
    print(f"Test rows sampled : {len(sampled)}")

    # Two runners — one for cached path (warmed), one for baseline.
    cached_runner = ProfiledRunner(
        model_id=args.model_id,
        summarize=not args.skip_summary,
        summary_max_chars=args.summary_max_chars,
        step_response_max_chars=args.step_response_max_chars,
        asteria_enabled=True,
    )
    baseline_runner = ProfiledRunner(
        model_id=args.model_id,
        summarize=not args.skip_summary,
        summary_max_chars=args.summary_max_chars,
        step_response_max_chars=args.step_response_max_chars,
        asteria_enabled=False,
    )

    t0 = time.perf_counter()
    await _run_seed(cached_runner, seed_rows, label="seed")
    print(f"\n[seed] done in {time.perf_counter() - t0:.1f}s")

    baseline_summary = await _run_sample_pass(baseline_runner, sampled, label="baseline")
    cached_summary   = await _run_sample_pass(cached_runner,   sampled, label="cached")

    _print_compare(baseline_summary, cached_summary)


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
