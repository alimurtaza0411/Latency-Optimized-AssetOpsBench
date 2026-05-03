"""
Cache benchmark — baseline vs Asteria over a two-stage paraphrase corpus.

Workflow:
  1. Seed pass: run each row of cache_seed.csv through a ProfiledRunner with
     Asteria enabled.  Cache is populated with real pipeline answers (subject
     to staticity gate / temporal-bucket policy).  Use --max-seed-rows only
     for quick smoke runs; omit it to run the full seed CSV.
  2. Test pass: by default **every** row of cache_test.csv (after --test-types
     filter).  Pass --sample-count N to randomly subsample instead.
  3. Baseline pass: run the chosen test rows with Asteria off.
  4. Cached pass: run the same rows with the warmed Asteria runner.
  5. Print side-by-side summary and (by default) an Asteria reliability table:
     TP/TN/FP/FN vs seed parent_id, plus precision / recall / F1 / specificity.
     Latency speedup is summarized on **cache hits only** (paired baseline vs
     cached); misses report mean extra latency vs baseline.

Ground truth for the matrix (see --confusion-mode):
  intent — y=1 if test row's parent_id appears in any *executed* seed row.
  strict — y=1 only if that parent had ≥1 successful Asteria insert during seed.

Usage (full seed + full test — no subsampling):
  PYTHONPATH=src:. uv run python bench_cache.py \\
      --seed-csv cache_seed.csv \\
      --test-csv cache_test.csv

Optional random subsample of test rows only:
  ... --sample-count 50 --sample-seed 42

Disable the reliability table with --no-confusion-matrix.

Common options:
  --skip-summary             Skip the LLM summarization step (faster).
  --seed-types IoT,Workorder Restrict seed pass to these types.
  --test-types IoT,Workorder Restrict test CSV to these types.
  --sample-count N           Randomly take N test rows (omit = use all).
  --sample-seed N            RNG for --sample-count (default: 42 if sampling).
  --max-seed-rows N          Cap seed pass for fast smoke runs.
  --confusion-matrix         TP/TN/FP/FN + precision/recall (on by default).
  --no-confusion-matrix      Skip reliability table.
  --debug                    Per row: scenario ids, temporal bucket/window, Asteria hit/miss.
  --verbose                  Full Asteria decision trail (use with --debug).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as _dt
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

from dotenv import load_dotenv

from sample_queries import sample_rows
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


@dataclass
class SeedStats:
    """parent_id sets from the Asteria-on seed pass (one runner, fresh cache)."""

    parents_executed: set[str] = field(default_factory=set)
    parents_warmed: set[str] = field(default_factory=set)


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


def _parse_run_at(row: dict) -> Optional[_dt.datetime]:
    """Read query_run_at from a CSV row.  Returns None if absent or invalid."""
    raw = (row.get("query_run_at") or "").strip()
    if not raw:
        return None
    try:
        return _dt.datetime.fromisoformat(raw)
    except ValueError:
        return None


def _print_scenario_debug(
    row: dict,
    run_at: Optional[_dt.datetime],
    timing: RunTiming,
    *,
    phase: str,
) -> None:
    """One block: which scenario ran, temporal bucket, Asteria outcome."""
    pfx = f"  [debug:{phase}]"
    q = (row.get("text") or "").strip()
    snip = q[:140] + ("…" if len(q) > 140 else "")
    print(
        f"{pfx} scenario row_id={row.get('id')} parent_id={row.get('parent_id')} "
        f"type={row.get('type')} tier={row.get('similarity_tier')}",
        flush=True,
    )
    print(f"{pfx} text     {snip!r}", flush=True)
    print(
        f"{pfx} now      {run_at.isoformat() if run_at else 'wall_clock'}",
        flush=True,
    )

    td = timing.temporal_debug or {}
    print(
        f"{pfx} temporal bucket={td.get('display_tag', '?')} "
        f"enum={td.get('asteria_bucket', '?')} "
        f"window={td.get('time_window', '—')}",
        flush=True,
    )

    dbg = timing.asteria_lookup_debug
    if dbg is None:
        print(f"{pfx} asteria  (lookup skipped — baseline / Asteria off)", flush=True)
    else:
        scores = dbg.get("judger_scores") or []
        best = f"{max(scores):.3f}" if scores else "—"
        print(
            f"{pfx} asteria  hit={timing.asteria_hit} "
            f"decision={dbg.get('lookup_decision', '?')} "
            f"temporal_bypass={dbg.get('temporal_bypass', False)} "
            f"prefilter_n={dbg.get('temporal_prefilter_size', '—')} "
            f"ann≥τ_sim={dbg.get('ann_candidates', '—')} "
            f"judger_best={best}",
            flush=True,
        )
        if timing.asteria_hit:
            mq = dbg.get("matched_query")
            if mq:
                msnip = mq[:100] + ("…" if len(mq) > 100 else "")
                print(
                    f"{pfx}          matched_cache_query={msnip!r} "
                    f"score={dbg.get('matched_score', '—')}",
                    flush=True,
                )

    ins = timing.asteria_insert_debug
    if ins:
        print(
            f"{pfx} insert   inserted={ins.get('inserted')} "
            f"skip_reason={ins.get('skip_reason')} "
            f"ins_bucket={ins.get('temporal_bucket')} "
            f"staticity={ins.get('staticity', '—')}",
            flush=True,
        )
    print(f"{pfx} latency  total={timing.total_s:.2f}s", flush=True)


def _print_cache_state(runner: ProfiledRunner, label: str) -> None:
    """Show whether the runner's cache is COLD (empty) or WARM (≥1 SE)."""
    cache = getattr(runner, "_asteria_cache", None)
    if cache is None:
        print(f"\n[{label}] Cache state: DISABLED (Asteria off)")
        return
    n = len(cache.ses)
    state = "COLD" if n == 0 else "WARM"
    print(f"\n[{label}] Cache state: {state} ({n} entries)")


def _dump_cache(runner: ProfiledRunner) -> None:
    """Print every SE currently in the warmed cache.

    Includes query text, temporal bucket + window, staticity, TTL hours,
    frequency (cache-hit count), and answer preview.  Lets the operator
    verify exactly what the pre-warm pass deposited.
    """
    cache = getattr(runner, "_asteria_cache", None)
    if cache is None:
        print("\n[dump-cache] Asteria disabled — nothing to dump.")
        return
    ses = list(cache.ses.values())
    print("\n" + "═" * 70)
    print(f"  Cache dump — {len(ses)} entries")
    print("═" * 70)
    if not ses:
        print("  (cache is empty)")
        return
    for i, se in enumerate(ses, start=1):
        window = "—"
        if se.time_window_start and se.time_window_end:
            window = f"{se.time_window_start} → {se.time_window_end}"
        ttl_h = round(se.ttl_seconds / 3600.0, 1)
        print(f"\n  [{i}] bucket={se.temporal_bucket}  window={window}")
        print(f"      query    : {se.query[:100]!r}")
        print(f"      answer   : {se.answer[:100]!r}…")
        print(
            f"      staticity={se.staticity}  ttl_h={ttl_h}  "
            f"freq={se.frequency}  size_tokens={se.size_tokens}"
        )


def _print_decision_trail(
    runner: ProfiledRunner,
    text: str,
    run_at: Optional[_dt.datetime],
    timing,
    *,
    indent: str = "    ",
) -> None:
    """Print the cache decision pipeline for a single query.

    Sections (verbose mode):
      [temporal filter]   classifier verdict + resolved window
      [pre-filter]        bucket+window scope, surviving SE count
      [ANN]               top-K candidates with cosine
      [Judger]            per-candidate scores, threshold, best
      [decision]          HIT (which SE) | MISS | VOLATILE bypass
    """
    dbg = timing.asteria_lookup_debug
    td = timing.temporal_debug or {}

    print(f"{indent}text: {text[:90]!r}")
    print(f"{indent}now:  {run_at.isoformat() if run_at else 'wall clock'}")

    print(f"{indent}[temporal filter]")
    bucket = td.get("display_tag", "?")
    window = td.get("time_window", "—")
    print(f"{indent}  bucket: {bucket}    window: {window}")

    # If Asteria is disabled (baseline pass) there is no cache lookup; skip
    # all cache-pipeline sections to avoid printing misleading info.
    if dbg is None:
        print(f"{indent}[asteria] disabled — baseline pass")
        print(f"{indent}total: {timing.total_s:.2f}s")
        return

    if dbg.get("temporal_bypass"):
        print(f"{indent}[decision] VOLATILE bypass — no cache lookup")
        return

    print(f"{indent}[pre-filter]")
    pf = dbg.get("temporal_prefilter_size")
    if pf is None:
        print(f"{indent}  scope: full cache (STATIC query, no temporal narrowing)")
    else:
        print(f"{indent}  scope: ANCHORED window-overlap filter")
        print(f"{indent}  candidates after pre-filter: {pf}")

    print(f"{indent}[ANN]")
    n_cand = dbg.get("ann_candidates", 0)
    print(f"{indent}  candidates ≥ τ_sim: {n_cand}")

    scores = dbg.get("judger_scores", []) or []
    print(f"{indent}[Judger]")
    if scores:
        print(f"{indent}  scores: {scores}    threshold τ_lsm: 0.80")
        print(f"{indent}  best:   {max(scores):.3f}")
    else:
        print(f"{indent}  no candidates evaluated")

    print(f"{indent}[decision]")
    if timing.asteria_hit:
        print(f"{indent}  HIT (source={dbg.get('source','sine')})")
        matched_q = dbg.get("matched_query")
        matched_s = dbg.get("matched_score")
        matched_b = dbg.get("matched_bucket")
        if matched_q is not None:
            print(f"{indent}  matched cached query: {matched_q[:90]!r}")
            extras = []
            if matched_s is not None:
                extras.append(f"score={matched_s}")
            if matched_b is not None:
                extras.append(f"bucket={matched_b}")
            if extras:
                print(f"{indent}  ({'  '.join(extras)})")
    else:
        print(f"{indent}  MISS — full pipeline ran")
    print(f"{indent}total: {timing.total_s:.2f}s")


# ── seed + sample passes ─────────────────────────────────────────────────────

async def _runner_run_with_retry(
    runner: ProfiledRunner,
    text: str,
    now: Optional[_dt.datetime],
    *,
    retries: int,
    delay_s: float,
    row_label: str,
    parallel: bool = False,
    cache_discovery: bool = False,
):
    """Wrap runner.run() with explicit retry on exception (WatsonX
    rate-limit, transient network).  Logs each retry so latency-distorting
    retries are visible in bench output, distinguishing them from cache cost.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return await runner.run(
                text, now=now,
                parallel=parallel,
                cache_discovery=cache_discovery,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                print(
                    f"     [retry {attempt + 1}/{retries} after error: "
                    f"{type(exc).__name__}: {str(exc)[:100]}] "
                    f"sleeping {delay_s:.1f}s … row={row_label}",
                    flush=True,
                )
                await asyncio.sleep(delay_s)
                continue
            print(
                f"     [retries exhausted on {row_label}, raising "
                f"{type(exc).__name__}]",
                flush=True,
            )
            raise
    raise RuntimeError("unreachable") from last_exc


async def _run_seed(
    runner: ProfiledRunner,
    rows: list[dict],
    *,
    label: str,
    verbose: bool = False,
    debug: bool = False,
    retries: int = 2,
    retry_delay_s: float = 5.0,
    parallel: bool = False,
    cache_discovery: bool = False,
) -> SeedStats:
    stats = SeedStats()
    print(f"\n[{label}] seeding {len(rows)} rows …")
    for i, row in enumerate(rows, start=1):
        text = row.get("text", "").strip()
        if not text:
            continue
        run_at = _parse_run_at(row)
        t = await _runner_run_with_retry(
            runner, text, run_at,
            retries=retries, delay_s=retry_delay_s,
            row_label=f"id={row.get('id','?')}",
            parallel=parallel, cache_discovery=cache_discovery,
        )
        pid = str(row.get("parent_id") or "").strip()
        if pid:
            stats.parents_executed.add(pid)
            ins = t.asteria_insert_debug or {}
            if ins.get("inserted"):
                stats.parents_warmed.add(pid)
        print(
            f"  {i:>3}/{len(rows)}  "
            f"id={row.get('id','?')}  parent_id={row.get('parent_id','-')}  "
            f"tier={row.get('similarity_tier','-')}  "
            f"now={run_at.isoformat() if run_at else 'wall'}  "
            f"hit={t.asteria_hit}  total={t.total_s:.2f}s",
            flush=True,
        )
        if debug:
            _print_scenario_debug(row, run_at, t, phase="seed")
        if verbose:
            _print_decision_trail(runner, text, run_at, t)
    return stats


async def _run_sample_pass(
    runner: ProfiledRunner,
    sampled: list[dict],
    *,
    label: str,
    verbose: bool = False,
    debug: bool = False,
    retries: int = 2,
    retry_delay_s: float = 5.0,
    parallel: bool = False,
    cache_discovery: bool = False,
) -> PassSummary:
    out = PassSummary(label=label)
    print(f"\n[{label}] running {len(sampled)} sampled rows …")
    for i, row in enumerate(sampled, start=1):
        text = row.get("text", "").strip()
        if not text:
            continue
        run_at = _parse_run_at(row)
        t = await _runner_run_with_retry(
            runner, text, run_at,
            retries=retries, delay_s=retry_delay_s,
            row_label=f"id={row.get('id','?')}",
            parallel=parallel, cache_discovery=cache_discovery,
        )
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
        if debug:
            _print_scenario_debug(row, run_at, t, phase=label)
        if verbose:
            _print_decision_trail(runner, text, run_at, t)
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"avg={mean(values):.2f}s  med={median(values):.2f}s  "
        f"min={min(values):.2f}s  max={max(values):.2f}s"
    )


def _print_hit_focused_latency(
    baseline: PassSummary,
    cached: PassSummary,
    common_ids: list[str],
) -> None:
    """Latency interpretation: hits only for speedup; misses for overhead."""
    hit_pairs: list[tuple[float, float]] = []
    miss_deltas: list[float] = []
    for rid in common_ids:
        b = baseline.by_id[rid].total_s
        c = cached.by_id[rid].total_s
        if cached.by_id[rid].asteria_hit:
            if c > 0:
                hit_pairs.append((b, c))
        else:
            miss_deltas.append(c - b)

    print(
        "\n  Asteria latency (paired rows) — overall avg mixes hits + misses; "
        "use the rows below for end-user benefit."
    )

    if hit_pairs:
        speedups = [b / c for b, c in hit_pairs]
        savings_frac = [(b - c) / b if b > 0 else 0.0 for b, c in hit_pairs]
        geo_sp = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        abs_saved = [b - c for b, c in hit_pairs]
        print(
            f"  On cache HITs only (n={len(hit_pairs)}):\n"
            f"    speedup baseline/cached  mean={mean(speedups):.2f}x  "
            f"median={median(speedups):.2f}x  geom_mean={geo_sp:.2f}x\n"
            f"    time saved vs baseline   "
            f"mean={mean(savings_frac)*100:.1f}%  "
            f"median={median(savings_frac)*100:.1f}%  "
            f"(per-row (b−c)/b)\n"
            f"    seconds saved (hit rows) mean={mean(abs_saved):.2f}s  "
            f"median={median(abs_saved):.2f}s"
        )
    else:
        print("  On cache HITs only: n=0 — no speedup sample this run.")

    if miss_deltas:
        print(
            f"  On cache MISS (n={len(miss_deltas)}) — extra vs baseline:\n"
            f"    cached−baseline  "
            f"mean={mean(miss_deltas):+.2f}s  "
            f"median={median(miss_deltas):+.2f}s  "
            f"(lookup + pipeline; expected ≥ 0 often)"
        )


def _print_compare(baseline: PassSummary, cached: PassSummary) -> None:
    print("\n" + "═" * 70)
    print("  Benchmark Summary")
    print("═" * 70)
    print(f"  baseline runs   : {baseline.runs}    {_stats(baseline.totals)}")
    print(f"  cached   runs   : {cached.runs}    {_stats(cached.totals)}")
    print(f"  cache hits      : {cached.hits}/{cached.runs}  "
          f"({100.0 * cached.hits / max(1, cached.runs):.1f}%)")

    common = sorted(set(baseline.by_id) & set(cached.by_id))
    if common:
        _print_hit_focused_latency(baseline, cached, common)

    # Per-row diff
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


def _parse_confusion_tiers(arg: str | None) -> set[str]:
    if not arg:
        return {"paraphrase", "shifted_anchored"}
    return {x.strip() for x in arg.split(",") if x.strip()}


def _print_confusion_report(
    *,
    seed_stats: SeedStats,
    sampled: list[dict],
    cached_summary: PassSummary,
    mode: str,
    tiers: set[str],
    confusion_csv: Path | None,
) -> None:
    """Reliability table: ground truth from seed parent_ids vs Asteria hit on cached pass."""
    positives = (
        seed_stats.parents_executed
        if mode == "intent"
        else seed_stats.parents_warmed
    )
    mode_desc = (
        "intent: y=true if test.parent_id appears in any executed seed row"
        if mode == "intent"
        else "strict: y=true only if that parent had ≥1 successful Asteria insert in seed"
    )

    eval_rows: list[tuple[dict, int, int]] = []
    for row in sampled:
        tier = (row.get("similarity_tier") or "").strip()
        if tiers and tier not in tiers:
            continue
        rid = row.get("id", "")
        if not rid or rid not in cached_summary.by_id:
            continue
        pid = str(row.get("parent_id") or "").strip()
        y_pred = 1 if cached_summary.by_id[rid].asteria_hit else 0
        y_true = 1 if pid in positives else 0
        eval_rows.append((row, y_true, y_pred))

    tp = fp = tn = fn = 0
    for _, yt, yp in eval_rows:
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            tn += 1

    print("\n" + "═" * 70)
    print("  Asteria reliability (confusion vs seed parent_id)")
    print("═" * 70)
    print(f"  Label mode       : {mode_desc}")
    print(f"  Tiers in matrix  : {', '.join(sorted(tiers))}")
    print(f"  Rows evaluated   : {len(eval_rows)} (of {len(sampled)} sampled)")
    print(
        f"  Seed parents     : {len(seed_stats.parents_executed)} executed, "
        f"{len(seed_stats.parents_warmed)} warmed (insert ok)"
    )
    print(
        "\n                      Predicted miss   Predicted hit\n"
        f"  Actual negative     {tn:>6d}           {fp:>6d}\n"
        f"  Actual positive     {fn:>6d}           {tp:>6d}"
    )
    print(
        "\n  TN = no seed parent, no hit  |  TP = seed parent and hit\n"
        "  FN = seed parent, miss        |  FP = no seed parent but hit"
    )

    den_p = tp + fp
    den_r = tp + fn
    prec = tp / den_p if den_p else float("nan")
    rec = tp / den_r if den_r else float("nan")
    if not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = float("nan")
    den_s = tn + fp
    spec = tn / den_s if den_s else float("nan")

    print()
    if not math.isnan(prec):
        print(f"  Precision  : {prec:.4f}")
    else:
        print("  Precision  : n/a (no predicted hits)")
    if not math.isnan(rec):
        print(f"  Recall     : {rec:.4f}")
    else:
        print("  Recall     : n/a (no positive labels in sample)")
    if not math.isnan(f1):
        print(f"  F1         : {f1:.4f}")
    else:
        print("  F1         : n/a")
    if not math.isnan(spec):
        print(f"  Specificity: {spec:.4f}")
    else:
        print("  Specificity: n/a")

    if confusion_csv and eval_rows:
        confusion_csv.parent.mkdir(parents=True, exist_ok=True)
        with confusion_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "id",
                    "parent_id",
                    "similarity_tier",
                    "y_true",
                    "y_pred",
                    "cell",
                    "label_mode",
                ],
            )
            w.writeheader()
            for row, yt, yp in eval_rows:
                if yt == 1 and yp == 1:
                    cell = "TP"
                elif yt == 1 and yp == 0:
                    cell = "FN"
                elif yt == 0 and yp == 1:
                    cell = "FP"
                else:
                    cell = "TN"
                w.writerow(
                    {
                        "id": row.get("id", ""),
                        "parent_id": row.get("parent_id", ""),
                        "similarity_tier": row.get("similarity_tier", ""),
                        "y_true": yt,
                        "y_pred": yp,
                        "cell": cell,
                        "label_mode": mode,
                    }
                )
        print(f"\n  Wrote per-row labels → {confusion_csv}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_cache",
        description="Cache benchmark using paraphrase seed + test CSVs.",
    )
    p.add_argument("--seed-csv", type=Path, required=True,
                   help="CSV produced by generate_scenarios.py to pre-warm cache.")
    p.add_argument("--test-csv", type=Path, required=True,
                   help="CSV with paraphrases for the test phase.")
    p.add_argument(
        "--sample-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set: randomly sample N rows from the test CSV (after filters). "
            "If omitted: run **all** test rows — use this for full benchmarks."
        ),
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed when --sample-count is set (ignored for full test run).",
    )
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
    p.add_argument(
        "--verbose", action="store_true",
        help=(
            "Print the full cache decision trail per row "
            "(temporal filter → pre-filter → ANN → judger → decision)."
        ),
    )
    p.add_argument(
        "--llm-retries", type=int, default=2,
        help=(
            "Retries per row when the runner raises (typically WatsonX "
            "rate-limit).  Default: 2."
        ),
    )
    p.add_argument(
        "--retry-delay-s", type=float, default=5.0,
        help="Seconds to sleep between retries (default: 5.0).",
    )
    p.add_argument(
        "--dump-cache", action="store_true",
        help=(
            "After the seed pre-warm pass, print every SE in the warmed "
            "cache (query, bucket, window, staticity, ttl, frequency).  "
            "Lets you verify what the cache actually holds before the "
            "measurement passes run."
        ),
    )
    p.add_argument(
        "--confusion-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print TP/TN/FP/FN and precision/recall after the cached pass (default: on).",
    )
    p.add_argument(
        "--confusion-mode",
        choices=("intent", "strict"),
        default="intent",
        help="How y=true is chosen for reliability (see docstring in script header).",
    )
    p.add_argument(
        "--confusion-tiers",
        default="paraphrase,shifted_anchored",
        help="Comma-separated similarity_tier values counted in the matrix.",
    )
    p.add_argument(
        "--confusion-csv",
        type=Path,
        default=None,
        help="Optional path for per-row y_true / y_pred / cell.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=(
            "After each row, print scenario id/parent/tier, temporal bucket/window, "
            "Asteria hit/miss, ANN/judger hints, and insert outcome. "
            "Use --verbose for the full decision trail as well."
        ),
    )
    p.add_argument(
        "--parallel", action="store_true",
        help="Use DAG-based parallel execution + MCP connection pool for all runner passes.",
    )
    p.add_argument(
        "--cache-discovery", action="store_true",
        help="Enable disk-backed discovery-phase cache (24 h TTL) for all runner passes.",
    )
    p.add_argument(
        "--ablation", action="store_true",
        help=(
            "Run full ablation study: baseline (all optimizations OFF) then optimized "
            "(Parallel + Discovery Cache + Asteria all ON) on the same data. "
            "Writes a markdown report and PNG plots."
        ),
    )
    p.add_argument(
        "--report-path", type=Path, default=Path("ablation_report.md"),
        help="Path for the ablation markdown report (default: ablation_report.md).",
    )
    p.add_argument(
        "--plots-dir", type=Path, default=Path("ablation_plots"),
        help="Directory for ablation PNG plots (default: ablation_plots/).",
    )
    return p


# ── ablation utilities ───────────────────────────────────────────────────────

def _compute_ablation_metrics(baseline: PassSummary, cached: PassSummary) -> dict:
    metrics = {}
    metrics['baseline_runs'] = baseline.runs
    metrics['optimized_runs'] = cached.runs
    metrics['cache_hits'] = cached.hits
    metrics['hit_rate'] = (cached.hits / cached.runs) if cached.runs else 0.0

    metrics['baseline_avg'] = mean(baseline.totals) if baseline.totals else 0.0
    metrics['baseline_med'] = median(baseline.totals) if baseline.totals else 0.0
    metrics['baseline_min'] = min(baseline.totals) if baseline.totals else 0.0
    metrics['baseline_max'] = max(baseline.totals) if baseline.totals else 0.0

    metrics['optimized_avg'] = mean(cached.totals) if cached.totals else 0.0
    metrics['optimized_med'] = median(cached.totals) if cached.totals else 0.0
    metrics['optimized_min'] = min(cached.totals) if cached.totals else 0.0
    metrics['optimized_max'] = max(cached.totals) if cached.totals else 0.0

    common_ids = sorted(set(baseline.by_id) & set(cached.by_id))
    hit_pairs = []
    miss_deltas = []
    for rid in common_ids:
        b = baseline.by_id[rid].total_s
        c = cached.by_id[rid].total_s
        if cached.by_id[rid].asteria_hit:
            if c > 0:
                hit_pairs.append((b, c))
        else:
            miss_deltas.append(c - b)

    metrics['hit_speedups'] = [b/c for b, c in hit_pairs]
    metrics['hit_savings_frac'] = [(b - c) / b if b > 0 else 0.0 for b, c in hit_pairs]
    metrics['miss_deltas'] = miss_deltas

    if hit_pairs:
        metrics['mean_hit_speedup'] = mean(metrics['hit_speedups'])
        metrics['geom_mean_hit_speedup'] = math.exp(sum(math.log(s) for s in metrics['hit_speedups']) / len(metrics['hit_speedups']))
        metrics['mean_hit_savings'] = mean(metrics['hit_savings_frac'])
    else:
        metrics['mean_hit_speedup'] = 0.0
        metrics['geom_mean_hit_speedup'] = 0.0
        metrics['mean_hit_savings'] = 0.0

    if miss_deltas:
        metrics['mean_miss_delta'] = mean(miss_deltas)
    else:
        metrics['mean_miss_delta'] = 0.0

    return metrics


def _write_ablation_report(
    report_path: Path,
    args: argparse.Namespace,
    baseline: PassSummary,
    cached: PassSummary,
    metrics: dict,
    seed_stats: SeedStats,
    sampled: list[dict],
) -> None:
    mode = args.confusion_mode
    tiers = _parse_confusion_tiers(args.confusion_tiers)
    
    positives = (
        seed_stats.parents_executed
        if mode == "intent"
        else seed_stats.parents_warmed
    )
    tp = fp = tn = fn = 0
    for row in sampled:
        tier = (row.get("similarity_tier") or "").strip()
        if tiers and tier not in tiers:
            continue
        rid = row.get("id", "")
        if not rid or rid not in cached.by_id:
            continue
        pid = str(row.get("parent_id") or "").strip()
        y_pred = 1 if cached.by_id[rid].asteria_hit else 0
        y_true = 1 if pid in positives else 0
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        else:
            tn += 1

    den_p = tp + fp
    den_r = tp + fn
    prec = tp / den_p if den_p else float("nan")
    rec = tp / den_r if den_r else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0 else float("nan")
    den_s = tn + fp
    spec = tn / den_s if den_s else float("nan")

    lines = [
        "# Asteria Ablation Study Report",
        "",
        "## Run Configuration",
        "| Parameter | Value |",
        "|---|---|",
        f"| Sample count | {args.sample_count or 'All'} |",
        f"| Sample seed | {args.sample_seed} |",
        f"| Max seed rows | {args.max_seed_rows or 'All'} |",
        f"| Model | {args.model_id} |",
        f"| Skip summary | {args.skip_summary} |",
        "",
        "## Latency Summary",
        "| Metric | Baseline (all OFF) | Optimized (all ON) |",
        "|---|---|---|",
        f"| Runs | {metrics['baseline_runs']} | {metrics['optimized_runs']} |",
        f"| Avg | {metrics['baseline_avg']:.2f}s | {metrics['optimized_avg']:.2f}s |",
        f"| Median | {metrics['baseline_med']:.2f}s | {metrics['optimized_med']:.2f}s |",
        f"| Min | {metrics['baseline_min']:.2f}s | {metrics['optimized_min']:.2f}s |",
        f"| Max | {metrics['baseline_max']:.2f}s | {metrics['optimized_max']:.2f}s |",
        "",
        "## Aggregated Improvement",
        "| Metric | Value |",
        "|---|---|",
        f"| Cache Hits | {metrics['cache_hits']} / {metrics['optimized_runs']} ({metrics['hit_rate']*100:.1f}%) |",
        f"| Mean Speedup (Hit Rows) | {metrics['mean_hit_speedup']:.2f}x |",
        f"| Geom-Mean Speedup | {metrics['geom_mean_hit_speedup']:.2f}x |",
        f"| Mean Latency Reduction | {metrics['mean_hit_savings']*100:.1f}% |",
        f"| Mean Miss Overhead | {metrics['mean_miss_delta']:+.2f}s |",
        "",
        "## Reliability",
        "| TP | FP | FN | TN | Precision | Recall | F1 | Specificity |",
        "|---|---|---|---|---|---|---|---|",
        f"| {tp} | {fp} | {fn} | {tn} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {spec:.4f} |",
        "",
        "## Per-Row Results",
        "| ID | Tier | Baseline (s) | Optimized (s) | Hit |",
        "|---|---|---|---|---|",
    ]
    common = sorted(set(baseline.by_id) & set(cached.by_id))
    for rid in common:
        b = baseline.by_id[rid]
        c = cached.by_id[rid]
        hit = "YES" if c.asteria_hit else "no"
        lines.append(f"| {rid} | {b.similarity} | {b.total_s:.2f} | {c.total_s:.2f} | {hit} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Wrote ablation report → {report_path}")


def _generate_ablation_plots(
    plots_dir: Path,
    baseline: PassSummary,
    cached: PassSummary,
    metrics: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [warn] matplotlib not found; skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Latency Comparison
    plt.figure(figsize=(8, 5))
    labels = ['Average', 'Median']
    b_vals = [metrics['baseline_avg'], metrics['baseline_med']]
    o_vals = [metrics['optimized_avg'], metrics['optimized_med']]
    x = list(range(len(labels)))
    width = 0.35
    plt.bar([i - width/2 for i in x], b_vals, width, label='Baseline (all OFF)', color='lightcoral')
    plt.bar([i + width/2 for i in x], o_vals, width, label='Optimized (all ON)', color='mediumseagreen')
    plt.ylabel('Latency (s)')
    plt.title('Latency Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'latency_comparison.png')
    plt.close()

    # 2. Per-Row Latency
    common_ids = sorted(set(baseline.by_id) & set(cached.by_id))
    if common_ids:
        plt.figure(figsize=(8, 6))
        b_hits = []
        o_hits = []
        b_misses = []
        o_misses = []
        for rid in common_ids:
            b = baseline.by_id[rid].total_s
            o = cached.by_id[rid].total_s
            if cached.by_id[rid].asteria_hit:
                b_hits.append(b)
                o_hits.append(o)
            else:
                b_misses.append(b)
                o_misses.append(o)

        max_val = max([baseline.by_id[rid].total_s for rid in common_ids] + [cached.by_id[rid].total_s for rid in common_ids] + [0.1]) * 1.1

        plt.scatter(b_misses, o_misses, color='red', alpha=0.6, label='Miss')
        plt.scatter(b_hits, o_hits, color='green', alpha=0.6, label='Hit')
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x (No change)')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.xlabel('Baseline Latency (s)')
        plt.ylabel('Optimized Latency (s)')
        plt.title('Per-Row Latency Scatter')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'per_row_latency.png')
        plt.close()

    # 3. Speedup Distribution
    if metrics['hit_speedups']:
        plt.figure(figsize=(8, 5))
        plt.hist(metrics['hit_speedups'], bins=10, color='mediumseagreen', edgecolor='black')
        plt.axvline(metrics['mean_hit_speedup'], color='red', linestyle='dashed', linewidth=1, label=f"Mean: {metrics['mean_hit_speedup']:.2f}x")
        plt.xlabel('Speedup (x)')
        plt.ylabel('Count')
        plt.title('Speedup Distribution (Hits Only)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'speedup_distribution.png')
        plt.close()

    # 4. Hit Rate
    plt.figure(figsize=(6, 6))
    hits = metrics['cache_hits']
    misses = metrics['optimized_runs'] - hits
    if hits + misses > 0:
        plt.pie([hits, misses], labels=['Hits', 'Misses'], autopct='%1.1f%%', colors=['mediumseagreen', 'lightcoral'], startangle=90)
        plt.title('Cache Hit Rate')
        plt.tight_layout()
        plt.savefig(plots_dir / 'hit_rate.png')
    plt.close()

    print(f"  Wrote 4 plots → {plots_dir}/")


async def _run_ablation_study(args: argparse.Namespace, seed_rows: list[dict], sampled: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  Starting Ablation Study")
    print("  Baseline: Sequential, No Discovery Cache, No Asteria")
    print("  Optimized: Parallel, Discovery Cache, Asteria (warmed)")
    print("=" * 70)

    # 1. Baseline Run
    baseline_abl_runner = ProfiledRunner(
        model_id=args.model_id,
        summarize=not args.skip_summary,
        summary_max_chars=args.summary_max_chars,
        step_response_max_chars=args.step_response_max_chars,
        asteria_enabled=False,
    )
    abl_baseline_summary = await _run_sample_pass(
        baseline_abl_runner, sampled,
        label="ablation-baseline (ALL OFF)",
        verbose=args.verbose,
        debug=args.debug,
        retries=args.llm_retries,
        retry_delay_s=args.retry_delay_s,
        parallel=False,
        cache_discovery=False,
    )

    # 2. Seed Pass
    opt_runner = ProfiledRunner(
        model_id=args.model_id,
        summarize=not args.skip_summary,
        summary_max_chars=args.summary_max_chars,
        step_response_max_chars=args.step_response_max_chars,
        asteria_enabled=True,
    )
    seed_stats = await _run_seed(
        opt_runner, seed_rows,
        label="ablation-seed (WARM CACHE)",
        verbose=args.verbose,
        debug=args.debug,
        retries=args.llm_retries,
        retry_delay_s=args.retry_delay_s,
        parallel=True,
        cache_discovery=True,
    )

    # 3. Optimized Run
    abl_opt_summary = await _run_sample_pass(
        opt_runner, sampled,
        label="ablation-optimized (ALL ON)",
        verbose=args.verbose,
        debug=args.debug,
        retries=args.llm_retries,
        retry_delay_s=args.retry_delay_s,
        parallel=True,
        cache_discovery=True,
    )

    # 4. Report & Plots
    metrics = _compute_ablation_metrics(abl_baseline_summary, abl_opt_summary)
    
    _print_compare(abl_baseline_summary, abl_opt_summary)
    
    if args.confusion_matrix:
        _print_confusion_report(
            seed_stats=seed_stats,
            sampled=sampled,
            cached_summary=abl_opt_summary,
            mode=args.confusion_mode,
            tiers=_parse_confusion_tiers(args.confusion_tiers),
            confusion_csv=args.confusion_csv,
        )

    _write_ablation_report(
        report_path=args.report_path,
        args=args,
        baseline=abl_baseline_summary,
        cached=abl_opt_summary,
        metrics=metrics,
        seed_stats=seed_stats,
        sampled=sampled,
    )

    _generate_ablation_plots(
        plots_dir=args.plots_dir,
        baseline=abl_baseline_summary,
        cached=abl_opt_summary,
        metrics=metrics,
    )


def _clear_environment_caches() -> None:
    """Clear discovery cache and compiled Python files to ensure a clean run."""
    print("\n[cleanup] Clearing unnecessary cache files...")
    
    # 1. Clear discovery cache
    discovery_cache = Path(".discovery_cache.json")
    if discovery_cache.exists():
        discovery_cache.unlink()
        print(f"  Deleted {discovery_cache}")
        
    # 2. Clear __pycache__ directories
    count = 0
    for p in Path(".").rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p)
            count += 1
    if count > 0:
        print(f"  Deleted {count} __pycache__ directories")
            
    # 3. Clear pytest cache
    pytest_cache = Path(".pytest_cache")
    if pytest_cache.is_dir():
        shutil.rmtree(pytest_cache)
        print(f"  Deleted {pytest_cache}")


async def _main(args: argparse.Namespace) -> None:
    _clear_environment_caches()
    seed_rows = _filter_by_types(_load_rows(args.seed_csv), args.seed_types)
    test_rows = _filter_by_types(_load_rows(args.test_csv), args.test_types)

    if args.max_seed_rows:
        seed_rows = seed_rows[: args.max_seed_rows]

    if not seed_rows:
        sys.exit("No seed rows after filtering.")
    if not test_rows:
        sys.exit("No test rows after filtering.")

    if args.sample_count is not None and args.sample_count < 1:
        sys.exit("--sample-count must be >= 1 when provided.")

    if args.sample_count is None:
        sampled = list(test_rows)
        print(f"Test selection    : all rows ({len(sampled)}), no subsample")
    else:
        sampled = sample_rows(
            test_rows, count=args.sample_count, seed=args.sample_seed)
        print(
            f"Test selection    : random subsample n={args.sample_count} "
            f"(seed={args.sample_seed})"
        )

    print(f"Seed rows         : {len(seed_rows)}")
    print(f"Test rows total   : {len(test_rows)}")
    print(f"Test rows to run  : {len(sampled)}")

    if args.ablation:
        await _run_ablation_study(args, seed_rows, sampled)
        return

    print(
        "\n[independence] New ProfiledRunner instances — Asteria cache starts empty."
    )

    parallel = getattr(args, "parallel", False)
    cache_discovery = getattr(args, "cache_discovery", False)

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

    _print_cache_state(cached_runner, "before seed")
    t0 = time.perf_counter()
    seed_stats = await _run_seed(
        cached_runner, seed_rows,
        label="seed",
        verbose=args.verbose,
        debug=args.debug,
        retries=args.llm_retries,
        retry_delay_s=args.retry_delay_s,
    )
    print(f"\n[seed] done in {time.perf_counter() - t0:.1f}s")
    _print_cache_state(cached_runner, "after seed")

    if args.dump_cache:
        _dump_cache(cached_runner)

    _print_cache_state(baseline_runner, "before baseline pass")
    baseline_summary = await _run_sample_pass(
        baseline_runner, sampled,
        label="baseline",
        verbose=args.verbose,
        debug=args.debug,
        retries=args.llm_retries,
        retry_delay_s=args.retry_delay_s,
    )
    _print_cache_state(cached_runner, "before cached pass")
    cached_summary = await _run_sample_pass(
        cached_runner, sampled,
        label="cached",
        verbose=args.verbose,
        debug=args.debug,
        retries=args.llm_retries,
        retry_delay_s=args.retry_delay_s,
    )
    _print_cache_state(cached_runner, "after cached pass")

    _print_compare(baseline_summary, cached_summary)

    if args.confusion_matrix:
        _print_confusion_report(
            seed_stats=seed_stats,
            sampled=sampled,
            cached_summary=cached_summary,
            mode=args.confusion_mode,
            tiers=_parse_confusion_tiers(args.confusion_tiers),
            confusion_csv=args.confusion_csv,
        )


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
