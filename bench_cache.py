"""
Cache benchmark — baseline vs Asteria over a two-stage paraphrase corpus.

Workflow:
  1. Seed pass: run each row of cache_seed.csv through a ProfiledRunner with
     Asteria enabled.  Cache is populated with real pipeline answers (subject
     to staticity gate / temporal-bucket policy).  Use --max-seed-rows only
     for quick smoke runs; omit it to run the full seed CSV.
  2. Test pass: by default **every** row of cache_test.csv (after --test-types
     filter).  Pass --sample-count N for uniform random subsample, or
     --sample-warm A --sample-cold B for a stratified smoke split (parents in
     seed CSV vs not).
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

Optional stratified smoke (A warm + B cold test rows by parent_id vs seed CSV):
  ... --sample-warm 4 --sample-cold 4 --sample-seed 42

Disable the reliability table with --no-confusion-matrix.

Common options:
  --skip-summary             Skip the LLM summarization step (faster).
  --seed-types IoT,Workorder Restrict seed pass to these types.
  --test-types IoT,Workorder Restrict test CSV to these types.
  --sample-count N           Uniform random N rows (omit = use all).
  --sample-warm A            With --sample-cold B: A rows whose parent_id is in seed CSV.
  --sample-cold B            B rows whose parent_id is not in seed CSV.
  --sample-seed N            RNG for sampling (default: 42).
  --max-seed-rows N          Cap seed pass for fast smoke runs.
  --confusion-matrix         TP/TN/FP/FN + precision/recall (on by default).
  --no-confusion-matrix      Skip reliability table.
  --debug                    Per-row [debug:…] block (scenario, temporal, Asteria hints).
  --verbose                  Full Asteria decision trail (temporal → ANN → judger → decision).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as _dt
import math
import random
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
):
    """Wrap runner.run() with explicit retry on exception (WatsonX
    rate-limit, transient network).  Logs each retry so latency-distorting
    retries are visible in bench output, distinguishing them from cache cost.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return await runner.run(text, now=now)
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
    n = len(values)
    k = int(n * 0.05)
    trimmed = sorted(values)[k : n - k] if n - 2 * k >= 1 else values
    trimmed_avg = mean(trimmed)
    return (
        f"avg={mean(values):.2f}s  trimmed_avg={trimmed_avg:.2f}s  "
        f"med={median(values):.2f}s  min={min(values):.2f}s  max={max(values):.2f}s"
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


def _stratified_warm_cold_sample(
    test_rows: list[dict],
    seed_rows: list[dict],
    n_warm: int,
    n_cold: int,
    *,
    rng_seed: int,
) -> list[dict]:
    """Pick test rows by whether ``parent_id`` appears in *seed_rows* (CSV-wise warm vs cold)."""
    seed_parents = {
        str(r.get("parent_id") or "").strip()
        for r in seed_rows
        if str(r.get("parent_id") or "").strip()
    }
    warm_pool: list[dict] = []
    cold_pool: list[dict] = []
    for r in test_rows:
        pid = str(r.get("parent_id") or "").strip()
        if pid in seed_parents:
            warm_pool.append(r)
        else:
            cold_pool.append(r)

    rng = random.Random(rng_seed)
    out: list[dict] = []
    if n_warm:
        if n_warm > len(warm_pool):
            raise ValueError(
                f"--sample-warm={n_warm} but only {len(warm_pool)} test rows have "
                f"parent_id in the seed CSV (after filters / --max-seed-rows)."
            )
        out.extend(rng.sample(warm_pool, n_warm))
    if n_cold:
        if n_cold > len(cold_pool):
            raise ValueError(
                f"--sample-cold={n_cold} but only {len(cold_pool)} cold test rows "
                f"(parent_id not in seed CSV)."
            )
        out.extend(rng.sample(cold_pool, n_cold))
    rng.shuffle(out)
    return out


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
        help="RNG for --sample-count or --sample-warm/--sample-cold (default: 42).",
    )
    p.add_argument(
        "--sample-warm",
        type=int,
        default=None,
        metavar="A",
        help=(
            "Stratified smoke: take A test rows whose parent_id appears in the seed CSV "
            "(after filters and --max-seed-rows).  Must be used with --sample-cold; "
            "incompatible with --sample-count."
        ),
    )
    p.add_argument(
        "--sample-cold",
        type=int,
        default=None,
        metavar="B",
        help=(
            "Stratified smoke: take B test rows whose parent_id is not in the seed CSV. "
            "Must be used with --sample-warm."
        ),
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
        default="strict",
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
            "Per-row [debug:…] lines: scenario id/parent/tier, query snip, temporal bucket/window, "
            "Asteria hit/miss and lookup hints, insert outcome. "
            "For the full cache decision trail as well, pass --verbose."
        ),
    )
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

    if args.sample_count is not None and args.sample_count < 1:
        sys.exit("--sample-count must be >= 1 when provided.")

    strat_w, strat_c = args.sample_warm, args.sample_cold
    if (strat_w is not None) ^ (strat_c is not None):
        sys.exit("Pass both --sample-warm and --sample-cold, or neither.")
    if args.sample_count is not None and strat_w is not None:
        sys.exit("Use either --sample-count or --sample-warm/--sample-cold, not both.")

    if strat_w is not None:
        assert strat_c is not None
        if strat_w < 0 or strat_c < 0:
            sys.exit("--sample-warm and --sample-cold must be >= 0.")
        if strat_w + strat_c < 1:
            sys.exit("Stratified test sample must include at least one row.")
        try:
            sampled = _stratified_warm_cold_sample(
                test_rows,
                seed_rows,
                strat_w,
                strat_c,
                rng_seed=args.sample_seed,
            )
        except ValueError as exc:
            sys.exit(str(exc))
        print(
            f"Test selection    : stratified warm={strat_w} cold={strat_c} "
            f"(seed={args.sample_seed})"
        )
    elif args.sample_count is None:
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
    print(
        "\n[independence] New ProfiledRunner instances — Asteria cache starts empty."
    )

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
