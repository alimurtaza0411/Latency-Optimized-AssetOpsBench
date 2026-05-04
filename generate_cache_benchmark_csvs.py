#!/usr/bin/env python3
"""
Generate a paired cache_seed.csv and cache_test.csv for Asteria benchmarking.

  • cache_seed.csv — exactly one LLM paraphrase per scenario for N unique
    parent rows (default N=20).  No shifted_anchored rows.  Sequential warm-up
    therefore sees at most one row per parent_id → no within-seed Asteria hit
    from a second paraphrase of the same scenario.

  • cache_test.csv — by default **60%** of rows are paraphrases of the warm
    (seeded) parents; **40%** are paraphrases of held-out (cold) parents.
    Counts are computed in **rows** (e.g. 100 → 60 warm + 40 cold).

Paraphrase prompts, model, dedup, and row shape match generate_scenarios.py.

Usage:
  PYTHONPATH=. uv run python generate_cache_benchmark_csvs.py

  PYTHONPATH=. uv run python generate_cache_benchmark_csvs.py \\
      --warm-count 20 --test-total 100 --warm-test-pct 60 --parent-sample-seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import generate_scenarios as gs


FIELDNAMES = [
    "id", "parent_id", "type", "text",
    "category", "deterministic", "characteristic_form",
    "group", "entity", "note",
    "similarity_tier",
    "synthetic_window_start", "synthetic_window_end",
    "query_run_at",
]


def _alloc_warm_counts(warm_n: int, warm_test_rows: int) -> list[int]:
    """Split warm_test_rows across warm_n parents as evenly as possible."""
    if warm_n < 1:
        return []
    per = warm_test_rows // warm_n
    rem = warm_test_rows % warm_n
    return [per + (1 if i < rem else 0) for i in range(warm_n)]


def _pick_warm_and_cold(
    rows: list[dict],
    warm_n: int,
    cold_n: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    if len(rows) < warm_n + cold_n:
        sys.exit(
            f"Need at least {warm_n + cold_n} rows in input; got {len(rows)}"
        )
    shuffled = rows[:]
    rng.shuffle(shuffled)
    warm = shuffled[:warm_n]
    warm_ids = {r["id"] for r in warm}
    pool = [r for r in shuffled[warm_n:] if r["id"] not in warm_ids]
    if len(pool) < cold_n:
        sys.exit(f"Not enough cold parents after warm pick; need {cold_n}, have {len(pool)}")
    cold = rng.sample(pool, cold_n)
    return warm, cold


def _emit_seed_rows(
    warm_rows: list[dict],
    corpus: list[str],
    next_id: int,
) -> tuple[list[dict], dict[str, str], int]:
    """One paraphrase per row; seed_text[parent_id] = emitted text."""
    out: list[dict] = []
    seed_text: dict[str, str] = {}
    for i, row in enumerate(warm_rows, start=1):
        print(f"[seed {i}/{len(warm_rows)}] parent_id={row['id']} type={row['type']}")
        prompt = gs._build_paraphrase_prompt(row, 1)
        candidates = gs._call_llm(prompt)
        accepted = 0
        for cand in candidates:
            text = str(cand.get("text", "")).strip()
            if not text:
                continue
            if gs._is_near_duplicate(text, corpus, gs.DEDUP_THRESHOLD):
                continue
            corpus.append(text)
            pid = str(row["id"])
            seed_text[pid] = text
            out.append(
                gs._row_template(
                    row,
                    new_id=next_id,
                    similarity_tier="paraphrase",
                    text=text,
                    w_start=None,
                    w_end=None,
                    query_run_at=gs._generate_query_run_at(row["id"], 0),
                )
            )
            next_id += 1
            accepted = 1
            break
        if not accepted:
            prompt = gs._build_paraphrase_prompt(row, 3)
            candidates = gs._call_llm(prompt)
            for cand in candidates:
                text = str(cand.get("text", "")).strip()
                if not text:
                    continue
                if gs._is_near_duplicate(text, corpus, gs.DEDUP_THRESHOLD):
                    continue
                corpus.append(text)
                pid = str(row["id"])
                seed_text[pid] = text
                out.append(
                    gs._row_template(
                        row,
                        new_id=next_id,
                        similarity_tier="paraphrase",
                        text=text,
                        w_start=None,
                        w_end=None,
                        query_run_at=gs._generate_query_run_at(row["id"], 0),
                    )
                )
                next_id += 1
                accepted = 1
                break
        if not accepted:
            sys.exit(f"Could not obtain a paraphrase for parent id={row['id']}")
        print("    ok")
    return out, seed_text, next_id


def _collect_extra_paraphrases(
    row: dict,
    n_need: int,
    corpus: list[str],
    reject_near: list[str],
    variant_start: int,
    next_id: int,
) -> tuple[list[dict], int]:
    """Up to n_need paraphrase rows, avoiding near-duplicates of reject_near."""
    built: list[dict] = []
    variant_idx = variant_start
    rounds = 0
    while len(built) < n_need and rounds < 5:
        rounds += 1
        ask = max(8, n_need * 2)
        prompt = gs._build_paraphrase_prompt(row, ask)
        candidates = gs._call_llm(prompt)
        for cand in candidates:
            if len(built) >= n_need:
                break
            text = str(cand.get("text", "")).strip()
            if not text:
                continue
            if reject_near and gs._is_near_duplicate(
                text, reject_near, gs.DEDUP_THRESHOLD
            ):
                continue
            if gs._is_near_duplicate(text, corpus, gs.DEDUP_THRESHOLD):
                continue
            corpus.append(text)
            built.append(
                gs._row_template(
                    row,
                    new_id=next_id,
                    similarity_tier="paraphrase",
                    text=text,
                    w_start=None,
                    w_end=None,
                    query_run_at=gs._generate_query_run_at(row["id"], variant_idx),
                )
            )
            next_id += 1
            variant_idx += 1
    return built, next_id


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate cache_seed + cache_test CSVs for Asteria bench.",
    )
    p.add_argument("--input", type=Path, default=gs.INPUT_CSV)
    p.add_argument("--seed-output", type=Path, default=Path("cache_seed.csv"))
    p.add_argument("--test-output", type=Path, default=Path("cache_test.csv"))
    p.add_argument("--warm-count", type=int, default=20)
    p.add_argument("--test-total", type=int, default=100)
    p.add_argument(
        "--warm-test-pct",
        type=int,
        default=60,
        metavar="PCT",
        help=(
            "Approximate percentage of test CSV rows that paraphrase *seeded* "
            "parents (default: 60). Remaining rows use cold parents (40%% with default)."
        ),
    )
    p.add_argument("--parent-sample-seed", type=int, default=42)
    p.add_argument(
        "--types",
        default=None,
        help="Optional comma filter (same as generate_scenarios).",
    )
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    if gs.MODEL.startswith("watsonx/"):
        missing = [v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID")
                   if not os.environ.get(v)]
        if missing:
            sys.exit(f"Missing env: {', '.join(missing)}")

    test_total = args.test_total
    warm_n = args.warm_count
    wpct = max(1, min(99, args.warm_test_pct))
    warm_test_rows = test_total * wpct // 100
    cold_test_rows = test_total - warm_test_rows

    if warm_test_rows < 1 or cold_test_rows < 1:
        sys.exit(
            f"Need at least 1 warm and 1 cold test row; got warm={warm_test_rows}, "
            f"cold={cold_test_rows} (test-total={test_total}, warm-test-pct={wpct})."
        )
    if warm_n < 1:
        sys.exit("--warm-count must be >= 1")

    cold_n = cold_test_rows

    rows = gs._load_rows(args.input)
    if args.types:
        keep = {t.strip() for t in args.types.split(",")}
        rows = [r for r in rows if r.get("type") in keep]
    if not rows:
        sys.exit("No rows after type filter.")

    rng = random.Random(args.parent_sample_seed)
    warm_rows, cold_canonical = _pick_warm_and_cold(rows, warm_n, cold_n, rng)
    warm_counts = _alloc_warm_counts(warm_n, warm_test_rows)
    if sum(warm_counts) != warm_test_rows:
        sys.exit("internal: warm paraphrase allocation mismatch")

    corpus: list[str] = [r["text"] for r in rows]
    next_id = max(int(r["id"]) for r in rows) + 1

    print(f"Warm (seed) parents     : {warm_n}")
    print(
        f"Test row split            : {warm_test_rows} warm ({wpct}%) + "
        f"{cold_test_rows} cold ({100 - wpct}%) = {test_total}"
    )
    print(f"Cold (held-out) parents   : {cold_n} (one test row each)")

    seed_rows, seed_text_by_parent, next_id = _emit_seed_rows(
        warm_rows, corpus, next_id
    )
    _write_csv(args.seed_output, seed_rows)
    print(f"\nWrote {len(seed_rows)} rows → {args.seed_output}")
    print(f"  tiers: {dict(Counter(r['similarity_tier'] for r in seed_rows))}")

    test_rows: list[dict] = []
    variant_base = 10
    for row, n_para in zip(warm_rows, warm_counts):
        pid = str(row["id"])
        st = seed_text_by_parent.get(pid, "")
        reject = [st, row["text"]] if st else [row["text"]]
        if n_para < 1:
            continue
        chunk, next_id = _collect_extra_paraphrases(
            row,
            n_para,
            corpus,
            reject,
            variant_base,
            next_id,
        )
        variant_base += 100
        if len(chunk) < n_para:
            print(
                f"  WARNING parent {pid}: only {len(chunk)}/{n_para} test paraphrases",
                flush=True,
            )
        test_rows.extend(chunk)

    for i, row in enumerate(cold_canonical, start=1):
        print(f"[cold {i}/{cold_n}] parent_id={row['id']}")
        chunk, next_id = _collect_extra_paraphrases(
            row,
            1,
            corpus,
            [row["text"]],
            0,
            next_id,
        )
        if not chunk:
            prompt = gs._build_paraphrase_prompt(row, 3)
            for cand in gs._call_llm(prompt):
                text = str(cand.get("text", "")).strip()
                if not text:
                    continue
                if gs._is_near_duplicate(text, corpus, gs.DEDUP_THRESHOLD):
                    continue
                corpus.append(text)
                chunk = [
                    gs._row_template(
                        row,
                        new_id=next_id,
                        similarity_tier="paraphrase",
                        text=text,
                        w_start=None,
                        w_end=None,
                        query_run_at=gs._generate_query_run_at(row["id"], 0),
                    )
                ]
                next_id += 1
                break
        if not chunk:
            sys.exit(f"No paraphrase for cold parent id={row['id']}")
        test_rows.extend(chunk)
        print("    ok")

    _write_csv(args.test_output, test_rows)
    print(f"\nWrote {len(test_rows)} rows → {args.test_output}")
    print(f"  tiers: {dict(Counter(r['similarity_tier'] for r in test_rows))}")
    warm_p = {str(r["id"]) for r in warm_rows}
    warm_rows_cnt = sum(1 for r in test_rows if str(r["parent_id"]) in warm_p)
    cold_rows_cnt = len(test_rows) - warm_rows_cnt
    print(f"  test rows warm / cold  : {warm_rows_cnt} / {cold_rows_cnt} (target {warm_test_rows}/{cold_test_rows})")


if __name__ == "__main__":
    main()
