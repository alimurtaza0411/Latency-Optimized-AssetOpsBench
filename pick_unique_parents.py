"""
Filter all_utterance.csv to a semantically-unique subset.

Pairs of parents asking the same thing under different parent_ids cause
labeling-artifact "FPs" in the cache benchmark: the cache returns a correct
answer but the parent_id-based ground truth flags it wrong. This script
greedily picks parents whose max pairwise cosine to all already-kept parents
is below a threshold, within the same `type`.

Output:
  unique_utterance.csv — same schema as input, kept rows only
  Console report      — kept/dropped counts + a few example duplicate pairs

Usage:
  PYTHONPATH=. .venv/bin/python pick_unique_parents.py \
      --input all_utterance.csv \
      --output unique_utterance.csv \
      --threshold 0.80
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _greedy_unique(
    rows: list[dict],
    embeddings: np.ndarray,
    threshold: float,
    rng: random.Random,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Return (kept_indices, dropped_pairs).

    Greedy: walk indices in random order; keep i iff max cosine(i, kept) < threshold.
    dropped_pairs[i] = (dropped_index, kept_index_that_caused_drop, cosine).
    """
    n = len(rows)
    order = list(range(n))
    rng.shuffle(order)

    kept: list[int] = []
    dropped: list[tuple[int, int, float]] = []
    for i in order:
        if not kept:
            kept.append(i)
            continue
        sims = embeddings[i] @ embeddings[kept].T  # already L2-normalised
        j_best = int(np.argmax(sims))
        s_best = float(sims[j_best])
        if s_best < threshold:
            kept.append(i)
        else:
            dropped.append((i, kept[j_best], s_best))
    return sorted(kept), dropped


def main() -> None:
    p = argparse.ArgumentParser(prog="pick_unique_parents")
    p.add_argument("--input", type=Path, default=Path("all_utterance.csv"))
    p.add_argument("--output", type=Path, default=Path("unique_utterance.csv"))
    p.add_argument(
        "--threshold", type=float, default=0.80,
        help="Drop parent if max cosine to any already-kept parent ≥ threshold "
             "(within the same type). Default 0.80. Tighter = more aggressive dedup.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the greedy walk order (default 42).",
    )
    p.add_argument(
        "--show-examples", type=int, default=8,
        help="Print N example dropped (kept_query, dropped_query, cosine) tuples.",
    )
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    rows = _load_rows(args.input)
    if not rows:
        sys.exit("Input is empty.")
    fieldnames = list(rows[0].keys())
    if "type" not in fieldnames or "text" not in fieldnames:
        sys.exit("Input must have 'type' and 'text' columns.")

    # Encode all texts in one batch (fast on MPS).
    print(f"Loading embedding model …", flush=True)
    from asteria.embedding_model import EmbeddingModel
    emb = EmbeddingModel()

    print(f"Encoding {len(rows)} utterances …", flush=True)
    embeddings = emb.encode([r["text"] for r in rows])

    # Group by type so we only dedup within same agent domain.
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_type[r.get("type", "")].append(i)

    rng = random.Random(args.seed)

    all_kept: list[int] = []
    all_dropped: list[tuple[int, int, float]] = []
    for t, idxs in sorted(by_type.items()):
        if len(idxs) <= 1:
            all_kept.extend(idxs)
            continue
        sub_rows = [rows[i] for i in idxs]
        sub_emb = embeddings[idxs]
        kept_local, dropped_local = _greedy_unique(
            sub_rows, sub_emb, args.threshold, rng,
        )
        # Translate local indices back to global.
        all_kept.extend(idxs[k] for k in kept_local)
        for d_local, k_local, s in dropped_local:
            all_dropped.append((idxs[d_local], idxs[k_local], s))

    all_kept_set = set(all_kept)
    kept_rows = [rows[i] for i in sorted(all_kept_set)]

    _write_rows(args.output, kept_rows, fieldnames)

    # ── Report ──────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"Total parents      : {len(rows)}")
    print(f"Kept               : {len(kept_rows)}")
    print(f"Dropped (duplicate): {len(all_dropped)}")
    print(f"Threshold          : {args.threshold}")
    print(f"Output             : {args.output}")
    print()

    # By-type breakdown
    type_kept: dict[str, int] = defaultdict(int)
    type_total: dict[str, int] = defaultdict(int)
    for i, r in enumerate(rows):
        type_total[r.get("type", "")] += 1
        if i in all_kept_set:
            type_kept[r.get("type", "")] += 1
    print("Per-type kept / total:")
    for t in sorted(type_total):
        print(f"  {t:<12}  {type_kept[t]:>3} / {type_total[t]:>3}")
    print()

    # Example dropped pairs (highest similarity first)
    if all_dropped and args.show_examples > 0:
        print(f"Top {args.show_examples} drop reasons (highest cosine first):")
        all_dropped.sort(key=lambda x: -x[2])
        for d_idx, k_idx, sim in all_dropped[: args.show_examples]:
            d_text = rows[d_idx]["text"][:80]
            k_text = rows[k_idx]["text"][:80]
            print(f"  cos={sim:.3f}  type={rows[d_idx].get('type', '?')}")
            print(f"    DROPPED  id={rows[d_idx]['id']}: {d_text!r}")
            print(f"    kept    id={rows[k_idx]['id']}: {k_text!r}")


if __name__ == "__main__":
    main()
