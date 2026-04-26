"""Evaluate an Asteria cache on the labeled pair sets.

For each pair we:
  1. Build a *fresh* cache instance (so pairs are independent).
  2. Insert (seed_text, dummy_answer) into the cache.
  3. Look up probe_text — record HIT if a value is returned, MISS otherwise.
  4. Compare to the expected label, increment one of TP/FP/FN/TN.

Then we report precision, recall, F1, false-positive rate — overall and per
pair_type.

Usage (from the repo root)
──────────────────────────
    # baseline: ExactMatchCache (string-match only). Fast, validates the harness.
    PYTHONPATH=. python asteria_eval/run_asteria_eval.py --mode exact

    # full Asteria stack: embeddings + Sine + judger + LCFU + Markov.
    PYTHONPATH=. python asteria_eval/run_asteria_eval.py --mode full --capacity 256

The full mode requires an MPS or CUDA GPU and downloads ~1 GB of model weights
on first run.

Inputs
──────
    --pairs   default = asteria_eval/data/asteria_eval_pairs.csv  (97 pairs)
    --output  default = asteria_eval/data/asteria_eval_results.json

Hooking in your own cache
─────────────────────────
The script's only contract with the cache is the two methods on the wrapper:

    cache.insert(seed_text: str, answer: str) -> None
    cache.lookup(probe_text: str) -> tuple[Optional[str], dict]

Implement that contract for whichever cache variant you are testing and add a
new factory in this file (or pass it from another script).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── Cache factories ──────────────────────────────────────────────────────


def _exact_factory() -> Callable[[], Any]:
    """Self-contained string-match cache. Used as a sanity baseline; importing
    asteria.cache here would pull in sentence_transformers/torch which we don't
    need for the smoke test."""

    class _StringMatchCache:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        def insert(self, q: str, a: str) -> None:
            self.store[q] = a

        def lookup(self, q: str) -> str | None:
            return self.store.get(q)

    def make():
        return _ExactWrapper(_StringMatchCache())

    return make


def _full_factory(capacity: int = 256) -> Callable[[], Any]:
    """Build a fresh AsteriaCache (embeddings + judger + Sine + LCFU)."""
    # Heavy imports happen once; the *cache instance* is fresh per pair.
    from asteria.cache import AsteriaCache
    from asteria.config import DEFAULT_CONFIG, AsteriaConfig
    from asteria.embedding_model import EmbeddingModel
    from asteria.semantic_judger import SemanticJudger

    cfg = AsteriaConfig(**{**DEFAULT_CONFIG.__dict__, "cache_capacity": capacity})
    embed = EmbeddingModel()
    judger = SemanticJudger()

    def make():
        return _FullWrapper(AsteriaCache(embed, judger, cfg))

    return make


class _ExactWrapper:
    """Adapt ExactMatchCache to the same insert/lookup signature."""

    def __init__(self, c):
        self.c = c

    def insert(self, q: str, a: str) -> None:
        self.c.insert(q, a)

    def lookup(self, q: str) -> tuple[str | None, dict]:
        ans = self.c.lookup(q)
        return ans, {"hit": ans is not None}


class _FullWrapper:
    def __init__(self, c):
        self.c = c

    def insert(self, q: str, a: str) -> None:
        self.c.insert(q, a, cost=0.0, latency_ms=0.0)

    def lookup(self, q: str) -> tuple[str | None, dict]:
        ans, dbg = self.c.lookup(q)
        return ans, dbg


# ─── Evaluation core ──────────────────────────────────────────────────────


def evaluate(pairs: list[dict[str, Any]], factory: Callable[[], Any]) -> dict[str, Any]:
    cm: dict[str, int] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for p in pairs:
        cache = factory()
        cache.insert(p["seed_text"], f"<answer for seed {p['seed_id']}>")
        ans, dbg = cache.lookup(p["probe_text"])
        actual = "HIT" if ans is not None else "MISS"
        expected = p["expected"]
        if expected == "HIT" and actual == "HIT":
            key = "TP"
        elif expected == "HIT" and actual == "MISS":
            key = "FN"
        elif expected == "MISS" and actual == "HIT":
            key = "FP"
        else:
            key = "TN"
        cm[key] += 1
        by_type[p["pair_type"]][key] += 1
        rows.append({
            "pair_id": p["pair_id"],
            "pair_type": p["pair_type"],
            "expected": expected,
            "actual": actual,
            "label": key,
            "seed_id": p["seed_id"],
            "probe_id": p["probe_id"],
            "judger_score": dbg.get("judger_scores"),
        })

    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(cm)
    by_type_metrics = {k: compute_metrics(v) for k, v in by_type.items()}
    return {
        "elapsed_s": round(elapsed, 3),
        "n_pairs": len(pairs),
        "confusion_matrix": cm,
        "metrics": metrics,
        "per_pair_type": by_type_metrics,
        "rows": rows,
    }


def compute_metrics(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fpr, 3),
        "support": tp + fp + fn + tn,
    }


def load_pairs(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "pair_id": int(row["pair_id"]),
                "pair_type": row["pair_type"],
                "expected": row["expected"],
                "seed_id": int(row["seed_id"]),
                "seed_text": row["seed_text"],
                "probe_id": int(row["probe_id"]),
                "probe_text": row["probe_text"],
            })
    return out


def print_report(name: str, results: dict[str, Any]) -> None:
    cm = results["confusion_matrix"]
    m = results["metrics"]
    print()
    print(f"┏━━ {name}  ({results['n_pairs']} pairs, {results['elapsed_s']}s) ━━")
    print(f"┃ TP={cm['TP']:>4}   FP={cm['FP']:>4}   FN={cm['FN']:>4}   TN={cm['TN']:>4}")
    print(f"┃ precision={m['precision']:.3f}   recall={m['recall']:.3f}   "
          f"F1={m['f1']:.3f}   FPR={m['false_positive_rate']:.3f}")
    print("┃ per pair_type:")
    for ptype, pm in results["per_pair_type"].items():
        print(f"┃   {ptype:<18} support={pm['support']:>3}  "
              f"P={pm['precision']:.2f}  R={pm['recall']:.2f}  "
              f"F1={pm['f1']:.2f}  FPR={pm['false_positive_rate']:.2f}")
    print("┗")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DATA_DIR / "asteria_eval_pairs.csv"))
    ap.add_argument("--output", default=str(DATA_DIR / "asteria_eval_results.json"))
    ap.add_argument("--mode", choices=["exact", "full"], default="exact",
                    help="exact=ExactMatchCache baseline; full=AsteriaCache (embeddings+judger)")
    ap.add_argument("--capacity", type=int, default=256)
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    print(f"loaded {len(pairs)} pairs from {args.pairs}")
    by_type = Counter(p["pair_type"] for p in pairs)
    for k, v in by_type.items():
        print(f"  {k}: {v}")

    if args.mode == "exact":
        factory = _exact_factory()
        label = "ExactMatchCache (baseline)"
    else:
        factory = _full_factory(capacity=args.capacity)
        label = "AsteriaCache (full, embeddings+judger)"

    results = evaluate(pairs, factory)
    results["mode"] = args.mode
    print_report(label, results)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfull results written to {out_path}")


if __name__ == "__main__":
    main()
