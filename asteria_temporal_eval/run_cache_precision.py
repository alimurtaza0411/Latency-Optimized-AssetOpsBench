"""Measure cache precision on the paraphrase set — *no agent runs*.

This is the **fast** harness: it instantiates an Asteria cache directly, warms
it with the 20 originals (each inserted with a unique sentinel as the cached
"answer"), then looks up each of the 40 paraphrases. Because the inserted
"answer" encodes the parent_id, we can tell whether a paraphrase that hits
hit the *correct* parent or some other one.

Confusion matrix (paraphrases only):
    TP : paraphrase hit cache AND returned its own parent's sentinel  (correct route)
    FP : paraphrase hit cache BUT returned some other parent's sentinel (cross-routed)
    FN : paraphrase missed the cache entirely
    TN : N/A in this experiment — every paraphrase is *expected* to hit

Reported metrics:
    Precision = TP / (TP + FP)        — when cache says HIT, how often is the route correct?
    Recall    = TP / (TP + FN)        — of all paraphrases, how many were correctly served?
    F1        = harmonic mean
    Cross-route rate = FP / (TP + FP) — share of HITs that went to the wrong original

Per-temporal-class breakdowns let you see whether the cache mishandles a
specific temporal type (e.g., 'volatile' should never hit; 'static' should
always hit).

Usage (from the repo root):
    python asteria_temporal_eval/run_cache_precision.py
    python asteria_temporal_eval/run_cache_precision.py --capacity 256

Requires: torch, sentence-transformers, faiss-cpu, transformers (i.e. the
full Asteria stack). For a lightweight smoke test without those deps, pass
--mode lightweight to use the difflib-based QueryIntentCache instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SENTINEL = "PARENT_QID_{qid}"  # cached "answer" body; uniquely identifies parent


def build_full_asteria(capacity: int):
    """Construct a fresh AsteriaCache (paper stack: embeddings + Sine + judger)."""
    from asteria.cache import AsteriaCache
    from asteria.config import DEFAULT_CONFIG, AsteriaConfig
    from asteria.embedding_model import EmbeddingModel
    from asteria.semantic_judger import SemanticJudger

    cfg = AsteriaConfig(**{**DEFAULT_CONFIG.__dict__, "cache_capacity": capacity})
    return AsteriaCache(EmbeddingModel(), SemanticJudger(), cfg)


def build_lightweight():
    """Difflib-based pre-planner cache used by `timer.py --query-cache`."""
    from asteria.integrations.assetops import QueryIntentCache

    cache = QueryIntentCache()

    class _Adapter:
        def __init__(self) -> None:
            self.c = cache

        def insert(self, q: str, body: str, **kwargs) -> None:
            self.c.store(q, {"body": body})

        def lookup(self, q: str):
            hit, payload = self.c.lookup(q)
            if hit and payload is not None:
                return payload.get("body"), {"hit": True}
            return None, {"hit": False}

    return _Adapter()


def normalise_lookup_result(ans):
    """Both AsteriaCache.lookup and the lightweight adapter return (ans, dbg).
    AsteriaCache.lookup gives a tuple; the adapter also gives a tuple. This
    helper just makes sure we have (str|None, dict)."""
    if isinstance(ans, tuple) and len(ans) == 2:
        return ans
    return ans, {}


def load_paraphrase_set() -> list[dict[str, Any]]:
    path = DATA_DIR / "paraphrases.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            f"run `python asteria_temporal_eval/generate_paraphrases.py` first."
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "query_id":         int(r["query_id"]),
                "role":             r["role"],
                "parent_query_id":  int(r["parent_query_id"]),
                "mcp_type":         r["mcp_type"],
                "temporal_class":   r["temporal_class"],
                "gen_temperature":  float(r["gen_temperature"]) if r["gen_temperature"] else 0.0,
                "text":             r["text"],
            })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_paraphrase_set()
    originals  = [r for r in rows if r["role"] == "original"]
    paraphrases = [r for r in rows if r["role"] == "paraphrase" and r["text"].strip()]
    skipped    = [r for r in rows if r["role"] == "paraphrase" and not r["text"].strip()]

    print(f"loaded {len(rows)} rows: {len(originals)} originals, "
          f"{len(paraphrases)} paraphrases ({len(skipped)} skipped — empty text)")

    if args.mode == "full":
        print(f"building full AsteriaCache (capacity={args.capacity})…")
        cache = build_full_asteria(args.capacity)
    else:
        print("using lightweight QueryIntentCache (difflib)…")
        cache = build_lightweight()

    # ── warm the cache with originals ──────────────────────────────────────
    print("\nwarming cache with originals…")
    insert_costs = []
    for o in originals:
        body = SENTINEL.format(qid=o["query_id"])
        t0 = time.perf_counter()
        cache.insert(o["text"], body, cost=0.0, latency_ms=0.0)
        insert_costs.append(time.perf_counter() - t0)
    print(f"  inserted {len(originals)} originals (avg insert: "
          f"{1000 * sum(insert_costs)/max(1,len(insert_costs)):.1f} ms)")

    # ── lookup each paraphrase ─────────────────────────────────────────────
    print("\nlooking up paraphrases…")
    cm = {"TP": 0, "FP": 0, "FN": 0}
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    by_temp: dict[float, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    detail: list[dict[str, Any]] = []

    for p in paraphrases:
        expected_body = SENTINEL.format(qid=p["parent_query_id"])
        t0 = time.perf_counter()
        ans, _dbg = normalise_lookup_result(cache.lookup(p["text"]))
        lookup_ms = 1000 * (time.perf_counter() - t0)

        if ans is None:
            label = "FN"
            served_qid = None
        elif ans == expected_body:
            label = "TP"
            served_qid = p["parent_query_id"]
        else:
            label = "FP"
            try:
                served_qid = int(ans.split("_")[-1])
            except Exception:  # noqa: BLE001
                served_qid = -1

        cm[label] += 1
        by_class[p["temporal_class"]][label] += 1
        by_temp[p["gen_temperature"]][label] += 1
        by_type[p["mcp_type"]][label] += 1
        detail.append({
            "query_id":         p["query_id"],
            "parent_query_id":  p["parent_query_id"],
            "served_query_id":  served_qid,
            "temporal_class":   p["temporal_class"],
            "mcp_type":         p["mcp_type"],
            "gen_temperature":  p["gen_temperature"],
            "label":            label,
            "lookup_ms":        round(lookup_ms, 2),
            "text":             p["text"],
        })

    return {
        "mode":               args.mode,
        "n_originals":        len(originals),
        "n_paraphrases":      len(paraphrases),
        "n_skipped":          len(skipped),
        "confusion_matrix":   cm,
        "metrics":            metrics(cm),
        "per_temporal_class": {k: {**v, **metrics(v)} for k, v in by_class.items()},
        "per_temperature":    {f"T={k}": {**v, **metrics(v)} for k, v in by_temp.items()},
        "per_mcp_type":       {k: {**v, **metrics(v)} for k, v in by_type.items()},
        "detail":             detail,
    }


def metrics(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = cm.get("TP", 0), cm.get("FP", 0), cm.get("FN", 0)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    cross = fp / (tp + fp) if (tp + fp) else 0.0
    return {
        "precision":        round(p, 3),
        "recall":           round(r, 3),
        "f1":               round(f1, 3),
        "cross_route_rate": round(cross, 3),
        "support":          tp + fp + fn,
    }


def print_report(res: dict[str, Any]) -> None:
    cm, m = res["confusion_matrix"], res["metrics"]
    print(f"\n┏━━ cache precision ({res['mode']}) ━━ "
          f"{res['n_paraphrases']} paraphrases, {res['n_originals']} originals warmed ━━")
    print(f"┃ TP={cm['TP']:>3}  FP={cm['FP']:>3}  FN={cm['FN']:>3}")
    print(f"┃ precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}  cross_route={m['cross_route_rate']:.3f}")
    print("┃")
    print("┃ by temporal class:")
    for cls in ["static", "anchored", "volatile", "relative"]:
        if cls in res["per_temporal_class"]:
            v = res["per_temporal_class"][cls]
            print(f"┃   {cls:<10} support={v['support']:>2}  P={v['precision']:.2f}  "
                  f"R={v['recall']:.2f}  F1={v['f1']:.2f}  cross={v['cross_route_rate']:.2f}")
    print("┃")
    print("┃ by paraphrase generation temperature:")
    for T_label, v in sorted(res["per_temperature"].items()):
        print(f"┃   {T_label:<8} support={v['support']:>2}  P={v['precision']:.2f}  "
              f"R={v['recall']:.2f}  F1={v['f1']:.2f}")
    print("┃")
    print("┃ by MCP type:")
    for k, v in res["per_mcp_type"].items():
        print(f"┃   {k:<11} support={v['support']:>2}  P={v['precision']:.2f}  "
              f"R={v['recall']:.2f}  F1={v['f1']:.2f}")
    print("┗")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lightweight"], default="full",
                    help="full = AsteriaCache (paper stack); lightweight = QueryIntentCache")
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--output", default=str(DATA_DIR / "cache_precision_results.json"))
    args = ap.parse_args()

    res = run(args)
    print_report(res)

    out = Path(args.output)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
