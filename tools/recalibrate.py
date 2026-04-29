"""
Offline τ_lsm recalibration — Algorithm 1.

Builds a paraphrase-based validation set from cache_stress_test.QUERY_SCENARIOS,
scores every (query, candidate_answer) pair through the real SemanticJudger,
then computes the smallest τ_lsm that achieves the target precision.

Validation pair construction:
    Each scenario contributes one canonical answer (stub or hand-written).

    POSITIVES (is_correct=True):
        For every non-canonical variant in a scenario, pair it with that
        scenario's canonical answer.  Same intent → expected HIT.

    NEGATIVES (is_correct=False):
        Pair each variant with answers from other scenarios.  Different
        intent → expected MISS.  Sub-sampled to keep neg:pos ratio bounded.

The script does NOT mutate any cache or config.  It prints the suggested
τ_lsm and a precision curve; user updates asteria/config.py manually.

WARNING: Loads Qwen3-Reranker-0.6B (~30 s warm-up, ~3 GB RAM).  Run sparingly.

Usage:
    PYTHONPATH=src:. uv run python tools/recalibrate.py
    PYTHONPATH=src:. uv run python tools/recalibrate.py --target-precision 0.99
    PYTHONPATH=src:. uv run python tools/recalibrate.py --max-scenarios 5 --neg-ratio 2
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── stubbed canonical answers per scenario ───────────────────────────────────
# Realistic-shape stubs so the judger doesn't have to reason over nonsense.

_STUB_ANSWERS = {
    "assets_main":
        "Site MAIN contains the following assets: Chiller_3, Chiller_6, Chiller_9, "
        "CQPA AHU 1, CQPA AHU 2B.",
    "assets_main_rephrase2":
        "Asset IDs available at site MAIN: Chiller_3, Chiller_6, Chiller_9, "
        "CQPA AHU 1, CQPA AHU 2B.",
    "sensors_chiller6":
        "Sensors available for Chiller_6 at MAIN: Tonnage, % Loaded, Power Input, "
        "Supply Temperature, Return Temperature, Condenser Water Flow.",
    "sensors_chiller6_rephrase2":
        "Sensor tags on Chiller_6 in MAIN: Tonnage, % Loaded, Power Input, "
        "Supply Temperature, Return Temperature, Condenser Water Flow.",
    "sensors_chiller6_typoish":
        "Chiller_6 sensors at MAIN site: Tonnage, % Loaded, Power Input, "
        "Supply Temperature, Return Temperature, Condenser Water Flow.",
    "assets_plus_sensors_composite":
        "Site MAIN assets: Chiller_3, Chiller_6, Chiller_9, CQPA AHU 1, CQPA AHU 2B. "
        "Chiller_6 sensors: Tonnage, % Loaded, Power Input, Supply Temperature, "
        "Return Temperature, Condenser Water Flow.",
    "history_chiller6_window1":
        "Chiller_6 history at MAIN from 2020-06-01T00:00:00 to 2020-06-01T01:00:00: "
        "Tonnage avg 312.4, Power Input avg 215.8 kW, Supply Temp avg 6.7C.",
    "history_chiller6_window2":
        "Chiller_6 history at MAIN from 2020-06-01T01:00:00 to 2020-06-01T02:00:00: "
        "Tonnage avg 308.1, Power Input avg 213.2 kW, Supply Temp avg 6.8C.",
    "history_chiller6_window3":
        "Chiller_6 history at MAIN from 2020-06-01T02:00:00 to 2020-06-01T03:00:00: "
        "Tonnage avg 305.7, Power Input avg 210.4 kW, Supply Temp avg 6.9C.",
    "history_chiller6_open_ended":
        "Chiller_6 records at MAIN starting 2020-06-01T00:00:00: 24 hours of data, "
        "Tonnage range 280-340, Power Input range 195-225 kW.",
    "history_chiller6_typoish":
        "Chiller_6 historical observations MAIN 2020-06-01T00:00:00 to "
        "2020-06-01T01:00:00: Tonnage 312, Power 215 kW.",
    "history_chiller6_natural_date":
        "Chiller_6 readings at MAIN for June 1 2020 00:00 to 01:00: Tonnage avg 312.4, "
        "Power Input avg 215.8 kW.",
    "history_unknown_asset_control":
        "No history found for asset UNKNOWN_ASSET at site MAIN between 2020-06-01T00:00:00 "
        "and 2020-06-01T01:00:00 — asset does not exist in the inventory.",
    "sensors_unknown_asset_control":
        "No sensors registered for UNKNOWN_ASSET at site MAIN — asset does not exist.",
    "assets_invalid_site_control":
        "No assets found at site INVALID — site does not exist in the inventory.",
    "multi_turn_style_query":
        "MAIN assets: Chiller_3, Chiller_6, Chiller_9, CQPA AHU 1, CQPA AHU 2B. "
        "Chiller_6 sensors: Tonnage, % Loaded, Power Input, Supply Temperature, "
        "Return Temperature, Condenser Water Flow.",
    "assets_main_short":
        "MAIN assets: Chiller_3, Chiller_6, Chiller_9, CQPA AHU 1, CQPA AHU 2B.",
    "sensors_main_short":
        "Chiller_6 MAIN sensors: Tonnage, % Loaded, Power Input, Supply Temperature, "
        "Return Temperature, Condenser Water Flow.",
    "history_main_short":
        "Chiller_6 MAIN history 2020-06-01T00:00:00 to 2020-06-01T01:00:00: "
        "Tonnage 312, Power 215 kW.",
    "history_chiller6_window4":
        "Chiller_6 history at MAIN from 2020-06-01T03:00:00 to 2020-06-01T04:00:00: "
        "Tonnage avg 303.2, Power Input avg 208.9 kW, Supply Temp avg 7.0C.",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline τ_lsm recalibration (Algorithm 1).",
    )
    p.add_argument(
        "--target-precision",
        type=float,
        default=0.95,
        help="Required precision after recalibration (default: 0.95).",
    )
    p.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Cap scenarios processed (default: all).",
    )
    p.add_argument(
        "--neg-ratio",
        type=int,
        default=3,
        help="Max negatives per positive (default: 3).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for negative subsampling (default: 42).",
    )
    p.add_argument(
        "--curve-rows",
        type=int,
        default=20,
        help="Precision-curve rows to print (default: 20).",
    )
    return p


def _build_pairs(
    scenarios: list[dict],
    *,
    neg_ratio: int,
    seed: int,
) -> list[tuple[str, str, bool, str]]:
    """Build (query, candidate_answer, is_correct, label_scenario) tuples."""
    rng = random.Random(seed)
    pos: list[tuple[str, str, bool, str]] = []
    neg: list[tuple[str, str, bool, str]] = []

    for s in scenarios:
        name = s["name"]
        if name not in _STUB_ANSWERS:
            print(f"  skip scenario {name}: no stub answer", file=sys.stderr)
            continue
        own_answer = _STUB_ANSWERS[name]
        variants = s["variants"]
        # Positives — paraphrases of canonical query
        for variant in variants[1:]:
            pos.append((variant, own_answer, True, name))
        # Negatives — variants vs other scenarios' answers
        for other in scenarios:
            if other["name"] == name or other["name"] not in _STUB_ANSWERS:
                continue
            other_answer = _STUB_ANSWERS[other["name"]]
            for variant in variants[1:]:
                neg.append((variant, other_answer, False, name))

    cap = neg_ratio * len(pos)
    if cap and len(neg) > cap:
        neg = rng.sample(neg, cap)

    return pos + neg


def main() -> None:
    args = _build_parser().parse_args()

    # Lazy imports — Asteria deps are heavy.
    from cache_stress_test import QUERY_SCENARIOS
    from asteria.config import DEFAULT_CONFIG
    from asteria.recalibrator import Recalibrator
    from asteria.semantic_judger import SemanticJudger

    scenarios = QUERY_SCENARIOS
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]

    print(f"Building validation pairs over {len(scenarios)} scenarios …")
    pairs = _build_pairs(scenarios, neg_ratio=args.neg_ratio, seed=args.seed)
    n_pos = sum(1 for *_, ok, _name in pairs if ok)
    n_neg = len(pairs) - n_pos
    print(f"  positives = {n_pos}, negatives = {n_neg}, total = {len(pairs)}")

    print("Loading SemanticJudger (Qwen3-Reranker-0.6B) …")
    judger = SemanticJudger()

    recal = Recalibrator(target_precision=args.target_precision)

    print(f"Scoring {len(pairs)} pairs …")
    for i, (q, a, label, _name) in enumerate(pairs, start=1):
        score = judger.score(q, a)
        recal.record(q, a, score, label)
        if i % 25 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}", flush=True)

    summary = recal.summary()
    print("\n=== Recalibration result ===")
    print(f"  target precision   : {summary['target_precision']:.3f}")
    print(f"  suggested τ_lsm    : {summary['suggested_tau_lsm']}")
    print(f"  current   τ_lsm    : {DEFAULT_CONFIG.tau_lsm:.3f}")
    print(f"  validation entries : {summary['n_entries']} "
          f"(pos={summary['n_positives']}, neg={summary['n_negatives']})")

    curve = recal.precision_curve()
    print(f"\nPrecision curve (top {min(args.curve_rows, len(curve))} by score):")
    print(f"  {'τ':>7s}  {'precision':>9s}  {'kept':>5s}  {'tp':>5s}")
    for tau, prec, kept, tp in curve[: args.curve_rows]:
        print(f"  {tau:7.4f}  {prec:9.4f}  {kept:5d}  {tp:5d}")


if __name__ == "__main__":
    main()
