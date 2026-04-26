"""Build labeled pair sets for evaluating Asteria cache precision/recall.

Source of truth: `asteria_eval/data/all_utterance.jsonl` — the raw 152-scenario
file mirrored from `ibm-research/AssetOpsBench` on Hugging Face.

This script does NOT use heuristic clustering. Every paraphrase pair and every
"hard negative" pair is **manually curated** against the raw text — see the
TRUE_PARAPHRASES and TRUE_HARD_NEGATIVES tables below. Each row carries a
`kind` (positives) or `confusion_type` (hard negatives) column so the labelling
is auditable downstream.

Easy negatives and volatile-repeat pairs remain auto-generated; the rules for
those (different MCP type + no shared asset; regex-detected time-sensitivity)
are simple enough to be reliable.

Run from the repo root:
    python asteria_eval/build_asteria_eval_pairs.py
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC_JSONL = ROOT / "data" / "all_utterance.jsonl"
OUT_DIR = ROOT / "data"
SEED = 42

random.seed(SEED)


# ─── Manually curated PARAPHRASE pairs ──────────────────────────────────────
# Every pair below was verified against the raw HF text. Same intent → same
# expected answer. The eval harness should treat these as expected = HIT.

TRUE_PARAPHRASES: list[tuple[int, int, str, str]] = [
    # ── verbatim duplicates (HF dataset has these as exact dupes) ──
    (601, 602, "verbatim",   "exact-duplicate text"),
    (606, 616, "verbatim",   "exact-duplicate text"),
    # ── 1xx FMSA ↔ 6xx multiagent: only difference is 'at MAIN site' ──
    (107, 605, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (108, 606, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (108, 616, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (109, 607, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (110, 608, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (111, 609, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (112, 610, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (113, 611, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (114, 612, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (115, 614, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (116, 615, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (117, 617, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (118, 618, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    (119, 619, "site_added", "FMSA vs multiagent: only 'at MAIN site' added"),
    # ── reworded: different surface, same intent ──
    (  1,   2, "reworded",   "list IoT sites — two phrasings"),
    (501, 503, "reworded",   "anomaly in Chiller 6 Tonnage week 2020-04-27 — two phrasings"),
    (516, 520, "reworded",   "forecast Chiller 6 energy consumption next week — two phrasings"),
]


# ─── Manually curated HARD NEGATIVE pairs ───────────────────────────────────
# Texts are nearly identical (≥0.85 char similarity) but the intent genuinely
# differs along one explicit dimension (asset / parameter / metric / time / …).
# A correctly-working semantic cache must NOT confuse these. These are the
# trap cases the judger has to handle.

TRUE_HARD_NEGATIVES: list[tuple[int, int, str, str]] = [
    # ── different_asset (one digit / specifier changes the equipment) ──
    (509, 514, "different_asset",       "Chiller 6 ↔ Chiller 9 (otherwise identical)"),
    (501, 517, "different_asset",       "Chiller 6 ↔ Chiller 9 anomaly in Tonnage"),
    (101, 102, "different_specificity", "asset 'Chiller' ↔ 'Chiller 6'"),
    # ── different_param (numeric parameter changes) ──
    (209, 210, "different_param",       "context length 96 ↔ 1024"),
    # ── different_operation (verb-class differs) ──
    (402, 403, "different_operation",   "preventive ↔ corrective work order details"),
    # ── different_time (year / month / week range) ──
    (407, 408, "different_time",        "2021 ↔ May 2020"),
    (405, 410, "different_time",        "all of June ↔ first week of June"),
    # ── different_metric (which sensor / measurement is forecasted) ──
    (502, 505, "different_metric",      "Condenser Water Flow ↔ Tonnage forecast"),
    (502, 507, "different_metric",      "Condenser Water Flow ↔ energy consumption forecast"),
    (505, 507, "different_metric",      "Tonnage ↔ energy forecast"),
    (511, 515, "different_metric",      "energy usage ↔ performance prediction"),
    (505, 516, "different_metric",      "Tonnage ↔ energy consumption forecast"),
    # ── different_sensor / sensor_set (same Q, different sensor list) ──
    (605, 606, "different_sensor_set",  "temp sensors ↔ temp + power input sensors"),
    (605, 616, "different_sensor_set",  "temp sensors ↔ temp + power input sensors"),
    (106, 107, "different_sensor",      "Supply Temperature ↔ general temperature sensors"),
    (107, 108, "different_sensor_set",  "temp sensors ↔ temp + power input sensors"),
    # ── different_site ──
    (120, 620, "different_site",        "POKMAIN ↔ MAIN site"),
    # ── different_feature (which TSFM capability is asked about) ──
    (207, 208, "different_feature",     "Anomaly Detection ↔ Time Series Classification support"),
    (203, 207, "different_feature",     "forecasting ↔ anomaly detection support"),
    # ── different_model (which TSFM model is asked about) ──
    (204, 205, "different_model",       "TTM ↔ LSTM"),
    (205, 206, "different_model",       "LSTM ↔ Chronos"),
    (204, 206, "different_model",       "TTM ↔ Chronos"),
]


# ─── Volatile detection (regex; user can audit and tune) ───────────────────

VOLATILE_RE = re.compile(
    r"\b(latest|today|now|currently|current|recent|right\s+now|live)\b",
    re.IGNORECASE,
)


def is_volatile(text: str) -> bool:
    return bool(VOLATILE_RE.search(text))


# ─── Helpers ────────────────────────────────────────────────────────────────


def load_scenarios() -> dict[int, dict[str, Any]]:
    if not SRC_JSONL.exists():
        raise SystemExit(
            f"missing {SRC_JSONL}; download it from "
            f"https://huggingface.co/datasets/ibm-research/AssetOpsBench/"
            f"resolve/main/data/scenarios/all_utterance.jsonl"
        )
    return {
        int(r["id"]): r
        for r in (json.loads(l) for l in SRC_JSONL.open(encoding="utf-8"))
    }


def _text_norm(t: str) -> str:
    return " ".join(str(t).lower().split())


# ─── Pair builders ─────────────────────────────────────────────────────────


def build_paraphrase_pairs(scen: dict[int, dict]) -> list[dict]:
    out: list[dict] = []
    for sid, pid, kind, why in TRUE_PARAPHRASES:
        if sid not in scen or pid not in scen:
            raise SystemExit(f"missing scenario id in TRUE_PARAPHRASES: {sid} or {pid}")
        out.append({
            "pair_type": "paraphrase",
            "expected": "HIT",
            "kind": kind,
            "rationale": why,
            "seed": scen[sid],
            "probe": scen[pid],
        })
    return out


def build_hard_negatives(scen: dict[int, dict]) -> list[dict]:
    out: list[dict] = []
    for sid, pid, ctype, why in TRUE_HARD_NEGATIVES:
        if sid not in scen or pid not in scen:
            raise SystemExit(f"missing scenario id in TRUE_HARD_NEGATIVES: {sid} or {pid}")
        out.append({
            "pair_type": "hard_negative",
            "expected": "MISS",
            "kind": ctype,
            "rationale": why,
            "seed": scen[sid],
            "probe": scen[pid],
        })
    return out


def build_easy_negatives(scen: dict[int, dict], target: int = 50) -> list[dict]:
    """Different MCP type AND no asset overlap. Heuristic auto-generation here
    is fine — the bar is "obviously different topic", which keyword filters
    handle reliably."""
    rows = list(scen.values())
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(out) < target and attempts < 10000:
        attempts += 1
        a, b = random.sample(rows, 2)
        if int(a["id"]) == int(b["id"]):
            continue
        key = tuple(sorted((int(a["id"]), int(b["id"]))))
        if key in seen:
            continue
        if is_volatile(a["text"]) or is_volatile(b["text"]):
            continue
        if _text_norm(a["text"]) == _text_norm(b["text"]):
            continue
        if str(a["type"]).lower() == str(b["type"]).lower():
            continue
        # crude shared-asset exclusion: scan for the same equipment id
        ta, tb = a["text"].lower(), b["text"].lower()
        if any(asset in ta and asset in tb for asset in
               ["chiller 6", "chiller 9", "chiller 7", "cwc04009", "cwc04013",
                "wind turbine"]):
            continue
        out.append({
            "pair_type": "easy_negative",
            "expected": "MISS",
            "kind": "different_topic",
            "rationale": (
                f"different MCP type ({a['type']} vs {b['type']}); no shared asset"
            ),
            "seed": a,
            "probe": b,
        })
        seen.add(key)
    return out


def build_volatile_pairs(scen: dict[int, dict]) -> list[dict]:
    """Volatile queries asked twice in a row (verbatim repeat). A correctly
    functioning cache must refuse to serve stale time-sensitive answers."""
    out: list[dict] = []
    for r in scen.values():
        if not is_volatile(r["text"]):
            continue
        out.append({
            "pair_type": "volatile_repeat",
            "expected": "MISS",
            "kind": "volatile_repeat",
            "rationale": "time-sensitive query; staticity gate must refuse to serve",
            "seed": r,
            "probe": r,
        })
    return out


# ─── Output ────────────────────────────────────────────────────────────────


CSV_COLS = [
    "pair_id", "pair_type", "expected", "kind", "rationale",
    "seed_id", "seed_type", "seed_text",
    "probe_id", "probe_type", "probe_text",
    "shared_type",
]


def to_row(idx: int, pair: dict[str, Any]) -> dict[str, Any]:
    s, p = pair["seed"], pair["probe"]
    return {
        "pair_id": idx,
        "pair_type": pair["pair_type"],
        "expected": pair["expected"],
        "kind": pair["kind"],
        "rationale": pair["rationale"],
        "seed_id": s["id"],
        "seed_type": s["type"],
        "seed_text": s["text"],
        "probe_id": p["id"],
        "probe_type": p["type"],
        "probe_text": p["text"],
        "shared_type": str(s["type"]).lower() == str(p["type"]).lower(),
    }


def write_csv(path: Path, pairs: list[dict], start_id: int) -> int:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for i, pair in enumerate(pairs):
            w.writerow(to_row(start_id + i, pair))
    return start_id + len(pairs)


def main() -> None:
    scen = load_scenarios()
    print(f"loaded {len(scen)} scenarios from {SRC_JSONL}")

    positives = build_paraphrase_pairs(scen)
    hard_neg = build_hard_negatives(scen)
    easy_neg = build_easy_negatives(scen, target=50)
    volatile = build_volatile_pairs(scen)

    print(f"  paraphrase      (HIT) : {len(positives):>3}  ← manually curated")
    print(f"  hard_negative  (MISS) : {len(hard_neg):>3}  ← manually curated")
    print(f"  easy_negative  (MISS) : {len(easy_neg):>3}")
    print(f"  volatile_repeat(MISS) : {len(volatile):>3}")
    total = len(positives) + len(hard_neg) + len(easy_neg) + len(volatile)
    print(f"  TOTAL pairs           : {total}")
    print(f"  expected distribution : HIT={len(positives)}  "
          f"MISS={len(hard_neg) + len(easy_neg) + len(volatile)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    next_id = write_csv(OUT_DIR / "asteria_eval_positives.csv", positives, 1)
    next_id = write_csv(OUT_DIR / "asteria_eval_hard_negatives.csv", hard_neg, next_id)
    next_id = write_csv(OUT_DIR / "asteria_eval_easy_negatives.csv", easy_neg, next_id)
    next_id = write_csv(OUT_DIR / "asteria_eval_volatile.csv", volatile, next_id)

    unified = positives + hard_neg + easy_neg + volatile
    write_csv(OUT_DIR / "asteria_eval_pairs.csv", unified, 1)

    summary = {
        "seed": SEED,
        "source": str(SRC_JSONL.relative_to(ROOT.parent)),
        "method": "manual_curation",
        "counts": {
            "paraphrase": len(positives),
            "hard_negative": len(hard_neg),
            "easy_negative": len(easy_neg),
            "volatile_repeat": len(volatile),
            "total": total,
        },
        "expected_distribution": dict(Counter(p["expected"] for p in unified)),
        "paraphrase_kind_breakdown": dict(
            Counter(p["kind"] for p in positives)
        ),
        "hard_negative_kind_breakdown": dict(
            Counter(p["kind"] for p in hard_neg)
        ),
        "files": {
            "unified": "asteria_eval/data/asteria_eval_pairs.csv",
            "positives": "asteria_eval/data/asteria_eval_positives.csv",
            "hard_negatives": "asteria_eval/data/asteria_eval_hard_negatives.csv",
            "easy_negatives": "asteria_eval/data/asteria_eval_easy_negatives.csv",
            "volatile": "asteria_eval/data/asteria_eval_volatile.csv",
        },
        "metric_definitions": {
            "TP": "expected=HIT, actual=HIT  → cache correctly served paraphrase",
            "FP": "expected=MISS, actual=HIT → cache served wrong / stale answer",
            "FN": "expected=HIT, actual=MISS → cache missed an obvious paraphrase",
            "TN": "expected=MISS, actual=MISS → cache correctly skipped",
            "precision": "TP / (TP + FP)",
            "recall": "TP / (TP + FN)",
            "f1": "2 · P · R / (P + R)",
            "false_positive_rate": "FP / (FP + TN)",
        },
    }
    (OUT_DIR / "asteria_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'asteria_eval_pairs.csv'} and per-category CSVs")
    print(f"summary: {OUT_DIR / 'asteria_eval_summary.json'}")


if __name__ == "__main__":
    main()
