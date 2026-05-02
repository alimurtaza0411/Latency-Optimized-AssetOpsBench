#!/usr/bin/env python3
"""
Scenario Generator for AssetOpsBench Cache Testing
==================================================

Builds an augmented CSV from all_utterance.csv covering ALL query types
(IoT, Workorder, TSFM, multiagent, FMSA) — not just IoT.

For each input row the script can produce two kinds of derived rows, both
tagged with a `parent_id` pointing back to the source row so a cache test
can assert that paraphrase rows hit the same SE as their canonical:

  similarity_tier="paraphrase"
      Same intent / entity / parameters, only the natural-language phrasing
      changes.  These are strong cache-HIT candidates.

  similarity_tier="shifted_anchored"
      Only emitted for temporally-eligible queries (those that already
      contain — or could contain — a concrete date window).  The synthetic
      timestamp window is injected explicitly into the paraphrase so the
      Asteria temporal classifier routes it through the ANCHORED bucket.
      Each shifted_anchored row also stores the synthetic window in the
      output columns `synthetic_window_start` / `synthetic_window_end`,
      enabling window-match cache tests.

Why this matters for caching:
    The original 152 utterances are a *seed* — production queries will be
    paraphrases of these, often with different concrete windows.  The
    augmented CSV gives a controlled benchmark for both semantic-only
    cache hits (paraphrase) and temporal-gate behaviour (shifted_anchored).

The script is type-agnostic: it picks a system context based on the row's
`type` column (IoT|Workorder|TSFM|multiagent|FMSA).  For unsupported types
it falls back to a generic prompt.

Requirements:
    litellm — already a project dependency

Usage:
    PYTHONPATH=. uv run python generate_scenarios.py
    PYTHONPATH=. uv run python generate_scenarios.py --types IoT,Workorder
    PYTHONPATH=. uv run python generate_scenarios.py --paraphrases-per-row 5
    PYTHONPATH=. uv run python generate_scenarios.py --no-shifted-anchored

Inputs:
    all_utterance.csv  (must exist in this directory)

Outputs:
    augmented_utterances.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

try:
    import litellm
except ImportError:
    sys.exit(
        "Error: 'litellm' package not found.  Run:  uv sync"
    )


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV  = Path(__file__).parent / "all_utterance.csv"
OUTPUT_CSV = Path(__file__).parent / "augmented_utterances.csv"

MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

# Per-row generation budget.  Tunable via CLI.
DEFAULT_PARAPHRASES_PER_ROW    = 3
DEFAULT_ANCHORED_SHIFTS_PER_ROW = 2

DEDUP_THRESHOLD = 0.85   # SequenceMatcher ratio above which a candidate is dropped


# ── Type-specific system context ──────────────────────────────────────────────

_SYSTEM_CONTEXTS = {
    "IoT": (
        "Domain: IoT sensor data at site MAIN. "
        "Assets include Chiller 3, Chiller 6, Chiller 9, CQPA AHU 1, CQPA AHU 2B. "
        "Sensors include Tonnage, % Loaded, Power Input, Supply/Return Temperature, "
        "Condenser Water Flow, Supply Humidity. "
        "Data spans 2015–2020."
    ),
    "Workorder": (
        "Domain: maintenance work-order operations.  Equipment is identified by "
        "asset codes such as CWC04009.  Work orders link to alerts, anomalies, and "
        "failure modes.  Typical queries summarise events, recommend new work "
        "orders, or analyse failure histories."
    ),
    "TSFM": (
        "Domain: time-series foundation-model operations.  Queries cover model "
        "support (TTM/Granite), forecasting, anomaly detection over historical "
        "sensor streams, and model selection."
    ),
    "FMSA": (
        "Domain: failure-mode and symptom analysis.  Queries cover failure modes, "
        "diagnostic recommendations, fault-tree relationships, and symptoms for "
        "industrial equipment."
    ),
    "multiagent": (
        "Domain: multi-agent industrial workflows that combine IoT readings, "
        "work-order management, time-series forecasting, and failure-mode analysis. "
        "Queries are typically multi-step decision-support requests."
    ),
}

_DEFAULT_CONTEXT = (
    "Domain: industrial asset operations.  Combines IoT telemetry, work orders, "
    "time-series forecasting, and failure-mode analysis."
)


# ── Output guidelines (embedded in every prompt) ──────────────────────────────

_OUTPUT_GUIDELINES = """
Return a JSON ARRAY of OBJECTS.  Each object must have EXACTLY these keys:

  text  — the rephrased natural-language query (string)

Do NOT add any other keys.  Do NOT wrap the array.  Output ONLY the JSON
array, no markdown, no commentary.
"""


# ── Prompt templates ──────────────────────────────────────────────────────────

_PARAPHRASE_PROMPT = """\
Generate {n} natural-language paraphrases of the query below.

PARAPHRASE RULES:
  • Keep the same entity, operation type, parameters, and time references.
  • Change only the wording: synonyms, reorder clauses, change voice or tone,
    switch between question and imperative.
  • Each paraphrase MUST be semantically identical to the original.
  • Do NOT change asset names, sensor names, equipment codes, dates, or sites.
  • Do NOT generate near-duplicate paraphrases of each other.

{guidelines}

Domain context:
{context}

Original query:
{original}
"""

_ANCHORED_SHIFT_PROMPT = """\
Generate {n} natural-language paraphrases of the query below, but REPLACE
the original time references with the explicit ISO time window provided.

REWRITE RULES:
  • Keep the same entity, operation type, parameters.
  • The rewritten query MUST contain the new ISO window explicitly:
        "{window_start}"  to  "{window_end}"
  • Vary the phrasing of how the window is introduced ("from X to Y",
    "between X and Y", "for the period X – Y", etc.).
  • Do NOT keep relative phrases like "yesterday" or "last week" — replace
    them with the absolute window.
  • Do NOT change asset names, sensor names, equipment codes, or sites.

{guidelines}

Domain context:
{context}

Original query:
{original}
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_temporally_eligible(query: str) -> bool:
    """A query is eligible for shifted-anchored augmentation if it contains
    or strongly implies a time component but is NOT a live-state query."""
    q = query.lower()
    # Skip live-state queries — those should never produce ANCHORED variants.
    if re.search(
        r"\b(current(ly)?|right now|live|real[\-\s]?time|latest|now)\b", q
    ):
        return False
    # Eligible if it has any temporal signal we could anchor.
    return bool(
        re.search(
            r"\b(\d{4}|\d{4}-\d{2}-\d{2}|january|february|march|april|may|"
            r"june|july|august|september|october|november|december|"
            r"yesterday|today|last\s+(week|month|year|hour|day|night|shift)|"
            r"this\s+(week|month|year|morning|afternoon)|"
            r"past\s+\d+|history|trend|log|"
            r"between|from|to|during|over)\b",
            q,
        )
    )


def _generate_synthetic_window(
    parent_id: str | int,
    shift_index: int,
    span_hours_choices: tuple[int, ...] = (1, 3, 6, 24, 168),
) -> tuple[str, str]:
    """Pick an absolute window in 2018–2020 deterministically from parent_id.

    Same (parent_id, shift_index) always produces the same window — guarantees
    that seed and test CSVs aligned by parent_id will share windows so that
    shifted_anchored test queries can hit the cache by exact-window match.
    """
    seed_str = f"{parent_id}-{shift_index}"
    rng = random.Random(seed_str)
    start_year = rng.choice([2018, 2019, 2020])
    start_month = rng.randint(1, 11)
    start_day = rng.randint(1, 27)
    start_hour = rng.randint(0, 23)
    span_h = rng.choice(span_hours_choices)
    start = _dt.datetime(start_year, start_month, start_day, start_hour, 0, 0)
    end = start + _dt.timedelta(hours=span_h)
    iso_fmt = "%Y-%m-%dT%H:%M:%S"
    return start.strftime(iso_fmt), end.strftime(iso_fmt)


def _generate_query_run_at(
    parent_id: str | int,
    variant_index: int,
    *,
    range_start: _dt.datetime = _dt.datetime(2024, 1, 1),
    range_end:   _dt.datetime = _dt.datetime(2026, 4, 1),
) -> str:
    """Pick a simulated 'when was this query asked' timestamp.

    Deterministic per (parent_id, variant_index) so seed and test CSVs
    aligned by parent + variant share the same query_run_at — and so
    relative phrases like 'yesterday' resolve to the same concrete window
    on both sides of the benchmark, enabling cache HITs.
    """
    seed_str = f"qra-{parent_id}-{variant_index}"
    rng = random.Random(seed_str)
    span_seconds = int((range_end - range_start).total_seconds())
    offset = rng.randint(0, span_seconds)
    chosen = range_start + _dt.timedelta(seconds=offset)
    return chosen.replace(microsecond=0).isoformat()


def _is_near_duplicate(candidate: str, corpus: list[str], threshold: float) -> bool:
    c = candidate.lower().strip()
    for existing in corpus:
        if SequenceMatcher(None, c, existing.lower().strip()).ratio() >= threshold:
            return True
    return False


def _call_llm(prompt: str) -> list[dict]:
    kwargs: dict = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens":  2048,
    }
    if MODEL.startswith("watsonx/"):
        kwargs["api_key"]    = os.environ["WATSONX_APIKEY"]
        kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
        if url := os.environ.get("WATSONX_URL"):
            kwargs["api_base"] = url
    else:
        kwargs["api_key"]  = os.environ["LITELLM_API_KEY"]
        kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

    response = litellm.completion(**kwargs)
    raw = response.choices[0].message.content.strip()
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_paraphrase_prompt(row: dict, n: int) -> str:
    return _PARAPHRASE_PROMPT.format(
        n=n,
        guidelines=_OUTPUT_GUIDELINES,
        context=_SYSTEM_CONTEXTS.get(row["type"], _DEFAULT_CONTEXT),
        original=row["text"],
    )


def _build_anchored_shift_prompt(row: dict, n: int, w_start: str, w_end: str) -> str:
    return _ANCHORED_SHIFT_PROMPT.format(
        n=n,
        guidelines=_OUTPUT_GUIDELINES,
        context=_SYSTEM_CONTEXTS.get(row["type"], _DEFAULT_CONTEXT),
        original=row["text"],
        window_start=w_start,
        window_end=w_end,
    )


def _row_template(parent: dict, *, new_id: int, similarity_tier: str,
                  text: str, w_start: str | None, w_end: str | None,
                  query_run_at: str) -> dict:
    return {
        "id":                       new_id,
        "parent_id":                parent["id"],
        "type":                     parent["type"],
        "text":                     text,
        "category":                 parent.get("category", ""),
        "deterministic":            parent.get("deterministic", ""),
        "characteristic_form":      parent.get("characteristic_form", ""),
        "group":                    parent.get("group", ""),
        "entity":                   parent.get("entity", ""),
        "note":                     parent.get("note", ""),
        "similarity_tier":          similarity_tier,
        "synthetic_window_start":   w_start or "",
        "synthetic_window_end":     w_end or "",
        "query_run_at":             query_run_at,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate paraphrase + synthetic-window scenarios from all_utterance.csv.",
    )
    p.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[INPUT_CSV],
        help=(
            "One or more input CSV paths.  When multiple are given they are "
            "concatenated in order before processing — useful for mixing "
            "temporal and static utterances at controlled ratios."
        ),
    )
    p.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Output CSV path.")
    p.add_argument(
        "--types",
        default=None,
        help="Comma-separated subset of types to process (default: all). "
             "Example: IoT,Workorder",
    )
    p.add_argument(
        "--paraphrases-per-row",
        type=int,
        default=DEFAULT_PARAPHRASES_PER_ROW,
    )
    p.add_argument(
        "--anchored-shifts-per-row",
        type=int,
        default=DEFAULT_ANCHORED_SHIFTS_PER_ROW,
    )
    p.add_argument(
        "--no-shifted-anchored",
        action="store_true",
        help="Skip the synthetic-window augmentation pass.",
    )
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap input rows processed (default: all).")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    for p in args.input:
        if not p.exists():
            sys.exit(f"Error: input file not found — {p}")

    if MODEL.startswith("watsonx/"):
        missing = [v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID")
                   if not os.environ.get(v)]
        if missing:
            sys.exit(
                f"Error: missing env var(s): {', '.join(missing)}.  "
                "Check your .env file."
            )

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for p in args.input:
        chunk = _load_rows(p)
        rows.extend(chunk)
        print(f"  loaded {len(chunk)} rows from {p}")
    if args.types:
        keep = {t.strip() for t in args.types.split(",")}
        rows = [r for r in rows if r["type"] in keep]
    if args.max_rows:
        rows = rows[: args.max_rows]

    print(f"Processing {len(rows)} input rows "
          f"(types={Counter(r['type'] for r in rows)})")

    next_id = max(int(r["id"]) for r in rows) + 1
    corpus = [r["text"] for r in rows]
    out_rows: list[dict] = []

    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] id={row['id']} type={row['type']} : "
              f"{row['text'][:60]!r}")

        # 1. Paraphrase pass — always run.
        paraphrase_prompt = _build_paraphrase_prompt(row, args.paraphrases_per_row)
        candidates = _call_llm(paraphrase_prompt)
        accepted_paraphrases = 0
        variant_idx = 0
        for cand in candidates:
            text = str(cand.get("text", "")).strip()
            if not text:
                continue
            if _is_near_duplicate(text, corpus, DEDUP_THRESHOLD):
                continue
            corpus.append(text)
            out_rows.append(
                _row_template(
                    row,
                    new_id=next_id,
                    similarity_tier="paraphrase",
                    text=text,
                    w_start=None,
                    w_end=None,
                    query_run_at=_generate_query_run_at(row["id"], variant_idx),
                )
            )
            next_id += 1
            variant_idx += 1
            accepted_paraphrases += 1
        print(f"    paraphrases accepted: {accepted_paraphrases}")

        # 2. Shifted-anchored pass — opt-out, eligibility-gated.
        if args.no_shifted_anchored:
            continue
        if not _is_temporally_eligible(row["text"]):
            continue

        for shift_index in range(args.anchored_shifts_per_row):
            w_start, w_end = _generate_synthetic_window(row["id"], shift_index)
            shift_prompt = _build_anchored_shift_prompt(row, 1, w_start, w_end)
            shift_candidates = _call_llm(shift_prompt)
            for cand in shift_candidates:
                text = str(cand.get("text", "")).strip()
                if not text:
                    continue
                if _is_near_duplicate(text, corpus, DEDUP_THRESHOLD):
                    continue
                # Sanity: emitted text must mention either window endpoint.
                if w_start not in text and w_end not in text:
                    continue
                corpus.append(text)
                # query_run_at is independent of the shifted window — it
                # represents when the test harness will simulate the user
                # asking this query.
                qra_idx = 1000 + shift_index   # offset to keep stable
                out_rows.append(
                    _row_template(
                        row,
                        new_id=next_id,
                        similarity_tier="shifted_anchored",
                        text=text,
                        w_start=w_start,
                        w_end=w_end,
                        query_run_at=_generate_query_run_at(row["id"], qra_idx),
                    )
                )
                next_id += 1

    if not out_rows:
        sys.exit("No augmented rows generated.  Check API keys + model access.")

    fieldnames = [
        "id", "parent_id", "type", "text",
        "category", "deterministic", "characteristic_form",
        "group", "entity", "note",
        "similarity_tier",
        "synthetic_window_start", "synthetic_window_end",
        "query_run_at",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = Counter(r["similarity_tier"] for r in out_rows)
    print(f"\nWrote {len(out_rows)} rows to {args.output}")
    for tier, n in counts.items():
        print(f"  {tier:<20s}  {n}")


if __name__ == "__main__":
    main()
