#!/usr/bin/env python3
"""
Split all_utterance.csv into two CSVs based on temporal eligibility.

Produces:
  temporal_utterances.csv  — rows whose text contains a date / "yesterday" /
                             "history" / "trend" / etc., AND is NOT a live-state
                             query.  These are the rows that, when fed through
                             generate_scenarios.py, will produce both paraphrase
                             and shifted_anchored variants — the rows that
                             actually exercise the temporal filter at cache time.

  static_utterances.csv    — everything else (knowledge lookups, reference
                             queries, no time component).  Fed through
                             generate_scenarios.py these become STATIC-bucket
                             cache entries.

Both CSVs share the original schema (id, type, text, ...), so they can be
fed back to generate_scenarios.py via --input.

Usage:
    PYTHONPATH=. uv run python split_utterances.py
    PYTHONPATH=. uv run python split_utterances.py --input all_utterance.csv \\
        --temporal-out temporal_utterances.csv \\
        --static-out static_utterances.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reuse the eligibility regex already proven in generate_scenarios.py.
from generate_scenarios import _is_temporally_eligible


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split a utterance CSV into temporal vs static halves."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("all_utterance.csv"),
        help="Source CSV (default: all_utterance.csv).",
    )
    p.add_argument(
        "--temporal-out",
        type=Path,
        default=Path("temporal_utterances.csv"),
        help="Output path for temporally-eligible rows.",
    )
    p.add_argument(
        "--static-out",
        type=Path,
        default=Path("static_utterances.csv"),
        help="Output path for non-temporal rows.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.input.exists():
        sys.exit(f"error: input CSV not found — {args.input}")

    with args.input.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        sys.exit("error: input CSV is empty")

    temporal = [r for r in rows if _is_temporally_eligible(r["text"])]
    static = [r for r in rows if not _is_temporally_eligible(r["text"])]

    for path, subset in (
        (args.temporal_out, temporal),
        (args.static_out, static),
    ):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    print(f"Source       : {args.input}  ({len(rows)} rows)")
    print(f"Temporal out : {args.temporal_out}  ({len(temporal)} rows)")
    print(f"Static   out : {args.static_out}  ({len(static)} rows)")


if __name__ == "__main__":
    main()
