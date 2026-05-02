"""Sample random queries from all_utterance.csv.

Examples:
    ./.venv/bin/python sample_queries.py
    ./.venv/bin/python sample_queries.py --count 5 --seed 42
    ./.venv/bin/python sample_queries.py --type IoT --count 10
    ./.venv/bin/python sample_queries.py --category "Data Query" --entity Chiller
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


DEFAULT_CSV = Path("all_utterance.csv")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample random queries from all_utterance.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to the utterance CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of random queries to return (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable sampling.",
    )
    parser.add_argument("--type", dest="type_filter", help="Filter by type, e.g. IoT.")
    parser.add_argument(
        "--category",
        dest="category_filter",
        help='Filter by category, e.g. "Data Query".',
    )
    parser.add_argument(
        "--entity",
        dest="entity_filter",
        help="Filter by entity, e.g. Chiller.",
    )
    parser.add_argument(
        "--group",
        dest="group_filter",
        help="Filter by group, e.g. retrospective.",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Print only matching row ids.",
    )
    return parser


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Public wrapper so other tools can reuse the CSV loader."""
    return _load_rows(csv_path)


def _matches(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.type_filter and row.get("type") != args.type_filter:
        return False
    if args.category_filter and row.get("category") != args.category_filter:
        return False
    if args.entity_filter and row.get("entity") != args.entity_filter:
        return False
    if args.group_filter and row.get("group") != args.group_filter:
        return False
    return True


def row_matches(
    row: dict[str, str],
    *,
    type_filter: str | None = None,
    category_filter: str | None = None,
    entity_filter: str | None = None,
    group_filter: str | None = None,
) -> bool:
    """Reusable row filter for callers outside the CLI."""
    if type_filter and row.get("type") != type_filter:
        return False
    if category_filter and row.get("category") != category_filter:
        return False
    if entity_filter and row.get("entity") != entity_filter:
        return False
    if group_filter and row.get("group") != group_filter:
        return False
    return True


def filter_rows(
    rows: list[dict[str, str]],
    *,
    type_filter: str | None = None,
    category_filter: str | None = None,
    entity_filter: str | None = None,
    group_filter: str | None = None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row_matches(
            row,
            type_filter=type_filter,
            category_filter=category_filter,
            entity_filter=entity_filter,
            group_filter=group_filter,
        )
    ]


def sample_rows(
    rows: list[dict[str, str]],
    *,
    count: int,
    seed: int | None = None,
) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    sample_size = min(count, len(rows))
    return rng.sample(rows, sample_size)


def main() -> None:
    args = _build_parser().parse_args()

    if not args.csv.exists():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        raise SystemExit(1)

    if args.count < 1:
        print("error: --count must be >= 1", file=sys.stderr)
        raise SystemExit(1)

    rows = _load_rows(args.csv)
    matches = [row for row in rows if _matches(row, args)]

    if not matches:
        print("No matching queries found.", file=sys.stderr)
        raise SystemExit(1)

    chosen = sample_rows(matches, count=args.count, seed=args.seed)

    for row in chosen:
        if args.ids_only:
            print(row["id"])
            continue
        print(f"[{row['id']}] {row['type']} | {row['category']} | {row['entity']}")
        print(row["text"])
        print()


if __name__ == "__main__":
    main()
