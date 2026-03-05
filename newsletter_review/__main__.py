"""CLI entry point for the newsletter review TUI.

Usage::

    python -m newsletter_review
    python -m newsletter_review --tier poor
    python -m newsletter_review --theme scripture --sender dm.org
    python -m newsletter_review --file path/to/assessments.jsonl
"""

import argparse
import sys
from pathlib import Path

from newsletter_review.tui import apply_filters, load_assessments, run_review_tui


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Browse newsletter classification assessments in a TUI.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/newsletter_assessments.jsonl"),
        help="Path to the assessments JSONL file (default: data/newsletter_assessments.jsonl)",
    )
    parser.add_argument(
        "--tier",
        choices=["excellent", "good", "fair", "poor"],
        help="Filter by overall tier",
    )
    parser.add_argument(
        "--theme",
        help="Filter by story theme (case-insensitive)",
    )
    parser.add_argument(
        "--sender",
        help="Filter by sender (substring match, case-insensitive)",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        records = load_assessments(args.file)
    except Exception as exc:
        print(f"Error loading {args.file}: {exc}", file=sys.stderr)
        sys.exit(1)

    records = apply_filters(records, tier=args.tier, theme=args.theme, sender=args.sender)

    if not records:
        print("No records match the given filters.")
        sys.exit(0)

    run_review_tui(records)


if __name__ == "__main__":
    cli()
