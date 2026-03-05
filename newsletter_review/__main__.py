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

from newsletter_review.tui import load_assessments, run_review_tui


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

    run_review_tui(records, init_tier=args.tier, init_theme=args.theme, init_sender=args.sender)


if __name__ == "__main__":
    cli()
