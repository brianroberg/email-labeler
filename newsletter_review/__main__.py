"""CLI entry point for the newsletter review TUI.

Usage::

    python -m newsletter_review
    python -m newsletter_review --tier poor
    python -m newsletter_review --theme scripture --sender dm.org
    python -m newsletter_review --file path/to/assessments.jsonl
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from newsletter_review.tui import load_assessments, run_review_tui


def _since_date(value: str) -> str:
    """argparse ``type`` for ``--since``: validate ``YYYY-MM-DD`` and normalize to
    zero-padded ISO, matching the in-TUI since input."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date '{value}' — use YYYY-MM-DD") from None


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
    parser.add_argument(
        "--since",
        type=_since_date,
        help="Filter to records sent on or after this local date (YYYY-MM-DD)",
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

    run_review_tui(
        records,
        init_tier=args.tier,
        init_theme=args.theme,
        init_sender=args.sender,
        init_since=args.since,
    )


if __name__ == "__main__":
    cli()
