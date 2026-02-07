"""Interactive CLI to review and correct golden set labels.

Usage:
    python -m evals.review
    python -m evals.review --unreviewed-only
    python -m evals.review --filter-label needs_response
    python -m evals.review --start-at 5
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from evals.schemas import GoldenThread
from gmail_utils import decode_body

SENDER_TYPES = ["person", "service"]
LABELS = ["needs_response", "fyi", "low_priority", "unwanted"]
BODY_PREVIEW_CHARS = 500


def load_golden_set(path: Path) -> list[GoldenThread]:
    """Load golden set from JSONL file."""
    threads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                threads.append(GoldenThread.from_dict(json.loads(line)))
    return threads


def save_golden_set(threads: list[GoldenThread], path: Path) -> None:
    """Save golden set atomically (temp file + rename)."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for thread in threads:
                f.write(json.dumps(thread.to_dict()) + "\n")
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def display_thread(thread: GoldenThread, index: int, total: int) -> None:
    """Display a thread for review."""
    print(f"\n{'=' * 60}")
    print(f"Thread {index + 1}/{total}  (id: {thread.thread_id})")
    print(f"{'=' * 60}")
    print(f"Subject:  {thread.subject}")
    print(f"Senders:  {', '.join(thread.senders)}")
    print(f"Snippet:  {thread.snippet}")
    print(f"Source:   {thread.source}")
    print(f"Reviewed: {thread.reviewed}")
    if thread.notes:
        print(f"Notes:    {thread.notes}")

    # Body preview
    print("\n--- Body preview ---")
    for i, msg in enumerate(thread.messages):
        body = decode_body(msg.get("payload", {}))
        preview = body[:BODY_PREVIEW_CHARS]
        if len(body) > BODY_PREVIEW_CHARS:
            preview += "..."
        print(f"[Message {i + 1}] {preview}")

    print("\nCurrent labels:")
    print(f"  Sender type: {thread.expected_sender_type}")
    print(f"  Label:       {thread.expected_label}")


def prompt_action() -> str:
    """Prompt user for review action."""
    print("\nActions: [Enter] confirm  [s] change sender type  [l] change label")
    print("         [n] add notes    [q] quit and save")
    try:
        return input("> ").strip().lower()
    except EOFError:
        return "q"


def prompt_choice(prompt: str, options: list[str]) -> str | None:
    """Prompt user to pick from a list."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    try:
        choice = input("> ").strip()
    except EOFError:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]
    return None


def review_loop(threads: list[GoldenThread], path: Path, start_at: int = 0) -> None:
    """Main interactive review loop."""
    total = len(threads)
    i = start_at

    while i < total:
        thread = threads[i]
        display_thread(thread, i, total)

        action = prompt_action()

        if action == "" or action == "y":
            # Confirm current labels
            thread.reviewed = True
            i += 1

        elif action == "s":
            new_type = prompt_choice("Select sender type:", SENDER_TYPES)
            if new_type:
                thread.expected_sender_type = new_type
                print(f"  -> Sender type set to: {new_type}")
            thread.reviewed = True
            i += 1

        elif action == "l":
            new_label = prompt_choice("Select label:", LABELS)
            if new_label:
                thread.expected_label = new_label
                print(f"  -> Label set to: {new_label}")
            thread.reviewed = True
            i += 1

        elif action == "n":
            try:
                notes = input("Notes: ").strip()
            except EOFError:
                notes = ""
            thread.notes = notes
            print("  -> Notes saved. (Thread not yet confirmed — press Enter to confirm)")
            continue  # Re-show same thread

        elif action == "q":
            break

        else:
            print(f"  Unknown action: {action!r}")
            continue

    save_golden_set(threads, path)
    reviewed_count = sum(1 for t in threads if t.reviewed)
    print(f"\nSaved. {reviewed_count}/{total} threads reviewed.")


def cli():
    parser = argparse.ArgumentParser(description="Review and correct golden set labels")
    parser.add_argument("--golden-set", default="evals/golden_set.jsonl", help="Path to golden set JSONL")
    parser.add_argument("--unreviewed-only", action="store_true", help="Show only unreviewed threads")
    parser.add_argument("--filter-label", choices=LABELS, help="Show only threads with this label")
    parser.add_argument("--start-at", type=int, default=0, help="Start at thread index (0-based)")
    args = parser.parse_args()

    path = Path(args.golden_set)
    if not path.exists():
        print(f"Golden set not found: {path}", file=sys.stderr)
        print("Run 'python -m evals.harvest' first to create it.", file=sys.stderr)
        sys.exit(1)

    threads = load_golden_set(path)
    if not threads:
        print("Golden set is empty.", file=sys.stderr)
        sys.exit(1)

    # Apply filters (preserve original indices for save)
    if args.unreviewed_only:
        threads = [t for t in threads if not t.reviewed]
    if args.filter_label:
        threads = [t for t in threads if t.expected_label == args.filter_label]

    if not threads:
        print("No threads match the filters.", file=sys.stderr)
        sys.exit(0)

    # When filtering, we need to save the full set back
    # So reload the full set, apply edits to matching threads, then save
    if args.unreviewed_only or args.filter_label:
        all_threads = load_golden_set(path)
        # Build lookup by thread_id
        filtered_map = {t.thread_id: t for t in threads}
        review_loop(threads, path=Path("/dev/null"), start_at=args.start_at)  # Don't save yet
        # Apply changes back to full set
        for i, t in enumerate(all_threads):
            if t.thread_id in filtered_map:
                all_threads[i] = filtered_map[t.thread_id]
        save_golden_set(all_threads, path)
    else:
        review_loop(threads, path, start_at=args.start_at)


if __name__ == "__main__":
    cli()
