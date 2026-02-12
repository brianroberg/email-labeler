"""Interactive CLI to review and correct golden set labels.

Usage:
    python -m evals.review                    # blind mode (default)
    python -m evals.review --show-labels      # see existing labels
    python -m evals.review --stage 1          # review sender type only
    python -m evals.review --stage 2          # review label only
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

import readchar

from evals.schemas import GoldenThread
from gmail_utils import decode_body

SENDER_TYPES = ["person", "service"]
LABELS = ["needs_response", "fyi", "low_priority"]

_SENDER_KEY_MAP = {"p": "person", "s": "service"}
_LABEL_KEY_MAP = {"n": "needs_response", "f": "fyi", "l": "low_priority"}

# Mutable fields that get snapshotted for undo
_SNAPSHOT_FIELDS = ("expected_sender_type", "expected_label", "reviewed", "notes", "skipped")


def _capture_snapshot(thread: GoldenThread, index: int) -> dict:
    """Capture mutable fields before a mutation so they can be restored."""
    return {"index": index, **{f: getattr(thread, f) for f in _SNAPSHOT_FIELDS}}


def _restore_snapshot(threads: list[GoldenThread], snapshot: dict) -> int:
    """Restore a thread from *snapshot* and return the index to revisit."""
    thread = threads[snapshot["index"]]
    for f in _SNAPSHOT_FIELDS:
        setattr(thread, f, snapshot[f])
    return snapshot["index"]


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


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def get_hotkey() -> str:
    """Read a single keypress without requiring Enter.

    Returns a lowercase single character, or "" for Enter.
    Returns "q" on EOF / KeyboardInterrupt / unexpected errors.
    """
    try:
        key = readchar.readkey()
    except Exception:
        return "q"
    if key in ("\r", "\n"):
        return ""
    # Ignore multi-byte special keys (arrows, etc.)
    if len(key) > 1:
        return ""
    return key.lower()


def prompt_hotkey_menu(header: str, options: list[tuple[str, str]]) -> str:
    """Display a hotkey menu and return the pressed key.

    *options* is a list of ``(key, description)`` tuples, e.g.
    ``[("p", "person"), ("s", "service")]``.
    """
    menu = "  ".join(f"[{k}] {desc}" for k, desc in options)
    print(f"\n{header}")
    print(menu, end="", flush=True)
    key = get_hotkey()
    print()  # newline after keypress
    return key


def _prompt_sender_type() -> str | None:
    """Hotkey sub-menu for sender type. Returns value or None on cancel."""
    key = prompt_hotkey_menu("Select sender type:", [("p", "person"), ("s", "service")])
    return _SENDER_KEY_MAP.get(key)


def _prompt_label() -> str | None:
    """Hotkey sub-menu for label. Returns value or None on cancel."""
    key = prompt_hotkey_menu(
        "Select label:", [("n", "needs_response"), ("f", "fyi"), ("l", "low_priority")]
    )
    return _LABEL_KEY_MAP.get(key)


def _prompt_notes() -> str:
    """Prompt for free-text notes (uses regular input with Enter)."""
    try:
        return input("Notes: ").strip()
    except EOFError:
        return ""


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_thread(
    thread: GoldenThread, index: int, total: int, *, show_labels: bool = True, stage: int | None = None
) -> None:
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

    # Body
    print("\n--- Body ---")
    for i, msg in enumerate(thread.messages):
        body = decode_body(msg.get("payload", {}))
        print(f"[Message {i + 1}] {body}")

    if show_labels:
        print("\nCurrent labels:")
        if stage != 2:
            print(f"  Sender type: {thread.expected_sender_type}")
        if stage != 1:
            print(f"  Label:       {thread.expected_label}")


# ---------------------------------------------------------------------------
# Normal review mode (one thread at a time)
# ---------------------------------------------------------------------------

def review_thread_normal(
    thread: GoldenThread, index: int, total: int, *, stage: int | None = None, undo_stack: list | None = None,
) -> str:
    """Normal review: show labels, prompt for action.

    When *stage* is ``1``, only show/allow editing sender type.
    When *stage* is ``2``, only show/allow editing label.

    Returns ``"advance"``, ``"quit"``, ``"back"``, or ``"stay"``.
    """
    display_thread(thread, index, total, show_labels=True, stage=stage)

    # Build action menu based on which stages are being reviewed
    actions: list[tuple[str, str]] = [("", "confirm")]
    if stage != 2:
        actions.append(("s", "sender"))
    if stage != 1:
        actions.append(("l", "label"))
    actions.extend([("n", "notes"), ("z", "undo"), ("x", "skip"), ("q", "quit")])

    while True:
        key = prompt_hotkey_menu("Actions:", actions)

        if key == "" or key == "y":
            if undo_stack is not None:
                undo_stack.append(_capture_snapshot(thread, index))
            thread.reviewed = True
            return "advance"

        if key == "s" and stage != 2:
            new_type = _prompt_sender_type()
            if new_type:
                if undo_stack is not None:
                    undo_stack.append(_capture_snapshot(thread, index))
                thread.expected_sender_type = new_type
                print(f"  -> Sender type set to: {new_type}")
                thread.reviewed = True
                return "advance"
            continue  # cancelled — re-show menu

        if key == "l" and stage != 1:
            new_label = _prompt_label()
            if new_label:
                if undo_stack is not None:
                    undo_stack.append(_capture_snapshot(thread, index))
                thread.expected_label = new_label
                print(f"  -> Label set to: {new_label}")
                thread.reviewed = True
                return "advance"
            continue  # cancelled — re-show menu

        if key == "n":
            thread.notes = _prompt_notes()
            print("  -> Notes saved. (Thread not yet confirmed — press Enter to confirm)")
            continue  # re-show action menu (thread already displayed)

        if key == "z":
            return "back"

        if key == "x":
            if undo_stack is not None:
                undo_stack.append(_capture_snapshot(thread, index))
            thread.skipped = True
            thread.reviewed = True
            print("  -> Thread marked as skipped.")
            return "advance"

        if key == "q":
            return "quit"

        print(f"  Unknown action: {key!r}")


# ---------------------------------------------------------------------------
# Blind review mode
# ---------------------------------------------------------------------------

def review_thread_blind(
    thread: GoldenThread, index: int, total: int, *, stage: int | None = None, undo_stack: list | None = None,
) -> str:
    """Blind review: hide labels, prompt sender type then label.

    When *stage* is ``1``, only prompt for sender type.
    When *stage* is ``2``, only prompt for label.

    Returns ``"advance"``, ``"quit"``, ``"back"``, or ``"stay"``.
    """
    display_thread(thread, index, total, show_labels=False)

    # Step 1: sender type
    if stage != 2:
        while True:
            key = prompt_hotkey_menu(
                "Sender type:",
                [
                    ("p", "person"), ("s", "service"), ("z", "undo"),
                    ("n", "notes"), ("x", "skip"), ("q", "quit"),
                ],
            )
            if key in _SENDER_KEY_MAP:
                if undo_stack is not None:
                    undo_stack.append(_capture_snapshot(thread, index))
                thread.expected_sender_type = _SENDER_KEY_MAP[key]
                print(f"  -> {thread.expected_sender_type}")
                break
            if key == "z":
                return "back"
            if key == "n":
                thread.notes = _prompt_notes()
                print("  -> Notes saved.")
                continue
            if key == "x":
                if undo_stack is not None:
                    undo_stack.append(_capture_snapshot(thread, index))
                thread.skipped = True
                thread.reviewed = True
                print("  -> Thread marked as skipped.")
                return "advance"
            if key == "q":
                return "quit"
            print(f"  Invalid key: {key!r}")

    # Step 2: label
    if stage != 1:
        while True:
            key = prompt_hotkey_menu(
                "Label:",
                [
                    ("n", "needs_response"), ("f", "fyi"),
                    ("l", "low_priority"), ("z", "undo"), ("q", "quit"),
                ],
            )
            if key in _LABEL_KEY_MAP:
                if undo_stack is not None:
                    undo_stack.append(_capture_snapshot(thread, index))
                thread.expected_label = _LABEL_KEY_MAP[key]
                thread.reviewed = True
                print(f"  -> Classified as {thread.expected_sender_type} / {thread.expected_label}")
                return "advance"
            if key == "z":
                return "back"
            if key == "q":
                return "quit"
            print(f"  Invalid key: {key!r}")

    # stage==1 only: mark reviewed after sender type
    # (snapshot already captured before the sender type mutation above)
    thread.reviewed = True
    return "advance"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def review_loop(
    threads: list[GoldenThread], *, start_at: int = 0, blind: bool = False, stage: int | None = None
) -> None:
    """Main interactive review loop.  Does NOT save — caller is responsible."""
    review_fn = review_thread_blind if blind else review_thread_normal
    undo_stack: list[dict] = []
    total = len(threads)
    i = start_at

    while i < total:
        result = review_fn(threads[i], i, total, stage=stage, undo_stack=undo_stack)
        if result == "quit":
            break
        if result == "advance":
            i += 1
        elif result == "back":
            if not undo_stack:
                print("  Nothing to undo.")
            else:
                i = _restore_snapshot(threads, undo_stack.pop())
                print(f"  <- Undone. Back to thread {i + 1}/{total}.")
        # "stay" → loop again on the same thread


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Review and correct golden set labels")
    parser.add_argument("--golden-set", default="evals/golden_set.jsonl", help="Path to golden set JSONL")
    parser.add_argument(
        "--show-labels", action="store_true", help="Show existing labels (default is blind mode)",
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], default=None,
        help="Review only stage 1 (sender) or stage 2 (label)",
    )
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

    # Apply filters
    filtered = False
    if args.unreviewed_only:
        threads = [t for t in threads if not t.reviewed]
        filtered = True
    if args.filter_label:
        threads = [t for t in threads if t.expected_label == args.filter_label]
        filtered = True

    if not threads:
        print("No threads match the filters.", file=sys.stderr)
        sys.exit(0)

    # Review, then save
    review_loop(threads, start_at=args.start_at, blind=not args.show_labels, stage=args.stage)

    if filtered:
        # Merge changes from filtered subset back into the full set
        all_threads = load_golden_set(path)
        filtered_map = {t.thread_id: t for t in threads}
        for i, t in enumerate(all_threads):
            if t.thread_id in filtered_map:
                all_threads[i] = filtered_map[t.thread_id]
        save_golden_set(all_threads, path)
    else:
        save_golden_set(threads, path)

    reviewed_count = sum(1 for t in threads if t.reviewed)
    print(f"\nSaved. {reviewed_count}/{len(threads)} threads reviewed.")


if __name__ == "__main__":
    cli()
