"""Interactive CLI to review and correct golden set labels.

Usage:
    python -m evals.review                    # blind mode (default)
    python -m evals.review --show-labels      # see existing labels
    python -m evals.review --edit             # TUI for editing reviewed threads
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

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from evals.schemas import GoldenThread
from gmail_utils import decode_body
from tui_common import CANCEL, KeyMenuScreen, PromptLineScreen

SENDER_TYPES = ["person", "service"]
LABELS = ["needs_response", "fyi", "low_priority"]

_SENDER_KEY_MAP = {"p": "person", "s": "service"}
_LABEL_KEY_MAP = {"r": "needs_response", "f": "fyi", "l": "low_priority"}

# Mutable fields that get snapshotted for undo
_SNAPSHOT_FIELDS = ("expected_sender_type", "expected_label", "reviewed", "notes", "excluded")


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
# Review session: cursor + one-snapshot-per-thread undo (no UI)
# ---------------------------------------------------------------------------

class ReviewSession:
    """Cursor + undo stack for the review queue, decoupled from the UI.

    Undo is owned here, not in the key handlers: each thread gets exactly
    one snapshot captured on entry, regardless of how the review mutates it
    (confirm, classify, skip, exclude, notes).  This keeps the undo stack in
    lock-step with the cursor — one advance pushes one snapshot — so a later
    undo walks back one classification at a time without over-rewinding past
    skips or leaving a half-applied blind-mode sender behind.
    """

    def __init__(self, threads: list[GoldenThread], *, start_at: int = 0):
        self.threads = threads
        self.index = start_at
        self.undo_stack: list[dict] = []
        self._entry = (
            _capture_snapshot(threads[start_at], start_at) if start_at < len(threads) else None
        )

    @property
    def done(self) -> bool:
        return self.index >= len(self.threads)

    @property
    def thread(self) -> GoldenThread:
        return self.threads[self.index]

    def advance(self) -> None:
        """Record the pre-review snapshot of the current thread and move on."""
        self.undo_stack.append(self._entry)
        self.index += 1
        if not self.done:
            self._entry = _capture_snapshot(self.threads[self.index], self.index)

    def undo(self) -> str:
        """Discard in-progress edits, then step back to the previous decision."""
        _restore_snapshot(self.threads, self._entry)
        if not self.undo_stack:
            return "  Nothing to undo."
        self.index = _restore_snapshot(self.threads, self.undo_stack.pop())
        self._entry = _capture_snapshot(self.threads[self.index], self.index)
        return f"  <- Undone. Back to thread {self.index + 1}/{len(self.threads)}."


# ---------------------------------------------------------------------------
# Display / menu builders (pure)
# ---------------------------------------------------------------------------

def build_review_lines(
    thread: GoldenThread, index: int, total: int, *, show_labels: bool = True, stage: int | None = None
) -> list[str]:
    """Content lines for one thread under review."""
    lines = [
        "=" * 60,
        f"Thread {index + 1}/{total}  (id: {thread.thread_id})",
        "=" * 60,
        f"Subject:  {thread.subject}",
        f"Senders:  {', '.join(thread.senders)}",
        f"Snippet:  {thread.snippet}",
        f"Source:   {thread.source}",
        f"Reviewed: {thread.reviewed}",
    ]
    if thread.notes:
        lines.append(f"Notes:    {thread.notes}")
    lines.append("")
    lines.append("--- Body ---")
    for i, msg in enumerate(thread.messages):
        body = decode_body(msg.get("payload", {}))
        lines.append(f"[Message {i + 1}] {body}")
    if show_labels:
        lines.append("")
        lines.append("Current labels:")
        if stage != 2:
            lines.append(f"  Sender type: {thread.expected_sender_type}")
        if stage != 1:
            lines.append(f"  Label:       {thread.expected_label}")
    return lines


def build_menu(header: str, options: list[tuple[str, str]]) -> str:
    """Menu legend, e.g. ``Actions:`` then ``[s] sender  [l] label ...``."""
    return header + "\n" + "  ".join(f"[{k}] {desc}" for k, desc in options)


_QUEUE_OPTIONS = [("n", "notes"), ("z", "undo"), ("k", "skip"), ("e", "exclude"), ("q", "quit")]


# ---------------------------------------------------------------------------
# Textual UI layer
# ---------------------------------------------------------------------------

class ReviewApp(App):
    """Sequential golden-set review: one thread at a time, blind or normal.

    Key handling mirrors the old readchar loop: single lowercase hotkeys,
    Enter confirms in normal mode.  Deliberate change from readchar: arrow /
    page keys scroll the thread body instead of acting like Enter.
    """

    DEFAULT_CSS = """
    ReviewApp #review-scroll {
        height: 1fr;
    }
    ReviewApp #status {
        color: $text-muted;
    }
    """

    def __init__(self, threads, *, start_at: int = 0, blind: bool = False, stage: int | None = None):
        super().__init__()
        self.session = ReviewSession(threads, start_at=start_at)
        self.blind = blind
        self.stage = stage
        self._step = "actions"  # "actions" (normal) | "sender" | "label" (blind)

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static(id="thread-content", markup=False), id="review-scroll")
        yield Static(id="menu", markup=False)
        yield Static(id="status", markup=False)

    def on_mount(self) -> None:
        self._show_thread()

    # -- rendering -----------------------------------------------------------

    def _show_thread(self) -> None:
        self._refresh_content()
        self.query_one("#review-scroll", VerticalScroll).scroll_home(animate=False)
        if self.blind:
            self._step = "label" if self.stage == 2 else "sender"
        else:
            self._step = "actions"
        self._refresh_menu()

    def _refresh_content(self) -> None:
        s = self.session
        lines = build_review_lines(
            s.thread, s.index, len(s.threads), show_labels=not self.blind, stage=self.stage
        )
        self.query_one("#thread-content", Static).update("\n".join(lines))

    def _refresh_menu(self) -> None:
        if self._step == "actions":
            actions: list[tuple[str, str]] = [("", "confirm")]
            if self.stage != 2:
                actions.append(("s", "sender"))
            if self.stage != 1:
                actions.append(("l", "label"))
            actions.extend(_QUEUE_OPTIONS)
            menu = build_menu("Actions:", actions)
        elif self._step == "sender":
            menu = build_menu("Sender type:", [("p", "person"), ("s", "service")] + _QUEUE_OPTIONS)
        else:
            menu = build_menu(
                "Label:",
                [("r", "needs_response"), ("f", "fyi"), ("l", "low_priority")] + _QUEUE_OPTIONS,
            )
        self.query_one("#menu", Static).update(menu)

    def _status(self, msg: str) -> None:
        self.query_one("#status", Static).update(msg)

    # -- queue movement --------------------------------------------------------

    def _advance(self) -> None:
        self.session.advance()
        if self.session.done:
            self.exit("done")
        else:
            self._show_thread()

    def _undo(self) -> None:
        msg = self.session.undo()
        self._show_thread()
        self._status(msg)

    # -- key dispatch ------------------------------------------------------------

    def on_key(self, event) -> None:
        if self.screen is not self.screen_stack[0]:
            return  # a modal (submenu / notes) owns the keys
        key = event.key
        if key == "enter":
            hot = ""
        elif len(key) == 1:
            hot = key.lower()
        else:
            scroll = self.query_one("#review-scroll", VerticalScroll)
            if key == "up":
                scroll.scroll_relative(y=-1, animate=False)
            elif key == "down":
                scroll.scroll_relative(y=1, animate=False)
            elif key == "pageup":
                scroll.scroll_page_up(animate=False)
            elif key == "pagedown":
                scroll.scroll_page_down(animate=False)
            elif key == "home":
                scroll.scroll_home(animate=False)
            elif key == "end":
                scroll.scroll_end(animate=False)
            return
        self._status("")
        if self._step == "actions":
            self._handle_actions(hot)
        elif self._step == "sender":
            self._handle_sender(hot)
        else:
            self._handle_label(hot)

    def _handle_queue_keys(self, hot: str, invalid_prefix: str) -> None:
        """The n/z/k/e/q keys shared by every prompt."""
        thread = self.session.thread
        if hot == "n":
            self._edit_notes()
        elif hot == "z":
            self._undo()
        elif hot == "k":
            self._status("  -> Skipped (no judgment).")
            self._advance()
        elif hot == "e":
            thread.excluded = True
            thread.reviewed = True
            self._status("  -> Thread excluded (permanently set aside).")
            self._advance()
        elif hot == "q":
            self.exit("quit")
        else:
            self._status(f"  {invalid_prefix}: {hot!r}")

    def _handle_actions(self, hot: str) -> None:
        thread = self.session.thread
        if hot in ("", "y"):
            thread.reviewed = True
            self._advance()
        elif hot == "s" and self.stage != 2:
            def apply(result) -> None:
                if result != CANCEL:
                    thread.expected_sender_type = result
                    thread.reviewed = True
                    self._status(f"  -> Sender type set to: {result}")
                    self._advance()

            self.push_screen(
                KeyMenuScreen(
                    "Select sender type:  [p] person  [s] service  (other key cancels)",
                    _SENDER_KEY_MAP,
                ),
                apply,
            )
        elif hot == "l" and self.stage != 1:
            def apply(result) -> None:
                if result != CANCEL:
                    thread.expected_label = result
                    thread.reviewed = True
                    self._status(f"  -> Label set to: {result}")
                    self._advance()

            self.push_screen(
                KeyMenuScreen(
                    "Select label:  [r] needs_response  [f] fyi  [l] low_priority"
                    "  (other key cancels)",
                    _LABEL_KEY_MAP,
                ),
                apply,
            )
        else:
            self._handle_queue_keys(hot, "Unknown action")

    def _handle_sender(self, hot: str) -> None:
        thread = self.session.thread
        if hot in _SENDER_KEY_MAP:
            thread.expected_sender_type = _SENDER_KEY_MAP[hot]
            self._status(f"  -> {thread.expected_sender_type}")
            if self.stage == 1:
                thread.reviewed = True
                self._advance()
            else:
                self._step = "label"
                self._refresh_menu()
        else:
            self._handle_queue_keys(hot, "Invalid key")

    def _handle_label(self, hot: str) -> None:
        thread = self.session.thread
        if hot in _LABEL_KEY_MAP:
            thread.expected_label = _LABEL_KEY_MAP[hot]
            thread.reviewed = True
            self._status(
                f"  -> Classified as {thread.expected_sender_type} / {thread.expected_label}"
            )
            self._advance()
        else:
            self._handle_queue_keys(hot, "Invalid key")

    def _edit_notes(self) -> None:
        thread = self.session.thread
        in_blind = self.blind

        def apply(result) -> None:
            if result is None:
                self._status("  Notes unchanged.")
                return
            thread.notes = result
            self._refresh_content()  # keep the Notes: line current (step unchanged)
            if in_blind:
                self._status("  -> Notes saved.")
            else:
                self._status(
                    "  -> Notes saved. (Thread not yet confirmed — press Enter to confirm)"
                )

        self.push_screen(PromptLineScreen("Notes:", initial=thread.notes), apply)


def select_review_threads(
    threads: list[GoldenThread], *, unreviewed_only: bool = False, filter_label: str | None = None
) -> list[GoldenThread]:
    """Threads to queue for review.

    Excluded threads are permanently set aside and are never queued, regardless
    of the other filters.
    """
    result = [t for t in threads if not t.excluded]
    if unreviewed_only:
        result = [t for t in result if not t.reviewed]
    if filter_label:
        result = [t for t in result if t.expected_label == filter_label]
    return result


def review_loop(
    threads: list[GoldenThread], *, start_at: int = 0, blind: bool = False, stage: int | None = None
) -> None:
    """Launch the review TUI.  Does NOT save — caller is responsible."""
    if not threads or start_at >= len(threads):
        return
    ReviewApp(threads, start_at=start_at, blind=blind, stage=stage).run()


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
    parser.add_argument(
        "--start-at", type=int, default=0,
        help="Start at this index into the review queue (0-based, after excluded "
             "threads and any --filter-label/--unreviewed-only filters are applied)",
    )
    parser.add_argument("--edit", action="store_true", help="TUI for editing reviewed threads")
    args = parser.parse_args()

    path = Path(args.golden_set)
    if not path.exists():
        print(f"Golden set not found: {path}", file=sys.stderr)
        print("Run 'python -m evals.harvest' first to create it.", file=sys.stderr)
        sys.exit(1)

    all_threads = load_golden_set(path)
    if not all_threads:
        print("Golden set is empty.", file=sys.stderr)
        sys.exit(1)

    # Edit mode: default to reviewed-only, but respect explicit filters
    if args.edit:
        from evals.edit_tui import run_edit_tui

        threads = all_threads
        if args.unreviewed_only:
            threads = [t for t in threads if not t.reviewed]
        elif not args.filter_label:
            # Default: show only reviewed threads
            threads = [t for t in threads if t.reviewed]
        if args.filter_label:
            threads = [t for t in threads if t.expected_label == args.filter_label]

        if not threads:
            print("No threads match the filters.", file=sys.stderr)
            sys.exit(0)

        run_edit_tui(threads, all_threads, path)
        return

    # Regular review mode. Excluded threads are never queued; un-exclude via --edit.
    threads = select_review_threads(
        all_threads, unreviewed_only=args.unreviewed_only, filter_label=args.filter_label
    )

    if not threads:
        print("No threads match the filters.", file=sys.stderr)
        sys.exit(0)

    # Review, then save. select_review_threads returns the SAME thread objects
    # held in all_threads, so review_loop's in-place edits are already reflected
    # there — saving all_threads directly preserves excluded/filtered-out
    # threads, original order, and any duplicate thread_ids untouched.
    review_loop(threads, start_at=args.start_at, blind=not args.show_labels, stage=args.stage)
    save_golden_set(all_threads, path)

    reviewed_count = sum(1 for t in threads if t.reviewed)
    print(f"\nSaved. {reviewed_count}/{len(threads)} threads reviewed.")


if __name__ == "__main__":
    cli()
