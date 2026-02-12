"""Curses-based TUI for editing already-reviewed golden set threads.

Launched via ``python -m evals.review --edit``.  Provides a paginated list
view of threads with keyboard navigation and a detail view for editing
stage 1 (sender type) and stage 2 (label) decisions.
"""

import curses
from pathlib import Path

from evals.schemas import GoldenThread
from gmail_utils import decode_body

# Abbreviation maps for compact list display
_SENDER_ABBREV = {"person": "PER", "service": "SVC"}
_LABEL_ABBREV = {"needs_response": "NR", "fyi": "FYI", "low_priority": "LP"}

# Hotkey maps (same keys as regular review)
_SENDER_KEY_MAP = {"p": "person", "s": "service"}
_LABEL_KEY_MAP = {"n": "needs_response", "f": "fyi", "l": "low_priority"}

# Column widths for list view
_COL_S1 = 5       # "PER" / "SVC"
_COL_S2 = 5       # "NR" / "FYI" / "LP"
_COL_SENDER = 32  # sender name + email
_COL_GAP = 2      # space between columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* chars, adding ``...`` if needed."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _safe_addstr(win, y: int, x: int, text: str, attr: int = curses.A_NORMAL) -> None:
    """Write *text* to *win* at (*y*, *x*), clipping to avoid curses errors."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x >= max_x:
        return
    available = max_x - x - 1  # -1 avoids writing to last cell of last row
    if available <= 0:
        return
    win.addnstr(y, x, text, available, attr)


def _format_list_row(thread: GoldenThread, max_x: int) -> str:
    """Format a single thread as one list-view line."""
    s1 = _SENDER_ABBREV.get(thread.expected_sender_type, "???")
    s2 = _LABEL_ABBREV.get(thread.expected_label, "???")

    sender = thread.senders[0] if thread.senders else "?"
    sender = _truncate(sender, _COL_SENDER)

    fixed_width = _COL_S1 + _COL_S2 + _COL_SENDER + _COL_GAP * 3
    subject_width = max(10, max_x - fixed_width)
    subject = _truncate(thread.subject, subject_width)

    return (
        f"{s1:<{_COL_S1}}"
        f"  {s2:<{_COL_S2}}"
        f"  {sender:<{_COL_SENDER}}"
        f"  {subject}"
    )


def _build_detail_lines(thread: GoldenThread, index: int, total: int) -> list[str]:
    """Build content lines for the detail view."""
    lines = [
        f"Thread {index + 1}/{total}  (id: {thread.thread_id})",
        "=" * 60,
        f"Subject:     {thread.subject}",
        f"Senders:     {', '.join(thread.senders)}",
        f"Snippet:     {thread.snippet}",
        f"Source:      {thread.source}",
        f"Reviewed:    {thread.reviewed}",
    ]
    if thread.notes:
        lines.append(f"Notes:       {thread.notes}")
    if thread.skipped:
        lines.append("Skipped:     True")

    lines.append("")
    lines.append(f"Sender type: {thread.expected_sender_type}")
    lines.append(f"Label:       {thread.expected_label}")
    lines.append("")
    lines.append("--- Body ---")

    for i, msg in enumerate(thread.messages):
        body = decode_body(msg.get("payload", {}))
        header = f"[Message {i + 1}]"
        lines.append(header)
        lines.extend(body.splitlines())
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Curses input prompts
# ---------------------------------------------------------------------------

def _prompt_sender_type_curses(stdscr) -> str | None:
    """Show sender-type menu on the bottom line; return value or ``None``."""
    max_y, max_x = stdscr.getmaxyx()
    prompt = "[p]erson  [s]ervice  (other key cancels)"
    _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
    _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
    stdscr.refresh()
    key = stdscr.getch()
    ch = chr(key) if 0 <= key < 256 else ""
    return _SENDER_KEY_MAP.get(ch.lower())


def _prompt_label_curses(stdscr) -> str | None:
    """Show label menu on the bottom line; return value or ``None``."""
    max_y, max_x = stdscr.getmaxyx()
    prompt = "[n]eeds_response  [f]yi  [l]ow_priority  (other key cancels)"
    _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
    _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
    stdscr.refresh()
    key = stdscr.getch()
    ch = chr(key) if 0 <= key < 256 else ""
    return _LABEL_KEY_MAP.get(ch.lower())


# ---------------------------------------------------------------------------
# Auto-save
# ---------------------------------------------------------------------------

def _auto_save(all_threads: list[GoldenThread], path: Path, stdscr) -> None:
    """Atomically save the full golden set, showing status on failure."""
    from evals.review import save_golden_set

    try:
        save_golden_set(all_threads, path)
    except Exception as exc:
        max_y, max_x = stdscr.getmaxyx()
        _safe_addstr(stdscr, max_y - 1, 0, f"Save failed: {exc}", curses.A_REVERSE)
        stdscr.refresh()
        stdscr.getch()


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------

def _detail_view(
    stdscr,
    threads: list[GoldenThread],
    index: int,
    all_threads: list[GoldenThread],
    path: Path,
) -> str:
    """Render the detail screen for ``threads[index]``.

    Returns ``"back"`` or ``"quit"``.
    """
    thread = threads[index]
    total = len(threads)
    scroll_y = 0

    while True:
        lines = _build_detail_lines(thread, index, total)
        max_y, max_x = stdscr.getmaxyx()
        content_rows = max(1, max_y - 2)  # reserve 2 lines for help + status

        stdscr.clear()
        for row_i in range(content_rows):
            line_i = scroll_y + row_i
            if line_i >= len(lines):
                break
            _safe_addstr(stdscr, row_i, 0, lines[line_i])

        # Help bar
        help_text = "\u2191/\u2193:Scroll  [s]ender  [l]abel  Esc:Back  q:Quit"
        _safe_addstr(stdscr, max_y - 2, 0, help_text, curses.A_DIM)

        # Status bar
        scroll_pct = ""
        max_scroll = max(0, len(lines) - content_rows)
        if max_scroll > 0:
            pct = int(scroll_y / max_scroll * 100)
            scroll_pct = f"  ({pct}%)"
        status = f"Sender: {thread.expected_sender_type}  Label: {thread.expected_label}{scroll_pct}"
        _safe_addstr(stdscr, max_y - 1, 0, status, curses.A_BOLD)

        stdscr.refresh()

        key = stdscr.getch()

        # Navigation
        if key == curses.KEY_UP and scroll_y > 0:
            scroll_y -= 1
        elif key == curses.KEY_DOWN and scroll_y < max_scroll:
            scroll_y += 1
        elif key == curses.KEY_PPAGE:
            scroll_y = max(0, scroll_y - content_rows)
        elif key == curses.KEY_NPAGE:
            scroll_y = min(max_scroll, scroll_y + content_rows)
        elif key == curses.KEY_HOME:
            scroll_y = 0
        elif key == curses.KEY_END:
            scroll_y = max_scroll

        # Edit actions
        elif key == ord("s"):
            new_val = _prompt_sender_type_curses(stdscr)
            if new_val:
                thread.expected_sender_type = new_val
                _auto_save(all_threads, path, stdscr)
        elif key == ord("l"):
            new_val = _prompt_label_curses(stdscr)
            if new_val:
                thread.expected_label = new_val
                _auto_save(all_threads, path, stdscr)

        # Exit
        elif key == 27:  # Esc
            return "back"
        elif key == ord("q"):
            return "quit"

        # Resize
        elif key == curses.KEY_RESIZE:
            pass  # max_y/max_x refreshed at top of loop


# ---------------------------------------------------------------------------
# List view
# ---------------------------------------------------------------------------

def _list_view(
    stdscr,
    threads: list[GoldenThread],
    all_threads: list[GoldenThread],
    path: Path,
) -> None:
    """Render the list screen and handle navigation + drill-down."""
    cursor = 0
    scroll_offset = 0

    while True:
        max_y, max_x = stdscr.getmaxyx()
        header_rows = 2  # title + column header
        footer_rows = 1  # help line
        page_size = max(1, max_y - header_rows - footer_rows)

        stdscr.clear()

        # Title
        title = f"Edit Mode \u2014 {len(threads)} threads"
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)

        # Column headers
        hdr = (
            f"{'S1':<{_COL_S1}}"
            f"  {'S2':<{_COL_S2}}"
            f"  {'Sender':<{_COL_SENDER}}"
            f"  Subject"
        )
        _safe_addstr(stdscr, 1, 0, hdr, curses.A_UNDERLINE)

        # Rows
        for vi in range(page_size):
            ti = scroll_offset + vi  # thread index
            if ti >= len(threads):
                break
            row_text = _format_list_row(threads[ti], max_x)
            attr = curses.A_REVERSE if ti == cursor else curses.A_NORMAL
            _safe_addstr(stdscr, header_rows + vi, 0, row_text, attr)

        # Help / footer
        help_text = "\u2191/\u2193:Nav  PgUp/PgDn:Page  Enter:Detail  q:Quit"
        _safe_addstr(stdscr, max_y - 1, 0, help_text, curses.A_DIM)

        stdscr.refresh()

        key = stdscr.getch()

        # Navigation
        if key == curses.KEY_UP and cursor > 0:
            cursor -= 1
            if cursor < scroll_offset:
                scroll_offset = cursor
        elif key == curses.KEY_DOWN and cursor < len(threads) - 1:
            cursor += 1
            if cursor >= scroll_offset + page_size:
                scroll_offset = cursor - page_size + 1
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - page_size)
            scroll_offset = max(0, scroll_offset - page_size)
        elif key == curses.KEY_NPAGE:
            cursor = min(len(threads) - 1, cursor + page_size)
            scroll_offset = min(max(0, len(threads) - page_size), scroll_offset + page_size)
        elif key == curses.KEY_HOME:
            cursor = 0
            scroll_offset = 0
        elif key == curses.KEY_END:
            cursor = len(threads) - 1
            scroll_offset = max(0, len(threads) - page_size)

        # Drill down
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            result = _detail_view(stdscr, threads, cursor, all_threads, path)
            if result == "quit":
                return

        # Quit
        elif key == ord("q"):
            return

        # Resize
        elif key == curses.KEY_RESIZE:
            pass  # recalculated at top of loop


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _curses_main(stdscr, threads: list[GoldenThread], all_threads: list[GoldenThread], path: Path) -> None:
    """Curses wrapper target."""
    curses.curs_set(0)
    stdscr.keypad(True)
    _list_view(stdscr, threads, all_threads, path)


def run_edit_tui(threads: list[GoldenThread], all_threads: list[GoldenThread], path: Path) -> None:
    """Launch the curses edit TUI.

    *threads* is the (possibly filtered) list displayed in the TUI.
    *all_threads* is the full golden set used for saving.
    *path* is the golden set file path for auto-save.
    """
    if not threads:
        print("No threads to edit.")
        return
    curses.wrapper(_curses_main, threads, all_threads, path)
