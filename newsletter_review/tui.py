"""Newsletter assessment review TUI.

Read-only curses interface for browsing newsletter classification results,
including per-story quality scores, themes, and chain-of-thought reasoning.
"""

import curses
import json
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Column widths for list view
# ---------------------------------------------------------------------------

_COL_TIER = 10
_COL_SENDER = 30
_COL_STORIES = 9  # "N stories"
_COL_GAP = 2


# ---------------------------------------------------------------------------
# Pure data functions (no curses, fully testable)
# ---------------------------------------------------------------------------

def load_assessments(path: Path) -> list[dict]:
    """Load newsletter assessment records from a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_filters(
    records: list[dict],
    *,
    tier: str | None = None,
    theme: str | None = None,
    sender: str | None = None,
) -> list[dict]:
    """Filter assessment records. All active filters are ANDed."""
    result = records
    if tier is not None:
        result = [r for r in result if r.get("overall_tier") == tier]
    if theme is not None:
        theme_lower = theme.lower()
        result = [
            r for r in result
            if any(
                theme_lower in [t.lower() for t in s.get("themes", [])]
                for s in r.get("stories", [])
            )
        ]
    if sender is not None:
        sender_lower = sender.lower()
        result = [r for r in result if sender_lower in r.get("from", "").lower()]
    return result


def _truncate(text: str, width: int) -> str:
    """Truncate text to width chars, adding ``...`` if needed."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to width, preserving existing newlines."""
    if not text:
        return [""]
    lines = []
    for paragraph in text.splitlines():
        if width <= 0:
            lines.append(paragraph)
        else:
            wrapped = textwrap.wrap(paragraph, width) or [""]
            lines.extend(wrapped)
    return lines


def format_list_row(record: dict, max_x: int) -> str:
    """Format one assessment record as a list-view line."""
    tier = record.get("overall_tier") or "—"
    sender = _truncate(record.get("from", "?"), _COL_SENDER)
    story_count = len(record.get("stories", []))
    stories_str = f"{story_count} stor{'y' if story_count == 1 else 'ies'}"

    fixed_width = _COL_TIER + _COL_SENDER + _COL_STORIES + _COL_GAP * 3
    subject_width = max(10, max_x - fixed_width)
    subject = _truncate(record.get("subject", ""), subject_width)

    return (
        f"{tier:<{_COL_TIER}}"
        f"  {sender:<{_COL_SENDER}}"
        f"  {stories_str:<{_COL_STORIES}}"
        f"  {subject}"
    )


def build_detail_lines(record: dict, width: int = 80) -> list[str]:
    """Build content lines for the detail view. Pure function, no curses."""
    subject = record.get("subject", "")
    sender = record.get("from", "")
    timestamp = record.get("timestamp", "")
    overall_tier = record.get("overall_tier") or "—"
    stories = record.get("stories", [])

    lines = [
        subject,
        "=" * min(60, width),
        f"From:    {sender}",
        f"Date:    {timestamp}",
        f"Overall: {overall_tier}",
        f"Stories: {len(stories)}",
        "",
    ]

    if not stories:
        lines.append("No stories extracted from this newsletter.")
        return lines

    for i, story in enumerate(stories):
        title = story.get("title", "Untitled")
        tier = story.get("tier") or "—"
        avg = story.get("average_score")
        avg_str = f"{avg:.1f}" if avg is not None else "—"
        themes = story.get("themes", [])

        lines.append(f"--- Story {i + 1}/{len(stories)}: {title} [{tier}, avg: {avg_str}] ---")

        if themes:
            lines.append(f"Themes: {', '.join(themes)}")

        lines.append("")

        # Story text
        text = story.get("text", "")
        if text:
            lines.append("Text:")
            lines.extend(wrap_text(text, width - 2))
            lines.append("")

        # Quality scores
        scores = story.get("scores")
        if scores:
            parts = [f"{k}={v}" for k, v in scores.items()]
            lines.append(f"Quality scores: {' '.join(parts)}")
        else:
            lines.append("Quality scores: —")
        lines.append("")

        # Quality CoT
        quality_cot = story.get("quality_cot", "")
        if quality_cot:
            lines.append("-- Quality CoT --")
            lines.extend(wrap_text(quality_cot, width - 2))
            lines.append("")

        # Theme CoT
        theme_cot = story.get("theme_cot", "")
        if theme_cot:
            lines.append("-- Theme CoT --")
            lines.extend(wrap_text(theme_cot, width - 2))
            lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Curses helpers
# ---------------------------------------------------------------------------

def _safe_addstr(win, y: int, x: int, text: str, attr: int = curses.A_NORMAL) -> None:
    """Write text to win at (y, x), clipping to avoid curses errors."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x >= max_x:
        return
    available = max_x - x - 1
    if available <= 0:
        return
    win.addnstr(y, x, text, available, attr)


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------

def _detail_view(stdscr, records: list[dict], index: int) -> str:
    """Render the detail screen for records[index]. Returns 'back' or 'quit'."""
    scroll_y = 0
    max_y, max_x = stdscr.getmaxyx()
    lines = build_detail_lines(records[index], width=max_x)

    while True:
        max_y, max_x = stdscr.getmaxyx()
        content_rows = max(1, max_y - 2)
        max_scroll = max(0, len(lines) - content_rows)

        stdscr.clear()
        for row_i in range(content_rows):
            line_i = scroll_y + row_i
            if line_i >= len(lines):
                break
            _safe_addstr(stdscr, row_i, 0, lines[line_i])

        help_text = "\u2191/\u2193:Scroll  PgUp/PgDn:Page  Home/End  Esc:Back  q:Quit"
        _safe_addstr(stdscr, max_y - 2, 0, help_text, curses.A_DIM)

        scroll_pct = ""
        if max_scroll > 0:
            pct = int(scroll_y / max_scroll * 100)
            scroll_pct = f"  ({pct}%)"
        subject = _truncate(records[index].get("subject", ""), max_x // 2)
        status = f"Newsletter {index + 1}/{len(records)}  |  {subject}{scroll_pct}"
        _safe_addstr(stdscr, max_y - 1, 0, status, curses.A_BOLD)

        stdscr.refresh()
        key = stdscr.getch()

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
        elif key == 27:
            return "back"
        elif key == ord("q"):
            return "quit"
        elif key == curses.KEY_RESIZE:
            max_y, max_x = stdscr.getmaxyx()
            lines = build_detail_lines(records[index], width=max_x)


# ---------------------------------------------------------------------------
# List view
# ---------------------------------------------------------------------------

def _list_view(stdscr, records: list[dict]) -> None:
    """Render the list screen and handle navigation + drill-down."""
    cursor = 0
    scroll_offset = 0

    while True:
        max_y, max_x = stdscr.getmaxyx()
        header_rows = 2
        footer_rows = 1
        page_size = max(1, max_y - header_rows - footer_rows)

        stdscr.clear()

        title = f"Newsletter Assessments \u2014 {len(records)} records"
        _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)

        hdr = (
            f"{'Tier':<{_COL_TIER}}"
            f"  {'Sender':<{_COL_SENDER}}"
            f"  {'Stories':<{_COL_STORIES}}"
            f"  Subject"
        )
        _safe_addstr(stdscr, 1, 0, hdr, curses.A_UNDERLINE)

        for vi in range(page_size):
            ti = scroll_offset + vi
            if ti >= len(records):
                break
            row_text = format_list_row(records[ti], max_x)
            attr = curses.A_REVERSE if ti == cursor else curses.A_NORMAL
            _safe_addstr(stdscr, header_rows + vi, 0, row_text, attr)

        help_text = "\u2191/\u2193:Nav  PgUp/PgDn:Page  Home/End  Enter:Detail  q:Quit"
        _safe_addstr(stdscr, max_y - 1, 0, help_text, curses.A_DIM)

        stdscr.refresh()
        key = stdscr.getch()

        if key == curses.KEY_UP and cursor > 0:
            cursor -= 1
            if cursor < scroll_offset:
                scroll_offset = cursor
        elif key == curses.KEY_DOWN and cursor < len(records) - 1:
            cursor += 1
            if cursor >= scroll_offset + page_size:
                scroll_offset = cursor - page_size + 1
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - page_size)
            scroll_offset = max(0, scroll_offset - page_size)
        elif key == curses.KEY_NPAGE:
            cursor = min(len(records) - 1, cursor + page_size)
            scroll_offset = min(max(0, len(records) - page_size), scroll_offset + page_size)
        elif key == curses.KEY_HOME:
            cursor = 0
            scroll_offset = 0
        elif key == curses.KEY_END:
            cursor = len(records) - 1
            scroll_offset = max(0, len(records) - page_size)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            result = _detail_view(stdscr, records, cursor)
            if result == "quit":
                return
        elif key == ord("q"):
            return
        elif key == curses.KEY_RESIZE:
            pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _curses_main(stdscr, records: list[dict]) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    _list_view(stdscr, records)


def run_review_tui(records: list[dict]) -> None:
    """Launch the newsletter review TUI."""
    if not records:
        print("No assessment records to display.")
        return
    curses.wrapper(_curses_main, records)
