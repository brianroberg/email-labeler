"""Newsletter assessment review TUI.

Read-only Textual interface for browsing newsletter classification results,
including per-story quality scores, themes, and chain-of-thought reasoning.
"""

import json
import textwrap
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from tui_common import CANCEL, KeyMenuScreen, PageListView, PromptLineScreen

# ---------------------------------------------------------------------------
# Column widths for list view
# ---------------------------------------------------------------------------

_COL_TIER = 10
_COL_SENDER = 30
_COL_STORIES = 9  # "N stories"
_COL_GAP = 2


# ---------------------------------------------------------------------------
# Pure data functions (no UI, fully testable)
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


def format_filter_summary(
    *,
    tier: str | None = None,
    theme: str | None = None,
    sender: str | None = None,
) -> str:
    """Build a human-readable summary of active filters."""
    parts = []
    if tier is not None:
        parts.append(f"tier:{tier}")
    if theme is not None:
        parts.append(f"theme:{theme}")
    if sender is not None:
        parts.append(f"sender:{sender}")
    return "  ".join(parts)


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
    """Build content lines for the detail view. Pure function, no UI."""
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
# Filter key maps
# ---------------------------------------------------------------------------

_FILTER_TYPE_KEYS = {"t": "tier", "h": "theme", "s": "sender"}

_TIER_KEYS = {
    "e": "excellent",
    "g": "good",
    "f": "fair",
    "p": "poor",
    "c": None,  # clear
}

_THEME_KEYS = {
    "s": "scripture",
    "c": "christlikeness",
    "h": "church",
    "v": "vocation_family",
    "d": "disciple_making",
    "x": None,  # clear
}


# ---------------------------------------------------------------------------
# Filter prompts (match the curses prompt text)
# ---------------------------------------------------------------------------

_FILTER_TYPE_PROMPT = "Filter: [t]ier  [h]eme  [s]ender  (other key cancels)"
_TIER_PROMPT = "[e]xcellent  [g]ood  [f]air  [p]oor  [c]lear  (other cancels)"
_THEME_PROMPT = "[s]cripture [c]hristlikeness [h]urch [v]ocation_family [d]isciple_making [x]clear"


class DetailScreen(Screen):
    """Scrollable detail view for one assessment record."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.quit_app", "Quit"),
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("pageup,ctrl+b", "page_up", "Page up", show=False),
        Binding("pagedown,ctrl+f", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    DEFAULT_CSS = """
    DetailScreen > #detail-scroll {
        height: 1fr;
    }
    """

    def __init__(self, record: dict, position: int, total: int) -> None:
        super().__init__()
        self.record = record
        self.position = position  # 1-based index within the (filtered) set
        self.total = total

    def compose(self) -> ComposeResult:
        width = max(20, self.app.size.width)
        lines = build_detail_lines(self.record, width=width)
        yield VerticalScroll(
            *[Static(line or " ", markup=False) for line in lines],
            id="detail-scroll",
        )
        help_text = "↑/↓:Scroll  PgUp/PgDn  Home/End  Esc:Back  q:Quit"
        yield Static(help_text, id="detail-help", markup=False)
        subject = _truncate(self.record.get("subject", ""), width // 2)
        status = f"Newsletter {self.position}/{self.total}  |  {subject}"
        yield Static(status, id="detail-status", markup=False)

    def _scroll(self):
        return self.query_one("#detail-scroll", VerticalScroll)

    def action_scroll_up(self) -> None:
        self._scroll().scroll_relative(y=-1, animate=False)

    def action_scroll_down(self) -> None:
        self._scroll().scroll_relative(y=1, animate=False)

    def action_page_up(self) -> None:
        self._scroll().scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self._scroll().scroll_page_down(animate=False)

    def action_scroll_home(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_end(self) -> None:
        self._scroll().scroll_end(animate=False)


class ReviewApp(App):
    """Newsletter assessment browser: filterable list of records, drill into detail."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("f", "filter", "Filter"),
    ]

    DEFAULT_CSS = """
    ReviewApp #records {
        height: 1fr;
    }
    ReviewApp #header {
        text-style: underline;
    }
    """

    def __init__(
        self,
        records: list[dict],
        *,
        init_tier: str | None = None,
        init_theme: str | None = None,
        init_sender: str | None = None,
    ) -> None:
        super().__init__()
        self.all_records = records
        self.filtered = records
        self.f_tier = init_tier
        self.f_theme = init_theme
        self.f_sender = init_sender

    def compose(self) -> ComposeResult:
        yield Static(id="title", markup=False)
        hdr = (
            f"{'Tier':<{_COL_TIER}}"
            f"  {'Sender':<{_COL_SENDER}}"
            f"  {'Stories':<{_COL_STORIES}}"
            f"  Subject"
        )
        yield Static(hdr, id="header", markup=False)
        yield PageListView(id="records")
        help_text = "↑/↓:Nav  PgUp/PgDn  Enter:Detail  [f]ilter  q:Quit"
        yield Static(help_text, id="help", markup=False)

    def on_mount(self) -> None:
        self._refresh_list()

    def on_resize(self, event) -> None:
        # Re-render rows so column truncation tracks the new width. self.size
        # is still the OLD size while this handler runs — use the event's.
        self._refresh_list(width=event.size.width)

    def _refresh_list(self, *, reset_cursor: bool = False, width: int | None = None) -> None:
        self.filtered = apply_filters(
            self.all_records,
            tier=self.f_tier, theme=self.f_theme, sender=self.f_sender,
        )
        width = max(40, width if width is not None else self.size.width)
        listview = self.query_one(ListView)
        cursor = 0 if reset_cursor else (listview.index or 0)
        listview.clear()
        listview.extend(
            ListItem(Label(format_list_row(record, width), markup=False))
            for record in self.filtered
        )
        if self.filtered:
            listview.index = min(cursor, len(self.filtered) - 1)

        filter_str = format_filter_summary(
            tier=self.f_tier, theme=self.f_theme, sender=self.f_sender,
        )
        if filter_str:
            count = f"{len(self.filtered)}/{len(self.all_records)} records"
            title = f"Newsletter Assessments — {count}  [{filter_str}]"
        else:
            title = f"Newsletter Assessments — {len(self.filtered)} records"
        self.query_one("#title", Static).update(title)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if len(self.screen_stack) > 1:
            return  # a detail/modal is already up (Enter auto-repeat)
        index = self.query_one(ListView).index
        if index is not None and self.filtered:
            self.push_screen(
                DetailScreen(self.filtered[index], index + 1, len(self.filtered))
            )

    def action_filter(self) -> None:
        if len(self.screen_stack) > 1:
            return  # only from the list screen (parity with curses)

        def on_tier(result) -> None:
            if result != CANCEL:
                self.f_tier = result
                self._refresh_list(reset_cursor=True)

        def on_theme(result) -> None:
            if result != CANCEL:
                self.f_theme = result
                self._refresh_list(reset_cursor=True)

        def on_sender(result) -> None:
            if result is None:
                return  # Esc = cancel
            self.f_sender = result or None  # empty string clears the filter
            self._refresh_list(reset_cursor=True)

        def on_type(result) -> None:
            if result == "tier":
                self.push_screen(KeyMenuScreen(_TIER_PROMPT, _TIER_KEYS), on_tier)
            elif result == "theme":
                self.push_screen(KeyMenuScreen(_THEME_PROMPT, _THEME_KEYS), on_theme)
            elif result == "sender":
                self.push_screen(
                    PromptLineScreen("Sender filter  (Enter=confirm, empty=clear, Esc=cancel)"),
                    on_sender,
                )

        self.push_screen(KeyMenuScreen(_FILTER_TYPE_PROMPT, _FILTER_TYPE_KEYS), on_type)

    def action_quit_app(self) -> None:
        self.exit("quit")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_review_tui(
    records: list[dict],
    *,
    init_tier: str | None = None,
    init_theme: str | None = None,
    init_sender: str | None = None,
) -> None:
    """Launch the newsletter review TUI.

    *records* is the full (unfiltered) set. Initial filter values from CLI
    args are passed via init_tier/init_theme/init_sender.
    """
    if not records:
        print("No assessment records to display.")
        return
    ReviewApp(
        records,
        init_tier=init_tier, init_theme=init_theme, init_sender=init_sender,
    ).run()
