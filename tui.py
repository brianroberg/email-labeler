"""Newsletter assessment TUI -- browse and filter classified newsletter stories."""

from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static
from textual import on

from tui_data import (
    Assessment,
    Story,
    filter_by_tier,
    filter_by_theme,
    format_detail,
    load_assessments,
)

TIER_CYCLE = [None, "excellent", "good", "fair", "poor"]
THEME_CYCLE = [
    None, "scripture", "christlikeness", "church",
    "vocation_family", "disciple_making",
]


class AssessmentApp(App):
    """TUI for browsing newsletter assessment data."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #filter-bar {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
    }
    #newsletters {
        height: 1fr;
        min-height: 5;
    }
    #stories {
        height: 1fr;
        min-height: 5;
    }
    #detail-scroll {
        height: 2fr;
        min-height: 8;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "cycle_tier", "Cycle Tier"),
        Binding("h", "cycle_theme", "Cycle Theme"),
    ]

    def __init__(self, assessments: list[Assessment]):
        super().__init__()
        self.all_assessments = assessments
        self.filtered_assessments = list(assessments)
        self._tier_filter: str | None = None
        self._theme_filter: str | None = None
        self._row_to_assessment: dict = {}
        self._row_to_story: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="filter-bar")
        yield DataTable(id="newsletters")
        yield DataTable(id="stories")
        with VerticalScroll(id="detail-scroll"):
            yield Static("Select a story to view details", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        nl_table = self.query_one("#newsletters", DataTable)
        nl_table.cursor_type = "row"
        nl_table.add_columns("Subject", "From", "Tier", "Date")

        story_table = self.query_one("#stories", DataTable)
        story_table.cursor_type = "row"
        story_table.add_columns("Title", "Tier", "Avg", "Themes")

        self._populate_newsletters()
        self._update_filter_bar()

    def _populate_newsletters(self) -> None:
        table = self.query_one("#newsletters", DataTable)
        table.clear()
        self._row_to_assessment.clear()

        for assessment in self.filtered_assessments:
            try:
                dt = datetime.fromisoformat(assessment.timestamp)
                date_str = dt.strftime("%b %d")
            except ValueError:
                date_str = "\u2014"
            row_key = table.add_row(
                assessment.subject,
                assessment.sender,
                assessment.overall_tier or "\u2014",
                date_str,
            )
            self._row_to_assessment[row_key] = assessment

    def _update_filter_bar(self) -> None:
        tier_text = self._tier_filter or "All"
        theme_text = self._theme_filter or "All"
        count = len(self.filtered_assessments)
        total = len(self.all_assessments)
        self.query_one("#filter-bar", Static).update(
            f"Tier: {tier_text}  |  Theme: {theme_text}"
            f"  |  {count} of {total} newsletters"
        )
