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

    @on(DataTable.RowHighlighted, "#newsletters")
    def on_newsletter_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key in self._row_to_assessment:
            self._populate_stories(self._row_to_assessment[event.row_key])

    @on(DataTable.RowHighlighted, "#stories")
    def on_story_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key in self._row_to_story:
            self._show_detail(self._row_to_story[event.row_key])

    def _populate_stories(self, assessment: Assessment) -> None:
        table = self.query_one("#stories", DataTable)
        table.clear()
        self._row_to_story.clear()

        for story in assessment.stories:
            avg = f"{story.average_score:.2f}" if story.average_score is not None else "\u2014"
            themes = ", ".join(story.themes) if story.themes else "\u2014"
            row_key = table.add_row(story.title, story.tier or "\u2014", avg, themes)
            self._row_to_story[row_key] = story

        if assessment.stories:
            self._show_detail(assessment.stories[0])
        else:
            self.query_one("#detail", Static).update("No stories in this newsletter")

    def _show_detail(self, story: Story) -> None:
        self.query_one("#detail", Static).update(format_detail(story))

    def action_cycle_tier(self) -> None:
        idx = TIER_CYCLE.index(self._tier_filter)
        self._tier_filter = TIER_CYCLE[(idx + 1) % len(TIER_CYCLE)]
        self._apply_filters()

    def action_cycle_theme(self) -> None:
        idx = THEME_CYCLE.index(self._theme_filter)
        self._theme_filter = THEME_CYCLE[(idx + 1) % len(THEME_CYCLE)]
        self._apply_filters()

    def _apply_filters(self) -> None:
        filtered = self.all_assessments
        if self._tier_filter:
            filtered = filter_by_tier(filtered, self._tier_filter)
        if self._theme_filter:
            filtered = filter_by_theme(filtered, self._theme_filter)
        self.filtered_assessments = filtered
        self._populate_newsletters()
        self._update_filter_bar()

        if not self.filtered_assessments:
            story_table = self.query_one("#stories", DataTable)
            story_table.clear()
            self._row_to_story.clear()
            self.query_one("#detail", Static).update(
                "No newsletters match current filters"
            )


def main():
    """CLI entry point for the newsletter assessment TUI."""
    import argparse

    parser = argparse.ArgumentParser(description="Browse newsletter assessments")
    parser.add_argument(
        "file",
        nargs="?",
        default="data/newsletter_assessments.jsonl",
        help="Path to the JSONL assessment file (default: data/newsletter_assessments.jsonl)",
    )
    args = parser.parse_args()

    assessments = load_assessments(args.file)
    app = AssessmentApp(assessments)
    app.run()


if __name__ == "__main__":
    main()
