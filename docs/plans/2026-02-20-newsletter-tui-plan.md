# Newsletter Assessment TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Textual TUI to browse and filter newsletter assessment JSONL data with tier/theme filtering and CoT inspection.

**Architecture:** Two flat modules: `tui_data.py` (pure data loading, filtering, formatting) and `tui.py` (Textual App with three-panel List→Detail layout). Entry point via `[project.scripts]`.

**Tech Stack:** Python 3.14, Textual, pytest, pytest-asyncio

---

### Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add textual to dependencies**

Add `"textual>=1.0.0"` to the `dependencies` list in `pyproject.toml`.

**Step 2: Install and verify**

Run: `cd /workspaces/email-labeler && uv sync`
Then: `uv run python -c "import textual; print(textual.__version__)"`
Expected: Version number printed

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add textual dependency for newsletter assessment TUI"
```

---

### Task 2: Data Models and JSONL Loading (TDD)

**Files:**
- Create: `tests/test_tui_data.py`
- Create: `tui_data.py`

**Step 1: Write failing tests**

Create `tests/test_tui_data.py`:

```python
"""Tests for TUI data loading, filtering, and formatting."""

import json

import pytest

from tui_data import Assessment, Story, load_assessments


def _write_jsonl(tmp_path, records):
    """Helper: write a list of dicts as JSONL to a temp file, return path."""
    path = tmp_path / "assessments.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _make_story_dict(**overrides):
    """Create a story dict with sensible defaults."""
    base = {
        "title": "A Story",
        "text": "Story content.",
        "scores": {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
        "average_score": 3.5,
        "tier": "good",
        "themes": ["scripture"],
        "quality_cot": "Quality reasoning.",
        "theme_cot": "Theme reasoning.",
    }
    base.update(overrides)
    return base


def _make_record(**overrides):
    """Create an assessment record dict with sensible defaults."""
    base = {
        "timestamp": "2026-02-19T14:30:00+00:00",
        "message_id": "msg001",
        "thread_id": "t001",
        "from": "john@dm.org",
        "subject": "Feb Update",
        "overall_tier": "good",
        "stories": [_make_story_dict()],
    }
    base.update(overrides)
    return base


class TestLoadAssessments:
    def test_loads_valid_jsonl(self, tmp_path):
        path = _write_jsonl(tmp_path, [_make_record(), _make_record(message_id="msg002")])
        assessments = load_assessments(path)
        assert len(assessments) == 2
        assert isinstance(assessments[0], Assessment)
        assert assessments[0].message_id == "msg001"
        assert assessments[1].message_id == "msg002"

    def test_parses_story_fields(self, tmp_path):
        path = _write_jsonl(tmp_path, [_make_record()])
        story = load_assessments(path)[0].stories[0]
        assert isinstance(story, Story)
        assert story.title == "A Story"
        assert story.scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}
        assert story.average_score == 3.5
        assert story.tier == "good"
        assert story.themes == ["scripture"]
        assert story.quality_cot == "Quality reasoning."
        assert story.theme_cot == "Theme reasoning."

    def test_maps_from_field_to_sender(self, tmp_path):
        record = _make_record()
        record["from"] = "jane@dm.org"
        path = _write_jsonl(tmp_path, [record])
        assert load_assessments(path)[0].sender == "jane@dm.org"

    def test_handles_null_fields(self, tmp_path):
        story = _make_story_dict(scores=None, average_score=None, tier=None)
        path = _write_jsonl(tmp_path, [_make_record(overall_tier=None, stories=[story])])
        assessment = load_assessments(path)[0]
        assert assessment.overall_tier is None
        assert assessment.stories[0].scores is None
        assert assessment.stories[0].tier is None

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_assessments(str(path)) == []

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        valid = json.dumps(_make_record())
        path.write_text(f"{valid}\nnot valid json\n{valid}\n")
        assert len(load_assessments(str(path))) == 2

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_assessments(str(tmp_path / "nonexistent.jsonl")) == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py -v`
Expected: `ModuleNotFoundError: No module named 'tui_data'`

**Step 3: Implement dataclasses and load_assessments**

Create `tui_data.py`:

```python
"""Data loading, filtering, and formatting for the newsletter assessment TUI."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Story:
    title: str
    text: str
    scores: dict[str, int] | None
    average_score: float | None
    tier: str | None
    themes: list[str]
    quality_cot: str
    theme_cot: str


@dataclass
class Assessment:
    timestamp: str
    message_id: str
    thread_id: str
    sender: str
    subject: str
    overall_tier: str | None
    stories: list[Story]


def load_assessments(path: str) -> list[Assessment]:
    """Load newsletter assessments from a JSONL file.

    Skips malformed lines with a warning. Returns empty list if file
    doesn't exist or is empty.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    assessments = []
    for line_num, line in enumerate(file_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            stories = [
                Story(
                    title=s["title"],
                    text=s["text"],
                    scores=s.get("scores"),
                    average_score=s.get("average_score"),
                    tier=s.get("tier"),
                    themes=s.get("themes", []),
                    quality_cot=s.get("quality_cot", ""),
                    theme_cot=s.get("theme_cot", ""),
                )
                for s in record.get("stories", [])
            ]
            assessments.append(
                Assessment(
                    timestamp=record["timestamp"],
                    message_id=record["message_id"],
                    thread_id=record["thread_id"],
                    sender=record["from"],
                    subject=record["subject"],
                    overall_tier=record.get("overall_tier"),
                    stories=stories,
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Skipping malformed line %d: %s", line_num, exc)

    return assessments
```

**Step 4: Run tests to verify they pass**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add tui_data.py tests/test_tui_data.py
git commit -m "Add data models and JSONL loading for newsletter TUI"
```

---

### Task 3: Filter Functions (TDD)

**Files:**
- Modify: `tests/test_tui_data.py`
- Modify: `tui_data.py`

**Step 1: Write failing tests**

Add imports at top of `tests/test_tui_data.py`:

```python
from tui_data import available_themes, available_tiers, filter_by_theme, filter_by_tier
```

Append test classes:

```python
@pytest.fixture
def three_assessments(tmp_path):
    """Three assessments covering different tiers and themes."""
    records = [
        _make_record(
            message_id="msg001",
            subject="Penn State",
            overall_tier="excellent",
            stories=[
                _make_story_dict(themes=["christlikeness", "disciple_making"]),
                _make_story_dict(tier="fair", themes=["church"]),
            ],
        ),
        _make_record(
            message_id="msg002",
            subject="Ohio State",
            overall_tier="good",
            stories=[_make_story_dict(themes=["scripture"])],
        ),
        _make_record(
            message_id="msg003",
            subject="Michigan",
            overall_tier=None,
            stories=[_make_story_dict(scores=None, tier=None, themes=["vocation_family"])],
        ),
    ]
    path = _write_jsonl(tmp_path, records)
    return load_assessments(path)


class TestFilterByTier:
    def test_filters_excellent(self, three_assessments):
        result = filter_by_tier(three_assessments, "excellent")
        assert len(result) == 1
        assert result[0].subject == "Penn State"

    def test_filters_good(self, three_assessments):
        result = filter_by_tier(three_assessments, "good")
        assert len(result) == 1
        assert result[0].subject == "Ohio State"

    def test_no_matches_returns_empty(self, three_assessments):
        assert filter_by_tier(three_assessments, "poor") == []

    def test_null_tier_not_matched(self, three_assessments):
        result = filter_by_tier(three_assessments, "excellent")
        assert all(a.overall_tier is not None for a in result)


class TestFilterByTheme:
    def test_filters_by_theme(self, three_assessments):
        result = filter_by_theme(three_assessments, "scripture")
        assert len(result) == 1
        assert result[0].subject == "Ohio State"

    def test_matches_any_story_theme(self, three_assessments):
        result = filter_by_theme(three_assessments, "church")
        assert len(result) == 1
        assert result[0].subject == "Penn State"

    def test_no_matches_returns_empty(self, three_assessments):
        assert filter_by_theme(three_assessments, "nonexistent") == []


class TestAvailableTiers:
    def test_returns_unique_tiers(self, three_assessments):
        assert set(available_tiers(three_assessments)) == {"excellent", "good"}

    def test_excludes_none(self, three_assessments):
        assert None not in available_tiers(three_assessments)

    def test_empty_list(self):
        assert available_tiers([]) == []


class TestAvailableThemes:
    def test_returns_unique_themes(self, three_assessments):
        assert set(available_themes(three_assessments)) == {
            "christlikeness", "disciple_making", "church", "scripture", "vocation_family",
        }

    def test_empty_list(self):
        assert available_themes([]) == []
```

**Step 2: Run to verify new tests fail**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py -v`
Expected: ImportError for `filter_by_tier` etc.

**Step 3: Implement filter functions**

Append to `tui_data.py`:

```python
def filter_by_tier(assessments: list[Assessment], tier: str) -> list[Assessment]:
    """Return assessments matching the given overall tier."""
    return [a for a in assessments if a.overall_tier == tier]


def filter_by_theme(assessments: list[Assessment], theme: str) -> list[Assessment]:
    """Return assessments where any story has the given theme."""
    return [a for a in assessments if any(theme in s.themes for s in a.stories)]


def available_tiers(assessments: list[Assessment]) -> list[str]:
    """Return sorted unique non-None tier values present in assessments."""
    return sorted({a.overall_tier for a in assessments if a.overall_tier is not None})


def available_themes(assessments: list[Assessment]) -> list[str]:
    """Return sorted unique theme values across all stories."""
    themes: set[str] = set()
    for a in assessments:
        for s in a.stories:
            themes.update(s.themes)
    return sorted(themes)
```

**Step 4: Run to verify they pass**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tui_data.py tests/test_tui_data.py
git commit -m "Add filtering and utility functions for newsletter TUI data"
```

---

### Task 4: Detail Formatting (TDD)

**Files:**
- Modify: `tests/test_tui_data.py`
- Modify: `tui_data.py`

**Step 1: Write failing tests**

Add import at top of `tests/test_tui_data.py`:

```python
from tui_data import format_detail
```

Append:

```python
class TestFormatDetail:
    def test_includes_title(self):
        story = Story(
            title="Sarah's Journey", text="Campus story.",
            scores={"simple": 5, "concrete": 4, "personal": 5, "dynamic": 5},
            average_score=4.75, tier="excellent",
            themes=["christlikeness", "disciple_making"],
            quality_cot="Follows one person.", theme_cot="Reflects Christlikeness.",
        )
        assert "Sarah's Journey" in format_detail(story)

    def test_includes_scores(self):
        story = Story(
            title="T", text="X",
            scores={"simple": 5, "concrete": 4, "personal": 5, "dynamic": 5},
            average_score=4.75, tier="excellent",
            themes=[], quality_cot="", theme_cot="",
        )
        result = format_detail(story)
        assert "simple: 5" in result
        assert "concrete: 4" in result

    def test_null_scores_shows_placeholder(self):
        story = Story(
            title="T", text="X", scores=None, average_score=None, tier=None,
            themes=["scripture"], quality_cot="", theme_cot="R.",
        )
        assert "not available" in format_detail(story).lower()

    def test_includes_cot_text(self):
        story = Story(
            title="T", text="X", scores=None, average_score=None, tier=None,
            themes=[], quality_cot="Quality reasoning here.",
            theme_cot="Theme reasoning here.",
        )
        result = format_detail(story)
        assert "Quality reasoning here." in result
        assert "Theme reasoning here." in result

    def test_empty_cot_shows_placeholder(self):
        story = Story(
            title="T", text="X", scores=None, average_score=None, tier=None,
            themes=[], quality_cot="", theme_cot="",
        )
        assert "no reasoning recorded" in format_detail(story).lower()

    def test_includes_story_text(self):
        story = Story(
            title="T", text="Full story content here.",
            scores=None, average_score=None, tier=None,
            themes=[], quality_cot="", theme_cot="",
        )
        assert "Full story content here." in format_detail(story)
```

**Step 2: Run to verify they fail**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py::TestFormatDetail -v`
Expected: ImportError for `format_detail`

**Step 3: Implement format_detail**

Append to `tui_data.py`:

```python
def format_detail(story: Story) -> str:
    """Format a Story as readable text for the detail panel."""
    tier = story.tier or "\u2014"
    avg = f"{story.average_score:.2f}" if story.average_score is not None else "\u2014"

    if story.scores:
        scores_str = "  ".join(f"{k}: {v}" for k, v in story.scores.items())
    else:
        scores_str = "(not available)"

    themes_str = ", ".join(story.themes) if story.themes else "(none)"
    quality_cot = story.quality_cot or "(no reasoning recorded)"
    theme_cot = story.theme_cot or "(no reasoning recorded)"

    return (
        f"{story.title}\n"
        f"Tier: {tier}  |  Average: {avg}\n"
        f"Scores: {scores_str}\n"
        f"Themes: {themes_str}\n"
        f"\n\u2500\u2500 Quality CoT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{quality_cot}\n"
        f"\n\u2500\u2500 Theme CoT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{theme_cot}\n"
        f"\n\u2500\u2500 Story Text \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{story.text}\n"
    )
```

**Step 4: Run to verify they pass**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tui_data.py tests/test_tui_data.py
git commit -m "Add detail formatting function for newsletter TUI"
```

---

### Task 5: TUI App Shell with Newsletter List (TDD)

**Files:**
- Create: `tests/test_tui.py`
- Create: `tui.py`

**Step 1: Write failing tests**

Create `tests/test_tui.py`:

```python
"""Tests for the newsletter assessment TUI app."""

import pytest

from textual.widgets import DataTable, Static

from tui import AssessmentApp
from tui_data import Assessment, Story


def _story(**overrides):
    base = dict(
        title="A Story", text="Content.",
        scores={"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
        average_score=3.5, tier="good", themes=["scripture"],
        quality_cot="Quality.", theme_cot="Theme.",
    )
    base.update(overrides)
    return Story(**base)


@pytest.fixture
def sample_assessments():
    return [
        Assessment(
            timestamp="2026-02-19T14:30:00+00:00", message_id="msg001",
            thread_id="t001", sender="john@dm.org",
            subject="Feb Update - Penn State", overall_tier="excellent",
            stories=[
                _story(
                    title="Sarah's Journey", tier="excellent", average_score=4.75,
                    themes=["christlikeness", "disciple_making"],
                    quality_cot="Follows one person.", theme_cot="Reflects Christlikeness.",
                ),
                _story(
                    title="Campus Outreach Week", tier="fair", average_score=2.25,
                    themes=["church"],
                    quality_cot="Multiple events.", theme_cot="Fellowship.",
                ),
            ],
        ),
        Assessment(
            timestamp="2026-01-15T10:00:00+00:00", message_id="msg002",
            thread_id="t002", sender="jane@dm.org",
            subject="Jan Report - Ohio State", overall_tier="good",
            stories=[_story(title="Jake's Bible Study", themes=["scripture"])],
        ),
        Assessment(
            timestamp="2025-12-20T08:00:00+00:00", message_id="msg003",
            thread_id="t003", sender="mike@dm.org",
            subject="Dec Newsletter - Michigan", overall_tier=None,
            stories=[
                _story(
                    title="Graduation Reflections", scores=None,
                    average_score=None, tier=None, themes=["vocation_family"],
                    quality_cot="", theme_cot="Career decisions.",
                ),
            ],
        ),
    ]


class TestAppLaunch:
    async def test_shows_newsletter_table(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            table = app.query_one("#newsletters", DataTable)
            assert table.row_count == 3

    async def test_newsletter_table_has_correct_columns(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            table = app.query_one("#newsletters", DataTable)
            labels = [col.label.plain for col in table.columns.values()]
            assert "Subject" in labels
            assert "Tier" in labels

    async def test_shows_filter_bar(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            bar = app.query_one("#filter-bar", Static)
            assert "3" in str(bar.renderable)
```

**Step 2: Run to verify they fail**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui.py -v`
Expected: `ModuleNotFoundError: No module named 'tui'` (the app module)

**Step 3: Implement AssessmentApp**

Create `tui.py`:

```python
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
        self.current_tier: str | None = None
        self.current_theme: str | None = None
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
        tier_text = self.current_tier or "All"
        theme_text = self.current_theme or "All"
        count = len(self.filtered_assessments)
        total = len(self.all_assessments)
        self.query_one("#filter-bar", Static).update(
            f"Tier: {tier_text}  |  Theme: {theme_text}"
            f"  |  {count} of {total} newsletters"
        )
```

**Step 4: Run to verify they pass**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add tui.py tests/test_tui.py
git commit -m "Add TUI app shell with newsletter list and filter bar"
```

---

### Task 6: Story and Detail Drill-Down (TDD)

**Files:**
- Modify: `tests/test_tui.py`
- Modify: `tui.py`

**Step 1: Write failing tests**

Append to `tests/test_tui.py`:

```python
class TestDrillDown:
    async def test_first_newsletter_stories_shown_on_mount(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            story_table = app.query_one("#stories", DataTable)
            assert story_table.row_count == 2

    async def test_first_story_detail_shown_on_mount(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            detail = app.query_one("#detail", Static)
            assert "Sarah's Journey" in str(detail.renderable)

    async def test_navigating_newsletters_updates_stories(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            story_table = app.query_one("#stories", DataTable)
            assert story_table.row_count == 1

    async def test_detail_shows_cot_text(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            detail = app.query_one("#detail", Static)
            text = str(detail.renderable)
            assert "Follows one person" in text
            assert "Reflects Christlikeness" in text
```

**Step 2: Run to verify they fail**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui.py::TestDrillDown -v`
Expected: FAIL -- story table empty, detail not updated

**Step 3: Implement RowHighlighted handlers**

Add these methods to `AssessmentApp` in `tui.py`:

```python
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
```

**Step 4: Run to verify they pass**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tui.py tests/test_tui.py
git commit -m "Add newsletter -> story -> detail drill-down navigation"
```

---

### Task 7: Tier and Theme Filtering (TDD)

**Files:**
- Modify: `tests/test_tui.py`
- Modify: `tui.py`

**Step 1: Write failing tests**

Append to `tests/test_tui.py`:

```python
class TestFiltering:
    async def test_tier_filter_narrows_list(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nl_table = app.query_one("#newsletters", DataTable)
            assert nl_table.row_count == 3

            await pilot.press("t")  # All -> excellent
            await pilot.pause()
            assert nl_table.row_count == 1

    async def test_tier_filter_cycles_back_to_all(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nl_table = app.query_one("#newsletters", DataTable)
            for _ in range(5):  # All -> excellent -> good -> fair -> poor -> All
                await pilot.press("t")
                await pilot.pause()
            assert nl_table.row_count == 3

    async def test_theme_filter_narrows_list(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nl_table = app.query_one("#newsletters", DataTable)

            await pilot.press("h")  # All -> scripture
            await pilot.pause()
            assert nl_table.row_count == 1

    async def test_filter_bar_updates(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one("#filter-bar", Static)
            assert "All" in str(bar.renderable)

            await pilot.press("t")
            await pilot.pause()
            assert "excellent" in str(bar.renderable)
```

**Step 2: Run to verify they fail**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui.py::TestFiltering -v`
Expected: FAIL -- actions not implemented

**Step 3: Implement filter actions**

Add these methods to `AssessmentApp` in `tui.py`:

```python
    def action_cycle_tier(self) -> None:
        idx = TIER_CYCLE.index(self.current_tier)
        self.current_tier = TIER_CYCLE[(idx + 1) % len(TIER_CYCLE)]
        self._apply_filters()

    def action_cycle_theme(self) -> None:
        idx = THEME_CYCLE.index(self.current_theme)
        self.current_theme = THEME_CYCLE[(idx + 1) % len(THEME_CYCLE)]
        self._apply_filters()

    def _apply_filters(self) -> None:
        filtered = self.all_assessments
        if self.current_tier:
            filtered = filter_by_tier(filtered, self.current_tier)
        if self.current_theme:
            filtered = filter_by_theme(filtered, self.current_theme)
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
```

**Step 4: Run to verify they pass**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tui.py tests/test_tui.py
git commit -m "Add tier and theme filter cycling with t and h key bindings"
```

---

### Task 8: CLI Entry Point

**Files:**
- Modify: `tui.py`
- Modify: `pyproject.toml`

**Step 1: Add main() with argparse**

Add to the bottom of `tui.py`:

```python
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
```

**Step 2: Add [project.scripts] to pyproject.toml**

Add after the `[project]` section:

```toml
[project.scripts]
tui = "tui:main"
```

**Step 3: Verify it runs**

Run: `cd /workspaces/email-labeler && uv run tui --help`
Expected: Shows argparse help with file argument

**Step 4: Run full test suite**

Run: `cd /workspaces/email-labeler && uv run --extra dev pytest tests/test_tui_data.py tests/test_tui.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tui.py pyproject.toml uv.lock
git commit -m "Add CLI entry point: uv run tui [file]"
```
