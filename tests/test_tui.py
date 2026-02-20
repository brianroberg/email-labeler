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
            assert "3" in bar.content


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
            assert "Sarah's Journey" in detail.content

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
            text = detail.content
            assert "Follows one person" in text
            assert "Reflects Christlikeness" in text


class TestFiltering:
    async def test_tier_filter_narrows_list(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nl_table = app.query_one("#newsletters", DataTable)
            assert nl_table.row_count == 3

            await pilot.press("t")
            await pilot.pause()
            assert nl_table.row_count == 1

    async def test_tier_filter_cycles_back_to_all(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nl_table = app.query_one("#newsletters", DataTable)
            for _ in range(5):
                await pilot.press("t")
                await pilot.pause()
            assert nl_table.row_count == 3

    async def test_theme_filter_narrows_list(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nl_table = app.query_one("#newsletters", DataTable)

            await pilot.press("h")
            await pilot.pause()
            assert nl_table.row_count == 1

    async def test_filter_bar_updates(self, sample_assessments):
        app = AssessmentApp(sample_assessments)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one("#filter-bar", Static)
            assert "All" in bar.content

            await pilot.press("t")
            await pilot.pause()
            assert "excellent" in bar.content
