"""Tests for newsletter_review TUI — pure data functions only."""

import json

import pytest

from newsletter_review.tui import (
    apply_filters,
    build_detail_lines,
    format_filter_summary,
    format_list_row,
    load_assessments,
    wrap_text,
)


def _make_record(
    *,
    subject="February Update",
    sender="john@dm.org",
    overall_tier="good",
    stories=None,
    thread_id="t_001",
    message_id="msg_001",
    timestamp="2026-02-20T12:00:00Z",
):
    if stories is None:
        stories = [
            {
                "title": "A Great Story",
                "text": "Once upon a time in a campus ministry...",
                "scores": {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
                "average_score": 3.5,
                "tier": "good",
                "themes": ["scripture", "church"],
                "quality_cot": "The story focuses on one idea.",
                "theme_cot": "This illustrates Scripture study.",
            }
        ]
    return {
        "timestamp": timestamp,
        "message_id": message_id,
        "thread_id": thread_id,
        "from": sender,
        "subject": subject,
        "overall_tier": overall_tier,
        "stories": stories,
    }


# ---------------------------------------------------------------------------
# load_assessments
# ---------------------------------------------------------------------------

class TestLoadAssessments:
    def test_loads_jsonl_records(self, tmp_path):
        f = tmp_path / "assessments.jsonl"
        r1 = _make_record(thread_id="t1")
        r2 = _make_record(thread_id="t2")
        f.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")

        records = load_assessments(f)
        assert len(records) == 2
        assert records[0]["thread_id"] == "t1"
        assert records[1]["thread_id"] == "t2"

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "assessments.jsonl"
        r = _make_record()
        f.write_text("\n" + json.dumps(r) + "\n\n")

        records = load_assessments(f)
        assert len(records) == 1

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_assessments(tmp_path / "nonexistent.jsonl")

    def test_returns_empty_for_empty_file(self, tmp_path):
        f = tmp_path / "assessments.jsonl"
        f.write_text("")

        records = load_assessments(f)
        assert records == []


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------

class TestApplyFilters:
    def test_no_filters_returns_all(self):
        records = [_make_record(thread_id="t1"), _make_record(thread_id="t2")]
        result = apply_filters(records)
        assert len(result) == 2

    def test_tier_filter(self):
        records = [
            _make_record(overall_tier="good"),
            _make_record(overall_tier="poor"),
            _make_record(overall_tier="good"),
        ]
        result = apply_filters(records, tier="good")
        assert len(result) == 2

    def test_tier_filter_excludes_none_tier(self):
        records = [_make_record(overall_tier=None)]
        result = apply_filters(records, tier="good")
        assert result == []

    def test_theme_filter_matches_story_theme(self):
        records = [
            _make_record(stories=[{
                "title": "S", "text": "T", "scores": None,
                "average_score": None, "tier": None,
                "themes": ["scripture", "church"],
                "quality_cot": "", "theme_cot": "",
            }]),
            _make_record(stories=[{
                "title": "S", "text": "T", "scores": None,
                "average_score": None, "tier": None,
                "themes": ["disciple_making"],
                "quality_cot": "", "theme_cot": "",
            }]),
        ]
        result = apply_filters(records, theme="scripture")
        assert len(result) == 1

    def test_theme_filter_case_insensitive(self):
        records = [_make_record()]  # default has "scripture" theme
        result = apply_filters(records, theme="SCRIPTURE")
        assert len(result) == 1

    def test_sender_filter_substring_match(self):
        records = [
            _make_record(sender="john@dm.org"),
            _make_record(sender="jane@other.org"),
        ]
        result = apply_filters(records, sender="dm.org")
        assert len(result) == 1
        assert result[0]["from"] == "john@dm.org"

    def test_sender_filter_case_insensitive(self):
        records = [_make_record(sender="John@DM.org")]
        result = apply_filters(records, sender="john@dm.org")
        assert len(result) == 1

    def test_multiple_filters_are_anded(self):
        records = [
            _make_record(overall_tier="good", sender="john@dm.org"),
            _make_record(overall_tier="poor", sender="john@dm.org"),
            _make_record(overall_tier="good", sender="jane@other.org"),
        ]
        result = apply_filters(records, tier="good", sender="john")
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        result = apply_filters([], tier="good")
        assert result == []


# ---------------------------------------------------------------------------
# _format_list_row
# ---------------------------------------------------------------------------

class TestFormatListRow:
    def test_includes_tier(self):
        row = format_list_row(_make_record(overall_tier="excellent"), 120)
        assert "excellent" in row

    def test_includes_sender(self):
        row = format_list_row(_make_record(sender="john@dm.org"), 120)
        assert "john@dm.org" in row

    def test_includes_story_count(self):
        record = _make_record()  # 1 story
        row = format_list_row(record, 120)
        assert "1" in row

    def test_truncates_long_subject(self):
        record = _make_record(subject="A" * 200)
        row = format_list_row(record, 80)
        assert "..." in row
        assert len(row) <= 80

    def test_handles_none_tier(self):
        row = format_list_row(_make_record(overall_tier=None), 120)
        # Should not crash; should show a placeholder
        assert row  # non-empty string


# ---------------------------------------------------------------------------
# _build_detail_lines
# ---------------------------------------------------------------------------

class TestBuildDetailLines:
    def test_includes_subject_and_sender(self):
        lines = build_detail_lines(_make_record(subject="Feb Update", sender="john@dm.org"))
        text = "\n".join(lines)
        assert "Feb Update" in text
        assert "john@dm.org" in text

    def test_includes_overall_tier(self):
        lines = build_detail_lines(_make_record(overall_tier="excellent"))
        text = "\n".join(lines)
        assert "excellent" in text

    def test_includes_story_title_and_tier(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "A Great Story" in text
        assert "good" in text

    def test_includes_quality_scores(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "simple" in text.lower()
        assert "4" in text

    def test_includes_quality_cot(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "The story focuses on one idea." in text

    def test_includes_theme_cot(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "This illustrates Scripture study." in text

    def test_includes_themes(self):
        record = _make_record()
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "scripture" in text

    def test_handles_missing_scores(self):
        record = _make_record(stories=[{
            "title": "No Scores", "text": "Content",
            "scores": None, "average_score": None, "tier": None,
            "themes": [], "quality_cot": "", "theme_cot": "",
        }])
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "No Scores" in text

    def test_handles_multiple_stories(self):
        stories = [
            {
                "title": "Story A", "text": "Content A",
                "scores": {"simple": 5, "concrete": 5, "personal": 5, "dynamic": 5},
                "average_score": 5.0, "tier": "excellent",
                "themes": ["scripture"], "quality_cot": "cot A", "theme_cot": "theme A",
            },
            {
                "title": "Story B", "text": "Content B",
                "scores": {"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
                "average_score": 2.0, "tier": "fair",
                "themes": ["church"], "quality_cot": "cot B", "theme_cot": "theme B",
            },
        ]
        record = _make_record(stories=stories)
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "Story A" in text
        assert "Story B" in text
        assert "1/2" in text
        assert "2/2" in text

    def test_handles_empty_stories_list(self):
        record = _make_record(stories=[])
        lines = build_detail_lines(record)
        text = "\n".join(lines)
        assert "No stories" in text.lower() or "0" in text


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# format_filter_summary
# ---------------------------------------------------------------------------

class TestFormatFilterSummary:
    def test_no_filters(self):
        assert format_filter_summary(tier=None, theme=None, sender=None) == ""

    def test_tier_only(self):
        result = format_filter_summary(tier="good", theme=None, sender=None)
        assert "tier:good" in result

    def test_theme_only(self):
        result = format_filter_summary(tier=None, theme="scripture", sender=None)
        assert "theme:scripture" in result

    def test_sender_only(self):
        result = format_filter_summary(tier=None, theme=None, sender="dm.org")
        assert "sender:dm.org" in result

    def test_all_filters(self):
        result = format_filter_summary(tier="poor", theme="church", sender="john")
        assert "tier:poor" in result
        assert "theme:church" in result
        assert "sender:john" in result

    def test_returns_empty_for_all_none(self):
        assert format_filter_summary() == ""


# ---------------------------------------------------------------------------
# wrap_text
# ---------------------------------------------------------------------------

class TestWrapText:
    def test_wraps_long_line(self):
        text = "word " * 30  # ~150 chars
        lines = wrap_text(text, 40)
        assert all(len(line) <= 40 for line in lines)
        assert len(lines) > 1

    def test_preserves_existing_newlines(self):
        text = "line one\nline two\nline three"
        lines = wrap_text(text, 80)
        assert len(lines) >= 3

    def test_zero_width_returns_raw_lines(self):
        text = "hello\nworld"
        lines = wrap_text(text, 0)
        assert "hello" in lines
        assert "world" in lines

    def test_empty_string(self):
        lines = wrap_text("", 80)
        assert lines == [] or lines == [""]
