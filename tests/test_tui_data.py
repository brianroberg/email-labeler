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
