"""Tests for the pre-#53 assessment-record migration (scripts/migrate_assessments.py).

The production assessments JSONL is append-only and can never be regenerated —
it is the only copy of every grading ever made. When #53 replaced the 1-5
dimension scores with 3-value Poor/OK/Good and list-shaped story themes with
theme->grade dicts, the readers were made to reject the old shapes outright, on
the assumption that the data would be regenerated. That holds for the eval
golden set; it cannot hold for this file, so its pre-#53 history needs
converting in place instead.
"""

import json

import pytest

from newsletter_review.tui import load_assessments
from scripts.migrate_assessments import (
    is_old_scheme,
    migrate_file,
    migrate_record,
)


def _old_record(**overrides):
    """A pre-#53 record: 1-5 dimension scores, list-shaped themes, no
    send_date/model keys (both landed with the same release)."""
    record = {
        "timestamp": "2026-06-02T12:00:00+00:00",
        "message_id": "msg_old",
        "thread_id": "t_old",
        "from": "staff@dm.org",
        "subject": "June Update",
        "overall_tier": "good",
        "stories": [
            {
                "text": "A story about ministry.",
                "scores": {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
                "average_score": 3.5,
                "tier": "good",
                "themes": ["vocation_family", "scripture"],
                "quality_cot": "reasoning",
                "theme_cot": "reasoning",
            }
        ],
    }
    record.update(overrides)
    return record


def _new_record(**overrides):
    record = {
        "timestamp": "2026-07-29T12:00:00+00:00",
        "message_id": "msg_new",
        "thread_id": "t_new",
        "from": "staff@dm.org",
        "subject": "God has done it!",
        "send_date": "2026-07-29T09:00:00+00:00",
        "model": "claude-sonnet-4-6",
        "overall_tier": "good",
        "stories": [
            {
                "text": "A story about ministry.",
                "scores": {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 2},
                "average_score": 2.5,
                "tier": "good",
                "themes": {"scripture": "emphasized"},
                "quality_cot": "",
                "theme_cot": "",
            }
        ],
    }
    record.update(overrides)
    return record


class TestIsOldScheme:
    def test_list_themes_identify_the_old_scheme(self):
        assert is_old_scheme(_old_record()) is True

    def test_empty_list_themes_still_identify_it(self):
        # The old grader stored [] for a story with no themes; the new one stores
        # {}. The list *shape* is the marker, not its contents.
        r = _old_record()
        r["stories"][0]["themes"] = []
        r["stories"][0]["scores"] = {"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2}
        assert is_old_scheme(r) is True

    def test_out_of_range_scores_identify_it_even_with_dict_themes(self):
        r = _new_record()
        r["stories"][0]["scores"] = {"simple": 5, "concrete": 4, "personal": 3, "dynamic": 2}
        assert is_old_scheme(r) is True

    def test_new_scheme_record_is_not_flagged(self):
        assert is_old_scheme(_new_record()) is False

    def test_record_with_no_stories_is_not_flagged(self):
        # A no-stories newsletter has nothing scheme-specific in it, so there is
        # nothing to convert and no way (or need) to tell which scheme wrote it.
        assert is_old_scheme(_new_record(stories=[])) is False


class TestMigrateRecord:
    def test_list_themes_become_present_grades(self):
        out, changed = migrate_record(_old_record())
        assert changed is True
        assert out["stories"][0]["themes"] == {
            "vocation_family": "present", "scripture": "present",
        }

    def test_empty_list_themes_become_an_empty_dict(self):
        r = _old_record()
        r["stories"][0]["themes"] = []
        out, _ = migrate_record(r)
        assert out["stories"][0]["themes"] == {}

    def test_scores_are_bucketed_into_the_three_value_rubric(self):
        r = _old_record()
        r["stories"][0]["scores"] = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        out, _ = migrate_record(r)
        # 1-2 -> Poor(1), 3 -> OK(2), 4-5 -> Good(3).
        assert out["stories"][0]["scores"] == {"a": 1, "b": 1, "c": 2, "d": 3, "e": 3}

    def test_average_score_is_recomputed_from_the_bucketed_scores(self):
        out, _ = migrate_record(_old_record())
        # {4,3,5,2} -> {3,2,3,1} -> mean 2.25
        assert out["stories"][0]["average_score"] == pytest.approx(2.25)

    def test_recorded_tiers_are_preserved_not_recomputed(self):
        """The tier is what the grader concluded under the old rubric, and it is
        what got applied to the email as a Gmail label. Recomputing it from
        lossily re-bucketed dimensions would put a grade in the record that
        disagrees with the label actually on the message."""
        out, _ = migrate_record(_old_record(overall_tier="excellent"))
        assert out["overall_tier"] == "excellent"
        assert out["stories"][0]["tier"] == "good"

    def test_missing_scores_are_left_alone(self):
        r = _old_record()
        r["stories"][0]["scores"] = None
        r["stories"][0]["average_score"] = None
        out, changed = migrate_record(r)
        assert changed is True  # themes still migrated
        assert out["stories"][0]["scores"] is None
        assert out["stories"][0]["average_score"] is None

    def test_provenance_is_recorded(self):
        """A migrated record's Poor/OK/Good labels are re-bucketed 5-point
        judgments, not what the grader actually emitted — the record has to say
        so, or the detail view silently overstates its own precision."""
        out, _ = migrate_record(_old_record())
        assert out["migrated_from"] == "pre-#53"

    def test_other_fields_are_untouched(self):
        out, _ = migrate_record(_old_record())
        for key in ("timestamp", "message_id", "thread_id", "from", "subject"):
            assert out[key] == _old_record()[key]
        assert out["stories"][0]["quality_cot"] == "reasoning"

    def test_new_scheme_record_passes_through_unchanged(self):
        out, changed = migrate_record(_new_record())
        assert changed is False
        assert out == _new_record()
        assert "migrated_from" not in out

    def test_input_record_is_not_mutated(self):
        r = _old_record()
        migrate_record(r)
        assert r["stories"][0]["themes"] == ["vocation_family", "scripture"]


class TestMigrateFile:
    def _write(self, path, records):
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_reports_counts_without_touching_the_file_by_default(self, tmp_path):
        f = tmp_path / "a.jsonl"
        self._write(f, [_old_record(), _new_record(), _old_record()])
        before = f.read_text()

        stats = migrate_file(f)

        assert (stats.total, stats.migrated) == (3, 2)
        assert f.read_text() == before

    def test_in_place_rewrites_the_file_and_keeps_a_backup(self, tmp_path):
        f = tmp_path / "a.jsonl"
        self._write(f, [_old_record(), _new_record()])
        before = f.read_text()

        migrate_file(f, in_place=True)

        assert json.loads(f.read_text().splitlines()[0])["stories"][0]["themes"] == {
            "vocation_family": "present", "scripture": "present",
        }
        assert (tmp_path / "a.jsonl.bak").read_text() == before

    def test_record_order_is_preserved(self, tmp_path):
        f = tmp_path / "a.jsonl"
        self._write(f, [_old_record(thread_id="t1"), _new_record(thread_id="t2"),
                        _old_record(thread_id="t3")])
        migrate_file(f, in_place=True)
        ids = [json.loads(line)["thread_id"] for line in f.read_text().splitlines()]
        assert ids == ["t1", "t2", "t3"]

    def test_blank_lines_are_dropped(self, tmp_path):
        f = tmp_path / "a.jsonl"
        f.write_text(json.dumps(_old_record()) + "\n\n\n")
        stats = migrate_file(f, in_place=True)
        assert stats.total == 1
        assert len(f.read_text().splitlines()) == 1

    def test_migrated_file_loads_through_the_review_tui(self, tmp_path):
        """The point of the exercise: the file the reader rejected must open."""
        f = tmp_path / "a.jsonl"
        self._write(f, [_old_record(), _new_record()])
        with pytest.raises(ValueError, match="old-scheme"):
            load_assessments(f)

        migrate_file(f, in_place=True)

        records = load_assessments(f)
        assert len(records) == 2

    def test_a_malformed_line_aborts_before_anything_is_written(self, tmp_path):
        """Half-migrating the only copy of the grading history is not an option."""
        f = tmp_path / "a.jsonl"
        f.write_text(json.dumps(_old_record()) + "\nnot json\n")
        before = f.read_text()

        with pytest.raises(ValueError, match="line 2"):
            migrate_file(f, in_place=True)

        assert f.read_text() == before
        assert not (tmp_path / "a.jsonl.bak").exists()

    def test_out_writes_a_separate_file_and_leaves_the_input_alone(self, tmp_path):
        f = tmp_path / "a.jsonl"
        out = tmp_path / "migrated.jsonl"
        self._write(f, [_old_record()])
        before = f.read_text()

        migrate_file(f, out=out)

        assert f.read_text() == before
        assert load_assessments(out)
