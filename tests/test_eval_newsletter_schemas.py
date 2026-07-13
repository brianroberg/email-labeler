"""Tests for evals.newsletter_schemas — round-trip serialization + defaults."""

import json

import pytest

from evals.newsletter_schemas import (
    ExtractionPrediction,
    GoldenNewsletter,
    GoldenStory,
    NewsletterRunMeta,
    NewsletterThinkingEntry,
    StoryPrediction,
)


class TestGoldenStory:
    def test_round_trip_via_json(self):
        s = GoldenStory(
            story_id="t_001:0",
            text="The full story text goes here.",
            expected_scores={"simple": 3, "concrete": 3, "personal": 2, "dynamic": 1},
            expected_tier="good",
            expected_themes={"scripture": "emphasized", "church": "present"},
            reviewed=True,
            notes="clear story",
            excluded=False,
        )
        d = s.to_dict()
        assert "title" not in d
        restored = GoldenStory.from_dict(json.loads(json.dumps(d)))
        assert restored.story_id == "t_001:0"
        assert restored.text == "The full story text goes here."
        assert restored.expected_scores == {
            "simple": 3,
            "concrete": 3,
            "personal": 2,
            "dynamic": 1,
        }
        assert restored.expected_tier == "good"
        assert restored.expected_themes == {"scripture": "emphasized", "church": "present"}
        assert restored.reviewed is True
        assert restored.notes == "clear story"
        assert restored.excluded is False

    def test_defaults_when_keys_missing(self):
        s = GoldenStory.from_dict({"story_id": "x:0", "text": "B"})
        assert s.expected_scores is None
        assert s.expected_tier is None
        assert s.expected_themes == {}
        assert s.reviewed is False
        assert s.notes == ""
        assert s.excluded is False

    def test_legacy_title_key_is_tolerated(self):
        # Golden sets written before titles were removed still carry a "title"
        # key; loading them must ignore the extra key, not crash.
        s = GoldenStory.from_dict({"story_id": "x:0", "title": "Old Title", "text": "B"})
        assert s.text == "B"
        assert not hasattr(s, "title")


class TestOldSchemeThemeGuard:
    """Loading old-scheme (pre-#53) theme LISTS or invalid grades must raise a
    clear, actionable error naming the record — not an opaque dict-update
    ValueError (Finding 1) or a silently-accepted junk grade that crashes the
    labeling TUI mid-session (Finding 2)."""

    def test_golden_list_themes_raise_actionable_error(self):
        with pytest.raises(ValueError, match="old-scheme") as exc:
            GoldenStory.from_dict(
                {"story_id": "t_001:2", "text": "B", "expected_themes": ["church"]}
            )
        msg = str(exc.value)
        assert "t_001:2" in msg  # story_id so the record is locatable
        assert "church" in msg  # the offending value

    def test_prediction_list_themes_raise_actionable_error(self):
        with pytest.raises(ValueError, match="old-scheme") as exc:
            StoryPrediction.from_dict(
                {
                    "story_id": "t_001:3",
                    "thread_id": "t_001",
                    "predicted_themes": ["scripture"],
                }
            )
        assert "t_001:3" in str(exc.value)

    def test_invalid_theme_grade_rejected(self):
        # Case variant: _THEME_CYCLE/_GRADE_ABBR/parse_themes only ever produce
        # exactly "present"/"emphasized"; "Present" would KeyError the TUI.
        with pytest.raises(ValueError, match="present.*emphasized|grade") as exc:
            GoldenStory.from_dict(
                {
                    "story_id": "t_001:4",
                    "text": "B",
                    "expected_themes": {"scripture": "Present"},
                }
            )
        msg = str(exc.value)
        assert "t_001:4" in msg
        assert "scripture" in msg
        assert "Present" in msg

    def test_absent_grade_rejected(self):
        # "absent" is represented by omission and is never stored as a grade.
        with pytest.raises(ValueError):
            GoldenStory.from_dict(
                {
                    "story_id": "t_001:5",
                    "text": "B",
                    "expected_themes": {"scripture": "absent"},
                }
            )

    def test_prediction_invalid_predicted_grade_rejected(self):
        with pytest.raises(ValueError):
            StoryPrediction.from_dict(
                {
                    "story_id": "t_001:6",
                    "thread_id": "t_001",
                    "predicted_themes": {"church": "maybe"},
                }
            )


class TestOldSchemeScoreGuard:
    """Legacy 1-5 scores must be rejected loudly at load, not accepted silently
    (Finding 3) — otherwise the report computes MAE against the wrong scale and
    the TUI shows out-of-range keys."""

    def test_out_of_range_expected_score_rejected(self):
        with pytest.raises(ValueError, match="1..3|old-scheme|re-migrat") as exc:
            GoldenStory.from_dict(
                {
                    "story_id": "t_001:7",
                    "text": "B",
                    "expected_scores": {"simple": 3, "concrete": 5},
                }
            )
        msg = str(exc.value)
        assert "t_001:7" in msg
        assert "concrete" in msg  # the offending dimension
        assert "5" in msg  # the offending value

    def test_out_of_range_predicted_score_rejected(self):
        with pytest.raises(ValueError) as exc:
            StoryPrediction.from_dict(
                {
                    "story_id": "t_001:8",
                    "thread_id": "t_001",
                    "predicted_scores": {"simple": 4},
                }
            )
        assert "t_001:8" in str(exc.value)

    def test_valid_scores_and_missing_scores_still_load(self):
        # 1-3 scores pass; None/missing stays None (unchanged behavior).
        s = GoldenStory.from_dict(
            {
                "story_id": "t_001:9",
                "text": "B",
                "expected_scores": {"simple": 1, "concrete": 2, "personal": 3},
            }
        )
        assert s.expected_scores == {"simple": 1, "concrete": 2, "personal": 3}
        s2 = GoldenStory.from_dict({"story_id": "t_001:10", "text": "B"})
        assert s2.expected_scores is None


class TestGoldenNewsletter:
    def test_round_trip_nested_stories_survive_json(self):
        nl = GoldenNewsletter(
            thread_id="t_001",
            message_id="m_001",
            sender="News <news@dm.org>",
            subject="July update",
            body="raw body verbatim",
            stories=[
                GoldenStory(story_id="t_001:0", text="one"),
                GoldenStory(
                    story_id="t_001:1",
                    text="two",
                    expected_themes={"vocation_family": "present"},
                ),
            ],
            source="harvested",
            harvested_at="2026-07-01T00:00:00+00:00",
            reviewed=True,
            notes="good newsletter",
            excluded=False,
        )
        d = nl.to_dict()
        assert d["type"] == "golden_newsletter"
        restored = GoldenNewsletter.from_dict(json.loads(json.dumps(d)))
        assert restored.thread_id == "t_001"
        assert restored.message_id == "m_001"
        assert restored.sender == "News <news@dm.org>"
        assert restored.subject == "July update"
        assert restored.body == "raw body verbatim"
        assert restored.source == "harvested"
        assert restored.harvested_at == "2026-07-01T00:00:00+00:00"
        assert restored.reviewed is True
        assert restored.notes == "good newsletter"
        assert restored.excluded is False
        # Nested stories survive as GoldenStory instances.
        assert len(restored.stories) == 2
        assert all(isinstance(s, GoldenStory) for s in restored.stories)
        assert restored.stories[0].story_id == "t_001:0"
        assert restored.stories[1].story_id == "t_001:1"
        assert restored.stories[1].text == "two"
        assert restored.stories[1].expected_themes == {"vocation_family": "present"}

    def test_defaults_when_keys_missing(self):
        nl = GoldenNewsletter.from_dict(
            {
                "thread_id": "t",
                "message_id": "m",
                "sender": "s",
                "subject": "sub",
                "body": "b",
            }
        )
        assert nl.stories == []
        assert nl.source == "harvested"
        assert nl.harvested_at == ""
        assert nl.reviewed is False
        assert nl.notes == ""
        assert nl.excluded is False


class TestStoryPrediction:
    def test_round_trip_via_json(self):
        p = StoryPrediction(
            story_id="t_001:0",
            thread_id="t_001",
            expected_scores={"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1},
            expected_tier="good",
            expected_themes={"scripture": "present"},
            predicted_scores={"simple": 2, "concrete": 2, "personal": 3, "dynamic": 1},
            predicted_tier="fair",
            predicted_themes={"scripture": "emphasized", "church": "present"},
            scores_raw="SIMPLE: OK ...",
            themes_raw="scripture: emphasized",
            duration_seconds=1.5,
            error=None,
        )
        d = p.to_dict()
        assert d["type"] == "story_prediction"
        restored = StoryPrediction.from_dict(json.loads(json.dumps(d)))
        assert restored.story_id == "t_001:0"
        assert restored.thread_id == "t_001"
        assert restored.expected_scores == {
            "simple": 3,
            "concrete": 2,
            "personal": 3,
            "dynamic": 1,
        }
        assert restored.expected_tier == "good"
        assert restored.expected_themes == {"scripture": "present"}
        assert restored.predicted_scores == {
            "simple": 2,
            "concrete": 2,
            "personal": 3,
            "dynamic": 1,
        }
        assert restored.predicted_tier == "fair"
        assert restored.predicted_themes == {"scripture": "emphasized", "church": "present"}
        assert restored.scores_raw == "SIMPLE: OK ..."
        assert restored.themes_raw == "scripture: emphasized"
        assert restored.duration_seconds == 1.5
        assert restored.error is None

    def test_defaults_when_keys_missing(self):
        p = StoryPrediction.from_dict({"story_id": "x:0", "thread_id": "x"})
        assert p.expected_scores is None
        assert p.expected_tier is None
        assert p.expected_themes == {}
        assert p.predicted_scores is None
        assert p.predicted_tier is None
        assert p.predicted_themes == {}
        assert p.scores_raw is None
        assert p.themes_raw is None
        assert p.duration_seconds == 0.0
        assert p.error is None


class TestExtractionPrediction:
    def test_round_trip_via_json(self):
        p = ExtractionPrediction(
            thread_id="t_001",
            golden_stories=[{"story_id": "t_001:0", "text": "aaa"}],
            predicted_stories=[{"text": "aaa"}, {"text": "bbb"}],
            duration_seconds=2.0,
            error=None,
        )
        d = p.to_dict()
        assert d["type"] == "extraction_prediction"
        restored = ExtractionPrediction.from_dict(json.loads(json.dumps(d)))
        assert restored.thread_id == "t_001"
        assert restored.golden_stories == [
            {"story_id": "t_001:0", "text": "aaa"}
        ]
        assert restored.predicted_stories == [
            {"text": "aaa"},
            {"text": "bbb"},
        ]
        assert restored.duration_seconds == 2.0
        assert restored.error is None

    def test_defaults_when_keys_missing(self):
        p = ExtractionPrediction.from_dict({"thread_id": "x"})
        assert p.golden_stories == []
        assert p.predicted_stories == []
        assert p.duration_seconds == 0.0
        assert p.error is None


class TestNewsletterThinkingEntry:
    def test_round_trip_via_json(self):
        t = NewsletterThinkingEntry(
            story_id="t_001:0",
            quality_cot="thinking about quality",
            theme_cot="thinking about themes",
        )
        d = t.to_dict()
        assert d["type"] == "thinking"
        restored = NewsletterThinkingEntry.from_dict(json.loads(json.dumps(d)))
        assert restored.story_id == "t_001:0"
        assert restored.quality_cot == "thinking about quality"
        assert restored.theme_cot == "thinking about themes"

    def test_defaults_when_keys_missing(self):
        t = NewsletterThinkingEntry.from_dict({"story_id": "x:0"})
        assert t.quality_cot == ""
        assert t.theme_cot == ""
        assert t.thread_id == ""
        assert t.extraction_cot == ""

    def test_extraction_entry_round_trip_keyed_by_thread(self):
        """An extraction cot entry is per-newsletter: thread_id set, no story_id."""
        t = NewsletterThinkingEntry(
            thread_id="t_001",
            extraction_cot="reasoning about how to segment the newsletter",
        )
        d = t.to_dict()
        assert d["type"] == "thinking"
        restored = NewsletterThinkingEntry.from_dict(json.loads(json.dumps(d)))
        assert restored.thread_id == "t_001"
        assert restored.extraction_cot == "reasoning about how to segment the newsletter"
        assert restored.story_id == ""
        assert restored.quality_cot == ""
        assert restored.theme_cot == ""

    def test_from_dict_tolerates_missing_story_id(self):
        """Older/extraction-only rows have no story_id key at all."""
        t = NewsletterThinkingEntry.from_dict(
            {"thread_id": "t_001", "extraction_cot": "seg cot"}
        )
        assert t.story_id == ""
        assert t.thread_id == "t_001"
        assert t.extraction_cot == "seg cot"


class TestNewsletterRunMeta:
    def test_round_trip_via_json(self):
        meta = NewsletterRunMeta(
            run_id="run123456",
            timestamp="2026-07-01T10:00:00+00:00",
            config_hash="cfghash1234",
            config_path="config.toml",
            newsletter_model="claude-sonnet-4-6",
            golden_set_path="evals/newsletter_golden_set.jsonl",
            golden_set_count=12,
            story_count=40,
            mode="all",
            prompt_hash="promptabc123",
            temperature=0.3,
            max_tokens=8096,
            extra_body={"top_p": 0.9},
            parallelism=4,
            tag="baseline",
            extraction_system_prompt="extract prompt",
            quality_system_prompt="quality prompt",
            theme_system_prompt="theme prompt",
        )
        d = meta.to_dict()
        assert d["type"] == "run_meta"
        restored = NewsletterRunMeta.from_dict(json.loads(json.dumps(d)))
        assert restored.run_id == "run123456"
        assert restored.timestamp == "2026-07-01T10:00:00+00:00"
        assert restored.config_hash == "cfghash1234"
        assert restored.config_path == "config.toml"
        assert restored.newsletter_model == "claude-sonnet-4-6"
        assert restored.golden_set_path == "evals/newsletter_golden_set.jsonl"
        assert restored.golden_set_count == 12
        assert restored.story_count == 40
        assert restored.mode == "all"
        assert restored.prompt_hash == "promptabc123"
        assert restored.temperature == 0.3
        assert restored.max_tokens == 8096
        assert restored.extra_body == {"top_p": 0.9}
        assert restored.parallelism == 4
        assert restored.tag == "baseline"
        assert restored.extraction_system_prompt == "extract prompt"
        assert restored.quality_system_prompt == "quality prompt"
        assert restored.theme_system_prompt == "theme prompt"

    def test_defaults_when_keys_missing(self):
        meta = NewsletterRunMeta.from_dict(
            {
                "run_id": "r",
                "timestamp": "t",
                "config_hash": "h",
                "config_path": "p",
                "newsletter_model": "m",
                "golden_set_path": "g",
                "golden_set_count": 5,
                "story_count": 20,
            }
        )
        assert meta.mode == "all"
        assert meta.prompt_hash == ""
        assert meta.temperature == 0.0
        assert meta.max_tokens == 0
        assert meta.extra_body is None
        assert meta.parallelism == 1
        assert meta.tag == ""
        assert meta.extraction_system_prompt == ""
        assert meta.quality_system_prompt == ""
        assert meta.theme_system_prompt == ""


class TestSeededFromRemoved:
    """Issue #59: seeding is gone, so the provenance field no longer serializes.
    Old JSONL containing the key stays readable (from_dict ignores unknown keys)."""

    def test_golden_newsletter_omits_seeded_from(self):
        nl = GoldenNewsletter(
            thread_id="t", message_id="m", sender="s", subject="sub", body="b"
        )
        assert "seeded_from" not in nl.to_dict()

    def test_run_meta_omits_seeded_from(self):
        meta = NewsletterRunMeta(
            run_id="r1",
            timestamp="2026-07-13T00:00:00+00:00",
            config_hash="h",
            config_path="config.toml",
            newsletter_model="model",
            golden_set_path="golden.jsonl",
            golden_set_count=1,
            story_count=1,
        )
        assert "seeded_from" not in meta.to_dict()

    def test_old_scheme_key_is_ignored_on_load(self):
        nl = GoldenNewsletter.from_dict(
            {
                "thread_id": "t",
                "message_id": "m",
                "sender": "s",
                "subject": "sub",
                "body": "b",
                "seeded_from": "parse_stories",
            }
        )
        assert not hasattr(nl, "seeded_from")
