"""Tests for evals.newsletter_schemas — round-trip serialization + defaults."""

import json

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
            title="A missionary in Nairobi",
            text="The full story text goes here.",
            expected_scores={"simple": 4, "concrete": 5, "personal": 3, "dynamic": 2},
            expected_tier="good",
            expected_themes=["scripture", "church"],
            reviewed=True,
            notes="clear story",
            excluded=False,
        )
        d = s.to_dict()
        restored = GoldenStory.from_dict(json.loads(json.dumps(d)))
        assert restored.story_id == "t_001:0"
        assert restored.title == "A missionary in Nairobi"
        assert restored.text == "The full story text goes here."
        assert restored.expected_scores == {
            "simple": 4,
            "concrete": 5,
            "personal": 3,
            "dynamic": 2,
        }
        assert restored.expected_tier == "good"
        assert restored.expected_themes == ["scripture", "church"]
        assert restored.reviewed is True
        assert restored.notes == "clear story"
        assert restored.excluded is False

    def test_defaults_when_keys_missing(self):
        s = GoldenStory.from_dict({"story_id": "x:0", "title": "T", "text": "B"})
        assert s.expected_scores is None
        assert s.expected_tier is None
        assert s.expected_themes == []
        assert s.reviewed is False
        assert s.notes == ""
        assert s.excluded is False


class TestGoldenNewsletter:
    def test_round_trip_nested_stories_survive_json(self):
        nl = GoldenNewsletter(
            thread_id="t_001",
            message_id="m_001",
            sender="News <news@dm.org>",
            subject="July update",
            body="raw body verbatim",
            stories=[
                GoldenStory(story_id="t_001:0", title="First", text="one"),
                GoldenStory(
                    story_id="t_001:1",
                    title="Second",
                    text="two",
                    expected_themes=["vocation-family"],
                ),
            ],
            source="harvested",
            harvested_at="2026-07-01T00:00:00+00:00",
            seeded_from="parse_stories",
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
        assert restored.seeded_from == "parse_stories"
        assert restored.reviewed is True
        assert restored.notes == "good newsletter"
        assert restored.excluded is False
        # Nested stories survive as GoldenStory instances.
        assert len(restored.stories) == 2
        assert all(isinstance(s, GoldenStory) for s in restored.stories)
        assert restored.stories[0].story_id == "t_001:0"
        assert restored.stories[1].title == "Second"
        assert restored.stories[1].expected_themes == ["vocation-family"]

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
        assert nl.seeded_from == ""
        assert nl.reviewed is False
        assert nl.notes == ""
        assert nl.excluded is False


class TestStoryPrediction:
    def test_round_trip_via_json(self):
        p = StoryPrediction(
            story_id="t_001:0",
            thread_id="t_001",
            expected_scores={"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
            expected_tier="good",
            expected_themes=["scripture"],
            predicted_scores={"simple": 3, "concrete": 3, "personal": 4, "dynamic": 2},
            predicted_tier="fair",
            predicted_themes=["scripture", "church"],
            scores_raw="SIMPLE: 3 ...",
            themes_raw="scripture, church",
            duration_seconds=1.5,
            error=None,
        )
        d = p.to_dict()
        assert d["type"] == "story_prediction"
        restored = StoryPrediction.from_dict(json.loads(json.dumps(d)))
        assert restored.story_id == "t_001:0"
        assert restored.thread_id == "t_001"
        assert restored.expected_scores == {
            "simple": 4,
            "concrete": 3,
            "personal": 5,
            "dynamic": 2,
        }
        assert restored.expected_tier == "good"
        assert restored.expected_themes == ["scripture"]
        assert restored.predicted_scores == {
            "simple": 3,
            "concrete": 3,
            "personal": 4,
            "dynamic": 2,
        }
        assert restored.predicted_tier == "fair"
        assert restored.predicted_themes == ["scripture", "church"]
        assert restored.scores_raw == "SIMPLE: 3 ..."
        assert restored.themes_raw == "scripture, church"
        assert restored.duration_seconds == 1.5
        assert restored.error is None

    def test_defaults_when_keys_missing(self):
        p = StoryPrediction.from_dict({"story_id": "x:0", "thread_id": "x"})
        assert p.expected_scores is None
        assert p.expected_tier is None
        assert p.expected_themes == []
        assert p.predicted_scores is None
        assert p.predicted_tier is None
        assert p.predicted_themes == []
        assert p.scores_raw is None
        assert p.themes_raw is None
        assert p.duration_seconds == 0.0
        assert p.error is None


class TestExtractionPrediction:
    def test_round_trip_via_json(self):
        p = ExtractionPrediction(
            thread_id="t_001",
            golden_stories=[{"story_id": "t_001:0", "title": "A", "text": "aaa"}],
            predicted_stories=[{"title": "A", "text": "aaa"}, {"title": "B", "text": "bbb"}],
            duration_seconds=2.0,
            error=None,
        )
        d = p.to_dict()
        assert d["type"] == "extraction_prediction"
        restored = ExtractionPrediction.from_dict(json.loads(json.dumps(d)))
        assert restored.thread_id == "t_001"
        assert restored.golden_stories == [
            {"story_id": "t_001:0", "title": "A", "text": "aaa"}
        ]
        assert restored.predicted_stories == [
            {"title": "A", "text": "aaa"},
            {"title": "B", "text": "bbb"},
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
            seeded_from="parse_stories",
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
        assert restored.seeded_from == "parse_stories"
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
        assert meta.seeded_from == ""
        assert meta.extraction_system_prompt == ""
        assert meta.quality_system_prompt == ""
        assert meta.theme_system_prompt == ""
