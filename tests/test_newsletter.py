"""Tests for newsletter classifier — parsing functions and data models."""

import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

from newsletter import (
    NewsletterTier,
    StoryResult,
    parse_stories,
    parse_quality_scores,
    parse_themes,
    compute_tier,
)


class TestParseStories:
    def test_single_story(self):
        raw = "TITLE: Sarah's Journey\nTEXT: Sarah came to campus as a freshman..."
        stories = parse_stories(raw)
        assert len(stories) == 1
        assert stories[0][0] == "Sarah's Journey"
        assert "Sarah came to campus" in stories[0][1]

    def test_multiple_stories(self):
        raw = (
            "TITLE: First Story\n"
            "TEXT: Content of first story.\n"
            "\n"
            "TITLE: Second Story\n"
            "TEXT: Content of second story."
        )
        stories = parse_stories(raw)
        assert len(stories) == 2
        assert stories[0][0] == "First Story"
        assert stories[1][0] == "Second Story"

    def test_no_stories(self):
        stories = parse_stories("NO_STORIES")
        assert stories == []

    def test_no_stories_with_whitespace(self):
        stories = parse_stories("  NO_STORIES  ")
        assert stories == []

    def test_multiline_story_text(self):
        raw = (
            "TITLE: A Long Story\n"
            "TEXT: First paragraph of the story.\n"
            "Second paragraph continues here.\n"
            "Third paragraph wraps up."
        )
        stories = parse_stories(raw)
        assert len(stories) == 1
        assert "First paragraph" in stories[0][1]
        assert "Third paragraph" in stories[0][1]

    def test_empty_input(self):
        stories = parse_stories("")
        assert stories == []

    def test_garbage_input(self):
        stories = parse_stories("This is not formatted correctly at all")
        assert stories == []


class TestParseQualityScores:
    def test_valid_scores(self):
        raw = "SIMPLE: 4\nCONCRETE: 3\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}

    def test_scores_with_extra_whitespace(self):
        raw = "  SIMPLE:  4 \n CONCRETE: 3\n  PERSONAL:5\n DYNAMIC :2  "
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}

    def test_scores_with_trailing_text(self):
        raw = "SIMPLE: 4 - very focused\nCONCRETE: 3\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}

    def test_missing_dimension_returns_none(self):
        raw = "SIMPLE: 4\nCONCRETE: 3\nPERSONAL: 5"
        scores = parse_quality_scores(raw)
        assert scores is None

    def test_invalid_score_returns_none(self):
        raw = "SIMPLE: 4\nCONCRETE: six\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores is None

    def test_score_out_of_range_clamped(self):
        raw = "SIMPLE: 7\nCONCRETE: 0\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 5, "concrete": 1, "personal": 5, "dynamic": 2}

    def test_empty_input_returns_none(self):
        assert parse_quality_scores("") is None

    def test_last_line_fallback(self):
        raw = "Here are my scores:\n\nSIMPLE: 4\nCONCRETE: 3\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}


class TestParseThemes:
    def test_single_theme(self):
        themes = parse_themes("SCRIPTURE")
        assert themes == ["scripture"]

    def test_multiple_themes(self):
        themes = parse_themes("SCRIPTURE\nCHRISTLIKENESS")
        assert themes == ["scripture", "christlikeness"]

    def test_all_themes(self):
        raw = "SCRIPTURE\nCHRISTLIKENESS\nCHURCH\nVOCATION_FAMILY\nDISCIPLE_MAKING"
        themes = parse_themes(raw)
        assert len(themes) == 5

    def test_none_response(self):
        themes = parse_themes("NONE")
        assert themes == []

    def test_none_with_whitespace(self):
        themes = parse_themes("  NONE  ")
        assert themes == []

    def test_ignores_invalid_themes(self):
        themes = parse_themes("SCRIPTURE\nINVALID_THEME\nCHURCH")
        assert themes == ["scripture", "church"]

    def test_empty_input(self):
        themes = parse_themes("")
        assert themes == []

    def test_with_extra_whitespace(self):
        themes = parse_themes("  SCRIPTURE \n  CHURCH  ")
        assert themes == ["scripture", "church"]


class TestComputeTier:
    def test_excellent(self):
        assert compute_tier({"simple": 5, "concrete": 4, "personal": 4, "dynamic": 5}) == NewsletterTier.EXCELLENT

    def test_good(self):
        assert compute_tier({"simple": 3, "concrete": 3, "personal": 4, "dynamic": 3}) == NewsletterTier.GOOD

    def test_fair(self):
        assert compute_tier({"simple": 2, "concrete": 2, "personal": 3, "dynamic": 2}) == NewsletterTier.FAIR

    def test_poor(self):
        assert compute_tier({"simple": 1, "concrete": 1, "personal": 2, "dynamic": 1}) == NewsletterTier.POOR

    def test_boundary_excellent(self):
        assert compute_tier({"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}) == NewsletterTier.EXCELLENT

    def test_boundary_good(self):
        assert compute_tier({"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}) == NewsletterTier.GOOD

    def test_boundary_fair(self):
        assert compute_tier({"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2}) == NewsletterTier.FAIR


from newsletter import NewsletterClassifier


@pytest.fixture
def mock_cloud_llm():
    return AsyncMock()


@pytest.fixture
def newsletter_config():
    return {
        "newsletter": {
            "recipient": "newsletters@dm.org",
            "output_file": "data/newsletter_assessments.jsonl",
            "labels": {},
            "prompts": {
                "story_extraction": {
                    "system": "Extract stories.",
                    "user_template": "Newsletter content:\n{body}",
                },
                "quality_assessment": {
                    "system": "Score the story.",
                    "user_template": "Story title: {title}\nStory text:\n{text}",
                },
                "theme_classification": {
                    "system": "Classify themes.",
                    "user_template": "Story title: {title}\nStory text:\n{text}",
                },
            },
        }
    }


@pytest.fixture
def nl_classifier(mock_cloud_llm, newsletter_config):
    return NewsletterClassifier(cloud_llm=mock_cloud_llm, config=newsletter_config)


class TestExtractStories:
    async def test_extracts_stories(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = (
            "TITLE: A Great Story\nTEXT: Once upon a time...",
            "thinking about stories",
        )
        stories = await nl_classifier.extract_stories("newsletter body text")
        assert len(stories) == 1
        assert stories[0][0] == "A Great Story"
        mock_cloud_llm.complete.assert_called_once()

    async def test_returns_empty_for_no_stories(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("NO_STORIES", "")
        stories = await nl_classifier.extract_stories("administrative newsletter")
        assert stories == []

    async def test_passes_body_to_prompt(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("NO_STORIES", "")
        await nl_classifier.extract_stories("the newsletter body")
        user_content = mock_cloud_llm.complete.call_args.args[1]
        assert "the newsletter body" in user_content


class TestAssessQuality:
    async def test_scores_story(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = (
            "SIMPLE: 4\nCONCRETE: 3\nPERSONAL: 5\nDYNAMIC: 2",
            "quality reasoning",
        )
        scores, cot = await nl_classifier.assess_quality("Title", "Story text")
        assert scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}
        assert cot == "quality reasoning"

    async def test_returns_none_on_parse_failure(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("garbled output", "")
        scores, cot = await nl_classifier.assess_quality("Title", "Story text")
        assert scores is None

    async def test_passes_title_and_text(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = (
            "SIMPLE: 3\nCONCRETE: 3\nPERSONAL: 3\nDYNAMIC: 3", "",
        )
        await nl_classifier.assess_quality("My Title", "My story text")
        user_content = mock_cloud_llm.complete.call_args.args[1]
        assert "My Title" in user_content
        assert "My story text" in user_content


class TestClassifyThemes:
    async def test_classifies_themes(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("SCRIPTURE\nCHURCH", "theme reasoning")
        themes, cot = await nl_classifier.classify_themes("Title", "Story text")
        assert themes == ["scripture", "church"]
        assert cot == "theme reasoning"

    async def test_returns_empty_for_none(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("NONE", "")
        themes, cot = await nl_classifier.classify_themes("Title", "Story text")
        assert themes == []


class TestClassifyNewsletter:
    async def test_full_pipeline(self, nl_classifier, mock_cloud_llm):
        """Full pipeline: extract 1 story, score it, tag themes."""
        mock_cloud_llm.complete.side_effect = [
            ("TITLE: Test Story\nTEXT: A student named Jake...", ""),
            ("SIMPLE: 4\nCONCRETE: 5\nPERSONAL: 4\nDYNAMIC: 3", "quality cot"),
            ("CHRISTLIKENESS\nDISCIPLE_MAKING", "theme cot"),
        ]
        results = await nl_classifier.classify_newsletter("newsletter body")
        assert len(results) == 1
        assert results[0].title == "Test Story"
        assert results[0].scores == {"simple": 4, "concrete": 5, "personal": 4, "dynamic": 3}
        assert results[0].average_score == 4.0
        assert results[0].tier == NewsletterTier.EXCELLENT
        assert results[0].themes == ["christlikeness", "disciple_making"]
        assert results[0].quality_cot == "quality cot"
        assert results[0].theme_cot == "theme cot"
        assert mock_cloud_llm.complete.call_count == 3

    async def test_no_stories_returns_empty(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("NO_STORIES", "")
        results = await nl_classifier.classify_newsletter("admin newsletter")
        assert results == []
        assert mock_cloud_llm.complete.call_count == 1

    async def test_quality_failure_still_classifies_themes(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.side_effect = [
            ("TITLE: Story\nTEXT: Content here", ""),
            ("garbled quality output", ""),
            ("SCRIPTURE", "theme cot"),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 1
        assert results[0].scores is None
        assert results[0].tier is None
        assert results[0].themes == ["scripture"]

    async def test_theme_failure_preserves_quality(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.side_effect = [
            ("TITLE: Story\nTEXT: Content", ""),
            ("SIMPLE: 3\nCONCRETE: 3\nPERSONAL: 3\nDYNAMIC: 3", "quality cot"),
            RuntimeError("LLM error"),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 1
        assert results[0].scores is not None
        assert results[0].themes == []

    async def test_multiple_stories(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.side_effect = [
            ("TITLE: Story A\nTEXT: Content A\n\nTITLE: Story B\nTEXT: Content B", ""),
            ("SIMPLE: 5\nCONCRETE: 5\nPERSONAL: 5\nDYNAMIC: 5", ""),
            ("SCRIPTURE", ""),
            ("SIMPLE: 2\nCONCRETE: 2\nPERSONAL: 2\nDYNAMIC: 2", ""),
            ("CHURCH", ""),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 2
        assert results[0].tier == NewsletterTier.EXCELLENT
        assert results[1].tier == NewsletterTier.FAIR
        assert mock_cloud_llm.complete.call_count == 5


from newsletter import write_assessment


class TestWriteAssessment:
    def test_writes_jsonl_record(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        stories = [
            StoryResult(
                title="Test Story",
                text="Story content",
                scores={"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2},
                average_score=3.5,
                tier=NewsletterTier.GOOD,
                themes=["scripture", "church"],
                quality_cot="quality reasoning",
                theme_cot="theme reasoning",
            )
        ]
        write_assessment(
            output_file=str(output_file),
            message_id="msg_001",
            thread_id="thread_001",
            sender="john@dm.org",
            subject="February Update",
            overall_tier=NewsletterTier.GOOD,
            stories=stories,
        )

        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["message_id"] == "msg_001"
        assert record["thread_id"] == "thread_001"
        assert record["from"] == "john@dm.org"
        assert record["subject"] == "February Update"
        assert record["overall_tier"] == "good"
        assert len(record["stories"]) == 1
        assert record["stories"][0]["title"] == "Test Story"
        assert record["stories"][0]["scores"]["simple"] == 4
        assert record["stories"][0]["themes"] == ["scripture", "church"]
        assert record["stories"][0]["quality_cot"] == "quality reasoning"
        assert "timestamp" in record

    def test_appends_to_existing_file(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        story = StoryResult(title="S", text="T", tier=NewsletterTier.FAIR)
        for i in range(3):
            write_assessment(
                output_file=str(output_file),
                message_id=f"msg_{i}",
                thread_id=f"thread_{i}",
                sender="a@b.com",
                subject="Subj",
                overall_tier=NewsletterTier.FAIR,
                stories=[story],
            )
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_creates_parent_directories(self, tmp_path):
        output_file = tmp_path / "sub" / "dir" / "assessments.jsonl"
        story = StoryResult(title="S", text="T", tier=NewsletterTier.POOR)
        write_assessment(
            output_file=str(output_file),
            message_id="msg_001",
            thread_id="t_001",
            sender="a@b.com",
            subject="Subj",
            overall_tier=NewsletterTier.POOR,
            stories=[story],
        )
        assert output_file.exists()

    def test_story_without_scores(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        story = StoryResult(
            title="No Scores",
            text="Content",
            scores=None,
            average_score=None,
            tier=None,
            themes=["scripture"],
        )
        write_assessment(
            output_file=str(output_file),
            message_id="msg_001",
            thread_id="t_001",
            sender="a@b.com",
            subject="Subj",
            overall_tier=None,
            stories=[story],
        )
        record = json.loads(output_file.read_text().strip())
        assert record["stories"][0]["scores"] is None
        assert record["stories"][0]["tier"] is None
        assert record["overall_tier"] is None


from newsletter import is_newsletter


class TestIsNewsletter:
    def test_detects_to_header(self):
        messages = [
            {"payload": {"headers": [
                {"name": "To", "value": "newsletters@dm.org"},
                {"name": "From", "value": "john@dm.org"},
            ]}}
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_detects_in_cc(self):
        messages = [
            {"payload": {"headers": [
                {"name": "To", "value": "someone@dm.org"},
                {"name": "Cc", "value": "newsletters@dm.org"},
            ]}}
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_not_newsletter(self):
        messages = [
            {"payload": {"headers": [
                {"name": "To", "value": "other@dm.org"},
                {"name": "From", "value": "john@dm.org"},
            ]}}
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is False

    def test_case_insensitive(self):
        messages = [
            {"payload": {"headers": [
                {"name": "To", "value": "Newsletters@DM.org"},
            ]}}
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_multiple_recipients(self):
        messages = [
            {"payload": {"headers": [
                {"name": "To", "value": "someone@dm.org, newsletters@dm.org, other@dm.org"},
            ]}}
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_checks_all_messages_in_thread(self):
        messages = [
            {"payload": {"headers": [{"name": "To", "value": "other@dm.org"}]}},
            {"payload": {"headers": [{"name": "To", "value": "newsletters@dm.org"}]}},
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_missing_to_header(self):
        messages = [
            {"payload": {"headers": [{"name": "From", "value": "john@dm.org"}]}}
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is False

    def test_empty_messages(self):
        assert is_newsletter([], "newsletters@dm.org") is False
