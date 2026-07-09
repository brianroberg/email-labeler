"""Tests for newsletter classifier — parsing functions and data models."""

import json
from unittest.mock import AsyncMock

import pytest

from llm_client import LLMContentError, LLMUnavailableError
from newsletter import (
    NewsletterClassifier,
    NewsletterTier,
    StoryResult,
    aggregate_theme_grades,
    compute_tier,
    is_newsletter,
    parse_quality_scores,
    parse_send_date,
    parse_stories,
    parse_themes,
    write_assessment,
)


class TestParseSendDate:
    # Normalize the email send-date to ISO-8601 UTC (issue #35/#36 enabler).
    def test_rfc2822_header_to_iso_utc(self):
        assert parse_send_date("Mon, 1 Jan 2024 12:00:00 +0000") == "2024-01-01T12:00:00+00:00"

    def test_offset_converted_to_utc(self):
        assert parse_send_date("Mon, 1 Jan 2024 12:00:00 +0500") == "2024-01-01T07:00:00+00:00"

    def test_naive_header_assumed_utc(self):
        assert parse_send_date("1 Jan 2024 12:00:00") == "2024-01-01T12:00:00+00:00"

    def test_falls_back_to_internal_date_ms(self):
        assert parse_send_date("", "1704067200000") == "2024-01-01T00:00:00+00:00"

    def test_unparseable_header_falls_back_to_internal(self):
        assert parse_send_date("not a date at all", "1704067200000") == "2024-01-01T00:00:00+00:00"

    def test_none_when_nothing_usable(self):
        assert parse_send_date("", None) is None
        assert parse_send_date("garbage", None) is None

    def test_extreme_year_offset_does_not_raise(self):
        # A near-max-year date with a west offset overflows past year 9999 when
        # shifted to UTC; the tz conversion must be guarded so this degrades to a
        # fallback, not an uncaught OverflowError.
        assert parse_send_date("Fri, 31 Dec 9999 23:59:59 -1200", None) is None
        # ...and falls back to internalDate when available.
        assert parse_send_date(
            "Fri, 31 Dec 9999 23:59:59 -1200", "1704067200000"
        ) == "2024-01-01T00:00:00+00:00"


class TestParseStories:
    def test_single_story(self):
        raw = "STORY: Sarah came to campus as a freshman..."
        stories = parse_stories(raw)
        assert len(stories) == 1
        assert "Sarah came to campus" in stories[0]

    def test_multiple_stories(self):
        raw = (
            "STORY: Content of first story.\n"
            "\n"
            "STORY: Content of second story."
        )
        stories = parse_stories(raw)
        assert len(stories) == 2
        assert stories[0] == "Content of first story."
        assert stories[1] == "Content of second story."

    def test_no_stories(self):
        stories = parse_stories("NO_STORIES")
        assert stories == []

    def test_no_stories_with_whitespace(self):
        stories = parse_stories("  NO_STORIES  ")
        assert stories == []

    def test_multiline_story_text(self):
        raw = (
            "STORY: First paragraph of the story.\n"
            "Second paragraph continues here.\n"
            "Third paragraph wraps up."
        )
        stories = parse_stories(raw)
        assert len(stories) == 1
        assert "First paragraph" in stories[0]
        assert "Third paragraph" in stories[0]

    def test_empty_input(self):
        stories = parse_stories("")
        assert stories == []

    def test_garbage_input(self):
        stories = parse_stories("This is not formatted correctly at all")
        assert stories == []

    def test_inner_line_starting_with_story_stays_one_story(self):
        # A story whose own body contains a line starting with "STORY:" must not
        # be split — delimiters are only recognized after a blank line.
        raw = (
            "STORY: Line one of the story.\n"
            "STORY: is a prefix that starts this line inside the same story."
        )
        stories = parse_stories(raw)
        assert len(stories) == 1
        assert "prefix that starts this line" in stories[0]

    def test_multi_paragraph_story_stays_one_story(self):
        raw = "STORY: First paragraph.\n\n\nSecond paragraph after a blank line."
        # A blank line inside a single STORY block (no following STORY:) is part
        # of the story, not a delimiter.
        stories = parse_stories(raw)
        assert len(stories) == 1
        assert "Second paragraph" in stories[0]

    def test_preamble_glued_to_first_story_is_not_dropped(self):
        # The model prefaces the list with a line glued to the first STORY: by a
        # single newline (no blank line). The first story must still be parsed,
        # not silently dropped.
        raw = "Here are the stories:\nSTORY: Alpha\n\nSTORY: Bravo"
        stories = parse_stories(raw)
        assert stories == ["Alpha", "Bravo"]

    def test_crlf_separated_stories_are_split(self):
        # Windows-style CRLF line endings must not collapse every story into one.
        raw = "STORY: Alpha\r\n\r\nSTORY: Bravo\r\n\r\nSTORY: Charlie"
        stories = parse_stories(raw)
        assert stories == ["Alpha", "Bravo", "Charlie"]


class TestParseQualityScores:
    # Poor/OK/Good tokens map to 1/2/3 (issue #53).
    def test_valid_tokens(self):
        raw = "SIMPLE: GOOD\nCONCRETE: OK\nPERSONAL: GOOD\nDYNAMIC: POOR"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1}

    def test_tokens_with_extra_whitespace(self):
        raw = "  SIMPLE:  GOOD \n CONCRETE: OK\n  PERSONAL:GOOD\n DYNAMIC : POOR  "
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1}

    def test_case_insensitive(self):
        raw = "simple: good\nconcrete: ok\npersonal: Good\ndynamic: poor"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1}

    def test_tokens_with_trailing_text(self):
        raw = "SIMPLE: GOOD - very focused\nCONCRETE: OK\nPERSONAL: GOOD\nDYNAMIC: POOR"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1}

    def test_missing_dimension_returns_none(self):
        raw = "SIMPLE: GOOD\nCONCRETE: OK\nPERSONAL: GOOD"
        scores = parse_quality_scores(raw)
        assert scores is None

    def test_unknown_token_returns_none(self):
        raw = "SIMPLE: GREAT\nCONCRETE: OK\nPERSONAL: GOOD\nDYNAMIC: POOR"
        scores = parse_quality_scores(raw)
        assert scores is None

    def test_legacy_digit_value_returns_none(self):
        # The old 1-5 numeric format is no longer accepted.
        raw = "SIMPLE: 4\nCONCRETE: 3\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores is None

    def test_empty_input_returns_none(self):
        assert parse_quality_scores("") is None

    def test_scores_after_preamble(self):
        raw = "Here are my scores:\n\nSIMPLE: GOOD\nCONCRETE: OK\nPERSONAL: GOOD\nDYNAMIC: POOR"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1}


class TestParseThemes:
    # Themes are graded Absent/Present/Emphasized (issue #53); parse returns a
    # dict theme->grade with Absent omitted.
    def test_single_present_theme(self):
        assert parse_themes("SCRIPTURE: PRESENT") == {"scripture": "present"}

    def test_emphasized_theme(self):
        assert parse_themes("SCRIPTURE: EMPHASIZED") == {"scripture": "emphasized"}

    def test_multiple_graded_themes(self):
        raw = "SCRIPTURE: EMPHASIZED\nCHURCH: PRESENT"
        assert parse_themes(raw) == {"scripture": "emphasized", "church": "present"}

    def test_absent_is_omitted(self):
        raw = "SCRIPTURE: ABSENT\nCHURCH: PRESENT"
        assert parse_themes(raw) == {"church": "present"}

    def test_all_absent_returns_empty(self):
        raw = (
            "SCRIPTURE: ABSENT\nCHRISTLIKENESS: ABSENT\nCHURCH: ABSENT\n"
            "VOCATION_FAMILY: ABSENT\nDISCIPLE_MAKING: ABSENT"
        )
        assert parse_themes(raw) == {}

    def test_none_response(self):
        assert parse_themes("NONE") == {}

    def test_none_with_whitespace(self):
        assert parse_themes("  NONE  ") == {}

    def test_ignores_invalid_theme_names(self):
        raw = "SCRIPTURE: PRESENT\nINVALID_THEME: EMPHASIZED\nCHURCH: PRESENT"
        assert parse_themes(raw) == {"scripture": "present", "church": "present"}

    def test_ignores_unparseable_lines(self):
        raw = "SCRIPTURE: PRESENT\nsome commentary\nCHURCH: EMPHASIZED"
        assert parse_themes(raw) == {"scripture": "present", "church": "emphasized"}

    def test_legacy_bare_theme_name_is_ignored(self):
        # The old bare-name (present-only) format no longer conveys a grade.
        assert parse_themes("SCRIPTURE\nCHURCH") == {}

    def test_empty_input(self):
        assert parse_themes("") == {}

    def test_with_extra_whitespace(self):
        raw = "  SCRIPTURE : PRESENT \n  CHURCH:EMPHASIZED  "
        assert parse_themes(raw) == {"scripture": "present", "church": "emphasized"}


class TestComputeTier:
    # Poor/OK/Good -> 1/2/3; tier = mean of the 4 dimensions, banded (issue #53):
    #   excellent >= 2.75, good >= 2.25, fair >= 1.75, poor < 1.75.
    def test_excellent_all_good(self):
        scores = {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}
        assert compute_tier(scores) == NewsletterTier.EXCELLENT

    def test_excellent_boundary_three_good_one_ok(self):
        # avg 2.75
        scores = {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 2}
        assert compute_tier(scores) == NewsletterTier.EXCELLENT

    def test_good_two_good_two_ok(self):
        # avg 2.5
        assert compute_tier({"simple": 3, "concrete": 3, "personal": 2, "dynamic": 2}) == NewsletterTier.GOOD

    def test_good_tolerates_one_poor(self):
        # 3,3,3,1 -> avg 2.5
        assert compute_tier({"simple": 3, "concrete": 3, "personal": 3, "dynamic": 1}) == NewsletterTier.GOOD

    def test_good_boundary(self):
        # 3,3,2,1 -> avg 2.25
        assert compute_tier({"simple": 3, "concrete": 3, "personal": 2, "dynamic": 1}) == NewsletterTier.GOOD

    def test_fair_all_ok(self):
        # avg 2.0
        assert compute_tier({"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2}) == NewsletterTier.FAIR

    def test_fair_boundary(self):
        # 2,2,2,1 -> avg 1.75
        assert compute_tier({"simple": 2, "concrete": 2, "personal": 2, "dynamic": 1}) == NewsletterTier.FAIR

    def test_poor_two_ok_two_poor(self):
        # 2,2,1,1 -> avg 1.5
        assert compute_tier({"simple": 2, "concrete": 2, "personal": 1, "dynamic": 1}) == NewsletterTier.POOR

    def test_poor_all_poor(self):
        assert compute_tier({"simple": 1, "concrete": 1, "personal": 1, "dynamic": 1}) == NewsletterTier.POOR


class TestAggregateThemeGrades:
    # Cross-story theme merge takes the strongest grade per theme (issue #53).
    def test_empty(self):
        assert aggregate_theme_grades([]) == {}

    def test_single_story(self):
        s = StoryResult(text="a", themes={"scripture": "present", "church": "emphasized"})
        assert aggregate_theme_grades([s]) == {"scripture": "present", "church": "emphasized"}

    def test_takes_strongest_grade_across_stories(self):
        s1 = StoryResult(text="a", themes={"scripture": "present"})
        s2 = StoryResult(text="b", themes={"scripture": "emphasized", "church": "present"})
        assert aggregate_theme_grades([s1, s2]) == {"scripture": "emphasized", "church": "present"}

    def test_emphasized_not_downgraded_by_later_present(self):
        s1 = StoryResult(text="a", themes={"scripture": "emphasized"})
        s2 = StoryResult(text="b", themes={"scripture": "present"})
        assert aggregate_theme_grades([s1, s2]) == {"scripture": "emphasized"}


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
                    "user_template": "Story text:\n{text}",
                },
                "theme_classification": {
                    "system": "Classify themes.",
                    "user_template": "Story text:\n{text}",
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
            "STORY: Once upon a time...",
            "thinking about stories",
        )
        stories = await nl_classifier.extract_stories("newsletter body text")
        assert len(stories) == 1
        assert stories[0] == "Once upon a time..."
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
            "SIMPLE: GOOD\nCONCRETE: OK\nPERSONAL: GOOD\nDYNAMIC: POOR",
            "quality reasoning",
        )
        scores, cot = await nl_classifier.assess_quality("Story text")
        assert scores == {"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1}
        assert cot == "quality reasoning"

    async def test_returns_none_on_parse_failure(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("garbled output", "")
        scores, cot = await nl_classifier.assess_quality("Story text")
        assert scores is None

    async def test_passes_text(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = (
            "SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK",
            "",
        )
        await nl_classifier.assess_quality("My story text")
        user_content = mock_cloud_llm.complete.call_args.args[1]
        assert "My story text" in user_content


class TestClassifyThemes:
    async def test_classifies_themes(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = (
            "SCRIPTURE: EMPHASIZED\nCHURCH: PRESENT",
            "theme reasoning",
        )
        themes, cot = await nl_classifier.classify_themes("Story text")
        assert themes == {"scripture": "emphasized", "church": "present"}
        assert cot == "theme reasoning"

    async def test_returns_empty_for_none(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.return_value = ("NONE", "")
        themes, cot = await nl_classifier.classify_themes("Story text")
        assert themes == {}


class TestClassifyNewsletter:
    async def test_full_pipeline(self, nl_classifier, mock_cloud_llm):
        """Full pipeline: extract 1 story, score it, tag themes."""
        mock_cloud_llm.complete.side_effect = [
            ("STORY: A student named Jake...", ""),
            ("SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: GOOD\nDYNAMIC: OK", "quality cot"),
            ("CHRISTLIKENESS: PRESENT\nDISCIPLE_MAKING: EMPHASIZED", "theme cot"),
        ]
        results = await nl_classifier.classify_newsletter("newsletter body")
        assert len(results) == 1
        assert results[0].text == "A student named Jake..."
        assert results[0].scores == {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 2}
        assert results[0].average_score == 2.75
        assert results[0].tier == NewsletterTier.EXCELLENT
        assert results[0].themes == {"christlikeness": "present", "disciple_making": "emphasized"}
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
            ("STORY: Content here", ""),
            ("garbled quality output", ""),
            ("SCRIPTURE: PRESENT", "theme cot"),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 1
        assert results[0].scores is None
        assert results[0].tier is None
        assert results[0].themes == {"scripture": "present"}

    async def test_theme_failure_preserves_quality(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content", ""),
            ("SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK", "quality cot"),
            RuntimeError("LLM error"),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 1
        assert results[0].scores is not None
        assert results[0].themes == {}

    async def test_multiple_stories(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content A\n\nSTORY: Content B", ""),
            ("SIMPLE: GOOD\nCONCRETE: GOOD\nPERSONAL: GOOD\nDYNAMIC: GOOD", ""),
            ("SCRIPTURE: PRESENT", ""),
            ("SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK", ""),
            ("CHURCH: PRESENT", ""),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 2
        assert results[0].tier == NewsletterTier.EXCELLENT
        assert results[1].tier == NewsletterTier.FAIR
        assert mock_cloud_llm.complete.call_count == 5


class TestClassifyNewsletterTransientOutage:
    """Issue #18: a transient LLM outage mid-newsletter must propagate so the daemon
    retries the whole thread — not be swallowed into a permanently mis-graded result.
    Genuinely per-story/permanent failures stay isolated as before."""

    async def test_transient_quality_outage_propagates(self, nl_classifier, mock_cloud_llm):
        """LLMUnavailableError during quality assessment propagates (not swallowed)."""
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content", ""),                      # extract_stories
            LLMUnavailableError("cloud endpoint down"),  # assess_quality
        ]
        with pytest.raises(LLMUnavailableError):
            await nl_classifier.classify_newsletter("body")

    async def test_transient_theme_outage_propagates(self, nl_classifier, mock_cloud_llm):
        """LLMUnavailableError during theme classification propagates too."""
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content", ""),                                        # extract_stories
            ("SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK", ""),    # assess_quality
            LLMUnavailableError("cloud endpoint down"),                    # classify_themes
        ]
        with pytest.raises(LLMUnavailableError):
            await nl_classifier.classify_newsletter("body")

    async def test_content_error_during_quality_propagates(self, nl_classifier, mock_cloud_llm):
        """A content-less LLMContentError during quality propagates (issue #30) so the
        daemon routes the whole newsletter to give-up rather than committing an empty
        grade indistinguishable from a genuine NO_STORIES."""
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content", ""),                       # extract_stories
            LLMContentError("model returned no content"),  # assess_quality
        ]
        with pytest.raises(LLMContentError):
            await nl_classifier.classify_newsletter("body")

    async def test_content_error_during_theme_propagates(self, nl_classifier, mock_cloud_llm):
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content", ""),                                        # extract_stories
            ("SIMPLE: OK\nCONCRETE: OK\nPERSONAL: OK\nDYNAMIC: OK", ""),    # assess_quality
            LLMContentError("model returned no content"),                  # classify_themes
        ]
        with pytest.raises(LLMContentError):
            await nl_classifier.classify_newsletter("body")

    async def test_non_transient_quality_error_stays_isolated(self, nl_classifier, mock_cloud_llm):
        """A non-transient per-story error (bare RuntimeError) is still swallowed: the
        story gets no scores but themes are still classified and the result is returned.

        (Issue #30 will add a dedicated LLMContentError subclass that DOES propagate; a
        bare RuntimeError like this one must remain isolated.)"""
        mock_cloud_llm.complete.side_effect = [
            ("STORY: Content", ""),  # extract_stories
            RuntimeError("malformed story crashed scoring"),  # assess_quality
            ("SCRIPTURE: PRESENT", "theme cot"),  # classify_themes still runs
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 1
        assert results[0].scores is None
        assert results[0].themes == {"scripture": "present"}


class TestWriteAssessment:
    def test_writes_jsonl_record(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        stories = [
            StoryResult(
                text="Story content",
                scores={"simple": 3, "concrete": 2, "personal": 3, "dynamic": 1},
                average_score=2.25,
                tier=NewsletterTier.GOOD,
                themes={"scripture": "present", "church": "emphasized"},
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
        assert "title" not in record["stories"][0]
        assert record["stories"][0]["text"] == "Story content"
        assert record["stories"][0]["scores"]["simple"] == 3
        assert record["stories"][0]["themes"] == {"scripture": "present", "church": "emphasized"}
        assert record["stories"][0]["quality_cot"] == "quality reasoning"
        assert "timestamp" in record

    def test_writes_send_date_and_model(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        write_assessment(
            output_file=str(output_file),
            message_id="m", thread_id="t", sender="a@b.com", subject="Subj",
            overall_tier=NewsletterTier.GOOD, stories=[StoryResult(text="x")],
            send_date="2024-01-01T12:00:00+00:00", model="claude-sonnet-4-6",
        )
        record = json.loads(output_file.read_text().strip())
        assert record["send_date"] == "2024-01-01T12:00:00+00:00"
        assert record["model"] == "claude-sonnet-4-6"

    def test_send_date_and_model_default_to_none(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        write_assessment(
            output_file=str(output_file),
            message_id="m", thread_id="t", sender="a@b.com", subject="Subj",
            overall_tier=None, stories=[StoryResult(text="x")],
        )
        record = json.loads(output_file.read_text().strip())
        assert record["send_date"] is None
        assert record["model"] is None

    def test_appends_to_existing_file(self, tmp_path):
        output_file = tmp_path / "assessments.jsonl"
        story = StoryResult(text="T", tier=NewsletterTier.FAIR)
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
        story = StoryResult(text="T", tier=NewsletterTier.POOR)
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
            text="Content",
            scores=None,
            average_score=None,
            tier=None,
            themes={"scripture": "present"},
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


class TestIsNewsletter:
    def test_detects_to_header(self):
        messages = [
            {
                "payload": {
                    "headers": [
                        {"name": "To", "value": "newsletters@dm.org"},
                        {"name": "From", "value": "john@dm.org"},
                    ]
                }
            }
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_detects_in_cc(self):
        messages = [
            {
                "payload": {
                    "headers": [
                        {"name": "To", "value": "someone@dm.org"},
                        {"name": "Cc", "value": "newsletters@dm.org"},
                    ]
                }
            }
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_not_newsletter(self):
        messages = [
            {
                "payload": {
                    "headers": [
                        {"name": "To", "value": "other@dm.org"},
                        {"name": "From", "value": "john@dm.org"},
                    ]
                }
            }
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is False

    def test_case_insensitive(self):
        messages = [
            {
                "payload": {
                    "headers": [
                        {"name": "To", "value": "Newsletters@DM.org"},
                    ]
                }
            }
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_multiple_recipients(self):
        messages = [
            {
                "payload": {
                    "headers": [
                        {"name": "To", "value": "someone@dm.org, newsletters@dm.org, other@dm.org"},
                    ]
                }
            }
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_checks_all_messages_in_thread(self):
        messages = [
            {"payload": {"headers": [{"name": "To", "value": "other@dm.org"}]}},
            {"payload": {"headers": [{"name": "To", "value": "newsletters@dm.org"}]}},
        ]
        assert is_newsletter(messages, "newsletters@dm.org") is True

    def test_missing_to_header(self):
        messages = [{"payload": {"headers": [{"name": "From", "value": "john@dm.org"}]}}]
        assert is_newsletter(messages, "newsletters@dm.org") is False

    def test_empty_messages(self):
        assert is_newsletter([], "newsletters@dm.org") is False
