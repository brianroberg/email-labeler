"""Tests for newsletter classifier — parsing functions and data models."""

import pytest

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
