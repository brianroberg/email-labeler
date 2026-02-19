# Newsletter Story Classification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a parallel classification pipeline that detects newsletter emails (sent to `newsletters@dm.org`), extracts individual stories, scores each story on a 4-dimension quality rubric, and tags each story with Ends Statement themes.

**Architecture:** Branch-in-loop — the daemon's existing `process_single_thread()` gains an early detection step that checks the `To:` header. Newsletter emails skip the existing priority classifier and instead route to a new `NewsletterClassifier` in `newsletter.py`. Results are stored as Gmail labels (quality tier + themes) and as detailed JSONL records.

**Tech Stack:** Python 3.14+, asyncio, httpx, existing LLMClient and GmailProxyClient, JSONL for structured output.

---

### Task 1: Add newsletter config to config.toml

**Files:**
- Modify: `config.toml` (after line 88, end of file)

**Step 1: Add the newsletter configuration section**

Append to `config.toml`:

```toml
[newsletter]
recipient = "newsletters@dm.org"
output_file = "data/newsletter_assessments.jsonl"

[newsletter.labels]
newsletter = "agent/newsletter"
excellent = "agent/newsletter/excellent"
good = "agent/newsletter/good"
fair = "agent/newsletter/fair"
poor = "agent/newsletter/poor"
no_stories = "agent/newsletter/no-stories"

[newsletter.labels.themes]
scripture = "agent/newsletter/theme/scripture"
christlikeness = "agent/newsletter/theme/christlikeness"
church = "agent/newsletter/theme/church"
vocation_family = "agent/newsletter/theme/vocation-family"
disciple_making = "agent/newsletter/theme/disciple-making"

[newsletter.prompts.story_extraction]
system = """You are a newsletter analyst. Given a ministry newsletter email, extract each distinct story from the content.

A "story" is a narrative segment about people, events, or ministry activities. Skip non-story content like:
- Headers, footers, signatures
- Donation appeals or fundraising asks
- Event calendar listings
- Administrative announcements
- Contact information

For each story, provide:
1. A short title (5-10 words)
2. The full text of the story as it appears in the newsletter

Respond in this exact format (one story per block, separated by blank lines):

TITLE: [short title]
TEXT: [full story text]

If the newsletter contains no stories, respond with exactly: NO_STORIES"""

user_template = """Newsletter content:
{body}"""

[newsletter.prompts.quality_assessment]
system = """You are a writing quality assessor for ministry newsletter stories. Score the following story on four dimensions, each on a 1-5 scale.

Dimensions:
- SIMPLE (1-5): Does the story focus on one key idea or progression? A 5 means no tangents or extraneous details. A 1 means it tries to cover too many things and feels scattered.
- CONCRETE (1-5): Does the story narrate particular events involving particular people at particular times and places? A 5 means vivid specifics. A 1 means abstract and generalized.
- PERSONAL (1-5): Does the story center around one person or a few people, such that what's important to them is what's important to the story? A 5 means deeply personal. A 1 means it's mainly about events, numbers, or ideas disconnected from people.
- DYNAMIC (1-5): Does the story describe how a person changes over time? A 5 means a clear arc of transformation. A 1 means static — no change, no before-and-after.

Think step by step about each dimension. Use <think> tags for your reasoning, then respond in this exact format:

SIMPLE: [1-5]
CONCRETE: [1-5]
PERSONAL: [1-5]
DYNAMIC: [1-5]"""

user_template = """Story title: {title}
Story text:
{text}"""

[newsletter.prompts.theme_classification]
system = """You are a ministry theme classifier. Given a story from a ministry newsletter, identify which themes from the organization's Ends Statement it illustrates.

The Ends Statement themes are:

- SCRIPTURE: Study the Scriptures correctly, apply them to all of life, and teach them to others
- CHRISTLIKENESS: Exhibit increasing Christlikeness in response to the Gospel
- CHURCH: Serve not only in their campus fellowship but also in a Bible-believing local church
- VOCATION_FAMILY: Honor God in their vocation and family relationships
- DISCIPLE_MAKING: Continue to make disciples wherever God takes them

A story may match multiple themes, or none at all. Only select themes that the story clearly illustrates — do not force a match.

Think step by step. Use <think> tags for your reasoning, then respond with ONLY the matching theme names, one per line. If no themes match, respond with: NONE"""

user_template = """Story title: {title}
Story text:
{text}"""
```

**Step 2: Verify config loads correctly**

Run: `cd /workspaces/email-labeler && uv run python -c "import tomllib; c = tomllib.load(open('config.toml','rb')); print(list(c['newsletter'].keys()))"`

Expected: `['recipient', 'output_file', 'labels', 'prompts']`

**Step 3: Update config loading test**

Add to `tests/test_daemon.py` at the end of `TestLoadConfig`:

```python
def test_config_has_newsletter_section(self):
    config = load_config()
    assert "newsletter" in config
    assert "recipient" in config["newsletter"]
    assert "output_file" in config["newsletter"]
    assert "labels" in config["newsletter"]
    assert "prompts" in config["newsletter"]

def test_config_has_newsletter_labels(self):
    config = load_config()
    nl = config["newsletter"]["labels"]
    assert "newsletter" in nl
    assert "excellent" in nl
    assert "good" in nl
    assert "fair" in nl
    assert "poor" in nl
    assert "no_stories" in nl
    assert "themes" in nl
    assert len(nl["themes"]) == 5

def test_config_has_newsletter_prompts(self):
    config = load_config()
    prompts = config["newsletter"]["prompts"]
    assert "story_extraction" in prompts
    assert "quality_assessment" in prompts
    assert "theme_classification" in prompts
    for key in ("story_extraction", "quality_assessment", "theme_classification"):
        assert "system" in prompts[key]
        assert "user_template" in prompts[key]
```

**Step 4: Run tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_daemon.py::TestLoadConfig -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
cd /workspaces/email-labeler
git add config.toml tests/test_daemon.py
git commit -m "Add newsletter classification config and prompts"
```

---

### Task 2: Newsletter data models and parsers

**Files:**
- Create: `newsletter.py`
- Create: `tests/test_newsletter.py`

**Step 1: Write tests for data models and parsers**

Create `tests/test_newsletter.py`:

```python
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


# ── Story extraction parsing ────────────────────────────────────────────


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


# ── Quality score parsing ────────────────────────────────────────────────


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
        """Scores may appear after preamble text, like email label parsing."""
        raw = "Here are my scores:\n\nSIMPLE: 4\nCONCRETE: 3\nPERSONAL: 5\nDYNAMIC: 2"
        scores = parse_quality_scores(raw)
        assert scores == {"simple": 4, "concrete": 3, "personal": 5, "dynamic": 2}


# ── Theme parsing ────────────────────────────────────────────────────────


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


# ── Tier computation ─────────────────────────────────────────────────────


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
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'newsletter'`

**Step 3: Create newsletter.py with data models and parsers**

Create `newsletter.py`:

```python
"""Newsletter story classification pipeline.

Extracts stories from ministry newsletters, scores them on a quality rubric,
and tags them with Ends Statement themes. All LLM calls use the cloud endpoint
(newsletter content is not privacy-sensitive).
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class NewsletterTier(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


_VALID_THEMES = {
    "SCRIPTURE", "CHRISTLIKENESS", "CHURCH",
    "VOCATION_FAMILY", "DISCIPLE_MAKING",
}

_DIMENSIONS = ("simple", "concrete", "personal", "dynamic")


@dataclass
class StoryResult:
    title: str
    text: str
    scores: dict[str, int] | None = None
    average_score: float | None = None
    tier: NewsletterTier | None = None
    themes: list[str] = field(default_factory=list)
    quality_cot: str = ""
    theme_cot: str = ""


def parse_stories(raw: str) -> list[tuple[str, str]]:
    """Parse LLM story extraction output into (title, text) pairs.

    Expected format:
        TITLE: <title>
        TEXT: <story text, possibly multi-line>

    Returns empty list for NO_STORIES or unparseable input.
    """
    stripped = raw.strip()
    if not stripped or stripped.upper() == "NO_STORIES":
        return []

    stories = []
    # Split on TITLE: markers (lookahead to handle multiple stories)
    blocks = re.split(r"(?=^TITLE:)", stripped, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"^TITLE:\s*(.+)\nTEXT:\s*(.*)", block, flags=re.DOTALL)
        if match:
            title = match.group(1).strip()
            text = match.group(2).strip()
            if title and text:
                stories.append((title, text))

    return stories


def parse_quality_scores(raw: str) -> dict[str, int] | None:
    """Parse LLM quality assessment output into dimension scores.

    Expected format (one per line, order doesn't matter):
        SIMPLE: 4
        CONCRETE: 3
        PERSONAL: 5
        DYNAMIC: 2

    Returns None if any dimension is missing or has an unparseable value.
    Clamps scores to 1-5 range.
    """
    if not raw.strip():
        return None

    scores = {}
    for dim in _DIMENSIONS:
        pattern = rf"{dim.upper()}\s*:\s*(\d+)"
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            value = int(match.group(1))
        except ValueError:
            return None
        scores[dim] = max(1, min(5, value))

    return scores


def parse_themes(raw: str) -> list[str]:
    """Parse LLM theme classification output into theme labels.

    Returns list of lowercase theme names. Ignores unrecognized themes.
    Returns empty list for NONE response.
    """
    stripped = raw.strip()
    if not stripped or stripped.upper() == "NONE":
        return []

    themes = []
    for line in stripped.splitlines():
        token = line.strip().upper()
        if token in _VALID_THEMES:
            themes.append(token.lower())

    return themes


def compute_tier(scores: dict[str, int]) -> NewsletterTier:
    """Derive quality tier from dimension scores."""
    avg = sum(scores.values()) / len(scores)
    if avg >= 4.0:
        return NewsletterTier.EXCELLENT
    if avg >= 3.0:
        return NewsletterTier.GOOD
    if avg >= 2.0:
        return NewsletterTier.FAIR
    return NewsletterTier.POOR
```

**Step 4: Run tests to verify they pass**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
cd /workspaces/email-labeler
git add newsletter.py tests/test_newsletter.py
git commit -m "Add newsletter data models and LLM output parsers"
```

---

### Task 3: NewsletterClassifier — story extraction, quality, and theme LLM calls

**Files:**
- Modify: `newsletter.py` (add NewsletterClassifier class)
- Modify: `tests/test_newsletter.py` (add classifier tests)

**Step 1: Write tests for NewsletterClassifier**

Append to `tests/test_newsletter.py`:

```python
from unittest.mock import AsyncMock

from newsletter import NewsletterClassifier, NewsletterTier


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
            # extract_stories
            ("TITLE: Test Story\nTEXT: A student named Jake...", ""),
            # assess_quality
            ("SIMPLE: 4\nCONCRETE: 5\nPERSONAL: 4\nDYNAMIC: 3", "quality cot"),
            # classify_themes
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
        assert mock_cloud_llm.complete.call_count == 1  # only extraction call

    async def test_quality_failure_still_classifies_themes(self, nl_classifier, mock_cloud_llm):
        """If quality assessment fails, themes still run."""
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
        """If theme classification raises, quality scores are preserved."""
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
        """Two stories each get quality + theme calls."""
        mock_cloud_llm.complete.side_effect = [
            # extraction
            ("TITLE: Story A\nTEXT: Content A\n\nTITLE: Story B\nTEXT: Content B", ""),
            # quality for story A
            ("SIMPLE: 5\nCONCRETE: 5\nPERSONAL: 5\nDYNAMIC: 5", ""),
            # themes for story A
            ("SCRIPTURE", ""),
            # quality for story B
            ("SIMPLE: 2\nCONCRETE: 2\nPERSONAL: 2\nDYNAMIC: 2", ""),
            # themes for story B
            ("CHURCH", ""),
        ]
        results = await nl_classifier.classify_newsletter("body")
        assert len(results) == 2
        assert results[0].tier == NewsletterTier.EXCELLENT
        assert results[1].tier == NewsletterTier.FAIR
        assert mock_cloud_llm.complete.call_count == 5  # 1 + 2*2
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py::TestExtractStories -v`

Expected: FAIL — `ImportError: cannot import name 'NewsletterClassifier'`

**Step 3: Implement NewsletterClassifier**

Add to `newsletter.py` (after the existing parser functions):

```python
from llm_client import LLMClient


class NewsletterClassifier:
    """Classifies newsletter stories for quality and themes."""

    def __init__(self, cloud_llm: LLMClient, config: dict):
        self.cloud_llm = cloud_llm
        nl_config = config["newsletter"]
        self.extraction_config = nl_config["prompts"]["story_extraction"]
        self.quality_config = nl_config["prompts"]["quality_assessment"]
        self.theme_config = nl_config["prompts"]["theme_classification"]

    async def extract_stories(self, body: str) -> list[tuple[str, str]]:
        """Extract individual stories from a newsletter body.

        Returns list of (title, text) tuples. Empty list if no stories found.
        """
        user_content = self.extraction_config["user_template"].format(body=body)
        raw, _ = await self.cloud_llm.complete(
            self.extraction_config["system"], user_content, include_thinking=True,
        )
        return parse_stories(raw)

    async def assess_quality(self, title: str, text: str) -> tuple[dict[str, int] | None, str]:
        """Score a story on the 4-dimension quality rubric.

        Returns (scores_dict, chain_of_thought). scores_dict is None if parsing fails.
        """
        user_content = self.quality_config["user_template"].format(title=title, text=text)
        raw, cot = await self.cloud_llm.complete(
            self.quality_config["system"], user_content, include_thinking=True,
        )
        scores = parse_quality_scores(raw)
        return scores, cot

    async def classify_themes(self, title: str, text: str) -> tuple[list[str], str]:
        """Tag a story with Ends Statement themes.

        Returns (themes_list, chain_of_thought).
        """
        user_content = self.theme_config["user_template"].format(title=title, text=text)
        raw, cot = await self.cloud_llm.complete(
            self.theme_config["system"], user_content, include_thinking=True,
        )
        return parse_themes(raw), cot

    async def classify_newsletter(self, body: str) -> list[StoryResult]:
        """Run the full newsletter classification pipeline.

        1. Extract stories from body
        2. For each story: assess quality + classify themes
        3. Return list of StoryResult

        Individual story failures are isolated — a quality failure doesn't
        prevent theme classification, and vice versa.
        """
        stories = await self.extract_stories(body)
        if not stories:
            return []

        results = []
        for title, text in stories:
            result = StoryResult(title=title, text=text)

            # Quality assessment
            try:
                scores, quality_cot = await self.assess_quality(title, text)
                result.quality_cot = quality_cot
                if scores:
                    result.scores = scores
                    result.average_score = sum(scores.values()) / len(scores)
                    result.tier = compute_tier(scores)
            except Exception:
                log.warning("Quality assessment failed for story: %s", title)

            # Theme classification
            try:
                themes, theme_cot = await self.classify_themes(title, text)
                result.themes = themes
                result.theme_cot = theme_cot
            except Exception:
                log.warning("Theme classification failed for story: %s", title)

            results.append(result)

        return results
```

**Step 4: Run tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
cd /workspaces/email-labeler
git add newsletter.py tests/test_newsletter.py
git commit -m "Add NewsletterClassifier with story extraction, quality, and theme LLM calls"
```

---

### Task 4: JSONL result writer

**Files:**
- Modify: `newsletter.py` (add `write_assessment` function)
- Modify: `tests/test_newsletter.py` (add writer tests)

**Step 1: Write tests for JSONL writer**

Append to `tests/test_newsletter.py`:

```python
import json
from pathlib import Path

from newsletter import write_assessment, StoryResult, NewsletterTier


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
        """Story with failed quality assessment still serializes."""
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py::TestWriteAssessment -v`

Expected: FAIL — `ImportError: cannot import name 'write_assessment'`

**Step 3: Implement write_assessment**

Add to `newsletter.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path


def write_assessment(
    output_file: str,
    message_id: str,
    thread_id: str,
    sender: str,
    subject: str,
    overall_tier: NewsletterTier | None,
    stories: list[StoryResult],
) -> None:
    """Append a newsletter assessment record to the JSONL output file.

    Creates parent directories if they don't exist.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": message_id,
        "thread_id": thread_id,
        "from": sender,
        "subject": subject,
        "overall_tier": overall_tier.value if overall_tier else None,
        "stories": [
            {
                "title": s.title,
                "scores": s.scores,
                "average_score": s.average_score,
                "tier": s.tier.value if s.tier else None,
                "themes": s.themes,
                "quality_cot": s.quality_cot,
                "theme_cot": s.theme_cot,
            }
            for s in stories
        ],
    }

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
```

**Step 4: Run tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
cd /workspaces/email-labeler
git add newsletter.py tests/test_newsletter.py
git commit -m "Add JSONL assessment writer for newsletter classification results"
```

---

### Task 5: Extend LabelManager for newsletter labels

**Files:**
- Modify: `labeler.py` (add newsletter label verification and application methods)
- Modify: `tests/test_labeler.py` (add newsletter label tests)

**Step 1: Write tests for newsletter label verification and application**

Append to `tests/test_labeler.py`:

```python
from newsletter import NewsletterTier


@pytest.fixture
def newsletter_config():
    return {
        "labels": {
            "needs_response": "agent/needs-response",
            "fyi": "agent/fyi",
            "low_priority": "agent/low-priority",
            "processed": "agent/processed",
            "personal": "agent/personal",
            "non_personal": "agent/non-personal",
            "actions": {
                "needs_response": "inbox",
                "fyi": "inbox",
                "low_priority": "archive",
            },
        },
        "newsletter": {
            "labels": {
                "newsletter": "agent/newsletter",
                "excellent": "agent/newsletter/excellent",
                "good": "agent/newsletter/good",
                "fair": "agent/newsletter/fair",
                "poor": "agent/newsletter/poor",
                "no_stories": "agent/newsletter/no-stories",
                "themes": {
                    "scripture": "agent/newsletter/theme/scripture",
                    "christlikeness": "agent/newsletter/theme/christlikeness",
                    "church": "agent/newsletter/theme/church",
                    "vocation_family": "agent/newsletter/theme/vocation-family",
                    "disciple_making": "agent/newsletter/theme/disciple-making",
                },
            },
        },
    }


@pytest.fixture
def all_labels_with_newsletter():
    """Gmail API response with all labels including newsletter labels."""
    return {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
            {"id": "Label_2", "name": "agent/fyi", "type": "user"},
            {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
            {"id": "Label_4", "name": "agent/processed", "type": "user"},
            {"id": "Label_5", "name": "agent/personal", "type": "user"},
            {"id": "Label_6", "name": "agent/non-personal", "type": "user"},
            {"id": "Label_10", "name": "agent/newsletter", "type": "user"},
            {"id": "Label_11", "name": "agent/newsletter/excellent", "type": "user"},
            {"id": "Label_12", "name": "agent/newsletter/good", "type": "user"},
            {"id": "Label_13", "name": "agent/newsletter/fair", "type": "user"},
            {"id": "Label_14", "name": "agent/newsletter/poor", "type": "user"},
            {"id": "Label_15", "name": "agent/newsletter/no-stories", "type": "user"},
            {"id": "Label_20", "name": "agent/newsletter/theme/scripture", "type": "user"},
            {"id": "Label_21", "name": "agent/newsletter/theme/christlikeness", "type": "user"},
            {"id": "Label_22", "name": "agent/newsletter/theme/church", "type": "user"},
            {"id": "Label_23", "name": "agent/newsletter/theme/vocation-family", "type": "user"},
            {"id": "Label_24", "name": "agent/newsletter/theme/disciple-making", "type": "user"},
        ]
    }


@pytest.fixture
def newsletter_label_manager(mock_proxy, newsletter_config):
    return LabelManager(proxy_client=mock_proxy, config=newsletter_config)


class TestNewsletterVerifyLabels:
    async def test_all_newsletter_labels_present(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        missing = await newsletter_label_manager.verify_labels()
        assert missing == []

    async def test_missing_newsletter_labels_detected(self, newsletter_label_manager, mock_proxy):
        mock_proxy.list_labels.return_value = {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
                {"id": "Label_2", "name": "agent/fyi", "type": "user"},
                {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
                {"id": "Label_4", "name": "agent/processed", "type": "user"},
                {"id": "Label_5", "name": "agent/personal", "type": "user"},
                {"id": "Label_6", "name": "agent/non-personal", "type": "user"},
                # newsletter labels missing
            ]
        }
        missing = await newsletter_label_manager.verify_labels()
        assert "agent/newsletter" in missing
        assert "agent/newsletter/excellent" in missing
        assert len(missing) == 12


class TestNewsletterApplyLabels:
    async def test_apply_newsletter_excellent(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001"],
            tier=NewsletterTier.EXCELLENT,
            themes=["scripture", "christlikeness"],
        )

        mock_proxy.modify_message.assert_called_once()
        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        add_ids = call_kwargs["add_label_ids"]
        assert "Label_4" in add_ids   # processed
        assert "Label_10" in add_ids  # newsletter marker
        assert "Label_11" in add_ids  # excellent
        assert "Label_20" in add_ids  # theme/scripture
        assert "Label_21" in add_ids  # theme/christlikeness
        assert "INBOX" in call_kwargs["remove_label_ids"]  # archived

    async def test_apply_newsletter_no_stories(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001"],
            tier=None,
            themes=[],
        )

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        add_ids = call_kwargs["add_label_ids"]
        assert "Label_4" in add_ids   # processed
        assert "Label_10" in add_ids  # newsletter marker
        assert "Label_15" in add_ids  # no-stories
        assert "INBOX" in call_kwargs["remove_label_ids"]

    async def test_apply_to_multiple_messages(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001", "msg_002"],
            tier=NewsletterTier.GOOD,
            themes=["church"],
        )
        assert mock_proxy.modify_message.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_labeler.py::TestNewsletterVerifyLabels -v`

Expected: FAIL — `AttributeError` or similar

**Step 3: Extend LabelManager**

Modify `labeler.py` — add newsletter label handling to `verify_labels()` and a new `apply_newsletter_classification()` method.

In `verify_labels()` (after line 68, before `return missing`), add newsletter label collection:

```python
# Newsletter labels (if configured)
nl_config = self.config.get("newsletter", {}).get("labels", {})
if nl_config:
    # Flat labels (newsletter, excellent, good, fair, poor, no_stories)
    for key in ("newsletter", "excellent", "good", "fair", "poor", "no_stories"):
        if key in nl_config:
            required_names.add(nl_config[key])
    # Theme labels
    for theme_name in nl_config.get("themes", {}).values():
        required_names.add(theme_name)
```

Also update `__init__` to store the full config:

```python
def __init__(self, proxy_client: GmailProxyClient, config: dict):
    self.proxy = proxy_client
    self.config = config
    self.labels_config = config["labels"]
    self.label_ids: dict[str, str] = {}
```

Add the `apply_newsletter_classification` method:

```python
async def apply_newsletter_classification(
    self,
    message_ids: list[str],
    tier: "NewsletterTier | None",
    themes: list[str],
) -> None:
    """Apply newsletter classification labels to message(s).

    Args:
        message_ids: Message IDs to label.
        tier: Quality tier of the best story, or None for no-stories.
        themes: List of theme keys (e.g. ["scripture", "church"]).
    """
    nl_labels = self.config["newsletter"]["labels"]
    processed_name = self.labels_config["processed"]

    add_label_ids = [
        self.label_ids[processed_name],
        self.label_ids[nl_labels["newsletter"]],
    ]

    if tier is not None:
        tier_name = nl_labels[tier.value]
        add_label_ids.append(self.label_ids[tier_name])
    else:
        add_label_ids.append(self.label_ids[nl_labels["no_stories"]])

    for theme in themes:
        theme_name = nl_labels["themes"].get(theme)
        if theme_name and theme_name in self.label_ids:
            add_label_ids.append(self.label_ids[theme_name])

    for msg_id in message_ids:
        await self.proxy.modify_message(
            message_id=msg_id,
            add_label_ids=add_label_ids,
            remove_label_ids=["INBOX"],
        )
```

Import at the top of `labeler.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newsletter import NewsletterTier
```

**Step 4: Run tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_labeler.py -v`

Expected: All tests PASS (both existing and new).

**Step 5: Commit**

```bash
cd /workspaces/email-labeler
git add labeler.py tests/test_labeler.py
git commit -m "Extend LabelManager with newsletter label verification and application"
```

---

### Task 6: Newsletter detection helper

**Files:**
- Modify: `newsletter.py` (add `is_newsletter` function)
- Modify: `tests/test_newsletter.py` (add detection tests)

**Step 1: Write tests for newsletter detection**

Append to `tests/test_newsletter.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py::TestIsNewsletter -v`

Expected: FAIL — `ImportError: cannot import name 'is_newsletter'`

**Step 3: Implement is_newsletter**

Add to `newsletter.py`:

```python
from gmail_utils import get_header


def is_newsletter(messages: list[dict], recipient: str) -> bool:
    """Check if any message in a thread was sent to the newsletter address.

    Checks both To and Cc headers, case-insensitive.
    """
    target = recipient.lower()
    for msg in messages:
        headers = msg.get("payload", {}).get("headers", [])
        for header_name in ("To", "Cc"):
            value = get_header(headers, header_name).lower()
            if target in value:
                return True
    return False
```

**Step 4: Run tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_newsletter.py -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
cd /workspaces/email-labeler
git add newsletter.py tests/test_newsletter.py
git commit -m "Add newsletter detection via To/Cc header check"
```

---

### Task 7: Integrate newsletter pipeline into daemon

**Files:**
- Modify: `daemon.py` (add newsletter branch to `process_single_thread`, wire up `NewsletterClassifier` in `run_daemon`)
- Modify: `tests/test_daemon.py` (add newsletter routing tests)

**Step 1: Write tests for daemon newsletter routing**

Append to `tests/test_daemon.py`:

```python
from unittest.mock import patch
from newsletter import NewsletterTier, StoryResult


@pytest.fixture
def mock_newsletter_classifier():
    return AsyncMock()


@pytest.fixture
def newsletter_thread_response():
    """Thread sent to newsletters@dm.org."""
    body = "This month's campus update features Sarah's journey..."
    return {
        "id": "thread_nl",
        "snippet": "This month's campus update...",
        "messages": [
            {
                "id": "msg_nl_001",
                "threadId": "thread_nl",
                "internalDate": "1704067200000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "John Staff <john@dm.org>"},
                        {"name": "To", "value": "newsletters@dm.org"},
                        {"name": "Subject", "value": "February Campus Update"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body.encode()).decode(),
                    },
                },
            },
        ],
    }


class TestNewsletterRouting:
    async def test_newsletter_skips_priority_classification(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        newsletter_thread_response,
    ):
        """Newsletter emails skip sender/priority classification entirely."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                title="Test", text="Content",
                scores={"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
                average_score=4.0, tier=NewsletterTier.EXCELLENT,
                themes=["scripture"],
            )
        ]

        result = await process_single_thread(
            "thread_nl", ["msg_nl_001"],
            mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
        )

        assert result is True
        # Priority classifier should NOT be called
        mock_classifier.classify_sender.assert_not_called()
        mock_classifier.classify.assert_not_called()
        # Newsletter classifier should be called
        mock_newsletter_classifier.classify_newsletter.assert_called_once()
        # Newsletter labels should be applied
        mock_label_manager.apply_newsletter_classification.assert_called_once()

    async def test_non_newsletter_uses_priority_pipeline(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Non-newsletter emails continue through the existing pipeline."""
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001", ["msg_001", "msg_002"],
            mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
        )

        assert result is True
        mock_classifier.classify_sender.assert_called_once()
        mock_classifier.classify.assert_called_once()
        mock_newsletter_classifier.classify_newsletter.assert_not_called()

    async def test_newsletter_no_stories(
        self, mock_proxy, mock_classifier, mock_label_manager,
        mock_newsletter_classifier, cloud_sem, local_sem,
        newsletter_thread_response,
    ):
        """Newsletter with no stories gets labeled and archived."""
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = []

        result = await process_single_thread(
            "thread_nl", ["msg_nl_001"],
            mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
        )

        assert result is True
        call_kwargs = mock_label_manager.apply_newsletter_classification.call_args.kwargs
        assert call_kwargs["tier"] is None
        assert call_kwargs["themes"] == []

    async def test_newsletter_without_classifier_falls_through(
        self, mock_proxy, mock_classifier, mock_label_manager,
        cloud_sem, local_sem, newsletter_thread_response,
    ):
        """If no newsletter_classifier is provided, newsletters go through priority pipeline."""
        mock_proxy.get_thread.return_value = newsletter_thread_response

        result = await process_single_thread(
            "thread_nl", ["msg_nl_001"],
            mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=50000,
        )

        assert result is True
        mock_classifier.classify_sender.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `cd /workspaces/email-labeler && uv run pytest tests/test_daemon.py::TestNewsletterRouting -v`

Expected: FAIL — `process_single_thread() got an unexpected keyword argument 'newsletter_classifier'`

**Step 3: Modify daemon.py — add newsletter branch to process_single_thread**

Add new parameters to `process_single_thread()` signature (after `max_thread_chars`):

```python
async def process_single_thread(
    thread_id: str,
    msg_ids: list[str],
    proxy_client: GmailProxyClient,
    classifier: EmailClassifier,
    label_manager: LabelManager,
    cloud_sem: asyncio.Semaphore,
    local_sem: asyncio.Semaphore,
    max_thread_chars: int,
    newsletter_classifier: "NewsletterClassifier | None" = None,
    newsletter_recipient: str = "",
    newsletter_output_file: str = "",
) -> bool:
```

Add import at top of `daemon.py`:

```python
from newsletter import NewsletterClassifier, NewsletterTier, is_newsletter, write_assessment
```

Inside `process_single_thread`, after the messages are sorted chronologically (line 135) and before the priority check (line 138), insert the newsletter detection branch:

```python
        # Newsletter detection — route to newsletter pipeline if applicable
        if newsletter_classifier and newsletter_recipient:
            if is_newsletter(messages, newsletter_recipient):
                all_msg_ids = [msg["id"] for msg in messages]
                first_headers = messages[0]["payload"]["headers"]
                subject = get_header(first_headers, "Subject")
                sender = get_header(first_headers, "From")
                transcript = format_thread_transcript(messages, max_thread_chars)

                async with cloud_sem:
                    story_results = await newsletter_classifier.classify_newsletter(transcript)

                # Determine overall tier (best story's tier)
                best_tier = None
                all_themes = []
                for sr in story_results:
                    if sr.tier is not None:
                        if best_tier is None or _TIER_RANK.get(sr.tier, 0) > _TIER_RANK.get(best_tier, 0):
                            best_tier = sr.tier
                    all_themes.extend(sr.themes)
                all_themes = list(dict.fromkeys(all_themes))  # dedupe, preserve order

                await label_manager.apply_newsletter_classification(
                    message_ids=all_msg_ids,
                    tier=best_tier,
                    themes=all_themes,
                )

                # Write structured results
                if newsletter_output_file:
                    try:
                        write_assessment(
                            output_file=newsletter_output_file,
                            message_id=all_msg_ids[0],
                            thread_id=thread_id,
                            sender=sender,
                            subject=subject,
                            overall_tier=best_tier,
                            stories=story_results,
                        )
                    except Exception:
                        log.exception("Failed to write newsletter assessment for thread %s", thread_id)

                story_count = len(story_results)
                log.info(
                    "Newsletter thread %s: %d stories, tier=%s, themes=%s — %s",
                    thread_id, story_count,
                    best_tier.value if best_tier else "no-stories",
                    all_themes, subject,
                )
                return True
```

Add the tier ranking dict near the top of `daemon.py` (after imports):

```python
_TIER_RANK = {
    NewsletterTier.POOR: 0,
    NewsletterTier.FAIR: 1,
    NewsletterTier.GOOD: 2,
    NewsletterTier.EXCELLENT: 3,
}
```

**Step 4: Wire up NewsletterClassifier in run_daemon()**

In `run_daemon()`, after creating the `classifier` (line 254), add:

```python
    # Newsletter classifier (if configured)
    nl_config = config.get("newsletter")
    newsletter_classifier = None
    newsletter_recipient = ""
    newsletter_output_file = ""
    if nl_config:
        newsletter_classifier = NewsletterClassifier(cloud_llm=cloud_llm, config=config)
        newsletter_recipient = nl_config["recipient"]
        newsletter_output_file = nl_config.get("output_file", "")
        log.info("Newsletter classification enabled for: %s", newsletter_recipient)
```

Update the `process_single_thread` call inside the polling loop (around line 309) to pass the new arguments:

```python
            results = await asyncio.gather(
                *(
                    process_single_thread(
                        tid,
                        msg_ids,
                        proxy_client,
                        classifier,
                        label_manager,
                        cloud_sem,
                        local_sem,
                        max_thread_chars,
                        newsletter_classifier=newsletter_classifier,
                        newsletter_recipient=newsletter_recipient,
                        newsletter_output_file=newsletter_output_file,
                    )
                    for tid, msg_ids in threads.items()
                ),
                return_exceptions=True,
            )
```

**Step 5: Run all tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/ -v`

Expected: All tests PASS (existing + new).

**Step 6: Commit**

```bash
cd /workspaces/email-labeler
git add daemon.py tests/test_daemon.py
git commit -m "Integrate newsletter classification pipeline into daemon loop"
```

---

### Task 8: Run full test suite and verify

**Step 1: Run all tests**

Run: `cd /workspaces/email-labeler && uv run pytest tests/ -v`

Expected: All tests PASS.

**Step 2: Run linter**

Run: `cd /workspaces/email-labeler && uv run ruff check .`

Expected: No errors (fix any that appear).

**Step 3: Run formatter**

Run: `cd /workspaces/email-labeler && uv run ruff format --check .`

Expected: No formatting issues (fix any that appear).

**Step 4: Verify config loads**

Run: `cd /workspaces/email-labeler && uv run python -c "from daemon import load_config; c = load_config(); print('Newsletter:', c['newsletter']['recipient']); print('Labels:', list(c['newsletter']['labels'].keys())); print('Prompts:', list(c['newsletter']['prompts'].keys()))"`

Expected:
```
Newsletter: newsletters@dm.org
Labels: ['newsletter', 'excellent', 'good', 'fair', 'poor', 'no_stories', 'themes']
Prompts: ['story_extraction', 'quality_assessment', 'theme_classification']
```

**Step 5: Commit any fixups**

If any fixes were needed, commit them:

```bash
cd /workspaces/email-labeler
git add -A
git commit -m "Fix lint/format issues in newsletter classification"
```
