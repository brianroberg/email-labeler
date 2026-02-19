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
