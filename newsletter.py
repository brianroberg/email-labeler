"""Newsletter story classification pipeline.

Extracts stories from ministry newsletters, scores them on a quality rubric,
and tags them with Ends Statement themes. All LLM calls use the cloud endpoint
(newsletter content is not privacy-sensitive).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from gmail_utils import get_header
from llm_client import LLMClient

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


def write_assessment(
    output_file: str,
    message_id: str,
    thread_id: str,
    sender: str,
    subject: str,
    overall_tier: NewsletterTier | None,
    stories: list[StoryResult],
) -> None:
    """Append a newsletter assessment record to the JSONL output file."""
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


class NewsletterClassifier:
    """Classifies newsletter stories for quality and themes."""

    def __init__(self, cloud_llm: LLMClient, config: dict):
        self.cloud_llm = cloud_llm
        nl_config = config["newsletter"]
        self.extraction_config = nl_config["prompts"]["story_extraction"]
        self.quality_config = nl_config["prompts"]["quality_assessment"]
        self.theme_config = nl_config["prompts"]["theme_classification"]

    async def extract_stories(self, body: str) -> list[tuple[str, str]]:
        """Extract individual stories from a newsletter body."""
        user_content = self.extraction_config["user_template"].format(body=body)
        raw, _ = await self.cloud_llm.complete(
            self.extraction_config["system"], user_content, include_thinking=True,
        )
        return parse_stories(raw)

    async def assess_quality(self, title: str, text: str) -> tuple[dict[str, int] | None, str]:
        """Score a story on the 4-dimension quality rubric."""
        user_content = self.quality_config["user_template"].format(title=title, text=text)
        raw, cot = await self.cloud_llm.complete(
            self.quality_config["system"], user_content, include_thinking=True,
        )
        scores = parse_quality_scores(raw)
        return scores, cot

    async def classify_themes(self, title: str, text: str) -> tuple[list[str], str]:
        """Tag a story with Ends Statement themes."""
        user_content = self.theme_config["user_template"].format(title=title, text=text)
        raw, cot = await self.cloud_llm.complete(
            self.theme_config["system"], user_content, include_thinking=True,
        )
        return parse_themes(raw), cot

    async def classify_newsletter(self, body: str) -> list[StoryResult]:
        """Run the full newsletter classification pipeline.

        Individual story failures are isolated — a quality failure doesn't
        prevent theme classification, and vice versa.
        """
        stories = await self.extract_stories(body)
        if not stories:
            return []

        results = []
        for title, text in stories:
            result = StoryResult(title=title, text=text)

            try:
                scores, quality_cot = await self.assess_quality(title, text)
                result.quality_cot = quality_cot
                if scores:
                    result.scores = scores
                    result.average_score = sum(scores.values()) / len(scores)
                    result.tier = compute_tier(scores)
            except Exception:
                log.warning("Quality assessment failed for story: %s", title)

            try:
                themes, theme_cot = await self.classify_themes(title, text)
                result.themes = themes
                result.theme_cot = theme_cot
            except Exception:
                log.warning("Theme classification failed for story: %s", title)

            results.append(result)

        return results
