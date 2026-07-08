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
from email.utils import parsedate_to_datetime
from enum import Enum
from pathlib import Path

from gmail_utils import get_header
from llm_client import LLMClient, LLMContentError, LLMUnavailableError

log = logging.getLogger(__name__)


class NewsletterTier(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


_VALID_THEMES = {
    "SCRIPTURE",
    "CHRISTLIKENESS",
    "CHURCH",
    "VOCATION_FAMILY",
    "DISCIPLE_MAKING",
}

_DIMENSIONS = ("simple", "concrete", "personal", "dynamic")

# Storytelling dimensions are graded on a 3-value rubric (issue #53). The tokens
# map to integers so the tier average and eval metrics stay simple arithmetic.
_SCORE_TOKENS = {"POOR": 1, "OK": 2, "GOOD": 3}

# Ends-Statement themes are graded on a 3-value rubric (issue #53). "absent" is
# represented by omission, so only these two grades ever appear in a stored dict.
_THEME_GRADES = {"PRESENT": "present", "EMPHASIZED": "emphasized"}


@dataclass
class StoryResult:
    text: str
    scores: dict[str, int] | None = None
    average_score: float | None = None
    tier: NewsletterTier | None = None
    themes: dict[str, str] = field(default_factory=dict)
    quality_cot: str = ""
    theme_cot: str = ""


def parse_stories(raw: str) -> list[str]:
    """Parse LLM story extraction output into a list of story texts.

    Expected format (one story per block, blocks separated by blank lines):
        STORY: <story text, possibly multi-line>

    Returns empty list for NO_STORIES or unparseable input.
    """
    # Normalize CRLF/CR to LF first so the blank-line split and the verbatim
    # story slices behave identically regardless of the model's line endings.
    stripped = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not stripped or stripped.upper() == "NO_STORIES":
        return []

    stories = []
    # Split only at a STORY: that begins the text or follows a blank line (the
    # documented "one story per block, separated by blank lines" format). This
    # keeps a story whose own body contains a line starting with "STORY:" intact
    # instead of splitting it in two.
    blocks = re.split(r"(?:\A|\n[ \t]*\n)(?=STORY:)", stripped)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # ``search`` (not ``match``): a preamble line the model glued onto the
        # first STORY: with a single newline is skipped rather than dropping the
        # whole block. The first STORY: wins and DOTALL keeps any later
        # "STORY:"-prefixed body line as part of this story's text.
        match = re.search(r"^STORY:\s*(.*)", block, flags=re.DOTALL | re.MULTILINE)
        if match:
            text = match.group(1).strip()
            if text:
                stories.append(text)

    return stories


def parse_quality_scores(raw: str) -> dict[str, int] | None:
    """Parse LLM quality assessment output into dimension scores.

    Each dimension is graded POOR/OK/GOOD (mapped to 1/2/3). Returns None if any
    dimension is missing or its value is not one of those tokens (a legacy digit
    or an unknown word is a parse failure, not a clamped score).
    """
    if not raw.strip():
        return None

    scores = {}
    for dim in _DIMENSIONS:
        pattern = rf"{dim.upper()}\s*:\s*(POOR|OK|GOOD)\b"
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            return None
        scores[dim] = _SCORE_TOKENS[match.group(1).upper()]

    return scores


def parse_themes(raw: str) -> dict[str, str]:
    """Parse LLM theme classification output into graded theme labels.

    Each recognized theme line is ``THEME: ABSENT|PRESENT|EMPHASIZED``. Returns a
    dict mapping lowercase theme name -> "present"/"emphasized"; Absent themes and
    unrecognized names are omitted. Returns an empty dict for a NONE response.
    """
    stripped = raw.strip()
    if not stripped or stripped.upper() == "NONE":
        return {}

    themes: dict[str, str] = {}
    for line in stripped.splitlines():
        match = re.match(
            r"\s*([A-Za-z_]+)\s*:\s*(ABSENT|PRESENT|EMPHASIZED)\b",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        name = match.group(1).upper()
        grade = match.group(2).upper()
        if name in _VALID_THEMES and grade in _THEME_GRADES:
            themes[name.lower()] = _THEME_GRADES[grade]

    return themes


def compute_tier(scores: dict[str, int]) -> NewsletterTier:
    """Derive quality tier from the 3-value dimension scores.

    Dimensions are 1/2/3 (Poor/OK/Good); the tier is banded off their mean:
    excellent >= 2.75, good >= 2.25, fair >= 1.75, else poor (issue #53).
    """
    avg = sum(scores.values()) / len(scores)
    if avg >= 2.75:
        return NewsletterTier.EXCELLENT
    if avg >= 2.25:
        return NewsletterTier.GOOD
    if avg >= 1.75:
        return NewsletterTier.FAIR
    return NewsletterTier.POOR


# Ordering of theme grades for cross-story aggregation (higher = stronger).
THEME_GRADE_RANK = {"present": 1, "emphasized": 2}


def aggregate_theme_grades(stories: list["StoryResult"]) -> dict[str, str]:
    """Merge per-story graded themes into one dict for the whole newsletter.

    Each theme takes the strongest grade seen across all stories
    (emphasized > present); themes absent from every story are omitted.
    """
    merged: dict[str, str] = {}
    for story in stories:
        for theme, grade in story.themes.items():
            if THEME_GRADE_RANK.get(grade, 0) > THEME_GRADE_RANK.get(merged.get(theme, ""), 0):
                merged[theme] = grade
    return merged


def parse_send_date(date_header: str, internal_date_ms: str | None = None) -> str | None:
    """Normalize an email's send date to an ISO-8601 UTC string.

    Prefers the RFC-2822 ``Date`` header (a header with no timezone is assumed
    UTC); falls back to Gmail's ``internalDate`` (epoch milliseconds) when the
    header is missing or unparseable. Returns None if neither yields a valid date.
    ISO-8601 UTC sorts lexicographically = chronologically, which the review TUI
    relies on for date sorting/filtering (issue #35/#36).
    """
    if date_header:
        try:
            dt = parsedate_to_datetime(date_header)
        except (ValueError, TypeError):
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()

    if internal_date_ms:
        try:
            seconds = int(internal_date_ms) / 1000
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (ValueError, TypeError, OverflowError, OSError):
            pass

    return None


def write_assessment(
    output_file: str,
    message_id: str,
    thread_id: str,
    sender: str,
    subject: str,
    overall_tier: NewsletterTier | None,
    stories: list[StoryResult],
    *,
    send_date: str | None = None,
    model: str | None = None,
) -> None:
    """Append a newsletter assessment record to the JSONL output file.

    ``timestamp`` is the *processed* time (UTC now). ``send_date`` is the email's
    own send date (ISO-8601 UTC, email-intrinsic) and ``model`` is the classifier
    model — both used by the review TUI (issue #35/#36). Old records lacking these
    keys are read with ``.get()`` fallbacks by consumers.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": message_id,
        "thread_id": thread_id,
        "from": sender,
        "subject": subject,
        "send_date": send_date,
        "model": model,
        "overall_tier": overall_tier.value if overall_tier else None,
        "stories": [
            {
                "text": s.text,
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

    async def extract_stories(self, body: str) -> list[str]:
        """Extract individual story texts from a newsletter body."""
        user_content = self.extraction_config["user_template"].format(body=body)
        raw, _ = await self.cloud_llm.complete(
            self.extraction_config["system"],
            user_content,
            include_thinking=True,
        )
        return parse_stories(raw)

    async def assess_quality(self, text: str) -> tuple[dict[str, int] | None, str]:
        """Score a story on the 4-dimension quality rubric."""
        user_content = self.quality_config["user_template"].format(text=text)
        raw, cot = await self.cloud_llm.complete(
            self.quality_config["system"],
            user_content,
            include_thinking=True,
        )
        scores = parse_quality_scores(raw)
        return scores, cot

    async def classify_themes(self, text: str) -> tuple[dict[str, str], str]:
        """Tag a story with graded Ends Statement themes (theme -> grade)."""
        user_content = self.theme_config["user_template"].format(text=text)
        raw, cot = await self.cloud_llm.complete(
            self.theme_config["system"],
            user_content,
            include_thinking=True,
        )
        return parse_themes(raw), cot

    async def classify_newsletter(self, body: str) -> list[StoryResult]:
        """Run the full newsletter classification pipeline over story texts.

        Individual *per-story* failures are isolated — a quality failure doesn't
        prevent theme classification, and vice versa.

        A *transient* LLM outage (LLMUnavailableError) is NOT isolated: it
        propagates so the daemon retries the whole newsletter thread next cycle,
        rather than committing a permanently mis-graded newsletter (empty
        tier/themes) and marking it processed. Mirrors the email pipeline's
        transient-outage guarantee.
        """
        stories = await self.extract_stories(body)
        if not stories:
            return []

        results = []
        for text in stories:
            result = StoryResult(text=text)

            try:
                scores, quality_cot = await self.assess_quality(text)
                result.quality_cot = quality_cot
                if scores:
                    result.scores = scores
                    result.average_score = sum(scores.values()) / len(scores)
                    result.tier = compute_tier(scores)
            except (LLMUnavailableError, LLMContentError):
                # transient outage OR a content-less response (issue #30): both
                # affect every story, so propagate to give-up rather than commit
                # a permanently mis-graded (empty) newsletter.
                raise
            except Exception:
                log.warning("Quality assessment failed for story: %s", text[:60])

            try:
                themes, theme_cot = await self.classify_themes(text)
                result.themes = themes
                result.theme_cot = theme_cot
            except (LLMUnavailableError, LLMContentError):
                raise
            except Exception:
                log.warning("Theme classification failed for story: %s", text[:60])

            results.append(result)

        return results
