"""Dataclasses for the newsletter golden set and evaluation results.

All types support JSONL serialization via to_dict() / from_dict().
from_dict is tolerant of missing optional keys (via d.get(...) defaults) so
older golden sets and result files keep loading as the schema grows.
"""

from dataclasses import dataclass, field


def _coerce_themes(value) -> dict[str, str]:
    """Normalize a stored themes value to a grade dict (theme -> "present"/
    "emphasized"), defaulting a missing/empty value to ``{}`` (issue #53)."""
    return dict(value) if value else {}


@dataclass
class GoldenStory:
    """One story within a golden newsletter (ground truth)."""

    story_id: str  # stable, f"{thread_id}:{index}"
    text: str
    expected_scores: dict[str, int] | None = None  # simple/concrete/personal/dynamic, 1-3 (Poor/OK/Good)
    expected_tier: str | None = None  # "excellent" / "good" / "fair" / "poor"
    expected_themes: dict[str, str] = field(default_factory=dict)  # theme -> "present"/"emphasized"
    reviewed: bool = False
    notes: str = ""
    excluded: bool = False  # drop from quality/theme scoring, keep as extraction truth

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "text": self.text,
            "expected_scores": self.expected_scores,
            "expected_tier": self.expected_tier,
            "expected_themes": self.expected_themes,
            "reviewed": self.reviewed,
            "notes": self.notes,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenStory":
        # ``title`` was dropped from the schema; older golden sets that still
        # carry it load fine because the extra key is simply ignored here.
        return cls(
            story_id=d["story_id"],
            text=d["text"],
            expected_scores=d.get("expected_scores"),
            expected_tier=d.get("expected_tier"),
            expected_themes=_coerce_themes(d.get("expected_themes")),
            reviewed=d.get("reviewed", False),
            notes=d.get("notes", ""),
            excluded=d.get("excluded", False),
        )


@dataclass
class GoldenNewsletter:
    """One newsletter in the golden set; owns its list of GoldenStory."""

    thread_id: str
    message_id: str
    sender: str
    subject: str
    body: str  # raw body fed verbatim to extract_stories
    stories: list[GoldenStory] = field(default_factory=list)
    source: str = "harvested"
    harvested_at: str = ""  # ISO 8601
    seeded_from: str = ""  # provenance of Phase-A story seeding, e.g. "parse_stories"
    reviewed: bool = False  # story list confirmed = authoritative extraction truth
    notes: str = ""
    excluded: bool = False

    def to_dict(self) -> dict:
        return {
            "type": "golden_newsletter",
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "stories": [s.to_dict() for s in self.stories],
            "source": self.source,
            "harvested_at": self.harvested_at,
            "seeded_from": self.seeded_from,
            "reviewed": self.reviewed,
            "notes": self.notes,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenNewsletter":
        return cls(
            thread_id=d["thread_id"],
            message_id=d["message_id"],
            sender=d["sender"],
            subject=d["subject"],
            body=d["body"],
            stories=[GoldenStory.from_dict(s) for s in d.get("stories", [])],
            source=d.get("source", "harvested"),
            harvested_at=d.get("harvested_at", ""),
            seeded_from=d.get("seeded_from", ""),
            reviewed=d.get("reviewed", False),
            notes=d.get("notes", ""),
            excluded=d.get("excluded", False),
        )


@dataclass
class StoryPrediction:
    """One prediction per golden story in quality/theme mode."""

    story_id: str
    thread_id: str
    expected_scores: dict[str, int] | None = None
    expected_tier: str | None = None
    expected_themes: dict[str, str] = field(default_factory=dict)
    predicted_scores: dict[str, int] | None = None
    predicted_tier: str | None = None
    predicted_themes: dict[str, str] = field(default_factory=dict)
    scores_raw: str | None = None
    themes_raw: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "story_prediction",
            "story_id": self.story_id,
            "thread_id": self.thread_id,
            "expected_scores": self.expected_scores,
            "expected_tier": self.expected_tier,
            "expected_themes": self.expected_themes,
            "predicted_scores": self.predicted_scores,
            "predicted_tier": self.predicted_tier,
            "predicted_themes": self.predicted_themes,
            "scores_raw": self.scores_raw,
            "themes_raw": self.themes_raw,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StoryPrediction":
        return cls(
            story_id=d["story_id"],
            thread_id=d["thread_id"],
            expected_scores=d.get("expected_scores"),
            expected_tier=d.get("expected_tier"),
            expected_themes=_coerce_themes(d.get("expected_themes")),
            predicted_scores=d.get("predicted_scores"),
            predicted_tier=d.get("predicted_tier"),
            predicted_themes=_coerce_themes(d.get("predicted_themes")),
            scores_raw=d.get("scores_raw"),
            themes_raw=d.get("themes_raw"),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
        )


@dataclass
class ExtractionPrediction:
    """One prediction per newsletter in extraction mode."""

    thread_id: str
    golden_stories: list[dict] = field(default_factory=list)  # [{"story_id","text"}]
    predicted_stories: list[dict] = field(default_factory=list)  # [{"text"}]
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "extraction_prediction",
            "thread_id": self.thread_id,
            "golden_stories": self.golden_stories,
            "predicted_stories": self.predicted_stories,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionPrediction":
        return cls(
            thread_id=d["thread_id"],
            golden_stories=d.get("golden_stories", []),
            predicted_stories=d.get("predicted_stories", []),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
        )


@dataclass
class NewsletterThinkingEntry:
    """Chain-of-thought content for one eval-run unit of work.

    Story-level entries (quality/theme scoring) set story_id; newsletter-level
    entries (extraction segmentation) set thread_id + extraction_cot instead.
    Stored in sidecar .cot.jsonl files alongside main results.
    """

    story_id: str = ""
    quality_cot: str = ""
    theme_cot: str = ""
    thread_id: str = ""
    extraction_cot: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "thinking",
            "story_id": self.story_id,
            "quality_cot": self.quality_cot,
            "theme_cot": self.theme_cot,
            "thread_id": self.thread_id,
            "extraction_cot": self.extraction_cot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NewsletterThinkingEntry":
        return cls(
            story_id=d.get("story_id", ""),
            quality_cot=d.get("quality_cot", ""),
            theme_cot=d.get("theme_cot", ""),
            thread_id=d.get("thread_id", ""),
            extraction_cot=d.get("extraction_cot", ""),
        )


@dataclass
class NewsletterRunMeta:
    """Metadata for a newsletter evaluation run (first line of results JSONL)."""

    run_id: str
    timestamp: str  # ISO 8601
    config_hash: str
    config_path: str
    newsletter_model: str
    golden_set_path: str
    golden_set_count: int  # number of newsletters evaluated
    # Stories the run evaluated: Phase-B-labeled, non-excluded stories for story
    # modes (quality/themes/all); ALL golden stories for extraction-only runs,
    # since extraction matches against the full confirmed story list.
    story_count: int
    mode: str = "all"  # "extraction" / "quality" / "themes" / "all"
    prompt_hash: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    extra_body: dict | None = None
    parallelism: int = 1
    tag: str = ""
    seeded_from: str = ""
    # System prompts (constant across a run, deterministic from config)
    extraction_system_prompt: str = ""
    quality_system_prompt: str = ""
    theme_system_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "run_meta",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "config_path": self.config_path,
            "newsletter_model": self.newsletter_model,
            "golden_set_path": self.golden_set_path,
            "golden_set_count": self.golden_set_count,
            "story_count": self.story_count,
            "mode": self.mode,
            "prompt_hash": self.prompt_hash,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "parallelism": self.parallelism,
            "tag": self.tag,
            "seeded_from": self.seeded_from,
            "extraction_system_prompt": self.extraction_system_prompt,
            "quality_system_prompt": self.quality_system_prompt,
            "theme_system_prompt": self.theme_system_prompt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NewsletterRunMeta":
        return cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            config_hash=d["config_hash"],
            config_path=d["config_path"],
            newsletter_model=d["newsletter_model"],
            golden_set_path=d["golden_set_path"],
            golden_set_count=d["golden_set_count"],
            story_count=d["story_count"],
            mode=d.get("mode", "all"),
            prompt_hash=d.get("prompt_hash", ""),
            temperature=d.get("temperature", 0.0),
            max_tokens=d.get("max_tokens", 0),
            extra_body=d.get("extra_body"),
            parallelism=d.get("parallelism", 1),
            tag=d.get("tag", ""),
            seeded_from=d.get("seeded_from", ""),
            extraction_system_prompt=d.get("extraction_system_prompt", ""),
            quality_system_prompt=d.get("quality_system_prompt", ""),
            theme_system_prompt=d.get("theme_system_prompt", ""),
        )
