"""Data loading, filtering, and formatting for the newsletter assessment TUI."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Story:
    title: str
    text: str
    scores: dict[str, int] | None
    average_score: float | None
    tier: str | None
    themes: list[str]
    quality_cot: str
    theme_cot: str


@dataclass
class Assessment:
    timestamp: str
    message_id: str
    thread_id: str
    sender: str
    subject: str
    overall_tier: str | None
    stories: list[Story]


def load_assessments(path: str) -> list[Assessment]:
    """Load newsletter assessments from a JSONL file.

    Skips malformed lines with a warning. Returns empty list if file
    doesn't exist or is empty.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    assessments = []
    for line_num, line in enumerate(file_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            stories = [
                Story(
                    title=s["title"],
                    text=s["text"],
                    scores=s.get("scores"),
                    average_score=s.get("average_score"),
                    tier=s.get("tier"),
                    themes=s.get("themes", []),
                    quality_cot=s.get("quality_cot", ""),
                    theme_cot=s.get("theme_cot", ""),
                )
                for s in record.get("stories", [])
            ]
            assessments.append(
                Assessment(
                    timestamp=record["timestamp"],
                    message_id=record["message_id"],
                    thread_id=record["thread_id"],
                    sender=record["from"],
                    subject=record["subject"],
                    overall_tier=record.get("overall_tier"),
                    stories=stories,
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Skipping malformed line %d: %s", line_num, exc)

    return assessments


def filter_by_tier(assessments: list[Assessment], tier: str) -> list[Assessment]:
    """Return assessments matching the given overall tier."""
    return [a for a in assessments if a.overall_tier == tier]


def filter_by_theme(assessments: list[Assessment], theme: str) -> list[Assessment]:
    """Return assessments where any story has the given theme."""
    return [a for a in assessments if any(theme in s.themes for s in a.stories)]


def available_tiers(assessments: list[Assessment]) -> list[str]:
    """Return sorted unique non-None tier values present in assessments."""
    return sorted({a.overall_tier for a in assessments if a.overall_tier is not None})


def available_themes(assessments: list[Assessment]) -> list[str]:
    """Return sorted unique theme values across all stories."""
    themes: set[str] = set()
    for a in assessments:
        for s in a.stories:
            themes.update(s.themes)
    return sorted(themes)


def format_detail(story: Story) -> str:
    """Format a Story as readable text for the detail panel."""
    tier = story.tier or "\u2014"
    avg = f"{story.average_score:.2f}" if story.average_score is not None else "\u2014"

    if story.scores:
        scores_str = "  ".join(f"{k}: {v}" for k, v in story.scores.items())
    else:
        scores_str = "(not available)"

    themes_str = ", ".join(story.themes) if story.themes else "(none)"
    quality_cot = story.quality_cot or "(no reasoning recorded)"
    theme_cot = story.theme_cot or "(no reasoning recorded)"

    return (
        f"{story.title}\n"
        f"Tier: {tier}  |  Average: {avg}\n"
        f"Scores: {scores_str}\n"
        f"Themes: {themes_str}\n"
        f"\n\u2500\u2500 Quality CoT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{quality_cot}\n"
        f"\n\u2500\u2500 Theme CoT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{theme_cot}\n"
        f"\n\u2500\u2500 Story Text \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"{story.text}\n"
    )
