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
