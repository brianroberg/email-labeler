"""Dataclasses for golden set and evaluation results.

All types support JSONL serialization via to_dict() / from_dict().
"""

from dataclasses import dataclass


@dataclass
class GoldenThread:
    """One thread in the golden set (ground truth)."""

    thread_id: str
    messages: list[dict]  # Raw Gmail message resources from proxy_client.get_thread()
    senders: list[str]
    subject: str
    snippet: str
    expected_sender_type: str  # "person" or "service"
    expected_label: str  # "needs_response" / "fyi" / "low_priority" / "unwanted"
    source: str = "harvested"  # "harvested" or "manual"
    harvested_at: str = ""  # ISO 8601
    reviewed: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "senders": self.senders,
            "subject": self.subject,
            "snippet": self.snippet,
            "expected_sender_type": self.expected_sender_type,
            "expected_label": self.expected_label,
            "source": self.source,
            "harvested_at": self.harvested_at,
            "reviewed": self.reviewed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenThread":
        return cls(
            thread_id=d["thread_id"],
            messages=d["messages"],
            senders=d["senders"],
            subject=d["subject"],
            snippet=d["snippet"],
            expected_sender_type=d["expected_sender_type"],
            expected_label=d["expected_label"],
            source=d.get("source", "harvested"),
            harvested_at=d.get("harvested_at", ""),
            reviewed=d.get("reviewed", False),
            notes=d.get("notes", ""),
        )


@dataclass
class PredictionResult:
    """One prediction from an evaluation run."""

    thread_id: str
    expected_sender_type: str
    expected_label: str
    predicted_sender_type: str | None = None
    predicted_label: str | None = None
    predicted_sender_type_raw: str | None = None
    predicted_label_raw: str | None = None
    sender_type_correct: bool | None = None
    label_correct: bool | None = None
    privacy_violation: bool = False  # True if expected=person, predicted=service
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "prediction",
            "thread_id": self.thread_id,
            "expected_sender_type": self.expected_sender_type,
            "expected_label": self.expected_label,
            "predicted_sender_type": self.predicted_sender_type,
            "predicted_label": self.predicted_label,
            "predicted_sender_type_raw": self.predicted_sender_type_raw,
            "predicted_label_raw": self.predicted_label_raw,
            "sender_type_correct": self.sender_type_correct,
            "label_correct": self.label_correct,
            "privacy_violation": self.privacy_violation,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionResult":
        return cls(
            thread_id=d["thread_id"],
            expected_sender_type=d["expected_sender_type"],
            expected_label=d["expected_label"],
            predicted_sender_type=d.get("predicted_sender_type"),
            predicted_label=d.get("predicted_label"),
            predicted_sender_type_raw=d.get("predicted_sender_type_raw"),
            predicted_label_raw=d.get("predicted_label_raw"),
            sender_type_correct=d.get("sender_type_correct"),
            label_correct=d.get("label_correct"),
            privacy_violation=d.get("privacy_violation", False),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
        )


@dataclass
class RunMeta:
    """Metadata for an evaluation run (first line of results JSONL)."""

    run_id: str
    timestamp: str  # ISO 8601
    config_hash: str
    config_path: str
    cloud_model: str
    local_model: str
    golden_set_path: str
    golden_set_count: int
    stages: str = "full"  # "full" / "stage1_only" / "stage2_only"
    parallelism: int = 1
    tag: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "run_meta",
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "config_path": self.config_path,
            "cloud_model": self.cloud_model,
            "local_model": self.local_model,
            "golden_set_path": self.golden_set_path,
            "golden_set_count": self.golden_set_count,
            "stages": self.stages,
            "parallelism": self.parallelism,
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunMeta":
        return cls(
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            config_hash=d["config_hash"],
            config_path=d["config_path"],
            cloud_model=d["cloud_model"],
            local_model=d["local_model"],
            golden_set_path=d["golden_set_path"],
            golden_set_count=d["golden_set_count"],
            stages=d.get("stages", "full"),
            parallelism=d.get("parallelism", 1),
            tag=d.get("tag", ""),
        )
