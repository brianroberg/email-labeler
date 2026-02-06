"""Two-tier email classification logic.

Stage 1: Cloud LLM classifies sender as PERSON or SERVICE (metadata only).
Stage 2: Full body sent to local LLM (person) or cloud LLM (service) for classification.

Privacy invariant: Person email bodies NEVER leave the local network.
"""

import re
from dataclasses import dataclass
from enum import Enum

from llm_client import LLMClient


class SenderType(Enum):
    PERSON = "person"
    SERVICE = "service"


class EmailLabel(Enum):
    NEEDS_RESPONSE = "needs_response"
    FYI = "fyi"
    LOW_PRIORITY = "low_priority"
    UNWANTED = "unwanted"


@dataclass
class EmailMetadata:
    message_id: str
    sender: str
    subject: str
    snippet: str


@dataclass
class ClassificationResult:
    sender_type: SenderType
    sender_type_raw: str
    label: EmailLabel
    label_raw: str


def parse_sender(from_header: str) -> tuple[str, str]:
    """Parse a From header into (name, email).

    Handles formats:
        "John Doe <john@example.com>" -> ("John Doe", "john@example.com")
        "john@example.com" -> ("", "john@example.com")
        '"John Doe" <john@example.com>' -> ("John Doe", "john@example.com")
    """
    if not from_header:
        return ("", "")

    match = re.match(r'^"?([^"<]*?)"?\s*<([^>]+)>$', from_header)
    if match:
        name = match.group(1).strip()
        email = match.group(2).strip()
        return (name, email)

    # No angle brackets — treat entire string as email if it has @
    if "@" in from_header:
        return ("", from_header.strip())

    return ("", from_header.strip())


def parse_sender_type(raw_llm_output: str) -> SenderType:
    """Parse LLM output into SenderType. Defaults to SERVICE (safe)."""
    cleaned = raw_llm_output.strip().upper()
    if cleaned.startswith("PERSON"):
        return SenderType.PERSON
    if cleaned.startswith("SERVICE"):
        return SenderType.SERVICE
    return SenderType.SERVICE


def parse_email_label(raw_llm_output: str) -> EmailLabel:
    """Parse LLM output into EmailLabel. Defaults to LOW_PRIORITY (safe)."""
    cleaned = raw_llm_output.strip().upper()
    if cleaned.startswith("NEEDS_RESPONSE"):
        return EmailLabel.NEEDS_RESPONSE
    if cleaned.startswith("FYI"):
        return EmailLabel.FYI
    if cleaned.startswith("LOW_PRIORITY"):
        return EmailLabel.LOW_PRIORITY
    if cleaned.startswith("UNWANTED"):
        return EmailLabel.UNWANTED
    return EmailLabel.LOW_PRIORITY


class EmailClassifier:
    """Two-tier email classifier using cloud and local LLMs."""

    def __init__(self, cloud_llm: LLMClient, local_llm: LLMClient, config: dict):
        self.cloud_llm = cloud_llm
        self.local_llm = local_llm
        self.sender_config = config["prompts"]["sender_classification"]
        self.email_config = config["prompts"]["email_classification"]

    async def classify_sender(self, metadata: EmailMetadata) -> tuple[SenderType, str]:
        """Stage 1: Classify sender as PERSON or SERVICE using cloud LLM.

        Only metadata (sender, subject, snippet) is sent — never the body.
        """
        user_content = self.sender_config["user_template"].format(
            sender=metadata.sender,
            subject=metadata.subject,
            snippet=metadata.snippet,
        )
        raw = await self.cloud_llm.complete(self.sender_config["system"], user_content)
        return (parse_sender_type(raw), raw)

    async def classify_email(
        self, metadata: EmailMetadata, body: str, sender_type: SenderType
    ) -> tuple[EmailLabel, str]:
        """Stage 2: Classify email content. Routes based on sender type.

        Person emails -> local LLM (privacy preserved)
        Service emails -> cloud LLM
        """
        llm = self.local_llm if sender_type == SenderType.PERSON else self.cloud_llm
        user_content = self.email_config["user_template"].format(
            sender=metadata.sender,
            subject=metadata.subject,
            body=body,
        )
        raw = await llm.complete(self.email_config["system"], user_content)
        return (parse_email_label(raw), raw)

    async def classify(
        self,
        metadata: EmailMetadata,
        body: str,
        sender_type: SenderType | None = None,
        sender_type_raw: str | None = None,
    ) -> ClassificationResult:
        """Full two-stage classification pipeline.

        If sender_type is provided, Stage 1 is skipped (avoids duplicate LLM calls
        when the caller has already classified the sender).
        """
        if sender_type is None:
            sender_type, sender_type_raw = await self.classify_sender(metadata)
        label, label_raw = await self.classify_email(metadata, body, sender_type)
        return ClassificationResult(
            sender_type=sender_type,
            sender_type_raw=sender_type_raw or "",
            label=label,
            label_raw=label_raw,
        )
