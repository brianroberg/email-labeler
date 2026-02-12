"""Two-tier email classification logic.

Stage 1: Cloud LLM classifies sender as PERSON or SERVICE (metadata only).
Stage 2: Full body sent to local LLM (person) or cloud LLM (service) for classification.

Privacy invariant: Person email bodies NEVER leave the local network.
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum

from llm_client import LLMClient

log = logging.getLogger(__name__)


class SenderType(Enum):
    PERSON = "person"
    SERVICE = "service"


class EmailLabel(Enum):
    NEEDS_RESPONSE = "needs_response"
    FYI = "fyi"
    LOW_PRIORITY = "low_priority"


@dataclass
class EmailMetadata:
    message_id: str
    sender: str
    subject: str
    snippet: str


@dataclass
class ThreadMetadata:
    """Metadata for a Gmail thread (multiple messages)."""

    thread_id: str
    senders: list[str]  # All unique From headers in the thread
    subject: str  # Subject from the first message
    snippet: str  # Snippet from the thread


@dataclass
class ClassificationResult:
    sender_type: SenderType
    sender_type_raw: str
    label: EmailLabel
    label_raw: str
    sender_cot: str = ""  # Raw Stage 1 CoT (with think tags)
    label_cot: str = ""  # Raw Stage 2 CoT (with think tags)


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


_SENDER_TYPE_VALID = {"PERSON", "SERVICE"}


def _match_sender_type(text: str) -> SenderType | None:
    """Return SenderType if text starts with a known keyword, else None."""
    if text.startswith("PERSON"):
        return SenderType.PERSON
    if text.startswith("SERVICE"):
        return SenderType.SERVICE
    return None


def parse_sender_type(raw_llm_output: str) -> SenderType:
    """Parse LLM output into SenderType. Defaults to SERVICE (safe).

    Checks the full output first, then falls back to the last non-empty line
    to handle models that emit preamble text before their final answer.
    """
    cleaned = raw_llm_output.strip().upper()
    result = _match_sender_type(cleaned)
    if result is None:
        # Fall back to last non-empty line (where models put their final answer)
        last_line = cleaned.rsplit("\n", 1)[-1].strip()
        result = _match_sender_type(last_line)
    if result is None:
        result = SenderType.SERVICE
        log.warning(
            "Unexpected sender type output (interpreting as %s): %.40s",
            result.name, raw_llm_output.strip(),
        )
    return result


_EMAIL_LABEL_VALID = {"NEEDS_RESPONSE", "FYI", "LOW_PRIORITY"}


def _match_email_label(text: str) -> EmailLabel | None:
    """Return EmailLabel if text starts with a known keyword, else None."""
    if text.startswith("NEEDS_RESPONSE"):
        return EmailLabel.NEEDS_RESPONSE
    if text.startswith("FYI"):
        return EmailLabel.FYI
    if text.startswith("LOW_PRIORITY"):
        return EmailLabel.LOW_PRIORITY
    if text.startswith("UNWANTED"):
        return EmailLabel.LOW_PRIORITY  # backward compat: UNWANTED merged into LOW_PRIORITY
    return None


def parse_email_label(raw_llm_output: str) -> EmailLabel:
    """Parse LLM output into EmailLabel. Defaults to LOW_PRIORITY (safe).

    Checks the full output first, then falls back to the last non-empty line
    to handle models that emit preamble text before their final answer.
    """
    cleaned = raw_llm_output.strip().upper()
    result = _match_email_label(cleaned)
    if result is None:
        last_line = cleaned.rsplit("\n", 1)[-1].strip()
        result = _match_email_label(last_line)
    if result is None:
        result = EmailLabel.LOW_PRIORITY
        log.warning(
            "Unexpected email label output (interpreting as %s): %.40s",
            result.name, raw_llm_output.strip(),
        )
    return result


class EmailClassifier:
    """Two-tier email classifier using cloud and local LLMs."""

    def __init__(self, cloud_llm: LLMClient, local_llm: LLMClient, config: dict):
        self.cloud_llm = cloud_llm
        self.local_llm = local_llm
        self.sender_config = config["prompts"]["sender_classification"]
        self.email_config = config["prompts"]["email_classification"]
        vip_config = config.get("vip_senders", {})
        raw_addrs = os.environ.get("VIP_SENDERS", "")
        self.vip_addresses: set[str] = {addr.strip().lower() for addr in raw_addrs.split(",") if addr.strip()}
        self.vip_categories: list[str] = vip_config.get("categories", ["NEEDS_RESPONSE", "FYI"])

    def _is_vip(self, metadata: EmailMetadata | ThreadMetadata) -> bool:
        """Check if any sender in the metadata is in the VIP list."""
        if isinstance(metadata, ThreadMetadata):
            for sender in metadata.senders:
                _, email = parse_sender(sender)
                if email.lower() in self.vip_addresses:
                    return True
            return False
        _, email = parse_sender(metadata.sender)
        return email.lower() in self.vip_addresses

    def _build_email_prompt(self, category_names: list[str]) -> str:
        """Build the email classification system prompt for the given categories."""
        all_categories = self.email_config["categories"]
        lines = [self.email_config["preamble"], ""]
        for name in category_names:
            lines.append(f"- {name}: {all_categories[name]}")
        lines.append("")
        lines.append(f"Respond with ONLY one word: {', '.join(category_names)}")
        lines.append("")
        lines.append(self.email_config["postamble"])
        return "\n".join(lines)

    async def classify_sender(
        self, metadata: EmailMetadata | ThreadMetadata,
    ) -> tuple[SenderType, str, str]:
        """Stage 1: Classify sender(s) as PERSON or SERVICE.

        For ThreadMetadata: checks all unique senders. Short-circuits on VIP
        (free, no LLM call) then on first PERSON from LLM.
        Only metadata (sender, subject, snippet) is sent — never the body.

        Returns:
            (sender_type, raw_output, cot) where cot is the chain-of-thought reasoning.
        """
        # Single sender (EmailMetadata)
        if isinstance(metadata, EmailMetadata):
            if self._is_vip(metadata):
                return (SenderType.PERSON, "VIP", "")
            user_content = self.sender_config["user_template"].format(
                sender=metadata.sender,
                subject=metadata.subject,
                snippet=metadata.snippet,
            )
            raw, cot = await self.cloud_llm.complete(
                self.sender_config["system"], user_content, include_thinking=True,
            )
            return (parse_sender_type(raw), raw, cot)

        # Multiple senders (ThreadMetadata) — check VIP first (free)
        if self._is_vip(metadata):
            return (SenderType.PERSON, "VIP", "")

        # LLM-classify each unique sender, short-circuit on first PERSON
        last_raw, last_cot = "SERVICE", ""
        for sender in metadata.senders:
            user_content = self.sender_config["user_template"].format(
                sender=sender,
                subject=metadata.subject,
                snippet=metadata.snippet,
            )
            raw, cot = await self.cloud_llm.complete(
                self.sender_config["system"], user_content, include_thinking=True,
            )
            last_raw, last_cot = raw, cot
            if parse_sender_type(raw) == SenderType.PERSON:
                return (SenderType.PERSON, raw, cot)

        return (SenderType.SERVICE, last_raw, last_cot)

    async def classify_email(
        self, metadata: EmailMetadata | ThreadMetadata, body: str, sender_type: SenderType, vip: bool = False
    ) -> tuple[EmailLabel, str, str]:
        """Stage 2: Classify email content. Routes based on sender type.

        Person emails -> local LLM (privacy preserved)
        Service emails -> cloud LLM
        VIP emails use a narrowed prompt with fewer categories.

        Returns:
            (label, raw_output, cot) where cot is the chain-of-thought reasoning.
        """
        llm = self.local_llm if sender_type == SenderType.PERSON else self.cloud_llm
        category_names = self.vip_categories if vip else list(self.email_config["categories"].keys())
        system_prompt = self._build_email_prompt(category_names)

        # For threads, include all senders; for single messages, use the one sender
        if isinstance(metadata, ThreadMetadata):
            sender = ", ".join(metadata.senders) if metadata.senders else ""
        else:
            sender = metadata.sender

        user_content = self.email_config["user_template"].format(
            sender=sender,
            subject=metadata.subject,
            body=body,
        )
        raw, cot = await llm.complete(system_prompt, user_content, include_thinking=True)
        label = parse_email_label(raw)
        # Safety net: clamp VIP results to allowed categories
        if vip and label.name not in category_names:
            label = EmailLabel.FYI
        return (label, raw, cot)

    async def classify(
        self,
        metadata: EmailMetadata | ThreadMetadata,
        body: str,
        sender_type: SenderType | None = None,
        sender_type_raw: str | None = None,
    ) -> ClassificationResult:
        """Full two-stage classification pipeline.

        If sender_type is provided, Stage 1 is skipped (avoids duplicate LLM calls
        when the caller has already classified the sender).
        """
        vip = self._is_vip(metadata)
        sender_cot = ""
        if sender_type is None:
            sender_type, sender_type_raw, sender_cot = await self.classify_sender(metadata)
        label, label_raw, label_cot = await self.classify_email(metadata, body, sender_type, vip=vip)
        return ClassificationResult(
            sender_type=sender_type,
            sender_type_raw=sender_type_raw or "",
            label=label,
            label_raw=label_raw,
            sender_cot=sender_cot,
            label_cot=label_cot,
        )
