"""Label verification and application.

Manages Gmail labels for the email classification system.
All labels must be pre-created in Gmail (api-proxy blocks label creation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from classifier import EmailLabel, SenderType
from proxy_client import GmailProxyClient

if TYPE_CHECKING:
    from newsletter import NewsletterTier

# Mapping from EmailLabel enum to config key names
_LABEL_CONFIG_KEY = {
    EmailLabel.NEEDS_RESPONSE: "needs_response",
    EmailLabel.FYI: "fyi",
    EmailLabel.LOW_PRIORITY: "low_priority",
}


# Priority ordering: higher index = higher priority. Never downgrade.
_PRIORITY_ORDER = [
    EmailLabel.LOW_PRIORITY,  # 0 — lowest
    EmailLabel.FYI,  # 1
    EmailLabel.NEEDS_RESPONSE,  # 2 — highest
]


def _get_priority(label: EmailLabel) -> int:
    """Get priority rank for a label (higher = more important)."""
    return _PRIORITY_ORDER.index(label)


class LabelManager:
    """Manages Gmail label verification and application."""

    def __init__(self, proxy_client: GmailProxyClient, config: dict):
        self.proxy = proxy_client
        self.config = config
        self.labels_config = config["labels"]
        self.label_ids: dict[str, str] = {}

    async def verify_labels(self) -> list[str]:
        """Verify all required labels exist in Gmail.

        Populates the internal label name -> ID mapping.

        Returns:
            List of missing label names (empty if all present).
        """
        response = await self.proxy.list_labels()
        gmail_labels = {label["name"]: label["id"] for label in response["labels"]}

        required_names = set()
        for key in (
            "needs_response",
            "fyi",
            "low_priority",
            "processed",
            "personal",
            "non_personal",
        ):
            name = self.labels_config[key]
            required_names.add(name)

        # Newsletter labels (if configured)
        nl_config = self.config.get("newsletter", {}).get("labels", {})
        if nl_config:
            for key in ("newsletter", "excellent", "good", "fair", "poor", "no_stories"):
                if key in nl_config:
                    required_names.add(nl_config[key])
            for theme_name in nl_config.get("themes", {}).values():
                required_names.add(theme_name)

        missing = []
        for name in required_names:
            if name in gmail_labels:
                self.label_ids[name] = gmail_labels[name]
            else:
                missing.append(name)

        return missing

    async def apply_classification(
        self, message_ids: str | list[str], label: EmailLabel, sender_type: SenderType
    ) -> None:
        """Apply classification label and action to message(s).

        Applies in one modify_message call per message:
        - The classification label (e.g., agent/needs-response)
        - The processed marker label (agent/processed)
        - The sender-path label (agent/personal or agent/non-personal)
        - Any extra labels (if configured)
        - Archive action (remove INBOX) if configured

        Note: This method applies labels unconditionally. The caller is
        responsible for checking priority (get_existing_priority) and
        skipping downgrades before calling this method.

        Args:
            message_ids: Single message ID or list of message IDs in a thread.
            label: The classification result.
            sender_type: Whether the sender was classified as PERSON or SERVICE.
        """
        # Normalize to list
        if isinstance(message_ids, str):
            ids = [message_ids]
        else:
            ids = list(message_ids)

        config_key = _LABEL_CONFIG_KEY[label]
        label_name = self.labels_config[config_key]
        processed_name = self.labels_config["processed"]
        path_key = "personal" if sender_type == SenderType.PERSON else "non_personal"
        path_name = self.labels_config[path_key]

        add_label_ids = [
            self.label_ids[label_name],
            self.label_ids[processed_name],
            self.label_ids[path_name],
        ]

        # Add extra labels (if configured)
        extra_label_names = self.labels_config.get("extra_labels", {}).get(config_key, [])
        for extra_name in extra_label_names:
            add_label_ids.append(self.label_ids[extra_name])

        # Determine action
        action = self.labels_config["actions"][config_key]
        remove_label_ids = ["INBOX"] if action == "archive" else []

        # Apply to all messages
        for msg_id in ids:
            kwargs = {"message_id": msg_id, "add_label_ids": add_label_ids}
            if remove_label_ids:
                kwargs["remove_label_ids"] = remove_label_ids
            await self.proxy.modify_message(**kwargs)

    async def apply_newsletter_classification(
        self,
        message_ids: list[str],
        tier: NewsletterTier | None,
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

    def get_existing_priority(self, thread_messages: list[dict]) -> int | None:
        """Check existing classification labels on thread messages.

        Args:
            thread_messages: List of Gmail message resources (must include labelIds).

        Returns:
            The highest existing priority rank, or None if no classification labels found.
        """
        # Build reverse lookup: label_id -> EmailLabel
        id_to_label = {}
        for label_enum, config_key in _LABEL_CONFIG_KEY.items():
            label_name = self.labels_config[config_key]
            if label_name in self.label_ids:
                id_to_label[self.label_ids[label_name]] = label_enum

        max_priority = None
        for msg in thread_messages:
            for label_id in msg.get("labelIds", []):
                if label_id in id_to_label:
                    priority = _get_priority(id_to_label[label_id])
                    if max_priority is None or priority > max_priority:
                        max_priority = priority

        return max_priority
