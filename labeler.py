"""Label verification and application.

Manages Gmail labels for the email classification system.
All labels must be pre-created in Gmail (api-proxy blocks label creation).
"""

from classifier import EmailLabel, SenderType
from proxy_client import GmailProxyClient

# Mapping from EmailLabel enum to config key names
_LABEL_CONFIG_KEY = {
    EmailLabel.NEEDS_RESPONSE: "needs_response",
    EmailLabel.FYI: "fyi",
    EmailLabel.LOW_PRIORITY: "low_priority",
    EmailLabel.UNWANTED: "unwanted",
}


class LabelManager:
    """Manages Gmail label verification and application."""

    def __init__(self, proxy_client: GmailProxyClient, config: dict):
        self.proxy = proxy_client
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
            "needs_response", "fyi", "low_priority", "unwanted",
            "processed", "would_have_deleted", "personal", "non_personal",
        ):
            name = self.labels_config[key]
            required_names.add(name)

        missing = []
        for name in required_names:
            if name in gmail_labels:
                self.label_ids[name] = gmail_labels[name]
            else:
                missing.append(name)

        return missing

    async def apply_classification(
        self, message_id: str, label: EmailLabel, sender_type: SenderType
    ) -> None:
        """Apply classification label and action to a message.

        Applies in a single modify_message call:
        - The classification label (e.g., agent/needs-response)
        - The processed marker label (agent/processed)
        - The sender-path label (agent/personal or agent/non-personal)
        - Any extra labels (e.g., agent/would-have-deleted for unwanted)
        - Archive action (remove INBOX) if configured

        Args:
            message_id: Gmail message ID.
            label: The classification result.
            sender_type: Whether the sender was classified as PERSON or SERVICE.
        """
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

        # Add extra labels (e.g., would-have-deleted for unwanted)
        extra_label_names = self.labels_config.get("extra_labels", {}).get(config_key, [])
        for extra_name in extra_label_names:
            add_label_ids.append(self.label_ids[extra_name])

        # Determine action
        action = self.labels_config["actions"][config_key]
        kwargs = {"message_id": message_id, "add_label_ids": add_label_ids}

        if action == "archive":
            kwargs["remove_label_ids"] = ["INBOX"]

        await self.proxy.modify_message(**kwargs)
