"""Tests for daemon orchestration."""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from classifier import (
    ClassificationResult,
    EmailLabel,
    SenderType,
)
from daemon import format_thread_transcript, load_config, process_single_thread
from labeler import _get_priority


@pytest.fixture
def mock_proxy():
    return AsyncMock()


@pytest.fixture
def mock_classifier():
    classifier = AsyncMock()
    classifier.classify.return_value = ClassificationResult(
        sender_type=SenderType.PERSON,
        sender_type_raw="PERSON",
        label=EmailLabel.NEEDS_RESPONSE,
        label_raw="NEEDS_RESPONSE",
    )
    classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON", "")
    return classifier


@pytest.fixture
def mock_label_manager():
    mgr = AsyncMock()
    # get_existing_priority is synchronous — use MagicMock so it returns a value, not a coroutine
    mgr.get_existing_priority = MagicMock(return_value=None)
    return mgr


@pytest.fixture
def mock_thread_response():
    """Sample thread with two messages."""
    body1 = "Hey, can we meet tomorrow at 3pm?"
    body2 = "Sure, works for me. See you then!"
    return {
        "id": "thread_001",
        "snippet": "Sure, works for me. See you then!",
        "messages": [
            {
                "id": "msg_001",
                "threadId": "thread_001",
                "internalDate": "1704067200000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "John Doe <john@example.com>"},
                        {"name": "Subject", "value": "Meeting tomorrow"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body1.encode()).decode(),
                    },
                },
            },
            {
                "id": "msg_002",
                "threadId": "thread_001",
                "internalDate": "1704070800000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "Jane Smith <jane@example.com>"},
                        {"name": "Subject", "value": "Re: Meeting tomorrow"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 13:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body2.encode()).decode(),
                    },
                },
            },
        ],
    }


class TestFormatThreadTranscript:
    def test_formats_chronologically(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        transcript = format_thread_transcript(messages, 50000)
        # John's message should come before Jane's
        john_pos = transcript.index("John Doe")
        jane_pos = transcript.index("Jane Smith")
        assert john_pos < jane_pos

    def test_includes_sender_and_date(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        transcript = format_thread_transcript(messages, 50000)
        assert "John Doe <john@example.com>" in transcript
        assert "Mon, 1 Jan 2024 12:00:00 +0000" in transcript

    def test_includes_body(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        transcript = format_thread_transcript(messages, 50000)
        assert "Hey, can we meet tomorrow at 3pm?" in transcript
        assert "Sure, works for me" in transcript

    def test_truncates_oldest_first(self, mock_thread_response):
        messages = mock_thread_response["messages"]
        # Set a very small max to force truncation
        transcript = format_thread_transcript(messages, 100)
        assert "[Earlier messages truncated]" in transcript or len(transcript) <= 100


class TestProcessSingleThread:
    async def test_classifies_thread(
        self, mock_proxy, mock_classifier, mock_label_manager, mock_thread_response
    ):
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.PERSON,
            sender_type_raw="PERSON",
            label=EmailLabel.NEEDS_RESPONSE,
            label_raw="NEEDS_RESPONSE",
        )

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            mlx_available=True,
            max_thread_chars=50000,
        )

        assert result is True
        mock_proxy.get_thread.assert_called_once_with("thread_001")
        mock_classifier.classify.assert_called_once()
        # Labels applied to ALL messages in thread
        mock_label_manager.apply_classification.assert_called_once()
        call_args = mock_label_manager.apply_classification.call_args
        assert call_args.args[0] == ["msg_001", "msg_002"]  # all message IDs

    async def test_skips_person_thread_when_mlx_unavailable(
        self, mock_proxy, mock_classifier, mock_label_manager, mock_thread_response
    ):
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON", "")

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            mlx_available=False,
            max_thread_chars=50000,
        )

        assert result is False
        mock_classifier.classify.assert_not_called()
        mock_label_manager.apply_classification.assert_not_called()

    async def test_skips_downgrade(
        self, mock_proxy, mock_classifier, mock_label_manager, mock_thread_response
    ):
        """Thread already at FYI should not be downgraded to LOW_PRIORITY."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.SERVICE,
            sender_type_raw="SERVICE",
            label=EmailLabel.LOW_PRIORITY,
            label_raw="LOW_PRIORITY",
        )
        # Existing priority = FYI (2), new = LOW_PRIORITY (1) -> skip
        mock_label_manager.get_existing_priority.return_value = _get_priority(EmailLabel.FYI)

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            mlx_available=True,
            max_thread_chars=50000,
        )

        assert result is False
        mock_label_manager.apply_classification.assert_not_called()

    async def test_allows_upgrade(
        self, mock_proxy, mock_classifier, mock_label_manager, mock_thread_response
    ):
        """Thread at FYI can be upgraded to NEEDS_RESPONSE."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.PERSON,
            sender_type_raw="PERSON",
            label=EmailLabel.NEEDS_RESPONSE,
            label_raw="NEEDS_RESPONSE",
        )
        # Existing priority = FYI (2), new = NEEDS_RESPONSE (3) -> upgrade
        mock_label_manager.get_existing_priority.return_value = _get_priority(EmailLabel.FYI)

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            mlx_available=True,
            max_thread_chars=50000,
        )

        assert result is True
        mock_label_manager.apply_classification.assert_called_once()

    async def test_error_in_processing_returns_false(self, mock_proxy, mock_classifier, mock_label_manager):
        """Errors during processing don't crash — return False."""
        mock_proxy.get_thread.side_effect = RuntimeError("API error")

        result = await process_single_thread(
            "thread_001",
            ["msg_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            mlx_available=True,
            max_thread_chars=50000,
        )

        assert result is False

    async def test_service_thread_processed_when_mlx_unavailable(
        self, mock_proxy, mock_classifier, mock_label_manager, mock_thread_response
    ):
        """Service threads still processed even when MLX is down."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.return_value = (SenderType.SERVICE, "SERVICE", "")
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.SERVICE,
            sender_type_raw="SERVICE",
            label=EmailLabel.LOW_PRIORITY,
            label_raw="LOW_PRIORITY",
        )

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            mlx_available=False,
            max_thread_chars=50000,
        )

        assert result is True
        mock_classifier.classify.assert_called_once()
        mock_label_manager.apply_classification.assert_called_once()


class TestLoadConfig:
    def test_loads_config_toml(self):
        config = load_config()
        assert "daemon" in config
        assert "labels" in config
        assert "llm" in config
        assert "prompts" in config

    def test_config_has_required_daemon_keys(self):
        config = load_config()
        assert "poll_interval_seconds" in config["daemon"]
        assert "max_emails_per_cycle" in config["daemon"]
        assert "gmail_query" in config["daemon"]
        assert "healthcheck_file" in config["daemon"]

    def test_config_has_all_labels(self):
        config = load_config()
        for key in (
            "needs_response",
            "fyi",
            "low_priority",
            "processed",
            "personal",
            "non_personal",
        ):
            assert key in config["labels"]

    def test_config_has_prompts(self):
        config = load_config()
        assert "system" in config["prompts"]["sender_classification"]
        assert "user_template" in config["prompts"]["sender_classification"]
        email_config = config["prompts"]["email_classification"]
        assert "preamble" in email_config
        assert "postamble" in email_config
        assert "categories" in email_config
        assert "user_template" in email_config

    def test_config_has_vip_senders(self):
        config = load_config()
        assert "vip_senders" in config
        assert "categories" in config["vip_senders"]

    def test_config_has_max_thread_chars(self):
        config = load_config()
        assert "max_thread_chars" in config["daemon"]
        assert isinstance(config["daemon"]["max_thread_chars"], int)
