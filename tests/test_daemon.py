"""Tests for daemon orchestration."""

import base64
from unittest.mock import AsyncMock

import pytest

from classifier import (
    ClassificationResult,
    EmailLabel,
    SenderType,
)
from daemon import load_config, process_single_email


@pytest.fixture
def mock_proxy():
    proxy = AsyncMock()
    # Default: return a full message
    body_text = "Hey, can we meet tomorrow?"
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    proxy.get_message.return_value = {
        "id": "msg_001",
        "snippet": "Hey, can we meet tomorrow?",
        "payload": {
            "headers": [
                {"name": "From", "value": "John Doe <john@example.com>"},
                {"name": "Subject", "value": "Meeting tomorrow"},
            ],
            "body": {"data": encoded_body},
        },
    }
    return proxy


@pytest.fixture
def mock_classifier():
    classifier = AsyncMock()
    classifier.classify.return_value = ClassificationResult(
        sender_type=SenderType.PERSON,
        sender_type_raw="PERSON",
        label=EmailLabel.NEEDS_RESPONSE,
        label_raw="NEEDS_RESPONSE",
    )
    classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON")
    return classifier


@pytest.fixture
def mock_label_manager():
    return AsyncMock()


class TestProcessSingleEmail:
    async def test_service_email_processed(self, mock_proxy, mock_classifier, mock_label_manager):
        """Service emails go through full pipeline."""
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.SERVICE,
            sender_type_raw="SERVICE",
            label=EmailLabel.LOW_PRIORITY,
            label_raw="LOW_PRIORITY",
        )

        result = await process_single_email(
            "msg_001", mock_proxy, mock_classifier, mock_label_manager, mlx_available=True
        )

        assert result is True
        mock_proxy.get_message.assert_called_once_with("msg_001")
        mock_classifier.classify.assert_called_once()
        mock_label_manager.apply_classification.assert_called_once_with(
            "msg_001", EmailLabel.LOW_PRIORITY, SenderType.SERVICE
        )

    async def test_person_email_processed_when_mlx_available(
        self, mock_proxy, mock_classifier, mock_label_manager
    ):
        """Person emails are processed when MLX is available."""
        result = await process_single_email(
            "msg_001", mock_proxy, mock_classifier, mock_label_manager, mlx_available=True
        )

        assert result is True
        mock_classifier.classify.assert_called_once()
        mock_label_manager.apply_classification.assert_called_once()

    async def test_person_email_skipped_when_mlx_unavailable(
        self, mock_proxy, mock_classifier, mock_label_manager
    ):
        """Person emails are skipped when MLX is down — privacy preserved."""
        mock_classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON")

        result = await process_single_email(
            "msg_001", mock_proxy, mock_classifier, mock_label_manager, mlx_available=False
        )

        assert result is False
        # classify_sender should be called (cloud, metadata only)
        mock_classifier.classify_sender.assert_called_once()
        # Full classify should NOT be called (would need local LLM)
        mock_classifier.classify.assert_not_called()
        # No label should be applied
        mock_label_manager.apply_classification.assert_not_called()

    async def test_service_email_processed_when_mlx_unavailable(
        self, mock_proxy, mock_classifier, mock_label_manager
    ):
        """Service emails still processed even when MLX is down."""
        mock_classifier.classify_sender.return_value = (SenderType.SERVICE, "SERVICE")
        mock_classifier.classify.return_value = ClassificationResult(
            sender_type=SenderType.SERVICE,
            sender_type_raw="SERVICE",
            label=EmailLabel.UNWANTED,
            label_raw="UNWANTED",
        )

        result = await process_single_email(
            "msg_001", mock_proxy, mock_classifier, mock_label_manager, mlx_available=False
        )

        assert result is True
        # Pre-computed sender_type should be passed to avoid duplicate LLM call
        mock_classifier.classify.assert_called_once()
        call_kwargs = mock_classifier.classify.call_args
        assert call_kwargs[0][2] == SenderType.SERVICE  # sender_type arg
        assert call_kwargs[0][3] == "SERVICE"  # sender_type_raw arg
        mock_label_manager.apply_classification.assert_called_once_with(
            "msg_001", EmailLabel.UNWANTED, SenderType.SERVICE
        )

    async def test_error_in_processing_returns_false(
        self, mock_proxy, mock_classifier, mock_label_manager
    ):
        """Errors during processing don't crash — return False."""
        mock_proxy.get_message.side_effect = RuntimeError("API error")

        result = await process_single_email(
            "msg_001", mock_proxy, mock_classifier, mock_label_manager, mlx_available=True
        )

        assert result is False


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
            "needs_response", "fyi", "low_priority", "unwanted",
            "processed", "would_have_deleted", "personal", "non_personal",
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
