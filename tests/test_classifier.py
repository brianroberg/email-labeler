"""Tests for email classifier — parsing functions and EmailClassifier class."""

from unittest.mock import AsyncMock

import pytest

from classifier import (
    ClassificationResult,
    EmailClassifier,
    EmailLabel,
    EmailMetadata,
    SenderType,
    parse_email_label,
    parse_sender,
    parse_sender_type,
)

# ── Pure parsing function tests ──────────────────────────────────────────


class TestParseSender:
    def test_standard_format(self):
        name, email = parse_sender("John Doe <john@example.com>")
        assert name == "John Doe"
        assert email == "john@example.com"

    def test_email_only(self):
        name, email = parse_sender("john@example.com")
        assert name == ""
        assert email == "john@example.com"

    def test_quoted_name(self):
        name, email = parse_sender('"John Doe" <john@example.com>')
        assert name == "John Doe"
        assert email == "john@example.com"

    def test_empty_string(self):
        name, email = parse_sender("")
        assert name == ""
        assert email == ""

    def test_name_with_special_chars(self):
        name, email = parse_sender("O'Brien, Mary <mary@example.com>")
        assert name == "O'Brien, Mary"
        assert email == "mary@example.com"

    def test_no_angle_brackets_no_at(self):
        name, email = parse_sender("Just A Name")
        assert name == ""
        assert email == "Just A Name"


class TestParseSenderType:
    def test_person(self):
        assert parse_sender_type("PERSON") == SenderType.PERSON

    def test_service(self):
        assert parse_sender_type("SERVICE") == SenderType.SERVICE

    def test_lowercase(self):
        assert parse_sender_type("person") == SenderType.PERSON

    def test_mixed_case(self):
        assert parse_sender_type("Service") == SenderType.SERVICE

    def test_with_whitespace(self):
        assert parse_sender_type("  PERSON  ") == SenderType.PERSON

    def test_with_trailing_text(self):
        assert parse_sender_type("PERSON. The sender is a real person.") == SenderType.PERSON

    def test_unknown_defaults_to_service(self):
        assert parse_sender_type("UNKNOWN") == SenderType.SERVICE

    def test_empty_defaults_to_service(self):
        assert parse_sender_type("") == SenderType.SERVICE

    def test_garbage_defaults_to_service(self):
        assert parse_sender_type("asdfghjkl") == SenderType.SERVICE


class TestParseEmailLabel:
    def test_needs_response(self):
        assert parse_email_label("NEEDS_RESPONSE") == EmailLabel.NEEDS_RESPONSE

    def test_fyi(self):
        assert parse_email_label("FYI") == EmailLabel.FYI

    def test_low_priority(self):
        assert parse_email_label("LOW_PRIORITY") == EmailLabel.LOW_PRIORITY

    def test_unwanted(self):
        assert parse_email_label("UNWANTED") == EmailLabel.UNWANTED

    def test_lowercase(self):
        assert parse_email_label("needs_response") == EmailLabel.NEEDS_RESPONSE

    def test_mixed_case(self):
        assert parse_email_label("Fyi") == EmailLabel.FYI

    def test_with_whitespace(self):
        assert parse_email_label("  UNWANTED  ") == EmailLabel.UNWANTED

    def test_with_trailing_text(self):
        assert parse_email_label("LOW_PRIORITY. This is a newsletter.") == EmailLabel.LOW_PRIORITY

    def test_unknown_defaults_to_low_priority(self):
        assert parse_email_label("SOMETHING_ELSE") == EmailLabel.LOW_PRIORITY

    def test_empty_defaults_to_low_priority(self):
        assert parse_email_label("") == EmailLabel.LOW_PRIORITY


# ── EmailClassifier class tests ──────────────────────────────────────────


@pytest.fixture
def mock_cloud_llm():
    return AsyncMock()


@pytest.fixture
def mock_local_llm():
    return AsyncMock()


@pytest.fixture
def config():
    return {
        "prompts": {
            "sender_classification": {
                "system": "Classify as PERSON or SERVICE.",
                "user_template": "From: {sender}\nSubject: {subject}\nPreview: {snippet}",
            },
            "email_classification": {
                "system": "Classify the email.",
                "user_template": "From: {sender}\nSubject: {subject}\nBody:\n{body}",
            },
        }
    }


@pytest.fixture
def metadata():
    return EmailMetadata(
        message_id="msg_001",
        sender="John Doe <john@example.com>",
        subject="Meeting tomorrow",
        snippet="Hey, can we meet tomorrow?",
    )


@pytest.fixture
def classifier(mock_cloud_llm, mock_local_llm, config):
    return EmailClassifier(
        cloud_llm=mock_cloud_llm,
        local_llm=mock_local_llm,
        config=config,
    )


class TestClassifySender:
    async def test_routes_to_cloud_llm(self, classifier, mock_cloud_llm, metadata):
        mock_cloud_llm.complete.return_value = "PERSON"
        sender_type, raw = await classifier.classify_sender(metadata)

        assert sender_type == SenderType.PERSON
        mock_cloud_llm.complete.assert_called_once()
        # Verify the prompt was formatted with metadata
        call_args = mock_cloud_llm.complete.call_args
        assert "John Doe <john@example.com>" in call_args.args[1]
        assert "Meeting tomorrow" in call_args.args[1]

    async def test_returns_service_for_automated(self, classifier, mock_cloud_llm, metadata):
        mock_cloud_llm.complete.return_value = "SERVICE"
        sender_type, raw = await classifier.classify_sender(metadata)
        assert sender_type == SenderType.SERVICE

    async def test_returns_raw_llm_output(self, classifier, mock_cloud_llm, metadata):
        mock_cloud_llm.complete.return_value = "PERSON"
        _, raw = await classifier.classify_sender(metadata)
        assert raw == "PERSON"


class TestClassifyEmail:
    async def test_person_routes_to_local_llm(self, classifier, mock_local_llm, metadata):
        mock_local_llm.complete.return_value = "NEEDS_RESPONSE"
        label, raw = await classifier.classify_email(
            metadata, "Hey, can we meet tomorrow?", SenderType.PERSON
        )

        assert label == EmailLabel.NEEDS_RESPONSE
        mock_local_llm.complete.assert_called_once()

    async def test_service_routes_to_cloud_llm(self, classifier, mock_cloud_llm, metadata):
        mock_cloud_llm.complete.return_value = "LOW_PRIORITY"
        label, raw = await classifier.classify_email(
            metadata, "Your order has shipped!", SenderType.SERVICE
        )

        assert label == EmailLabel.LOW_PRIORITY
        mock_cloud_llm.complete.assert_called_once()

    async def test_formats_body_in_prompt(self, classifier, mock_local_llm, metadata):
        mock_local_llm.complete.return_value = "FYI"
        body_text = "Here's the project update."
        await classifier.classify_email(metadata, body_text, SenderType.PERSON)

        call_args = mock_local_llm.complete.call_args
        assert body_text in call_args.args[1]


class TestClassifyPipeline:
    async def test_full_person_pipeline(self, classifier, mock_cloud_llm, mock_local_llm, metadata):
        mock_cloud_llm.complete.return_value = "PERSON"
        mock_local_llm.complete.return_value = "NEEDS_RESPONSE"

        result = await classifier.classify(metadata, "Can we discuss the proposal?")

        assert isinstance(result, ClassificationResult)
        assert result.sender_type == SenderType.PERSON
        assert result.label == EmailLabel.NEEDS_RESPONSE
        assert result.sender_type_raw == "PERSON"
        assert result.label_raw == "NEEDS_RESPONSE"

    async def test_full_service_pipeline(self, classifier, mock_cloud_llm, metadata):
        mock_cloud_llm.complete.side_effect = ["SERVICE", "LOW_PRIORITY"]

        result = await classifier.classify(metadata, "Your order has shipped!")

        assert result.sender_type == SenderType.SERVICE
        assert result.label == EmailLabel.LOW_PRIORITY

    async def test_cloud_llm_used_for_both_stages_of_service(
        self, classifier, mock_cloud_llm, mock_local_llm, metadata
    ):
        """Service emails use cloud LLM for both sender classification and email classification."""
        mock_cloud_llm.complete.side_effect = ["SERVICE", "UNWANTED"]

        await classifier.classify(metadata, "Buy now!")

        assert mock_cloud_llm.complete.call_count == 2
        mock_local_llm.complete.assert_not_called()

    async def test_person_uses_cloud_then_local(
        self, classifier, mock_cloud_llm, mock_local_llm, metadata
    ):
        """Person emails use cloud for sender type, local for email classification."""
        mock_cloud_llm.complete.return_value = "PERSON"
        mock_local_llm.complete.return_value = "FYI"

        await classifier.classify(metadata, "Just wanted to share this article.")

        mock_cloud_llm.complete.assert_called_once()
        mock_local_llm.complete.assert_called_once()
