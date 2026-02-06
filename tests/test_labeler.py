"""Tests for label manager."""

from unittest.mock import AsyncMock

import pytest

from classifier import EmailLabel, SenderType
from labeler import LabelManager, _get_priority


@pytest.fixture
def config():
    return {
        "labels": {
            "needs_response": "agent/needs-response",
            "fyi": "agent/fyi",
            "low_priority": "agent/low-priority",
            "unwanted": "agent/unwanted",
            "processed": "agent/processed",
            "would_have_deleted": "agent/would-have-deleted",
            "personal": "agent/personal",
            "non_personal": "agent/non-personal",
            "actions": {
                "needs_response": "inbox",
                "fyi": "inbox",
                "low_priority": "archive",
                "unwanted": "archive",
            },
            "extra_labels": {
                "unwanted": ["agent/would-have-deleted"],
            },
        }
    }


@pytest.fixture
def all_labels_response():
    """Gmail API response with all required labels present."""
    return {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
            {"id": "Label_2", "name": "agent/fyi", "type": "user"},
            {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
            {"id": "Label_4", "name": "agent/unwanted", "type": "user"},
            {"id": "Label_5", "name": "agent/processed", "type": "user"},
            {"id": "Label_6", "name": "agent/would-have-deleted", "type": "user"},
            {"id": "Label_7", "name": "agent/personal", "type": "user"},
            {"id": "Label_8", "name": "agent/non-personal", "type": "user"},
        ]
    }


@pytest.fixture
def mock_proxy():
    return AsyncMock()


@pytest.fixture
def label_manager(mock_proxy, config):
    return LabelManager(proxy_client=mock_proxy, config=config)


class TestVerifyLabels:
    async def test_all_labels_present(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        missing = await label_manager.verify_labels()
        assert missing == []

    async def test_some_labels_missing(self, label_manager, mock_proxy):
        mock_proxy.list_labels.return_value = {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
                {"id": "Label_5", "name": "agent/processed", "type": "user"},
            ]
        }
        missing = await label_manager.verify_labels()
        assert set(missing) == {
            "agent/fyi",
            "agent/low-priority",
            "agent/unwanted",
            "agent/would-have-deleted",
            "agent/personal",
            "agent/non-personal",
        }

    async def test_no_labels_present(self, label_manager, mock_proxy):
        mock_proxy.list_labels.return_value = {"labels": [{"id": "INBOX", "name": "INBOX", "type": "system"}]}
        missing = await label_manager.verify_labels()
        assert len(missing) == 8

    async def test_builds_label_id_map(self, label_manager, mock_proxy, all_labels_response):
        """verify_labels populates the internal label name -> ID mapping."""
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        assert label_manager.label_ids["agent/needs-response"] == "Label_1"
        assert label_manager.label_ids["agent/processed"] == "Label_5"


class TestApplyClassification:
    async def test_needs_response_stays_in_inbox(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.NEEDS_RESPONSE, SenderType.PERSON)

        mock_proxy.modify_message.assert_called_once()
        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        assert "Label_1" in call_kwargs["add_label_ids"]  # needs-response
        assert "Label_5" in call_kwargs["add_label_ids"]  # processed
        assert "Label_7" in call_kwargs["add_label_ids"]  # personal
        assert "remove_label_ids" not in call_kwargs or "INBOX" not in call_kwargs.get("remove_label_ids", [])

    async def test_fyi_stays_in_inbox(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.FYI, SenderType.PERSON)

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        assert "Label_2" in call_kwargs["add_label_ids"]  # fyi
        assert "Label_5" in call_kwargs["add_label_ids"]  # processed
        assert "Label_7" in call_kwargs["add_label_ids"]  # personal
        assert "remove_label_ids" not in call_kwargs or "INBOX" not in call_kwargs.get("remove_label_ids", [])

    async def test_low_priority_gets_archived(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.LOW_PRIORITY, SenderType.SERVICE)

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        assert "Label_3" in call_kwargs["add_label_ids"]  # low-priority
        assert "Label_5" in call_kwargs["add_label_ids"]  # processed
        assert "Label_8" in call_kwargs["add_label_ids"]  # non-personal
        assert "INBOX" in call_kwargs["remove_label_ids"]

    async def test_unwanted_gets_archived_and_extra_label(
        self, label_manager, mock_proxy, all_labels_response
    ):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.UNWANTED, SenderType.SERVICE)

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        add_ids = call_kwargs["add_label_ids"]
        assert "Label_4" in add_ids  # unwanted
        assert "Label_5" in add_ids  # processed
        assert "Label_6" in add_ids  # would-have-deleted (extra)
        assert "Label_8" in add_ids  # non-personal
        assert "INBOX" in call_kwargs["remove_label_ids"]

    async def test_single_modify_call(self, label_manager, mock_proxy, all_labels_response):
        """Each classification should result in exactly one modify_message call."""
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.UNWANTED, SenderType.SERVICE)

        assert mock_proxy.modify_message.call_count == 1


class TestPriorityOrder:
    def test_priority_ordering(self):
        assert _get_priority(EmailLabel.UNWANTED) < _get_priority(EmailLabel.LOW_PRIORITY)
        assert _get_priority(EmailLabel.LOW_PRIORITY) < _get_priority(EmailLabel.FYI)
        assert _get_priority(EmailLabel.FYI) < _get_priority(EmailLabel.NEEDS_RESPONSE)

    def test_needs_response_highest(self):
        assert _get_priority(EmailLabel.NEEDS_RESPONSE) == 3

    def test_unwanted_lowest(self):
        assert _get_priority(EmailLabel.UNWANTED) == 0


class TestBatchApplyClassification:
    async def test_applies_to_multiple_messages(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification(
            ["msg_001", "msg_002", "msg_003"], EmailLabel.NEEDS_RESPONSE, SenderType.PERSON
        )

        assert mock_proxy.modify_message.call_count == 3

    async def test_single_string_still_works(self, label_manager, mock_proxy, all_labels_response):
        """Backward compat: single string message_id still works."""
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.FYI, SenderType.PERSON)

        mock_proxy.modify_message.assert_called_once()

    async def test_batch_applies_correct_labels(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification(
            ["msg_001", "msg_002"], EmailLabel.LOW_PRIORITY, SenderType.SERVICE
        )

        # Both messages should get the same labels
        for call in mock_proxy.modify_message.call_args_list:
            kwargs = call.kwargs
            assert "Label_3" in kwargs["add_label_ids"]  # low-priority
            assert "Label_5" in kwargs["add_label_ids"]  # processed
            assert "Label_8" in kwargs["add_label_ids"]  # non-personal
            assert "INBOX" in kwargs["remove_label_ids"]


class TestGetExistingPriority:
    async def test_no_classification_labels(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        messages = [{"labelIds": ["INBOX", "UNREAD"]}]
        assert label_manager.get_existing_priority(messages) is None

    async def test_finds_existing_label(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        messages = [{"labelIds": ["INBOX", "Label_2", "Label_5"]}]  # Label_2 = fyi
        priority = label_manager.get_existing_priority(messages)
        assert priority == _get_priority(EmailLabel.FYI)

    async def test_returns_highest_priority(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        messages = [
            {"labelIds": ["Label_3"]},  # low-priority
            {"labelIds": ["Label_1"]},  # needs-response
        ]
        priority = label_manager.get_existing_priority(messages)
        assert priority == _get_priority(EmailLabel.NEEDS_RESPONSE)
