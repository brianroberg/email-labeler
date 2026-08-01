"""Tests for label manager."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from classifier import EmailLabel, SenderType
from labeler import LabelManager, _get_priority
from newsletter import NewsletterTier


@pytest.fixture
def config():
    return {
        "labels": {
            "needs_response": "agent/needs-response",
            "fyi": "agent/fyi",
            "low_priority": "agent/low-priority",
            "processed": "agent/processed",
            "attempted": "agent/attempted",
            "personal": "agent/personal",
            "non_personal": "agent/non-personal",
            "actions": {
                "needs_response": "inbox",
                "fyi": "inbox",
                "low_priority": "archive",
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
            {"id": "Label_4", "name": "agent/processed", "type": "user"},
            {"id": "Label_5", "name": "agent/personal", "type": "user"},
            {"id": "Label_6", "name": "agent/non-personal", "type": "user"},
            {"id": "Label_7", "name": "agent/attempted", "type": "user"},
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
                {"id": "Label_4", "name": "agent/processed", "type": "user"},
            ]
        }
        missing = await label_manager.verify_labels()
        assert set(missing) == {
            "agent/fyi",
            "agent/low-priority",
            "agent/attempted",
            "agent/personal",
            "agent/non-personal",
        }

    async def test_no_labels_present(self, label_manager, mock_proxy):
        mock_proxy.list_labels.return_value = {"labels": [{"id": "INBOX", "name": "INBOX", "type": "system"}]}
        missing = await label_manager.verify_labels()
        assert len(missing) == 7

    async def test_builds_label_id_map(self, label_manager, mock_proxy, all_labels_response):
        """verify_labels populates the internal label name -> ID mapping."""
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        assert label_manager.label_ids["agent/needs-response"] == "Label_1"
        assert label_manager.label_ids["agent/processed"] == "Label_4"


class TestApplyClassification:
    async def test_needs_response_stays_in_inbox(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.NEEDS_RESPONSE, SenderType.PERSON)

        mock_proxy.modify_message.assert_called_once()
        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        assert "Label_1" in call_kwargs["add_label_ids"]  # needs-response
        assert "Label_4" in call_kwargs["add_label_ids"]  # processed
        assert "Label_5" in call_kwargs["add_label_ids"]  # personal
        assert "remove_label_ids" not in call_kwargs or "INBOX" not in call_kwargs.get("remove_label_ids", [])

    async def test_fyi_stays_in_inbox(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.FYI, SenderType.PERSON)

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        assert "Label_2" in call_kwargs["add_label_ids"]  # fyi
        assert "Label_4" in call_kwargs["add_label_ids"]  # processed
        assert "Label_5" in call_kwargs["add_label_ids"]  # personal
        assert "remove_label_ids" not in call_kwargs or "INBOX" not in call_kwargs.get("remove_label_ids", [])

    async def test_low_priority_gets_archived(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.LOW_PRIORITY, SenderType.SERVICE)

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        assert "Label_3" in call_kwargs["add_label_ids"]  # low-priority
        assert "Label_4" in call_kwargs["add_label_ids"]  # processed
        assert "Label_6" in call_kwargs["add_label_ids"]  # non-personal
        assert "INBOX" in call_kwargs["remove_label_ids"]

    async def test_single_modify_call(self, label_manager, mock_proxy, all_labels_response):
        """Each classification should result in exactly one modify_message call."""
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        mock_proxy.modify_message.return_value = {"id": "msg_001"}
        await label_manager.apply_classification("msg_001", EmailLabel.LOW_PRIORITY, SenderType.SERVICE)

        assert mock_proxy.modify_message.call_count == 1


class TestPriorityOrder:
    def test_priority_ordering(self):
        assert _get_priority(EmailLabel.LOW_PRIORITY) < _get_priority(EmailLabel.FYI)
        assert _get_priority(EmailLabel.FYI) < _get_priority(EmailLabel.NEEDS_RESPONSE)

    def test_needs_response_highest(self):
        assert _get_priority(EmailLabel.NEEDS_RESPONSE) == 2

    def test_low_priority_lowest(self):
        assert _get_priority(EmailLabel.LOW_PRIORITY) == 0


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
            assert "Label_4" in kwargs["add_label_ids"]  # processed
            assert "Label_6" in kwargs["add_label_ids"]  # non-personal
            assert "INBOX" in kwargs["remove_label_ids"]


class TestWriteSemaphore:
    """LabelManager owns the write semaphore (issue #33): each modify_message
    call acquires one slot, so `write_parallel` bounds writes in flight, not
    threads writing — the slot is released between a thread's messages.

    Every write path is covered, not just apply_classification: the marker
    writes (mark_processed/mark_attempted) and the newsletter write reach the
    proxy through the same ``_modify`` helper, and they are the ones that run
    under the 300 s human-approval WRITE_TIMEOUT most often (a give-up storm is
    all markers). A path that reached ``self.proxy.modify_message`` directly
    would silently lose the bound.
    """

    @staticmethod
    def _instrumented_modify(order, counters):
        """A modify_message stub recording call order and peak concurrency."""

        async def modify(message_id, **kwargs):
            counters["in_flight"] += 1
            counters["max_in_flight"] = max(counters["max_in_flight"], counters["in_flight"])
            order.append(message_id)
            await asyncio.sleep(0)  # hold the slot across a scheduling point
            counters["in_flight"] -= 1
            return {"id": message_id}

        return modify

    async def test_concurrent_applies_interleave_per_message(
        self, mock_proxy, config, all_labels_response
    ):
        order: list[str] = []
        in_flight = 0
        max_in_flight = 0

        async def modify(message_id, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            order.append(message_id)
            await asyncio.sleep(0)  # hold the slot across a scheduling point
            in_flight -= 1
            return {"id": message_id}

        manager = LabelManager(
            proxy_client=mock_proxy, config=config, write_sem=asyncio.Semaphore(1)
        )
        mock_proxy.list_labels.return_value = all_labels_response
        await manager.verify_labels()
        mock_proxy.modify_message.side_effect = modify

        await asyncio.gather(
            manager.apply_classification(
                ["a1", "a2"], EmailLabel.NEEDS_RESPONSE, SenderType.PERSON
            ),
            manager.apply_classification(
                ["b1", "b2"], EmailLabel.FYI, SenderType.SERVICE
            ),
        )

        assert sorted(order) == ["a1", "a2", "b1", "b2"]
        # The Semaphore(1) bound held per write...
        assert max_in_flight == 1
        # ...and the slot was released between messages: the second apply's
        # first write ran before the first apply's second write. Holding the
        # semaphore across a whole apply would force all a* before any b*.
        assert order.index("b1") < order.index("a2")

    async def test_concurrent_marker_writes_interleave_per_message(
        self, mock_proxy, config, all_labels_response
    ):
        """The marker path (mark_processed/mark_attempted, via _apply_marker) is
        bounded the same way. These are the writes a give-up storm issues most
        of, and they block on the same 300 s human-approval WRITE_TIMEOUT."""
        order: list[str] = []
        counters = {"in_flight": 0, "max_in_flight": 0}

        manager = LabelManager(
            proxy_client=mock_proxy, config=config, write_sem=asyncio.Semaphore(1)
        )
        mock_proxy.list_labels.return_value = all_labels_response
        await manager.verify_labels()
        mock_proxy.modify_message.side_effect = self._instrumented_modify(order, counters)

        await asyncio.gather(
            manager.mark_processed(["a1", "a2"]),
            manager.mark_attempted(["b1", "b2"]),
        )

        assert sorted(order) == ["a1", "a2", "b1", "b2"]
        assert counters["max_in_flight"] == 1
        assert order.index("b1") < order.index("a2")

    async def test_concurrent_newsletter_writes_interleave_per_message(
        self, mock_proxy, newsletter_config, all_labels_with_newsletter
    ):
        """And the newsletter write path, which builds its own label list rather
        than sharing apply_classification's."""
        order: list[str] = []
        counters = {"in_flight": 0, "max_in_flight": 0}

        manager = LabelManager(
            proxy_client=mock_proxy,
            config=newsletter_config,
            write_sem=asyncio.Semaphore(1),
        )
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await manager.verify_labels()
        mock_proxy.modify_message.side_effect = self._instrumented_modify(order, counters)

        await asyncio.gather(
            manager.apply_newsletter_classification(
                ["a1", "a2"], NewsletterTier.EXCELLENT, {"scripture": "emphasized"}
            ),
            manager.apply_newsletter_classification(
                ["b1", "b2"], NewsletterTier.POOR, {}
            ),
        )

        assert sorted(order) == ["a1", "a2", "b1", "b2"]
        assert counters["max_in_flight"] == 1
        assert order.index("b1") < order.index("a2")


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


class TestMarkProcessed:
    async def test_applies_only_processed_label(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        await label_manager.mark_processed(["msg1", "msg2"])
        assert mock_proxy.modify_message.call_count == 2
        for call in mock_proxy.modify_message.call_args_list:
            assert call.kwargs["add_label_ids"] == ["Label_4"]  # agent/processed
            assert "remove_label_ids" not in call.kwargs

    async def test_single_message_id(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        await label_manager.mark_processed("msg1")
        mock_proxy.modify_message.assert_called_once_with(
            message_id="msg1", add_label_ids=["Label_4"],
        )


class TestMarkAttempted:
    async def test_applies_only_attempted_label(self, label_manager, mock_proxy, all_labels_response):
        """Give-up applies agent/attempted (not agent/processed), so abandoned
        threads are findable in Gmail and distinct from successfully-processed mail."""
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        await label_manager.mark_attempted(["msg1", "msg2"])
        assert mock_proxy.modify_message.call_count == 2
        for call in mock_proxy.modify_message.call_args_list:
            assert call.kwargs["add_label_ids"] == ["Label_7"]  # agent/attempted
            assert "remove_label_ids" not in call.kwargs

    async def test_single_message_id(self, label_manager, mock_proxy, all_labels_response):
        mock_proxy.list_labels.return_value = all_labels_response
        await label_manager.verify_labels()

        await label_manager.mark_attempted("msg1")
        mock_proxy.modify_message.assert_called_once_with(
            message_id="msg1", add_label_ids=["Label_7"],
        )


@pytest.fixture
def newsletter_config():
    return {
        "labels": {
            "needs_response": "agent/needs-response",
            "fyi": "agent/fyi",
            "low_priority": "agent/low-priority",
            "processed": "agent/processed",
            "attempted": "agent/attempted",
            "personal": "agent/personal",
            "non_personal": "agent/non-personal",
            "actions": {
                "needs_response": "inbox",
                "fyi": "inbox",
                "low_priority": "archive",
            },
        },
        "newsletter": {
            "labels": {
                "newsletter": "agent/newsletter",
                "excellent": "agent/newsletter/excellent",
                "good": "agent/newsletter/good",
                "fair": "agent/newsletter/fair",
                "poor": "agent/newsletter/poor",
                "no_stories": "agent/newsletter/no-stories",
                "themes": {
                    "scripture": "agent/newsletter/theme/scripture",
                    "christlikeness": "agent/newsletter/theme/christlikeness",
                    "church": "agent/newsletter/theme/church",
                    "vocation_family": "agent/newsletter/theme/vocation-family",
                    "disciple_making": "agent/newsletter/theme/disciple-making",
                },
            },
        },
    }


@pytest.fixture
def all_labels_with_newsletter():
    return {
        "labels": [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
            {"id": "Label_2", "name": "agent/fyi", "type": "user"},
            {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
            {"id": "Label_4", "name": "agent/processed", "type": "user"},
            {"id": "Label_5", "name": "agent/personal", "type": "user"},
            {"id": "Label_6", "name": "agent/non-personal", "type": "user"},
            {"id": "Label_7", "name": "agent/attempted", "type": "user"},
            {"id": "Label_10", "name": "agent/newsletter", "type": "user"},
            {"id": "Label_11", "name": "agent/newsletter/excellent", "type": "user"},
            {"id": "Label_12", "name": "agent/newsletter/good", "type": "user"},
            {"id": "Label_13", "name": "agent/newsletter/fair", "type": "user"},
            {"id": "Label_14", "name": "agent/newsletter/poor", "type": "user"},
            {"id": "Label_15", "name": "agent/newsletter/no-stories", "type": "user"},
            {"id": "Label_20", "name": "agent/newsletter/theme/scripture", "type": "user"},
            {"id": "Label_21", "name": "agent/newsletter/theme/christlikeness", "type": "user"},
            {"id": "Label_22", "name": "agent/newsletter/theme/church", "type": "user"},
            {"id": "Label_23", "name": "agent/newsletter/theme/vocation-family", "type": "user"},
            {"id": "Label_24", "name": "agent/newsletter/theme/disciple-making", "type": "user"},
        ]
    }


@pytest.fixture
def newsletter_label_manager(mock_proxy, newsletter_config):
    return LabelManager(proxy_client=mock_proxy, config=newsletter_config)


class TestNewsletterVerifyLabels:
    async def test_all_newsletter_labels_present(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        missing = await newsletter_label_manager.verify_labels()
        assert missing == []

    async def test_missing_newsletter_labels_detected(self, newsletter_label_manager, mock_proxy):
        mock_proxy.list_labels.return_value = {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {"id": "Label_1", "name": "agent/needs-response", "type": "user"},
                {"id": "Label_2", "name": "agent/fyi", "type": "user"},
                {"id": "Label_3", "name": "agent/low-priority", "type": "user"},
                {"id": "Label_4", "name": "agent/processed", "type": "user"},
                {"id": "Label_5", "name": "agent/personal", "type": "user"},
                {"id": "Label_6", "name": "agent/non-personal", "type": "user"},
            ]
        }
        missing = await newsletter_label_manager.verify_labels()
        assert "agent/newsletter" in missing
        assert "agent/newsletter/excellent" in missing
        assert "agent/attempted" in missing
        assert len(missing) == 12


class TestNewsletterApplyLabels:
    async def test_apply_newsletter_excellent(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001"],
            tier=NewsletterTier.EXCELLENT,
            themes={"scripture": "emphasized", "christlikeness": "present"},
        )

        mock_proxy.modify_message.assert_called_once()
        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        add_ids = call_kwargs["add_label_ids"]
        assert "Label_4" in add_ids  # processed
        assert "Label_10" in add_ids  # newsletter marker
        assert "Label_11" in add_ids  # excellent
        assert "Label_20" in add_ids  # theme/scripture (emphasized -> labeled)
        # christlikeness is only "present", not "emphasized" — issue #53 applies the
        # Gmail theme label ONLY for emphasized themes.
        assert "Label_21" not in add_ids
        assert "INBOX" in call_kwargs["remove_label_ids"]

    async def test_present_theme_not_labeled(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001"],
            tier=NewsletterTier.GOOD,
            themes={"scripture": "present", "church": "present"},
        )

        add_ids = mock_proxy.modify_message.call_args.kwargs["add_label_ids"]
        assert "Label_20" not in add_ids  # theme/scripture not emphasized
        assert "Label_22" not in add_ids  # theme/church not emphasized

    async def test_apply_newsletter_no_stories(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001"],
            tier=None,
            themes={},
        )

        call_kwargs = mock_proxy.modify_message.call_args.kwargs
        add_ids = call_kwargs["add_label_ids"]
        assert "Label_4" in add_ids  # processed
        assert "Label_10" in add_ids  # newsletter marker
        assert "Label_15" in add_ids  # no-stories
        assert "INBOX" in call_kwargs["remove_label_ids"]

    async def test_apply_to_multiple_messages(
        self, newsletter_label_manager, mock_proxy, all_labels_with_newsletter
    ):
        mock_proxy.list_labels.return_value = all_labels_with_newsletter
        await newsletter_label_manager.verify_labels()
        mock_proxy.modify_message.return_value = {"id": "msg_001"}

        await newsletter_label_manager.apply_newsletter_classification(
            message_ids=["msg_001", "msg_002"],
            tier=NewsletterTier.GOOD,
            themes={"church": "emphasized"},
        )
        assert mock_proxy.modify_message.call_count == 2
