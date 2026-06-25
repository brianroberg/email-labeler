"""Tests for daemon orchestration."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import daemon
from classifier import (
    ClassificationResult,
    EmailLabel,
    SenderType,
)
from daemon import (
    FailureTracker,
    format_thread_transcript,
    load_config,
    process_single_thread,
    resolve_int_env,
    summarize_cycle,
)
from labeler import _get_priority
from llm_client import LLMClient, LLMUnavailableError
from newsletter import NewsletterTier, StoryResult
from proxy_client import ProxyAuthError, ProxyError, ProxyUnavailableError


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
def cloud_sem():
    return asyncio.Semaphore(2)


@pytest.fixture
def local_sem():
    return asyncio.Semaphore(1)


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
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
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
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_proxy.get_thread.assert_called_once_with("thread_001")
        mock_classifier.classify.assert_called_once()
        # Labels applied to ALL messages in thread
        mock_label_manager.apply_classification.assert_called_once()
        call_args = mock_label_manager.apply_classification.call_args
        assert call_args.args[0] == ["msg_001", "msg_002"]  # all message IDs

    async def test_person_thread_returns_false_when_mlx_unreachable(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """PERSON threads fail gracefully when local LLM is unreachable."""
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.return_value = (SenderType.PERSON, "PERSON", "")
        mock_classifier.classify.side_effect = httpx.ConnectError("Connection refused")

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is False
        mock_label_manager.apply_classification.assert_not_called()

    async def test_skips_downgrade(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
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
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_label_manager.apply_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_called_once_with(["msg_001", "msg_002"])

    async def test_allows_upgrade(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
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
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_label_manager.apply_classification.assert_called_once()

    async def test_error_in_processing_returns_false(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
    ):
        """Errors during processing don't crash — return False."""
        mock_proxy.get_thread.side_effect = RuntimeError("API error")

        result = await process_single_thread(
            "thread_001",
            ["msg_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is False

    async def test_gives_up_on_thread_after_repeated_failures(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """A thread that keeps failing is marked processed after max_failures, breaking the loop."""
        mock_proxy.get_thread.side_effect = RuntimeError("API error")
        tracker = FailureTracker(max_failures=3)

        results = [
            await process_single_thread(
                "thread_stuck", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, failure_tracker=tracker,
            )
            for _ in range(3)
        ]

        # First two failures: retry next cycle (not handled), nothing marked processed.
        assert results[0] is False
        assert results[1] is False
        # Third failure hits the threshold: give up — mark agent/attempted (not
        # agent/processed) so the abandoned thread is findable, and report handled.
        assert results[2] is True
        mock_label_manager.mark_attempted.assert_called_once_with(["msg_1"])
        mock_label_manager.mark_processed.assert_not_called()
        # The give-up is recorded so the cycle summary can report it distinctly.
        assert tracker.take_given_up() == ["thread_stuck"]

    async def test_connect_error_does_not_count_toward_give_up(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """An endpoint outage (ConnectError) is transient and must never trigger give-up."""
        mock_proxy.get_thread.side_effect = httpx.ConnectError("connection refused")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            result = await process_single_thread(
                "thread_down", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, failure_tracker=tracker,
            )
            assert result is False

        mock_label_manager.mark_processed.assert_not_called()

    async def test_llm_unavailable_does_not_count_toward_give_up(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """An LLM endpoint outage (LLMUnavailableError) is transient and must never give up.

        Covers review finding #1: a connect-timeout / dropped connection from a down
        local MLX server surfaces as LLMUnavailableError, which must be retried, not
        counted toward marking the thread processed.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = LLMUnavailableError("MLX endpoint down")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            result = await process_single_thread(
                "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, failure_tracker=tracker,
            )
            assert result is False

        mock_label_manager.mark_processed.assert_not_called()

    async def test_proxy_unavailable_does_not_count_toward_give_up(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """A transient api-proxy outage (ProxyUnavailableError) must defer, never give up.

        Covers issue #16: proxy-side timeouts / dropped connections / 5xx now surface
        as ProxyUnavailableError, which is transient like LLMUnavailableError — it must
        be retried next cycle, not counted toward abandoning the thread.
        """
        mock_proxy.get_thread.side_effect = ProxyUnavailableError("proxy 503")
        tracker = FailureTracker(max_failures=2)

        for _ in range(5):
            result = await process_single_thread(
                "thread_down", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, failure_tracker=tracker,
            )
            assert result is False

        mock_label_manager.mark_processed.assert_not_called()
        mock_label_manager.mark_attempted.assert_not_called()

    async def test_proxy_4xx_is_give_up_eligible(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
    ):
        """A request-specific proxy 4xx (plain ProxyError) stays give-up-eligible.

        The transient subclass defers; the base ProxyError (e.g. a 404 for a thread
        deleted between listing and fetching) should still be bounded by the
        FailureTracker so it isn't retried forever.
        """
        mock_proxy.get_thread.side_effect = ProxyError("404 not found")
        tracker = FailureTracker(max_failures=2)

        results = [
            await process_single_thread(
                "thread_gone", ["msg_1"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, failure_tracker=tracker,
            )
            for _ in range(2)
        ]

        assert results[0] is False  # first failure: retry
        assert results[1] is True   # threshold hit: give up

    async def test_give_up_marks_all_thread_messages_not_just_query_stubs(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """When a fetched thread is given up, ALL its messages are marked processed.

        Covers review finding #3: the query may return a subset of a thread's messages,
        but once get_thread succeeds the full thread is known. Give-up must mark every
        message (so the thread stops re-matching the query) — not just the query stub.
        """
        # Thread has msg_001 + msg_002, but the query only surfaced msg_001.
        mock_proxy.get_thread.return_value = mock_thread_response
        mock_classifier.classify_sender.side_effect = RuntimeError("classification boom")
        tracker = FailureTracker(max_failures=2)

        results = [
            await process_single_thread(
                "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
                cloud_sem, local_sem, max_thread_chars=16000, failure_tracker=tracker,
            )
            for _ in range(2)
        ]

        assert results[0] is False  # first failure: retry
        assert results[1] is True   # threshold hit: give up
        mock_label_manager.mark_attempted.assert_called_once_with(["msg_001", "msg_002"])

    async def test_get_thread_is_bounded_by_fetch_sem(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """get_thread runs under fetch_sem, so proxy fetches can't fan out unbounded.

        Covers review finding #5: a large max_emails_per_cycle would otherwise burst
        one simultaneous get_thread per thread (the LLM semaphores gate only the
        classify calls, not the fetch).
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        exhausted = asyncio.Semaphore(0)  # no permits available

        task = asyncio.create_task(process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000, fetch_sem=exhausted,
        ))
        await asyncio.sleep(0.05)
        # Blocked acquiring the fetch semaphore — get_thread must not have run.
        mock_proxy.get_thread.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_processes_normally_with_available_fetch_sem(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """With a permitting fetch_sem, the thread is fetched and classified normally."""
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000, fetch_sem=asyncio.Semaphore(2),
        )

        assert result is True
        mock_proxy.get_thread.assert_called_once()

    async def test_label_application_is_bounded_by_write_sem(
        self, mock_proxy, mock_classifier, mock_label_manager, cloud_sem, local_sem,
        mock_thread_response,
    ):
        """Label-application writes run under write_sem, so they can't fan out unbounded.

        Covers issue #17: the cloud/local/fetch semaphores gate reads + classify, but
        the label-application phase (modify_message via apply_classification etc.)
        previously ran with no bound, so a large max_emails_per_cycle could burst many
        concurrent writes at the api-proxy / Gmail.
        """
        mock_proxy.get_thread.return_value = mock_thread_response
        exhausted = asyncio.Semaphore(0)  # no write permits available

        task = asyncio.create_task(process_single_thread(
            "thread_001", ["msg_001"], mock_proxy, mock_classifier, mock_label_manager,
            cloud_sem, local_sem, max_thread_chars=16000,
            fetch_sem=asyncio.Semaphore(2), write_sem=exhausted,
        ))
        await asyncio.sleep(0.05)
        # Fetch + classify ran, but blocked acquiring write_sem — no label write yet.
        mock_proxy.get_thread.assert_called_once()
        mock_label_manager.apply_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_service_thread_classified_via_cloud(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        """Service threads are classified via cloud LLM regardless of local LLM state."""
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
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_classifier.classify.assert_called_once()
        mock_label_manager.apply_classification.assert_called_once()


class TestFailureTracker:
    def test_gives_up_only_at_threshold(self):
        t = FailureTracker(max_failures=3)
        assert t.should_give_up("x") is False
        t.record_failure("x")
        assert t.should_give_up("x") is False
        t.record_failure("x")
        assert t.should_give_up("x") is False
        t.record_failure("x")
        assert t.should_give_up("x") is True

    def test_clear_resets_count(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("x")
        t.record_failure("x")
        assert t.should_give_up("x") is True
        t.clear("x")
        assert t.should_give_up("x") is False

    def test_threads_tracked_independently(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("a")
        t.record_failure("a")
        t.record_failure("b")
        assert t.should_give_up("a") is True
        assert t.should_give_up("b") is False

    def test_records_and_takes_give_ups(self):
        t = FailureTracker(max_failures=1)
        assert t.take_given_up() == []
        t.record_give_up("a")
        t.record_give_up("b")
        assert t.take_given_up() == ["a", "b"]
        assert t.take_given_up() == []  # draining resets the per-cycle list

    def test_prune_evicts_counts_for_absent_threads(self):
        # A thread that fails a few times then vanishes from the query must not
        # leak its count forever (review finding #7).
        t = FailureTracker(max_failures=2)
        t.record_failure("gone")
        t.record_failure("gone")
        assert t.should_give_up("gone") is True
        t.prune({"still_here"})
        assert t.should_give_up("gone") is False  # evicted

    def test_prune_keeps_active_threads(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("active")
        t.record_failure("active")
        t.prune({"active"})
        assert t.should_give_up("active") is True  # still counted toward give-up


class TestSummarizeCycle:
    def test_counts_handled_threads_and_drains_give_ups(self):
        t = FailureTracker(max_failures=1)
        t.record_give_up("gaveup")
        items = [("ok", ["1"]), ("retry", ["2"]), ("gaveup", ["3"])]
        results = [True, False, True]  # gaveup returned True (handled via give-up)
        processed, given_up = summarize_cycle(items, results, t)
        assert processed == 2  # ok + gaveup
        assert given_up == ["gaveup"]  # a subset of processed
        assert t.take_given_up() == []  # already drained

    def test_clears_counts_for_handled_threads(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("ok")  # failed last cycle, succeeds now
        summarize_cycle([("ok", ["1"])], [True], t)
        t.record_failure("ok")
        assert t.should_give_up("ok") is False  # count was reset on success

    def test_prunes_counts_for_threads_absent_this_cycle(self):
        t = FailureTracker(max_failures=2)
        t.record_failure("stale")
        t.record_failure("stale")  # at threshold from a prior cycle
        assert t.should_give_up("stale") is True
        # This cycle only has "active"; "stale" is gone from the query.
        summarize_cycle([("active", ["1"])], [False], t)
        assert t.should_give_up("stale") is False  # pruned


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

    def test_config_has_parallel_settings(self):
        config = load_config()
        assert "cloud_parallel" in config["daemon"]
        assert "local_parallel" in config["daemon"]
        assert config["daemon"]["cloud_parallel"] >= 1
        assert config["daemon"]["local_parallel"] >= 1

    def test_config_disables_native_thinking_on_local_classifier(self):
        # Issue #10: the eval showed native thinking on the local person-email
        # classifier is strictly worse (budget-split failures where reasoning
        # overruns max_tokens and no label is emitted). The classification prompt
        # already drives reasoning into the content channel, so native thinking is
        # disabled via request-level chat_template_kwargs (the form mlx_lm.server
        # honors). The cloud classifier is unaffected.
        config = load_config()
        local = config["llm"]["local"]
        # Layout guard: the flag must be in the nested chat_template_kwargs form.
        # mlx_lm.server (the real local server) honors this form and ignores a
        # top-level enable_thinking, so this specific nesting is load-bearing — a
        # refactor to the top-level form would silently re-enable thinking on the
        # local server even though llm_client treats both as no-think.
        ctk = local["extra_body"]["chat_template_kwargs"]
        assert ctk["enable_thinking"] is False
        # Behavior guard: the LLMClient the daemon builds from this config must
        # actually treat the request as thinking-disabled.
        client = LLMClient(
            base_url="", api_key="", model=local["model"],
            extra_body=local.get("extra_body"),
        )
        assert client._extra_body_disables_thinking() is True

    def test_config_has_newsletter_section(self):
        config = load_config()
        assert "newsletter" in config
        assert "recipient" in config["newsletter"]
        assert "output_file" in config["newsletter"]
        assert "labels" in config["newsletter"]
        assert "prompts" in config["newsletter"]

    def test_config_has_newsletter_labels(self):
        config = load_config()
        nl = config["newsletter"]["labels"]
        assert "newsletter" in nl
        assert "excellent" in nl
        assert "good" in nl
        assert "fair" in nl
        assert "poor" in nl
        assert "no_stories" in nl
        assert "themes" in nl
        assert len(nl["themes"]) == 5

    def test_config_has_newsletter_prompts(self):
        config = load_config()
        prompts = config["newsletter"]["prompts"]
        assert "story_extraction" in prompts
        assert "quality_assessment" in prompts
        assert "theme_classification" in prompts
        for key in ("story_extraction", "quality_assessment", "theme_classification"):
            assert "system" in prompts[key]
            assert "user_template" in prompts[key]


class TestResolveIntEnv:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("LOCAL_PARALLEL", raising=False)
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "6")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 6

    def test_blank_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "   ")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MAX_EMAILS_PER_CYCLE", "lots")
        assert resolve_int_env("MAX_EMAILS_PER_CYCLE", 10) == 10

    def test_strips_whitespace_around_value(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "  2 ")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 2

    def test_zero_falls_back_to_default(self, monkeypatch):
        # 0 parses fine but is out of range: Semaphore(0) deadlocks the daemon.
        monkeypatch.setenv("LOCAL_PARALLEL", "0")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_negative_falls_back_to_default(self, monkeypatch):
        # -1 parses fine but is out of range: Semaphore(-1) crashes at startup.
        monkeypatch.setenv("LOCAL_PARALLEL", "-1")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 4

    def test_value_at_minimum_is_allowed(self, monkeypatch):
        monkeypatch.setenv("LOCAL_PARALLEL", "1")
        assert resolve_int_env("LOCAL_PARALLEL", 4) == 1

    def test_max_emails_zero_falls_back_to_default(self, monkeypatch):
        # The lower-bound guard protects every numeric override, not just concurrency:
        # max_results=0 would make the daemon process nothing each cycle.
        monkeypatch.setenv("MAX_EMAILS_PER_CYCLE", "0")
        assert resolve_int_env("MAX_EMAILS_PER_CYCLE", 10) == 10


@pytest.fixture
def mock_newsletter_classifier():
    return AsyncMock()


@pytest.fixture
def newsletter_thread_response():
    body = "This month's campus update features Sarah's journey..."
    return {
        "id": "thread_nl",
        "snippet": "This month's campus update...",
        "messages": [
            {
                "id": "msg_nl_001",
                "threadId": "thread_nl",
                "internalDate": "1704067200000",
                "labelIds": ["INBOX", "UNREAD"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "John Staff <john@dm.org>"},
                        {"name": "To", "value": "newsletters@dm.org"},
                        {"name": "Subject", "value": "February Campus Update"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    ],
                    "body": {
                        "data": base64.urlsafe_b64encode(body.encode()).decode(),
                    },
                },
            },
        ],
    }


class TestNewsletterRouting:
    async def test_newsletter_skips_priority_classification(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                title="Test",
                text="Content",
                scores={"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
                average_score=4.0,
                tier=NewsletterTier.EXCELLENT,
                themes=["scripture"],
            )
        ]

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
        )

        assert result is True
        mock_classifier.classify_sender.assert_not_called()
        mock_classifier.classify.assert_not_called()
        mock_newsletter_classifier.classify_newsletter.assert_called_once()
        mock_label_manager.apply_newsletter_classification.assert_called_once()

    async def test_non_newsletter_uses_priority_pipeline(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
        )

        assert result is True
        mock_classifier.classify_sender.assert_called_once()
        mock_classifier.classify.assert_called_once()
        mock_newsletter_classifier.classify_newsletter.assert_not_called()

    async def test_newsletter_no_stories(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = []

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
        )

        assert result is True
        call_kwargs = mock_label_manager.apply_newsletter_classification.call_args.kwargs
        assert call_kwargs["tier"] is None
        assert call_kwargs["themes"] == []

    async def test_newsletter_only_skips_non_newsletter(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        mock_thread_response,
    ):
        mock_proxy.get_thread.return_value = mock_thread_response

        result = await process_single_thread(
            "thread_001",
            ["msg_001", "msg_002"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
            newsletter_only=True,
        )

        assert result is False
        mock_classifier.classify_sender.assert_not_called()
        mock_classifier.classify.assert_not_called()
        mock_newsletter_classifier.classify_newsletter.assert_not_called()

    async def test_newsletter_only_still_processes_newsletters(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        mock_newsletter_classifier,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response
        mock_newsletter_classifier.classify_newsletter.return_value = [
            StoryResult(
                title="Test",
                text="Content",
                scores={"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
                average_score=4.0,
                tier=NewsletterTier.EXCELLENT,
                themes=["scripture"],
            )
        ]

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=mock_newsletter_classifier,
            newsletter_recipient="newsletters@dm.org",
            newsletter_output_file="/tmp/test.jsonl",
            newsletter_only=True,
        )

        assert result is True
        mock_newsletter_classifier.classify_newsletter.assert_called_once()
        mock_classifier.classify_sender.assert_not_called()

    async def test_newsletter_without_classifier_falls_through(
        self,
        mock_proxy,
        mock_classifier,
        mock_label_manager,
        cloud_sem,
        local_sem,
        newsletter_thread_response,
    ):
        mock_proxy.get_thread.return_value = newsletter_thread_response

        result = await process_single_thread(
            "thread_nl",
            ["msg_nl_001"],
            mock_proxy,
            mock_classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        mock_classifier.classify_sender.assert_called_once()


class TestVerifyLabelsWithRetry:
    """Startup label verification must survive a transiently-unreachable api-proxy.

    Regression guard for the daemon crash-loop. Two boot-time conditions are
    transient and must be waited out rather than crashed on:
      * the proxy is slow/down — a transport fault (ConnectError/ConnectTimeout/
        read timeout/dropped connection); and
      * the proxy is up but its Gmail backend is still warming, answering 5xx,
        which surfaces as proxy_client.ProxyError (NOT an httpx.TransportError).
    Permanent failures (a misconfigured PROXY_URL → httpx.UnsupportedProtocol, a
    bad key → ProxyAuthError, a programming error) must surface immediately so a
    real misconfiguration is not masked as a silent, endless retry.
    """

    @staticmethod
    def _label_manager(side_effect):
        label_manager = AsyncMock()
        label_manager.proxy.proxy_url = "http://proxy:8000"
        label_manager.verify_labels.side_effect = side_effect
        return label_manager

    async def test_retries_on_connect_timeout(self):
        """ConnectTimeout (proxy slow/unreachable at startup) is retried, not propagated."""
        label_manager = self._label_manager([httpx.ConnectTimeout("connect timed out"), []])

        missing = await daemon.verify_labels_with_retry(
            label_manager, initial_backoff=0, max_backoff=0,
        )

        assert missing == []
        assert label_manager.verify_labels.call_count == 2

    async def test_retries_on_connect_error(self):
        """ConnectError stays retryable (existing transient-outage behavior preserved)."""
        label_manager = self._label_manager(
            [httpx.ConnectError("connection refused"), ["agent/processed"]]
        )

        missing = await daemon.verify_labels_with_retry(
            label_manager, initial_backoff=0, max_backoff=0,
        )

        assert missing == ["agent/processed"]
        assert label_manager.verify_labels.call_count == 2

    async def test_retries_on_proxy_5xx_error(self):
        """A warming proxy answers 5xx → ProxyError; this is transient and must retry.

        Regression for the original crash-loop: ProxyError is not an
        httpx.TransportError, so a TransportError-only catch let it propagate and
        the daemon exited — the exact failure this helper exists to prevent.
        """
        label_manager = self._label_manager([ProxyError("Proxy error: 503"), []])

        missing = await daemon.verify_labels_with_retry(
            label_manager, initial_backoff=0, max_backoff=0,
        )

        assert missing == []
        assert label_manager.verify_labels.call_count == 2

    async def test_propagates_unsupported_protocol(self):
        """A misconfigured PROXY_URL (UnsupportedProtocol) is permanent — fail fast.

        UnsupportedProtocol is an httpx.TransportError subclass, so a base-class
        catch would retry it forever and mask the misconfiguration as a hang.
        """
        label_manager = self._label_manager(
            httpx.UnsupportedProtocol("Request URL has no scheme")
        )

        with pytest.raises(httpx.UnsupportedProtocol):
            await daemon.verify_labels_with_retry(
                label_manager, initial_backoff=0, max_backoff=0,
            )

        assert label_manager.verify_labels.call_count == 1

    async def test_propagates_auth_error(self):
        """A bad PROXY_API_KEY (ProxyAuthError) is permanent — surface immediately."""
        label_manager = self._label_manager(ProxyAuthError("Unauthorized"))

        with pytest.raises(ProxyAuthError):
            await daemon.verify_labels_with_retry(
                label_manager, initial_backoff=0, max_backoff=0,
            )

        assert label_manager.verify_labels.call_count == 1

    async def test_propagates_programming_error(self):
        """A non-transient programming error must surface immediately, not retry forever."""
        label_manager = self._label_manager(RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await daemon.verify_labels_with_retry(
                label_manager, initial_backoff=0, max_backoff=0,
            )

        assert label_manager.verify_labels.call_count == 1


class TestNewsletterLLMEndpoint:
    """The newsletter grader can use a different provider than the cloud classifier.

    Newsletter quality grading is configured for a Claude model
    (config.toml [newsletter.llm] model = "claude-sonnet-4-6"), but the cloud
    classification endpoint (CLOUD_LLM_URL) points at a provider that doesn't
    serve Claude (e.g. Novita) — so requesting that model there 404s. These env
    vars let the newsletter LLM target its own Claude-serving endpoint.
    """

    def test_defaults_to_cloud_endpoint(self, monkeypatch):
        """Without NEWSLETTER_LLM_*, the newsletter LLM shares the cloud endpoint."""
        monkeypatch.setenv("CLOUD_LLM_URL", "https://novita.example/v1/chat/completions")
        monkeypatch.setenv("CLOUD_LLM_API_KEY", "novita-key")
        monkeypatch.delenv("NEWSLETTER_LLM_URL", raising=False)
        monkeypatch.delenv("NEWSLETTER_LLM_API_KEY", raising=False)

        url, key = daemon.resolve_newsletter_llm_endpoint()

        assert url == "https://novita.example/v1/chat/completions"
        assert key == "novita-key"

    def test_overrides_with_newsletter_env(self, monkeypatch):
        """NEWSLETTER_LLM_* point the newsletter LLM at its own provider (e.g. Anthropic)."""
        monkeypatch.setenv("CLOUD_LLM_URL", "https://novita.example/v1/chat/completions")
        monkeypatch.setenv("CLOUD_LLM_API_KEY", "novita-key")
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "https://api.anthropic.com/v1/chat/completions")
        monkeypatch.setenv("NEWSLETTER_LLM_API_KEY", "sk-ant-newsletter")

        url, key = daemon.resolve_newsletter_llm_endpoint()

        assert url == "https://api.anthropic.com/v1/chat/completions"
        assert key == "sk-ant-newsletter"

    def test_partial_override_does_not_borrow_cloud_key(self, monkeypatch):
        """An override URL must never silently pair with the cloud provider's key.

        The override is atomic: setting NEWSLETTER_LLM_URL alone targets the new
        endpoint with an empty key (auth fails clearly) rather than the cloud key
        (which would authenticate against the wrong provider and 401 confusingly).
        """
        monkeypatch.setenv("CLOUD_LLM_URL", "https://novita.example/v1/chat/completions")
        monkeypatch.setenv("CLOUD_LLM_API_KEY", "novita-key")
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "https://api.anthropic.com/v1/chat/completions")
        monkeypatch.delenv("NEWSLETTER_LLM_API_KEY", raising=False)

        url, key = daemon.resolve_newsletter_llm_endpoint()

        assert url == "https://api.anthropic.com/v1/chat/completions"
        assert key != "novita-key"
        assert key == ""
