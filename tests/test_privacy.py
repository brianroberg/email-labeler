"""Negative-form privacy tests (registry D2, D3 — docs/decisions.md).

The guarantee under test, stated honestly (D2): bodies of threads
*classified as* person are processed only by the local LLM; before routing,
the cloud sees only sender/subject/snippet. Stage 1 routes on metadata and
can be wrong, and unparseable Stage 1 output deliberately defaults to
SERVICE (availability first) — that default is pinned here, not "fixed".

The snippet is Gmail's body-derived preview of the latest message
(daemon.py builds it from the message's own ``snippet`` field), so these
tests never assert "no body text reaches the cloud"; they assert nothing
*beyond* sender/subject/snippet does. Every fixture body carries two
sentinel markers — one inside the first ~100 characters, one after several
hundred characters of filler — so truncated leaks (a ``body[:200]``-style
interpolation) and full-body leaks are both caught. The fixture snippet
contains neither marker.
"""

import asyncio
import base64
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classifier import EmailClassifier, EmailMetadata, SenderType, ThreadMetadata
from daemon import process_single_thread
from llm_client import LLMUnavailableError

EARLY_MARKER = "EARLY-BODY-MARKER-3c9d"
LATE_MARKER = "LATE-BODY-MARKER-7f3a"
MARKERS = (EARLY_MARKER, LATE_MARKER)
SNIPPET_TEXT = "Preview text containing neither marker"
FILLER = "filler words to push the late marker well past snippet range " * 8


def _body(salutation: str) -> str:
    """A body with an early marker (first ~100 chars) and a late one."""
    return f"{EARLY_MARKER} {salutation}\n{FILLER}\n{LATE_MARKER} regards"


def _calls_leaking_markers(mock_llm: AsyncMock) -> list:
    """Every complete() call on the mock whose args or kwargs carry a marker."""
    leaks = []
    for call in mock_llm.complete.call_args_list:
        blob = " ".join(str(a) for a in call.args) + " " + " ".join(
            str(v) for v in call.kwargs.values()
        )
        if any(marker in blob for marker in MARKERS):
            leaks.append(call)
    return leaks


CONFIG = {
    "prompts": {
        "sender_classification": {
            "system": "Classify as PERSON or SERVICE.",
            "user_template": "From: {sender}\nSubject: {subject}\nPreview: {snippet}",
        },
        "email_classification": {
            "preamble": "Classify the email into one category:",
            "postamble": "Think carefully.",
            "user_template": "From: {sender}\nSubject: {subject}\nThread transcript:\n{body}",
            "categories": {
                "NEEDS_RESPONSE": "Requires a reply.",
                "FYI": "Informational, no action needed.",
                "LOW_PRIORITY": "Low importance or unwanted.",
            },
        },
    },
    "vip_senders": {
        "categories": ["NEEDS_RESPONSE", "FYI"],
    },
}


@pytest.fixture
def mock_cloud_llm():
    llm = AsyncMock()
    llm.complete.return_value = ("SERVICE", "")  # every mock configured (tuple contract)
    return llm


@pytest.fixture
def mock_local_llm():
    llm = AsyncMock()
    llm.complete.return_value = ("FYI", "")  # every mock configured (tuple contract)
    return llm


@pytest.fixture
def classifier(mock_cloud_llm, mock_local_llm):
    with patch.dict("os.environ", {"VIP_SENDERS": "vip@example.com"}):
        return EmailClassifier(cloud_llm=mock_cloud_llm, local_llm=mock_local_llm, config=CONFIG)


@pytest.fixture
def mock_proxy():
    return AsyncMock()


@pytest.fixture
def mock_label_manager():
    mgr = AsyncMock()
    mgr.get_existing_priority = MagicMock(return_value=None)
    return mgr


@pytest.fixture
def cloud_sem():
    return asyncio.Semaphore(2)


@pytest.fixture
def local_sem():
    return asyncio.Semaphore(1)


def _message(msg_id: str, sender: str, subject: str, body: str, date: str, to: str = "") -> dict:
    headers = [
        {"name": "From", "value": sender},
        {"name": "Subject", "value": subject},
        {"name": "Date", "value": date},
    ]
    if to:
        headers.append({"name": "To", "value": to})
    return {
        "id": msg_id,
        "threadId": "thread_privacy",
        "internalDate": "1704067200000" if msg_id.endswith("1") else "1704070800000",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": headers,
            "body": {"data": base64.urlsafe_b64encode(body.encode()).decode()},
        },
    }


def _thread_dict(senders: list[str], to: str = "") -> dict:
    """Two-message thread; marker-bearing bodies; snippet on the latest message."""
    messages = [
        _message("msg_1", senders[0], "Quarterly update", _body("hello"),
                 "Mon, 1 Jan 2024 12:00:00 +0000", to=to),
        _message("msg_2", senders[-1], "Re: Quarterly update", _body("thanks"),
                 "Mon, 1 Jan 2024 13:00:00 +0000", to=to),
    ]
    messages[-1]["snippet"] = SNIPPET_TEXT  # daemon reads the latest message's snippet
    return {"id": "thread_privacy", "messages": messages}


# ── Person routing: bodies of threads classified as person stay local ────


class TestPersonRouting:
    async def test_person_thread_body_reaches_only_local_llm(
        self, classifier, mock_cloud_llm, mock_local_llm
    ):
        """Classifier level: a person-classified body goes to the local tier only."""
        metadata = ThreadMetadata(
            thread_id="t1",
            senders=["Alice Example <alice@example.com>"],
            subject="Quarterly update",
            snippet=SNIPPET_TEXT,
        )
        mock_cloud_llm.complete.side_effect = [("PERSON", "sender cot")]
        mock_local_llm.complete.return_value = ("NEEDS_RESPONSE", "label cot")

        result = await classifier.classify(metadata, _body("hello"))

        assert result.sender_type == SenderType.PERSON
        local_user = mock_local_llm.complete.call_args.args[1]
        assert all(marker in local_user for marker in MARKERS)
        assert _calls_leaking_markers(mock_cloud_llm) == []

    async def test_person_thread_end_to_end_daemon(
        self, mock_cloud_llm, mock_local_llm, classifier,
        mock_proxy, mock_label_manager, cloud_sem, local_sem,
    ):
        """Daemon level: real classifier through process_single_thread.

        Each Stage 1 cloud call is checked by whole-call equality — system
        string, rendered user_template, kwargs — so a widening of the
        pre-routing cloud payload in any slot fails, not just one carrying
        the sentinels.
        """
        senders = ["Notification Bot <bot@example.com>", "Alice Example <alice@example.com>"]
        mock_proxy.get_thread.return_value = _thread_dict(senders)
        # First sender parses SERVICE, second PERSON (exercises the loop + short-circuit).
        mock_cloud_llm.complete.side_effect = [("SERVICE", ""), ("PERSON", "cot")]
        mock_local_llm.complete.return_value = ("NEEDS_RESPONSE", "")

        result = await process_single_thread(
            "thread_privacy",
            ["msg_1", "msg_2"],
            mock_proxy,
            classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert result is True
        sender_cfg = CONFIG["prompts"]["sender_classification"]
        assert mock_cloud_llm.complete.call_count == 2
        for call, sender in zip(mock_cloud_llm.complete.call_args_list, senders):
            assert call.args[0] == sender_cfg["system"]
            assert call.args[1] == sender_cfg["user_template"].format(
                sender=sender, subject="Quarterly update", snippet=SNIPPET_TEXT
            )
            assert call.kwargs == {"include_thinking": True}
        local_user = mock_local_llm.complete.call_args.args[1]
        assert all(marker in local_user for marker in MARKERS)
        assert _calls_leaking_markers(mock_cloud_llm) == []
        mock_label_manager.apply_classification.assert_called_once()

    async def test_person_thread_local_failure_never_falls_back_to_cloud(
        self, mock_cloud_llm, mock_local_llm, classifier,
        mock_proxy, mock_label_manager, cloud_sem, local_sem,
    ):
        """A local-tier outage defers the thread; it must not reroute the body
        to the cloud (the most plausible D2 regression: a well-meaning
        resilience fallback)."""
        mock_proxy.get_thread.return_value = _thread_dict(
            ["Alice Example <alice@example.com>", "Alice Example <alice@example.com>"]
        )
        # Second element exists only so a mutated cloud-fallback path has
        # something to consume — the unmutated flow never reaches it.
        mock_cloud_llm.complete.side_effect = [("PERSON", "cot"), ("FYI", "")]
        mock_local_llm.complete.side_effect = LLMUnavailableError("local LLM down")

        result = await process_single_thread(
            "thread_privacy",
            ["msg_1", "msg_2"],
            mock_proxy,
            classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
        )

        assert _calls_leaking_markers(mock_cloud_llm) == []
        assert result is False
        mock_label_manager.apply_classification.assert_not_called()
        mock_label_manager.mark_processed.assert_not_called()


# ── Service routing: the honest other arm of the tier split ──────────────


class TestServiceRouting:
    async def test_service_thread_body_goes_to_cloud_and_local_unused(
        self, classifier, mock_cloud_llm, mock_local_llm
    ):
        """Service-classified bodies go to the cloud by design (D2 states the
        guarantee for person-classified threads only); the local tier is not
        involved."""
        metadata = ThreadMetadata(
            thread_id="t2",
            senders=["Shop Notifications <noreply@shop.example>"],
            subject="Your order",
            snippet=SNIPPET_TEXT,
        )
        mock_cloud_llm.complete.side_effect = [("SERVICE", ""), ("LOW_PRIORITY", "")]
        mock_local_llm.complete.return_value = ("FYI", "")  # configured, must stay unused

        result = await classifier.classify(metadata, _body("your order"))

        assert result.sender_type == SenderType.SERVICE
        assert mock_local_llm.complete.call_count == 0
        stage2_user = mock_cloud_llm.complete.call_args_list[1].args[1]
        assert all(marker in stage2_user for marker in MARKERS)

    async def test_unparseable_stage1_defaults_to_service_route(
        self, classifier, mock_cloud_llm, mock_local_llm
    ):
        """Pins the adjudicated D2 default: keywordless Stage 1 output routes
        as SERVICE (availability first). Do not "fix" this to PERSON — D2
        forecloses that; the residual misroute risk is measured by the eval
        suite's privacy-violation rate, not denied."""
        metadata = ThreadMetadata(
            thread_id="t3",
            senders=["Ambiguous <someone@example.com>"],
            subject="Hello",
            snippet=SNIPPET_TEXT,
        )
        mock_cloud_llm.complete.side_effect = [
            ("I think this may be a human being", ""),
            ("FYI", ""),
        ]
        mock_local_llm.complete.return_value = ("FYI", "")  # configured, must stay unused

        result = await classifier.classify(metadata, _body("hello"))

        assert result.sender_type == SenderType.SERVICE
        assert mock_local_llm.complete.call_count == 0


# ── VIP short-circuit: no cloud involvement at all ───────────────────────


class TestVipRouting:
    async def test_vip_sender_skips_cloud_entirely(
        self, classifier, mock_cloud_llm, mock_local_llm
    ):
        """A VIP sender is PERSON without a Stage 1 call: zero cloud calls,
        body only on the local tier."""
        metadata = ThreadMetadata(
            thread_id="t4",
            senders=["VIP Person <vip@example.com>"],
            subject="Catching up",
            snippet=SNIPPET_TEXT,
        )
        mock_cloud_llm.complete.return_value = ("SERVICE", "")  # configured, must stay unused
        mock_local_llm.complete.return_value = ("NEEDS_RESPONSE", "")

        result = await classifier.classify(metadata, _body("hi"))

        assert result.sender_type == SenderType.PERSON
        assert mock_cloud_llm.complete.call_count == 0
        local_user = mock_local_llm.complete.call_args.args[1]
        assert all(marker in local_user for marker in MARKERS)


# ── Newsletter ownership (D3): cloud by design, before person/service ────


class TestNewsletterOwnership:
    async def test_newsletter_thread_bypasses_stage1_and_local(
        self, mock_cloud_llm, mock_local_llm, classifier,
        mock_proxy, mock_label_manager, cloud_sem, local_sem,
    ):
        """A thread To-addressed to the newsletter recipient is organizational
        content (D3): its full transcript — person-written replies included —
        goes to the newsletter pipeline's cloud client by design, and the
        email pipeline (Stage 1, local tier) is never involved.

        The fixture address exactly equals the recipient so this test is
        unaffected by Wave 2's D3 exact-address matching change."""
        recipient = "newsletters@example.org"
        mock_proxy.get_thread.return_value = _thread_dict(
            ["Alice Example <alice@example.com>", "Bob Staff <bob@example.org>"],
            to=recipient,
        )
        newsletter_classifier = AsyncMock()
        newsletter_classifier.classify_newsletter.return_value = []

        result = await process_single_thread(
            "thread_privacy",
            ["msg_1", "msg_2"],
            mock_proxy,
            classifier,
            mock_label_manager,
            cloud_sem,
            local_sem,
            max_thread_chars=50000,
            newsletter_classifier=newsletter_classifier,
            newsletter_recipient=recipient,
        )

        assert result is True
        assert mock_cloud_llm.complete.call_count == 0
        assert mock_local_llm.complete.call_count == 0
        transcript = newsletter_classifier.classify_newsletter.call_args.args[0]
        assert all(marker in transcript for marker in MARKERS)


# ── Metadata shape guard: the pre-routing cloud payload cannot grow ──────


class TestMetadataShape:
    def test_metadata_shapes_cannot_carry_bodies(self):
        """Body text travels only as the explicit ``body`` argument of
        classify()/classify_email(); the metadata objects Stage 1 sees hold
        exactly sender/subject/snippet identity fields. Growing either class
        is a privacy-review event, and a body-carrying field is foreclosed
        (D2) — update this allowlist only with a registry-aware review."""
        assert {f.name for f in fields(ThreadMetadata)} == {
            "thread_id", "senders", "subject", "snippet",
        }
        assert {f.name for f in fields(EmailMetadata)} == {
            "message_id", "sender", "subject", "snippet",
        }
