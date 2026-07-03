"""Tests for evals.newsletter_harvest — newsletter filtering and deduplication."""

import argparse
import json
import logging

import pytest

import evals.newsletter_harvest as harvest_mod
from daemon import format_thread_transcript
from evals.newsletter_harvest import deduplicate, harvest_newsletters, write_golden_set
from evals.newsletter_schemas import GoldenNewsletter
from newsletter import is_newsletter

RECIPIENT = "newsletters@dm.org"

CONFIG = {
    "newsletter": {"recipient": RECIPIENT},
    "daemon": {"max_thread_chars": 16000},
}


def _msg(msg_id, internal_date, to_value, from_value="ministry@example.com",
         subject="Update", body_text="Story body"):
    return {
        "id": msg_id,
        "internalDate": internal_date,
        "snippet": "snip",
        "payload": {
            "headers": [
                {"name": "From", "value": from_value},
                {"name": "To", "value": to_value},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Mon, 1 Jul 2026 10:00:00 +0000"},
            ],
            "body": {"data": ""},
        },
    }


class TestIsNewsletterGuard:
    """The harvest guard accepts a To/Cc-matching thread and rejects others."""

    def test_accepts_matching_recipient(self):
        messages = [_msg("m1", "1000", to_value=RECIPIENT)]
        assert is_newsletter(messages, RECIPIENT) is True

    def test_rejects_non_matching_recipient(self):
        messages = [_msg("m1", "1000", to_value="someone-else@dm.org")]
        assert is_newsletter(messages, RECIPIENT) is False


class TestLoadEvalConfigReuse:
    def test_reuses_email_harvest_helper(self):
        """newsletter_harvest must reuse evals.harvest.load_eval_config, not
        carry a byte-for-byte copy of it."""
        import evals.harvest as email_harvest_mod

        assert harvest_mod.load_eval_config is email_harvest_mod.load_eval_config


class TestDeduplicate:
    def _make_golden(self, thread_id: str) -> GoldenNewsletter:
        return GoldenNewsletter(
            thread_id=thread_id,
            message_id="m1",
            sender="ministry@example.com",
            subject="Update",
            body="body",
        )

    def test_no_existing_file(self, tmp_path):
        goldens = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(goldens, tmp_path / "nonexistent.jsonl")
        assert len(result) == 2

    def test_dedup_removes_existing(self, tmp_path):
        existing_file = tmp_path / "golden.jsonl"
        existing_file.write_text(
            json.dumps({"thread_id": "t1"}) + "\n"
        )
        goldens = [self._make_golden("t1"), self._make_golden("t2")]
        result = deduplicate(goldens, existing_file)
        assert {g.thread_id for g in result} == {"t2"}

    def test_skips_malformed_line(self, tmp_path):
        """A corrupt/partial line must not crash dedup."""
        existing_file = tmp_path / "golden.jsonl"
        existing_file.write_text(
            json.dumps({"thread_id": "t1"}) + "\n"
            + '{"thread_id": "t2", "body":\n'  # truncated, invalid JSON
        )
        goldens = [self._make_golden("t1"), self._make_golden("t2"), self._make_golden("t3")]
        result = deduplicate(goldens, existing_file)
        # t1 recognized as existing; malformed t2 line skipped so t2 treated as new.
        assert {g.thread_id for g in result} == {"t2", "t3"}


class _OneNewsletterProxy:
    """Fake proxy surfacing one thread addressed to the newsletter recipient."""

    def __init__(self, *args, **kwargs):
        pass

    async def list_messages(self, user_id="me", max_results=10, q=None, label_ids=None):
        return {"messages": [{"id": "m2", "threadId": "t1"}]}

    async def get_thread(self, thread_id, user_id="me", format="full"):
        return {
            "messages": [
                _msg("m1", "1000", to_value=RECIPIENT, subject="First"),
                _msg("m2", "2000", to_value=RECIPIENT, subject="Second"),
            ]
        }


class _NonMatchingProxy(_OneNewsletterProxy):
    async def get_thread(self, thread_id, user_id="me", format="full"):
        return {
            "messages": [_msg("m1", "1000", to_value="other@dm.org")]
        }


class TestHarvestNewsletters:
    async def test_seeds_empty_unreviewed_newsletter(self):
        proxy = _OneNewsletterProxy()
        goldens = await harvest_newsletters(proxy, CONFIG, max_threads=10)
        assert len(goldens) == 1
        g = goldens[0]
        assert g.stories == []
        assert g.reviewed is False
        assert g.seeded_from == ""

    async def test_body_matches_format_thread_transcript(self):
        proxy = _OneNewsletterProxy()
        goldens = await harvest_newsletters(proxy, CONFIG, max_threads=10)
        messages = [
            _msg("m1", "1000", to_value=RECIPIENT, subject="First"),
            _msg("m2", "2000", to_value=RECIPIENT, subject="Second"),
        ]
        messages.sort(key=lambda m: int(m["internalDate"]))
        expected = format_thread_transcript(messages, 16000)
        assert goldens[0].body == expected

    async def test_captures_last_message_id_and_metadata(self):
        proxy = _OneNewsletterProxy()
        goldens = await harvest_newsletters(proxy, CONFIG, max_threads=10)
        g = goldens[0]
        assert g.thread_id == "t1"
        assert g.message_id == "m2"  # last message chronologically
        assert g.sender == "ministry@example.com"
        assert g.subject == "First"  # first message subject

    async def test_query_uses_recipient(self):
        class RecordingProxy(_OneNewsletterProxy):
            last_query = None

            async def list_messages(self, user_id="me", max_results=10, q=None, label_ids=None):
                RecordingProxy.last_query = q
                return {"messages": []}

        proxy = RecordingProxy()
        await harvest_newsletters(proxy, CONFIG, max_threads=10)
        assert proxy.last_query == f"to:{RECIPIENT}"

    async def test_non_newsletter_thread_skipped(self):
        proxy = _NonMatchingProxy()
        goldens = await harvest_newsletters(proxy, CONFIG, max_threads=10)
        assert goldens == []


class TestWriteGoldenSet:
    def _make_golden(self, thread_id: str) -> GoldenNewsletter:
        return GoldenNewsletter(
            thread_id=thread_id,
            message_id="m1",
            sender="ministry@example.com",
            subject="Update",
            body="body",
        )

    def test_appends_to_existing_file(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        write_golden_set([self._make_golden("t1")], path)
        write_golden_set([self._make_golden("t2")], path)
        ids = [json.loads(line)["thread_id"] for line in path.read_text().splitlines() if line]
        assert ids == ["t1", "t2"]


def _main_args(output):
    return argparse.Namespace(
        output=str(output),
        max_threads=10,
        recipient=None,
        config=None,
        proxy_url="http://x",
    )


class TestMainDeduplicates:
    async def test_rerun_does_not_duplicate_thread(self, tmp_path, monkeypatch):
        monkeypatch.setattr(harvest_mod, "GmailProxyClient", _OneNewsletterProxy)
        monkeypatch.setattr(harvest_mod, "load_eval_config", lambda config_path=None: CONFIG)
        output = tmp_path / "golden.jsonl"
        args = _main_args(output)
        await harvest_mod.main(args)
        await harvest_mod.main(args)
        ids = [json.loads(line)["thread_id"] for line in output.read_text().splitlines() if line]
        assert ids == ["t1"]

    async def test_rerun_skips_fetching_existing_threads(self, tmp_path, monkeypatch):
        """Threads already in the golden set must be skipped BEFORE get_thread,
        so re-runs don't re-download every thread in the file."""
        fetched: list[str] = []

        class CountingProxy(_OneNewsletterProxy):
            async def get_thread(self, thread_id, user_id="me", format="full"):
                fetched.append(thread_id)
                return await super().get_thread(thread_id, user_id=user_id, format=format)

        monkeypatch.setattr(harvest_mod, "GmailProxyClient", CountingProxy)
        monkeypatch.setattr(harvest_mod, "load_eval_config", lambda config_path=None: CONFIG)
        args = _main_args(tmp_path / "golden.jsonl")
        await harvest_mod.main(args)
        await harvest_mod.main(args)
        assert fetched == ["t1"]  # second run must not re-fetch t1


class TestCliUx:
    """Help text shows defaults; the run is quiet and ends with a next-step hint."""

    def test_help_shows_output_and_max_threads_defaults(self):
        help_text = harvest_mod.build_parser().format_help()
        assert "evals/newsletter_golden_set.jsonl" in help_text
        assert "default: 50" in help_text

    async def test_main_prints_next_step_hint(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(harvest_mod, "GmailProxyClient", _OneNewsletterProxy)
        monkeypatch.setattr(harvest_mod, "load_eval_config", lambda config_path=None: CONFIG)
        output = tmp_path / "golden.jsonl"
        await harvest_mod.main(_main_args(output))
        err = capsys.readouterr().err
        assert "evals.newsletter_label" in err
        assert f"--golden-set {output}" in err

    async def test_missing_proxy_api_key_exits_with_one_line_error(
        self, tmp_path, monkeypatch, capsys,
    ):
        """GmailProxyClient raises ProxyAuthError at construction when
        PROXY_API_KEY is unset; main() must print one actionable line and
        exit 1 instead of a 20-line traceback."""
        from proxy_client import ProxyAuthError

        class _NoKeyProxy:
            def __init__(self, *a, **kw):
                raise ProxyAuthError("PROXY_API_KEY environment variable is not set")

        monkeypatch.setattr(harvest_mod, "GmailProxyClient", _NoKeyProxy)
        monkeypatch.setattr(harvest_mod, "load_eval_config", lambda config_path=None: CONFIG)
        with pytest.raises(SystemExit) as excinfo:
            await harvest_mod.main(_main_args(tmp_path / "golden.jsonl"))
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "PROXY_API_KEY" in err
        assert "Traceback" not in err

    def test_quiet_http_logging_raises_httpx_loggers_to_warning(self, monkeypatch):
        """quiet_http_logging() itself must set the httpx AND httpcore logger
        levels to WARNING so per-request INFO lines are suppressed."""
        for name in ("httpx", "httpcore"):
            monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
        harvest_mod.quiet_http_logging()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    async def test_main_silences_httpx_request_logs(self, tmp_path, monkeypatch):
        """daemon's import-time basicConfig(INFO) lets httpx spam 'HTTP Request'
        lines; main() must raise the httpx/httpcore loggers to WARNING."""
        for name in ("httpx", "httpcore"):
            monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
        monkeypatch.setattr(harvest_mod, "GmailProxyClient", _OneNewsletterProxy)
        monkeypatch.setattr(harvest_mod, "load_eval_config", lambda config_path=None: CONFIG)
        await harvest_mod.main(_main_args(tmp_path / "golden.jsonl"))
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
