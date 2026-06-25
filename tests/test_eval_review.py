"""Tests for evals.review — skip/exclude actions, hotkeys, and queue selection.

The interactive prompt loops are driven by monkeypatching ``get_hotkey`` with a
queued key sequence; menu legends are captured via ``capsys``.
"""

import sys

import pytest

import evals.review as review
from evals.review import (
    load_golden_set,
    review_thread_blind,
    review_thread_normal,
    select_review_threads,
)
from evals.schemas import GoldenThread


def _golden(thread_id, **kw):
    base = dict(
        thread_id=thread_id, messages=[{"payload": {"headers": []}}], senders=["a@b.com"],
        subject="Subj", snippet="snip", expected_sender_type="person", expected_label="fyi",
    )
    base.update(kw)
    return GoldenThread(**base)


@pytest.fixture
def keyqueue(monkeypatch):
    """Queue keypresses returned by ``get_hotkey`` in order."""

    def _install(keys):
        seq = list(keys)
        monkeypatch.setattr(review, "get_hotkey", lambda: seq.pop(0))
        return seq

    return _install


# ---------------------------------------------------------------------------
# Legends: skip/exclude/notes must appear in every prompt
# ---------------------------------------------------------------------------

class TestLegends:
    def test_normal_menu_lists_skip_and_exclude(self, keyqueue, capsys):
        keyqueue(["k"])
        review_thread_normal(_golden("t"), 0, 1)
        out = capsys.readouterr().out
        assert "[k] skip" in out
        assert "[e] exclude" in out

    def test_blind_sender_prompt_lists_skip_and_exclude(self, keyqueue, capsys):
        keyqueue(["k"])
        review_thread_blind(_golden("t"), 0, 1, stage=1)
        out = capsys.readouterr().out
        assert "[k] skip" in out
        assert "[e] exclude" in out

    def test_blind_label_prompt_lists_skip_exclude_notes_and_r(self, keyqueue, capsys):
        keyqueue(["k"])
        review_thread_blind(_golden("t"), 0, 1, stage=2)
        out = capsys.readouterr().out
        assert "[k] skip" in out
        assert "[e] exclude" in out
        assert "[n] notes" in out
        assert "[r] needs_response" in out


# ---------------------------------------------------------------------------
# Skip (temporary) — no persisted state
# ---------------------------------------------------------------------------

class TestSkip:
    def test_normal_skip_leaves_no_judgment(self, keyqueue):
        keyqueue(["k"])
        t = _golden("t")
        result = review_thread_normal(t, 0, 1)
        assert result == "advance"
        assert t.reviewed is False
        assert t.excluded is False  # skip mutates nothing on the thread

    def test_blind_skip_leaves_no_judgment(self, keyqueue):
        keyqueue(["k"])
        t = _golden("t")
        result = review_thread_blind(t, 0, 1, stage=2)
        assert result == "advance"
        assert t.reviewed is False
        assert t.excluded is False


# ---------------------------------------------------------------------------
# Exclude (permanent)
# ---------------------------------------------------------------------------

class TestExclude:
    def test_normal_exclude_marks_excluded(self, keyqueue):
        keyqueue(["e"])
        t = _golden("t")
        result = review_thread_normal(t, 0, 1)
        assert result == "advance"
        assert t.excluded is True

    def test_blind_exclude_marks_excluded(self, keyqueue):
        keyqueue(["e"])
        t = _golden("t")
        result = review_thread_blind(t, 0, 1, stage=1)
        assert result == "advance"
        assert t.excluded is True


# ---------------------------------------------------------------------------
# needs_response moved to `r`; notes available in stage-2 label prompt
# ---------------------------------------------------------------------------

class TestLabelKeysAndNotes:
    def test_r_selects_needs_response(self, keyqueue):
        keyqueue(["r"])
        t = _golden("t")
        result = review_thread_blind(t, 0, 1, stage=2)
        assert result == "advance"
        assert t.expected_label == "needs_response"

    def test_n_records_note_in_label_prompt(self, keyqueue, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _="": "not a good test case")
        keyqueue(["n", "r"])  # add note, then classify to advance
        t = _golden("t")
        result = review_thread_blind(t, 0, 1, stage=2)
        assert result == "advance"
        assert t.notes == "not a good test case"
        assert t.expected_label == "needs_response"


# ---------------------------------------------------------------------------
# review_loop undo — one snapshot per thread, regardless of action
# ---------------------------------------------------------------------------

class TestReviewLoopUndo:
    def test_undo_after_skip_returns_to_skipped_thread_not_earlier(self, keyqueue, capsys):
        # Confirm t0, skip t1, then press undo on t2.  Undo must return to t1
        # (the skipped thread), NOT rewind past it and silently revert t0's
        # confirmed judgment.
        threads = [_golden("t0"), _golden("t1"), _golden("t2")]
        keyqueue(["", "k", "z", "q"])
        review.review_loop(threads, blind=False)
        assert threads[0].reviewed is True   # t0's confirmation survives
        assert threads[1].reviewed is False  # t1 was only skipped, no judgment
        assert "Back to thread 2/3" in capsys.readouterr().out  # cursor went to t1

    def test_blind_single_undo_fully_reverts_one_classification(self, keyqueue):
        # In blind full mode a thread is classified in two steps (sender, label).
        # A single undo must revert BOTH, leaving no half-applied sender behind.
        threads = [_golden("t0"), _golden("t1")]
        keyqueue(["s", "r", "z", "q"])  # t0: service/needs_response, then undo on t1
        review.review_loop(threads, blind=True)
        t0 = threads[0]
        assert t0.expected_sender_type == "person"   # reverted from service
        assert t0.expected_label == "fyi"            # reverted from needs_response
        assert t0.reviewed is False

    def test_blind_undo_at_label_step_reverts_current_and_steps_back(
        self, keyqueue, monkeypatch, capsys
    ):
        monkeypatch.setattr("builtins.input", lambda _="": "scratch")
        threads = [_golden("t0"), _golden("t1")]
        # t0: classify person/fyi.  t1: pick sender, jot a note, then undo.
        keyqueue(["p", "f", "p", "n", "z", "q"])
        review.review_loop(threads, blind=True)
        t1 = threads[1]
        # Undo reverts t1 entirely (no orphaned note, no half-set sender) ...
        assert t1.notes == ""
        assert t1.reviewed is False
        # ... and steps back to the previous decision (t0).
        assert "Back to thread 1/2" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Queue selection — excluded threads are never reviewed
# ---------------------------------------------------------------------------

class TestSelectReviewThreads:
    def test_excludes_excluded_threads(self):
        threads = [_golden("keep"), _golden("drop", excluded=True)]
        selected = select_review_threads(threads)
        assert [t.thread_id for t in selected] == ["keep"]

    def test_excludes_excluded_even_with_unreviewed_only(self):
        threads = [
            _golden("keep", reviewed=False),
            _golden("drop", reviewed=False, excluded=True),
        ]
        selected = select_review_threads(threads, unreviewed_only=True)
        assert [t.thread_id for t in selected] == ["keep"]

    def test_filter_label_still_applies(self):
        threads = [
            _golden("a", expected_label="fyi"),
            _golden("b", expected_label="needs_response"),
        ]
        selected = select_review_threads(threads, filter_label="needs_response")
        assert [t.thread_id for t in selected] == ["b"]


# ---------------------------------------------------------------------------
# cli() integration — excluded threads stay in the file, never queued
# ---------------------------------------------------------------------------

class TestCliPreservesExcluded:
    def test_excluded_thread_not_queued_and_preserved_on_save(self, tmp_path, monkeypatch):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "".join(__import__("json").dumps(t.to_dict()) + "\n" for t in [
                _golden("normal", reviewed=False),
                _golden("excluded", reviewed=True, excluded=True),
            ])
        )

        seen = {}

        def fake_review_loop(threads, **kwargs):
            seen["ids"] = [t.thread_id for t in threads]
            for t in threads:
                t.reviewed = True

        monkeypatch.setattr(review, "review_loop", fake_review_loop)
        monkeypatch.setattr(sys, "argv", ["review", "--golden-set", str(path)])
        review.cli()

        # Excluded thread was never handed to the review loop.
        assert seen["ids"] == ["normal"]
        # Both threads remain in the file; exclusion preserved.
        saved = {t.thread_id: t for t in load_golden_set(path)}
        assert set(saved) == {"normal", "excluded"}
        assert saved["excluded"].excluded is True
        assert saved["normal"].reviewed is True

    def test_duplicate_thread_ids_not_collapsed_on_filtered_save(self, tmp_path, monkeypatch):
        # Two distinct rows sharing a thread_id, plus an excluded thread that
        # reduces the review queue (triggering the merge-back save path).  The
        # old thread_id-keyed merge collapsed the duplicates into one row; a
        # direct save of the in-memory set must keep both, each with its own
        # label.
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "".join(__import__("json").dumps(t.to_dict()) + "\n" for t in [
                _golden("dup", expected_label="fyi", reviewed=False),
                _golden("dup", expected_label="needs_response", reviewed=False),
                _golden("excluded", reviewed=True, excluded=True),
            ])
        )

        def fake_review_loop(threads, **kwargs):
            for t in threads:
                t.reviewed = True

        monkeypatch.setattr(review, "review_loop", fake_review_loop)
        monkeypatch.setattr(sys, "argv", ["review", "--golden-set", str(path)])
        review.cli()

        saved = load_golden_set(path)
        assert len(saved) == 3  # nothing collapsed or dropped
        dup_labels = sorted(t.expected_label for t in saved if t.thread_id == "dup")
        assert dup_labels == ["fyi", "needs_response"]  # both judgments kept
        assert all(t.reviewed for t in saved if t.thread_id == "dup")
