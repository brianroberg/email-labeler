"""Tests for evals.review — session/undo semantics, queue selection, Pilot UI.

The pure ReviewSession (cursor + one-snapshot-per-thread undo) is tested
directly; the Textual UI layer is driven with Textual's Pilot: real key
presses in, thread mutations and rendered legends/statuses out.
"""

import sys

import evals.review as review
from evals.review import (
    ReviewApp,
    ReviewSession,
    load_golden_set,
    select_review_threads,
)
from evals.schemas import GoldenThread
from tui_common import PromptLineScreen


def _golden(thread_id, **kw):
    base = dict(
        thread_id=thread_id, messages=[{"payload": {"headers": []}}], senders=["a@b.com"],
        subject="Subj", snippet="snip", expected_sender_type="person", expected_label="fyi",
    )
    base.update(kw)
    return GoldenThread(**base)


SIZE = (100, 30)


def _screen_text(app) -> str:
    from textual.widgets import Static

    return "\n".join(str(w.render()) for w in app.screen.query(Static))


def _status(app) -> str:
    from textual.widgets import Static

    return str(app.query_one("#status", Static).render())


# ---------------------------------------------------------------------------
# ReviewSession: cursor + one-snapshot-per-thread undo (pure)
# ---------------------------------------------------------------------------

class TestReviewSession:
    def test_advance_pushes_one_snapshot_and_moves_cursor(self):
        threads = [_golden("t0"), _golden("t1")]
        session = ReviewSession(threads)
        threads[0].reviewed = True
        session.advance()
        assert session.index == 1
        assert len(session.undo_stack) == 1
        assert session.undo_stack[0]["reviewed"] is False  # pre-review state

    def test_undo_restores_previous_thread_and_steps_back(self):
        threads = [_golden("t0"), _golden("t1")]
        session = ReviewSession(threads)
        threads[0].expected_label = "needs_response"
        threads[0].reviewed = True
        session.advance()
        msg = session.undo()
        assert session.index == 0
        assert threads[0].expected_label == "fyi"
        assert threads[0].reviewed is False
        assert "Back to thread 1/2" in msg

    def test_undo_discards_in_progress_edits_first(self):
        threads = [_golden("t0")]
        session = ReviewSession(threads)
        threads[0].notes = "half-typed"
        msg = session.undo()
        assert threads[0].notes == ""  # entry snapshot restored
        assert "Nothing to undo" in msg


# ---------------------------------------------------------------------------
# Legends: skip/exclude/notes must appear in every prompt
# ---------------------------------------------------------------------------

class TestLegends:
    async def test_normal_menu_lists_skip_and_exclude(self, tmp_path):
        app = ReviewApp([_golden("t1")], blind=False)
        async with app.run_test(size=SIZE):
            text = _screen_text(app)
            assert "[k] skip" in text
            assert "[e] exclude" in text
            assert "[n] notes" in text
            assert "[] confirm" in text

    async def test_blind_sender_prompt_lists_skip_and_exclude(self, tmp_path):
        app = ReviewApp([_golden("t1")], blind=True)
        async with app.run_test(size=SIZE):
            text = _screen_text(app)
            assert "Sender type:" in text
            assert "[p] person" in text
            assert "[k] skip" in text
            assert "[e] exclude" in text

    async def test_blind_label_prompt_lists_skip_exclude_notes_and_r(self, tmp_path):
        app = ReviewApp([_golden("t1")], blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("p")  # answer the sender step
            text = _screen_text(app)
            assert "Label:" in text
            assert "[r] needs_response" in text
            assert "[k] skip" in text
            assert "[e] exclude" in text
            assert "[n] notes" in text

    async def test_blind_mode_hides_current_labels(self, tmp_path):
        app = ReviewApp([_golden("t1")], blind=True)
        async with app.run_test(size=SIZE):
            assert "Current labels:" not in _screen_text(app)

    async def test_normal_mode_shows_current_labels(self, tmp_path):
        app = ReviewApp([_golden("t1")], blind=False)
        async with app.run_test(size=SIZE):
            text = _screen_text(app)
            assert "Current labels:" in text
            assert "Sender type: person" in text


# ---------------------------------------------------------------------------
# Skip: no judgment is recorded
# ---------------------------------------------------------------------------

class TestSkip:
    async def test_normal_skip_leaves_no_judgment(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("k")
            assert threads[0].reviewed is False
            assert threads[0].excluded is False
            assert "Thread 2/2" in _screen_text(app)
            assert "Skipped (no judgment)" in _status(app)

    async def test_blind_skip_leaves_no_judgment(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("k")
            assert threads[0].reviewed is False
            assert threads[0].excluded is False
            assert "Thread 2/2" in _screen_text(app)


# ---------------------------------------------------------------------------
# Exclude: permanently set aside
# ---------------------------------------------------------------------------

class TestExclude:
    async def test_normal_exclude_marks_excluded(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("e")
            assert threads[0].excluded is True
            assert threads[0].reviewed is True
            assert "Thread 2/2" in _screen_text(app)

    async def test_blind_exclude_marks_excluded(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("e")
            assert threads[0].excluded is True
            assert threads[0].reviewed is True


# ---------------------------------------------------------------------------
# Classification keys, confirm, and notes
# ---------------------------------------------------------------------------

class TestClassifyAndNotes:
    async def test_blind_p_then_r_classifies(self, tmp_path):
        threads = [_golden("t1", expected_sender_type="service", expected_label="fyi")]
        app = ReviewApp(threads, blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("p", "r")
        assert threads[0].expected_sender_type == "person"
        assert threads[0].expected_label == "needs_response"
        assert threads[0].reviewed is True
        assert app.return_value == "done"  # queue finished

    async def test_normal_enter_confirms(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
        assert threads[0].reviewed is True
        assert app.return_value == "done"

    async def test_normal_y_confirms(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("y")
        assert threads[0].reviewed is True

    async def test_normal_s_submenu_sets_sender(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("s")
            assert "Select sender type:" in _screen_text(app)
            await pilot.press("s")  # service
            assert threads[0].expected_sender_type == "service"
            assert threads[0].reviewed is True
            assert "Sender type set to: service" in _status(app)

    async def test_normal_l_submenu_cancel_returns_to_menu(self, tmp_path):
        threads = [_golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("l", "z")  # z is not a label key -> cancel
            assert threads[0].expected_label == "fyi"
            assert threads[0].reviewed is False
            assert "Actions:" in _screen_text(app)  # back on the same thread

    async def test_normal_notes_do_not_confirm(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("n")
            assert isinstance(app.screen, PromptLineScreen)
            await pilot.press(*"watch this one")
            await pilot.press("enter")
            assert threads[0].notes == "watch this one"
            assert threads[0].reviewed is False
            assert "not yet confirmed" in _status(app)

    async def test_blind_notes_then_classify(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("p")
            await pilot.press("n")
            await pilot.press(*"note")
            await pilot.press("enter")
            assert threads[0].notes == "note"
            assert "Label:" in _screen_text(app)  # still at the label step
            await pilot.press("r")
        assert threads[0].expected_label == "needs_response"

    async def test_unknown_key_reports_it(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("x")
            assert "Unknown action" in _status(app)
            assert threads[0].reviewed is False


# ---------------------------------------------------------------------------
# Stage filters
# ---------------------------------------------------------------------------

class TestStages:
    async def test_blind_stage_1_reviews_after_sender_only(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=True, stage=1)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("s")  # service
        assert threads[0].expected_sender_type == "service"
        assert threads[0].reviewed is True
        assert app.return_value == "done"

    async def test_blind_stage_2_starts_at_label(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=True, stage=2)
        async with app.run_test(size=SIZE) as pilot:
            assert "Label:" in _screen_text(app)
            await pilot.press("f")
        assert threads[0].expected_label == "fyi"
        assert threads[0].reviewed is True


# ---------------------------------------------------------------------------
# Undo across the queue (the review_loop invariants, now in ReviewApp)
# ---------------------------------------------------------------------------

class TestReviewUndo:
    async def test_undo_after_skip_returns_to_skipped_thread_not_earlier(self, tmp_path):
        threads = [_golden("t0"), _golden("t1"), _golden("t2")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")  # confirm t0
            await pilot.press("k")  # skip t1
            await pilot.press("z")  # undo on t2
            assert "Back to thread 2/3" in _status(app)
            assert threads[0].reviewed is True  # earlier confirm untouched
            assert "Thread 2/3" in _screen_text(app)

    async def test_blind_single_undo_fully_reverts_one_classification(self, tmp_path):
        threads = [_golden("t0"), _golden("t1")]
        app = ReviewApp(threads, blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("s", "r")  # t0: service / needs_response
            assert threads[0].expected_sender_type == "service"
            await pilot.press("z")  # undo on t1
            assert threads[0].expected_sender_type == "person"
            assert threads[0].expected_label == "fyi"
            assert threads[0].reviewed is False
            assert "Thread 1/2" in _screen_text(app)

    async def test_blind_undo_at_label_step_reverts_current_and_steps_back(self, tmp_path):
        threads = [_golden("t0"), _golden("t1")]
        app = ReviewApp(threads, blind=True)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("p", "f")  # t0 done
            await pilot.press("p")  # t1: sender set
            await pilot.press("n")
            await pilot.press(*"scratch")
            await pilot.press("enter")
            await pilot.press("z")  # undo at the label step
            assert "Back to thread 1/2" in _status(app)
            assert threads[1].notes == ""  # in-progress edits discarded
            assert threads[1].expected_sender_type == "person"
            assert threads[0].reviewed is False  # t0 decision reverted

    async def test_undo_with_empty_stack_reports_nothing(self, tmp_path):
        threads = [_golden("t0")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("z")
            assert "Nothing to undo" in _status(app)


class TestQuit:
    async def test_q_quits(self, tmp_path):
        threads = [_golden("t0"), _golden("t1")]
        app = ReviewApp(threads, blind=False)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("q")
        assert app.return_value == "quit"
        assert threads[0].reviewed is False


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


class TestScrollAndStages:
    async def test_arrow_keys_scroll_the_thread_body(self, tmp_path):
        import base64

        body = "\n".join(f"line {i}" for i in range(80))
        data = base64.urlsafe_b64encode(body.encode()).decode()
        thread = _golden("t1", messages=[{"payload": {"mimeType": "text/plain", "body": {"data": data}}}])
        app = ReviewApp([thread], blind=False)
        async with app.run_test(size=SIZE) as pilot:
            scroll = app.query_one("#review-scroll")
            assert scroll.scroll_offset.y == 0
            await pilot.press("down", "down", "down")
            assert scroll.scroll_offset.y == 3
            await pilot.press("up")
            assert scroll.scroll_offset.y == 2

    async def test_stage_2_disables_the_sender_submenu(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=False, stage=2)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("s")
            assert "Unknown action" in _status(app)
            assert threads[0].expected_sender_type == "person"

    async def test_stage_1_disables_the_label_submenu(self, tmp_path):
        threads = [_golden("t1")]
        app = ReviewApp(threads, blind=False, stage=1)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("l")
            assert "Unknown action" in _status(app)
            assert threads[0].expected_label == "fyi"
