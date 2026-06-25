"""Tests for evals.edit_tui — detail rendering and the un-exclude action.

The curses loop is driven through a minimal fake screen that replays a queued
sequence of keypresses, so the real key handlers run without a terminal.
"""

from pathlib import Path

from evals.edit_tui import _build_detail_lines, _detail_view, _format_list_row
from evals.review import load_golden_set
from evals.schemas import GoldenThread


def _golden(thread_id, **kw):
    base = dict(
        thread_id=thread_id, messages=[], senders=[], subject="Subj", snippet="snip",
        expected_sender_type="person", expected_label="fyi", reviewed=True,
    )
    base.update(kw)
    return GoldenThread(**base)


class FakeStdscr:
    """Replay a queued list of getch() return values; no-op rendering."""

    def __init__(self, keys):
        self._keys = list(keys)

    def getmaxyx(self):
        return (24, 80)

    def getch(self):
        return self._keys.pop(0)

    def clear(self):
        pass

    def refresh(self):
        pass

    def addstr(self, *args, **kwargs):
        pass

    def addnstr(self, *args, **kwargs):
        pass


class TestBuildDetailLines:
    def test_excluded_thread_shows_excluded_label(self):
        thread = _golden("t1", excluded=True)
        lines = _build_detail_lines(thread, 0, 1)
        assert any(line.startswith("Excluded:") for line in lines)
        assert not any("Skipped:" in line for line in lines)

    def test_non_excluded_thread_omits_excluded_label(self):
        thread = _golden("t1", excluded=False)
        lines = _build_detail_lines(thread, 0, 1)
        assert not any(line.startswith("Excluded:") for line in lines)


class TestListRowExcludedMarker:
    def test_excluded_row_is_marked(self):
        row = _format_list_row(_golden("t1", excluded=True), max_x=80)
        assert row.lstrip().startswith("X")

    def test_non_excluded_row_has_no_marker(self):
        row = _format_list_row(_golden("t1", excluded=False), max_x=80)
        assert not row.lstrip().startswith("X")


class TestUnexclude:
    def test_e_key_unexcludes_and_saves(self, tmp_path):
        path: Path = tmp_path / "golden.jsonl"
        thread = _golden("t1", excluded=True)
        threads = [thread]
        all_threads = [thread]
        # Press 'e' (un-exclude), then Esc to leave the detail view.
        stdscr = FakeStdscr([ord("e"), 27])

        result = _detail_view(stdscr, threads, 0, all_threads, path)

        assert result == "back"
        assert thread.excluded is False
        # The change was persisted atomically.
        saved = load_golden_set(path)
        assert len(saved) == 1
        assert saved[0].excluded is False
