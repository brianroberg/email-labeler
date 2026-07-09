"""E2E driver for newsletter_review.tui (ReviewApp) — read-only browser, no model.

Drives list nav/paging, drill-in/back + detail scrolling, and the full filter
matrix (type menu -> tier/theme/sender), including CANCEL-vs-clear semantics,
the literal 'cancel' sender value, empty-clears, pre-applied init filters, and
an empty filtered result.
"""

import asyncio
import os
import sys
import time
from contextlib import contextmanager

import synth_data
from _e2e import report, run_scenarios

from newsletter_review.tui import DetailScreen, ReviewApp

SIZE = (100, 30)


def _recs():
    return synth_data.assessment_records()


def _by_id(*ids):
    """Fresh subset of assessment records selected (and ordered) by message_id."""
    by = {r["message_id"]: r for r in _recs()}
    return [by[i] for i in ids]


def _lv(app):
    from textual.widgets import ListView
    return app.query_one(ListView)


def _rows(app) -> list[str]:
    """Rendered text of each visible list row, top to bottom."""
    from textual.widgets import Label, ListView
    return [str(item.query_one(Label).render()) for item in app.query_one(ListView).children]


def _row_date(row: str) -> str:
    """The Date column (first whitespace-delimited token) of a rendered row."""
    return row.split()[0]


def _title(app) -> str:
    from textual.widgets import Static
    return str(app.query_one("#title", Static).render())


@contextmanager
def _tz(name: str):
    """Temporarily set the process timezone (Linux ``TZ`` + ``time.tzset()``),
    restoring the prior value on exit. ``_local_date`` / ``_days_ago_cutoff``
    read the process-local zone, so date-column and since-filter scenarios must
    pin it to a west-of-UTC zone to exercise the local-date conversion."""
    prev = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = prev
        time.tzset()


async def s_list_nav(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        chk.eq(len(_lv(app)), len(_recs()), "all records listed")
        await pilot.press("down")
        chk.eq(_lv(app).index, 1, "down moves cursor")
        await pilot.press("up")
        chk.eq(_lv(app).index, 0, "up moves back")
        await pilot.press("end")
        chk.eq(_lv(app).index, len(_recs()) - 1, "end -> last")
        await pilot.press("home")
        chk.eq(_lv(app).index, 0, "home -> first")


async def s_paging_small_terminal(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=(100, 8)) as pilot:
        await pilot.press("pagedown")
        chk.that(_lv(app).index > 0, "pagedown advanced a page on a short terminal")
        await pilot.press("pageup")
        chk.eq(_lv(app).index, 0, "pageup returned to top")


async def s_drill_and_back(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("down", "down")   # to index 2
        pre = _lv(app).index
        await pilot.press("enter")
        chk.that(isinstance(app.screen, DetailScreen), "enter opens detail")
        await pilot.press("escape")
        chk.that(not isinstance(app.screen, DetailScreen), "escape returns to list")
        chk.eq(_lv(app).index, pre, "list cursor preserved across drill-in")


async def s_detail_scroll(chk):
    # a8 has long quality/theme CoT -> scrollable.
    recs = [r for r in _recs() if r["thread_id"] == "a8"]
    app = ReviewApp(recs)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await pilot.press("down", "down", "pagedown", "end", "home")
        await pilot.pause()
        chk.that(app.is_running, "detail scroll does not crash")


async def s_filter_tier_set_and_clear(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        full = len(_lv(app))
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("t")              # tier submenu
        await pilot.pause()
        await pilot.press("g")              # good
        await pilot.pause()
        chk.that("tier:good" in _title(app), "tier filter applied to title")
        chk.that(len(_lv(app)) < full, "tier filter narrowed the list")
        chk.eq(_lv(app).index, 0, "cursor reset after filter")
        # Clear it.
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        await pilot.press("c")              # clear
        await pilot.pause()
        chk.that("tier:" not in _title(app), "tier filter cleared")
        chk.eq(len(_lv(app)), full, "list restored to full after clear")


async def s_filter_tier_cancel(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        full = len(_lv(app))
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        await pilot.press("z")              # unmapped -> CANCEL -> no change
        await pilot.pause()
        chk.that("tier:" not in _title(app), "cancel added no tier filter")
        chk.eq(len(_lv(app)), full, "list unchanged after cancel")


async def s_filter_theme(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        full = len(_lv(app))
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("h")              # theme submenu
        await pilot.pause()
        await pilot.press("s")              # scripture
        await pilot.pause()
        chk.that("theme:scripture" in _title(app), "theme filter applied")
        chk.that(len(_lv(app)) < full, "theme filter narrowed the list")
        # Clear via x.
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        chk.that("theme:" not in _title(app), "theme cleared via x")


async def s_filter_sender_and_cancel(chk):
    from textual.widgets import Input
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        full = len(_lv(app))
        # Set a sender substring filter.
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("s")              # sender prompt
        await pilot.pause()
        app.screen.query_one(Input).value = "dm.org"
        await pilot.press("enter")
        await pilot.pause()
        chk.that("sender:dm.org" in _title(app), "sender filter applied")
        chk.that(len(_lv(app)) < full, "sender filter narrowed the list")
        # Empty submit clears the filter.
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one(Input).value = ""
        await pilot.press("enter")
        await pilot.pause()
        chk.that("sender:" not in _title(app), "empty submit clears sender filter")
        chk.eq(len(_lv(app)), full, "list restored after clearing sender")
        # Esc cancels (leaves no filter).
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one(Input).value = "whatever"
        await pilot.press("escape")
        await pilot.pause()
        chk.that("sender:" not in _title(app), "Esc cancels the sender prompt")


async def s_sender_literal_cancel_is_a_value(chk):
    # The word 'cancel' typed into the sender prompt is a real filter value,
    # never confused with the CANCEL dismissal sentinel.
    from textual.widgets import Input
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        app.screen.query_one(Input).value = "cancel"
        await pilot.press("enter")
        await pilot.pause()
        chk.eq(app.f_sender, "cancel", "'cancel' is a literal filter value")
        chk.that("sender:cancel" in _title(app), "'cancel' shown as an active sender filter")


async def s_init_filters(chk):
    app = ReviewApp(_recs(), init_tier="good")
    async with app.run_test(size=SIZE):
        chk.that("tier:good" in _title(app), "init_tier applied on launch")
        chk.that(0 < len(_lv(app)) < len(_recs()), "init_tier pre-narrowed the list")


async def s_empty_filtered_result(chk):
    app = ReviewApp(_recs(), init_sender="nobody@nowhere")
    async with app.run_test(size=SIZE) as pilot:
        chk.eq(len(_lv(app)), 0, "no records match the init sender filter")
        await pilot.press("enter")          # no-op on empty list
        await pilot.pause()
        chk.that(not isinstance(app.screen, DetailScreen), "enter is a no-op on an empty list")
        chk.that(app.is_running, "empty filtered result does not crash")


async def s_quit(chk):
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("q")
    chk.eq(app.return_value, "quit", "q quits")


def _detail_content(app) -> str:
    from textual.widgets import Static
    return "\n".join(str(w.render()) for w in app.screen.query(Static))


async def s_detail_no_stories(chk):
    from textual.widgets import Label, ListView
    recs = [r for r in _recs() if r["thread_id"] == "a5"]  # overall_tier None, stories=[]
    app = ReviewApp(recs)
    async with app.run_test(size=SIZE) as pilot:
        row = str(app.query_one(ListView).children[0].query_one(Label).render())
        chk.that("0 stories" in row, "list row shows '0 stories' for a no-story record")
        chk.that(row.strip().startswith("—"), "list row shows '—' tier for None")
        await pilot.press("enter")
        c = _detail_content(app)
        chk.that("No stories extracted" in c, "no-stories detail branch rendered")
        chk.that("Overall: —" in c, "detail overall tier shows '—' for None")


async def s_detail_scroll_offset_and_quit(chk):
    from textual.containers import VerticalScroll
    recs = [r for r in _recs() if r["thread_id"] == "a8"]  # long CoT -> scrollable
    app = ReviewApp(recs)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        sc = app.screen.query_one("#detail-scroll", VerticalScroll)
        await pilot.press("down", "down", "pagedown")
        await pilot.pause()
        chk.that(sc.scroll_offset.y > 0, "detail scroll increased the offset")
        await pilot.press("up", "pageup", "home")   # reverse-scroll actions
        await pilot.pause()
        chk.eq(sc.scroll_offset.y, 0, "home returned detail scroll to the top")
        await pilot.press("q")                      # detail-level quit binding
    chk.eq(app.return_value, "quit", "q from the detail view quits")


async def s_detail_wide_and_multistory(chk):
    a6 = [r for r in _recs() if r["thread_id"] == "a6"]  # 2 stories, multi-theme
    app = ReviewApp(a6)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        c = _detail_content(app)
        chk.that("Story 2/2" in c, "multi-story detail shows the second story header")
        chk.that("christlikeness" in c, "multi-theme Themes line rendered")
    a7 = [r for r in _recs() if r["thread_id"] == "a7"]  # emoji / CJK subject + story
    app = ReviewApp(a7)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        chk.that(app.is_running, "emoji/CJK detail renders without crashing")
        chk.that("東京" in _detail_content(app), "wide-char story text rendered in detail")


async def s_date_column_local_rendering(chk):
    # The evening-UTC send (01:00 on 07-05 UTC) is 20:00 on 07-04 in Chicago, so
    # the Date column must show the LOCAL date 2026-07-04, never the UTC 07-05.
    # The no-send_date record renders "—".
    with _tz("America/Chicago"):
        app = ReviewApp(_by_id("nd-evening", "a5"))
        async with app.run_test(size=SIZE):
            rows = _rows(app)
            chk.eq(len(rows), 2, "both records listed")
            chk.eq(_row_date(rows[0]), "2026-07-04",
                   "evening-UTC send renders the PREVIOUS local calendar date")
            chk.that("2026-07-05" not in rows[0],
                     "the raw UTC date never leaks into the row")
            chk.eq(_row_date(rows[1]), "—", "the no-send_date record renders '—'")


async def s_list_sorted_by_send_date_desc(chk):
    # Fixed-date records (+ the no-date one) prove newest-first ordering with the
    # dateless record pinned last. Pinned to UTC so midday dates map to exact
    # local calendar dates for the column assertion.
    with _tz("UTC"):
        app = ReviewApp(_by_id("a8", "a5", "a6", "a7"))  # deliberately shuffled input
        async with app.run_test(size=SIZE):
            chk.eq([r["message_id"] for r in app.filtered], ["a6", "a7", "a8", "a5"],
                   "records sorted by send-date descending, no-date last")
            chk.eq([_row_date(r) for r in _rows(app)],
                   ["2026-08-01", "2026-07-15", "2026-06-01", "—"],
                   "visible Date column is newest-first with '—' last")


async def s_filter_date_past_window(chk):
    # a1 (now-2d) and a2 (now-10d) are inside a past-30d window; a3 (now-45d),
    # a4 (now-200d) and a5 (no date) are outside it.
    sub = _by_id("a1", "a2", "a3", "a4", "a5")
    app = ReviewApp(sub)
    async with app.run_test(size=SIZE) as pilot:
        full = len(_lv(app))
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("d")          # date submenu
        await pilot.pause()
        await pilot.press("3")          # past 30 days
        await pilot.pause()
        chk.that("since:" in _title(app), "date filter shows a since cutoff in the title")
        chk.eq([r["message_id"] for r in app.filtered], ["a1", "a2"],
               "only the two dynamically-recent records survive past-30d")
        # Clear via the date submenu's clear key.
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("x")          # clear
        await pilot.pause()
        chk.that("since:" not in _title(app), "date filter cleared")
        chk.eq(len(_lv(app)), full, "full list restored after clearing the date filter")


async def s_filter_date_since_prompt(chk):
    from textual.widgets import Input
    # West-of-UTC so the evening-UTC record's local date (07-04) differs from its
    # UTC date (07-05); the boundary must be judged on the LOCAL date.
    with _tz("America/Chicago"):
        sub = _by_id("nd-evening", "a6", "a8", "a5")  # local: 07-04, 08-01, 06-01, —
        app = ReviewApp(sub)
        async with app.run_test(size=SIZE) as pilot:
            async def _set_since(value: str):
                await pilot.press("f")
                await pilot.pause()
                await pilot.press("d")
                await pilot.pause()
                await pilot.press("s")      # "since…" prompt
                await pilot.pause()
                app.screen.query_one(Input).value = value
                await pilot.press("enter")
                await pilot.pause()

            # since == the UTC date of the evening record: it must be EXCLUDED
            # (its local date 07-04 is earlier); the 08-01 record still passes.
            await _set_since("2026-07-05")
            ids = [r["message_id"] for r in app.filtered]
            chk.eq(ids, ["a6"], "since=2026-07-05 keeps only the 08-01 record")
            chk.that("nd-evening" not in ids,
                     "UTC-date boundary excludes the evening record (local date is earlier)")

            # since == the LOCAL date of the evening record: now INCLUDED.
            await _set_since("2026-07-04")
            ids = [r["message_id"] for r in app.filtered]
            chk.eq(ids, ["a6", "nd-evening"],
                   "since=2026-07-04 (its local date) now includes the evening record")
            chk.that("a8" not in ids and "a5" not in ids,
                     "the 06-01 record and the dateless record stay excluded")


async def s_filter_date_cancel(chk):
    # An unmapped key in the date submenu dismisses with CANCEL -> no change.
    app = ReviewApp(_recs())
    async with app.run_test(size=SIZE) as pilot:
        full = len(_lv(app))
        await pilot.press("f")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("z")          # unmapped -> CANCEL
        await pilot.pause()
        chk.eq(app.f_since, None, "cancel left the since filter unset")
        chk.that("since:" not in _title(app), "no date filter after cancel")
        chk.eq(len(_lv(app)), full, "list unchanged after cancel")


async def s_init_since_prefilters(chk):
    # ReviewApp(init_since=...) pre-applies the date filter on launch (mirrors
    # init_tier). a6 (08-01) passes; a8 (06-01) and a5 (no date) do not.
    sub = _by_id("a6", "a8", "a5")
    app = ReviewApp(sub, init_since="2026-07-01")
    async with app.run_test(size=SIZE):
        chk.that("since:2026-07-01" in _title(app), "init_since shown in the title")
        chk.that(0 < len(_lv(app)) < len(sub), "init_since pre-narrowed the list")
        chk.eq([r["message_id"] for r in app.filtered], ["a6"],
               "only the record on/after the since date survives launch")


async def s_old_scheme_record_rejected_at_load(chk):
    # load_assessments must fail fast (not mid-render) on a pre-#53 record whose
    # story themes are a LIST, naming the file + line so the reader can re-migrate.
    import json
    import tempfile
    from pathlib import Path

    from newsletter_review.tui import load_assessments

    new_rec = {
        "timestamp": "2026-01-05T10:00:00+00:00", "message_id": "ok", "thread_id": "ok",
        "from": "a@dm.org", "subject": "New scheme", "overall_tier": "good",
        "stories": [synth_data._story_rec("A fine story.", synth_data._scores(3, 2, 3, 2),
                                          "good", {"scripture": "present"})],
    }
    old_rec = {
        "timestamp": "2026-01-06T10:00:00+00:00", "message_id": "old", "thread_id": "old",
        "from": "b@dm.org", "subject": "Old scheme", "overall_tier": "good",
        "stories": [{
            "text": "A legacy story.", "scores": synth_data._scores(2, 2, 2, 2),
            "average_score": 2.0, "tier": "good",
            "themes": ["scripture", "church"],  # old scheme: a LIST, not a dict
            "quality_cot": "", "theme_cot": "",
        }],
    }
    tmpdir = tempfile.mkdtemp(prefix="oldscheme_")
    path = Path(tmpdir) / "assessments.jsonl"
    path.write_text(json.dumps(new_rec) + "\n" + json.dumps(old_rec) + "\n")

    try:
        load_assessments(path)
        chk.that(False, "load_assessments should raise ValueError on an old-scheme record")
    except ValueError as exc:
        msg = str(exc)
        chk.that(str(path) in msg, "error names the offending file path")
        chk.that(":2:" in msg, "error names the offending line number (2)")
        chk.that("re-migrated" in msg, "error is actionable (mentions re-migration)")
    except Exception as exc:  # noqa: BLE001
        chk.that(False, f"expected ValueError, got {type(exc).__name__}: {exc}")


def scenarios():
    return [
        ("list_nav", s_list_nav),
        ("paging_small_terminal", s_paging_small_terminal),
        ("drill_and_back", s_drill_and_back),
        ("detail_scroll", s_detail_scroll),
        ("filter_tier_set_and_clear", s_filter_tier_set_and_clear),
        ("filter_tier_cancel", s_filter_tier_cancel),
        ("filter_theme", s_filter_theme),
        ("filter_sender_and_cancel", s_filter_sender_and_cancel),
        ("sender_literal_cancel_is_a_value", s_sender_literal_cancel_is_a_value),
        ("init_filters", s_init_filters),
        ("empty_filtered_result", s_empty_filtered_result),
        ("quit", s_quit),
        ("detail_no_stories", s_detail_no_stories),
        ("detail_scroll_offset_and_quit", s_detail_scroll_offset_and_quit),
        ("detail_wide_and_multistory", s_detail_wide_and_multistory),
        ("date_column_local_rendering", s_date_column_local_rendering),
        ("list_sorted_by_send_date_desc", s_list_sorted_by_send_date_desc),
        ("filter_date_past_window", s_filter_date_past_window),
        ("filter_date_since_prompt", s_filter_date_since_prompt),
        ("filter_date_cancel", s_filter_date_cancel),
        ("init_since_prefilters", s_init_since_prefilters),
        ("old_scheme_record_rejected_at_load", s_old_scheme_record_rejected_at_load),
    ]


if __name__ == "__main__":
    results = asyncio.run(run_scenarios(scenarios()))
    sys.exit(report("newsletter_review", results))
