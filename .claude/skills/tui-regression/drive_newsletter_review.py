"""E2E driver for newsletter_review.tui (ReviewApp) — read-only browser, no model.

Drives list nav/paging, drill-in/back + detail scrolling, and the full filter
matrix (type menu -> tier/theme/sender), including CANCEL-vs-clear semantics,
the literal 'cancel' sender value, empty-clears, pre-applied init filters, and
an empty filtered result.
"""

import asyncio
import sys

import synth_data
from _e2e import report, run_scenarios

from newsletter_review.tui import DetailScreen, ReviewApp

SIZE = (100, 30)


def _recs():
    return synth_data.assessment_records()


def _lv(app):
    from textual.widgets import ListView
    return app.query_one(ListView)


def _title(app) -> str:
    from textual.widgets import Static
    return str(app.query_one("#title", Static).render())


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
    ]


if __name__ == "__main__":
    results = asyncio.run(run_scenarios(scenarios()))
    sys.exit(report("newsletter_review", results))
