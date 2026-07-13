"""E2E driver for evals.newsletter_label (LabelApp).

Drives the full curate+label workflow over diverse synthetic newsletters:
browse/select, span add/edit/delete, clear-all, unlocatable text-edit fallback,
Phase-B scoring+themes, story/newsletter exclusion, notes, undo, accept, skip.
The TUI is manual-only (issue #59 removed LLM auto-seeding), so scenarios
pre-populate stories directly where a story-ful newsletter is needed.
"""

import asyncio
import tempfile
from pathlib import Path

import _e2e
import synth_data
from _e2e import SIZE, drain, report, run_scenarios

from evals.newsletter_label import (
    DetailScreen,
    LabelApp,
    row_for_body_line,
)
from evals.newsletter_schemas import GoldenStory

NLS = {n.thread_id: n for n in synth_data.newsletters()}


def _fresh():
    """Fresh copies of all newsletters (scenarios mutate, so re-generate)."""
    return {n.thread_id: n for n in synth_data.newsletters()}


def _app(queue):
    tmp = Path(tempfile.mkdtemp()) / "golden.jsonl"
    return LabelApp(queue, queue, tmp)


def _populate_from_paragraphs(nl):
    """Pre-populate one story per blank-line paragraph of the body.

    Stands in for the removed LLM auto-seed (issue #59) wherever a scenario
    needs a story-ful newsletter — the stories are exact body slices, so they
    all locate."""
    normalized = nl.body.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    nl.stories = [
        GoldenStory(story_id=f"{nl.thread_id}:{i}", text=p)
        for i, p in enumerate(paragraphs)
    ]
    return nl


def _detail(app) -> DetailScreen:
    assert isinstance(app.screen, DetailScreen), f"not on detail: {app.screen!r}"
    return app.screen


async def _goto_body(pilot, app, body_idx):
    screen = _detail(app)
    target = row_for_body_line(screen._rows, body_idx)
    lv = screen.query_one("#rows")
    cur = lv.index or 0
    delta = target - cur
    for _ in range(abs(delta)):
        await pilot.press("down" if delta > 0 else "up")


# ---------------------------------------------------------------------------

async def s_label_single_story(chk):
    nls = _fresh()
    nl = _populate_from_paragraphs(nls["nl-single"])
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        chk.eq(len(nl.stories), 1, "single-paragraph body -> one story")
        chk.that(nl.stories and "Sarah" in nl.stories[0].text, "story text is the body")
        # Phase B: score all-Good (1/2/3 = Poor/OK/Good) + scripture theme emphasized.
        await pilot.press("1", "l")
        await pilot.press("3", "3", "3", "3")
        await pilot.press("s", "s", "enter")   # scripture: present -> emphasized
        await drain(app, pilot)
        chk.eq(nl.stories[0].expected_scores, {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
               "scores assigned")
        chk.eq(nl.stories[0].expected_tier, "excellent", "tier derived from avg 3.0")
        chk.eq(nl.stories[0].expected_themes, {"scripture": "emphasized"}, "theme assigned")
        chk.that(nl.stories[0].reviewed, "story marked reviewed")
        await pilot.press("c")              # accept (all labeled -> silent) -> back to list
        await drain(app, pilot)
        chk.that(nl.reviewed, "newsletter confirmed reviewed after accept")


async def s_span_add_edit_delete_clear(chk):
    nls = _fresh()
    nl = _populate_from_paragraphs(nls["nl-multi"])
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        chk.eq(len(nl.stories), 3, "three-paragraph body -> three stories")
        start_n = len(nl.stories)
        # Add a new story via span: go to a body line, mark start, extend, commit.
        await _goto_body(pilot, app, 0)
        await pilot.press("a", "enter", "down", "enter")
        await drain(app, pilot)
        chk.eq(len(nl.stories), start_n + 1, "span-add created a story")
        # Edit selected story's bounds (extend end by one line).
        await pilot.press("1", "e", "down", "enter")
        await drain(app, pilot)
        chk.that(_detail(app).mode == "browse", "back to browse after span commit")
        # Delete a story.
        n_before = len(nl.stories)
        await pilot.press("2", "d")
        await drain(app, pilot)
        chk.eq(len(nl.stories), n_before - 1, "delete removed one story")
        # Clear all (confirm y).
        await pilot.press("C", "y")
        await drain(app, pilot)
        chk.eq(len(nl.stories), 0, "clear-all emptied the story list")


async def s_unlocatable_text_fallback(chk):
    nls = _fresh()
    nl = nls["nl-unlocatable"]
    original = nl.stories[0].text
    app = _app([nl])  # nl-unlocatable ships with its story pre-populated
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        # Accept the collapsed prefill unchanged -> must NOT flatten the multi-line text.
        await pilot.press("1", "e")         # opens the text-fallback modal (worker suspends on it)
        await pilot.pause()
        from textual.widgets import Input
        chk.that(app.screen.query(Input), "text fallback prompt opened for unlocatable story")
        await pilot.press("enter")          # accept prefill unchanged -> worker completes
        await drain(app, pilot)
        chk.eq(nl.stories[0].text, original, "multi-line text preserved on accept-prefill (bug #4 fix)")
        # Now genuinely edit it.
        await pilot.press("1", "e")
        await pilot.pause()
        app.screen.query_one(Input).value = "corrected single line"
        await pilot.press("enter")
        await drain(app, pilot)
        chk.eq(nl.stories[0].text, "corrected single line", "text updated on real edit")


async def s_story_less_open_stays_empty(chk):
    # An unreviewed, story-less newsletter opens with an empty strip and the
    # add-a-story hint — nothing auto-populates it (issue #59 removed seeding).
    nls = _fresh()
    empty = nls["nl-empty"]
    app = _app([empty])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        chk.eq(len(empty.stories), 0, "story-less newsletter stays empty on open")
        chk.that("[a]dd a story" in _e2e.screen_text(app), "manual add hint shown")
        chk.that(app.is_running, "empty newsletter opens without crashing")


async def s_many_stories_navigation(chk):
    nls = _fresh()
    nl = nls["nl-many"]
    app = _app([nl])  # nl-many ships with 12 pre-populated stories
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        await pilot.press("1")
        chk.eq(_detail(app).selected_story, 0, "number key selects story 1")
        await pilot.press("9")
        chk.eq(_detail(app).selected_story, 8, "number key selects story 9")
        await pilot.press("n")
        chk.eq(_detail(app).selected_story, 9, "n advances selection")
        await pilot.press("p", "p")
        chk.eq(_detail(app).selected_story, 7, "p moves selection back")


async def s_exclude_and_undo(chk):
    nls = _fresh()
    nl = nls["nl-mixed"]
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        await pilot.press("2", "u")         # exclude story 2
        await drain(app, pilot)
        chk.that(nl.stories[1].excluded, "story 2 excluded")
        await pilot.press("z")              # undo
        await drain(app, pilot)
        chk.that(not nl.stories[1].excluded, "undo restored story 2 inclusion")
        await pilot.press("X")              # toggle newsletter excluded
        await drain(app, pilot)
        chk.that(nl.excluded, "newsletter excluded via X")
        await pilot.press("z")
        await drain(app, pilot)
        chk.that(not nl.excluded, "undo restored newsletter inclusion")


async def s_notes_and_skip(chk):
    nls = _fresh()
    nl = _populate_from_paragraphs(nls["nl-single"])
    other = nls["nl-multi"]
    app = _app([nl, other])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        await pilot.press("N")              # notes prompt (worker suspends on the modal)
        await pilot.pause()
        from textual.widgets import Input
        app.screen.query_one(Input).value = "a reviewer note"
        await pilot.press("enter")
        await drain(app, pilot)
        chk.eq(nl.notes, "a reviewer note", "notes saved")
        # Skip advances to the next newsletter without marking reviewed.
        await pilot.press("k")
        await drain(app, pilot)
        chk.that(not nl.reviewed, "skip does not mark reviewed")
        chk.that(isinstance(app.screen, DetailScreen), "skip opened the next newsletter")
        chk.that(_detail(app).newsletter is other, "skip advanced to the NEXT newsletter, not the same one")


async def s_accept_confirm_when_unlabeled(chk):
    nls = _fresh()
    nl = nls["nl-mixed"]  # has one labeled + one unlabeled story
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        # Accept with an unlabeled story -> confirmation; decline (any non-y).
        await pilot.press("c", "n")
        await drain(app, pilot)
        chk.that(not nl.reviewed, "declined accept did not confirm")
        chk.that(isinstance(app.screen, DetailScreen), "stayed on detail after declining")
        # Accept anyway.
        await pilot.press("c", "y")
        await drain(app, pilot)
        chk.that(nl.reviewed, "accept-anyway confirmed the list")
        chk.that(not isinstance(app.screen, DetailScreen), "accept returned to the list screen")


async def s_span_commits_exact_text(chk):
    nls = _fresh()
    nl = _populate_from_paragraphs(nls["nl-multi"])
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        body_lines = nl.body.splitlines()
        n0 = len(nl.stories)
        # New span body line 0..2 inclusive via mark-start/mark-end.
        await _goto_body(pilot, app, 0)
        await pilot.press("a", "enter")     # mark start at body 0
        await _goto_body(pilot, app, 2)     # end tracks to body 2
        await pilot.press("enter")          # commit
        await drain(app, pilot)
        chk.eq(len(nl.stories), n0 + 1, "span-add created a story")
        chk.eq(nl.stories[-1].text, "\n".join(body_lines[0:3]),
               "committed span text is the EXACT inclusive body slice")
        # Fine-adjust the new story's end back to body 0 via 'e' (adjust stage).
        await pilot.press("e")              # edit bounds of the just-created (selected) story
        await pilot.pause()
        chk.that(_detail(app).mode == "span", "e enters span adjust mode")
        await _goto_body(pilot, app, 0)
        await pilot.press("e", "enter")     # set END to body 0, commit
        await drain(app, pilot)
        chk.eq(nl.stories[-1].text, body_lines[0], "e fine-adjust narrowed to the single-line slice")


async def s_esc_cancels_span_and_backs_out(chk):
    nls = _fresh()
    nl = _populate_from_paragraphs(nls["nl-multi"])
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        n0 = len(nl.stories)
        await _goto_body(pilot, app, 0)
        await pilot.press("a", "enter")     # enter span mode, mark start
        chk.that(_detail(app).mode == "span", "in span mode after 'a'")
        await pilot.press("escape")         # cancel the span edit
        chk.that(_detail(app).mode == "browse", "esc cancels span -> browse mode")
        chk.eq(len(nl.stories), n0, "no story created when span cancelled")
        await pilot.press("escape")         # browse-mode esc -> back to list
        chk.that(not isinstance(app.screen, DetailScreen), "esc from browse returns to the list")


async def s_relabel_and_score_cancel(chk):
    nls = _fresh()
    nl = nls["nl-reviewed"]                  # already reviewed + one labeled story
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        chk.that("Reviewed: True" in _e2e.screen_text(app), "reviewed=True header rendered")
        orig = dict(nl.stories[0].expected_scores)
        # Cancel score entry with a non-1-3 key -> labels untouched.
        await pilot.press("1", "l")
        await pilot.press("9")
        await drain(app, pilot)
        chk.eq(nl.stories[0].expected_scores, orig, "non-digit cancels score entry (labels intact)")
        # Re-label: new scores, and ThemeScreen starts from the existing themes (toggle one off).
        await pilot.press("1", "l")
        await pilot.press("2", "2", "2", "2")
        await pilot.press("d", "enter")      # 'd'=disciple_making was on -> toggled OFF
        await drain(app, pilot)
        chk.eq(nl.stories[0].expected_tier, "fair", "re-scored 2/2/2/2 -> fair tier")
        chk.that("disciple_making" not in nl.stories[0].expected_themes, "existing theme toggled off")


async def s_diverse_data_render(chk):
    # Actually OPEN the wide-char / emoji / CRLF newsletters and confirm their
    # stories locate and render without crashing (display-width wrapping + CRLF
    # splitlines).
    for tid in ("nl-emoji", "nl-crlf"):
        nls = _fresh()
        nl = _populate_from_paragraphs(nls[tid])
        app = _app([nl])
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await drain(app, pilot)
            chk.that(app.is_running, f"{tid} opens without crashing")
            chk.that(len(nl.stories) >= 2, f"{tid} has its stories")
            located = [s for s in _detail(app)._spans if s is not None]
            chk.eq(len(located), len(nl.stories), f"{tid} every story located in the body for rendering")


async def s_guard_and_decline_paths(chk):
    nls = _fresh()
    nl = _populate_from_paragraphs(nls["nl-multi"])
    app = _app([nl])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        # Label with nothing selected -> guard hint.
        await pilot.press("l")
        await drain(app, pilot)
        chk.that("Select a story" in _e2e.static_text(app, "#status"), "label guarded when no story selected")
        # Out-of-range number.
        await pilot.press("8")
        chk.that("No story 8" in _e2e.static_text(app, "#status"), "out-of-range number rejected")
        n0 = len(nl.stories)
        # Decline clear-all -> stories kept.
        await pilot.press("C", "n")
        await drain(app, pilot)
        chk.eq(len(nl.stories), n0, "declining clear-all keeps stories")
        # Undo with an empty stack.
        while _detail(app).undo_stack:
            await pilot.press("z")
            await drain(app, pilot)
        await pilot.press("z")
        chk.that("Nothing to undo" in _e2e.static_text(app, "#status"), "empty-undo hint shown")


async def s_list_nav_and_quit(chk):
    from textual.widgets import ListView
    nls = _fresh()
    queue = [nls["nl-single"], nls["nl-multi"], nls["nl-long"]]
    app = _app(queue)
    async with app.run_test(size=SIZE) as pilot:
        lv = app.query_one("#newsletters", ListView)
        chk.eq(len(lv), 3, "all queued newsletters listed")
        await pilot.press("down")
        chk.eq(lv.index, 1, "list cursor down")
        await pilot.press("end")
        chk.eq(lv.index, 2, "list end -> last")
        await pilot.press("home")
        chk.eq(lv.index, 0, "list home -> first")
        await pilot.press("q")
    chk.eq(app.return_value, "quit", "q quits from the list")


async def s_quit_from_detail(chk):
    nls = _fresh()
    app = _app([nls["nl-reviewed"]])
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("enter")
        await drain(app, pilot)
        await pilot.press("q")
    chk.eq(app.return_value, "quit", "q from the detail screen quits")


def scenarios():
    return [
        ("label_single_story", s_label_single_story),
        ("span_add_edit_delete_clear", s_span_add_edit_delete_clear),
        ("unlocatable_text_fallback", s_unlocatable_text_fallback),
        ("story_less_open_stays_empty", s_story_less_open_stays_empty),
        ("many_stories_navigation", s_many_stories_navigation),
        ("exclude_and_undo", s_exclude_and_undo),
        ("notes_and_skip", s_notes_and_skip),
        ("accept_confirm_when_unlabeled", s_accept_confirm_when_unlabeled),
        ("span_commits_exact_text", s_span_commits_exact_text),
        ("esc_cancels_span_and_backs_out", s_esc_cancels_span_and_backs_out),
        ("relabel_and_score_cancel", s_relabel_and_score_cancel),
        ("diverse_data_render", s_diverse_data_render),
        ("guard_and_decline_paths", s_guard_and_decline_paths),
        ("list_nav_and_quit", s_list_nav_and_quit),
        ("quit_from_detail", s_quit_from_detail),
    ]


if __name__ == "__main__":
    import sys
    results = asyncio.run(run_scenarios(scenarios()))
    sys.exit(report("newsletter_label", results))
