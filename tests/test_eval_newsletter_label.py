"""Tests for evals.newsletter_label — pure state transitions + Pilot UI tests.

Every state transition lives in a pure function that mutates a
GoldenNewsletter / GoldenStory in place, tested directly without a terminal.
The Textual UI layer on top is driven with Textual's Pilot: real key presses
in, widget state / rendered content / on-disk golden-set effects out.
"""

import sys

import pytest

from evals.newsletter_label import (
    SpanEdit,
    accept_confirmation_message,
    add_story,
    assign_scores_and_themes,
    begin_add_span,
    begin_edit_span,
    browse_mode_bar,
    build_detail_rows,
    capture_snapshot,
    clear_stories,
    commit_span_edit,
    confirm_status_message,
    confirm_story_list,
    create_story_from_body,
    delete_story,
    edit_story,
    exclude_story,
    format_list_row,
    format_story_strip,
    format_theme_legend,
    label_progress,
    load_golden_set,
    locate_story_span,
    locate_story_spans,
    newsletter_exclude_status,
    restore_snapshot,
    row_for_body_line,
    save_golden_set,
    seed_confirmation_message,
    seed_from_extractor,
    seed_outcome_message,
    seed_stories,
    select_label_newsletters,
    span_cursor_moved,
    span_mark,
    span_mode_bar,
    span_range,
    span_set_end,
    span_set_start,
    story_at_body_line,
    story_excerpt,
    toggle_newsletter_excluded,
    unlabeled_story_count,
    wrap_text,
)
from evals.newsletter_schemas import GoldenNewsletter, GoldenStory
from newsletter import NewsletterTier, compute_tier


def _newsletter(thread_id="t1", **kw):
    base = dict(
        thread_id=thread_id,
        message_id="m1",
        sender="a@dm.org",
        subject="Subj",
        body="raw body",
    )
    base.update(kw)
    return GoldenNewsletter(**base)


def _story(story_id="t1:0", **kw):
    base = dict(story_id=story_id, text="Text")
    base.update(kw)
    return GoldenStory(**base)


class TestSeedStories:
    def test_seeds_candidates_with_stable_ids_and_provenance(self):
        nl = _newsletter("tS")
        texts = ["text a", "text b"]

        seed_stories(nl, texts)

        assert [s.story_id for s in nl.stories] == ["tS:0", "tS:1"]
        assert [s.text for s in nl.stories] == texts
        assert nl.seeded_from == "parse_stories"
        assert nl.reviewed is False  # seeding is not confirmation
        assert all(not s.reviewed for s in nl.stories)

    def test_seeding_a_reviewed_newsletter_clears_reviewed(self):
        # Re-seeding replaces the confirmed story list with an uncurated machine
        # seed — it must NOT stay authoritative extraction truth for
        # reviewed-only runs. Undo still restores via the snapshot.
        nl = _newsletter("tS", reviewed=True)

        seed_stories(nl, ["text a"])

        assert nl.reviewed is False

    def test_seeding_replaces_prior_candidates(self):
        nl = _newsletter("tS")
        nl.stories = [_story("tS:0", text="old")]

        seed_stories(nl, ["new text"])

        assert [s.text for s in nl.stories] == ["new text"]
        assert nl.stories[0].story_id == "tS:0"


class TestAssignScoresAndThemes:
    def test_assigns_scores_themes_reviewed_and_derives_tier(self):
        story = _story()
        scores = {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}

        # A legacy present-only list is coerced to graded "present" (issue #53).
        assign_scores_and_themes(story, scores, ["scripture", "church"])

        assert story.expected_scores == scores
        assert story.expected_themes == {"scripture": "present", "church": "present"}
        assert story.reviewed is True
        assert story.expected_tier == compute_tier(scores).value
        assert story.expected_tier == NewsletterTier.EXCELLENT.value

    def test_assigns_graded_themes_dict(self):
        story = _story()
        assign_scores_and_themes(
            story, _scores(), {"scripture": "emphasized", "church": "present"}
        )
        assert story.expected_themes == {"scripture": "emphasized", "church": "present"}

    def test_derives_fair_tier_for_midrange_scores(self):
        story = _story()
        # all-OK -> avg 2.0 -> fair (issue #53 bands).
        scores = {"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2}

        assign_scores_and_themes(story, scores, {})

        assert story.expected_tier == "fair"

    def test_scores_are_copied_not_aliased(self):
        story = _story()
        scores = {"simple": 1, "concrete": 1, "personal": 1, "dynamic": 1}

        assign_scores_and_themes(story, scores, {})
        scores["simple"] = 3

        assert story.expected_scores["simple"] == 1


class TestSelectLabelNewsletters:
    def test_excludes_excluded_newsletters(self):
        nls = [_newsletter("keep"), _newsletter("drop", excluded=True)]
        selected = select_label_newsletters(nls)
        assert [n.thread_id for n in selected] == ["keep"]

    def test_unreviewed_only_drops_reviewed(self):
        nls = [_newsletter("a", reviewed=False), _newsletter("b", reviewed=True)]
        selected = select_label_newsletters(nls, unreviewed_only=True)
        assert [n.thread_id for n in selected] == ["a"]

    def test_excluded_skipped_even_when_unreviewed(self):
        nls = [
            _newsletter("keep", reviewed=False),
            _newsletter("drop", reviewed=False, excluded=True),
        ]
        selected = select_label_newsletters(nls, unreviewed_only=True)
        assert [n.thread_id for n in selected] == ["keep"]

    def test_include_excluded_queues_excluded_newsletters(self):
        nls = [_newsletter("a"), _newsletter("b", excluded=True)]
        selected = select_label_newsletters(nls, include_excluded=True)
        assert [n.thread_id for n in selected] == ["a", "b"]

    def test_include_excluded_still_honors_unreviewed_only(self):
        nls = [
            _newsletter("a", reviewed=True),
            _newsletter("b", excluded=True, reviewed=False),
        ]
        selected = select_label_newsletters(
            nls, unreviewed_only=True, include_excluded=True,
        )
        assert [n.thread_id for n in selected] == ["b"]


class TestToggleNewsletterExcluded:
    def test_toggles_on_and_off(self):
        nl = _newsletter("t")
        assert toggle_newsletter_excluded(nl) is True
        assert nl.excluded is True
        assert toggle_newsletter_excluded(nl) is False
        assert nl.excluded is False

    def test_leaves_reviewed_untouched(self):
        nl = _newsletter("t", reviewed=True)
        toggle_newsletter_excluded(nl)
        assert nl.reviewed is True

    def test_status_messages(self):
        nl = _newsletter("t", excluded=True)
        assert "restore" in newsletter_exclude_status(nl).lower()
        assert "excluded" in newsletter_exclude_status(nl).lower()
        nl.excluded = False
        assert "restored" in newsletter_exclude_status(nl).lower()


class TestUndo:
    def test_undo_restores_prior_story_list_and_reviewed(self):
        nl = _newsletter("tU")
        seed_stories(nl, ["a", "b"])
        snap = capture_snapshot(nl)

        delete_story(nl, 1)
        edit_story(nl, 0, text="a-edited")
        confirm_story_list(nl)
        assert nl.reviewed is True
        assert [s.text for s in nl.stories] == ["a-edited"]

        restore_snapshot(nl, snap)

        assert nl.reviewed is False
        assert [s.text for s in nl.stories] == ["a", "b"]

    def test_undo_restores_per_story_labels(self):
        nl = _newsletter("tU")
        seed_stories(nl, ["a"])
        snap = capture_snapshot(nl)

        assign_scores_and_themes(
            nl.stories[0],
            {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
            ["scripture"],
        )
        assert nl.stories[0].reviewed is True

        restore_snapshot(nl, snap)

        assert nl.stories[0].reviewed is False
        assert nl.stories[0].expected_scores is None
        assert nl.stories[0].expected_themes == {}

    def test_snapshot_is_independent_of_later_mutations(self):
        nl = _newsletter("tU")
        seed_stories(nl, ["a"])
        snap = capture_snapshot(nl)
        nl.stories[0].text = "changed"

        restore_snapshot(nl, snap)

        assert nl.stories[0].text == "a"


class TestExcludeStory:
    def test_exclude_sets_flag(self):
        story = _story()
        exclude_story(story)
        assert story.excluded is True


class TestDeleteStory:
    def test_delete_removes_candidate(self):
        nl = _newsletter("tD")
        seed_stories(nl, ["a", "b"])

        delete_story(nl, 0)

        assert [s.text for s in nl.stories] == ["b"]


class TestClearStories:
    def test_clear_empties_the_list(self):
        nl = _newsletter("tC")
        seed_stories(nl, ["a", "b", "c"])

        clear_stories(nl)

        assert nl.stories == []


class TestEditStory:
    def test_edit_updates_text(self):
        nl = _newsletter("tE")
        seed_stories(nl, ["a"])

        edit_story(nl, 0, text="a2")

        assert nl.stories[0].text == "a2"

    def test_edit_leaves_unspecified_field_untouched(self):
        nl = _newsletter("tE")
        seed_stories(nl, ["a"])

        edit_story(nl, 0)

        assert nl.stories[0].text == "a"


class TestWrapText:
    def test_long_line_wraps_to_multiple_lines(self):
        wrapped = wrap_text("aaaa bbbb cccc", 5)
        assert len(wrapped) >= 2
        assert all(len(line) <= 5 for line in wrapped)

    def test_newlines_preserved(self):
        wrapped = wrap_text("one\ntwo", 80)
        assert wrapped == ["one", "two"]

    def test_width_zero_leaves_unchanged(self):
        wrapped = wrap_text("a very long line here", 0)
        assert wrapped == ["a very long line here"]

    def test_empty_text_yields_single_empty_string(self):
        assert wrap_text("", 80) == [""]


class TestCreateStoryFromBody:
    def test_multi_line_inclusive_span_joined_with_newlines(self):
        nl = _newsletter("tB", body="l0\nl1\nl2\nl3")

        story = create_story_from_body(nl, 1, 2)

        assert story.text == "l1\nl2"
        assert nl.stories[-1] is story

    def test_single_line_span(self):
        nl = _newsletter("tB", body="l0\nl1\nl2")

        story = create_story_from_body(nl, 1, 1)

        assert story.text == "l1"

    def test_start_greater_than_end_normalized(self):
        nl = _newsletter("tB", body="l0\nl1\nl2\nl3")

        story = create_story_from_body(nl, 3, 1)

        assert story.text == "l1\nl2\nl3"

    def test_out_of_range_clamped(self):
        nl = _newsletter("tB", body="l0\nl1\nl2")

        story = create_story_from_body(nl, -5, 99)

        assert story.text == "l0\nl1\nl2"

    def test_small_negative_start_clamps_to_zero_not_from_end(self):
        # Guards the *lower* clamp specifically: without max(0, ...) a small
        # negative start would slice from the end (Python slice semantics),
        # silently dropping the leading body lines.
        nl = _newsletter("tB", body="l0\nl1\nl2")

        story = create_story_from_body(nl, -1, 1)

        assert story.text == "l0\nl1"

    def test_span_entirely_past_end_collapses_to_last_line(self):
        # Both endpoints out of range on the same (high) side must clamp to the
        # boundary line, not produce an empty slice.
        nl = _newsletter("tB", body="l0\nl1\nl2")
        assert create_story_from_body(nl, 99, 99).text == "l2"

    def test_span_entirely_below_start_collapses_to_first_line(self):
        nl = _newsletter("tB", body="l0\nl1\nl2")
        assert create_story_from_body(nl, -9, -3).text == "l0"

    def test_stable_story_id_and_reviewed_untouched(self):
        nl = _newsletter("tB", body="l0\nl1")
        nl.stories = [_story("tB:0")]

        story = create_story_from_body(nl, 0, 0)

        assert story.story_id == "tB:1"
        assert nl.reviewed is False


class TestLocateStorySpan:
    def test_verbatim_multiline_span_located(self):
        body = "greeting\nstory line one\nstory line two\nsign-off".splitlines()
        assert locate_story_span(body, "story line one\nstory line two") == (1, 2)

    def test_single_line_span(self):
        body = "a\nb\nc".splitlines()
        assert locate_story_span(body, "b") == (1, 1)

    def test_blank_lines_between_matched_lines_are_skipped(self):
        body = "intro\npara one\n\npara two\noutro".splitlines()
        # The story's two paragraphs are separated by a blank body line.
        assert locate_story_span(body, "para one\n\npara two") == (1, 3)

    def test_unlocatable_text_returns_none(self):
        body = "a\nb\nc".splitlines()
        assert locate_story_span(body, "text that is nowhere in the body") is None

    def test_empty_story_text_returns_none(self):
        body = "a\nb".splitlines()
        assert locate_story_span(body, "   ") is None

    def test_first_matching_run_wins(self):
        body = "dup\nother\ndup".splitlines()
        assert locate_story_span(body, "dup") == (0, 0)

    def test_whitespace_variant_seed_still_locates(self):
        # An LLM seed may re-wrap/re-space a line; the substring fallback still
        # locates it as long as the sequence is contiguous.
        body = "The quick brown fox\njumped over it".splitlines()
        assert locate_story_span(body, "The quick brown fox\njumped over it") == (0, 1)

    def test_trailing_blank_body_lines_do_not_extend_span(self):
        body = "para\n\n\nnext".splitlines()
        assert locate_story_span(body, "para") == (0, 0)

    def test_exact_match_preferred_over_earlier_fuzzy_match(self):
        # A byte-for-byte verbatim slice must locate at its true region even
        # when an earlier region only matches via the whitespace-collapse
        # fallback — the exact run wins.
        body = ["a  b", "x", "a b", "x"]  # line 0 double-spaced, line 2 single
        assert locate_story_span(body, "a b\nx") == (2, 3)


class TestLocateStorySpans:
    def test_one_entry_per_story_with_none_for_unlocatable(self):
        nl = _newsletter("t", body="l0\nl1\nl2")
        nl.stories = [_story("t:0", text="l1"), _story("t:1", text="not here")]
        assert locate_story_spans(nl) == [(1, 1), None]


class TestStoryAtBodyLine:
    def test_returns_first_story_containing_the_line(self):
        spans = [(0, 2), (4, 6)]
        assert story_at_body_line(spans, 1) == 0
        assert story_at_body_line(spans, 5) == 1

    def test_returns_none_outside_all_spans(self):
        assert story_at_body_line([(0, 1)], 3) is None

    def test_first_wins_on_overlap(self):
        assert story_at_body_line([(0, 3), (2, 5)], 3) == 0


class TestStoryExcerpt:
    def test_short_text_returned_whole(self):
        assert story_excerpt("hello world", 40) == "hello world"

    def test_collapses_whitespace(self):
        assert story_excerpt("a\n  b\tc", 40) == "a b c"

    def test_long_text_truncated_with_ellipsis(self):
        out = story_excerpt("word " * 40, 20)
        assert len(out) <= 20
        assert out.endswith("…")

    def test_empty_text_placeholder(self):
        assert story_excerpt("   ", 40) == "(empty)"


class TestFormatStoryStrip:
    def test_lists_each_story_with_range_and_selected_marker(self):
        nl = _newsletter("t", body="l0\nl1\nl2")
        nl.stories = [_story("t:0", text="l0"), _story("t:1", text="l2")]
        spans = locate_story_spans(nl)
        lines = format_story_strip(nl, spans, selected=1, width=80)
        assert len(lines) == 2
        assert "L1-1" in lines[0]
        assert lines[1].startswith("▶")  # story 2 is selected

    def test_unlocatable_story_flagged(self):
        nl = _newsletter("t", body="l0")
        nl.stories = [_story("t:0", text="nowhere")]
        lines = format_story_strip(nl, locate_story_spans(nl), selected=None, width=80)
        assert "not found" in lines[0]

    def test_labeled_and_excluded_flags_shown(self):
        nl = _newsletter("t", body="l0")
        nl.stories = [_story("t:0", text="l0")]
        assign_scores_and_themes(
            nl.stories[0], {"simple": 2, "concrete": 2, "personal": 3, "dynamic": 1}, {}
        )
        nl.stories[0].excluded = True
        lines = format_story_strip(nl, locate_story_spans(nl), selected=None, width=80)
        assert "2/2/3/1" in lines[0]
        assert "EXCL" in lines[0]

    def test_no_stories_hint(self):
        nl = _newsletter("t", body="l0")
        lines = format_story_strip(nl, [], selected=None, width=80)
        assert len(lines) == 1
        assert "none" in lines[0].lower()


class TestSpanEdit:
    def test_begin_add_span_starts_at_cursor_in_mark_start(self):
        edit = begin_add_span(3)
        assert edit == SpanEdit(story_index=None, start=3, end=3, stage="mark-start")

    def test_begin_edit_span_seeds_from_located_span(self):
        edit = begin_edit_span([(2, 5)], 0)
        assert edit == SpanEdit(story_index=0, start=2, end=5, stage="adjust")

    def test_begin_edit_span_none_when_unlocatable(self):
        assert begin_edit_span([None], 0) is None

    def test_two_press_add_flow(self):
        # Enter marks the start (advances to mark-end), then Enter commits.
        edit = begin_add_span(1)
        edit, commit = span_mark(edit, 1)
        assert commit is False
        assert edit.stage == "mark-end"
        edit = span_cursor_moved(edit, 3)  # cursor drags the end
        edit, commit = span_mark(edit, 3)
        assert commit is True
        assert span_range(edit) == (1, 3)

    def test_mark_start_cursor_drags_both_ends(self):
        edit = span_cursor_moved(begin_add_span(1), 5)
        assert (edit.start, edit.end) == (5, 5)

    def test_adjust_commits_immediately(self):
        edit = SpanEdit(story_index=0, start=2, end=4, stage="adjust")
        edit, commit = span_mark(edit, 9)  # cursor ignored in adjust commit
        assert commit is True
        assert span_range(edit) == (2, 4)

    def test_set_start_and_end(self):
        edit = SpanEdit(story_index=0, start=2, end=4, stage="adjust")
        assert span_set_start(edit, 1).start == 1
        assert span_set_end(edit, 7).end == 7


class TestCommitSpanEdit:
    def test_new_story_appends_verbatim_slice(self):
        nl = _newsletter("t", body="l0\nl1\nl2\nl3")
        story = commit_span_edit(nl, SpanEdit(None, 1, 2, "mark-end"))
        assert story.text == "l1\nl2"
        assert nl.stories[-1] is story

    def test_edit_replaces_text_preserving_id_and_labels(self):
        nl = _newsletter("t", body="l0\nl1\nl2\nl3")
        nl.stories = [_story("t:5", text="l1")]
        assign_scores_and_themes(
            nl.stories[0], {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3},
            ["church"],
        )
        story = commit_span_edit(nl, SpanEdit(0, 1, 2, "adjust"))
        assert story.text == "l1\nl2"
        assert story.story_id == "t:5"  # id preserved
        assert story.expected_themes == {"church": "present"}  # labels preserved
        assert story.expected_scores is not None


class TestAcceptConfirmationMessage:
    def test_none_when_all_labeled(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a"])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}, []
        )
        assert accept_confirmation_message(nl) is None

    def test_warns_with_unlabeled_count(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b"])
        msg = accept_confirmation_message(nl)
        assert "2" in msg
        assert "y/N" in msg


class TestModeBars:
    def test_browse_bar_names_mode_and_core_keys(self):
        joined = " ".join(browse_mode_bar(_newsletter("t"), None, 200))
        assert "BROWSE" in joined
        assert "[a]dd" in joined
        assert "[c]" in joined  # accept + next
        assert "[r]eseed" in joined

    def test_browse_bar_shows_selected_story(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b"])
        joined = " ".join(browse_mode_bar(nl, 1, 200))
        assert "story 2" in joined

    def test_span_bar_names_stage(self):
        joined = " ".join(span_mode_bar(begin_add_span(0), 200))
        assert "SPAN" in joined
        assert "Enter" in joined

    def test_bars_wrap_to_narrow_width(self):
        lines = browse_mode_bar(_newsletter("t"), None, 40)
        assert len(lines) >= 2
        assert all(len(line) <= 40 for line in lines)


class TestBuildDetailRows:
    def _spans(self, nl):
        return locate_story_spans(nl)

    def test_header_rows_have_none_body_idx(self):
        nl = _newsletter("tR", body="body line one\nbody line two")
        rows = build_detail_rows(nl, 0, 1, 80, spans=self._spans(nl))
        assert rows[0].body_idx is None
        assert any(r.body_idx is None for r in rows)

    def test_long_body_line_wraps_sharing_body_idx(self):
        nl = _newsletter("tR", body="aaaa bbbb cccc dddd")
        rows = build_detail_rows(nl, 0, 1, 6, spans=self._spans(nl))
        body_rows = [r for r in rows if r.body_idx == 0]
        assert len(body_rows) >= 2
        assert all(len(r.text) <= 6 for r in body_rows)

    def test_distinct_body_idx_count_matches_body_lines(self):
        body = "line0\nline1\nline2"
        nl = _newsletter("tR", body=body)
        rows = build_detail_rows(nl, 0, 1, 80, spans=self._spans(nl))
        distinct = {r.body_idx for r in rows if r.body_idx is not None}
        assert len(distinct) == len(body.splitlines())

    def test_story_span_gets_marker_row_and_membership(self):
        nl = _newsletter("tT", body="b0\nb1\nb2")
        nl.stories = [_story("tT:0", text="b1")]
        rows = build_detail_rows(nl, 0, 1, 80, spans=self._spans(nl))
        markers = [r for r in rows if r.kind == "marker"]
        assert len(markers) == 1
        assert "Story 1" in markers[0].text
        # The body row for b1 is tagged with story 0.
        b1 = next(r for r in rows if r.body_idx == 1 and r.kind == "body")
        assert b1.story_idx == 0
        # b0/b2 are outside the span.
        b0 = next(r for r in rows if r.body_idx == 0 and r.kind == "body")
        assert b0.story_idx is None

    def test_story_text_not_shown_inline_in_header(self):
        # The redesign removed the inline story-text block from the header; the
        # body itself (highlighted) is the only place the story text appears.
        nl = _newsletter("tT", body="b0")
        nl.stories = [_story("tT:0", text="unique-marker-text")]
        rows = build_detail_rows(nl, 0, 1, 80, spans=self._spans(nl))
        header_text = " ".join(r.text for r in rows if r.kind == "header")
        assert "unique-marker-text" not in header_text


class TestConfirmStoryList:
    def test_confirm_marks_newsletter_reviewed(self):
        nl = _newsletter("tC")
        seed_stories(nl, ["a", "b"])
        assert nl.reviewed is False

        confirm_story_list(nl)

        assert nl.reviewed is True

    def test_confirm_preserves_existing_ids_after_delete(self):
        nl = _newsletter("tC")
        seed_stories(nl, ["a", "b", "c"])
        delete_story(nl, 1)

        confirm_story_list(nl)

        assert [s.story_id for s in nl.stories] == ["tC:0", "tC:2"]

    def test_confirm_repairs_duplicate_ids_with_fresh_unique_ones(self):
        nl = _newsletter("tC")
        nl.stories = [
            _story("tC:0", text="A"),
            _story("tC:0", text="A-dup"),
            _story("tC:1", text="B"),
        ]

        confirm_story_list(nl)

        ids = [s.story_id for s in nl.stories]
        assert len(set(ids)) == len(ids)
        assert ids[0] == "tC:0"
        assert ids[2] == "tC:1"
        assert ids[1] == "tC:2"


class TestAddStory:
    def test_add_after_delete_never_duplicates_an_existing_id(self):
        nl = _newsletter("t1")
        seed_stories(nl, ["a", "b", "c"])
        delete_story(nl, 0)

        added = add_story(nl, "d")

        ids = [s.story_id for s in nl.stories]
        assert len(set(ids)) == len(ids)
        assert added.story_id == "t1:3"

    def test_create_from_body_after_delete_never_duplicates_an_existing_id(self):
        nl = _newsletter("t1", body="l0\nl1")
        seed_stories(nl, ["a", "b", "c"])
        delete_story(nl, 0)

        story = create_story_from_body(nl, 0, 0)

        ids = [s.story_id for s in nl.stories]
        assert len(set(ids)) == len(ids)
        assert story.story_id == "t1:3"

    def test_appends_story_with_stable_id_and_leaves_unreviewed(self):
        nl = _newsletter("tXY")
        nl.stories = [_story("tXY:0")]

        add_story(nl, text="New text")

        assert len(nl.stories) == 2
        added = nl.stories[1]
        assert added.story_id == "tXY:1"
        assert added.text == "New text"
        assert added.reviewed is False
        assert nl.reviewed is False


class TestLoadSaveRoundTrip:
    def test_save_then_load_preserves_nested_stories(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        nl = _newsletter("t1")
        seed_stories(nl, ["a", "b"])
        assign_scores_and_themes(
            nl.stories[0],
            {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
            ["scripture"],
        )
        confirm_story_list(nl)

        save_golden_set([nl], path)
        loaded = load_golden_set(path)

        assert len(loaded) == 1
        assert loaded[0].thread_id == "t1"
        assert [s.text for s in loaded[0].stories] == ["a", "b"]
        assert loaded[0].stories[0].expected_tier == "excellent"
        assert loaded[0].reviewed is True

    def test_load_tolerates_blank_lines(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        nl = _newsletter("t1")
        save_golden_set([nl], path)
        with open(path, "a") as f:
            f.write("\n")

        loaded = load_golden_set(path)
        assert len(loaded) == 1


class TestSeedFromExtractor:
    def test_runs_extractor_output_through_parse_stories(self):
        nl = _newsletter("tX")
        raw = "STORY: first story\n\nSTORY: second story"

        captured = {}

        def fake_extract(body):
            captured["body"] = body
            return raw

        seed_from_extractor(nl, fake_extract)

        assert captured["body"] == "raw body"
        assert [s.text for s in nl.stories] == ["first story", "second story"]
        assert nl.seeded_from == "parse_stories"
        assert nl.reviewed is False

    def test_no_stories_seeds_empty_list(self):
        nl = _newsletter("tX")
        seed_from_extractor(nl, lambda body: "NO_STORIES")
        assert nl.stories == []
        assert nl.seeded_from == "parse_stories"


class TestSeedConfirmationMessage:
    def test_no_confirmation_needed_for_empty_story_list(self):
        assert seed_confirmation_message(_newsletter("t")) is None

    def test_warns_with_story_count_when_stories_exist(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b"])
        msg = seed_confirmation_message(nl)
        assert msg is not None
        assert "2" in msg
        assert "y/N" in msg

    def test_mentions_labeled_stories_at_risk(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b"])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}, []
        )
        assert "1 labeled" in seed_confirmation_message(nl)

    def test_singular_story_in_warning(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a"])
        msg = seed_confirmation_message(nl)
        assert "Replace 1 story " in msg
        assert "stories" not in msg


class TestConfirmStatusMessage:
    def test_all_labeled_confirms_plainly(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a"])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}, []
        )
        assert confirm_status_message(nl) == "Story list confirmed."

    def test_singular_unlabeled(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a"])
        msg = confirm_status_message(nl)
        assert "1 story still unlabeled" in msg
        assert "stories" not in msg

    def test_plural_unlabeled(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b"])
        assert "2 stories still unlabeled" in confirm_status_message(nl)


class TestSeedOutcomeMessage:
    def test_reports_story_count_on_success(self):
        assert "2" in seed_outcome_message("STORY: a\n\nSTORY: b", 2)

    def test_singular_story_count(self):
        assert seed_outcome_message("STORY: a", 1) == "Seeded 1 story."

    def test_distinguishes_no_stories_verdict(self):
        assert "NO_STORIES" in seed_outcome_message("NO_STORIES", 0)

    def test_distinguishes_unparseable_output(self):
        msg = seed_outcome_message("prose with no story blocks", 0)
        assert "NO_STORIES" not in msg
        assert "no parseable" in msg.lower()


class TestRowForBodyLine:
    def _rows(self, pairs):
        from evals.newsletter_label import DetailRow

        return [DetailRow(text, bi, None, "header" if bi is None else "body")
                for text, bi in pairs]

    def test_finds_first_row_carrying_body_idx(self):
        rows = self._rows([("hdr", None), ("a", 0), ("b", 1), ("b2", 1), ("c", 2)])
        assert row_for_body_line(rows, 1) == 2

    def test_missing_body_idx_falls_back_to_last_row(self):
        rows = self._rows([("hdr", None), ("a", 0)])
        assert row_for_body_line(rows, 99) == 1

    def test_empty_rows_fall_back_to_zero(self):
        assert row_for_body_line([], 3) == 0


class TestLabelProgress:
    def test_counts_labeled_over_total_excluding_excluded(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b", "c"])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}, []
        )
        nl.stories[2].excluded = True
        assert label_progress(nl) == (1, 2)

    def test_unlabeled_story_count_ignores_excluded(self):
        nl = _newsletter("t")
        seed_stories(nl, ["a", "b"])
        nl.stories[1].excluded = True
        assert unlabeled_story_count(nl) == 1


class TestFormatListRow:
    def test_shows_labeled_over_total_and_sender(self):
        nl = _newsletter("t", sender="paul@dm.org", subject="Fall update")
        seed_stories(nl, ["a", "b", "c"])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}, []
        )
        assign_scores_and_themes(
            nl.stories[1], {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}, []
        )
        row = format_list_row(nl)
        assert "2/3" in row
        assert "paul@dm.org" in row
        assert "Fall update" in row

    def test_reviewed_flag_leads_the_row(self):
        nl = _newsletter("t", reviewed=True)
        assert format_list_row(nl).startswith("Y")

    def test_excluded_marker_takes_precedence(self):
        nl = _newsletter("t", reviewed=True, excluded=True)
        assert format_list_row(nl).startswith("X")


class TestNewsletterExclusionVisibility:
    def test_detail_rows_flag_excluded_newsletter(self):
        nl = _newsletter("t", body="b0", excluded=True)
        rows = build_detail_rows(nl, 0, 1, 80, spans=locate_story_spans(nl))
        joined = " ".join(r.text for r in rows)
        assert "Excluded:" in joined

    def test_detail_rows_omit_excluded_line_when_not_excluded(self):
        nl = _newsletter("t", body="b0")
        rows = build_detail_rows(nl, 0, 1, 80, spans=locate_story_spans(nl))
        joined = " ".join(r.text for r in rows)
        assert "Excluded:" not in joined

    def test_mode_bar_mentions_newsletter_exclude_hotkey(self):
        joined = "  ".join(browse_mode_bar(_newsletter("t"), None, 500))
        assert "[X]" in joined


class TestFormatThemeLegend:
    def test_full_legend_when_it_fits(self):
        legend = format_theme_legend({}, 200)
        assert "[s]scripture" in legend
        assert "[d]disciple_making" in legend
        assert "Enter=done" in legend

    def test_shows_selected_theme_grade(self):
        legend = format_theme_legend({"scripture": "emphasized"}, 200)
        assert "scripture=E" in legend

    def test_narrow_terminal_gets_compact_legend_with_all_keys(self):
        legend = format_theme_legend({"scripture": "present"}, 60)
        assert len(legend) <= 60
        for key in "schvd":
            assert f"[{key}]" in legend
        assert "Enter" in legend


class TestWrapTextDisplayWidth:
    def test_emoji_wrap_respects_display_width(self):
        wrapped = wrap_text("🎉🎉🎉🎉", 4)
        assert wrapped == ["🎉🎉", "🎉🎉"]

    def test_no_characters_are_lost_when_wrapping_wide_text(self):
        text = "José y Anaïs 🎉🎊✨ celebraron juntos"
        wrapped = wrap_text(text, 10)
        assert "".join("".join(line.split()) for line in wrapped) == "".join(text.split())


class TestBuildDetailRowsDivider:
    def test_divider_fits_narrow_terminals(self):
        nl = _newsletter("tS", body="b0")
        rows = build_detail_rows(nl, 0, 1, 30, spans=locate_story_spans(nl))
        dividers = [r.text for r in rows if set(r.text) == {"="}]
        assert len(dividers) == 1
        assert len(dividers[0]) <= 30


class TestSeedFromExtractorRaw:
    def test_returns_stories_and_raw_output_for_status_reporting(self):
        nl = _newsletter("tX")
        raw = "STORY: first story"
        stories, returned_raw = seed_from_extractor(nl, lambda body: raw)
        assert returned_raw == raw
        assert [s.text for s in stories] == ["first story"]


class TestBuildExtractorClientConfig:
    def test_passes_config_timeout_and_1024_max_tokens_default(self, monkeypatch):
        import llm_client as llm_client_module
        from evals.newsletter_label import build_extractor

        captured = {}
        real_client = llm_client_module.LLMClient

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(llm_client_module, "LLMClient", spy)
        monkeypatch.setenv("NEWSLETTER_LLM_URL", "http://llm.example")
        monkeypatch.setenv("NEWSLETTER_LLM_API_KEY", "k")
        prompt = {"system": "s", "user_template": "{body}"}
        quality = {"system": "s", "user_template": "{text}"}
        config = {
            "newsletter": {
                "llm": {"model": "m", "timeout": 123},
                "prompts": {
                    "story_extraction": prompt,
                    "quality_assessment": quality,
                    "theme_classification": quality,
                },
            }
        }

        build_extractor(config)

        assert captured["timeout"] == 123
        assert captured["max_tokens"] == 1024


class TestBuildExtractorPreflight:
    def test_missing_endpoint_fails_fast_naming_env_vars(self, monkeypatch):
        from evals.newsletter_label import build_extractor

        monkeypatch.delenv("NEWSLETTER_LLM_URL", raising=False)
        monkeypatch.delenv("CLOUD_LLM_URL", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            build_extractor({"newsletter": {"llm": {"model": "m"}}})
        msg = str(exc_info.value)
        assert "NEWSLETTER_LLM_URL" in msg
        assert "CLOUD_LLM_URL" in msg
        assert "--edit" in msg


# ---------------------------------------------------------------------------
# Pilot UI tests — drive the real Textual app: key presses in, widget state,
# rendered content, and on-disk golden-set effects out.
# ---------------------------------------------------------------------------

SIZE = (100, 30)


def _scores(n=3):
    return {"simple": n, "concrete": n, "personal": n, "dynamic": n}


def _label_app(newsletters, tmp_path, extract_fn=None):
    from evals.newsletter_label import LabelApp

    return LabelApp(
        newsletters, newsletters, tmp_path / "golden.jsonl", extract_fn=extract_fn
    )


def _detail(app):
    from evals.newsletter_label import DetailScreen

    assert isinstance(app.screen, DetailScreen), f"not on detail: {app.screen!r}"
    return app.screen


def _status(app) -> str:
    from textual.widgets import Static

    return str(app.screen.query_one("#status", Static).render())


def _screen_text(app) -> str:
    from textual.widgets import Label, Static

    parts = [str(w.render()) for w in app.screen.query(Static)]
    parts += [str(w.render()) for w in app.screen.query(Label)]
    return "\n".join(parts)


def _cursor_row_text(app) -> str:
    from textual.widgets import Label

    screen = _detail(app)
    lv = screen.query_one("#rows")
    item = lv.children[lv.index]
    return str(item.query_one(Label).render())


async def _drain(app, pilot):
    """Wait for background workers (e.g. seeding) to finish, then settle."""
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _goto_body(pilot, app, body_idx):
    """Move the detail cursor onto the row for a given body line."""
    screen = _detail(app)
    target = row_for_body_line(screen._rows, body_idx)
    lv = screen.query_one("#rows")
    cur = lv.index or 0
    delta = target - cur
    for _ in range(abs(delta)):
        await pilot.press("down" if delta > 0 else "up")


class TestDetailAutoSeed:
    async def test_opening_unreviewed_story_less_newsletter_auto_seeds(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "STORY: b0")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _drain(app, pilot)
            assert [s.text for s in nl.stories] == ["b0"]

    async def test_no_auto_seed_when_stories_present(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        seed_stories(nl, ["existing"])
        calls = []
        app = _label_app([nl], tmp_path, extract_fn=lambda body: calls.append(1) or "STORY: x")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _drain(app, pilot)
            assert calls == []  # never extracted
            assert [s.text for s in nl.stories] == ["existing"]

    async def test_no_auto_seed_when_reviewed(self, tmp_path):
        nl = _newsletter("tg", body="b0", reviewed=True)
        calls = []
        app = _label_app([nl], tmp_path, extract_fn=lambda body: calls.append(1) or "STORY: x")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _drain(app, pilot)
            assert calls == []

    async def test_no_auto_seed_in_edit_mode(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=None)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _drain(app, pilot)
            assert nl.stories == []


class TestDetailReseed:
    async def test_reseed_over_existing_stories_requires_confirmation(self, tmp_path):
        nl = _newsletter("tg", body="b0\nb1")
        seed_stories(nl, ["b0"])
        assign_scores_and_themes(nl.stories[0], _scores(), ["scripture"])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "STORY: b1")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "r")
            await pilot.press("n")  # y/N: keep the curated story
            await _drain(app, pilot)
            assert [s.text for s in nl.stories] == ["b0"]
            assert nl.stories[0].expected_scores == _scores()
            assert "Seed cancelled" in _status(app)

    async def test_confirmed_reseed_replaces_and_reports_count(self, tmp_path):
        nl = _newsletter("tg", body="b0\nb1")
        seed_stories(nl, ["b0"])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "STORY: b1")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "r")
            await pilot.press("y")
            await _drain(app, pilot)
            assert [s.text for s in nl.stories] == ["b1"]
            assert "Seeded 1" in _status(app)

    async def test_no_stories_verdict_is_reported_distinctly(self, tmp_path):
        nl = _newsletter("tg", body="admin only")
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "NO_STORIES")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")  # auto-seed returns NO_STORIES
            await _drain(app, pilot)
            assert "NO_STORIES" in _status(app)

    async def test_edit_mode_reseed_key_reports_seeding_disabled(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        seed_stories(nl, ["b0"])
        app = _label_app([nl], tmp_path, extract_fn=None)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "r")
            await _drain(app, pilot)
            assert [s.text for s in nl.stories] == ["b0"]
            assert "edit mode" in _status(app).lower()

    async def test_seed_failure_keeps_stories_and_undo_clean(self, tmp_path):
        def boom(body):
            raise RuntimeError("connection dropped")

        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=boom)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")  # auto-seed fails
            await _drain(app, pilot)
            assert "Seed failed" in _screen_text(app)
            await pilot.press("x")  # any key dismisses the hint
            await _drain(app, pilot)
            assert nl.stories == []
            await pilot.press("z")
            assert "Nothing to undo" in _status(app)

    async def test_second_reseed_while_seeding_is_ignored(self, tmp_path):
        import threading

        release = threading.Event()
        calls = []

        def slow_extract(body):
            calls.append(1)
            release.wait(timeout=5)
            return "STORY: b0"

        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=slow_extract)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")  # auto-seed starts (in flight)
            await pilot.pause()
            await pilot.press("r")  # while the first seed is in flight
            release.set()
            await _drain(app, pilot)
            assert len(calls) == 1
            assert [s.text for s in nl.stories] == ["b0"]


class TestDetailUndo:
    async def test_undo_stack_restores_multiple_edits(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        seed_stories(nl, ["a", "b", "c"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "d")   # select story 1, delete
            await pilot.press("1", "d")   # select (new) story 1, delete
            assert [s.text for s in nl.stories] == ["c"]
            await pilot.press("z", "z")
            assert [s.text for s in nl.stories] == ["a", "b", "c"]

    async def test_cancelled_prompt_does_not_clobber_undo(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        seed_stories(nl, ["a", "b"])
        assign_scores_and_themes(nl.stories[1], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "d")          # delete unlabeled story A
            await pilot.press("1", "d", "n")     # try delete labeled B, cancel
            await pilot.press("z")               # undo must restore A
            assert [s.text for s in nl.stories] == ["a", "b"]

    async def test_undo_with_empty_stack_reports_nothing_to_undo(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "z")
            assert "Nothing to undo" in _status(app)


class TestDetailDelete:
    async def test_deleting_labeled_story_requires_confirmation(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, ["a"])
        assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "d", "n")
            assert [s.text for s in nl.stories] == ["a"]

    async def test_confirmed_delete_of_labeled_story_reports_what_was_deleted(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, ["a"])
        assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "d", "y")
            assert nl.stories == []
            assert "Deleted" in _status(app)

    async def test_delete_without_selection_reports_hint(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d")
            assert "select a story" in _status(app).lower()
            assert [s.text for s in nl.stories] == ["a"]

    async def test_autosave_persists_mutation_to_disk_before_quit(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, ["a", "b"])
        path = tmp_path / "golden.jsonl"
        save_golden_set([nl], path)
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "d")
            on_disk = load_golden_set(path)
            assert [s.text for s in on_disk[0].stories] == ["b"]


class TestDetailSpanNew:
    async def test_two_press_flow_makes_verbatim_multiline_story(self, tmp_path):
        nl = _newsletter("tA", body="para one\npara two\npara three")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 0)     # cursor on "para one"
            await pilot.press("a")              # enter span mode
            await pilot.press("enter")          # mark start at body 0
            await pilot.press("down")           # move end to body 1
            await pilot.press("enter")          # commit
            assert len(nl.stories) == 1
            assert nl.stories[0].text == "para one\npara two"

    async def test_add_story_from_header_row_anchors_to_body(self, tmp_path):
        # On open the cursor sits on the metadata header (body_idx None).
        # Pressing `a` there must move the cursor onto the first body line so the
        # two-press flow can commit, instead of stranding it on the header where
        # Enter can never mark a boundary.
        nl = _newsletter("tA", body="b0\nb1")
        app = _label_app([nl], tmp_path)  # no extract_fn -> no auto-seed
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")           # open detail; cursor on header
            await pilot.press("a")               # add-story from the header row
            await pilot.press("enter")           # mark start (first body line)
            await pilot.press("enter")           # mark end -> commit
            assert [s.text for s in nl.stories] == ["b0"]

    async def test_esc_cancels_span_without_creating_a_story(self, tmp_path):
        nl = _newsletter("tA", body="para one\npara two")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 0)
            await pilot.press("a", "enter", "escape")
            assert nl.stories == []
            assert _detail(app).mode == "browse"
            assert "cancelled" in _status(app).lower()

    async def test_s_key_overrides_the_span_start_mid_flow(self, tmp_path):
        # After marking the start at body 1 and moving the end to body 3,
        # pressing s re-sets the start to the cursor (body 3), so the committed
        # story is just body line 3.
        nl = _newsletter("tA", body="l0\nl1\nl2\nl3")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 1)
            await pilot.press("a")          # enter span mode (mark-start at body 1)
            await pilot.press("enter")      # fix start at body 1
            await _goto_body(pilot, app, 3)  # end tracks to body 3
            await pilot.press("s")          # re-set start to the cursor (body 3)
            await pilot.press("enter")      # commit start=3, end=3
            assert len(nl.stories) == 1
            assert nl.stories[0].text == "l3"


class TestDetailSpanEdit:
    async def test_edit_extends_selected_story_end_preserving_labels(self, tmp_path):
        nl = _newsletter("tE", body="l0\nl1\nl2\nl3")
        seed_stories(nl, ["l1"])  # story spans body line 1 only
        assign_scores_and_themes(nl.stories[0], _scores(), ["church"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1")          # select story 1
            await pilot.press("e")          # edit boundaries (adjust, cursor at end=body1)
            await pilot.press("down")       # move end to body 2
            await pilot.press("enter")      # commit
            assert nl.stories[0].text == "l1\nl2"
            assert nl.stories[0].expected_themes == {"church": "present"}  # labels preserved
            assert nl.stories[0].expected_scores == _scores()

    async def test_in_span_e_sets_end_not_start(self, tmp_path):
        # Distinguishes span_set_end from span_set_start at the keystroke level:
        # in adjust mode, `e` at the cursor sets the END, keeping the start.
        nl = _newsletter("tE", body="l0\nl1\nl2\nl3")
        seed_stories(nl, ["l1"])  # story spans body line 1
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "e")     # adjust, start=end=body1
            await _goto_body(pilot, app, 3)  # cursor to body 3
            await pilot.press("e")          # set END to body 3 (start stays 1)
            await pilot.press("enter")      # commit
            # If `e` had set the START, the text would be just "l3".
            assert nl.stories[0].text == "l1\nl2\nl3"

    async def test_edit_unlocatable_story_falls_back_to_text_prompt(self, tmp_path):
        nl = _newsletter("tE", body="l0\nl1")
        seed_stories(nl, ["text that is not in the body"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "e")     # can't locate -> text prompt
            from textual.widgets import Input

            app.screen.query_one(Input).value = "corrected text"
            await pilot.press("enter")
            await _drain(app, pilot)
            assert nl.stories[0].text == "corrected text"

    async def test_edit_unlocatable_accept_prefill_keeps_multiline_text(self, tmp_path):
        # An unlocatable MULTI-line story falls back to the single-line text
        # prompt, prefilled with a whitespace-collapsed copy. Accepting that
        # prefill unchanged must keep the original multi-line text rather than
        # silently flattening the newlines into spaces.
        nl = _newsletter("tE", body="l0\nl1")
        seed_stories(nl, ["multi\nline story not in body"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "e")     # can't locate -> text prompt (prefill collapsed)
            await pilot.press("enter")      # accept the prefill unchanged
            await _drain(app, pilot)
            assert nl.stories[0].text == "multi\nline story not in body"

    async def test_unlocatable_fallback_escape_is_a_clean_cancel(self, tmp_path):
        # Escaping the fallback text prompt must not mutate the story or push an
        # undo level.
        nl = _newsletter("tE", body="l0\nl1")
        seed_stories(nl, ["text that is not in the body"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "e", "escape")
            await _drain(app, pilot)
            assert nl.stories[0].text == "text that is not in the body"
            assert "cancelled" in _status(app).lower()
            await pilot.press("z")          # nothing was pushed to undo
            assert "Nothing to undo" in _status(app)

    async def test_unlocatable_fallback_unchanged_text_is_a_noop(self, tmp_path):
        # Submitting the prefilled (unchanged) text is a no-op: no undo, no save.
        nl = _newsletter("tE", body="l0\nl1")
        seed_stories(nl, ["text that is not in the body"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "e")     # prompt prefilled with the story text
            await pilot.press("enter")      # submit unchanged
            await _drain(app, pilot)
            assert nl.stories[0].text == "text that is not in the body"
            await pilot.press("z")
            assert "Nothing to undo" in _status(app)

    async def test_edit_without_selection_reports_hint(self, tmp_path):
        nl = _newsletter("tE", body="l0")
        seed_stories(nl, ["l0"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "e")
            assert "select a story" in _status(app).lower()


class TestDetailClearAll:
    async def test_clear_all_after_confirm_empties_stories(self, tmp_path):
        nl = _newsletter("tX", body="l0\nl1")
        seed_stories(nl, ["l0", "l1"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("C", "y")
            await _drain(app, pilot)
            assert nl.stories == []
            await pilot.press("z")          # undoable
            assert [s.text for s in nl.stories] == ["l0", "l1"]

    async def test_clear_all_cancel_keeps_stories(self, tmp_path):
        nl = _newsletter("tX", body="l0")
        seed_stories(nl, ["l0"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("C", "n")
            await _drain(app, pilot)
            assert [s.text for s in nl.stories] == ["l0"]


class TestDetailSelection:
    async def test_number_and_np_keys_select_stories(self, tmp_path):
        nl = _newsletter("tS", body="l0\nl1\nl2")
        seed_stories(nl, ["l0", "l1", "l2"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("2")
            assert _detail(app).selected_story == 1
            await pilot.press("n")          # next
            assert _detail(app).selected_story == 2
            await pilot.press("p")          # prev
            assert _detail(app).selected_story == 1

    async def test_strip_marks_selected_story(self, tmp_path):
        nl = _newsletter("tS", body="l0\nl1")
        seed_stories(nl, ["l0", "l1"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "2")
            from textual.widgets import Static

            strip = str(app.screen.query_one("#storystrip", Static).render())
            assert "▶2." in strip

    async def test_enter_on_a_story_body_row_selects_it(self, tmp_path):
        nl = _newsletter("tS", body="l0\nl1\nl2")
        seed_stories(nl, ["l1"])  # story on body line 1
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 1)
            await pilot.press("enter")      # browse: select story under cursor
            assert _detail(app).selected_story == 0


class TestDetailCursorAnchor:
    async def test_cursor_stays_on_its_body_line_after_making_a_story(self, tmp_path):
        nl = _newsletter("tA", body="para one\npara two\npara three")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 1)     # onto "para two"
            await pilot.press("a", "enter", "enter")  # span body1..body1, commit
            assert len(nl.stories) == 1
            assert nl.stories[0].text == "para two"
            assert "para two" in _cursor_row_text(app)


class TestDetailRowCache:
    def _counting(self, monkeypatch):
        import evals.newsletter_label as label

        calls = {"rows": 0, "spans": 0}
        real_rows = label.build_detail_rows
        real_spans = label.locate_story_spans

        def counting_rows(*args, **kwargs):
            calls["rows"] += 1
            return real_rows(*args, **kwargs)

        def counting_spans(*args, **kwargs):
            calls["spans"] += 1
            return real_spans(*args, **kwargs)

        monkeypatch.setattr(label, "build_detail_rows", counting_rows)
        monkeypatch.setattr(label, "locate_story_spans", counting_spans)
        return calls

    async def test_pure_cursor_movement_reuses_cached_rows(self, tmp_path, monkeypatch):
        calls = self._counting(monkeypatch)
        nl = _newsletter("tP", body="b0\nb1\nb2\nb3")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("down", "down", "down", "down", "up", "up")
            assert calls["rows"] == 1
            assert calls["spans"] == 1

    async def test_mutation_rebuilds_rows(self, tmp_path, monkeypatch):
        calls = self._counting(monkeypatch)
        nl = _newsletter("tP", body="b0\nb1")
        seed_stories(nl, ["b0"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "d")
            assert calls["rows"] == 2  # initial build + rebuild after delete

    async def test_resize_rewraps_body(self, tmp_path):
        words = [f"word{i:02d}" for i in range(24)]
        nl = _newsletter("tP", body=" ".join(words))
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("enter")
            await pilot.resize_terminal(40, 30)
            await pilot.pause()
            from textual.widgets import Label

            lv = _detail(app).query_one("#rows")
            row_texts = [str(item.query_one(Label).render()) for item in lv.children]
            rendered = " ".join(row_texts)
            for word in words:
                assert word in rendered
            assert all(len(t) <= 40 for t in row_texts), max(row_texts, key=len)


class TestDetailNotes:
    async def test_notes_prompt_prefills_existing_note(self, tmp_path):
        nl = _newsletter("tN", body="b0", notes="important note")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("N", "enter")  # Shift-N; Enter keeps
            assert nl.notes == "important note"
            assert "unchanged" in _status(app).lower()


class TestDetailLabeling:
    async def test_label_flow_assigns_scores_and_saves(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "l")           # select story 1, label
            await pilot.press("1", "2", "3", "2")  # four dimension scores (Poor/OK/Good=1/2/3)
            await pilot.press("enter")             # finish themes
            assert nl.stories[0].expected_scores == {
                "simple": 1, "concrete": 2, "personal": 3, "dynamic": 2,
            }
            assert nl.stories[0].reviewed is True

    async def test_label_without_selection_reports_hint(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "l")
            assert "select a story" in _status(app).lower()

    async def test_relabeling_shows_current_scores(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a"])
        assign_scores_and_themes(
            nl.stories[0],
            {"simple": 2, "concrete": 3, "personal": 3, "dynamic": 1},
            {},
        )
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "l")
            assert "now 2" in _screen_text(app)
            await pilot.press("x")  # cancel

    async def test_cancelled_score_entry_reports_it(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "l")
            await pilot.press("1", "x")  # any out-of-range/other key cancels everything
            assert nl.stories[0].expected_scores is None
            assert "cancelled" in _status(app).lower()

    async def test_theme_cycle_absent_present_emphasized(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "l")
            await pilot.press("1", "2", "3", "2")   # dimension scores
            # scripture cycles present->emphasized (2 presses); christlikeness present;
            # church present->emphasized->absent (3 presses, removed).
            await pilot.press("s", "s", "c", "h", "h", "h")
            await pilot.press("enter")
            assert nl.stories[0].expected_themes == {
                "scripture": "emphasized",
                "christlikeness": "present",
            }


class TestDetailAccept:
    async def test_accept_marks_reviewed_and_advances(self, tmp_path):
        nls = [_newsletter("t0", body="b0"), _newsletter("t1", body="b1")]
        seed_stories(nls[0], ["b0"])
        assign_scores_and_themes(nls[0].stories[0], _scores(), [])
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")     # open t0
            await pilot.press("c")         # accept + advance
            await _drain(app, pilot)
            assert nls[0].reviewed is True
            assert "Newsletter 2/2" in _screen_text(app)

    async def test_accept_warns_about_unlabeled_stories(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a", "b"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("c")         # warns: 2 unlabeled
            await pilot.press("y")         # accept anyway
            await _drain(app, pilot)
            assert nl.reviewed is True

    async def test_accept_warning_declined_keeps_editing(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("c", "n")    # decline the warning
            await _drain(app, pilot)
            assert nl.reviewed is False
            assert isinstance(app.screen, _detail(app).__class__)

    async def test_accept_on_last_newsletter_returns_to_list(self, tmp_path):
        from evals.newsletter_label import DetailScreen

        nl = _newsletter("t0", body="b0")
        seed_stories(nl, ["b0"])
        assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "c")
            await _drain(app, pilot)
            assert not isinstance(app.screen, DetailScreen)
            assert "Y" in _screen_text(app)  # reviewed flag updated in list

    async def test_mashed_accept_does_not_crash_or_double_advance(self, tmp_path):
        # Auto-repeating `c` on a fully-labeled newsletter (accept has no confirm
        # modal, so nothing awaits) must not double-dismiss the popped screen
        # (a crash) nor accept-and-skip the next newsletter.
        from textual import events

        nls = [_newsletter(f"t{i}", body=f"b{i}") for i in range(3)]
        for nl in nls:
            seed_stories(nl, [f"b{nls.index(nl)}"])
            assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")           # open t0
            app.post_message(events.Key("c", "c"))
            app.post_message(events.Key("c", "c"))
            await pilot.pause()
            await pilot.pause()
            assert app.is_running                 # no crash (was a double-dismiss)
            assert nls[0].reviewed is True
            assert nls[1].reviewed is False       # second `c` did NOT accept t1
            assert "Newsletter 2/2" not in _screen_text(app)  # not skipped past t1
            assert "Newsletter 2/3" in _screen_text(app)      # advanced exactly one

    async def test_mashed_accept_does_not_re_run_on_the_popped_screen(self, tmp_path, monkeypatch):
        # The await-free accept path releases the _busy latch before the second
        # queued `c` worker runs; that worker must NOT re-confirm/save/refresh the
        # already-dismissed screen (a latent NoMatches crash + a redundant write).
        from textual import events

        import evals.newsletter_label as NL

        calls = []
        orig = NL.confirm_story_list
        monkeypatch.setattr(NL, "confirm_story_list", lambda nl: calls.append(nl) or orig(nl))

        nls = [_newsletter(f"t{i}", body=f"b{i}") for i in range(3)]
        for i, nl in enumerate(nls):
            seed_stories(nl, [f"b{i}"])
            assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")            # open t0 (fully labeled -> no confirm)
            app.post_message(events.Key("c", "c"))
            app.post_message(events.Key("c", "c"))
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()
            assert app.is_running
            # Exactly one accept committed: the second worker bailed instead of
            # re-running confirm_story_list on the popped t0 screen.
            assert len(calls) == 1


class TestDetailStoryExclusion:
    async def test_u_toggles_selected_story_exclusion(self, tmp_path):
        nl = _newsletter("tX", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "u")
            assert nl.stories[0].excluded is True
            assert "excluded from" in _status(app)
            await pilot.press("1", "u")
            assert nl.stories[0].excluded is False
            assert "included in" in _status(app)

    async def test_shift_x_toggles_newsletter_exclusion(self, tmp_path):
        nl = _newsletter("tX", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("X")
            assert nl.excluded is True
            await pilot.press("z")  # exclusion is undoable
            assert nl.excluded is False


class TestDetailModeBar:
    async def test_browse_mode_bar_on_screen(self, tmp_path):
        nl = _newsletter("tH", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "BROWSE" in _screen_text(app)

    async def test_span_mode_bar_on_screen(self, tmp_path):
        nl = _newsletter("tH", body="b0\nb1")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 0)
            await pilot.press("a")
            assert "SPAN" in _screen_text(app)


class TestDetailModeGating:
    async def test_browse_action_keys_are_inert_in_span_mode(self, tmp_path):
        # While marking a span, the browse action keys (delete / accept / clear /
        # label / select) must not fire — only span controls act.
        nl = _newsletter("tG", body="l0\nl1\nl2")
        seed_stories(nl, ["l0"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 1)
            await pilot.press("a")               # enter span mode
            assert _detail(app).mode == "span"
            await pilot.press("d", "c", "C", "l", "1", "k")  # all inert here
            await pilot.pause()
            assert _detail(app).mode == "span"   # still marking; nothing happened
            assert nl.stories == [nl.stories[0]]  # story list untouched
            assert nl.reviewed is False           # accept did not fire
            from evals.newsletter_label import DetailScreen

            assert isinstance(app.screen, DetailScreen)  # no modal stacked

    async def test_state_changing_browse_keys_are_inert_in_span_mode(self, tmp_path):
        # The remaining span-guarded browse keys (r reseed, n/p select, N notes,
        # u exclude, z undo, X exclude-nl) must not mutate state while a span is
        # being marked — a stray undo/reseed there would desync span_edit's
        # story_index from a rebuilt story list.
        nl = _newsletter("tG", body="l0\nl1\nl2", notes="keep")
        seed_stories(nl, ["l0", "l2"])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "STORY: l1")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1")               # select story 1 (build undo isn't needed)
            await pilot.press("e")               # enter span mode (adjust story 1)
            assert _detail(app).mode == "span"
            span_before = _detail(app).span_edit
            sel_before = _detail(app).selected_story
            await pilot.press("r", "n", "p", "N", "u", "z", "X")
            await pilot.pause()
            assert _detail(app).mode == "span"        # still marking
            assert _detail(app).span_edit == span_before   # boundary untouched
            assert _detail(app).selected_story == sel_before
            assert [s.text for s in nl.stories] == ["l0", "l2"]  # r/z didn't fire
            assert nl.notes == "keep"                 # N didn't fire
            assert nl.excluded is False               # X didn't fire
            assert nl.stories[0].excluded is False    # u didn't fire


class TestDetailSpanResize:
    async def test_resize_mid_span_mark_preserves_the_boundary(self, tmp_path):
        # Resizing while marking a span must not drag the in-progress end to
        # whatever body line the old physical row maps to after the rewrap.
        words = [f"word{i:02d}" for i in range(24)]
        nl = _newsletter("tR", body="\n".join(words))
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("enter")
            await _goto_body(pilot, app, 0)
            await pilot.press("a", "enter")   # mark start at body 0 (stage mark-end)
            await _goto_body(pilot, app, 5)   # end tracks to body 5
            assert _detail(app).span_edit.end == 5
            await pilot.resize_terminal(40, 30)
            await pilot.pause()
            assert _detail(app).span_edit.end == 5   # boundary unchanged by resize


class TestListView:
    async def test_list_shows_newsletters_and_q_quits(self, tmp_path):
        from textual.widgets import ListView

        nls = [_newsletter(f"t{i}", subject=f"subject-{i}") for i in range(3)]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            assert len(app.query_one(ListView)) == 3
            assert "3 newsletters" in _screen_text(app)
            await pilot.press("q")
        assert app.return_value == "quit"

    async def test_page_down_moves_cursor_by_a_page(self, tmp_path):
        from textual.widgets import ListView

        nls = [_newsletter(f"t{i}", subject=f"subject-{i}") for i in range(10)]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=(100, 8)) as pilot:
            await pilot.press("pagedown")
            index = app.query_one(ListView).index
            assert index and index > 1

    async def test_enter_opens_detail_and_esc_returns(self, tmp_path):
        from evals.newsletter_label import DetailScreen

        nls = [_newsletter("t0", subject="first"), _newsletter("t1", subject="second")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("down", "enter")
            assert isinstance(app.screen, DetailScreen)
            assert "Newsletter 2/2" in _screen_text(app)
            await pilot.press("escape")
            assert not isinstance(app.screen, DetailScreen)

    async def test_skip_advances_to_next_newsletter(self, tmp_path):
        from evals.newsletter_label import DetailScreen

        nls = [_newsletter("t0"), _newsletter("t1")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "Newsletter 1/2" in _screen_text(app)
            await pilot.press("k")
            assert isinstance(app.screen, DetailScreen)
            assert "Newsletter 2/2" in _screen_text(app)
            assert nls[0].reviewed is False
            await pilot.press("k")
            assert not isinstance(app.screen, DetailScreen)

    async def test_q_from_detail_quits_whole_app(self, tmp_path):
        nls = [_newsletter("t0")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "q")
        assert app.return_value == "quit"

    async def test_list_flags_refresh_after_accept_in_detail(self, tmp_path):
        nls = [_newsletter("t0"), _newsletter("t1")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "c")   # accept t0 -> advances to t1
            await _drain(app, pilot)
            await pilot.press("escape")       # back to list
            assert nls[0].reviewed is True
            assert "Y" in _screen_text(app)


class TestCli:
    def test_excluded_not_queued_and_preserved_on_save(self, tmp_path, monkeypatch):
        import evals.newsletter_label as label

        path = tmp_path / "golden.jsonl"
        save_golden_set(
            [
                _newsletter("normal", reviewed=False),
                _newsletter("excluded", reviewed=True, excluded=True),
            ],
            path,
        )

        seen = {}

        def fake_loop(newsletters, all_newsletters, p, **kwargs):
            seen["ids"] = [n.thread_id for n in newsletters]
            for n in newsletters:
                n.reviewed = True

        monkeypatch.setattr(label, "label_loop", fake_loop)
        monkeypatch.setattr(
            sys, "argv", ["newsletter_label", "--golden-set", str(path), "--edit"]
        )
        label.cli()

        assert seen["ids"] == ["normal"]
        saved = {n.thread_id: n for n in load_golden_set(path)}
        assert set(saved) == {"normal", "excluded"}
        assert saved["excluded"].excluded is True
        assert saved["normal"].reviewed is True

    def test_missing_golden_set_hint_mentions_matching_harvest_output(
        self, tmp_path, monkeypatch, capsys
    ):
        import evals.newsletter_label as label

        monkeypatch.setattr(
            sys, "argv",
            ["newsletter_label", "--golden-set", str(tmp_path / "nope.jsonl"), "--edit"],
        )
        with pytest.raises(SystemExit):
            label.cli()

        err = capsys.readouterr().err
        assert "--output" in err
        assert "--golden-set" in err

    def test_unreviewed_only_flag_filters_queue(self, tmp_path, monkeypatch):
        import evals.newsletter_label as label

        path = tmp_path / "golden.jsonl"
        save_golden_set(
            [
                _newsletter("done", reviewed=True),
                _newsletter("todo", reviewed=False),
            ],
            path,
        )

        seen = {}

        def fake_loop(newsletters, all_newsletters, p, **kwargs):
            seen["ids"] = [n.thread_id for n in newsletters]

        monkeypatch.setattr(label, "label_loop", fake_loop)
        monkeypatch.setattr(
            sys, "argv",
            ["newsletter_label", "--golden-set", str(path), "--unreviewed-only", "--edit"],
        )
        label.cli()

        assert seen["ids"] == ["todo"]


class TestModalRobustness:
    async def test_score_screen_survives_held_digit_key(self, tmp_path):
        from textual import events

        nl = _newsletter("tR", body="b0")
        seed_stories(nl, ["a"])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("1", "l")
            for _ in range(6):
                app.post_message(events.Key("3", "3"))
            await pilot.pause()
            assert app.is_running
            await pilot.press("enter")  # finish themes
            assert nl.stories[0].expected_scores == {
                "simple": 3, "concrete": 3, "personal": 3, "dynamic": 3,
            }

    async def test_prompt_line_survives_double_enter(self, tmp_path):
        from textual import events

        nl = _newsletter("tR", body="b0", notes="keep")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "N")
            app.post_message(events.Key("enter", None))
            app.post_message(events.Key("enter", None))
            await pilot.pause()
            assert app.is_running
            assert nl.notes == "keep"
            from evals.newsletter_label import DetailScreen

            assert isinstance(app.screen, DetailScreen)

    async def test_confirm_screen_survives_double_key(self, tmp_path):
        from textual import events

        nl = _newsletter("tR", body="b0")
        seed_stories(nl, ["b0"])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "STORY: x")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "r")
            app.post_message(events.Key("n", "n"))
            app.post_message(events.Key("n", "n"))
            await _drain(app, pilot)
            assert app.is_running
            assert [s.text for s in nl.stories] == ["b0"]
            from evals.newsletter_label import DetailScreen

            assert isinstance(app.screen, DetailScreen)

    async def test_mashed_action_key_opens_a_single_prompt(self, tmp_path):
        from textual import events

        nl = _newsletter("tR", body="b0")
        seed_stories(nl, ["a", "b"])
        assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "1")   # select labeled story A
            app.post_message(events.Key("d", "d"))
            app.post_message(events.Key("d", "d"))
            await pilot.pause()
            await pilot.press("y")            # confirm the single delete prompt
            await pilot.pause()
            assert app.is_running
            assert [s.text for s in nl.stories] == ["b"]
            from evals.newsletter_label import DetailScreen

            assert isinstance(app.screen, DetailScreen)

    async def test_notes_control_characters_never_persist(self, tmp_path):
        from textual.widgets import Input

        nl = _newsletter("tR", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "N")
            app.screen.query_one(Input).value = "good\x1b[31mbad\x07note"
            await pilot.press("enter")
            assert "\x1b" not in nl.notes
            assert "\x07" not in nl.notes
            assert "good" in nl.notes and "note" in nl.notes


class TestSeedAbort:
    async def test_escape_during_seed_discards_the_seed_result(self, tmp_path):
        import threading

        release = threading.Event()

        def slow_extract(body):
            release.wait(timeout=5)
            return "STORY: Machine seed"

        nl = _newsletter("tA", body="b0")
        seed_stories(nl, ["Hand curated"])
        nl.reviewed = True
        app = _label_app([nl], tmp_path, extract_fn=slow_extract)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "r", "y")  # confirm reseed, in flight
            await pilot.pause()
            await pilot.press("escape")            # abandon mid-seed
            release.set()
            await _drain(app, pilot)
            assert [s.text for s in nl.stories] == ["Hand curated"]
            assert nl.reviewed is True


class TestStatusLifecycle:
    async def test_status_clears_on_next_navigation_keypress(self, tmp_path):
        nl = _newsletter("tS", body="b0\nb1")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "z")  # -> "Nothing to undo."
            assert "Nothing to undo" in _status(app)
            await pilot.press("down")
            assert "Nothing to undo" not in _status(app)
            assert "row 2/" in _status(app)


class TestListPaging:
    async def test_page_down_lands_exactly_one_page_down(self, tmp_path):
        from textual.widgets import ListView

        nls = [_newsletter(f"t{i}", subject=f"subject-{i}") for i in range(10)]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=(100, 8)) as pilot:
            await pilot.press("pagedown")
            assert app.query_one(ListView).index == 5
            await pilot.press("end")
            assert app.query_one(ListView).index == 9
            await pilot.press("home")
            assert app.query_one(ListView).index == 0


class TestLabelAppReviewFindings:
    async def test_enter_auto_repeat_opens_a_single_detail(self, tmp_path):
        from textual import events

        from evals.newsletter_label import DetailScreen

        nls = [_newsletter("t0"), _newsletter("t1")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            app.post_message(events.Key("enter", None))
            app.post_message(events.Key("enter", None))
            await pilot.pause()
            assert len(app.screen_stack) == 2  # base + ONE detail
            await pilot.press("escape")
            assert not isinstance(app.screen, DetailScreen)
