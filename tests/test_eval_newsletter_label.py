"""Tests for evals.newsletter_label — pure state transitions + Pilot UI tests.

Every state transition lives in a pure function that mutates a
GoldenNewsletter / GoldenStory in place, tested directly without a terminal.
The Textual UI layer on top is driven with Textual's Pilot: real key presses
in, widget state / rendered content / on-disk golden-set effects out.
"""

import sys

import pytest

from evals.newsletter_label import (
    add_story,
    assign_scores_and_themes,
    build_detail_rows,
    capture_snapshot,
    confirm_status_message,
    confirm_story_list,
    covered_body_lines,
    create_story_from_body,
    delete_story,
    edit_story,
    exclude_story,
    format_help_lines,
    format_list_row,
    format_theme_legend,
    label_progress,
    load_golden_set,
    newsletter_exclude_status,
    restore_snapshot,
    row_for_body_line,
    save_golden_set,
    seed_confirmation_message,
    seed_from_extractor,
    seed_outcome_message,
    seed_stories,
    select_label_newsletters,
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
    base = dict(story_id=story_id, title="Title", text="Text")
    base.update(kw)
    return GoldenStory(**base)


class TestSeedStories:
    def test_seeds_candidates_with_stable_ids_and_provenance(self):
        nl = _newsletter("tS")
        pairs = [("A", "text a"), ("B", "text b")]

        seed_stories(nl, pairs)

        assert [s.story_id for s in nl.stories] == ["tS:0", "tS:1"]
        assert [(s.title, s.text) for s in nl.stories] == pairs
        assert nl.seeded_from == "parse_stories"
        assert nl.reviewed is False  # seeding is not confirmation
        assert all(not s.reviewed for s in nl.stories)

    def test_seeding_a_reviewed_newsletter_clears_reviewed(self):
        # Re-seeding replaces the confirmed story list with an uncurated
        # machine seed — it must NOT stay authoritative extraction truth for
        # reviewed-only runs. Undo still restores via the snapshot.
        nl = _newsletter("tS", reviewed=True)

        seed_stories(nl, [("A", "text a")])

        assert nl.reviewed is False

    def test_seeding_replaces_prior_candidates(self):
        nl = _newsletter("tS")
        nl.stories = [_story("tS:0", title="old")]

        seed_stories(nl, [("new", "new text")])

        assert [s.title for s in nl.stories] == ["new"]
        assert nl.stories[0].story_id == "tS:0"


class TestAssignScoresAndThemes:
    def test_assigns_scores_themes_reviewed_and_derives_tier(self):
        story = _story()
        scores = {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}

        assign_scores_and_themes(story, scores, ["scripture", "church"])

        assert story.expected_scores == scores
        assert story.expected_themes == ["scripture", "church"]
        assert story.reviewed is True
        # tier auto-derived from scores (avg 4.0 -> excellent)
        assert story.expected_tier == compute_tier(scores).value
        assert story.expected_tier == NewsletterTier.EXCELLENT.value

    def test_derives_fair_tier_for_midrange_scores(self):
        story = _story()
        # avg 2.5 -> fair (>=2.0, <3.0)
        scores = {"simple": 2, "concrete": 3, "personal": 2, "dynamic": 3}

        assign_scores_and_themes(story, scores, [])

        assert story.expected_tier == "fair"

    def test_scores_are_copied_not_aliased(self):
        story = _story()
        scores = {"simple": 1, "concrete": 1, "personal": 1, "dynamic": 1}

        assign_scores_and_themes(story, scores, [])
        scores["simple"] = 5  # mutate caller's dict

        assert story.expected_scores["simple"] == 1  # snapshot, not alias


class TestSelectLabelNewsletters:
    def test_excludes_excluded_newsletters(self):
        nls = [_newsletter("keep"), _newsletter("drop", excluded=True)]
        selected = select_label_newsletters(nls)
        assert [n.thread_id for n in selected] == ["keep"]

    def test_unreviewed_only_drops_reviewed(self):
        nls = [
            _newsletter("a", reviewed=False),
            _newsletter("b", reviewed=True),
        ]
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
        assert "restore" in newsletter_exclude_status(nl).lower()  # X to restore
        assert "excluded" in newsletter_exclude_status(nl).lower()
        nl.excluded = False
        assert "restored" in newsletter_exclude_status(nl).lower()


class TestUndo:
    def test_undo_restores_prior_story_list_and_reviewed(self):
        nl = _newsletter("tU")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        snap = capture_snapshot(nl)

        # Mutate: delete a story, edit another, confirm the list.
        delete_story(nl, 1)
        edit_story(nl, 0, title="A-edited")
        confirm_story_list(nl)
        assert nl.reviewed is True
        assert [s.title for s in nl.stories] == ["A-edited"]

        restore_snapshot(nl, snap)

        assert nl.reviewed is False
        assert [(s.title, s.text) for s in nl.stories] == [("A", "a"), ("B", "b")]

    def test_undo_restores_per_story_labels(self):
        nl = _newsletter("tU")
        seed_stories(nl, [("A", "a")])
        snap = capture_snapshot(nl)

        assign_scores_and_themes(
            nl.stories[0],
            {"simple": 5, "concrete": 5, "personal": 5, "dynamic": 5},
            ["scripture"],
        )
        assert nl.stories[0].reviewed is True

        restore_snapshot(nl, snap)

        assert nl.stories[0].reviewed is False
        assert nl.stories[0].expected_scores is None
        assert nl.stories[0].expected_themes == []

    def test_snapshot_is_independent_of_later_mutations(self):
        # Capturing then mutating a story must not bleed into the snapshot.
        nl = _newsletter("tU")
        seed_stories(nl, [("A", "a")])
        snap = capture_snapshot(nl)
        nl.stories[0].title = "changed"

        restore_snapshot(nl, snap)

        assert nl.stories[0].title == "A"


class TestExcludeStory:
    def test_exclude_sets_flag(self):
        story = _story()
        exclude_story(story)
        assert story.excluded is True


class TestDeleteStory:
    def test_delete_removes_candidate(self):
        nl = _newsletter("tD")
        seed_stories(nl, [("A", "a"), ("B", "b")])

        delete_story(nl, 0)

        assert [s.title for s in nl.stories] == ["B"]


class TestEditStory:
    def test_edit_updates_title_and_text(self):
        nl = _newsletter("tE")
        seed_stories(nl, [("A", "a")])

        edit_story(nl, 0, title="A2", text="a2")

        assert nl.stories[0].title == "A2"
        assert nl.stories[0].text == "a2"

    def test_edit_leaves_unspecified_field_untouched(self):
        nl = _newsletter("tE")
        seed_stories(nl, [("A", "a")])

        edit_story(nl, 0, title="A2")

        assert nl.stories[0].title == "A2"
        assert nl.stories[0].text == "a"  # unchanged


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

        story = create_story_from_body(nl, 1, 2, "Title")

        assert story.text == "l1\nl2"
        assert nl.stories[-1] is story

    def test_single_line_span(self):
        nl = _newsletter("tB", body="l0\nl1\nl2")

        story = create_story_from_body(nl, 1, 1, "T")

        assert story.text == "l1"

    def test_start_greater_than_end_normalized(self):
        nl = _newsletter("tB", body="l0\nl1\nl2\nl3")

        story = create_story_from_body(nl, 3, 1, "T")

        assert story.text == "l1\nl2\nl3"

    def test_out_of_range_clamped(self):
        nl = _newsletter("tB", body="l0\nl1\nl2")

        story = create_story_from_body(nl, -5, 99, "T")

        assert story.text == "l0\nl1\nl2"

    def test_small_negative_start_clamps_to_zero_not_from_end(self):
        # Guards the *lower* clamp specifically: without max(0, ...) a small
        # negative start would slice from the end (Python slice semantics),
        # silently dropping the leading body lines.
        nl = _newsletter("tB", body="l0\nl1\nl2")

        story = create_story_from_body(nl, -1, 1, "T")

        assert story.text == "l0\nl1"

    def test_stable_story_id_and_reviewed_untouched(self):
        nl = _newsletter("tB", body="l0\nl1")
        nl.stories = [_story("tB:0")]

        story = create_story_from_body(nl, 0, 0, "T")

        assert story.story_id == "tB:1"
        assert nl.reviewed is False

    def test_empty_title_auto_derived_from_first_words(self):
        body = " ".join(f"w{i}" for i in range(20))
        nl = _newsletter("tB", body=body)

        story = create_story_from_body(nl, 0, 0, "")

        # First ~8 words, single line, trimmed, non-empty.
        assert story.title
        assert "\n" not in story.title
        assert story.title.startswith("w0 w1 w2")
        assert len(story.title.split()) <= 8

    def test_empty_title_on_whitespace_only_segment_is_non_empty(self):
        # Selecting blank/whitespace-only body lines (common: blank separator
        # rows) with a blank title must still yield a usable, non-empty title.
        nl = _newsletter("tB", body="   \n\t\n  ")

        story = create_story_from_body(nl, 0, 2, "")

        assert story.title.strip()  # non-empty after stripping


class TestBuildDetailRows:
    def test_header_rows_have_none_body_idx(self):
        nl = _newsletter("tR", body="body line one\nbody line two")
        rows = build_detail_rows(nl, 0, 1, 80)

        # The first rows (subject/sender/etc.) are header rows.
        assert rows[0][1] is None
        assert any(idx is None for _, idx in rows)

    def test_long_body_line_wraps_sharing_body_idx(self):
        nl = _newsletter("tR", body="aaaa bbbb cccc dddd")
        rows = build_detail_rows(nl, 0, 1, 6)

        body_rows = [r for r in rows if r[1] == 0]
        assert len(body_rows) >= 2
        assert all(len(text) <= 6 for text, _ in body_rows)

    def test_distinct_body_idx_count_matches_body_lines(self):
        body = "line0\nline1\nline2"
        nl = _newsletter("tR", body=body)
        rows = build_detail_rows(nl, 0, 1, 80)

        distinct = {idx for _, idx in rows if idx is not None}
        assert len(distinct) == len(body.splitlines())

    def test_story_text_shown_inline_wrapped(self):
        nl = _newsletter("tT", body="b0")
        nl.stories = [
            _story(
                story_id="tT:0",
                title="Alpha",
                text="the quick brown fox jumped over the lazy dog",
            )
        ]
        rows = build_detail_rows(nl, 0, 1, 20)
        header_rows = [text for text, idx in rows if idx is None]
        joined = " ".join(header_rows)

        # Title still shown, and the FULL parsed text is shown inline (not truncated).
        assert "Alpha" in joined
        for word in ("quick", "brown", "fox", "lazy", "dog"):
            assert word in joined
        # Wrapped within the width and spanning multiple rows.
        assert all(len(text) <= 20 for text in header_rows)
        text_rows = [t for t in header_rows if any(w in t for w in ("quick", "fox", "lazy"))]
        assert len(text_rows) >= 2


class TestConfirmStoryList:
    def test_confirm_marks_newsletter_reviewed(self):
        nl = _newsletter("tC")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        assert nl.reviewed is False

        confirm_story_list(nl)

        assert nl.reviewed is True

    def test_confirm_preserves_existing_ids_after_delete(self):
        # Confirm must NOT positionally renumber: deleting a story and
        # re-confirming would rename all later stories, breaking the stable
        # story_id contract that cross-run comparisons key on.
        nl = _newsletter("tC")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
        delete_story(nl, 1)  # remove B -> gap in ids is fine

        confirm_story_list(nl)

        assert [s.story_id for s in nl.stories] == ["tC:0", "tC:2"]

    def test_confirm_repairs_duplicate_ids_with_fresh_unique_ones(self):
        # Only duplicates/collisions get a fresh id; the first holder keeps its.
        nl = _newsletter("tC")
        nl.stories = [
            _story("tC:0", title="A"),
            _story("tC:0", title="A-dup"),
            _story("tC:1", title="B"),
        ]

        confirm_story_list(nl)

        ids = [s.story_id for s in nl.stories]
        assert len(set(ids)) == len(ids)  # unique
        assert ids[0] == "tC:0"  # first holder kept
        assert ids[2] == "tC:1"  # untouched non-duplicate kept
        assert ids[1] == "tC:2"  # dup repaired via the next-id helper


class TestAddStory:
    def test_add_after_delete_never_duplicates_an_existing_id(self):
        # Deleting story 0 of [t1:0, t1:1, t1:2] then adding must NOT mint a
        # second "t1:2" — downstream dicts keyed by story_id would silently
        # collapse two stories. The next id comes from max(suffix)+1.
        nl = _newsletter("t1")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
        delete_story(nl, 0)

        added = add_story(nl, "D", "d")

        ids = [s.story_id for s in nl.stories]
        assert len(set(ids)) == len(ids)  # all unique
        assert added.story_id == "t1:3"

    def test_create_from_body_after_delete_never_duplicates_an_existing_id(self):
        # Same defect via the body-span path: it must share add_story's id
        # derivation instead of re-deriving from len(stories).
        nl = _newsletter("t1", body="l0\nl1")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
        delete_story(nl, 0)

        story = create_story_from_body(nl, 0, 0, "T")

        ids = [s.story_id for s in nl.stories]
        assert len(set(ids)) == len(ids)
        assert story.story_id == "t1:3"

    def test_appends_story_with_stable_id_and_leaves_unreviewed(self):
        nl = _newsletter("tXY")
        nl.stories = [_story("tXY:0")]

        add_story(nl, title="New", text="New text")

        assert len(nl.stories) == 2
        added = nl.stories[1]
        assert added.story_id == "tXY:1"  # stable f"{thread_id}:{index}"
        assert added.title == "New"
        assert added.text == "New text"
        assert added.reviewed is False
        assert nl.reviewed is False  # adding does not confirm the list



class TestLoadSaveRoundTrip:
    def test_save_then_load_preserves_nested_stories(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        nl = _newsletter("t1")
        seed_stories(nl, [("A", "a"), ("B", "b")])
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
        assert [s.title for s in loaded[0].stories] == ["A", "B"]
        assert loaded[0].stories[0].expected_tier == "excellent"
        assert loaded[0].reviewed is True

    def test_load_tolerates_blank_lines(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        nl = _newsletter("t1")
        save_golden_set([nl], path)
        with open(path, "a") as f:
            f.write("\n")  # trailing blank line

        loaded = load_golden_set(path)
        assert len(loaded) == 1


class TestSeedFromExtractor:
    def test_runs_extractor_output_through_parse_stories(self):
        # The injected extractor returns the *raw* LLM extraction string; the
        # tool must parse it with the production parse_stories, so tests never
        # hit a network.
        nl = _newsletter("tX")
        raw = "TITLE: One\nTEXT: first story\n\nTITLE: Two\nTEXT: second story"

        captured = {}

        def fake_extract(body):
            captured["body"] = body
            return raw

        seed_from_extractor(nl, fake_extract)

        assert captured["body"] == "raw body"  # body fed verbatim
        assert [(s.title, s.text) for s in nl.stories] == [
            ("One", "first story"),
            ("Two", "second story"),
        ]
        assert nl.seeded_from == "parse_stories"
        assert nl.reviewed is False

    def test_no_stories_seeds_empty_list(self):
        nl = _newsletter("tX")
        seed_from_extractor(nl, lambda body: "NO_STORIES")
        assert nl.stories == []
        assert nl.seeded_from == "parse_stories"


class TestFormatHelpLines:
    def test_documents_space_seed_and_paging_hotkeys(self):
        # The Space (LLM-seed) key is Phase A's entry point and PgUp/PgDn exist;
        # both must be discoverable from the on-screen help.
        joined = " ".join(format_help_lines(200))
        assert "Space" in joined
        assert "seed" in joined.lower()
        assert "PgUp" in joined

    def test_wide_terminal_fits_on_one_line(self):
        assert len(format_help_lines(500)) == 1

    def test_narrow_terminal_wraps_to_multiple_full_lines(self):
        lines = format_help_lines(60)
        assert len(lines) >= 2
        assert all(len(line) <= 60 for line in lines)
        # Nothing is lost by wrapping: every key listed wide is listed narrow.
        assert set(" ".join(lines).split()) == set(" ".join(format_help_lines(500)).split())


class TestSeedConfirmationMessage:
    def test_no_confirmation_needed_for_empty_story_list(self):
        assert seed_confirmation_message(_newsletter("t")) is None

    def test_warns_with_story_count_when_stories_exist(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        msg = seed_confirmation_message(nl)
        assert msg is not None
        assert "2" in msg
        assert "y/N" in msg

    def test_mentions_labeled_stories_at_risk(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}, []
        )
        msg = seed_confirmation_message(nl)
        assert "1 labeled" in msg

    def test_singular_story_in_warning(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a")])
        msg = seed_confirmation_message(nl)
        assert "Replace 1 story " in msg
        assert "stories" not in msg


class TestConfirmStatusMessage:
    def test_all_labeled_confirms_plainly(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a")])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4}, []
        )
        assert confirm_status_message(nl) == "Story list confirmed."

    def test_singular_unlabeled(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a")])
        msg = confirm_status_message(nl)
        assert "1 story still unlabeled" in msg
        assert "stories" not in msg

    def test_plural_unlabeled(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        assert "2 stories still unlabeled" in confirm_status_message(nl)


class TestSeedOutcomeMessage:
    def test_reports_story_count_on_success(self):
        assert "2" in seed_outcome_message("TITLE: A\nTEXT: a\n\nTITLE: B\nTEXT: b", 2)

    def test_singular_story_count(self):
        assert seed_outcome_message("TITLE: A\nTEXT: a", 1) == "Seeded 1 story."

    def test_distinguishes_no_stories_verdict(self):
        msg = seed_outcome_message("NO_STORIES", 0)
        assert "NO_STORIES" in msg

    def test_distinguishes_unparseable_output(self):
        msg = seed_outcome_message("TITLE: dangling title with no text", 0)
        assert "NO_STORIES" not in msg
        assert "no parseable" in msg.lower()


class TestRowForBodyLine:
    def test_finds_first_row_carrying_body_idx(self):
        rows = [("hdr", None), ("a", 0), ("b wrapped", 1), ("b wrapped2", 1), ("c", 2)]
        assert row_for_body_line(rows, 1) == 2

    def test_missing_body_idx_falls_back_to_last_row(self):
        rows = [("hdr", None), ("a", 0)]
        assert row_for_body_line(rows, 99) == 1

    def test_empty_rows_fall_back_to_zero(self):
        assert row_for_body_line([], 3) == 0


class TestCoveredBodyLines:
    def test_lines_verbatim_in_a_story_are_covered(self):
        nl = _newsletter("t", body="greeting\nstory line one\nstory line two\nsign-off")
        nl.stories = [_story("t:0", text="story line one\nstory line two")]
        assert covered_body_lines(nl) == {1, 2}

    def test_blank_lines_are_never_marked(self):
        nl = _newsletter("t", body="story line\n\nother")
        nl.stories = [_story("t:0", text="story line\n\nother")]
        assert 1 not in covered_body_lines(nl)

    def test_no_stories_means_nothing_covered(self):
        nl = _newsletter("t", body="a\nb")
        assert covered_body_lines(nl) == set()


class TestLabelProgress:
    def test_counts_labeled_over_total_excluding_excluded(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
        assign_scores_and_themes(
            nl.stories[0], {"simple": 3, "concrete": 3, "personal": 3, "dynamic": 3}, []
        )
        nl.stories[2].excluded = True
        assert label_progress(nl) == (1, 2)

    def test_unlabeled_story_count_ignores_excluded(self):
        nl = _newsletter("t")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        nl.stories[1].excluded = True
        assert unlabeled_story_count(nl) == 1


class TestFormatListRow:
    def test_shows_labeled_over_total_and_sender(self):
        nl = _newsletter("t", sender="paul@dm.org", subject="Fall update")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
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
        rows = build_detail_rows(nl, 0, 1, 80)
        joined = " ".join(text for text, _ in rows)
        assert "Excluded:" in joined

    def test_detail_rows_omit_excluded_line_when_not_excluded(self):
        nl = _newsletter("t", body="b0")
        rows = build_detail_rows(nl, 0, 1, 80)
        joined = " ".join(text for text, _ in rows)
        assert "Excluded:" not in joined

    def test_help_mentions_newsletter_exclude_hotkey(self):
        joined = "  ".join(format_help_lines(500))
        assert "[X]" in joined


class TestFormatThemeLegend:
    def test_full_legend_when_it_fits(self):
        legend = format_theme_legend([], 200)
        assert "[s]scripture" in legend
        assert "[d]disciple_making" in legend
        assert "Enter=done" in legend

    def test_narrow_terminal_gets_compact_legend_with_all_keys(self):
        legend = format_theme_legend(["scripture"], 60)
        assert len(legend) <= 60
        for key in "schvd":
            assert f"[{key}]" in legend
        assert "Enter" in legend


class TestWrapTextDisplayWidth:
    def test_emoji_wrap_respects_display_width(self):
        # Each emoji occupies 2 terminal cells; 4 cells fit only 2 of them.
        wrapped = wrap_text("🎉🎉🎉🎉", 4)
        assert wrapped == ["🎉🎉", "🎉🎉"]

    def test_no_characters_are_lost_when_wrapping_wide_text(self):
        text = "José y Anaïs 🎉🎊✨ celebraron juntos"
        wrapped = wrap_text(text, 10)
        assert "".join("".join(line.split()) for line in wrapped) == "".join(text.split())


class TestBuildDetailRowsLabels:
    def test_assigned_scores_are_shown_on_the_story(self):
        nl = _newsletter("tS", body="b0")
        nl.stories = [_story("tS:0")]
        assign_scores_and_themes(
            nl.stories[0], {"simple": 4, "concrete": 4, "personal": 5, "dynamic": 3}, []
        )
        joined = " ".join(text for text, _ in build_detail_rows(nl, 0, 1, 80))
        assert "4/4/5/3" in joined

    def test_divider_fits_narrow_terminals(self):
        nl = _newsletter("tS", body="b0")
        rows = build_detail_rows(nl, 0, 1, 30)
        dividers = [text for text, _ in rows if set(text) == {"="}]
        # A single divider row sized to the terminal — not a 60-char rule
        # wrapped into multiple ragged rows.
        assert len(dividers) == 1
        assert len(dividers[0]) <= 30


class TestSeedFromExtractorRaw:
    def test_returns_stories_and_raw_output_for_status_reporting(self):
        nl = _newsletter("tX")
        raw = "TITLE: One\nTEXT: first story"
        stories, returned_raw = seed_from_extractor(nl, lambda body: raw)
        assert returned_raw == raw
        assert [s.title for s in stories] == ["One"]


class TestBuildExtractorClientConfig:
    def test_passes_config_timeout_and_1024_max_tokens_default(self, monkeypatch):
        # Must match daemon.run_daemon / newsletter_run.build_classifier:
        # timeout=nl_llm.get("timeout", 60) and max_tokens default 1024.
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
        quality = {"system": "s", "user_template": "{title}{text}"}
        config = {
            "newsletter": {
                "llm": {"model": "m", "timeout": 123},  # no max_tokens
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


def _scores(n=4):
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


# Body rows start after: title, divider, subject, sender, reviewed, seeded,
# blank, "--- Stories (0) ---", blank, "--- Body ---" = 10 header rows.
BODY_START = 10


class TestDetailSeedGuard:
    async def test_reseed_over_existing_stories_requires_confirmation(self, tmp_path):
        nl = _newsletter("tg", body="b0\nb1")
        seed_stories(nl, [("Keep", "b0")])
        assign_scores_and_themes(nl.stories[0], _scores(), ["scripture"])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "TITLE: New\nTEXT: b1")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await pilot.press("n")  # y/N prompt: keep the curated story
            await _drain(app, pilot)
            assert [s.title for s in nl.stories] == ["Keep"]
            assert nl.stories[0].expected_scores == _scores()
            assert "Seed cancelled" in _status(app)

    async def test_confirmed_reseed_replaces_and_reports_count(self, tmp_path):
        nl = _newsletter("tg", body="b0\nb1")
        seed_stories(nl, [("Old", "b0")])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "TITLE: New\nTEXT: b1")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await pilot.press("y")
            await _drain(app, pilot)
            assert [s.title for s in nl.stories] == ["New"]
            assert "Seeded 1" in _status(app)

    async def test_empty_story_list_seeds_without_confirmation(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "TITLE: A\nTEXT: b0")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await _drain(app, pilot)
            assert [s.title for s in nl.stories] == ["A"]

    async def test_no_stories_verdict_is_reported_distinctly(self, tmp_path):
        nl = _newsletter("tg", body="admin only")
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "NO_STORIES")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await _drain(app, pilot)
            assert "NO_STORIES" in _status(app)

    async def test_edit_mode_space_reports_seeding_disabled(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        seed_stories(nl, [("Keep", "b0")])
        app = _label_app([nl], tmp_path, extract_fn=None)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await _drain(app, pilot)
            assert [s.title for s in nl.stories] == ["Keep"]
            assert "edit mode" in _status(app).lower()

    async def test_seed_failure_keeps_stories_and_undo_clean(self, tmp_path):
        def boom(body):
            raise RuntimeError("connection dropped")

        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=boom)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await _drain(app, pilot)
            assert "Seed failed" in _screen_text(app)  # blocking hint
            await pilot.press("x")  # any key dismisses the hint
            await _drain(app, pilot)
            assert nl.stories == []
            await pilot.press("z")
            assert "Nothing to undo" in _status(app)

    async def test_second_space_while_seeding_is_ignored(self, tmp_path):
        import threading

        release = threading.Event()
        calls = []

        def slow_extract(body):
            calls.append(1)
            release.wait(timeout=5)
            return "TITLE: A\nTEXT: b0"

        nl = _newsletter("tg", body="b0")
        app = _label_app([nl], tmp_path, extract_fn=slow_extract)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            await pilot.pause()
            await pilot.press("space")  # while the first seed is in flight
            release.set()
            await _drain(app, pilot)
            assert len(calls) == 1
            assert [s.title for s in nl.stories] == ["A"]


class TestDetailUndo:
    async def test_undo_stack_restores_multiple_edits(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "0", "enter")
            await pilot.press("d", "0", "enter")
            assert [s.title for s in nl.stories] == ["C"]
            await pilot.press("z", "z")
            assert [s.title for s in nl.stories] == ["A", "B", "C"]

    async def test_cancelled_prompt_does_not_clobber_undo(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "1", "enter")  # delete B
            await pilot.press("E", "escape")  # open edit, cancel at Story #
            await pilot.press("z")  # undo must restore B
            assert [s.title for s in nl.stories] == ["A", "B"]

    async def test_undo_with_empty_stack_reports_nothing_to_undo(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "z")
            assert "Nothing to undo" in _status(app)


class TestDetailDelete:
    async def test_deleting_labeled_story_requires_confirmation(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("Labeled", "a")])
        assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "0", "enter", "n")
            assert [s.title for s in nl.stories] == ["Labeled"]

    async def test_confirmed_delete_of_labeled_story_reports_what_was_deleted(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("Labeled", "a")])
        assign_scores_and_themes(nl.stories[0], _scores(), [])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "0", "enter", "y")
            assert nl.stories == []
            assert "Deleted" in _status(app)
            assert "Labeled" in _status(app)

    async def test_invalid_index_reports_instead_of_silent_noop(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "9", "enter")
            assert [s.title for s in nl.stories] == ["A"]
            assert "story #" in _status(app).lower()

    async def test_autosave_persists_mutation_to_disk_before_quit(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        path = tmp_path / "golden.jsonl"
        save_golden_set([nl], path)
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "0", "enter")
            # Autosave on the mutation itself — the app is still running.
            on_disk = load_golden_set(path)
            assert [s.title for s in on_disk[0].stories] == ["B"]


class TestDetailCursorAnchor:
    async def test_cursor_stays_on_its_body_line_after_making_a_story(self, tmp_path):
        nl = _newsletter("tA", body="para one\npara two\npara three")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press(*["down"] * (BODY_START + 1))  # onto "para two"
            await pilot.press("s", "enter")  # start selection, make story
            await pilot.press("enter")  # Title (blank=auto)
            assert len(nl.stories) == 1
            assert nl.stories[0].text == "para two"
            # The story list above the body grew, shifting every row down;
            # the cursor must still sit on the "para two" BODY line.
            assert "para two" in _cursor_row_text(app)


class TestDetailRowCache:
    def _counting(self, monkeypatch):
        import evals.newsletter_label as label

        calls = {"rows": 0, "covered": 0}
        real_rows = label.build_detail_rows
        real_covered = label.covered_body_lines

        def counting_rows(*args, **kwargs):
            calls["rows"] += 1
            return real_rows(*args, **kwargs)

        def counting_covered(*args, **kwargs):
            calls["covered"] += 1
            return real_covered(*args, **kwargs)

        monkeypatch.setattr(label, "build_detail_rows", counting_rows)
        monkeypatch.setattr(label, "covered_body_lines", counting_covered)
        return calls

    async def test_pure_cursor_movement_reuses_cached_rows(self, tmp_path, monkeypatch):
        calls = self._counting(monkeypatch)
        nl = _newsletter("tP", body="b0\nb1\nb2\nb3")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("down", "down", "down", "down", "up", "up")
            assert calls["rows"] == 1
            assert calls["covered"] == 1

    async def test_mutation_rebuilds_rows(self, tmp_path, monkeypatch):
        calls = self._counting(monkeypatch)
        nl = _newsletter("tP", body="b0\nb1")
        seed_stories(nl, [("A", "b0")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("d", "0", "enter")
            assert calls["rows"] == 2  # initial build + rebuild after the delete

    async def test_resize_rewraps_body(self, tmp_path):
        # A terminal resize must rewrap the body to the new width so no
        # content is lost to clipping.
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
            # The rows were actually rebuilt at the new width (marker col +
            # width-2 content = 39 cols max) — not just soft-wrapped visually.
            assert all(len(t) <= 39 for t in row_texts), max(row_texts, key=len)


class TestDetailSelection:
    async def test_esc_clears_selection_before_leaving(self, tmp_path):
        from evals.newsletter_label import DetailScreen

        nl = _newsletter("tE", body="para one\npara two")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press(*["down"] * BODY_START)
            await pilot.press("s")
            await pilot.press("escape")  # first Esc only clears the selection
            assert isinstance(app.screen, DetailScreen)
            assert "Selection cleared" in _status(app)
            await pilot.press("escape")  # second Esc leaves
            assert not isinstance(app.screen, DetailScreen)

    async def test_esc_at_title_prompt_cancels_story_creation(self, tmp_path):
        nl = _newsletter("tE", body="para one")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press(*["down"] * BODY_START)
            await pilot.press("s", "enter")  # selection + make story
            await pilot.press("escape")  # cancel at the title prompt
            assert nl.stories == []  # no story, and certainly no "\x1b" title

    async def test_selection_keys_require_a_body_line(self, tmp_path):
        nl = _newsletter("tE", body="para one")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")  # cursor starts on the header row
            await pilot.press("s")
            assert "body line" in _status(app)

    async def test_make_story_without_selection_reports_hint(self, tmp_path):
        nl = _newsletter("tE", body="para one")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("enter")  # make-story with no selection start
            assert "selection start" in _status(app).lower()
            assert nl.stories == []

    async def test_span_selection_makes_multiline_story(self, tmp_path):
        nl = _newsletter("tE", body="para one\npara two\npara three")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press(*["down"] * BODY_START)  # "para one"
            await pilot.press("s")
            await pilot.press("down", "e")  # end on "para two"
            await pilot.press("enter")  # make story
            await pilot.press(*"My Story")
            await pilot.press("enter")
            assert len(nl.stories) == 1
            assert nl.stories[0].title == "My Story"
            assert nl.stories[0].text == "para one\npara two"


class TestDetailNotes:
    async def test_notes_prompt_prefills_existing_note(self, tmp_path):
        nl = _newsletter("tN", body="b0", notes="important note")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("n", "enter")  # Enter keeps, does not wipe
            assert nl.notes == "important note"
            assert "unchanged" in _status(app).lower()


class TestDetailLabeling:
    async def test_label_flow_assigns_scores_and_saves(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "0", "enter")  # story index
            await pilot.press("4", "4", "5", "3")  # four dimension scores
            await pilot.press("enter")  # finish themes
            assert nl.stories[0].expected_scores == {
                "simple": 4, "concrete": 4, "personal": 5, "dynamic": 3,
            }
            assert nl.stories[0].reviewed is True

    async def test_score_entry_echoes_accepted_digits(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "0", "enter")
            await pilot.press("4", "4", "5")
            # By the last prompt the three accepted digits are visible.
            assert "4/4/5" in _screen_text(app)
            await pilot.press("3", "enter")

    async def test_relabeling_shows_current_scores(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])
        assign_scores_and_themes(
            nl.stories[0],
            {"simple": 2, "concrete": 3, "personal": 4, "dynamic": 5},
            [],
        )
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "0", "enter")
            assert "now 2" in _screen_text(app)
            await pilot.press("x")  # cancel

    async def test_cancelled_score_entry_reports_it(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "0", "enter")
            await pilot.press("4", "x")  # any non-digit cancels everything
            assert nl.stories[0].expected_scores is None
            assert "cancelled" in _status(app).lower()

    async def test_theme_toggle_on_and_off(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "0", "enter")
            await pilot.press("4", "4", "4", "4")
            await pilot.press("s", "c", "s")  # scripture on, christlikeness on, scripture off
            await pilot.press("enter")
            assert nl.stories[0].expected_themes == ["christlikeness"]

    async def test_confirm_warns_about_unlabeled_stories(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("c")
            assert nl.reviewed is True
            assert "2" in _status(app)
            assert "unlabeled" in _status(app).lower()


class TestDetailStoryExclusion:
    async def test_u_toggles_story_exclusion(self, tmp_path):
        nl = _newsletter("tX", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("u", "0", "enter")
            assert nl.stories[0].excluded is True
            assert "excluded from" in _status(app)
            await pilot.press("u", "0", "enter")
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


class TestDetailHelp:
    async def test_space_seed_hotkey_is_on_screen(self, tmp_path):
        nl = _newsletter("tH", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            assert "Space:seed" in _screen_text(app)


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
            assert index and index > 1  # moved by a page, not a single row

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
            await pilot.press("k")  # linear skip-through: straight to the next
            assert isinstance(app.screen, DetailScreen)
            assert "Newsletter 2/2" in _screen_text(app)
            assert nls[0].reviewed is False  # skip never marks reviewed
            await pilot.press("k")  # skip on the last -> back to the list
            assert not isinstance(app.screen, DetailScreen)

    async def test_q_from_detail_quits_whole_app(self, tmp_path):
        nls = [_newsletter("t0")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "q")
        assert app.return_value == "quit"

    async def test_list_flags_refresh_after_confirm_in_detail(self, tmp_path):
        nls = [_newsletter("t0"), _newsletter("t1")]
        app = _label_app(nls, tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "c", "escape")  # confirm, back to list
            assert nls[0].reviewed is True
            assert "Y" in _screen_text(app)  # reviewed flag column updated


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

        assert seen["ids"] == ["normal"]  # excluded never queued
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
        # The harvest hint must say the harvest --output has to match this
        # tool's --golden-set path, or the user harvests into the wrong file.
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
        # Auto-repeat can queue a 5th digit behind the dismissal of the 4th;
        # it must be ignored, not IndexError-crash the app.
        from textual import events

        nl = _newsletter("tR", body="b0")
        seed_stories(nl, [("A", "a")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            await pilot.press("l", "0", "enter")
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
            await pilot.press("enter", "n")
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
        seed_stories(nl, [("Keep", "b0")])
        app = _label_app([nl], tmp_path, extract_fn=lambda body: "TITLE: New\nTEXT: x")
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space")
            app.post_message(events.Key("n", "n"))
            app.post_message(events.Key("n", "n"))
            await _drain(app, pilot)
            assert app.is_running
            assert [s.title for s in nl.stories] == ["Keep"]
            from evals.newsletter_label import DetailScreen

            # the queued second key must not pop the detail screen underneath
            assert isinstance(app.screen, DetailScreen)

    async def test_mashed_action_key_opens_a_single_prompt(self, tmp_path):
        # Two queued 'd' presses must not spawn two concurrent delete flows.
        from textual import events

        nl = _newsletter("tR", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b")])
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter")
            app.post_message(events.Key("d", "d"))
            app.post_message(events.Key("d", "d"))
            await pilot.pause()
            await pilot.press("0", "enter")
            await pilot.pause()
            assert app.is_running
            assert [s.title for s in nl.stories] == ["B"]
            from evals.newsletter_label import DetailScreen

            assert isinstance(app.screen, DetailScreen)  # no second prompt stacked

    async def test_notes_control_characters_never_persist(self, tmp_path):
        # A paste can carry control chars into the Input; they must be
        # stripped before landing in the golden set.
        from textual.widgets import Input

        nl = _newsletter("tR", body="b0")
        app = _label_app([nl], tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "n")
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
            return "TITLE: Machine seed\nTEXT: b0"

        nl = _newsletter("tA", body="b0")
        seed_stories(nl, [("Hand curated", "b0")])
        nl.reviewed = True
        app = _label_app([nl], tmp_path, extract_fn=slow_extract)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("enter", "space", "y")  # confirm reseed, in flight
            await pilot.pause()
            await pilot.press("escape")  # abandon the newsletter mid-seed
            release.set()
            await _drain(app, pilot)
            # The late extractor result must NOT clobber the curated stories.
            assert [s.title for s in nl.stories] == ["Hand curated"]
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
            # title + header + help = 3 rows -> list height 5
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
