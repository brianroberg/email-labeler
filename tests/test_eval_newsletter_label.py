"""Tests for evals.newsletter_label — pure state-transition functions.

The curses UI is untested here by design; every behavior lives in a pure
function that mutates a GoldenNewsletter / GoldenStory in place, so these
tests exercise the labeling logic without a terminal.
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
    restore_snapshot,
    row_for_body_line,
    save_golden_set,
    seed_confirmation_message,
    seed_from_extractor,
    seed_outcome_message,
    seed_stories,
    select_label_newsletters,
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

    def test_confirm_reindexes_story_ids_after_edits(self):
        # Delete leaves a gap in ids; confirming makes them contiguous.
        nl = _newsletter("tC")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])
        delete_story(nl, 1)  # remove B -> ids may be stale

        confirm_story_list(nl)

        assert [s.story_id for s in nl.stories] == ["tC:0", "tC:1"]


class TestAddStory:
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


class TestLineBuffer:
    def _buf(self, initial=""):
        from evals.newsletter_label import LineBuffer
        return LineBuffer(initial)

    def test_prefill_places_cursor_at_end_and_keeps_text(self):
        buf = self._buf("existing note")
        assert buf.text() == "existing note"
        buf.insert("!")
        assert buf.text() == "existing note!"

    def test_insert_backspace_and_cursor_movement(self):
        buf = self._buf("ab")
        buf.left()
        buf.insert("X")
        assert buf.text() == "aXb"
        buf.backspace()
        assert buf.text() == "ab"
        buf.right()
        buf.insert("Y")
        assert buf.text() == "abY"

    def test_control_characters_are_rejected(self):
        buf = self._buf()
        buf.insert("\x1b")  # Esc must never end up in the golden set
        buf.insert("\n")
        buf.insert("a")
        assert buf.text() == "a"

    def test_visible_scrolls_horizontally_so_input_is_never_truncated(self):
        buf = self._buf("abcdefghij")  # 10 chars, window of 4
        text, cur = buf.visible(4)
        assert "j" in text  # window follows the cursor at the end
        assert len(text) <= 4
        assert 0 <= cur <= 4
        for _ in range(10):
            buf.left()
        text, cur = buf.visible(4)
        assert text.startswith("a")  # window follows the cursor back to the start
        assert cur == 0


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


class FakeScreen:
    """Minimal curses-window stand-in: scripted keys, recorded frames.

    ``clear()`` starts a new frame; every ``addnstr`` is recorded as
    ``(y, x, text, attr)`` in the current frame, so tests can assert on both
    transient messages (any frame) and the final rendered state (last frame).
    """

    def __init__(self, keys, height=30, width=100):
        self._keys = list(keys)
        self.height = height
        self.width = width
        self.frames = [[]]

    def getmaxyx(self):
        return (self.height, self.width)

    def clear(self):
        self.frames.append([])

    def refresh(self):
        pass

    def keypad(self, flag):
        pass

    def move(self, y, x):
        pass

    def addnstr(self, y, x, text, n, attr=0):
        self.frames[-1].append((y, x, text[:n], attr))

    def _pop(self):
        if not self._keys:
            raise AssertionError("key script exhausted — the TUI asked for more input")
        return self._keys.pop(0)

    def getch(self):
        key = self._pop()
        return ord(key) if isinstance(key, str) else key

    def get_wch(self):
        return self._pop()

    def all_writes(self):
        return [w for frame in self.frames for w in frame]

    def texts(self):
        return " | ".join(w[2] for w in self.all_writes())


def _run_detail(keys, newsletters, tmp_path, extract_fn=None, index=0):
    from evals.newsletter_label import _newsletter_detail

    screen = FakeScreen(keys)
    path = tmp_path / "golden.jsonl"
    result = _newsletter_detail(
        screen, newsletters, index, newsletters, path, extract_fn=extract_fn
    )
    return screen, result


def _scores(n=4):
    return {"simple": n, "concrete": n, "personal": n, "dynamic": n}


class TestDetailSeedGuard:
    def test_reseed_over_existing_stories_requires_confirmation(self, tmp_path):
        nl = _newsletter("tg", body="b0\nb1")
        seed_stories(nl, [("Keep", "b0")])
        assign_scores_and_themes(nl.stories[0], _scores(), ["scripture"])

        screen, _ = _run_detail(
            [" ", "n", "q"], [nl], tmp_path, extract_fn=lambda body: "TITLE: New\nTEXT: b1"
        )

        # 'n' at the y/N prompt keeps the curated story AND its labels.
        assert [s.title for s in nl.stories] == ["Keep"]
        assert nl.stories[0].expected_scores == _scores()

    def test_confirmed_reseed_replaces_and_reports_count(self, tmp_path):
        nl = _newsletter("tg", body="b0\nb1")
        seed_stories(nl, [("Old", "b0")])

        screen, _ = _run_detail(
            [" ", "y", "q"], [nl], tmp_path, extract_fn=lambda body: "TITLE: New\nTEXT: b1"
        )

        assert [s.title for s in nl.stories] == ["New"]
        assert "Seeded 1" in screen.texts()

    def test_empty_story_list_seeds_without_confirmation(self, tmp_path):
        nl = _newsletter("tg", body="b0")

        _run_detail([" ", "q"], [nl], tmp_path, extract_fn=lambda body: "TITLE: A\nTEXT: b0")

        assert [s.title for s in nl.stories] == ["A"]

    def test_no_stories_verdict_is_reported_distinctly(self, tmp_path):
        nl = _newsletter("tg", body="admin only")

        screen, _ = _run_detail([" ", "q"], [nl], tmp_path, extract_fn=lambda body: "NO_STORIES")

        assert "NO_STORIES" in screen.texts()

    def test_edit_mode_space_reports_seeding_disabled(self, tmp_path):
        nl = _newsletter("tg", body="b0")
        seed_stories(nl, [("Keep", "b0")])

        screen, _ = _run_detail([" ", "q"], [nl], tmp_path, extract_fn=None)

        assert [s.title for s in nl.stories] == ["Keep"]
        assert "edit mode" in screen.texts().lower()

    def test_seed_failure_keeps_stories_and_undo_clean(self, tmp_path):
        def boom(body):
            raise RuntimeError("connection dropped")

        nl = _newsletter("tg", body="b0")
        # Space (empty list, no confirm) -> failure hint (any key) -> z -> q
        screen, _ = _run_detail([" ", "x", "z", "q"], [nl], tmp_path, extract_fn=boom)

        assert nl.stories == []
        assert "Seed failed" in screen.texts()
        assert "Nothing to undo" in screen.texts()


class TestDetailUndo:
    def test_undo_stack_restores_multiple_edits(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])

        keys = ["d", "0", "\n", "d", "0", "\n", "z", "z", "q"]
        _run_detail(keys, [nl], tmp_path)

        assert [s.title for s in nl.stories] == ["A", "B", "C"]

    def test_cancelled_prompt_does_not_clobber_undo(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b")])

        # Delete B, then open E and cancel it with Esc, then undo.
        keys = ["d", "1", "\n", "E", "\x1b", "z", "q"]
        _run_detail(keys, [nl], tmp_path)

        assert [s.title for s in nl.stories] == ["A", "B"]

    def test_undo_with_empty_stack_reports_nothing_to_undo(self, tmp_path):
        nl = _newsletter("tU", body="b0")
        screen, _ = _run_detail(["z", "q"], [nl], tmp_path)
        assert "Nothing to undo" in screen.texts()


class TestDetailDelete:
    def test_deleting_labeled_story_requires_confirmation(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("Labeled", "a")])
        assign_scores_and_themes(nl.stories[0], _scores(), [])

        _run_detail(["d", "0", "\n", "n", "q"], [nl], tmp_path)

        assert [s.title for s in nl.stories] == ["Labeled"]

    def test_confirmed_delete_of_labeled_story_reports_what_was_deleted(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("Labeled", "a")])
        assign_scores_and_themes(nl.stories[0], _scores(), [])

        screen, _ = _run_detail(["d", "0", "\n", "y", "q"], [nl], tmp_path)

        assert nl.stories == []
        assert "Labeled" in screen.texts()
        assert "Deleted" in screen.texts()

    def test_invalid_index_reports_instead_of_silent_noop(self, tmp_path):
        nl = _newsletter("tD", body="b0")
        seed_stories(nl, [("A", "a")])

        screen, _ = _run_detail(["d", "9", "\n", "q"], [nl], tmp_path)

        assert [s.title for s in nl.stories] == ["A"]
        assert "story #" in screen.texts().lower()


class TestDetailCursorAnchor:
    # Header rows before the body: title, divider, subject, sender, reviewed,
    # seeded, blank, "--- Stories (0) ---", blank, "--- Body ---" = 10 rows.
    BODY_START = 10

    def test_cursor_stays_on_its_body_line_after_making_a_story(self, tmp_path):
        import curses

        nl = _newsletter("tA", body="para one\npara two\npara three")
        keys = [curses.KEY_DOWN] * (self.BODY_START + 1) + ["s", 10, "\n", "q"]
        screen, _ = _run_detail(keys, [nl], tmp_path)

        assert len(nl.stories) == 1
        assert nl.stories[0].text == "para two"
        # Final frame: the cursor (reverse video) is still on "para two", even
        # though the story list above grew and shifted every row down.
        reversed_rows = [
            w[2] for w in screen.frames[-1] if w[3] == curses.A_REVERSE and w[1] <= 1
        ]
        assert any("para two" in text for text in reversed_rows)


class TestDetailSelection:
    def test_esc_clears_selection_before_leaving(self, tmp_path):
        import curses

        nl = _newsletter("tE", body="para one\npara two")
        keys = [curses.KEY_DOWN] * TestDetailCursorAnchor.BODY_START + ["s", 27, 27]
        screen, result = _run_detail(keys, [nl], tmp_path)

        assert result == "back"  # second Esc leaves; first only cleared
        assert "election cleared" in screen.texts()

    def test_esc_at_title_prompt_cancels_story_creation(self, tmp_path):
        import curses

        nl = _newsletter("tE", body="para one")
        keys = [curses.KEY_DOWN] * TestDetailCursorAnchor.BODY_START + ["s", 10, "\x1b", "q"]
        _run_detail(keys, [nl], tmp_path)

        assert nl.stories == []  # no story, and certainly no "\x1b" title


class TestDetailNotes:
    def test_notes_prompt_prefills_existing_note(self, tmp_path):
        nl = _newsletter("tN", body="b0", notes="important note")

        _run_detail(["n", "\n", "q"], [nl], tmp_path)

        assert nl.notes == "important note"  # Enter keeps, does not wipe


class TestDetailLabeling:
    def test_label_flow_assigns_scores_and_saves(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])

        keys = ["l", "0", "\n", "4", "4", "5", "3", "\n", "q"]
        _run_detail(keys, [nl], tmp_path)

        assert nl.stories[0].expected_scores == {
            "simple": 4, "concrete": 4, "personal": 5, "dynamic": 3,
        }
        assert nl.stories[0].reviewed is True

    def test_score_entry_echoes_accepted_digits(self):
        from evals.newsletter_label import _prompt_scores

        screen = FakeScreen(["4", "4", "5", "3"])
        scores = _prompt_scores(screen)

        assert scores == {"simple": 4, "concrete": 4, "personal": 5, "dynamic": 3}
        # By the last prompt the three accepted digits are visible.
        assert any("4/4/5" in w[2] for w in screen.all_writes())

    def test_relabeling_shows_current_scores(self):
        from evals.newsletter_label import _prompt_scores

        screen = FakeScreen(["x"])  # cancel immediately; only the prompt matters
        _prompt_scores(screen, current={"simple": 2, "concrete": 3, "personal": 4, "dynamic": 5})

        assert any("now 2" in w[2] for w in screen.all_writes())

    def test_cancelled_score_entry_reports_it(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a")])

        screen, _ = _run_detail(["l", "0", "\n", "4", "x", "q"], [nl], tmp_path)

        assert nl.stories[0].expected_scores is None
        assert "cancelled" in screen.texts().lower()

    def test_confirm_warns_about_unlabeled_stories(self, tmp_path):
        nl = _newsletter("tL", body="b0")
        seed_stories(nl, [("A", "a"), ("B", "b")])

        screen, _ = _run_detail(["c", "q"], [nl], tmp_path)

        assert nl.reviewed is True
        assert "2" in screen.texts() and "unlabeled" in screen.texts().lower()


class TestDetailHelp:
    def test_space_seed_hotkey_is_on_screen(self, tmp_path):
        nl = _newsletter("tH", body="b0")
        screen, _ = _run_detail(["q"], [nl], tmp_path)
        assert "Space:seed" in screen.texts()


class TestListView:
    def test_page_down_moves_by_a_page(self, tmp_path):
        import curses

        from evals.newsletter_label import _list_view

        nls = [_newsletter(f"t{i}", subject=f"subject-{i}") for i in range(10)]
        screen = FakeScreen([curses.KEY_NPAGE, "q"], height=8, width=100)
        _list_view(screen, nls, nls, tmp_path / "golden.jsonl")

        final_reversed = [w[2] for w in screen.frames[-1] if w[3] == curses.A_REVERSE]
        assert any("subject-5" in text for text in final_reversed)


class TestLabelLoopEnvironment:
    def test_escdelay_configured_for_snappy_esc(self, monkeypatch):
        import curses
        import os

        from evals.newsletter_label import label_loop

        monkeypatch.delenv("ESCDELAY", raising=False)
        monkeypatch.setattr(curses, "wrapper", lambda *a, **kw: None)
        label_loop([_newsletter("t")], [_newsletter("t")], "unused-path")

        assert os.environ.get("ESCDELAY")


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
