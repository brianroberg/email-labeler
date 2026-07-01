"""Tests for evals.newsletter_label — pure state-transition functions.

The curses UI is untested here by design; every behavior lives in a pure
function that mutates a GoldenNewsletter / GoldenStory in place, so these
tests exercise the labeling logic without a terminal.
"""

import sys

from evals.newsletter_label import (
    add_story,
    assign_scores_and_themes,
    capture_snapshot,
    confirm_story_list,
    delete_story,
    edit_story,
    exclude_story,
    load_golden_set,
    merge_stories,
    restore_snapshot,
    save_golden_set,
    seed_from_extractor,
    seed_stories,
    select_label_newsletters,
    split_story,
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


class TestSplitStory:
    def test_split_replaces_one_candidate_with_two(self):
        nl = _newsletter("tSp")
        seed_stories(nl, [("Merged", "part one\npart two")])

        split_story(nl, 0, "part two")

        assert len(nl.stories) == 2
        assert nl.stories[0].text == "part one"
        assert nl.stories[1].text == "part two"

    def test_split_second_half_starts_at_marker(self):
        nl = _newsletter("tSp")
        seed_stories(nl, [("M", "aaa BREAK bbb")])

        split_story(nl, 0, "BREAK bbb")

        assert nl.stories[0].text == "aaa"
        assert nl.stories[1].text == "BREAK bbb"


class TestMergeStories:
    def test_merge_combines_candidate_with_successor(self):
        nl = _newsletter("tM")
        seed_stories(nl, [("A", "a"), ("B", "b"), ("C", "c")])

        merge_stories(nl, 0)

        assert len(nl.stories) == 2
        assert nl.stories[0].title == "A"
        assert nl.stories[0].text == "a\nb"
        assert nl.stories[1].title == "C"


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
