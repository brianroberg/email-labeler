"""Interactive CLI to build newsletter golden-set labels (two phases).

Usage:
    python -m evals.newsletter_label                     # curate + label
    python -m evals.newsletter_label --edit              # curses edit TUI
    python -m evals.newsletter_label --unreviewed-only

Phase A (curate stories / extraction truth): seed candidate stories by running
the production ``newsletter.parse_stories`` over a one-time LLM extraction of
the body, then split/merge/edit/add/delete candidates. Confirming the list
sets ``newsletter.reviewed=True`` and assigns each story a stable
``story_id = f"{thread_id}:{index}"``.

Phase B (per-story labels): assign the 4 dimension scores
(simple/concrete/personal/dynamic, 1-5) and multi-select themes; on save the
``expected_tier`` is derived via ``newsletter.compute_tier`` and
``story.reviewed`` is set.

The state transitions are factored into PURE functions (below) so they can be
unit-tested without curses.
"""

import argparse
import asyncio
import copy
import curses
import json
import os
import sys
import tempfile
from pathlib import Path

from evals.newsletter_schemas import GoldenNewsletter, GoldenStory
from newsletter import compute_tier, parse_stories

_DIMENSIONS = ("simple", "concrete", "personal", "dynamic")

# Theme hotkeys mirror newsletter_review's convention (s/c/h/v/d).
_THEME_KEYS = {
    "s": "scripture",
    "c": "christlikeness",
    "h": "church",
    "v": "vocation_family",
    "d": "disciple_making",
}


# ---------------------------------------------------------------------------
# Load / save (atomic temp-file + rename, mirroring evals.review)
# ---------------------------------------------------------------------------

def load_golden_set(path: Path) -> list[GoldenNewsletter]:
    """Load the newsletter golden set from a JSONL file."""
    newsletters = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                newsletters.append(GoldenNewsletter.from_dict(json.loads(line)))
    return newsletters


def save_golden_set(newsletters: list[GoldenNewsletter], path: Path) -> None:
    """Save the golden set atomically (temp file + rename)."""
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for nl in newsletters:
                f.write(json.dumps(nl.to_dict()) + "\n")
        os.rename(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Pure state transitions (Phase A — curate stories)
# ---------------------------------------------------------------------------

def _reindex_story_ids(newsletter):
    """Reassign every story's id to f"{thread_id}:{index}" for its position."""
    for i, story in enumerate(newsletter.stories):
        story.story_id = f"{newsletter.thread_id}:{i}"


def seed_from_extractor(newsletter, extract_fn):
    """Seed candidates by running *extract_fn* over the body, then parse_stories.

    *extract_fn* takes the raw body and returns the raw LLM extraction string
    (kept injectable so tests never touch a network). The production
    ``parse_stories`` turns that into (title, text) pairs.
    """
    raw = extract_fn(newsletter.body)
    return seed_stories(newsletter, parse_stories(raw))


def seed_stories(newsletter, story_pairs):
    """Populate candidate stories from ``parse_stories`` (title, text) pairs.

    Replaces any existing story list, assigns stable ids, and records the seed
    provenance. Does not confirm the list.
    """
    newsletter.stories = [
        GoldenStory(story_id=f"{newsletter.thread_id}:{i}", title=title, text=text)
        for i, (title, text) in enumerate(story_pairs)
    ]
    newsletter.seeded_from = "parse_stories"
    return newsletter.stories


def add_story(newsletter, title, text):
    """Append a new candidate story with a stable story_id.

    Does not confirm the list — ``newsletter.reviewed`` is untouched and the
    new story starts unreviewed.
    """
    index = len(newsletter.stories)
    story = GoldenStory(
        story_id=f"{newsletter.thread_id}:{index}",
        title=title,
        text=text,
    )
    newsletter.stories.append(story)
    return story


def split_story(newsletter, index, split_at):
    """Split an over-merged candidate into two at the first occurrence of
    *split_at* within its text.

    The second story begins at *split_at* (marker kept in the tail). Both
    halves inherit the original title; ids are reindexed on confirm.
    """
    story = newsletter.stories[index]
    pos = story.text.find(split_at)
    if pos < 0:
        return  # marker not found — leave unchanged
    head = story.text[:pos].strip()
    tail = story.text[pos:].strip()
    story.text = head
    new_story = GoldenStory(
        story_id=f"{newsletter.thread_id}:{index + 1}",
        title=story.title,
        text=tail,
    )
    newsletter.stories.insert(index + 1, new_story)
    return new_story


def merge_stories(newsletter, index):
    """Merge the candidate at *index* with its immediate successor.

    Texts are joined with a newline; the merged story keeps the first story's
    title. Ids are reindexed on confirm.
    """
    if index + 1 >= len(newsletter.stories):
        return
    first = newsletter.stories[index]
    second = newsletter.stories.pop(index + 1)
    first.text = f"{first.text}\n{second.text}"


def edit_story(newsletter, index, *, title=None, text=None):
    """Edit a candidate's title and/or text span. Unspecified fields are kept."""
    story = newsletter.stories[index]
    if title is not None:
        story.title = title
    if text is not None:
        story.text = text


def delete_story(newsletter, index):
    """Remove a wrongly-extracted candidate entirely (not the same as exclude).

    Leaves ``newsletter.reviewed`` untouched; ids are reindexed on confirm.
    """
    newsletter.stories.pop(index)


def confirm_story_list(newsletter):
    """Confirm the curated story list as authoritative extraction truth.

    Reindexes story ids to be contiguous and marks the newsletter reviewed.
    """
    _reindex_story_ids(newsletter)
    newsletter.reviewed = True


# ---------------------------------------------------------------------------
# Pure state transitions (Phase B — per-story labels)
# ---------------------------------------------------------------------------

def assign_scores_and_themes(story, scores, themes):
    """Assign the 4 dimension scores + themes to a story and confirm it.

    ``expected_tier`` is auto-derived from *scores* via ``compute_tier``; the
    story is marked reviewed. *scores*/*themes* are copied so later mutation of
    the caller's objects does not alias into the golden set.
    """
    story.expected_scores = dict(scores)
    story.expected_themes = list(themes)
    story.expected_tier = compute_tier(scores).value
    story.reviewed = True


def exclude_story(story):
    """Keep a story as extraction truth but drop it from quality/theme scoring.

    A *wrongly-extracted* candidate should be deleted (``delete_story``); this
    is for real stories that are simply unsuitable for grading.
    """
    story.excluded = True


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

# Mutable newsletter-level fields captured for undo (stories handled separately)
_NL_SNAPSHOT_FIELDS = ("seeded_from", "reviewed", "notes", "excluded")


def capture_snapshot(newsletter):
    """Deep-copy a newsletter's mutable state so a later edit can be undone.

    One snapshot per newsletter; captures the full story list plus the
    newsletter-level mutable fields.
    """
    return {
        "stories": [copy.deepcopy(s) for s in newsletter.stories],
        **{f: getattr(newsletter, f) for f in _NL_SNAPSHOT_FIELDS},
    }


def restore_snapshot(newsletter, snapshot):
    """Restore a newsletter in place from *snapshot* (see ``capture_snapshot``)."""
    newsletter.stories = [copy.deepcopy(s) for s in snapshot["stories"]]
    for f in _NL_SNAPSHOT_FIELDS:
        setattr(newsletter, f, snapshot[f])


# ---------------------------------------------------------------------------
# Queue selection
# ---------------------------------------------------------------------------

def select_label_newsletters(newsletters, *, unreviewed_only=False):
    """Newsletters to queue for labeling.

    Excluded newsletters are permanently set aside and never queued, regardless
    of *unreviewed_only*.
    """
    result = [n for n in newsletters if not n.excluded]
    if unreviewed_only:
        result = [n for n in result if not n.reviewed]
    return result


# ---------------------------------------------------------------------------
# Phase-A seed extractor (real LLM; injected for tests)
# ---------------------------------------------------------------------------

def build_extractor(config: dict):
    """Build a synchronous ``extract_fn(body) -> raw_str`` for Phase-A seeding.

    Wraps the production newsletter story-extraction LLM call, returning the
    *raw* extraction string (not parsed) so ``seed_from_extractor`` can run it
    through ``parse_stories`` exactly like production. Kept separate from the
    pure functions so tests inject a fake extractor and never hit the network.
    """
    from daemon import resolve_newsletter_llm_endpoint
    from llm_client import LLMClient
    from newsletter import NewsletterClassifier

    base_url, api_key = resolve_newsletter_llm_endpoint()
    nl_llm = config["newsletter"]["llm"]
    client = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=nl_llm["model"],
        temperature=nl_llm.get("temperature", 0.0),
        max_tokens=nl_llm.get("max_tokens", 2048),
        extra_body=nl_llm.get("extra_body"),
    )
    classifier = NewsletterClassifier(client, config)
    extraction_config = classifier.extraction_config

    def extract_fn(body: str) -> str:
        user_content = extraction_config["user_template"].format(body=body)

        async def _run():
            raw, _ = await client.complete(
                extraction_config["system"], user_content, include_thinking=True
            )
            return raw

        return asyncio.run(_run())

    return extract_fn


# ---------------------------------------------------------------------------
# Curses helpers (mirroring evals.edit_tui)
# ---------------------------------------------------------------------------

def _safe_addstr(win, y, x, text, attr=curses.A_NORMAL):
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x >= max_x:
        return
    available = max_x - x - 1
    if available <= 0:
        return
    win.addnstr(y, x, text, available, attr)


def _prompt_line(stdscr, prompt: str) -> str:
    """Read a line of text at the bottom of the screen (Enter-terminated)."""
    max_y, max_x = stdscr.getmaxyx()
    _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
    _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
    stdscr.refresh()
    curses.echo()
    curses.curs_set(1)
    try:
        raw = stdscr.getstr(max_y - 1, len(prompt) + 1)
    finally:
        curses.noecho()
        curses.curs_set(0)
    return raw.decode("utf-8", "replace").strip()


def _newsletter_lines(newsletter, index, total) -> list[str]:
    lines = [
        f"Newsletter {index + 1}/{total}  (id: {newsletter.thread_id})",
        "=" * 60,
        f"Subject:  {newsletter.subject}",
        f"Sender:   {newsletter.sender}",
        f"Reviewed: {newsletter.reviewed}",
        f"Seeded:   {newsletter.seeded_from or '-'}",
    ]
    if newsletter.notes:
        lines.append(f"Notes:    {newsletter.notes}")
    lines.append("")
    lines.append(f"--- Stories ({len(newsletter.stories)}) ---")
    for i, s in enumerate(newsletter.stories):
        flags = []
        if s.excluded:
            flags.append("EXCLUDED")
        if s.reviewed:
            flags.append(s.expected_tier or "reviewed")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"[{i}] {s.title}{flag_str}")
        lines.append(f"    themes: {', '.join(s.expected_themes) or '-'}")
    lines.append("")
    lines.append("--- Body ---")
    lines.extend(newsletter.body.splitlines())
    return lines


def _prompt_scores(stdscr) -> dict[str, int] | None:
    """Prompt for the 4 dimension scores 1-5. Returns None on cancel."""
    scores = {}
    for dim in _DIMENSIONS:
        max_y, max_x = stdscr.getmaxyx()
        prompt = f"{dim} [1-5] (other cancels): "
        _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
        _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
        stdscr.refresh()
        key = stdscr.getch()
        ch = chr(key) if 0 <= key < 256 else ""
        if ch not in "12345":
            return None
        scores[dim] = int(ch)
    return scores


def _prompt_themes(stdscr) -> list[str]:
    """Multi-select themes by toggling s/c/h/v/d; Enter to finish."""
    selected: list[str] = []
    while True:
        max_y, max_x = stdscr.getmaxyx()
        legend = "  ".join(f"[{k}]{v}" for k, v in _THEME_KEYS.items())
        prompt = f"Themes {selected}: {legend}  Enter=done"
        _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
        _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return selected
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if ch in _THEME_KEYS:
            theme = _THEME_KEYS[ch]
            if theme in selected:
                selected.remove(theme)
            else:
                selected.append(theme)


# ---------------------------------------------------------------------------
# Detail view — Phase A (curate) then Phase B (label) for one newsletter
# ---------------------------------------------------------------------------

def _newsletter_detail(stdscr, newsletters, index, all_newsletters, path, *, extract_fn=None):
    """Curate + label one newsletter. Returns "back" or "quit"."""
    newsletter = newsletters[index]
    total = len(newsletters)
    scroll_y = 0
    snapshot = None  # one snapshot per newsletter for undo

    def save():
        _auto_save(all_newsletters, path, stdscr)

    while True:
        lines = _newsletter_lines(newsletter, index, total)
        max_y, max_x = stdscr.getmaxyx()
        content_rows = max(1, max_y - 2)
        stdscr.clear()
        for row_i in range(content_rows):
            li = scroll_y + row_i
            if li >= len(lines):
                break
            _safe_addstr(stdscr, row_i, 0, lines[li])
        help_text = (
            "[Space]seed [a]dd [e]dit [x]split [m]erge [d]el [c]onfirm "
            "[l]abel [u]nexclude/exclude [n]notes [z]undo Esc:Back q:Quit"
        )
        _safe_addstr(stdscr, max_y - 2, 0, help_text, curses.A_DIM)
        max_scroll = max(0, len(lines) - content_rows)
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and scroll_y > 0:
            scroll_y -= 1
        elif key == curses.KEY_DOWN and scroll_y < max_scroll:
            scroll_y += 1
        elif key == curses.KEY_NPAGE:
            scroll_y = min(max_scroll, scroll_y + content_rows)
        elif key == curses.KEY_PPAGE:
            scroll_y = max(0, scroll_y - content_rows)

        elif key == ord(" "):  # seed via extractor
            if extract_fn is not None:
                snapshot = capture_snapshot(newsletter)
                try:
                    seed_from_extractor(newsletter, extract_fn)
                    save()
                except Exception as exc:
                    _safe_addstr(stdscr, max_y - 1, 0, f"Seed failed: {exc}", curses.A_REVERSE)
                    stdscr.refresh()
                    stdscr.getch()

        elif key == ord("a"):  # add story
            snapshot = capture_snapshot(newsletter)
            title = _prompt_line(stdscr, "New title:")
            text = _prompt_line(stdscr, "New text:")
            if title and text:
                add_story(newsletter, title, text)
                save()

        elif key == ord("e"):  # edit story
            snapshot = capture_snapshot(newsletter)
            si = _prompt_index(stdscr, newsletter)
            if si is not None:
                title = _prompt_line(stdscr, "Title (blank=keep):")
                text = _prompt_line(stdscr, "Text (blank=keep):")
                edit_story(
                    newsletter, si,
                    title=title or None,
                    text=text or None,
                )
                save()

        elif key == ord("x"):  # split
            snapshot = capture_snapshot(newsletter)
            si = _prompt_index(stdscr, newsletter)
            if si is not None:
                marker = _prompt_line(stdscr, "Split at text:")
                if marker:
                    split_story(newsletter, si, marker)
                    save()

        elif key == ord("m"):  # merge with successor
            snapshot = capture_snapshot(newsletter)
            si = _prompt_index(stdscr, newsletter)
            if si is not None:
                merge_stories(newsletter, si)
                save()

        elif key == ord("d"):  # delete
            snapshot = capture_snapshot(newsletter)
            si = _prompt_index(stdscr, newsletter)
            if si is not None:
                delete_story(newsletter, si)
                save()

        elif key == ord("c"):  # confirm story list (Phase A done)
            snapshot = capture_snapshot(newsletter)
            confirm_story_list(newsletter)
            save()

        elif key == ord("l"):  # label a story (Phase B)
            si = _prompt_index(stdscr, newsletter)
            if si is not None:
                snapshot = capture_snapshot(newsletter)
                scores = _prompt_scores(stdscr)
                if scores is not None:
                    themes = _prompt_themes(stdscr)
                    assign_scores_and_themes(newsletter.stories[si], scores, themes)
                    save()

        elif key == ord("u"):  # toggle exclude on a story
            si = _prompt_index(stdscr, newsletter)
            if si is not None:
                snapshot = capture_snapshot(newsletter)
                story = newsletter.stories[si]
                story.excluded = not story.excluded
                save()

        elif key == ord("n"):  # newsletter notes
            snapshot = capture_snapshot(newsletter)
            newsletter.notes = _prompt_line(stdscr, "Notes:")
            save()

        elif key == ord("z"):  # undo one snapshot
            if snapshot is not None:
                restore_snapshot(newsletter, snapshot)
                snapshot = None
                save()

        elif key == 27:  # Esc
            return "back"
        elif key == ord("q"):
            return "quit"


def _prompt_index(stdscr, newsletter) -> int | None:
    """Prompt for a story index; return it or None if out of range/blank."""
    raw = _prompt_line(stdscr, "Story #:")
    if not raw.isdigit():
        return None
    si = int(raw)
    if 0 <= si < len(newsletter.stories):
        return si
    return None


def _auto_save(all_newsletters, path, stdscr) -> None:
    try:
        save_golden_set(all_newsletters, path)
    except Exception as exc:
        max_y, _ = stdscr.getmaxyx()
        _safe_addstr(stdscr, max_y - 1, 0, f"Save failed: {exc}", curses.A_REVERSE)
        stdscr.refresh()
        stdscr.getch()


# ---------------------------------------------------------------------------
# List view + loop
# ---------------------------------------------------------------------------

def _list_view(stdscr, newsletters, all_newsletters, path, *, extract_fn=None):
    cursor = 0
    scroll_offset = 0
    while True:
        max_y, max_x = stdscr.getmaxyx()
        page_size = max(1, max_y - 3)
        stdscr.clear()
        _safe_addstr(stdscr, 0, 0, f"Newsletter Label — {len(newsletters)} newsletters",
                     curses.A_BOLD)
        _safe_addstr(stdscr, 1, 0, f"{'R':<2}  {'Stories':<8}  Subject", curses.A_UNDERLINE)
        for vi in range(page_size):
            ti = scroll_offset + vi
            if ti >= len(newsletters):
                break
            n = newsletters[ti]
            r = "Y" if n.reviewed else " "
            row = f"{r:<2}  {len(n.stories):<8}  {n.subject}"
            attr = curses.A_REVERSE if ti == cursor else curses.A_NORMAL
            _safe_addstr(stdscr, 2 + vi, 0, row, attr)
        _safe_addstr(stdscr, max_y - 1, 0,
                     "↑/↓:Nav  Enter:Open  q:Quit", curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and cursor > 0:
            cursor -= 1
            if cursor < scroll_offset:
                scroll_offset = cursor
        elif key == curses.KEY_DOWN and cursor < len(newsletters) - 1:
            cursor += 1
            if cursor >= scroll_offset + page_size:
                scroll_offset = cursor - page_size + 1
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            result = _newsletter_detail(
                stdscr, newsletters, cursor, all_newsletters, path, extract_fn=extract_fn
            )
            if result == "quit":
                return
        elif key == ord("q"):
            return


def _curses_main(stdscr, newsletters, all_newsletters, path, extract_fn):
    curses.curs_set(0)
    stdscr.keypad(True)
    _list_view(stdscr, newsletters, all_newsletters, path, extract_fn=extract_fn)


def label_loop(newsletters, all_newsletters, path, *, extract_fn=None):
    """Launch the curses labeling TUI. Saving is atomic + auto on each edit."""
    if not newsletters:
        print("No newsletters to label.")
        return
    curses.wrapper(_curses_main, newsletters, all_newsletters, path, extract_fn)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Label newsletter golden set (curate + score)")
    parser.add_argument(
        "--golden-set", default="evals/newsletter_golden_set.jsonl",
        help="Path to newsletter golden set JSONL",
    )
    parser.add_argument("--edit", action="store_true",
                        help="Edit mode: no LLM seeding (curate/label existing stories)")
    parser.add_argument("--unreviewed-only", action="store_true",
                        help="Queue only unreviewed newsletters")
    parser.add_argument("--config", help="Path to config.toml (default: ./config.toml)")
    args = parser.parse_args()

    path = Path(args.golden_set)
    if not path.exists():
        print(f"Golden set not found: {path}", file=sys.stderr)
        print("Run 'python -m evals.newsletter_harvest' first to create it.", file=sys.stderr)
        sys.exit(1)

    all_newsletters = load_golden_set(path)
    if not all_newsletters:
        print("Golden set is empty.", file=sys.stderr)
        sys.exit(1)

    newsletters = select_label_newsletters(all_newsletters, unreviewed_only=args.unreviewed_only)
    if not newsletters:
        print("No newsletters match the filters.", file=sys.stderr)
        sys.exit(0)

    # Build the Phase-A seed extractor unless --edit (edit mode never hits an LLM).
    extract_fn = None
    if not args.edit:
        from evals.newsletter_harvest import load_eval_config

        extract_fn = build_extractor(load_eval_config(args.config))

    label_loop(newsletters, all_newsletters, path, extract_fn=extract_fn)

    # The TUI auto-saves on each edit; save once more so a monkeypatched loop
    # (and any final in-memory state) is persisted. Saving all_newsletters keeps
    # excluded/filtered-out newsletters and their order untouched.
    save_golden_set(all_newsletters, path)

    reviewed_count = sum(1 for n in newsletters if n.reviewed)
    print(f"\nSaved. {reviewed_count}/{len(newsletters)} newsletters reviewed.")


if __name__ == "__main__":
    cli()
