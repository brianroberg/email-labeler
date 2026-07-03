"""Interactive CLI to build newsletter golden-set labels (two phases).

Usage:
    python -m evals.newsletter_label                     # curate + label
    python -m evals.newsletter_label --edit              # manual-only (no LLM seeding)
    python -m evals.newsletter_label --unreviewed-only

Phase A (curate stories / extraction truth): press ``Space`` to seed candidate
stories by running the production ``newsletter.parse_stories`` over a fresh
LLM extraction of the body (an uncached call on every press; the seed is a
deletable starting point, and re-seeding over an existing list asks for
confirmation), then build the authoritative story list
by marking body segments — move the cursor over the rendered body, press ``s``/
``e`` to set the selection start/end line, and ``Enter`` to make a story from
that inclusive span — plus add/edit/delete candidates. Body lines already
covered by a story are dimmed, so paragraphs the seed omitted stand out.
Confirming the list sets ``newsletter.reviewed=True`` and assigns each story a
stable ``story_id = f"{thread_id}:{index}"``. A newsletter can also be skipped
(``k``) without marking it reviewed, so it resurfaces in a later pass.

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
import unicodedata
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
    ``parse_stories`` turns that into (title, text) pairs. Returns
    ``(stories, raw)`` so the caller can report the outcome (e.g. distinguish
    a NO_STORIES verdict from unparseable output — see ``seed_outcome_message``).
    """
    raw = extract_fn(newsletter.body)
    return seed_stories(newsletter, parse_stories(raw)), raw


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


def create_story_from_body(newsletter, start_line, end_line, title):
    """Build a candidate story from an inclusive span of body lines.

    *start_line* / *end_line* are indices into ``newsletter.body.splitlines()``;
    they are normalized (min/max) and clamped to the valid range, so order and
    out-of-range values are tolerated. The span text is the inclusive
    ``lo..hi`` slice joined with newlines. A falsy *title* is auto-derived from
    the first ~8 words of the segment. Appends the story with a stable
    ``story_id``; ``newsletter.reviewed`` is left untouched.
    """
    body_lines = newsletter.body.splitlines()
    last = max(0, len(body_lines) - 1)
    lo = max(0, min(start_line, end_line))
    hi = min(last, max(start_line, end_line))
    text = "\n".join(body_lines[lo:hi + 1])
    if not title:
        title = " ".join(text.split()[:8]).strip() or "(untitled)"
    story = GoldenStory(
        story_id=f"{newsletter.thread_id}:{len(newsletter.stories)}",
        title=title,
        text=text,
    )
    newsletter.stories.append(story)
    return story


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
# Pure helpers for the TUI (no curses; fully testable)
# ---------------------------------------------------------------------------

_HELP_ITEMS = (
    "↑/↓:move", "PgUp/PgDn:page", "s/e:select", "Enter:make-story", "Space:seed",
    "[a]dd", "[E]dit", "[d]el", "[c]onfirm", "[l]abel", "[u]exclude", "[n]otes",
    "[z]undo", "[k]skip", "Esc:back", "q:quit",
)


def format_help_lines(width: int) -> list[str]:
    """Pack the detail-view hotkey help into lines of at most *width* chars.

    Wide terminals get one line; narrow ones wrap so no hotkey is ever hidden.
    """
    lines: list[str] = []
    current = ""
    for item in _HELP_ITEMS:
        candidate = f"{current}  {item}" if current else item
        if current and len(candidate) > max(1, width):
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def seed_confirmation_message(newsletter) -> str | None:
    """Confirmation to show before Space re-seeds over existing stories.

    Returns None when seeding is safe (empty story list); otherwise a y/N
    question stating how many stories — and how many carrying Phase-B labels —
    would be discarded.
    """
    n = len(newsletter.stories)
    if n == 0:
        return None
    labeled = sum(1 for s in newsletter.stories if s.expected_scores is not None)
    detail = f" ({labeled} labeled)" if labeled else ""
    return f"Replace {_stories(n)}{detail} with a fresh seed? y/N"


def _stories(n: int) -> str:
    """'1 story' / 'N stories' — mirror the run/report modules' _plural."""
    return "1 story" if n == 1 else f"{n} stories"


def seed_outcome_message(raw: str, story_count: int) -> str:
    """Status line after a seed: distinguishes NO_STORIES from a parse washout."""
    if story_count:
        return f"Seeded {_stories(story_count)}."
    if raw.strip().upper() == "NO_STORIES":
        return "Extractor returned NO_STORIES."
    return "Extractor output had no parseable story blocks."


def confirm_status_message(newsletter) -> str:
    """Status line after ``c`` confirms the story list (Phase A done).

    Mentions how many stories still lack Phase-B labels so a user doesn't quit
    thinking the newsletter is fully done.
    """
    unlabeled = unlabeled_story_count(newsletter)
    status = "Story list confirmed."
    if unlabeled:
        status += (
            f" {_stories(unlabeled)} still unlabeled — press l to score them."
        )
    return status


def row_for_body_line(rows, body_idx) -> int:
    """First rendered-row index carrying *body_idx* (see ``build_detail_rows``).

    Used to re-anchor the cursor to the body line it was on after the story
    list above the body grows/shrinks. Falls back to the last row (or 0).
    """
    for ri, (_, bi) in enumerate(rows):
        if bi == body_idx:
            return ri
    return max(0, len(rows) - 1)


def covered_body_lines(newsletter) -> set[int]:
    """Body-line indices whose text already appears in some story.

    Good seeds are verbatim spans of the body, so substring matching of each
    non-blank body line against the story texts marks what is covered — the
    view dims covered lines so OMITTED paragraphs stand out.
    """
    texts = [s.text for s in newsletter.stories]
    covered = set()
    for i, line in enumerate(newsletter.body.splitlines()):
        stripped = line.strip()
        if stripped and any(stripped in t for t in texts):
            covered.add(i)
    return covered


def label_progress(newsletter) -> tuple[int, int]:
    """(labeled, total) over the newsletter's non-excluded stories."""
    stories = [s for s in newsletter.stories if not s.excluded]
    labeled = sum(1 for s in stories if s.expected_scores is not None)
    return labeled, len(stories)


def unlabeled_story_count(newsletter) -> int:
    """Non-excluded stories still missing Phase-B scores."""
    labeled, total = label_progress(newsletter)
    return total - labeled


def format_list_header() -> str:
    return f"{'R':<2} {'Lbl':<6} {'Sender':<24} Subject"


def format_list_row(newsletter) -> str:
    """One list-view row: reviewed flag, labeled/total, sender, subject."""
    r = "Y" if newsletter.reviewed else " "
    labeled, total = label_progress(newsletter)
    sender = (newsletter.sender or "")[:24]
    return f"{r:<2} {f'{labeled}/{total}':<6} {sender:<24} {newsletter.subject}"


class LineBuffer:
    """Pure single-line editor state for prompts (no curses).

    Supports prefilling with the current value (so edits are incremental, not
    blind retyping), horizontal scrolling via ``visible`` (so input longer than
    the terminal is never silently truncated), and rejects control characters
    (so a stray Esc can never end up as a literal ``\\x1b`` in the golden set).
    """

    def __init__(self, initial: str = ""):
        self._chars = [c for c in str(initial) if c.isprintable()]
        self._pos = len(self._chars)

    def text(self) -> str:
        return "".join(self._chars)

    def insert(self, ch: str) -> None:
        if len(ch) == 1 and ch.isprintable():
            self._chars.insert(self._pos, ch)
            self._pos += 1

    def backspace(self) -> None:
        if self._pos > 0:
            self._pos -= 1
            self._chars.pop(self._pos)

    def left(self) -> None:
        self._pos = max(0, self._pos - 1)

    def right(self) -> None:
        self._pos = min(len(self._chars), self._pos + 1)

    def visible(self, width: int) -> tuple[str, int]:
        """(visible slice, cursor column) for a window of *width* cells."""
        width = max(1, width)
        start = self._pos - width + 1 if self._pos >= width else 0
        return "".join(self._chars[start:start + width]), self._pos - start


def format_theme_legend(selected, width: int) -> str:
    """Theme-picker prompt line; falls back to a compact form on narrow screens."""
    full_legend = "  ".join(f"[{k}]{v}" for k, v in _THEME_KEYS.items())
    full = f"Themes {sorted(selected)}: {full_legend}  Enter=done"
    if len(full) <= width:
        return full
    compact_legend = " ".join(f"[{k}]{v[:3]}" for k, v in _THEME_KEYS.items())
    compact = f"Themes {','.join(selected) or '-'}: {compact_legend} Enter=done"
    if len(compact) <= width:
        return compact
    return f"Themes({len(selected)}): {compact_legend} Enter=done"


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------

# Mutable newsletter-level fields captured for undo (stories handled separately)
_NL_SNAPSHOT_FIELDS = ("seeded_from", "reviewed", "notes", "excluded")


def capture_snapshot(newsletter):
    """Deep-copy a newsletter's mutable state so a later edit can be undone.

    Captures the full story list plus the newsletter-level mutable fields.
    The TUI keeps a STACK of these per newsletter, pushed only when a mutation
    actually happens — cancelled prompts never consume an undo level.
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
    if not base_url:
        # Fail fast, before curses starts — otherwise the missing endpoint only
        # surfaces as a truncated error when Space is pressed deep in the TUI.
        raise SystemExit(
            "No LLM endpoint configured for Phase-A seeding: set NEWSLETTER_LLM_URL "
            "or CLOUD_LLM_URL, or run with --edit to curate without seeding."
        )
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
# Pure rendering helpers (no curses; fully testable)
# ---------------------------------------------------------------------------

def display_width(text: str) -> int:
    """Terminal cell width of *text* (east-asian wide chars & emoji count 2)."""
    total = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _wrap_paragraph(paragraph: str, width: int) -> list[str]:
    """Greedy word-wrap by DISPLAY width so wide chars never overflow the row."""
    lines: list[str] = []
    cur_words: list[str] = []
    cur_w = 0
    for word in paragraph.split():
        w = display_width(word)
        if cur_words and cur_w + 1 + w <= width:
            cur_words.append(word)
            cur_w += 1 + w
            continue
        if cur_words:
            lines.append(" ".join(cur_words))
            cur_words = []
        # Word starts a fresh line; break it if wider than the screen.
        while display_width(word) > width:
            prefix = ""
            prefix_w = 0
            for ch in word:
                ch_w = display_width(ch)
                if prefix and prefix_w + ch_w > width:
                    break
                prefix += ch
                prefix_w += ch_w
            lines.append(prefix)
            word = word[len(prefix):]
        cur_words = [word] if word else []
        cur_w = display_width(word)
    if cur_words:
        lines.append(" ".join(cur_words))
    return lines or [""]


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap *text* to *width* terminal cells, preserving existing newlines.

    Mirrors ``newsletter_review.tui.wrap_text`` (``width <= 0`` disables
    wrapping and empty *text* yields ``[""]``) but wraps by display width, so
    emoji/wide characters are never clipped at the screen edge.
    """
    if not text:
        return [""]
    lines = []
    for paragraph in text.splitlines():
        if width <= 0:
            lines.append(paragraph)
        else:
            lines.extend(_wrap_paragraph(paragraph, width))
    return lines


def build_detail_rows(newsletter, index, total, width) -> list[tuple[str, int | None]]:
    """Build the wrapped detail-view rows with body-line provenance.

    Each returned row is ``(text, body_line_index_or_None)``. Header/metadata/
    story-list rows carry ``None``. Each logical body line (from
    ``body.splitlines()``) is wrapped to *width*, and every resulting physical
    row carries THAT body line's index — the provenance map the view uses for
    selection and highlighting.
    """
    rows: list[tuple[str, int | None]] = []

    def add_header(text: str) -> None:
        for line in wrap_text(text, width):
            rows.append((line, None))

    add_header(f"Newsletter {index + 1}/{total}  (id: {newsletter.thread_id})")
    add_header("=" * max(1, min(60, width)))
    add_header(f"Subject:  {newsletter.subject}")
    add_header(f"Sender:   {newsletter.sender}")
    add_header(f"Reviewed: {newsletter.reviewed}")
    add_header(f"Seeded:   {newsletter.seeded_from or '-'}")
    if newsletter.notes:
        add_header(f"Notes:    {newsletter.notes}")
    add_header("")
    add_header(f"--- Stories ({len(newsletter.stories)}) ---")
    for i, s in enumerate(newsletter.stories):
        flags = []
        if s.excluded:
            flags.append("EXCLUDED")
        if s.reviewed:
            flags.append(s.expected_tier or "reviewed")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        add_header(f"[{i}] {s.title}{flag_str}")
        # Show the full parsed text inline (indented), wrapped to the width.
        for text_line in wrap_text(s.text, max(1, width - 6)):
            add_header(f"      {text_line}")
        label_bits = []
        if s.expected_scores is not None:
            dims = "/".join(str(s.expected_scores.get(d, "?")) for d in _DIMENSIONS)
            label_bits.append(f"scores: {dims}")
        label_bits.append(f"themes: {', '.join(s.expected_themes) or '-'}")
        add_header("    " + "   ".join(label_bits))
    add_header("")
    add_header("--- Body ---")

    for body_idx, body_line in enumerate(newsletter.body.splitlines()):
        for physical in wrap_text(body_line, width):
            rows.append((physical, body_idx))
    return rows


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


def _set_cursor_visibility(visible: bool) -> None:
    try:
        curses.curs_set(1 if visible else 0)
    except curses.error:
        pass


def _prompt_line(stdscr, prompt: str, initial: str = "") -> str | None:
    """Line editor at the bottom of the screen. Returns None on Esc (cancel).

    Prefills with *initial* (edit the current value instead of retyping it
    blind), scrolls horizontally so long input is never silently truncated,
    and rejects control characters (a stray Esc can't pollute the golden set).
    """
    buf = LineBuffer(initial)
    _set_cursor_visibility(True)
    try:
        while True:
            max_y, max_x = stdscr.getmaxyx()
            avail = max(1, max_x - len(prompt) - 2)
            text, cur = buf.visible(avail)
            _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
            _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
            _safe_addstr(stdscr, max_y - 1, len(prompt) + 1, text)
            try:
                stdscr.move(max_y - 1, min(max_x - 1, len(prompt) + 1 + cur))
            except curses.error:
                pass
            stdscr.refresh()
            try:
                key = stdscr.get_wch()
            except curses.error:
                continue
            if isinstance(key, str):
                if key in ("\n", "\r"):
                    return buf.text().strip()
                if key == "\x1b":
                    return None
                if key in ("\x7f", "\x08"):
                    buf.backspace()
                else:
                    buf.insert(key)
            elif key == curses.KEY_ENTER:
                return buf.text().strip()
            elif key == curses.KEY_BACKSPACE:
                buf.backspace()
            elif key == curses.KEY_LEFT:
                buf.left()
            elif key == curses.KEY_RIGHT:
                buf.right()
    finally:
        _set_cursor_visibility(False)


def _confirm(stdscr, message: str) -> bool:
    """Bottom-line y/N confirmation; only y/Y confirms."""
    max_y, max_x = stdscr.getmaxyx()
    _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
    _safe_addstr(stdscr, max_y - 1, 0, message, curses.A_REVERSE)
    stdscr.refresh()
    return stdscr.getch() in (ord("y"), ord("Y"))


def _prompt_scores(stdscr, current=None) -> dict[str, int] | None:
    """Prompt for the 4 dimension scores 1-5. Returns None on cancel.

    When re-labeling, *current* shows the value already assigned per dimension;
    accepted digits are echoed in the prompt so entry is verifiable.
    """
    scores = {}
    for dim in _DIMENSIONS:
        max_y, max_x = stdscr.getmaxyx()
        now = f" (now {current[dim]})" if current and dim in current else ""
        entered = "/".join(str(scores[d]) for d in _DIMENSIONS if d in scores)
        so_far = f"[{entered}] " if entered else ""
        prompt = f"{so_far}{dim}{now} [1-5] (other cancels): "
        _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
        _safe_addstr(stdscr, max_y - 1, 0, prompt, curses.A_REVERSE)
        stdscr.refresh()
        key = stdscr.getch()
        ch = chr(key) if 0 <= key < 256 else ""
        if ch not in "12345":
            return None
        scores[dim] = int(ch)
    return scores


def _prompt_themes(stdscr, initial=None) -> list[str]:
    """Multi-select themes by toggling s/c/h/v/d; Enter to finish.

    Starts from *initial* (the story's current themes) so re-labeling edits
    rather than restarts. The legend compacts on narrow terminals.
    """
    selected: list[str] = list(initial or [])
    while True:
        max_y, max_x = stdscr.getmaxyx()
        prompt = format_theme_legend(selected, max_x - 1)
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

_MAX_UNDO = 100  # bound the per-newsletter undo stack


def _newsletter_detail(stdscr, newsletters, index, all_newsletters, path, *, extract_fn=None):
    """Curate + label one newsletter. Returns "back", "skip", or "quit"."""
    newsletter = newsletters[index]
    total = len(newsletters)
    scroll_y = 0
    cursor = 0  # index into rendered rows
    sel_start = None  # selected body-line index
    sel_end = None
    undo_stack: list[dict] = []  # snapshots, pushed only when a mutation happens
    status = ""  # one-shot feedback line (cleared on the next keypress)
    anchor_body = None  # body line to re-anchor the cursor to after a mutation

    def save():
        _auto_save(all_newsletters, path, stdscr)

    def push_undo():
        undo_stack.append(capture_snapshot(newsletter))
        del undo_stack[:-_MAX_UNDO]

    def hint(msg):
        # Blocking notice (used for errors): wrapped so nothing truncates.
        max_y, max_x = stdscr.getmaxyx()
        lines = wrap_text(f"{msg}  (press any key)", max_x - 1)
        start = max(0, max_y - len(lines))
        for i, line in enumerate(lines[:max_y]):
            _safe_addstr(stdscr, start + i, 0, " " * (max_x - 1))
            _safe_addstr(stdscr, start + i, 0, line, curses.A_REVERSE)
        stdscr.refresh()
        stdscr.getch()

    while True:
        max_y, max_x = stdscr.getmaxyx()
        rows = build_detail_rows(newsletter, index, total, max_x - 2)
        if anchor_body is not None:
            # The story list above the body grew/shrank: keep the cursor on
            # the same BODY line, not the same physical row.
            cursor = row_for_body_line(rows, anchor_body)
            anchor_body = None
        help_lines = format_help_lines(max_x - 1)
        content_rows = max(1, max_y - len(help_lines) - 1)
        cursor = max(0, min(cursor, len(rows) - 1))
        # Auto-scroll so the cursor row stays visible.
        if cursor < scroll_y:
            scroll_y = cursor
        elif cursor >= scroll_y + content_rows:
            scroll_y = cursor - content_rows + 1
        max_scroll = max(0, len(rows) - content_rows)

        lo = hi = None
        if sel_start is not None and sel_end is not None:
            lo, hi = min(sel_start, sel_end), max(sel_start, sel_end)
        elif sel_start is not None:
            lo = hi = sel_start
        covered = covered_body_lines(newsletter)

        stdscr.clear()
        for row_i in range(content_rows):
            ri = scroll_y + row_i
            if ri >= len(rows):
                break
            text, body_idx = rows[ri]
            selected = (
                body_idx is not None and lo is not None and lo <= body_idx <= hi
            )
            # Column 0 is reserved for the selection marker on EVERY row, so
            # text never shifts when a selection appears.
            if selected:
                _safe_addstr(stdscr, row_i, 0, "*", curses.A_BOLD)
            if ri == cursor:
                # `or " "` keeps the cursor visible on blank body lines.
                _safe_addstr(stdscr, row_i, 1, text or " ", curses.A_REVERSE)
            elif selected:
                _safe_addstr(stdscr, row_i, 1, text, curses.A_BOLD)
            elif body_idx is not None and body_idx in covered:
                # Dim body lines already captured by a story, so paragraphs
                # the seed OMITTED stand out at normal brightness.
                _safe_addstr(stdscr, row_i, 1, text, curses.A_DIM)
            else:
                _safe_addstr(stdscr, row_i, 1, text)
        for i, help_line in enumerate(help_lines):
            _safe_addstr(stdscr, max_y - 1 - len(help_lines) + i, 0, help_line, curses.A_DIM)
        if status:
            _safe_addstr(stdscr, max_y - 1, 0, status, curses.A_REVERSE)
        else:
            _safe_addstr(stdscr, max_y - 1, 0, f"row {cursor + 1}/{len(rows)}", curses.A_DIM)
        stdscr.refresh()

        key = stdscr.getch()
        status = ""

        if key == curses.KEY_UP and cursor > 0:
            cursor -= 1
        elif key == curses.KEY_DOWN and cursor < len(rows) - 1:
            cursor += 1
        elif key == curses.KEY_NPAGE:
            cursor = min(len(rows) - 1, cursor + content_rows)
            scroll_y = min(max_scroll, scroll_y + content_rows)
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - content_rows)
            scroll_y = max(0, scroll_y - content_rows)

        elif key == ord(" "):  # seed via extractor
            if extract_fn is None:
                status = "Seeding disabled in edit mode (run without --edit to seed)."
            else:
                confirm_msg = seed_confirmation_message(newsletter)
                if confirm_msg is not None and not _confirm(stdscr, confirm_msg):
                    status = "Seed cancelled — stories kept."
                else:
                    _safe_addstr(stdscr, max_y - 1, 0, " " * (max_x - 1))
                    _safe_addstr(stdscr, max_y - 1, 0, "Seeding…", curses.A_REVERSE)
                    stdscr.refresh()
                    push_undo()
                    try:
                        stories, raw = seed_from_extractor(newsletter, extract_fn)
                    except Exception as exc:
                        undo_stack.pop()  # nothing changed; keep prior undo intact
                        hint(f"Seed failed: {exc}")
                    else:
                        save()
                        status = seed_outcome_message(raw, len(stories))

        elif key == ord("s"):  # set selection start
            body_idx = rows[cursor][1]
            if body_idx is None:
                status = "Move the cursor onto a body line first."
            else:
                sel_start = body_idx
                status = f"Selection start = body line {body_idx} (Esc clears)."

        elif key == ord("e"):  # set selection end
            body_idx = rows[cursor][1]
            if body_idx is None:
                status = "Move the cursor onto a body line first."
            else:
                sel_end = body_idx
                status = f"Selection end = body line {body_idx} (Esc clears)."

        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):  # make story
            if sel_start is None:
                status = "Set a selection start with 's' first."
            else:
                s_line = sel_start
                e_line = sel_end if sel_end is not None else sel_start
                title = _prompt_line(stdscr, "Title (blank=auto):")
                if title is None:
                    status = "Story creation cancelled."
                else:
                    anchor_body = rows[cursor][1]
                    push_undo()
                    story = create_story_from_body(newsletter, s_line, e_line, title)
                    save()
                    sel_start = sel_end = None
                    status = f"Story created: {story.title}"

        elif key == ord("a"):  # add story
            title = _prompt_line(stdscr, "New title:")
            text = _prompt_line(stdscr, "New text:") if title else None
            if title and text:
                push_undo()
                add_story(newsletter, title, text)
                save()
                status = f"Story added: {title}"
            else:
                status = "Add cancelled — need a title and text."

        elif key == ord("E"):  # edit story
            si = _prompt_index(stdscr, newsletter)
            if si is None:
                status = "No valid story # — nothing edited."
            else:
                story = newsletter.stories[si]
                new_title = _prompt_line(stdscr, "Title (blank=keep):", initial=story.title)
                if new_title is None:
                    status = "Edit cancelled."
                else:
                    # Multi-line text can't be edited in a one-line prompt;
                    # fall back to blank=keep for it (body selection is the
                    # intended repair path for paragraph-level text).
                    text_initial = story.text if "\n" not in story.text else ""
                    new_text = _prompt_line(
                        stdscr, "Text (blank=keep):", initial=text_initial
                    )
                    if new_text is None:
                        status = "Edit cancelled."
                    else:
                        new_title = new_title or story.title
                        new_text = new_text or story.text
                        if new_title == story.title and new_text == story.text:
                            status = "No changes."
                        else:
                            push_undo()
                            edit_story(newsletter, si, title=new_title, text=new_text)
                            save()
                            status = f"Story [{si}] updated."

        elif key == ord("d"):  # delete
            si = _prompt_index(stdscr, newsletter)
            if si is None:
                status = "No valid story # — nothing deleted."
            else:
                story = newsletter.stories[si]
                if story.expected_scores is not None and not _confirm(
                    stdscr, f"[{si}] {story.title} has labels — delete anyway? y/N"
                ):
                    status = "Delete cancelled."
                else:
                    push_undo()
                    delete_story(newsletter, si)
                    save()
                    status = f"Deleted [{si}] {story.title} (z to undo)."

        elif key == ord("c"):  # confirm story list (Phase A done)
            push_undo()
            confirm_story_list(newsletter)
            save()
            status = confirm_status_message(newsletter)

        elif key == ord("l"):  # label a story (Phase B)
            si = _prompt_index(stdscr, newsletter)
            if si is None:
                status = "No valid story # — nothing labeled."
            else:
                story = newsletter.stories[si]
                scores = _prompt_scores(stdscr, current=story.expected_scores)
                if scores is None:
                    status = "Score entry cancelled — no changes saved."
                else:
                    themes = _prompt_themes(stdscr, initial=story.expected_themes)
                    push_undo()
                    assign_scores_and_themes(story, scores, themes)
                    save()
                    status = f"Labeled [{si}] {story.expected_tier}."

        elif key == ord("u"):  # toggle exclude on a story
            si = _prompt_index(stdscr, newsletter)
            if si is None:
                status = "No valid story # — nothing toggled."
            else:
                push_undo()
                story = newsletter.stories[si]
                story.excluded = not story.excluded
                save()
                status = f"[{si}] {'excluded from' if story.excluded else 'included in'} scoring."

        elif key == ord("n"):  # newsletter notes
            new_notes = _prompt_line(stdscr, "Notes:", initial=newsletter.notes)
            if new_notes is None:
                status = "Notes edit cancelled."
            elif new_notes == newsletter.notes:
                status = "Notes unchanged."
            else:
                push_undo()
                newsletter.notes = new_notes
                save()
                status = "Notes updated."

        elif key == ord("z"):  # undo the last mutation
            if undo_stack:
                restore_snapshot(newsletter, undo_stack.pop())
                save()
                status = f"Undone ({len(undo_stack)} more undo levels)."
            else:
                status = "Nothing to undo."

        elif key == ord("k"):  # skip this newsletter (never marks reviewed)
            return "skip"
        elif key == 27:  # Esc: clear an active selection first, then go back
            if sel_start is not None or sel_end is not None:
                sel_start = sel_end = None
                status = "Selection cleared."
            else:
                return "back"
        elif key == ord("q"):
            return "quit"


def _prompt_index(stdscr, newsletter) -> int | None:
    """Prompt for a story index; None if cancelled, non-numeric or out of range."""
    raw = _prompt_line(stdscr, "Story #:")
    if raw is None or not raw.isdigit():
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
        _safe_addstr(stdscr, 1, 0, format_list_header(), curses.A_UNDERLINE)
        for vi in range(page_size):
            ti = scroll_offset + vi
            if ti >= len(newsletters):
                break
            attr = curses.A_REVERSE if ti == cursor else curses.A_NORMAL
            _safe_addstr(stdscr, 2 + vi, 0, format_list_row(newsletters[ti]), attr)
        _safe_addstr(stdscr, max_y - 1, 0,
                     "↑/↓ PgUp/PgDn:Nav  Enter:Open  q:Quit", curses.A_DIM)
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
        elif key == curses.KEY_NPAGE:
            cursor = min(len(newsletters) - 1, cursor + page_size)
            if cursor >= scroll_offset + page_size:
                scroll_offset = cursor - page_size + 1
        elif key == curses.KEY_PPAGE:
            cursor = max(0, cursor - page_size)
            if cursor < scroll_offset:
                scroll_offset = cursor
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            while True:
                result = _newsletter_detail(
                    stdscr, newsletters, cursor, all_newsletters, path, extract_fn=extract_fn
                )
                if result == "quit":
                    return
                if result == "skip" and cursor < len(newsletters) - 1:
                    # Linear skip-through: advance and immediately open the next
                    # newsletter. Skipping never marks reviewed, so it resurfaces.
                    cursor += 1
                    if cursor >= scroll_offset + page_size:
                        scroll_offset = cursor - page_size + 1
                    continue
                # "back", or "skip" on the last newsletter -> return to list.
                break
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
    # Esc is the back key; the default ~1s ESCDELAY makes it feel broken.
    os.environ.setdefault("ESCDELAY", "25")
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
    parser.add_argument("--config",
                        help="Path to config.toml (default: the repo-root config.toml, "
                             "regardless of CWD)")
    args = parser.parse_args()

    path = Path(args.golden_set)
    if not path.exists():
        print(f"Golden set not found: {path}", file=sys.stderr)
        print(
            "Run 'python -m evals.newsletter_harvest' first to create it — its "
            "--output path must match the --golden-set path passed here.",
            file=sys.stderr,
        )
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
