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
Confirming the list sets ``newsletter.reviewed=True``; every story keeps its
stable ``story_id`` (``f"{thread_id}:{n}"``) and only missing/duplicate ids are
repaired with fresh unique ones. A newsletter can also be skipped
(``k``) without marking it reviewed, so it resurfaces in a later pass.

Phase B (per-story labels): assign the 4 dimension scores
(simple/concrete/personal/dynamic, 1-5) and multi-select themes; on save the
``expected_tier`` is derived via ``newsletter.compute_tier`` and
``story.reviewed`` is set.

The state transitions are factored into PURE functions (below) so they can be
unit-tested without a terminal; the Textual UI layer on top is tested with
Textual's Pilot driver.
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from evals import plural
from evals.newsletter_schemas import GoldenNewsletter, GoldenStory
from newsletter import compute_tier, parse_stories
from tui_common import BottomModal, HintScreen, PageListView, PromptLineScreen

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
    provenance. Replacing the list invalidates any prior confirmation, so
    ``newsletter.reviewed`` is reset to False — the uncurated machine seed must
    be re-confirmed before it counts as extraction truth (undo still restores
    the prior state, ``reviewed`` included, via the snapshot).
    """
    newsletter.stories = [
        GoldenStory(story_id=f"{newsletter.thread_id}:{i}", title=title, text=text)
        for i, (title, text) in enumerate(story_pairs)
    ]
    newsletter.seeded_from = "parse_stories"
    newsletter.reviewed = False
    return newsletter.stories


def _next_story_id(newsletter) -> str:
    """Next unique story id: ``max(existing numeric suffixes) + 1``.

    Deriving from the max — not ``len(stories)`` — means a delete followed by
    an add can never mint a duplicate id (downstream dicts key on story_id).
    Falls back to ``len(stories)`` when no story carries a numeric suffix.
    """
    prefix = f"{newsletter.thread_id}:"
    max_suffix = -1
    for story in newsletter.stories:
        sid = story.story_id or ""
        if sid.startswith(prefix) and sid[len(prefix):].isdigit():
            max_suffix = max(max_suffix, int(sid[len(prefix):]))
    if max_suffix < 0:
        return f"{prefix}{len(newsletter.stories)}"
    return f"{prefix}{max_suffix + 1}"


def add_story(newsletter, title, text):
    """Append a new candidate story with a stable, unique story_id.

    Does not confirm the list — ``newsletter.reviewed`` is untouched and the
    new story starts unreviewed.
    """
    story = GoldenStory(
        story_id=_next_story_id(newsletter),
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
    the first ~8 words of the segment. Appends via ``add_story`` (stable,
    unique ``story_id``); ``newsletter.reviewed`` is left untouched.
    """
    body_lines = newsletter.body.splitlines()
    last = max(0, len(body_lines) - 1)
    lo = max(0, min(start_line, end_line))
    hi = min(last, max(start_line, end_line))
    text = "\n".join(body_lines[lo:hi + 1])
    if not title:
        title = " ".join(text.split()[:8]).strip() or "(untitled)"
    return add_story(newsletter, title, text)


def edit_story(newsletter, index, *, title=None, text=None):
    """Edit a candidate's title and/or text span. Unspecified fields are kept."""
    story = newsletter.stories[index]
    if title is not None:
        story.title = title
    if text is not None:
        story.text = text


def delete_story(newsletter, index):
    """Remove a wrongly-extracted candidate entirely (not the same as exclude).

    Leaves ``newsletter.reviewed`` untouched; the remaining stories keep their
    ids (gaps are fine — ids are stable, never positional).
    """
    newsletter.stories.pop(index)


def confirm_story_list(newsletter):
    """Confirm the curated story list as authoritative extraction truth.

    Existing story ids are PRESERVED (cross-run comparisons key on them); only
    missing or duplicate ids are repaired with a fresh unique id from
    ``_next_story_id``. Marks the newsletter reviewed.
    """
    seen = set()
    for story in newsletter.stories:
        if not story.story_id or story.story_id in seen:
            story.story_id = _next_story_id(newsletter)
        seen.add(story.story_id)
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


def toggle_newsletter_excluded(newsletter) -> bool:
    """Toggle whole-newsletter exclusion; returns the new state.

    An excluded newsletter is dropped from the labeling queue (unless
    ``--include-excluded``) and from eval runs entirely. ``reviewed`` is left
    untouched — it records that the story list is confirmed extraction truth,
    which stays valid if the newsletter is later restored.
    """
    newsletter.excluded = not newsletter.excluded
    return newsletter.excluded


def newsletter_exclude_status(newsletter) -> str:
    """Status line after ``X`` toggles newsletter-level exclusion."""
    if newsletter.excluded:
        return (
            "Newsletter excluded from the queue and eval runs "
            "(X to restore; relaunch with --include-excluded to see it again)."
        )
    return "Newsletter restored to the queue and eval runs."


# ---------------------------------------------------------------------------
# Pure helpers for the TUI (no UI dependency; fully testable)
# ---------------------------------------------------------------------------

_HELP_ITEMS = (
    "↑/↓:move", "PgUp/PgDn:page", "s/e:select", "Enter:make-story", "Space:seed",
    "[a]dd", "[E]dit", "[d]el", "[c]onfirm", "[l]abel", "[u]exclude", "[n]otes",
    "[z]undo", "[k]skip", "[X]exclude-nl", "Esc:back", "q:quit",
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
    return f"Replace {plural(n, 'story', 'stories')}{detail} with a fresh seed? y/N"


def seed_outcome_message(raw: str, story_count: int) -> str:
    """Status line after a seed: distinguishes NO_STORIES from a parse washout."""
    if story_count:
        return f"Seeded {plural(story_count, 'story', 'stories')}."
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
            f" {plural(unlabeled, 'story', 'stories')} still unlabeled"
            " — press l to score them."
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
    """One list-view row: reviewed/excluded flag, labeled/total, sender, subject.

    ``X`` (excluded) takes precedence over ``Y`` (reviewed) in the flag column
    so an excluded newsletter is unmistakable when queued via
    ``--include-excluded``.
    """
    r = "X" if newsletter.excluded else ("Y" if newsletter.reviewed else " ")
    labeled, total = label_progress(newsletter)
    sender = (newsletter.sender or "")[:24]
    return f"{r:<2} {f'{labeled}/{total}':<6} {sender:<24} {newsletter.subject}"


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

def select_label_newsletters(newsletters, *, unreviewed_only=False, include_excluded=False):
    """Newsletters to queue for labeling.

    Excluded newsletters are set aside and not queued, regardless of
    *unreviewed_only* — unless *include_excluded*, which queues them so they
    can be inspected and restored (``X`` toggles exclusion in the detail view).
    """
    result = list(newsletters)
    if not include_excluded:
        result = [n for n in result if not n.excluded]
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
        # Fail fast, before the app starts — otherwise the missing endpoint only
        # surfaces as a truncated error when Space is pressed deep in the TUI.
        raise SystemExit(
            "No LLM endpoint configured for Phase-A seeding: set NEWSLETTER_LLM_URL "
            "or CLOUD_LLM_URL, or run with --edit to curate without seeding."
        )
    nl_llm = config["newsletter"]["llm"]
    # Mirror daemon.run_daemon / newsletter_run.build_classifier defaults.
    client = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=nl_llm["model"],
        temperature=nl_llm.get("temperature", 0.0),
        max_tokens=nl_llm.get("max_tokens", 1024),
        timeout=nl_llm.get("timeout", 60),
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
# Pure rendering helpers (no UI; fully testable)
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
    if newsletter.excluded:
        add_header("Excluded: True — skipped by the queue and eval runs (X to restore)")
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
# Textual UI layer
# ---------------------------------------------------------------------------

_MAX_UNDO = 100  # bound the per-newsletter undo stack


class ConfirmScreen(BottomModal):
    """y/N confirmation; only y/Y confirms, any other key is No."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(self._message, markup=False)

    def on_key(self, event) -> None:
        event.stop()
        self.dismiss_once(event.key.lower() == "y")


class ScoreScreen(BottomModal):
    """The 4 dimension scores, one keypress each; any non-1-5 key cancels all.

    When re-labeling, *current* shows the value already assigned per dimension;
    accepted digits are echoed in the prompt so entry is verifiable.
    """

    def __init__(self, current=None) -> None:
        super().__init__()
        self._current = current
        self._scores: dict[str, int] = {}

    def _prompt_text(self) -> str:
        dim = _DIMENSIONS[len(self._scores)]
        now = f" (now {self._current[dim]})" if self._current and dim in self._current else ""
        entered = "/".join(str(self._scores[d]) for d in _DIMENSIONS if d in self._scores)
        so_far = f"[{entered}] " if entered else ""
        return f"{so_far}{dim}{now} [1-5] (other cancels): "

    def compose(self) -> ComposeResult:
        yield Static(self._prompt_text(), id="score-prompt", markup=False)

    def on_key(self, event) -> None:
        event.stop()
        if self._dismissed:
            return  # key queued behind the dismissal (auto-repeat)
        if event.key not in ("1", "2", "3", "4", "5"):
            self.dismiss_once(None)
            return
        self._scores[_DIMENSIONS[len(self._scores)]] = int(event.key)
        if len(self._scores) == len(_DIMENSIONS):
            self.dismiss_once(dict(self._scores))
        else:
            self.query_one("#score-prompt", Static).update(self._prompt_text())


class ThemeScreen(BottomModal):
    """Multi-select themes by toggling s/c/h/v/d; Enter finishes (no cancel).

    Starts from *initial* (the story's current themes) so re-labeling edits
    rather than restarts; toggles preserve insertion order.
    """

    def __init__(self, initial=None) -> None:
        super().__init__()
        self._selected: list[str] = list(initial or [])

    def _legend(self) -> str:
        return format_theme_legend(self._selected, max(10, self.app.size.width - 1))

    def compose(self) -> ComposeResult:
        yield Static(self._legend(), id="theme-legend", markup=False)

    def on_key(self, event) -> None:
        event.stop()
        if self._dismissed:
            return  # key queued behind the dismissal (auto-repeat)
        if event.key == "enter":
            self.dismiss_once(list(self._selected))
            return
        theme = _THEME_KEYS.get(event.key.lower())
        if theme is not None:
            if theme in self._selected:
                self._selected.remove(theme)
            else:
                self._selected.append(theme)
            self.query_one("#theme-legend", Static).update(self._legend())


class DetailScreen(Screen):
    """Curate + label one newsletter. Dismisses with "back", "skip", or "quit"."""

    BINDINGS = [
        Binding("escape", "esc", "Back", show=False),
        Binding("q", "quit_app", "Quit", show=False),
        Binding("space", "seed", "Seed", show=False),
        Binding("s", "sel_start", "Selection start", show=False),
        Binding("e", "sel_end", "Selection end", show=False),
        Binding("a", "add_story", "Add story", show=False),
        Binding("E", "edit_story", "Edit story", show=False),
        Binding("d", "delete_story", "Delete story", show=False),
        Binding("c", "confirm_list", "Confirm list", show=False),
        Binding("l", "label_story", "Label story", show=False),
        Binding("u", "toggle_story_excluded", "Exclude story", show=False),
        Binding("n", "notes", "Notes", show=False),
        Binding("z", "undo", "Undo", show=False),
        Binding("X", "toggle_newsletter_excluded", "Exclude newsletter", show=False),
        Binding("k", "skip", "Skip", show=False),
    ]

    DEFAULT_CSS = """
    DetailScreen #rows {
        height: 1fr;
    }
    DetailScreen #rows .covered {
        color: $text-disabled;
    }
    DetailScreen #rows .selected {
        text-style: bold;
    }
    DetailScreen #help {
        color: $text-muted;
    }
    """

    def __init__(self, newsletters, index, all_newsletters, path, *, extract_fn=None):
        super().__init__()
        self.newsletters = newsletters
        self.nl_index = index
        self.newsletter = newsletters[index]
        self.all_newsletters = all_newsletters
        self.path = path
        self.extract_fn = extract_fn
        self.sel_start = None  # selected body-line index
        self.sel_end = None
        self.undo_stack: list[dict] = []  # snapshots, pushed only on real mutations
        self.anchor_body = None  # body line to re-anchor the cursor to after a mutation
        self._rows: list[tuple[str, int | None]] | None = None
        self._covered: set[int] = set()
        self._status_msg = ""
        self._busy = False  # a seed is in flight; mutating keys are ignored
        self._last_size: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield PageListView(id="rows")
        yield Static(id="help", markup=False)
        yield Static(id="status", markup=False)

    def on_mount(self) -> None:
        self._refresh(rebuild=True)
        self.query_one("#rows", ListView).focus()
        self._set_status("")

    def on_resize(self, event) -> None:
        size = (event.size.width, event.size.height)
        prev, self._last_size = self._last_size, size
        if prev is not None and prev != size and self._rows is not None:
            # Rewrap to the new width so no body content is lost to clipping.
            self._refresh(rebuild=True)

    # -- rendering ----------------------------------------------------------

    def _width(self) -> int:
        return max(20, self.app.size.width)

    def _refresh(self, rebuild: bool = False) -> None:
        if rebuild or self._rows is None:
            # Rows + covered-lines are expensive (display-width wrap of the
            # whole body, substring scan per line); rebuild only on mutation
            # or resize. Module-global lookups on purpose (test seam).
            self._rows = build_detail_rows(
                self.newsletter, self.nl_index, len(self.newsletters), self._width() - 2
            )
            self._covered = covered_body_lines(self.newsletter)

        lo = hi = None
        if self.sel_start is not None and self.sel_end is not None:
            lo, hi = min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)
        elif self.sel_start is not None:
            lo = hi = self.sel_start

        listview = self.query_one("#rows", ListView)
        if self.anchor_body is not None:
            # The story list above the body grew/shrank: keep the cursor on
            # the same BODY line, not the same physical row.
            cursor = row_for_body_line(self._rows, self.anchor_body)
            self.anchor_body = None
        else:
            cursor = listview.index or 0
        listview.clear()
        items = []
        for text, body_idx in self._rows:
            selected = body_idx is not None and lo is not None and lo <= body_idx <= hi
            # Column 0 is reserved for the selection marker on EVERY row, so
            # text never shifts when a selection appears.
            marker = "*" if selected else " "
            label = Label(marker + (text or " "), markup=False)
            if selected:
                label.add_class("selected")
            elif body_idx is not None and body_idx in self._covered:
                # Dim body lines already captured by a story, so paragraphs
                # the seed OMITTED stand out at normal brightness.
                label.add_class("covered")
            items.append(ListItem(label))
        listview.extend(items)
        if self._rows:
            listview.index = max(0, min(cursor, len(self._rows) - 1))
        self.query_one("#help", Static).update("\n".join(format_help_lines(self._width() - 1)))

    def _set_status(self, msg: str) -> None:
        self._status_msg = msg
        if msg:
            text = msg
        else:
            listview = self.query_one("#rows", ListView)
            text = f"row {(listview.index or 0) + 1}/{len(self._rows or [])}"
        self.query_one("#status", Static).update(text)

    def on_list_view_highlighted(self, event) -> None:
        if not self._status_msg:
            self._set_status("")  # keep the row counter current

    def on_page_list_view_user_navigated(self, event) -> None:
        # One-shot statuses clear on the next navigation keypress, restoring
        # the "row N/M" position counter (parity with the curses status line).
        self._set_status("")

    # -- persistence / undo ---------------------------------------------------

    def _save(self) -> None:
        try:
            save_golden_set(self.all_newsletters, self.path)
        except Exception as exc:
            # Non-fatal: the in-memory state survives, the next mutation retries.
            self.app.push_screen(HintScreen(f"Save failed: {exc}"))
        self._refresh(rebuild=True)

    def _push_undo(self) -> None:
        self.undo_stack.append(capture_snapshot(self.newsletter))
        del self.undo_stack[:-_MAX_UNDO]

    def _begin_flow(self) -> bool:
        """Claim the single mutation-flow slot.

        Two key events processed back-to-back (auto-repeat / mashing) each
        spawn a worker; the check-and-set runs before the worker's first
        await, so the second worker bails instead of stacking a second
        prompt over the first.
        """
        if self._busy:
            return False
        self._busy = True
        return True

    def _end_flow(self) -> None:
        self._busy = False

    def _cursor_body_idx(self):
        listview = self.query_one("#rows", ListView)
        if not self._rows or listview.index is None:
            return None
        return self._rows[listview.index][1]

    async def _prompt_story_index(self):
        raw = await self.app.push_screen_wait(PromptLineScreen("Story #:"))
        if raw is None or not raw.isdigit():
            return None
        si = int(raw)
        if 0 <= si < len(self.newsletter.stories):
            return si
        return None

    # -- navigation-ish actions ------------------------------------------------

    def action_esc(self) -> None:
        if self.sel_start is not None or self.sel_end is not None:
            self.sel_start = self.sel_end = None
            self._refresh()
            self._set_status("Selection cleared.")
        else:
            self.dismiss("back")

    def action_skip(self) -> None:
        self.dismiss("skip")  # never marks reviewed; list opens the next one

    def action_quit_app(self) -> None:
        self.dismiss("quit")

    # -- selection ------------------------------------------------------------

    def action_sel_start(self) -> None:
        body_idx = self._cursor_body_idx()
        if body_idx is None:
            self._set_status("Move the cursor onto a body line first.")
        else:
            self.sel_start = body_idx
            self._refresh()
            self._set_status(f"Selection start = body line {body_idx} (Esc clears).")

    def action_sel_end(self) -> None:
        body_idx = self._cursor_body_idx()
        if body_idx is None:
            self._set_status("Move the cursor onto a body line first.")
        else:
            self.sel_end = body_idx
            self._refresh()
            self._set_status(f"Selection end = body line {body_idx} (Esc clears).")

    def on_list_view_selected(self, event) -> None:
        event.stop()  # don't bubble to LabelApp's open-detail handler
        self._make_story()

    @work
    async def _make_story(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_make_story()
        finally:
            self._end_flow()

    async def _run_make_story(self) -> None:
        if self.sel_start is None:
            self._set_status("Set a selection start with 's' first.")
            return
        s_line = self.sel_start
        e_line = self.sel_end if self.sel_end is not None else self.sel_start
        title = await self.app.push_screen_wait(PromptLineScreen("Title (blank=auto):"))
        if title is None:
            self._set_status("Story creation cancelled.")
            return
        self.anchor_body = self._cursor_body_idx()
        self._push_undo()
        story = create_story_from_body(self.newsletter, s_line, e_line, title)
        self.sel_start = self.sel_end = None
        self._save()
        self._set_status(f"Story created: {story.title}")

    # -- seeding ---------------------------------------------------------------

    def action_seed(self) -> None:
        self._seed()

    @work
    async def _seed(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_seed()
        finally:
            self._end_flow()

    async def _run_seed(self) -> None:
        if self.extract_fn is None:
            self._set_status("Seeding disabled in edit mode (run without --edit to seed).")
            return
        confirm_msg = seed_confirmation_message(self.newsletter)
        if confirm_msg is not None and not await self.app.push_screen_wait(
            ConfirmScreen(confirm_msg)
        ):
            self._set_status("Seed cancelled — stories kept.")
            return
        self._set_status("Seeding…")
        try:
            # Only the network call runs in the thread; the newsletter is
            # mutated on the UI task AFTER the await, so dismissing the
            # screen mid-seed cancels this worker and the late extractor
            # result is discarded instead of clobbering curated stories.
            raw = await asyncio.to_thread(self.extract_fn, self.newsletter.body)
            pairs = parse_stories(raw)
        except Exception as exc:
            # No undo snapshot was pushed, so the stack stays clean.
            self._set_status("")
            self.app.push_screen(HintScreen(f"Seed failed: {exc}"))
        else:
            self._push_undo()
            stories = seed_stories(self.newsletter, pairs)
            self._save()
            self._set_status(seed_outcome_message(raw, len(stories)))

    # -- story curation ----------------------------------------------------------

    def action_add_story(self) -> None:
        self._add_story()

    @work
    async def _add_story(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_add_story()
        finally:
            self._end_flow()

    async def _run_add_story(self) -> None:
        title = await self.app.push_screen_wait(PromptLineScreen("New title:"))
        text = None
        if title:
            text = await self.app.push_screen_wait(PromptLineScreen("New text:"))
        if title and text:
            self._push_undo()
            add_story(self.newsletter, title, text)
            self._save()
            self._set_status(f"Story added: {title}")
        else:
            self._set_status("Add cancelled — need a title and text.")

    def action_edit_story(self) -> None:
        self._edit_story()

    @work
    async def _edit_story(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_edit_story()
        finally:
            self._end_flow()

    async def _run_edit_story(self) -> None:
        si = await self._prompt_story_index()
        if si is None:
            self._set_status("No valid story # — nothing edited.")
            return
        story = self.newsletter.stories[si]
        new_title = await self.app.push_screen_wait(
            PromptLineScreen("Title (blank=keep):", initial=story.title)
        )
        if new_title is None:
            self._set_status("Edit cancelled.")
            return
        # Multi-line text can't be edited in a one-line prompt; fall back to
        # blank=keep for it (body selection is the intended repair path).
        text_initial = story.text if "\n" not in story.text else ""
        new_text = await self.app.push_screen_wait(
            PromptLineScreen("Text (blank=keep):", initial=text_initial)
        )
        if new_text is None:
            self._set_status("Edit cancelled.")
            return
        new_title = new_title or story.title
        new_text = new_text or story.text
        if new_title == story.title and new_text == story.text:
            self._set_status("No changes.")
            return
        self._push_undo()
        edit_story(self.newsletter, si, title=new_title, text=new_text)
        self._save()
        self._set_status(f"Story [{si}] updated.")

    def action_delete_story(self) -> None:
        self._delete_story()

    @work
    async def _delete_story(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_delete_story()
        finally:
            self._end_flow()

    async def _run_delete_story(self) -> None:
        si = await self._prompt_story_index()
        if si is None:
            self._set_status("No valid story # — nothing deleted.")
            return
        story = self.newsletter.stories[si]
        if story.expected_scores is not None and not await self.app.push_screen_wait(
            ConfirmScreen(f"[{si}] {story.title} has labels — delete anyway? y/N")
        ):
            self._set_status("Delete cancelled.")
            return
        self._push_undo()
        delete_story(self.newsletter, si)
        self._save()
        self._set_status(f"Deleted [{si}] {story.title} (z to undo).")

    def action_confirm_list(self) -> None:
        if self._busy:
            return
        self._push_undo()
        confirm_story_list(self.newsletter)
        self._save()
        self._set_status(confirm_status_message(self.newsletter))

    # -- Phase B: labeling ------------------------------------------------------

    def action_label_story(self) -> None:
        self._label_story()

    @work
    async def _label_story(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_label_story()
        finally:
            self._end_flow()

    async def _run_label_story(self) -> None:
        si = await self._prompt_story_index()
        if si is None:
            self._set_status("No valid story # — nothing labeled.")
            return
        story = self.newsletter.stories[si]
        scores = await self.app.push_screen_wait(ScoreScreen(current=story.expected_scores))
        if scores is None:
            self._set_status("Score entry cancelled — no changes saved.")
            return
        themes = await self.app.push_screen_wait(ThemeScreen(initial=story.expected_themes))
        self._push_undo()
        assign_scores_and_themes(story, scores, themes)
        self._save()
        self._set_status(f"Labeled [{si}] {story.expected_tier}.")

    def action_toggle_story_excluded(self) -> None:
        self._toggle_story_excluded()

    @work
    async def _toggle_story_excluded(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_toggle_story_excluded()
        finally:
            self._end_flow()

    async def _run_toggle_story_excluded(self) -> None:
        si = await self._prompt_story_index()
        if si is None:
            self._set_status("No valid story # — nothing toggled.")
            return
        self._push_undo()
        story = self.newsletter.stories[si]
        story.excluded = not story.excluded
        self._save()
        self._set_status(
            f"[{si}] {'excluded from' if story.excluded else 'included in'} scoring."
        )

    def action_notes(self) -> None:
        self._notes()

    @work
    async def _notes(self) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_notes()
        finally:
            self._end_flow()

    async def _run_notes(self) -> None:
        new_notes = await self.app.push_screen_wait(
            PromptLineScreen("Notes:", initial=self.newsletter.notes)
        )
        if new_notes is None:
            self._set_status("Notes edit cancelled.")
        elif new_notes == self.newsletter.notes:
            self._set_status("Notes unchanged.")
        else:
            self._push_undo()
            self.newsletter.notes = new_notes
            self._save()
            self._set_status("Notes updated.")

    # -- undo / exclusion ---------------------------------------------------------

    def action_undo(self) -> None:
        if self._busy:
            return
        if self.undo_stack:
            restore_snapshot(self.newsletter, self.undo_stack.pop())
            self._save()
            self._set_status(f"Undone ({len(self.undo_stack)} more undo levels).")
        else:
            self._set_status("Nothing to undo.")

    def action_toggle_newsletter_excluded(self) -> None:
        if self._busy:
            return
        self._push_undo()
        toggle_newsletter_excluded(self.newsletter)
        self._save()
        self._set_status(newsletter_exclude_status(self.newsletter))


class LabelApp(App):
    """Newsletter labeling: list of newsletters, drill into curate+label detail."""

    BINDINGS = [Binding("q", "quit_app", "Quit")]

    DEFAULT_CSS = """
    LabelApp #newsletters {
        height: 1fr;
    }
    LabelApp #list-header {
        text-style: underline;
    }
    LabelApp #list-help {
        color: $text-muted;
    }
    """

    def __init__(self, newsletters, all_newsletters, path, *, extract_fn=None):
        super().__init__()
        self.newsletters = newsletters
        self.all_newsletters = all_newsletters
        self.path = Path(path)
        self.extract_fn = extract_fn

    def compose(self) -> ComposeResult:
        yield Static(
            f"Newsletter Label — {len(self.newsletters)} newsletters",
            id="list-title", markup=False,
        )
        yield Static(format_list_header(), id="list-header", markup=False)
        yield PageListView(id="newsletters")
        yield Static("↑/↓ PgUp/PgDn:Nav  Enter:Open  q:Quit", id="list-help", markup=False)

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        listview = self.query_one("#newsletters", ListView)
        cursor = listview.index or 0
        listview.clear()
        listview.extend(
            ListItem(Label(format_list_row(n), markup=False)) for n in self.newsletters
        )
        if self.newsletters:
            listview.index = min(cursor, len(self.newsletters) - 1)

    def on_list_view_selected(self, event) -> None:
        event.stop()
        index = self.query_one("#newsletters", ListView).index
        if index is not None:
            self._open_detail(index)

    def _open_detail(self, index: int) -> None:
        if len(self.screen_stack) > 1:
            return  # a detail/modal is already up (Enter auto-repeat)

        def on_dismiss(result) -> None:
            if result == "quit":
                self.exit("quit")
                return
            if result == "skip" and index < len(self.newsletters) - 1:
                # Linear skip-through: advance and immediately open the next
                # newsletter. Skipping never marks reviewed, so it resurfaces.
                self.query_one("#newsletters", ListView).index = index + 1
                self._open_detail(index + 1)
                return
            # "back", or "skip" on the last newsletter -> back to the list;
            # rebuild rows so reviewed/excluded flags are current.
            self._refresh_list()

        self.push_screen(
            DetailScreen(
                self.newsletters, index, self.all_newsletters, self.path,
                extract_fn=self.extract_fn,
            ),
            on_dismiss,
        )

    def action_quit_app(self) -> None:
        self.exit("quit")


def label_loop(newsletters, all_newsletters, path, *, extract_fn=None):
    """Launch the labeling TUI. Saving is atomic + auto on each edit."""
    if not newsletters:
        print("No newsletters to label.")
        return
    LabelApp(newsletters, all_newsletters, Path(path), extract_fn=extract_fn).run()



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
    parser.add_argument("--include-excluded", action="store_true",
                        help="Also queue excluded newsletters (to inspect or restore "
                             "them with the X hotkey; default: excluded are skipped)")
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

    newsletters = select_label_newsletters(
        all_newsletters,
        unreviewed_only=args.unreviewed_only,
        include_excluded=args.include_excluded,
    )
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
