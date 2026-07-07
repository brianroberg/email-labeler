"""Interactive CLI to build newsletter golden-set labels (two phases).

Usage:
    python -m evals.newsletter_label                     # curate + label
    python -m evals.newsletter_label --edit              # manual-only (no LLM seeding)
    python -m evals.newsletter_label --unreviewed-only

The detail screen is **body-centric**: it shows the newsletter body with each
story's span highlighted *in place*, so the extracted excerpts and their context
are visible together. A compact story strip above the body lists each story with
its located line range. Two explicit modes drive story refinement, and the mode
bar always names the active mode and its keys:

Phase A — curate stories / extraction truth.
  * **Auto-seed**: opening an unreviewed, story-less newsletter runs a fresh LLM
    extraction (``newsletter.parse_stories`` over the production extraction
    prompt) so the model's stories are already visible. Press ``r`` to re-seed.
  * **Browse mode** (default): ``n``/``p`` or a number key selects a story;
    ``a`` starts a new story, ``e`` edits the selected story's boundaries,
    ``d`` deletes it, ``C`` clears all stories, ``u`` excludes it. ``c`` accepts
    the story list (marks ``reviewed=True``) and advances to the next newsletter.
  * **Span mode** (via ``a``/``e``): pick a story's boundaries by marking the
    first line and the last line. A new story is two presses of ``Enter``
    (mark start, mark end); ``s``/``e`` fine-tune the start/end; ``Esc`` cancels.
    A committed story's text is the verbatim inclusive body slice.

Phase B — per-story labels: with a story selected, ``l`` assigns the 4 dimension
scores (simple/concrete/personal/dynamic, 1-5) and multi-select themes; on save
``expected_tier`` is derived via ``newsletter.compute_tier``.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

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
    ``parse_stories`` turns that into a list of story texts. Returns
    ``(stories, raw)`` so the caller can report the outcome (e.g. distinguish
    a NO_STORIES verdict from unparseable output — see ``seed_outcome_message``).
    """
    raw = extract_fn(newsletter.body)
    return seed_stories(newsletter, parse_stories(raw)), raw


def seed_stories(newsletter, story_texts):
    """Populate candidate stories from ``parse_stories`` story texts.

    Replaces any existing story list, assigns stable ids, and records the seed
    provenance. Replacing the list invalidates any prior confirmation, so
    ``newsletter.reviewed`` is reset to False — the uncurated machine seed must
    be re-confirmed before it counts as extraction truth (undo still restores
    the prior state, ``reviewed`` included, via the snapshot).
    """
    newsletter.stories = [
        GoldenStory(story_id=f"{newsletter.thread_id}:{i}", text=text)
        for i, text in enumerate(story_texts)
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


def add_story(newsletter, text):
    """Append a new candidate story with a stable, unique story_id.

    Does not confirm the list — ``newsletter.reviewed`` is untouched and the
    new story starts unreviewed.
    """
    story = GoldenStory(story_id=_next_story_id(newsletter), text=text)
    newsletter.stories.append(story)
    return story


def create_story_from_body(newsletter, start_line, end_line):
    """Build a candidate story from an inclusive span of body lines.

    *start_line* / *end_line* are indices into ``newsletter.body.splitlines()``;
    they are normalized (min/max) and clamped to the valid range, so order and
    out-of-range values are tolerated. The span text is the inclusive
    ``lo..hi`` slice joined with newlines. Appends via ``add_story`` (stable,
    unique ``story_id``); ``newsletter.reviewed`` is left untouched.
    """
    lo, hi = _clamp_span(newsletter, start_line, end_line)
    body_lines = newsletter.body.splitlines()
    text = "\n".join(body_lines[lo:hi + 1])
    return add_story(newsletter, text)


def _clamp_span(newsletter, start_line, end_line) -> tuple[int, int]:
    """Normalize + clamp an inclusive (start, end) body-line span.

    Both endpoints are clamped to ``[0, last]``, so a span that lies entirely
    past either end (e.g. ``(99, 99)``) collapses to the nearest boundary line
    rather than an empty slice — the clamp is monotonic, so ``lo <= hi`` holds.
    """
    body_lines = newsletter.body.splitlines()
    last = max(0, len(body_lines) - 1)
    lo = max(0, min(last, min(start_line, end_line)))
    hi = max(0, min(last, max(start_line, end_line)))
    return lo, hi


def edit_story(newsletter, index, *, text=None):
    """Edit a candidate's text span. An unspecified field is kept."""
    story = newsletter.stories[index]
    if text is not None:
        story.text = text


def delete_story(newsletter, index):
    """Remove a wrongly-extracted candidate entirely (not the same as exclude).

    Leaves ``newsletter.reviewed`` untouched; the remaining stories keep their
    ids (gaps are fine — ids are stable, never positional).
    """
    newsletter.stories.pop(index)


def clear_stories(newsletter):
    """Remove every story (start the story list over from scratch)."""
    newsletter.stories = []


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
# Span-locating (derive a story's body-line range from its text at runtime)
# ---------------------------------------------------------------------------

def _line_matches(body_stripped: str, story_stripped: str, *, fuzzy: bool) -> bool:
    """Whether a body line corresponds to a story line.

    Exact (stripped) equality — human-created stories are verbatim body slices,
    so they match exactly. With *fuzzy*, a whitespace-collapsed comparison also
    tolerates the minor internal-spacing variance an LLM seed may introduce,
    without the false positives a bare substring test would produce.
    """
    if body_stripped == story_stripped:
        return True
    if not fuzzy:
        return False
    return " ".join(body_stripped.split()) == " ".join(story_stripped.split())


def _locate_run(body_lines, story_lines, *, fuzzy: bool) -> tuple[int, int] | None:
    """First contiguous run of non-blank body lines matching *story_lines*."""
    n = len(body_lines)
    for start in range(n):
        if not body_lines[start].strip():
            continue
        si = 0
        bi = start
        last = None
        while bi < n and si < len(story_lines):
            bstr = body_lines[bi].strip()
            if not bstr:
                bi += 1
                continue
            if _line_matches(bstr, story_lines[si], fuzzy=fuzzy):
                last = bi
                si += 1
                bi += 1
            else:
                break
        if si == len(story_lines) and last is not None:
            return (start, last)
    return None


def locate_story_span(body_lines, story_text) -> tuple[int, int] | None:
    """Locate *story_text* as a contiguous run of body lines.

    The story's non-blank lines must match a run of non-blank body lines in
    order (blank body lines between them are skipped). Returns the inclusive
    ``(lo, hi)`` body-line range trimmed to the first/last matched non-blank
    line, or ``None`` when the text can't be located (e.g. a hand-edited story
    whose text no longer appears verbatim).

    An **exact** run is preferred over a whitespace-collapsed (fuzzy) one, so a
    story that is a verbatim body slice always locates at its true region even
    if an earlier region would fuzzy-match.
    """
    story_lines = [ln.strip() for ln in story_text.splitlines() if ln.strip()]
    if not story_lines:
        return None
    return (
        _locate_run(body_lines, story_lines, fuzzy=False)
        or _locate_run(body_lines, story_lines, fuzzy=True)
    )


def locate_story_spans(newsletter) -> list[tuple[int, int] | None]:
    """Locate every story's body-line span (see ``locate_story_span``).

    One entry per story, ``None`` where a story can't be located. Module-global
    so the row-cache path can be counted/patched in tests.
    """
    body_lines = newsletter.body.splitlines()
    return [locate_story_span(body_lines, s.text) for s in newsletter.stories]


def story_at_body_line(spans, body_idx) -> int | None:
    """Index of the first story whose located span contains *body_idx*."""
    for i, span in enumerate(spans):
        if span is not None and span[0] <= body_idx <= span[1]:
            return i
    return None


# ---------------------------------------------------------------------------
# Span-edit state machine (pure)
# ---------------------------------------------------------------------------

@dataclass
class SpanEdit:
    """Working state while marking/adjusting a story's body-line span.

    *story_index* is None for a brand-new story, else the story being re-bounded.
    *stage* is one of ``"mark-start"`` (pick the first line), ``"mark-end"``
    (pick the last line), or ``"adjust"`` (re-bounding an existing span).
    """

    story_index: int | None
    start: int
    end: int
    stage: str


def begin_add_span(cursor: int) -> SpanEdit:
    """Start a new-story span edit anchored at the cursor's body line."""
    return SpanEdit(story_index=None, start=cursor, end=cursor, stage="mark-start")


def begin_edit_span(spans, story_index) -> SpanEdit | None:
    """Start re-bounding an existing story, seeded with its located span.

    Returns None when the story can't be located in the body (nothing to seed).
    """
    span = spans[story_index] if 0 <= story_index < len(spans) else None
    if span is None:
        return None
    lo, hi = span
    return SpanEdit(story_index=story_index, start=lo, end=hi, stage="adjust")


def span_mark(edit: SpanEdit, cursor: int) -> tuple[SpanEdit, bool]:
    """Handle the Enter key in span mode; returns ``(new_edit, should_commit)``.

    In ``mark-start`` the cursor fixes the start and the edit advances to
    ``mark-end``. In ``mark-end`` the cursor fixes the end and the edit commits.
    In ``adjust`` the current start/end commit as-is.
    """
    if edit.stage == "mark-start":
        return replace(edit, start=cursor, end=cursor, stage="mark-end"), False
    if edit.stage == "mark-end":
        return replace(edit, end=cursor), True
    return edit, True  # adjust


def span_cursor_moved(edit: SpanEdit, cursor: int) -> SpanEdit:
    """Track cursor movement: both ends in ``mark-start``, else the end."""
    if edit.stage == "mark-start":
        return replace(edit, start=cursor, end=cursor)
    return replace(edit, end=cursor)


def span_set_start(edit: SpanEdit, cursor: int) -> SpanEdit:
    """Fine-adjust: set the span start to the cursor's body line."""
    return replace(edit, start=cursor)


def span_set_end(edit: SpanEdit, cursor: int) -> SpanEdit:
    """Fine-adjust: set the span end to the cursor's body line."""
    return replace(edit, end=cursor)


def span_range(edit: SpanEdit) -> tuple[int, int]:
    """The inclusive (lo, hi) highlight range for the current edit."""
    return min(edit.start, edit.end), max(edit.start, edit.end)


def commit_span_edit(newsletter, edit: SpanEdit) -> GoldenStory:
    """Commit a span edit to a story (new or re-bounded).

    A new story (``story_index is None``) is appended via ``create_story_from_body``.
    Re-bounding replaces the story's text with the verbatim inclusive body slice
    while PRESERVING its ``story_id``, labels, themes, notes, and excluded flag.
    """
    lo, hi = span_range(edit)
    if edit.story_index is None:
        return create_story_from_body(newsletter, lo, hi)
    lo, hi = _clamp_span(newsletter, lo, hi)
    body_lines = newsletter.body.splitlines()
    story = newsletter.stories[edit.story_index]
    story.text = "\n".join(body_lines[lo:hi + 1])
    return story


# ---------------------------------------------------------------------------
# Pure helpers for the TUI (no UI dependency; fully testable)
# ---------------------------------------------------------------------------

def _pack_bar(items, width: int) -> list[str]:
    """Pack *items* into lines of at most *width* chars (two-space separators)."""
    lines: list[str] = []
    current = ""
    for item in items:
        candidate = f"{current}  {item}" if current else item
        if current and len(candidate) > max(1, width):
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


_BROWSE_KEYS = (
    "n/p:pick-story", "1-9:pick", "[a]dd", "[e]dit-bounds", "[d]el", "[l]abel",
    "[u]excl-story", "[C]lear-all", "[r]eseed", "[c]:accept+next", "[k]skip",
    "[z]undo", "[N]otes", "[X]excl-nl", "Esc:back", "q:quit",
)


def browse_mode_bar(newsletter, selected, width: int) -> list[str]:
    """Footer help for browse mode; names the mode + the selected story."""
    if selected is not None:
        head = f"[BROWSE] story {selected + 1}/{len(newsletter.stories)} selected"
    elif newsletter.stories:
        head = "[BROWSE] no story selected (n/p or a number to pick)"
    else:
        head = "[BROWSE] no stories yet"
    return _pack_bar([head, *_BROWSE_KEYS], width)


def span_mode_bar(edit, width: int) -> list[str]:
    """Footer help for span mode; names the current marking stage."""
    stage_text = {
        "mark-start": "move to the FIRST line, Enter marks start",
        "mark-end": "move to the LAST line, Enter commits",
        "adjust": "move/[s]et-start/[e]nd, Enter commits",
    }.get(getattr(edit, "stage", ""), "")
    head = f"[SPAN] {stage_text}" if stage_text else "[SPAN]"
    return _pack_bar(
        [head, "arrows:move", "s:set-start", "e:set-end", "Enter:mark/commit",
         "Esc:cancel"],
        width,
    )


def seed_confirmation_message(newsletter) -> str | None:
    """Confirmation to show before ``r`` re-seeds over existing stories.

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


def accept_confirmation_message(newsletter) -> str | None:
    """Confirmation before ``c`` accepts a newsletter with unlabeled stories.

    Returns None when every non-excluded story is labeled (accept silently);
    otherwise a y/N question naming how many stories are still unlabeled.
    """
    n = unlabeled_story_count(newsletter)
    if n == 0:
        return None
    return f"{plural(n, 'story', 'stories')} still unlabeled — accept & advance anyway? y/N"


def row_for_body_line(rows, body_idx) -> int:
    """First rendered-row index carrying *body_idx* (see ``build_detail_rows``).

    Used to re-anchor the cursor to the body line it was on after the rows
    above the body grow/shrink. Falls back to the last row (or 0).
    """
    for ri, row in enumerate(rows):
        if row.body_idx == body_idx:
            return ri
    return max(0, len(rows) - 1)


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


def story_excerpt(text: str, width: int) -> str:
    """First-words excerpt of a story's text, collapsed and truncated to *width*."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty)"
    if len(collapsed) <= width:
        return collapsed
    return collapsed[:max(1, width - 1)].rstrip() + "…"


def format_story_strip(newsletter, spans, selected, width: int) -> list[str]:
    """One strip line per story: marker, number, located range, flags, excerpt.

    A story that can't be located in the body is flagged ``⚠ not found``; a
    labeled story shows its dimension scores; an excluded story shows ``EXCL``.
    With no stories, a single hint line is returned so the strip is never blank.
    """
    if not newsletter.stories:
        return ["Stories: none — [a]dd a story or [r]eseed from the model"]
    lines = []
    for i, s in enumerate(newsletter.stories):
        marker = "▶" if i == selected else " "
        span = spans[i] if i < len(spans) else None
        loc = f"L{span[0] + 1}-{span[1] + 1}" if span else "⚠ not found"
        flags = []
        if s.expected_scores is not None:
            flags.append("/".join(str(s.expected_scores.get(d, "?")) for d in _DIMENSIONS))
        if s.excluded:
            flags.append("EXCL")
        flag_str = f" [{'  '.join(flags)}]" if flags else ""
        prefix = f"{marker}{i + 1}. {loc}{flag_str}  "
        excerpt = story_excerpt(s.text, max(10, width - len(prefix)))
        lines.append((prefix + excerpt)[:width])
    return lines


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
        # surfaces as a truncated error when a seed is triggered deep in the TUI.
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


class DetailRow(NamedTuple):
    """One rendered detail-view row.

    *body_idx* is the source body-line index (``None`` for header/marker rows).
    *story_idx* is the owning story's index when the row falls inside a located
    span (``None`` otherwise). *kind* is ``"header"``, ``"marker"``, or ``"body"``.
    """

    text: str
    body_idx: int | None
    story_idx: int | None
    kind: str


def build_detail_rows(newsletter, index, total, width, *, spans) -> list[DetailRow]:
    """Build the wrapped detail-view rows with body-line + story provenance.

    A compact metadata header precedes the body. Each located story span gets a
    ``▶ Story N`` marker row at its first body line, and every body row inside a
    span carries that story's index (first-story-wins on overlap) — the map the
    view uses to tint story regions. Each logical body line is wrapped to *width*.
    """
    rows: list[DetailRow] = []

    def add_header(text: str) -> None:
        for line in wrap_text(text, width):
            rows.append(DetailRow(line, None, None, "header"))

    add_header(f"Newsletter {index + 1}/{total}  (id: {newsletter.thread_id})")
    add_header("=" * max(1, min(60, width)))
    add_header(f"Subject:  {newsletter.subject}")
    add_header(f"Sender:   {newsletter.sender}")
    add_header(f"Reviewed: {newsletter.reviewed}")
    if newsletter.excluded:
        add_header("Excluded: True — skipped by the queue and eval runs (X to restore)")
    if newsletter.notes:
        add_header(f"Notes:    {newsletter.notes}")
    add_header("--- Body ---")

    # Map body-line -> span-start story and -> owning story (first-wins).
    starts: dict[int, int] = {}
    membership: dict[int, int] = {}
    for si, span in enumerate(spans):
        if span is None:
            continue
        lo, hi = span
        starts.setdefault(lo, si)
        for b in range(lo, hi + 1):
            membership.setdefault(b, si)

    for body_idx, body_line in enumerate(newsletter.body.splitlines()):
        if body_idx in starts:
            si = starts[body_idx]
            rows.append(DetailRow(f"▶ Story {si + 1}", None, si, "marker"))
        sidx = membership.get(body_idx)
        for physical in wrap_text(body_line, width):
            rows.append(DetailRow(physical, body_idx, sidx, "body"))
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
    """Curate + label one newsletter. Dismisses with "back", "skip", "accepted",
    or "quit"."""

    BINDINGS = [
        Binding("escape", "esc", "Back", show=False),
        Binding("q", "quit_app", "Quit", show=False),
        Binding("r", "seed", "Reseed", show=False),
        Binding("a", "add_story", "Add story", show=False),
        Binding("e", "e_key", "Edit bounds / set end", show=False),
        Binding("s", "s_key", "Set start", show=False),
        Binding("n", "select_next", "Next story", show=False),
        Binding("p", "select_prev", "Prev story", show=False),
        Binding("1", "select_number(1)", "Story 1", show=False),
        Binding("2", "select_number(2)", "Story 2", show=False),
        Binding("3", "select_number(3)", "Story 3", show=False),
        Binding("4", "select_number(4)", "Story 4", show=False),
        Binding("5", "select_number(5)", "Story 5", show=False),
        Binding("6", "select_number(6)", "Story 6", show=False),
        Binding("7", "select_number(7)", "Story 7", show=False),
        Binding("8", "select_number(8)", "Story 8", show=False),
        Binding("9", "select_number(9)", "Story 9", show=False),
        Binding("d", "delete_story", "Delete story", show=False),
        Binding("l", "label_story", "Label story", show=False),
        Binding("u", "toggle_story_excluded", "Exclude story", show=False),
        Binding("C", "clear_all", "Clear all stories", show=False),
        Binding("c", "accept", "Accept + next", show=False),
        Binding("N", "notes", "Notes", show=False),
        Binding("z", "undo", "Undo", show=False),
        Binding("X", "toggle_newsletter_excluded", "Exclude newsletter", show=False),
        Binding("k", "skip", "Skip", show=False),
    ]

    DEFAULT_CSS = """
    DetailScreen #storystrip {
        color: $text-muted;
    }
    DetailScreen #rows {
        height: 1fr;
    }
    DetailScreen #rows .story-a {
        color: $secondary;
    }
    DetailScreen #rows .story-b {
        color: $primary;
    }
    DetailScreen #rows .story-selected {
        text-style: bold;
        background: $boost;
    }
    DetailScreen #rows .span-working {
        text-style: bold;
        background: $accent;
    }
    DetailScreen #rows .marker {
        color: $text-muted;
        text-style: italic;
    }
    DetailScreen #modebar {
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
        self.mode = "browse"  # "browse" | "span"
        self.selected_story: int | None = None
        self.span_edit: SpanEdit | None = None
        self.undo_stack: list[dict] = []  # snapshots, pushed only on real mutations
        self.anchor_body = None  # body line to re-anchor the cursor to after a mutation
        self._rows: list[DetailRow] | None = None
        self._spans: list = []
        self._status_msg = ""
        self._busy = False  # a seed/flow is in flight; mutating keys are ignored
        self._dismiss_guard = False  # guards a second dismiss on an already-popped screen
        self._last_size: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="storystrip", markup=False)
        yield PageListView(id="rows")
        yield Static(id="modebar", markup=False)
        yield Static(id="status", markup=False)

    def on_mount(self) -> None:
        self._refresh(rebuild=True)
        self.query_one("#rows", ListView).focus()
        self._set_status("")
        # Auto-seed: opening an unreviewed, story-less newsletter runs a fresh
        # extraction so the model's stories are visible without a keypress.
        if self.extract_fn and not self.newsletter.stories and not self.newsletter.reviewed:
            self._seed(auto=True)

    def on_resize(self, event) -> None:
        size = (event.size.width, event.size.height)
        prev, self._last_size = self._last_size, size
        if prev is not None and prev != size and self._rows is not None:
            # Rewrap to the new width so no body content is lost to clipping.
            # Re-anchor by BODY line (not physical row) so a resize while marking
            # a span doesn't drag the in-progress boundary to whatever body line
            # the old physical row now maps to after the rewrap.
            if self.anchor_body is None:
                self.anchor_body = self._cursor_body_idx()
            self._refresh(rebuild=True)

    # -- rendering ----------------------------------------------------------

    def _width(self) -> int:
        return max(20, self.app.size.width)

    def _clamp_selection(self) -> None:
        if self.selected_story is not None and self.selected_story >= len(self.newsletter.stories):
            self.selected_story = (
                len(self.newsletter.stories) - 1 if self.newsletter.stories else None
            )

    def _refresh(self, rebuild: bool = False) -> None:
        self._clamp_selection()
        if rebuild or self._rows is None:
            # Rows + spans are expensive (display-width wrap of the whole body,
            # locate scan per story); rebuild only on mutation or resize.
            # Module-global lookups on purpose (test seam).
            self._spans = locate_story_spans(self.newsletter)
            self._rows = build_detail_rows(
                self.newsletter, self.nl_index, len(self.newsletters),
                self._width() - 2, spans=self._spans,
            )

        span_lo = span_hi = None
        if self.mode == "span" and self.span_edit is not None:
            span_lo, span_hi = span_range(self.span_edit)

        listview = self.query_one("#rows", ListView)
        if self.anchor_body is not None:
            # The rows above the body grew/shrank: keep the cursor on the same
            # BODY line, not the same physical row.
            cursor = row_for_body_line(self._rows, self.anchor_body)
            self.anchor_body = None
        else:
            cursor = listview.index or 0
        listview.clear()
        items = []
        for row in self._rows:
            in_span = (
                row.body_idx is not None and span_lo is not None
                and span_lo <= row.body_idx <= span_hi
            )
            gutter, cls = self._row_gutter_class(row, in_span)
            label = Label(gutter + (row.text or " "), markup=False)
            if cls:
                label.add_class(cls)
            items.append(ListItem(label))
        listview.extend(items)
        if self._rows:
            listview.index = max(0, min(cursor, len(self._rows) - 1))

        self.query_one("#storystrip", Static).update(
            "\n".join(format_story_strip(
                self.newsletter, self._spans, self.selected_story, self._width() - 1,
            ))
        )
        if self.mode == "span":
            bar = span_mode_bar(self.span_edit, self._width() - 1)
        else:
            bar = browse_mode_bar(self.newsletter, self.selected_story, self._width() - 1)
        self.query_one("#modebar", Static).update("\n".join(bar))

    def _row_gutter_class(self, row: DetailRow, in_span: bool) -> tuple[str, str]:
        """The (2-cell gutter, CSS class) for a rendered row.

        Column 0 is reserved on every row so text never shifts. The span-working
        highlight wins, then the selected story, then plain story membership.
        """
        if row.kind == "marker":
            return "▸ ", "marker"
        if in_span:
            return "┃ ", "span-working"
        if row.story_idx is not None:
            if row.story_idx == self.selected_story:
                return "┃ ", "story-selected"
            return "│ ", ("story-a" if row.story_idx % 2 == 0 else "story-b")
        return "  ", ""

    def _set_status(self, msg: str) -> None:
        self._status_msg = msg
        if msg:
            text = msg
        elif self.mode == "span" and self.span_edit is not None:
            text = self._span_status()
        else:
            listview = self.query_one("#rows", ListView)
            text = f"row {(listview.index or 0) + 1}/{len(self._rows or [])}"
        self.query_one("#status", Static).update(text)

    def _span_status(self) -> str:
        edit = self.span_edit
        lo, hi = span_range(edit)
        if edit.stage == "mark-start":
            return "SPAN: move to the FIRST line, Enter marks start (Esc cancels)."
        if edit.stage == "mark-end":
            return (
                f"SPAN: start=L{edit.start + 1}; move to the LAST line, Enter commits. "
                "s/e adjust, Esc cancels."
            )
        return (
            f"SPAN: lines L{lo + 1}-L{hi + 1}; Enter commits, s/e adjust, Esc cancels."
        )

    def on_list_view_highlighted(self, event) -> None:
        if self.mode == "span" and self.span_edit is not None:
            bi = self._cursor_body_idx()
            if bi is not None:
                moved = span_cursor_moved(self.span_edit, bi)
                if (moved.start, moved.end) != (self.span_edit.start, self.span_edit.end):
                    self.span_edit = moved
                    self._refresh()  # highlight update only (no rebuild)
            self._set_status("")
            return
        if not self._status_msg:
            self._set_status("")  # keep the row counter current

    def on_page_list_view_user_navigated(self, event) -> None:
        # One-shot statuses clear on the next navigation keypress, restoring the
        # "row N/M" position counter (parity with the curses status line).
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

        Two key events processed back-to-back (auto-repeat / mashing) each spawn
        a worker; the check-and-set runs before the worker's first await, so the
        second worker bails instead of stacking a second prompt over the first.
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
        return self._rows[listview.index].body_idx

    def _require_selected(self, what: str):
        """Return the selected story index, or None with a status hint."""
        if self.selected_story is None or self.selected_story >= len(self.newsletter.stories):
            self._set_status(f"Select a story first (n/p or a number) to {what}.")
            return None
        return self.selected_story

    # -- navigation-ish actions ------------------------------------------------

    def action_esc(self) -> None:
        if self.mode == "span":
            self.mode = "browse"
            self.span_edit = None
            self._refresh()
            self._set_status("Span edit cancelled.")
        else:
            self._dismiss_once("back")

    def _dismiss_once(self, result) -> None:
        """Dismiss the screen at most once.

        The mutation-flow ``_busy`` latch only serializes workers that *await*
        between claim and release; an await-free ``_accept`` (nothing to confirm)
        releases the latch before a second queued worker runs, so two mashed
        ``c`` presses could both reach ``dismiss`` — the second on an
        already-popped screen (a crash). This one-shot guard prevents that.
        (Named ``_dismiss_guard``, not ``_closing`` — Textual owns ``_closing``
        on the message pump, and clobbering it jams the screen's teardown.)
        """
        if not self._dismiss_guard:
            self._dismiss_guard = True
            self.dismiss(result)

    def action_skip(self) -> None:
        if self.mode == "span" or self._busy:
            return
        self._dismiss_once("skip")  # never marks reviewed; list opens the next one

    def action_quit_app(self) -> None:
        self._dismiss_once("quit")

    # -- story selection -------------------------------------------------------

    def _select_story(self, si: int) -> None:
        if not self.newsletter.stories:
            self._set_status("No stories — press a to add or r to seed.")
            return
        si %= len(self.newsletter.stories)
        self.selected_story = si
        span = self._spans[si] if si < len(self._spans) else None
        if span is not None:
            self.anchor_body = span[0]  # scroll the story into view
        self._refresh()
        self._set_status(f"Story {si + 1}/{len(self.newsletter.stories)} selected.")

    def action_select_next(self) -> None:
        if self._busy or self.mode == "span":
            return
        cur = self.selected_story if self.selected_story is not None else -1
        self._select_story(cur + 1)

    def action_select_prev(self) -> None:
        if self._busy or self.mode == "span":
            return
        cur = self.selected_story if self.selected_story is not None else 0
        self._select_story(cur - 1)

    def action_select_number(self, n: int) -> None:
        if self._busy or self.mode == "span":
            return
        if 1 <= n <= len(self.newsletter.stories):
            self._select_story(n - 1)
        else:
            self._set_status(f"No story {n}.")

    # -- span mode (mark / edit boundaries) -----------------------------------

    def _enter_span_new(self) -> None:
        cursor = self._cursor_body_idx()
        if cursor is None:
            cursor = self._first_body_line()
        if cursor is None:
            self._set_status("This newsletter has no body to select from.")
            return
        self.mode = "span"
        self.span_edit = begin_add_span(cursor)
        self._refresh()
        self._set_status(self._span_status())

    def _first_body_line(self):
        for row in self._rows or []:
            if row.body_idx is not None:
                return row.body_idx
        return None

    def action_add_story(self) -> None:
        if self._busy or self.mode == "span":
            return
        self._enter_span_new()

    def action_e_key(self) -> None:
        # In span mode: set the end to the cursor. In browse: edit the selected
        # story's boundaries (enter span mode seeded with its located span).
        if self.mode == "span":
            self._span_adjust(span_set_end)
            return
        if self._busy:
            return
        si = self._require_selected("edit its boundaries")
        if si is None:
            return
        edit = begin_edit_span(self._spans, si)
        if edit is None:
            self._edit_text_fallback(si)
            return
        self.mode = "span"
        self.span_edit = edit
        span = self._spans[si]
        self.anchor_body = span[1]  # cursor at the current end
        self._refresh()
        self._set_status(self._span_status())

    def action_s_key(self) -> None:
        # Only meaningful in span mode (set the start to the cursor).
        if self.mode == "span":
            self._span_adjust(span_set_start)

    def _span_adjust(self, fn) -> None:
        cursor = self._cursor_body_idx()
        if cursor is None:
            self._set_status("Move the cursor onto a body line first.")
            return
        self.span_edit = fn(self.span_edit, cursor)
        self._refresh()
        self._set_status(self._span_status())

    def on_list_view_selected(self, event) -> None:
        event.stop()  # don't bubble to LabelApp's open-detail handler
        if self.mode == "span":
            self._span_enter()
        else:
            self._select_under_cursor()

    def _span_enter(self) -> None:
        cursor = self._cursor_body_idx()
        if cursor is None:
            self._set_status("Move the cursor onto a body line first.")
            return
        self.span_edit, commit = span_mark(self.span_edit, cursor)
        if commit:
            self._commit_span()
        else:
            self._refresh()
            self._set_status(self._span_status())

    def _commit_span(self) -> None:
        edit = self.span_edit
        self.anchor_body = self._cursor_body_idx()
        self._push_undo()
        story = commit_span_edit(self.newsletter, edit)
        # Select the story we just created/edited so labeling can follow.
        try:
            self.selected_story = self.newsletter.stories.index(story)
        except ValueError:
            self.selected_story = None
        self.mode = "browse"
        self.span_edit = None
        self._save()
        excerpt = story_excerpt(story.text, 40)
        verb = "updated" if edit.story_index is not None else "created"
        self._set_status(f"Story {verb}: {excerpt}")

    def _select_under_cursor(self) -> None:
        cursor = self._cursor_body_idx()
        si = story_at_body_line(self._spans, cursor) if cursor is not None else None
        if si is None:
            self._set_status("No story on this line — press a to add one.")
            return
        self._select_story(si)

    def _edit_text_fallback(self, si: int) -> None:
        """Boundary edit is impossible (story not locatable) — edit text directly."""
        self._run_edit_text_fallback(si)

    @work
    async def _run_edit_text_fallback(self, si: int) -> None:
        if not self._begin_flow():
            return
        try:
            story = self.newsletter.stories[si]
            collapsed = " ".join(story.text.split())
            new_text = await self.app.push_screen_wait(
                PromptLineScreen(
                    "Text (story not found in body; edit as text):", initial=collapsed,
                )
            )
            if new_text is None or new_text == story.text:
                self._set_status("Edit cancelled.")
                return
            self._push_undo()
            edit_story(self.newsletter, si, text=new_text)
            self._save()
            self._set_status(f"Story [{si + 1}] text updated.")
        finally:
            self._end_flow()

    # -- seeding ---------------------------------------------------------------

    def action_seed(self) -> None:
        if self.mode == "span":
            return
        self._seed(auto=False)

    @work
    async def _seed(self, auto: bool = False) -> None:
        if not self._begin_flow():
            return
        try:
            await self._run_seed(auto=auto)
        finally:
            self._end_flow()

    async def _run_seed(self, auto: bool = False) -> None:
        if self.extract_fn is None:
            if not auto:
                self._set_status("Seeding disabled in edit mode (run without --edit to seed).")
            return
        if not auto:
            confirm_msg = seed_confirmation_message(self.newsletter)
            if confirm_msg is not None and not await self.app.push_screen_wait(
                ConfirmScreen(confirm_msg)
            ):
                self._set_status("Seed cancelled — stories kept.")
                return
        self._set_status("Extracting stories…")
        try:
            # Only the network call runs in the thread; the newsletter is mutated
            # on the UI task AFTER the await, so dismissing the screen mid-seed
            # cancels this worker and the late extractor result is discarded
            # instead of clobbering curated stories.
            raw = await asyncio.to_thread(self.extract_fn, self.newsletter.body)
            texts = parse_stories(raw)
        except Exception as exc:
            # No undo snapshot was pushed, so the stack stays clean.
            self._set_status("")
            self.app.push_screen(HintScreen(f"Seed failed: {exc}"))
        else:
            self._push_undo()
            self.selected_story = None
            stories = seed_stories(self.newsletter, texts)
            self._save()
            self._set_status(seed_outcome_message(raw, len(stories)))

    # -- story curation --------------------------------------------------------

    def action_delete_story(self) -> None:
        if self.mode == "span":
            return
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
        si = self._require_selected("delete it")
        if si is None:
            return
        story = self.newsletter.stories[si]
        excerpt = story_excerpt(story.text, 40)
        if story.expected_scores is not None and not await self.app.push_screen_wait(
            ConfirmScreen(f"[{si + 1}] {excerpt} has labels — delete anyway? y/N")
        ):
            self._set_status("Delete cancelled.")
            return
        self._push_undo()
        delete_story(self.newsletter, si)
        self.selected_story = None
        self._save()
        self._set_status(f"Deleted [{si + 1}] {excerpt} (z to undo).")

    def action_clear_all(self) -> None:
        if self.mode == "span":
            return
        self._clear_all()

    @work
    async def _clear_all(self) -> None:
        if not self._begin_flow():
            return
        try:
            if not self.newsletter.stories:
                self._set_status("No stories to clear.")
                return
            n = len(self.newsletter.stories)
            if not await self.app.push_screen_wait(
                ConfirmScreen(f"Clear all {plural(n, 'story', 'stories')}? y/N")
            ):
                self._set_status("Clear cancelled.")
                return
            self._push_undo()
            clear_stories(self.newsletter)
            self.selected_story = None
            self._save()
            self._set_status("All stories cleared (z to undo). Press a to add one.")
        finally:
            self._end_flow()

    def action_accept(self) -> None:
        if self.mode == "span":
            return
        self._accept()

    @work
    async def _accept(self) -> None:
        if not self._begin_flow():
            return
        try:
            msg = accept_confirmation_message(self.newsletter)
            if msg is not None and not await self.app.push_screen_wait(ConfirmScreen(msg)):
                self._set_status("Not accepted — press l to label the remaining stories.")
                return
            confirm_story_list(self.newsletter)
            self._save()
            self._dismiss_once("accepted")
        finally:
            self._end_flow()

    # -- Phase B: labeling -----------------------------------------------------

    def action_label_story(self) -> None:
        if self.mode == "span":
            return
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
        si = self._require_selected("label it")
        if si is None:
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
        self._set_status(f"Labeled [{si + 1}] {story.expected_tier}.")

    def action_toggle_story_excluded(self) -> None:
        if self.mode == "span" or self._busy:
            return
        si = self._require_selected("exclude it")
        if si is None:
            return
        self._push_undo()
        story = self.newsletter.stories[si]
        story.excluded = not story.excluded
        self._save()
        self._set_status(
            f"[{si + 1}] {'excluded from' if story.excluded else 'included in'} scoring."
        )

    def action_notes(self) -> None:
        if self.mode == "span":
            return
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

    # -- undo / exclusion ------------------------------------------------------

    def action_undo(self) -> None:
        if self._busy or self.mode == "span":
            return
        if self.undo_stack:
            restore_snapshot(self.newsletter, self.undo_stack.pop())
            self.selected_story = None
            self._save()
            self._set_status(f"Undone ({len(self.undo_stack)} more undo levels).")
        else:
            self._set_status("Nothing to undo.")

    def action_toggle_newsletter_excluded(self) -> None:
        if self._busy or self.mode == "span":
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
            if result in ("skip", "accepted") and index < len(self.newsletters) - 1:
                # Linear advance: open the next newsletter. "skip" never marks
                # reviewed; "accepted" already did (confirm_story_list).
                self.query_one("#newsletters", ListView).index = index + 1
                self._open_detail(index + 1)
                return
            # "back", or advancing off the last newsletter -> back to the list;
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
