# `newsletter_label` story-refinement UX redesign

Design record for the redesign of the `evals.newsletter_label` TUI (issue #43).
This documents *why* the tool works the way it does now; the operational
reference (flags, hotkeys, behavior) lives in
[`evals/README-technical.md`](../evals/README-technical.md) under
`### newsletter_label`.

## Context

A feel-check on issue #43 found the original story-refinement UX "essentially
unusable": the reviewer could not form a mental model of it and was continually
surprised by how it responded to keypresses. The root causes were:

- **Invisible modal state** — the same keys did different things depending on
  hidden state, with no on-screen indication of what mode you were in.
- **No story-in-context view** — the model's extracted excerpts and the full
  newsletter were not visible together, so you couldn't judge story boundaries.
- **No boundary editing** — a story could only be re-typed via one-line text
  prompts, not adjusted relative to its existing span.
- **No clear-all, no save-and-advance** — common flows had no direct key.

The redesign maps the UI onto the reviewer's actual thought process (narrated in
the appendix): open a newsletter, see the model's stories already highlighted in
the body, agree or adjust their boundaries, and move on.

## Two decisions the user made

1. **Auto-seed on open.** When an unreviewed, story-less newsletter is opened,
   the tool automatically runs the production extractor (a live LLM call with the
   current prompt) so the model's stories are visible without a keypress. This
   satisfies "use the daemon's initial assessment as a starting point" — via a
   fresh extraction, not by reading the daemon's stored assessments.

2. **Remove story titles entirely.** Titles were model-invented headlines that
   were fed into the quality/theme scoring prompts. The user ruled that a
   headline should never influence scoring. Titles are removed **end-to-end**:
   the extraction prompt now emits `STORY:` blocks (was `TITLE:`/`TEXT:`), the
   quality/theme prompts take story text only, and the `title` field is gone from
   `StoryResult`, `GoldenStory`, both TUIs, and the report. Stories are identified
   everywhere by a first-words text excerpt. Accepted consequence: the
   extraction/quality/theme `prompt_hash` all changed, so runs recorded before
   the change are no longer prompt-comparable — a deliberate clean break. Old
   golden-set / assessment JSONL files that still carry a `title` key load fine
   (the extra key is ignored).

## The redesigned UX

**The detail screen is the newsletter body, with stories shown in place.** A
compact metadata header, then a **story strip** (one line per story: number,
located line range or `⚠ not found`, scores, excerpt), then the body in a
`PageListView` where each located story span is tinted with a gutter bar and a
`▶ Story N` marker row at its first line, then a **mode bar** that always names
the active mode and its keys, then the status line. Excerpt and context are the
same pixels.

A story's line span is **derived at runtime** by matching its text against the
body (`locate_story_span`); the golden set stores only the story text, never line
numbers. A story whose text can't be located (e.g. a hand-mangled seed) is flagged
`⚠ not found` and stays editable via a text-prompt fallback.

**Two explicit modes, always named in the mode bar** (the fix for invisible
state):

- **Browse mode** (default): `n`/`p` or a number key selects a story; `Enter` on
  a story's body row selects it. `a` starts a new story · `e` edits the selected
  story's boundaries · `d` deletes it (confirm if labeled) · `C` clears all
  stories (confirm, undoable) · `r` re-seeds (confirm if non-empty) · `l` labels
  the selected story · `u` toggles its exclusion · `c` **accepts** the story list
  (marks reviewed, warns first if any story is unlabeled) and **advances to the
  next newsletter** · `k` skips · `z` undo · `N` notes · `X` excludes the whole
  newsletter · `Esc` back · `q` quit.
- **Span mode** (entered by `a` for a new story, or `e` to re-bound the selected
  one): the working span live-highlights as the cursor moves. A new story is two
  presses of `Enter` — mark the first line, then the last line. `s`/`e`
  fine-adjust the start/end; `Enter` commits; `Esc` cancels. A committed story's
  text is the **verbatim inclusive body slice**; editing preserves the story's
  `story_id`, labels, themes, notes, and excluded flag.

## Implementation notes

- **No schema change.** `GoldenStory.text` remains the single source of truth;
  spans are recomputed each render. Persisting spans was rejected — it would be a
  second source of truth the eval pipeline never reads, and would force a
  migration.
- **Rendering.** The single `PageListView` row model is preserved. Rows are
  `DetailRow(text, body_idx, story_idx, kind)`; `build_detail_rows(..., spans=)`
  emits the header, `▶ Story N` marker rows, and story-tagged body rows.
  Selection and span highlighting are applied at label-construction time, so
  changing the selection or the working span re-styles cached rows **without a
  rebuild** — the row cache only rebuilds on a mutation or a resize.
- **Pure core.** The state transitions are pure functions (`locate_story_span`,
  `story_at_body_line`, `SpanEdit` + `span_mark`/`span_cursor_moved`/
  `span_set_start`/`span_set_end`, `commit_span_edit`, `format_story_strip`,
  `browse_mode_bar`/`span_mode_bar`, `accept_confirmation_message`), unit-tested
  without a terminal. The Textual layer is driven by Pilot tests.
- **Auto-seed safety.** The extraction runs in a worker; the newsletter is
  mutated only after the network await, so leaving the screen mid-seed discards
  the result, and `z` restores the pre-seed (empty) list.

## Verification

- Full test suite green (`uv run --extra dev pytest tests/`) and `ruff check .`
  clean.
- End-to-end smoke test driving the real TUI in a pseudo-terminal against a temp
  golden set with a fake extractor: auto-seed → highlighted spans + strip →
  `e` extend a story → `c` advance → `C` clear-all → two-press single-story
  creation → `c` → quit, confirming the on-disk JSONL reflects it.
- Each beat of the reviewer's narrated thought process (below) checked against
  the final UI.

## Appendix — the reviewer's narrated thought process (verbatim)

> - "OK, the daemon has processed a bunch of newsletters and I'd like to use the daemon's initial assessment of them as a starting point for adding more newsletters to my golden set."
> - (User opens the `newsletter_label` TUI.)
> - "OK, good, there's a bunch of emails here that have been processed. Let's look at the first one."
> - (User drills down into the first email on the list.)
> - "OK, this is a newsletter from my co-worker Joe. I see that the classifying model identified two stories. Let's see if I agree with the model about the boundaries of those two stories."
> - (At this point, the user needs to see two things clearly in order to compare them: the excerpts that the model identified as stories, and how those excerpts fit within the context of the full newsletter. Ideally both are visible simultaneously.)
> - (The user compares the pre-selected stories to the full context of the letter.)
> - "OK, story 1 looks good. But story 2 cut off too early. The following paragraph belongs with it. I need to fix that so that story 2 includes that additional paragraph."
> - (The user selects story 2 and indicates he wants to modify what text is included within its boundaries. The UI provides a way to start from the existing boundaries, then shift the end point so as to include the following paragraph.)
> - "OK, that looks good now. Let's go on to the next newsletter."
> - (The user hits a key to save the first newsletter and show the next.)
> - "OK, I see it found two stories for this one, too. But this time it was right. Let's go on to the next newsletter."
> - (The user hits a key to save that newsletter and show the next.)
> - "OK, wow, it found five stories. Let me look that over."
> - (The user reads the newsletter in its original form. This needs to be visible easily.)
> - "All right, I see why it divided these but these five segments really should be treated as one story. I'm going to clear out the stories and start from scratch."
> - (The user presses a key to clear the stories. After a confirmation gate, the application clears them so that no stories are defined.)
> - "OK, now let's select the beginning of the overall story."
> - (The user selects the line that begins the story.)
> - "And now the end."
> - (The user selects the line that ends the story. The application now shows one story.)
> - "OK, good, now it shows as one story. Let's move along to the next newsletter."
>
> This is not an exhaustive thought process (for example, I didn't mention skipping or excluding emails). But hopefully it provides guidance.

Each beat maps to the final UI: open → **auto-seed** shows the model's stories in
context; agree → the story strip + in-body highlight; "story 2 cut off too early"
→ select story 2, `e`, extend the end, `Enter`; "go on to the next" → `c` (accept
& advance); "these five should be one story" → `C` (clear all), then `a` + mark
start + mark end for the single combined story; "move along" → `c`.
