---
name: tui-regression
description: >-
  Feature- and regression-test the four Textual TUIs (evals.newsletter_label,
  evals.review, evals.edit_tui, newsletter_review) by driving each end-to-end
  through its full workflow with Pilot over diverse synthetic data and a mocked
  model. Use this after changing any TUI, tui_common, the newsletter/thread
  golden-set schemas, or the extraction/parse path — or when asked to "drive the
  TUIs", "exercise the full workflow", "smoke/regression test the TUIs end to
  end", or "make sure the TUIs still work". Runnable harness lives beside this
  file; the process below explains how to extend it for a new feature and how to
  read failures. For quick manual one-offs in a real terminal, prefer the
  smoke-test-terminal-app skill; use THIS for repeatable, asserting coverage.
---

# TUI regression harness

Four Textual TUIs, driven end-to-end via the framework's own **Pilot** driver
(`app.run_test()` → real key events) over **diverse synthetic data** with the
**model mocked**. This is the deterministic, headless, CI-friendly way to
exercise a Textual app's full workflow — complementary to the unit-level Pilot
tests already in `tests/` (those check one behavior each; these run realistic
multi-step workflows over the full data-state matrix and assert outcomes).

## Layout (all beside this file, in `.claude/skills/tui-regression/`)

| File | Role |
|------|------|
| `synth_data.py` | Diverse `GoldenNewsletter` / `GoldenThread` / assessment-record builders + `write_all(dir)`. Spans empty, typical, and stress-edge (emoji/CJK, CRLF, unlocatable story, >9 stories, every tier/theme/label, excluded/reviewed/labeled mixes). |
| `mocks.py` | The model seam. **Only `newsletter_label` calls a model** (the injected `extract_fn(body)->raw "STORY:" text`). Fakes cover parsed / NO_STORIES / unparseable / hard-failure. The other three TUIs have no model. |
| `_e2e.py` | Shared harness: `SIZE`, `drain`, `Check`, `run_scenarios` (per-scenario timeout), `report`. Puts repo root + this dir on `sys.path`. |
| `drive_newsletter_label.py` | Curate+label workflow: auto-seed, browse, span add/edit/delete, clear, reseed(+confirm), unlocatable text-edit fallback, Phase-B scoring+themes, exclude, notes, undo, accept, skip. |
| `drive_review.py` | Golden-thread review: confirm/sender/label/notes/skip/exclude/undo, normal + blind modes, stage 1/2, scroll, quit, last-thread double-key. |
| `drive_edit_tui.py` | Golden-thread editor: nav/paging, drill-in/back, sender+label edits (+ l/l collision, cancel-no-save), unexclude, scroll, filtered-save-all, quit. |
| `drive_newsletter_review.py` | Read-only browser: nav/paging, drill+scroll, full filter matrix (tier/theme/sender), CANCEL-vs-clear, literal-"cancel" value, init filters, empty result. |
| `run_all.py` | Orchestrator: writes the data, runs all drivers, prints a combined report, exits non-zero on any failure. |

## Run it

The repo pins `requires-python >=3.14`, but the code runs fine on 3.11–3.13 with
`textual>=8.2.8`. Make a throwaway venv if `uv run` refuses the interpreter:

```bash
python3 -m venv /tmp/tuivenv && /tmp/tuivenv/bin/pip install -q \
  httpx python-dotenv fastapi jinja2 uvicorn python-multipart "textual>=8.2.8" \
  pytest pytest-asyncio pytest-subtests ruff
```

Then, from the repo root:

```bash
PYTHONPATH=.:.claude/skills/tui-regression \
  /tmp/tuivenv/bin/python .claude/skills/tui-regression/run_all.py
# or a single TUI:
PYTHONPATH=.:.claude/skills/tui-regression \
  /tmp/tuivenv/bin/python .claude/skills/tui-regression/drive_review.py
# --emit DIR to keep the generated JSONL for inspection:
#   ... run_all.py --emit /tmp/tui_data
```

Green = every scenario passed and the app never crashed. The generated data is
also loaded back through the real `load_golden_set` / `load_assessments` paths,
so a schema break shows up here too.

## The process (do this for a new feature or a regression check)

1. **Extend the synthetic data first.** Add a `GoldenNewsletter` / `GoldenThread`
   / record to `synth_data.py` that puts the app in the new state (or the edge
   case you're guarding). Keep it diverse — cover the degenerate and the
   stress-edge, not just the happy path.
2. **Mock the model at the seam, never the network.** Only `newsletter_label`
   has one: pass an `extract_fn` from `mocks.py` (or a `lambda body: "STORY: …"`)
   into `LabelApp(...)`. Never let a driver reach `build_extractor` (it dials a
   real LLM). The other TUIs need no mock.
3. **Add a scenario** to the relevant `drive_*.py`: an `async def s_...(chk)` that
   builds the app, `async with app.run_test(size=SIZE) as pilot:`, presses keys,
   and asserts via `chk.eq/that`. Register it in that module's `scenarios()`.
   Drive the workflow the way a user would — multi-step, then assert the
   persisted/in-memory outcome (golden-set fields, `app.return_value`, widget
   state), not just "didn't crash".
4. **Run the one driver, then `run_all.py`, then the unit suite**
   (`pytest tests/`). Keep all three green.
5. **When something fails, decide: driver bug or app bug.**
   - A **hang / `WorkerCancelled`** almost always means you called `drain()`
     (which waits for `@work` workers to *complete*) while a worker is
     intentionally suspended on a modal it just opened. Fix: `await pilot.pause()`
     to let the modal mount, interact with it (set `Input.value`, press the key),
     *then* `drain()`. `drain()` is only for after a flow fully completes.
   - A **real assertion failure** that reproduces the same way a user would hit
     it is an app bug: write it up as a failing case, fix red/green per the
     repo's TDD rule (a matching `tests/` unit test is the durable home for the
     regression), and confirm the driver goes green.

## Pilot gotchas that bite these specific TUIs

- **`drain` vs `pause`.** `drain()` = `workers.wait_for_complete()` + `pause()`;
  use it after a flow finishes. Use bare `pause()` after any key that *opens* a
  modal via a `@work` flow (`newsletter_label`'s `l`, `N`, `e`-fallback, `r`,
  `c`, `d`, `C`, and auto-seed on open). Pressing the modal's keys immediately
  after `pause()` works because Pilot pauses between presses.
- **`newsletter_label` cursor starts on the metadata header** (row 0 is a
  header, not a body line). Use `_goto_body(pilot, app, idx)` before span keys.
  Auto-seed fires on `on_mount` for an unreviewed, story-less newsletter with an
  `extract_fn` — always `drain` after opening one.
- **Screen-stack guards.** All four apps guard re-entrant `push_screen` with
  `len(self.screen_stack) > 1` (review.py uses `self.screen is not
  screen_stack[0]`). To simulate key auto-repeat, `app.post_message(events.Key(
  k, None))` twice then `pause()`; assert `app.is_running` / a single pushed
  screen.
- **CANCEL vs None.** In filter/menu screens an unmapped key dismisses with the
  `CANCEL` sentinel (no change); the mapped clear keys dismiss with `None`
  (actively clear). `newsletter_review`'s sender prompt: Esc→None (no change),
  empty Enter→clear, non-empty Enter→set (the literal word "cancel" is a real
  filter value).
- **Data mutates.** Scenarios mutate the golden objects, so rebuild fresh data
  per scenario (`synth_data.newsletters()` etc.) rather than sharing one list.

## Coverage today

69 scenarios across the four drivers (newsletter_label 16, review 18, edit_tui
13, newsletter_review 22) exercise every binding/action in each module — browse
+ span sub-mode (with **exact committed-slice** assertions), blind + stage
variants, the full filter matrix (incl. the newsletter_review **date filter** —
past-N-days window, since-prompt boundary, CANCEL-vs-clear, and the
`--since`/`init_since` pre-filter), every modal, Esc/cancel/decline/guard
branches, undo (incl. empty-stack and undo-after-skip/exclude), scroll (asserting
the offset, not just liveness), quit paths, and the auto-repeat/last-item guards
— and each diverse data state is actually **opened and its rendered content
asserted** (emoji/CJK, CRLF, no-stories, multi-message, notes, reviewed, multi-
story), including the send-date column's **LOCAL-timezone rendering** (an
evening-UTC send shown on its prior local day) with newest-first sort/no-date-last
ordering, and the `load_assessments` **old-scheme guard** (a list-valued
`themes` rejected with a located `path:lineno` error). When you add a binding or
a workflow branch, add a scenario here in the same change and re-run
`run_all.py`.

The harness was itself built adversarially: an automated coverage critic flagged
every unopened data state and every liveness-only (`is_running`) assertion, and
those gaps were closed — so "green" means content-verified, not just crash-free.
