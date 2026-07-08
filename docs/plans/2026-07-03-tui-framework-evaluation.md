# TUI framework evaluation: Textual vs urwid (issue #40)

**Recommendation: migrate to Textual.** The spike validates the issue's leaning.
Both frameworks cleanly eliminate the hand-rolled wrapping/scroll/selection code
and both proved testable in CI, but Textual's official `Pilot` test driver, its
asyncio-native model (which fits `evals/newsletter_label.py`'s existing async LLM
calls and the repo's `pytest-asyncio` setup), and its much richer widget set for
the heavier migration targets outweigh urwid's lighter dependency footprint.

## Context

The repo has three hand-rolled `curses` TUIs — `newsletter_review/tui.py`
(~490 lines), `evals/newsletter_label.py` (~759 lines TUI code), `evals/edit_tui.py`
(~351 lines) — plus the `readchar`-based `evals/review.py`. The render/key-dispatch
layer is untestable today: each TUI factors state transitions into pure functions
for unit tests and leaves the actual UI covered only by manual pty smoke tests.

Historical note: `docs/plans/2026-02-20-newsletter-tui-design.md` originally
designed the newsletter review TUI **in Textual**, but the implementation shipped
on stdlib `curses`. This spike is a concrete re-evaluation of that fork in the road.

## Spike setup

One representative screen — the `newsletter_review` browser (list view, detail
view, tier filter, quit) — was ported to each framework with **identical scope**:

- `spikes/tui_eval/textual_app.py` + `tests/test_spike_tui_textual.py` (10 tests)
- `spikes/tui_eval/urwid_app.py` + `tests/test_spike_tui_urwid.py` (10 tests)

Both ports reuse the pure helpers from `newsletter_review.tui` **unchanged**
(`apply_filters`, `format_list_row`, `build_detail_lines`, `format_filter_summary`);
only the UI layer was reimplemented. Theme/sender filters were omitted from both
ports (the theme filter is mechanically identical to the tier filter; the sender
text-input is a stock widget in both frameworks — `Input` in Textual, `Edit` in
urwid — so neither omission hides effort).

Run: `uv run --extra spike python -m spikes.tui_eval.textual_app` (or `urwid_app`).
Test: `uv run --extra dev --extra spike pytest tests/test_spike_tui_*.py -v`.
The `spike` optional-dependency group keeps both frameworks out of runtime and dev
deps; the spike tests `pytest.importorskip` so the main suite is unaffected.

## Criterion 1: Testing (primary)

**Both frameworks let us test the real UI in CI — the thing curses cannot do.**
Every spike test drives actual key presses and asserts on rendered output or
widget state. A mutation check (disabling the filter-refresh in each app) made
the corresponding UI tests fail, proving the tests exercise real behavior.

Textual has an official, documented test driver (`Pilot`):

```python
async def test_tier_filter_narrows_list():
    app = ReviewApp(_records())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("f", "g")
        assert len(app.query_one(ListView)) == 3
        assert "tier:good" in str(app.query_one("#title", Static).render())
```

The `async` is zero-friction here: `asyncio_mode = "auto"` is already configured.
Textual also offers snapshot testing (`pytest-textual-snapshot`, not used in the
spike) for whole-screen regression coverage.

urwid has **no official driver**; the standard pattern is hand-rolled — call
`keypress()` on the top widget and render the canvas:

```python
def test_tier_filter_narrows_list():
    app = ReviewApp(_records())
    app.keypress(SIZE, "f")
    app.keypress(SIZE, "g")
    text = "\n".join(row.decode() for row in app.render(SIZE).text)
    assert "3/6 records" in text and "tier:good" in text
```

This worked well for a read-only screen — it is fully synchronous and fast — but
it is a convention we would own, and it constrains app structure (the spike had
to be built as a `WidgetPlaceholder` swapping frames precisely so tests could
avoid `MainLoop`). Canvas assertions are also lower-level: `canvas.text` gives
bytes rows; focus/attribute assertions require digging into canvas internals.

Test runtime for the same 10 behaviors: **urwid 0.11s, Textual 2.5s** (~0.25s
per Pilot test for app startup). Both are fine for CI; urwid is notably snappier.

Verdict: both testable; **Textual's is the supported, documented path** (plus
snapshots), urwid's is a do-it-yourself harness. Textual wins the primary criterion.

## Criterion 2: Wrapping / scrolling / list+detail widgets

| Hand-written in curses today | Textual | urwid |
|---|---|---|
| Cursor + scroll-offset math, selection highlight | `ListView` (free) | `ListBox` + `SimpleFocusListWalker` (free) |
| Page up/down, home/end | free (ListView/scrollables) | free (ListBox) |
| Scrollable detail with clipping (`_safe_addstr`) | `VerticalScroll` (free) | `ListBox` of `Text` (free) |
| Resize handling (`KEY_RESIZE` rebuild) | free (reactive layout) | free (render-time size) |
| Line wrapping | free (`Static`/CSS) | free (`Text` wraps by default) |
| Modal prompt overlay | `ModalScreen` + callback | hand-rolled key-mode flag (spike) or `Overlay` |

Both eliminate the exact code that caused the newsletter_label friction (wrapping,
scroll math, selection highlighting). Textual goes further for the *editing* TUIs:
`Input`, `DataTable`, `TextArea`, focus management, and CSS-based layout are stock;
urwid has equivalents (`Edit`, `Columns`/`Pile`) but at a lower level of abstraction.

Screen management differs meaningfully: Textual has a built-in screen stack
(`push_screen`/`pop_screen` with result callbacks — the spike's filter modal and
detail view are each ~10 lines of wiring). In urwid the spike hand-rolled both
(swap `original_widget`, a `_filter_mode` flag) — small, but it *is* the kind of
bespoke state machine we're trying to stop writing.

## Criterion 3: Dependency + architecture fit

Measured in the spike venv (installed size of the full closure):

| | packages | installed size | compiled extensions |
|---|---|---|---|
| textual 8.2.8 | 10 (rich, pygments, markdown-it-py, mdurl, linkify-it-py, uc-micro-py, mdit-py-plugins, platformdirs, typing-extensions) | 9.0 MB | none (pure Python) |
| urwid 4.0.3 | 3 (typing-extensions, wcwidth) | 3.0 MB | none (pure Python) |

**Python 3.14** (`requires-python = ">=3.14"`): both declare support — textual
8.2.8 (`requires-python >=3.9,<4.0`, `Programming Language :: Python :: 3.14`
classifier) and urwid 4.0.3 (`>=3.9`, 3.14 classifier) — and both closures are
pure Python (zero `.so` files), so there is no ABI risk on 3.14. Caveat: the
sandbox this spike ran in could not install a 3.14 interpreter (egress policy
blocks python.org and GitHub release downloads), so imports, the full test suite
(919 tests), and the pty smoke tests were verified on **Python 3.13.12**:

```
$ python -c "import sys, textual, urwid; print(sys.version.split()[0], textual.__version__, urwid.__version__)"
3.13.12 textual 8.2.8 urwid 4.0.3
```

Re-running `uv run --extra dev --extra spike pytest tests/ -v` on a real 3.14
machine is the one remaining checkbox; given the classifiers, upstream CI, and
pure-Python closures, the risk is negligible.

**Architecture**: Textual is asyncio-native. That is a shift from the synchronous
`getch` loops, but it *matches* the codebase's direction — the daemon is pure
asyncio, `evals/newsletter_label.py` already makes async LLM calls (currently
awkwardly bridged into a synchronous curses loop), and pytest-asyncio is already
configured. urwid is synchronous and closer to the current style; its asyncio
integration (`AsyncioEventLoop`) exists but is bolted on rather than native.

## Criterion 4: Migration effort / LOC

Counting method: non-blank, non-comment lines (`grep -cvE '^\s*(#|$)'`).
Curses baseline = the equivalent-scope slice of `newsletter_review/tui.py`
(curses layer lines 184–491, minus the omitted `_prompt_theme`/`_prompt_sender`).
Spike apps counted minus their `_sample_records`/`main` demo scaffolding.

| implementation | UI-layer LOC | vs curses |
|---|---|---|
| curses (equivalent scope) | 217 | — |
| Textual | 107 | **−51%** |
| urwid | 91 | **−58%** |

Both ports took comparable effort (a few hours each including tests) and neither
hit a wall. What disappeared entirely in both: `_safe_addstr`, all cursor/scroll
arithmetic, selection highlighting, page-size computation, `KEY_RESIZE` handling.
Full parity would add the theme filter (copy of tier) and sender input (stock
widget) — no hidden complexity in either framework.

Gotchas found: Textual parses `[f]` in labels as Rich markup (the pty smoke test
caught the help line rendering as "ilter tier"); anything displaying
record-derived text needs `markup=False`. urwid needed the `WidgetPlaceholder`
structure chosen up front for testability, and its `enter`/focus key routing
had to be intercepted in a hand-written `keypress` override.

Per-test LOC is a wash (~115 lines for 10 tests each).

## Criterion 5: Consistency

Four UIs need a home: `newsletter_review/tui.py`, `evals/newsletter_label.py`,
`evals/edit_tui.py`, `evals/review.py`. Either framework can host all four, so
consistency is achievable with both; the question is which stack we want to
standardize on. Textual's screen stack + widget library covers the whole range
(read-only browser → multi-phase labeling editor with text input and LLM calls)
with one set of idioms, and its async model means `newsletter_label`'s LLM calls
stop needing a sync/async bridge. During the transition the codebase temporarily
mixes curses and the new stack — the pure-helper split makes per-file ports
independent, so this is low-risk either way.

## Criterion 6: Runtime (container / SSH terminal)

Both apps were driven end-to-end in a real pty (tmux pane, 100×30, inside the
Docker-based sandbox; `TERM=screen` inside tmux, host `TERM=linux`):
launch → arrow-key navigation → Enter into detail → Esc back → `f`/`g` tier
filter → `f`/`c` clear → `q` quit. All 20 checks passed for both apps; both
exited with code 0 and restored the terminal. Mouse support (both frameworks)
was not exercised — keyboard-only operation is the requirement and it works.

## Recommendation

**Textual**, for the reasons ranked:

1. **Testing (primary criterion)**: official `Pilot` driver + optional snapshot
   testing vs a hand-rolled harness we'd own and have to keep teaching.
2. **Fit for the hard targets**: the real payoff is `newsletter_label` and
   `edit_tui` — editing flows, text input, modal phases, async LLM calls. Textual's
   screen stack, `Input`/`TextArea`, and asyncio-native loop map directly onto them.
3. **Framework does more of the work**: modal/screen management came free in the
   spike; in urwid we immediately started hand-rolling mode flags again.

Honest costs, accepted: ~9 MB / 10-package closure (vs urwid's 3 MB / 3), slower
UI tests (~0.25s each), the Rich-markup escaping gotcha for record-derived text,
and an async programming model in the UI layer.

urwid remains a respectable fallback: smallest footprint, synchronous, fastest
tests, and the spike port was actually the shorter one. If the dependency weight
ever becomes disqualifying, this spike shows the port is tractable.

## Next steps

- Phased migration (one TUI per PR), tracked in the follow-up issue — see issue
  linked from #40: Phase 0 promotes `textual` to runtime deps and sets shared
  conventions; then `newsletter_review` → `newsletter_label` → `edit_tui` →
  `review.py`, each replacing pure-function-only test modules with real Pilot
  UI tests (keeping the pure helpers and their tests where they still add value).
- Keep the pty smoke-test skill as a backstop until each port reaches parity
  (it caught a real rendering bug in the spike).
- Delete `spikes/` and the `spike` optional-dependency group once the first
  real port lands.
- Confirm the suite on a real Python 3.14 interpreter (one `uv run` on the
  user's machine; blocked only by sandbox egress policy here).
