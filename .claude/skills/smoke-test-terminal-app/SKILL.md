---
name: smoke-test-terminal-app
description: >-
  Smoke-test an interactive terminal/CLI/TUI application by driving it in a
  pseudo-terminal — sending real keystrokes and asserting on rendered output
  plus resulting state. Use this whenever you want to actually SEE an
  interactive terminal program work (not just run its unit tests): verifying a
  fix, confirming a feature, or sanity-checking a keypress/prompt loop or curses
  UI after changing the code. Trigger it for phrases like "smoke test the CLI",
  "drive the review tool", "confirm this works end to end", "Playwright but for
  the terminal", or any request to manually exercise a program that reads
  keystrokes (readchar, getch/curses, prompt loops, REPLs, menus). Because the
  driver is derived FRESH from the app's current code each time, it stays
  correct after edits where a saved script would go stale.
---

# Smoke-testing a terminal / TUI application

This is the terminal counterpart to using Playwright on a web app: launch the
**real** program inside a pseudo-terminal (pty), send keystrokes, watch the
rendered "screen," and assert on both what was displayed and what state changed.

## Why a pty, and why derive it fresh each time

Interactive terminal programs read input in ways that defeat naive `echo | app`
piping:

- **Raw single-key readers** (`readchar`, `termios` cbreak mode) read directly
  from a TTY. Piped stdin isn't a TTY, so they block or misbehave.
- **curses / `getch`** needs a real, sized terminal to render at all.

A pty gives the program a genuine terminal on the slave side while you drive the
master side programmatically. That's the whole trick.

The reason this lives as a skill rather than a committed script: the highest-value
moment to smoke-test is **right after you changed the app** — new keys, renamed
prompts, different saved fields. A frozen script written against yesterday's UI
sends the wrong keys and asserts the wrong things. So the workflow always starts
by reading the *current* code to map the interaction, then writes a throwaway
driver. Treat the driver as disposable; treat the **method** as the asset.

## Choosing the driver

In rough order of preference for this environment:

1. **`pexpect`** — cleanest expect/send semantics. It is usually not installed;
   add it for one run without touching the project:
   `uv run --with pexpect python driver.py ...`
2. **stdlib `pty` + `os`/`select`** — no dependency, more boilerplate. Use if
   `uv`/`pexpect` isn't an option.
3. **`tmux`** (if present) — spawn a detached session, `send-keys`, and
   `capture-pane`. Closest to "a real terminal you poke." Best for full-screen
   curses apps where you want to diff the rendered pane.

Check availability first (`which tmux`; try importing pexpect) and pick what's
there. Don't add a permanent dependency just to smoke-test.

## Workflow

### 1. Read the current code to map the interaction

Before writing anything, open the app's source and extract — for the *current*
version:

- **Entry point**: how it's launched (`python -m pkg.module ...`, flags it needs).
- **Prompts**: the literal strings printed before each input (your `expect`
  anchors). Pick anchors that are unique — avoid matching a substring that also
  appears in a header (e.g. a "Sender type:" prompt vs. a "Senders:" display line).
- **Keys**: which keypress maps to which action, per prompt/stage. This is the
  part most likely to have just changed.
- **Persisted state**: what the program writes and where (a JSONL file, a DB, a
  label) — so you can assert on the *effect*, not just the screen.
- **Inputs it needs**: does it require a data file / fixture to have anything to
  act on? Note what's the minimum.

### 2. Synthesize a minimal fixture (don't touch real data)

If the app reads a data file, build a tiny synthetic one in the scratchpad using
the app's *own* schema/serializer where possible (so the shape can't drift from
the code). Cover one case per behavior you want to exercise. Never point the
smoke test at the user's real data file — copy or generate.

### 3. Write the driver in the scratchpad

Put driver + fixtures under the session scratchpad, never in the repo. Structure:

- Spawn the real entry point in a pty (`pexpect.spawn(sys.executable, [...],
  cwd=repo, encoding="utf-8", timeout=15, dimensions=(40, 100))`). Using
  `sys.executable` under `uv run --with pexpect` reuses the project venv, so the
  app's own imports resolve without a nested `uv run`.
- Mirror the screen so you (and the user) can read the transcript:
  `child.logfile_read = sys.stdout`.
- For each step: `expect` the prompt anchor, then `send` the single key. Raw-key
  readers take one byte — `send("k")`, no newline. Line-based `input()` prompts
  need `sendline(...)`.
- End by expecting a completion marker (e.g. "Saved.") and then `EOF`.

### 4. Assert on rendered output AND persisted state

A smoke test that only checks the screen can miss a save bug; one that only
checks the file can miss a rendering/legend regression. Check both:

- **Output**: assert the expected prompts/legends/confirmations appeared in the
  transcript (`child.before`, or scan the captured log).
- **State**: reload the persisted file via the app's loader and assert the
  fields changed as intended — *and* that fields which must stay put didn't move.
  An action often has a "did" and a "didn't": a temporary skip must leave the
  judgment fields unset **and** not mark the record excluded; assert both, or a
  bug that flips the wrong field slips through. Also assert untouched *records*
  were preserved.

Print a clear PASS/FAIL line per assertion and exit non-zero on any failure, so
the run is self-evaluating.

### 5. Add cross-run / idempotency checks where they matter

Many bugs only show on the *second* interaction. If an action is supposed to
change what shows up next time (e.g. "this item should no longer appear",
"that one should resurface"), run the program again and assert on the new queue.
Re-running against the now-mutated fixture is often the strongest check.

When the program renders a progress/count line (e.g. `Thread 2/5`, `[3/10]`),
assert on that rendered count as a queue-size signal — watching the total drop
from one run to the next is direct visual proof the queue shrank, which a
file-only check shows far less convincingly. Scrape the per-item id/header lines
to capture exactly which items were presented.

### 6. Report and clean up

Summarize what was driven and the PASS/FAIL results to the user. Leave the
scratchpad driver in place for the session (handy to re-run after the next edit)
but don't commit it.

## Worked example: the eval review CLI

> The keys (`p`/`r`/`k`/`e`), prompt strings, and field names below are from one
> snapshot of this app. They illustrate the *method* — they are not a substitute
> for step 1. Verify every key and prompt against the current source before
> reusing this, or you'll send the wrong keystrokes after the next edit. That is
> the whole reason this is a skill and not a saved script.

The review tool (`python -m evals.review`) is a blind keypress loop using
`readchar`; it reads/writes `golden_set.jsonl`. A driver that classifies one
thread, skips one, and excludes one — then verifies the saved file and that the
excluded thread drops out of the queue on a second run:

```python
# run with: uv run --with pexpect python driver.py <golden.jsonl> <repo>
import json, sys, pexpect
GOLDEN, REPO = sys.argv[1], sys.argv[2]

child = pexpect.spawn(sys.executable, ["-m", "evals.review", "--golden-set", GOLDEN],
                      cwd=REPO, encoding="utf-8", timeout=15, dimensions=(40, 100))
child.logfile_read = sys.stdout                      # echo the "screen"

child.expect("Sender type:"); child.send("p")        # classify thread 1
child.expect("Label:");       child.send("r")
child.expect("Sender type:"); child.send("k")        # skip thread 2 (no judgment)
child.expect("Sender type:"); child.send("e")        # exclude thread 3
child.expect("Saved."); child.expect(pexpect.EOF)

data = {d["thread_id"]: d for d in
        (json.loads(l) for l in open(GOLDEN) if l.strip())}
assert data["t_keep"]["reviewed"] and data["t_keep"]["expected_label"] == "needs_response"
assert data["t_skip"]["reviewed"] is False           # skip resurfaces
assert data["t_excl"]["excluded"] is True            # exclude persists
print("PASS")
```

Build the fixture with the app's own schema so it can't drift:

```python
from evals.schemas import GoldenThread   # use the real serializer
# ... construct a few GoldenThread(...) and write t.to_dict() as JSONL lines
```

## Gotchas

- **No-TTY symptoms** (program hangs, ignores keys, "Inappropriate ioctl for
  device") mean stdin isn't a real terminal → you need a pty, not a pipe.
- **Single byte vs line**: `readchar`/`getch` want one keypress (`send`);
  `input()` wants a line (`sendline`).
- **Ambiguous anchors**: a too-short `expect` string can match a display line
  instead of the prompt. Anchor on the exact prompt text.
- **curses apps**: prefer `tmux` capture-pane, or a stdlib-`pty` driver with a
  set window size. For pure handler logic, a unit test with a fake screen object
  (stub `getch`/`addstr`/`getmaxyx`) is often faster than a full pty — use the
  pty when you specifically want to prove the real terminal path works.
- **Timeouts**: always set one on `spawn`; a wrong anchor otherwise hangs the run.
- **Isolation**: generate/copy fixtures into the scratchpad; never drive the
  program against real user data.
