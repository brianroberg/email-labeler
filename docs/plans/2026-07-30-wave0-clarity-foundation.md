# Wave 0 — Clarity foundation (docs only)

> **Status: executed 2026-07-30** (owner-approved; commits d766662..HEAD on
> `claude/wave0-clarity-foundation`; three-agent acceptance pass applied).
> First plan written under the frozen-history convention it introduces (T5).
> See "Execution record" at the end for accepted deviations and the Wave 3
> carry-list.

Companion to the Phase 1 clarity audit (2026-07-30, 110 verified findings) and
the Phase 2 owner dialogue that adjudicated groups A–E. This plan implements
**Wave 0 only**: the docs-only foundation. After it lands: **Wave 0.5** (a
triage of every open GitHub issue against the new registry — annotate each as
reversed / scheduled / absorbed / reframed / unaffected by a D-number, close
the mooted ones; its output feeds Wave 2 planning), then Waves 1–3 (tests/CI,
behavior changes, sweep), each with its own plan.

## Goal

Give every fact one authoritative home, record the adjudicated decisions where
reviews can see them, and make the primary docs state the system's actual
identity, privacy posture, and failure model — so that subsequent code work
lands inside a framework that stops regenerating review findings.

## Principles (binding on every task below)

- **P1 — No behavior changes.** No `.py` file is edited. `config.toml` is not
  edited (its values and comments are already authoritative and correct).
- **P2 — Never document unshipped behavior as current.** Decisions whose
  implementation is pending (Wave 1/2) are recorded in the registry with
  `implementation pending`. Descriptions of *current* behavior (e.g.
  README.md's Resilience section) stay accurate to today's code until the
  behavior actually changes.
- **P3 — One commit per task**, in task order (each leaves the repo coherent).
- **P4 — Full test suite after every task** (`uv run --extra dev pytest
  tests/`). The meta-tests parse README-technical.md and the READMEs — nothing
  parses `.env.example` or CLAUDE.md — and a doc edit can fail them. Known
  constraints are noted per task. (Known-failing baseline: issue #67's
  timezone-dependent test fails west of UTC; unrelated to this wave.)
- **P5 — TDD note.** Per CLAUDE.md, red/green applies to behavior changes;
  these are documentation changes with no production code, so no new tests are
  written in this wave. Wave 1 adds the enforcement tests.

## Task list

| # | Task | Files touched |
|---|---|---|
| T1 | Create the decisions registry | `docs/decisions.md` (new) |
| T2 | Rewrite CLAUDE.md as the constitution | `CLAUDE.md` |
| T3 | README.md: identity + honest privacy model | `README.md` |
| T4 | README-technical: fork declaration, de-duplicated settings blocks, stand-in warning | `README-technical.md`, `.env.example` |
| T5 | Freeze docs/plans (status headers, neutralized directives, roadmap banner) | 9 files in `docs/plans/` |
| T6 | Status header on the UX-redesign doc | `docs/newsletter-label-ux-redesign.md` |
| T7 | Acceptance check | — |

---

## T1 — `docs/decisions.md` (new file, seed content)

Create with exactly this content:

```markdown
# Decisions Registry

Adjudicated tradeoffs and product decisions. Per the Review Charter in
CLAUDE.md, reviewers check this registry before flagging: a behavior recorded
here is a decision, not a finding. Reversing a decision requires updating its
entry in the same change that reverses it.

Entry format — one `##` per decision:
`## D<n> — <title> (<date decided>)` followed by **Status**, the decision, and
what it forecloses. Status parts use: `implemented` · `implementation pending
(<wave/issue>)` · `reversed by D<m>`; a split decision combines them (e.g.
"docs implemented; enforcement pending (Wave 1)").

---

## D1 — One system, two independent functions (2026-07-30)

**Status:** implemented (docs, Wave 0).

The project is one system housing two independent functions — email triage and
newsletter grading — sharing a daemon and chassis. It is not two products and
will not be split into two repos; the shared chassis stays shared. Hybrid
operation (both functions in one daemon process) is the normal deployment.
Newsletter grading is enabled by the presence of `[newsletter]` in config.toml;
`NEWSLETTER_ONLY` is a *filter* that skips non-newsletter threads, not a mode
switch. A symmetric `INBOX_ONLY` filter (email triage alone) would be
legitimate — the asymmetry is an accident of history, not a decision. Neither
function depends on the other.
Forecloses: repo splits; treating newsletter grading as a test-only mode;
fixes that assume one function's values govern the other.

## D2 — Privacy posture: best-effort routing, stated honestly (2026-07-30)

**Status:** implemented (docs, Wave 0); test enforcement pending (Wave 1).

The email-triage privacy guarantee is: **bodies of threads classified as
person are processed only by the local LLM**. This is best-effort routing, not
an absolute guarantee — Stage 1 classifies from metadata and can be wrong, and
unparseable Stage 1 output deliberately defaults to SERVICE (availability
first). The residual misroute risk is *measured, not denied*: the eval reports
a privacy-violation rate. Docs must not state the invariant in absolute
("NEVER") form.
Forecloses: "hardening" fixes that flip the unknown-sender default to PERSON;
findings that treat the SERVICE default as a privacy bug.

## D3 — Newsletter ownership rule (2026-07-30)

**Status:** docs implemented (Wave 0); exact-address matching pending (Wave 2).

A thread with any message To/Cc-addressed to the configured newsletter
recipient belongs to the newsletter function and is organizational content —
its full transcript, **including person-written replies**, goes to the cloud
LLM by design, before and instead of person/service routing. The recipient
match must be an exact address comparison (current substring matching is a
convicted bug, fixed in Wave 2).
Forecloses: routing newsletter threads through Stage 1 "to restore the
invariant"; privacy findings about person replies inside newsletter threads.

## D4 — Public-API local stand-ins are non-production only (2026-07-30)

**Status:** implemented (docs, Wave 0).

Pointing `MLX_URL`/`MLX_API_KEY` at a public API (e.g. Novita.ai) sends person
email bodies off-network and is sanctioned **only for non-production/eval
use**. Docs describing the stand-in carry this warning.
Forecloses: presenting a public stand-in as a routine production configuration.

## D5 — Failure model: two rules and a scope (2026-07-30)

**Status:** model is the governing design now; corollaries pending (Wave 2).

**Rule 1 — Outcomes only come from successes.** A committed outcome (label,
archive, grade record, `agent/processed`) is only ever produced by a
successful classification. Failures never commit anything: they defer, set
aside findably (`agent/attempted`), or stop loudly.
**Rule 2 — Blame by correlation.** Siblings succeeding while one thread fails →
thread-scoped: bounded strikes, then `agent/attempted`. Siblings failing the
same way → shared cause (provider, proxy, disk, or our config/code): count no
strikes, get loud, keep the backlog intact.
**Scope — functions fail independently.** A fault that disables one function
(e.g. its provider's balance) fails that function loudly without stopping the
other.

Corollaries, each `implementation pending (Wave 2)` until landed:
- Exhausted 429/5xx (LLM and proxy) never count toward give-up. **This
  deliberately reverses issue #26.** proxy_client's transient-error docstrings
  become true; llm_client's non-200 docstring (which currently matches the old
  counting behavior) is rewritten to the new rule.
- Correlation is the attribution mechanism (detection mechanics designed in
  Wave 2). The single-thread masquerade (provider-shaped errors, siblings
  succeeding) retries forever, never abandoned, with a distinct ERROR
  escalation line repeated on the status heartbeat.
- The halt becomes per-function (today: daemon-wide).
- A keyword-free label reply raises instead of silently defaulting to
  LOW_PRIORITY→archive (Rule 1; completes the issue-#64 fail-loud direction).
  The unknown-sender→SERVICE default *stays* (D2).
- `agent/newsletter/no-stories` may only result from a successful extraction
  that found zero stories; all-grades-unparseable is a failure (extends the
  issue-#30 principle to the parse-to-None path).
- Assessment-sink faults are shared-cause: never counted, retried forever
  (docs already claim this; code currently deviates).
- `max_failures` (the strike bound, currently 5) becomes env-overridable
  (`MAX_FAILURES`) and documented with the other knobs.
Forecloses: per-cell relitigating of the failure table; new error paths that
commit outcomes on failure; give-up counting for provider-shaped faults.

## D6 — Proxy 403 is a human answer, not a failure (2026-07-09, reaffirmed 2026-07-30)

**Status:** implementation pending (Wave 2; originally issue #28 Option A).

A 403 on a gated write means an operator said "not now": log one clean line,
count nothing, re-offer next cycle. A rejection can never cause
`agent/attempted`. (Also a corollary of D5, kept separate for the #28 trail.)

## D7 — Documentation authority map + altitude rule (2026-07-30)

**Status:** implemented (Wave 0).

Every fact has exactly one home; other docs point rather than restate.
**CLAUDE.md** = constitution: identity, privacy posture, failure model, review
charter — principles, no literal values. **config.toml** = authoritative for
every operational value *and its rationale* (its comments). **README-technical**
= operational reference (env vars, config key meanings/constraints, procedures,
test coverage); it does not restate config.toml values. **README.md** = human
overview. Altitude rule: prose never quotes exact log strings or literal counts
outside runbooks.
Forecloses: duplicated fact tables; doc fixes that add a second home.

## D8 — docs/plans/ is frozen history (2026-07-30)

**Status:** implemented (Wave 0).

Plan/design docs are historical artifacts. Every file carries a one-line
status header (`proposed` / `executed` / `partially executed` / `superseded`,
with date); superseded plans' agent-directives are neutralized. The 2026-07-08
issue roadmap is frozen with a banner; GitHub issues are the living tracker.
New plans start with the status line.

## D9 — proxy_client.py / gmail_utils.py are a declared fork (2026-07-30)

**Status:** implemented (docs, Wave 0).

Both files were copied from email-agent (Feb 2026) and have since diverged
deliberately (error taxonomy, retry layer, body cleaning). They are
**independently maintained here with no sync obligation in either direction**,
and no re-convergence is intended. Never "sync" them with email-agent.

## D10 — Remove the eval web app (2026-07-30)

**Status:** implementation pending (Wave 2).

`evals/web_app.py`, `web_auth.py`, `web_data.py`, `run_web.py`,
`evals/templates/`, and evals/README.md §5 are removed; `fastapi`, `jinja2`,
`uvicorn` leave the runtime dependencies. Coupled removals the implementer
must include or the suite fails: drop `evals.run_web` from
`tests/test_eval_cli_docs.py`'s `_CLI_MODULES` (it imports the module), and
delete the `### run_web` section of evals/README-technical.md. Rationale:
forgotten by the owner, workflows CLI-covered, zero tests, and it contradicted
the no-web-server posture at the dependency level. CoT capture/sidecars are
unaffected.

## D11 — Minimal release identity (2026-07-30)

**Status:** implementation pending (Wave 2).

The Dockerfile bakes the git SHA (build-arg), the daemon logs it once at
startup, and notable releases get a lightweight git tag. No semver, no
changelog — just enough that "what is deployed?" has an answer in the logs.

## D12 — Assessments JSONL: documented schema + version field (2026-07-30)

**Status:** implementation pending (Wave 2).

The assessments JSONL (the only durable copy of gradings) gets a schema
document (home: README-technical) and a `schema_version` field on each record,
so the next schema evolution is a version bump + migration, not shape-sniffing.

## D13 — Adopt CI (2026-07-30)

**Status:** implementation pending (Wave 1).

A minimal GitHub Actions workflow (uv sync, pytest, ruff) on PRs and pushes to
main. The suite is fully mocked; no secrets. Its absence was not deliberate.

---

*Backfilled entries — decisions adjudicated before this registry existed:*

## D14 — Theme labels are Emphasized-only (issue #53, 2026-07-08)

**Status:** implemented.

`agent/newsletter/theme/*` is applied only when a theme grades Emphasized;
Present is recorded in the JSONL but not labeled; Absent is omitted from the
record. Deliberately changed the label's meaning from "present" to
"emphasized".

## D15 — Newsletter rubric: Poor/OK/Good + Absent/Present/Emphasized (issue #53, 2026-07-08)

**Status:** implemented.

Dimensions score as Poor/OK/Good stored as ints 1/2/3; tier bands ≥2.75 /
≥2.25 / ≥1.75 on the four-dimension mean. Themes as theme→grade dicts. No
backward compatibility in readers (production JSONL is migrated via
`scripts/migrate_assessments.py`, which preserves tiers verbatim).

## D16 — Local tier: native thinking disabled, max_tokens 4096, coupled (issue #64)

**Status:** implemented.

Thinking is disabled on the local classifier (both dialect forms sent);
`finish_reason: length` raises loudly; local `max_tokens` = 4096 sized to the
honest scaffold demand. These three are one coupled decision — see
config.toml's `[llm.local]` comments (authoritative).

## D17 — Cloud tier: native thinking enabled, max_tokens 1024 (issue #64 era)

**Status:** implemented.

Deliberately asymmetric with D16: glm-5's accuracy rides its native reasoning
channel; budget kept at 1024 with the loud length-finish as backstop. See
config.toml's `[llm.cloud]` comments (authoritative). Disable only behind an
eval gate.

## D18 — Assessment record before labels (PR #65)

**Status:** implemented. (An instance of D5 Rule 1 before it was named.)

The JSONL record is written before Gmail labels commit; a sink fault leaves
the thread unprocessed rather than labeled-but-lost. Dedup on read: newest
`timestamp` per `thread_id` wins.

## D19 — HTTP 429 never halts the daemon (PR #61)

**Status:** implemented.

Rate-limit phrasing is indistinguishable from quota exhaustion; a wrong
restart-only halt is worse than retry. Balance-signature 402/400/403 halts
(today daemon-wide; per-function under D5, pending).

## D20 — Content-less grading is a failure, not an outcome (issue #30, 2026-07-08)

**Status:** implemented for the exception path; parse-to-None path pending
(extended by D5, Wave 2).

Stories-exist-but-every-grade-errored raises and reaches the give-up path;
genuine zero-story extraction remains a valid `no-stories` outcome.
```

---

**Sequencing note (P3):** registry statuses reading `implemented (Wave 0)`
become true as T2–T6 land in this branch's subsequent commits; the registry is
committed first so those commits can cite D-numbers. The branch is coherent at
merge.

## T2 — CLAUDE.md rewrite (the constitution)

CLAUDE.md is rebuilt around principles, dropping every duplicated fact table.
No test parses CLAUDE.md (verify in T7), so the constraint is editorial only.

**New section order:** identity → documentation map → privacy posture →
failure model → architecture sketches → review charter → package management →
testing (unchanged) → pointers.

**§1 Identity — replaces the current opening paragraph:**

```markdown
# Email Labeler

One system, two independent functions sharing a daemon and chassis:

1. **Email triage** — polls Gmail, classifies each thread with a two-tier LLM
   pipeline, applies inbox/archive labels.
2. **Newsletter grading** — grades ministry-newsletter stories for writing
   quality and thematic alignment, appending assessments to a durable JSONL.

Both functions normally run in one daemon process — hybrid operation is the
default deployment. Newsletter grading is enabled by the presence of
`[newsletter]` in config.toml, **not** by `NEWSLETTER_ONLY`, which is a filter
that skips non-newsletter threads (for running the newsletter function alone).
Neither function depends on the other, and design values are scoped
per-function. The failure model (D5) scopes faults per-function as well — note
its per-function halt is still pending: today a provider-balance halt stops
the whole daemon (registry D5/D19).

Non-goals: this is a single-owner deployment, not a generic multi-user
product; org-specific content (the Ends Statement themes, the newsletter
recipient) is configuration, not something to abstract away.
```

**§2 Documentation map — replaces the current "Documentation" list.** Keeps the
five bullets (README, README-technical, evals READMEs, runbook) but adds the
authority rule:

```markdown
Authority map (decision D7): every fact has one home. CLAUDE.md holds
principles (no literal values). config.toml is authoritative for operational
values and their rationale — its comments are the design record for sizing.
README-technical.md is the operational reference (env vars, config key
meanings, procedures, test coverage). README.md is the human overview.
docs/decisions.md is the decisions registry. docs/plans/ is frozen history
(status header on every file). When editing docs: point, don't restate.
```

**§3 Privacy posture — replaces the current "Privacy Invariant" section:**

```markdown
## Privacy Posture (email-triage function)

Bodies of threads **classified as person** are processed only by the local
LLM; the cloud LLM sees metadata (sender, subject, snippet) for Stage 1
routing. This is best-effort routing, stated honestly (decision D2): Stage 1
can misclassify, and unparseable Stage 1 output deliberately defaults to
SERVICE. The residual misroute risk is measured — the eval suite reports a
privacy-violation rate — not denied. Do not restate this guarantee in
absolute ("never") form, and do not "fix" the SERVICE default to PERSON.

Ownership rule (D3): a thread with any message To/Cc-addressed to the
configured newsletter recipient belongs to the newsletter function and is
organizational content — its full transcript, including person-written
replies, goes to the cloud by design, without person/service routing.

Public-API stand-ins for the local endpoint (`MLX_API_KEY`, e.g. Novita.ai)
are non-production/eval-only (D4): pointing the local tier at a public API
sends person bodies off-network.
```

**§4 Failure model — new section (absorbs old decisions 2, 3, 5, 6):**

```markdown
## Failure Model (decision D5)

**Rule 1 — Outcomes only come from successes.** A committed outcome (label,
archive, grade record, `agent/processed`) is only ever produced by a
successful classification. Failures never commit anything: they defer, set
aside findably (`agent/attempted`), or stop the affected function loudly.

**Rule 2 — Blame by correlation.** When processing fails, ask whether sibling
threads are succeeding. Others succeed → the thread is the problem: bounded
strikes, then set aside under `agent/attempted` (findable, never silent).
Others failing the same way → the thread is innocent (provider, proxy, disk,
or our own config/code): no strikes for anyone, get loud, keep the backlog.

**Scope — functions fail independently.** A fault that disables one function
(e.g. its LLM provider's balance) stops that function loudly; the other
function continues.

The registry entry D5 lists the corollaries and their implementation status —
several are `pending`, and until they land, code deviates from this model
where D5 says so. Current-behavior descriptions live in README.md
(Resilience) and stay accurate to the code, not the model.
```

**§5 Architecture sketches.** Keep both existing pipeline sketches, corrected:
the email sketch unchanged; the newsletter sketch's introduction changes from
"When `NEWSLETTER_ONLY=1`, the daemon switches to a newsletter-specific
pipeline" to:

```markdown
Whenever `[newsletter]` is configured (it is in the shipped config), threads
To/Cc-matching the recipient are routed to the newsletter pipeline before
person/service routing. `NEWSLETTER_ONLY=1` additionally skips all
non-newsletter threads.
```

Also correct the sketch's tier line — the current one elides that the overall
tier is the *best story's* tier (per-story bands, then max). It becomes:
"Compute each story's tier from its averaged dimension scores; overall tier =
best story's tier (bands: registry D15 / config.toml)".

**§6 Review charter — new section:**

```markdown
## Review Charter

Before flagging any finding, reviewers (human or agent) check
`docs/decisions.md`. A behavior or tradeoff recorded there is a decision, not
a finding — do not re-litigate it. A finding that contradicts a registry entry
is valid only if it names the entry and argues the reversal explicitly. A
change that reverses a registry decision must update that entry in the same
change.
```

**§7–8 Package Management and Testing:** kept verbatim (the TDD rules are
untouched).

**§9 Pointers — replaces the dropped tables.** The following current CLAUDE.md
content is **deleted and replaced by one-line pointers**:

- *Project Structure list* → "Structure and per-module test coverage:
  README-technical.md (authoritative)." (Drops the list that omits
  newsletter.py, retry.py, config_utils.py, evals/, and two of three scripts.)
- *Environment Variables list* → "Env vars: README-technical.md (authoritative,
  test-guarded)." (Drops the list missing USER_NAME, VIP_SENDERS,
  EMAIL_LABELER_API_KEY.)
- *Labels lists (email + newsletter)* → "Label names/actions: config.toml
  `[labels]`, `[labels.actions]`, `[newsletter.labels]` (authoritative). All
  labels must be pre-created in Gmail; the daemon verifies at startup and
  exits if any are missing. Markers: `agent/processed` (success),
  `agent/attempted` (give-up, findable), both excluded from `gmail_query`."
- *Newsletter Review TUI hotkeys/flags* → "TUI usage: README.md (guarded by
  test_tui_docs)." Keep the one-line launch command.
- *Key Design Decisions 1–7* → absorbed: 1→§1/§3, 2/3/5/6→§4+D5 registry,
  4→D10, 7→one retained line: "Per-cycle processing is bounded by the
  cloud/local/fetch/write semaphores — sizing rationale lives in config.toml
  `[daemon]` comments (authoritative). `local_parallel` defaults to 1; see
  README-technical 'Local Model Serving & Memory' before raising it."
- *Newsletter scoring details (tier bands, label meanings)* → registry
  D14/D15 + config.toml; CLAUDE.md keeps the two-sentence pipeline summary.

The runbook pointer and the `scripts/migrate_assessments.py` description stay
(the runbook line already carries its do-not-execute-with-write-access
warning).

## T3 — README.md

1. **Opening paragraph** gains the identity sentence: "One system, two
   independent functions: email triage and newsletter grading (see below);
   both normally run in one daemon."
2. **Privacy Model section (lines 7–19) rewritten** to the honest form:

```markdown
## Privacy Model

The email-triage function keeps person-email bodies local, best-effort:

1. **Stage 1 (Cloud LLM)** sees only metadata — sender, subject, snippet —
   and classifies the sender as person or service. Bodies are never sent in
   this stage.
2. **Stage 2** routes by that result: service-classified bodies go to the
   cloud LLM; person-classified bodies go to the local MLX instance and do
   not leave the local network. If the local LLM is unavailable, person
   emails are skipped and retried — never sent to the cloud as a fallback.

This is best-effort routing, not an absolute guarantee: Stage 1 classifies
from metadata and can be wrong, and unparseable Stage 1 output deliberately
defaults to SERVICE. The residual misroute risk is measured — the eval suite
reports a privacy-violation rate (person threads that were routed as
service).

**Newsletter threads are exempt by rule** (decision D3): a thread addressed
(To/Cc) to the configured newsletter recipient is organizational content, and
its full transcript — including person-written replies in the thread — is
graded by the cloud LLM without person/service routing. (Today the recipient
match is a substring check against the To/Cc headers — broader than an exact
address; the exact-address fix is pending, decision D3.)
```

3. **The `NEWSLETTER_ONLY` paragraph (Docker section)** changes its framing
   from "To classify only newsletters" to "To run only the newsletter
   function (skip email triage):" — one line.
4. The architecture diagram already shows the newsletter branch correctly; no
   change.
5. **Resilience section: no changes** (P2 — it describes current behavior,
   which Wave 2 will change; it is updated there).

Constraint: `test_tui_docs.py` requires the `python -m newsletter_review`
fenced block to survive — it does (untouched).

## T4 — README-technical.md + .env.example

1. **Fork declaration** (lines 14–15): both file descriptions become
   "Gmail API proxy client (forked from email-agent, Feb 2026; independently
   maintained here — no sync obligation, see decision D9)" and likewise for
   gmail_utils.
2. **LLM settings section (lines 122–138): the stale TOML block is deleted**
   (it shows deepseek/8096/0.2 against the shipped glm-5/1024/0 — audit
   finding "LLM-settings block drifted from config.toml") and replaced with
   prose per the authority map:

```markdown
### LLM settings

`[llm.cloud]` and `[llm.local]` configure model, `max_tokens`, `temperature`,
and `timeout` per tier. **config.toml is authoritative for the values and —
via its comments — for their rationale**; they are deliberate calibrations
(issue #64), not free knobs: the token budgets are coupled to the
thinking-disable decision (registry D16/D17), and `finish_reason: length`
raises rather than silently truncating. Change them only with the config
comments in view. URL and API key come from `.env` (secrets); model names and
inference parameters stay in version-controlled config.toml.
```

   The `extra_body` subsection (dialects for disabling thinking) **stays** —
   it documents mechanism, not values, and is the home for that material.

   **Same pattern for the other two restated blocks** (this completes D7 for
   the file — the verifiers caught that removing only the LLM block would
   leave D7's "stops restating TOML blocks" claim under-implemented):

   The `[daemon]` fenced block (~lines 87–99) is replaced with:

   ```markdown
   `[daemon]` keys (values and rationale live in config.toml — authoritative):
   `poll_interval_seconds` (poll cadence) · `status_interval_seconds` (idle
   heartbeat cadence) · `max_emails_per_cycle` (per-cycle thread cap; env
   override `MAX_EMAILS_PER_CYCLE`) · `gmail_query` (the unprocessed-thread
   search; excludes the `agent/processed` and `agent/attempted` markers) ·
   `max_thread_chars` (transcript cap — see below) · `cloud_parallel` /
   `local_parallel` / `fetch_parallel` / `write_parallel` (concurrency
   semaphores; env overrides `LOCAL_PARALLEL` / `WRITE_PARALLEL`) ·
   `healthcheck_file` (heartbeat path).
   ```

   The surrounding prose (the `local_parallel`/KV-cache explanation, the
   `max_thread_chars` rationale) **stays** — it is meaning, not values.

   The `[newsletter]`/`[newsletter.llm]` fenced block (~lines 183–190) is
   replaced with:

   ```markdown
   `[newsletter]` configures `recipient` (the To/Cc match that routes a
   thread to the newsletter function) and `output_file` (the assessment
   JSONL sink); `[newsletter.llm]` configures the grading model and its
   `max_tokens` / `temperature` / `timeout` — values and sizing rationale in
   config.toml (authoritative; the budget/timeout comments there are
   deliberate issue-#64 calibration).
   ```

   The volume-mount warning and sink-preflight subsections that follow
   **stay** unchanged.
3. **`MLX_API_KEY` env-table row** gains the D4 warning: "Empty for real MLX.
   Setting it for a public API stand-in (e.g. Novita.ai) is
   **non-production/eval use only** — it sends person email bodies
   off-network (decision D4)."
4. **`NEWSLETTER_ONLY` row** reworded: "Run only the newsletter function:
   threads not matching the newsletter recipient are skipped. Newsletter
   grading itself is enabled by `[newsletter]` in config.toml, with or
   without this flag."
5. **`.env.example`**: add a commented `# MLX_API_KEY=` entry to the Local
   LLM section carrying the same D4 warning — the file currently has **no**
   `MLX_API_KEY` line at all. No test parses `.env.example`, so this is
   suite-safe; for the same reason, review it by hand.
6. **Not in this wave:** structure-tree/coverage-table backfill (waits for
   Wave 2's web-app removal so we don't list files about to be deleted;
   issue #39 covers the general guard) — *except* the two fork lines above.

Constraint: `test_env_var_docs.py` asserts every env var referenced in daemon
source/config.toml appears **by name** in README-technical's "## Environment
Variables" section — names only, one direction; it never reads `.env.example`,
which nothing guards (a Wave 1 candidate alongside CI). The edits here change
descriptions only, so it stays green. Run the suite after this task (P4).

## T5 — Freeze docs/plans/

Add a status header (blockquote, first line under the title) to every file,
and neutralize the two live agent-directives. Exact headers:

| File | Header |
|---|---|
| `2026-02-19-newsletter-classification-design.md` | `> **Status: executed Feb 2026; superseded in parts.** The shipped system diverges: JSONL write-before-label (inverse of the error table here; see D18), Poor/OK/Good + Emphasized-only rubric (D14/D15, not 1–5 + union), all-grades-errored is a failure (D20/D5). Historical record — do not implement.` |
| `2026-02-19-newsletter-classification-plan.md` | `> **Status: executed Feb 2026; superseded (see the design doc's header). Do not implement.**` — and the `> **For Claude:** REQUIRED SUB-SKILL…` line at line 3 is **deleted**. |
| `2026-02-20-newsletter-tui-design.md` | `> **Status: executed with major divergence; superseded.** Shipped as newsletter_review/tui.py (Textual, after a curses interlude — see 2026-07-03-tui-framework-evaluation.md); tui_data.py and the [project.scripts] entry point never existed; the t/h cycle-filter UX became the f filter menu. Do not implement.` |
| `2026-02-20-newsletter-tui-plan.md` | `> **Status: executed; superseded (see design doc header). Do not implement.**` — and its `For Claude:` directive line is **deleted**. |
| `2026-07-01-newsletter-eval-plan.md` | `> **Status: executed July 2026.** Correction: this doc says the pipeline is "active under NEWSLETTER_ONLY=1"; grading is actually enabled by [newsletter] config presence (D1).` |
| `2026-07-03-tui-framework-evaluation.md` | `> **Status: executed (decision adopted; migration done via issue #43). Historical record.**` |
| `2026-07-08-issue-roadmap.md` | `> **Status: frozen snapshot as of 2026-07-08 — GitHub issues are the living tracker.** Known-shipped since: #53, #52, #30, #35, #36, #41-triage (see the phase decision docs). Priorities and "current status" figures are historical.` |
| `2026-07-08-phase1-decisions.md` | `> **Status: executed (see per-item DONE markers). Decisions recorded here are superseded by docs/decisions.md where they overlap.**` |
| `2026-07-09-phase2-decisions.md` | `> **Status: partially executed** — #13 tooling/#24/#15 shipped; **#28 Option A was never implemented** (now registry D6, scheduled Wave 2).` |

This plan file itself already carries the convention's header.

## T6 — `docs/newsletter-label-ux-redesign.md`

Insert after the title:

```markdown
> **Status: partially superseded (2026-07-30).** Decision 1 (auto-seed on
> open) was reversed by issue #59 — the tool is manual-only; the `r` re-seed
> hotkey and "Auto-seed safety" notes below describe removed behavior.
> Decision 2 (remove story titles end-to-end) and the redesigned span-edit UX
> still reflect the shipped design.
```

## T7 — Acceptance

1. `uv run --extra dev pytest tests/ -v` fully green (expected: doc edits
   touch no asserted content; `test_env_var_docs`, `test_tui_docs`,
   `test_eval_cli_docs`, `test_newsletter_eval_docs` are the ones to watch).
2. `grep -rn "CLAUDE" tests/` to confirm no test parses CLAUDE.md before
   relying on that (expected: none — audit finding "Meta-tests guard
   README-technical only").
3. Cross-check: no doc states pending-D5 behavior as current (P2 spot-check
   of CLAUDE.md §4, README.md Resilience, registry statuses).
4. Audit findings this wave resolves (by catalog title), for the record:
   *No unified purpose statement · Newsletter pipeline: mode vs always-on
   branch (all variants) · Privacy invariant unqualified vs cloud newsletter
   bodies · Absolute invariant vs SERVICE-default · MLX public-API stand-in ·
   NEWSLETTER_ONLY mis-stated in CLAUDE.md · LLM-settings block drifted ·
   Stale LLM settings block · copied-vs-shared email-agent files · 2026-02-19
   design doc contradicts current behavior · 2026-02-20 TUI docs describe
   nonexistent modules · docs/plans status unstated · Plan/decision docs
   carry no completion status · UX-redesign auto-seed · No non-goals stated ·
   VIP-sender feature absent from CLAUDE.md (resolved by pointer: the env
   table that documents it becomes the single home) · Facts duplicated 3-4x
   (the CLAUDE.md copies) · Project-structure listings omit modules (the
   CLAUDE.md copy; README-technical backfill deferred to Wave 3).*
5. Remaining waves are unblocked: Wave 0.5 (open-issue triage against the
   registry — e.g. #26 is reversed by D5, #28 is scheduled as D6, #29/#33
   absorb into Wave 2, #39 is reframed by D7), Wave 1 (privacy tests,
   CI/D13), Wave 2 (D3 exact-match, D5 corollaries, D6, D10, D11, D12),
   Wave 3 (sweep of the Phase 1 findings catalog).

## Execution record (2026-07-30)

Executed T1–T6 in order, one commit each, suite green after every task
(constant known-failing baseline: issue #67's timezone test). A three-agent
acceptance pass (completeness / P2-honesty / coherence) ran before this
header flipped; its fixes landed in the final commit.

Accepted deviation from the plan text: CLAUDE.md's identity section carries
the D1 `INBOX_ONLY` sentence (added during plan verification to the registry
entry only); it is registry-consistent and kept.

Wave 3 carry-list from acceptance (out of this wave's docs-only scope):
- `classifier.py:6` and `daemon.py:6` module docstrings still state the
  absolute pre-D2 invariant ("NEVER leave the local network");
  `daemon.py:577` references "graceful degradation of the privacy invariant".
- `pyproject.toml:4` description still states the email-only identity D1
  replaced.
- README-technical's env table restates config.toml defaults; README.md
  restates the D15 tier bands and quotes literal counts/log lines — the
  remaining D7 altitude/duplication sweep (D7's status is scoped
  accordingly).
