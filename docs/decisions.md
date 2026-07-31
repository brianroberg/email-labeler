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

**Status:** implemented (docs, Wave 0; test enforcement, Wave 1).

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

**Status:** implemented (docs Wave 0; exact-address matching Wave 2).

A thread with any message To/Cc-addressed to the configured newsletter
recipient belongs to the newsletter function and is organizational content —
its full transcript, **including person-written replies**, goes to the cloud
LLM by design, before and instead of person/service routing. The recipient
match is an exact address comparison.
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
  (README-technical's write-before-label table currently documents the
  give-up ending, matching the code; this corollary changes both).
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

**Status:** implemented for the restated settings blocks (Wave 0); the
remaining altitude/duplication sweep (env-table defaults, README literals and
quoted log lines) lands with Wave 3.

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

**Status:** implemented (Wave 2).

`evals/web_app.py`, `web_auth.py`, `web_data.py`, `run_web.py`,
`evals/templates/`, and evals/README.md §5 are removed; `fastapi`, `jinja2`,
`uvicorn`, and `python-multipart` leave the runtime dependencies
(`python-multipart` was web-only in fact — FastAPI form parsing — though
this entry originally omitted it; the `multipart` strings in gmail_utils.py
are MIME types, not the package). Coupled removals the implementer
must include or the suite fails: drop `evals.run_web` from
`tests/test_eval_cli_docs.py`'s `_CLI_MODULES` (it imports the module), and
delete the `### run_web` section of evals/README-technical.md. Implementation
found two doc edits beyond this list: the "plus a web UI" clause in
evals/README.md's intro and the web-UI sentence in README-technical's
Chain-of-Thought section (the tui-regression skill's venv pip line also
dropped the four packages). Rationale:
forgotten by the owner, workflows CLI-covered, zero tests, and it contradicted
the no-web-server posture at the dependency level. CoT capture/sidecars are
unaffected.

## D11 — Minimal release identity (2026-07-30)

**Status:** implemented (Wave 2).

The Dockerfile bakes the git SHA (build-arg), the daemon logs it once at
startup, and notable releases get a lightweight git tag. No semver, no
changelog — just enough that "what is deployed?" has an answer in the logs.

## D12 — Assessments JSONL: documented schema + version field (2026-07-30)

**Status:** implementation pending (Wave 2).

The assessments JSONL (the only durable copy of gradings) gets a schema
document (home: README-technical) and a `schema_version` field on each record,
so the next schema evolution is a version bump + migration, not shape-sniffing.

## D13 — Adopt CI (2026-07-30)

**Status:** implemented (Wave 1).

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
