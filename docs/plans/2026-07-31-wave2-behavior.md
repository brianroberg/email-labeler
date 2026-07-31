# Wave 2 — Behavior changes under the new policy

> **Status: proposed (2026-07-31); owner-approved 2026-07-31 — execution
> pending.** Drafted from the Wave 0.5 triage feed on issue #68; adversarially
> verified (6-lens, 56 findings → corrections applied); supersedes nothing.
> Flips to `executed` with the T14 execution record (D8).

Fourth step of the 2026-07-30 clarity effort, after Wave 0 (docs foundation),
Wave 0.5 (issue triage), and Wave 1 (enforcement) — all executed. Wave 2 is
the **behavior wave**: every task changes production behavior to match the
policy the registry now records. Scope, from the issue-#68 roadmap plus the
Wave 0.5 planning feed: D3 (exact-address newsletter match), the D5
corollaries, D6 (proxy 403), D10 (eval web-app removal), D11 (release
identity), D12 (assessments schema versioning), and the absorbed #29+#33
write-path bundle with its ⚠ ordering interaction. The findings-catalog sweep
stays with Wave 3.

## Goal

Make the code obey the failure model and the registry:

1. **Rule 1 holds everywhere.** No failure path commits an outcome: keyword-free
   label replies raise (T10), false `no-stories` paths become failures (T11),
   a rejected write is a human answer, not a strike (T7).
2. **Rule 2 replaces per-exception blame.** Provider-shaped faults (exhausted
   429/5xx, LLM and proxy) never count toward give-up; thread-attributable
   faults are counted only when cycle-level correlation says the thread is the
   problem; the masquerade case gets a repeated, distinct escalation instead
   of silent abandonment (T8).
3. **Functions fail independently.** A provider-balance halt stops one
   function, loudly, while the other continues (T9).
4. The system stops paying for its own failures: label-write faults no longer
   discard finished classifications (T6), and the write bound means what
   config.toml says it means (T5).
5. The non-failure-model registry debts land: D10, D3, D11, D12 (T1–T4).

## Principles (binding on every task)

- **P1 — Red/green TDD.** Every behavior change starts with a failing test
  observed red for the expected reason (CLAUDE.md Testing). Each task section
  names its RED test(s). Exception, stated openly: T1 is a pure removal with
  no new behavior to test — its gate is the coupled-removal checklist plus the
  full suite staying green (the D10 entry names the couplings).
- **P2 — One commit per task, in order; full suite + `ruff check .` after
  every task.** CI's TZ matrix runs on the PR; local runs may be single-TZ.
- **P3 — Registry statuses flip in the implementing commit** (Review
  Charter). D5's corollaries flip individually: each task that lands one
  edits that corollary's `implementation pending (Wave 2)` marker in the same
  commit. Tests that pin behavior a task reverses are updated in that same
  commit, citing the entry (they are not "broken tests"; they pinned the old
  decision).
- **P4 — Current-behavior docs move with the code.** README.md (labels table,
  Resilience), README-technical (write-before-label, sink preflight, Health
  Checking), config.toml comments, and in-code docstrings that describe the
  old behavior are rewritten in the same commit as the behavior change.
  CLAUDE.md's pending-notes are removed as their subjects land — and no
  earlier: a note that still covers unlanded corollaries stays until the
  last of them lands.
- **P5 — What stays fixed.** D2's unparseable-Stage-1 → SERVICE default
  (classifier.py:110-115) and the never-count / quiet-local-deferral
  **semantics** of the LLMUnavailableError arm (daemon.py:574-590) are
  already the model. No task changes those behaviors, and
  tests/test_privacy.py plus the TestParseSenderType defaults must stay green
  untouched throughout. (T8c does add masquerade *bookkeeping* to the
  cloud/tier-less branch of that arm — bookkeeping only; the arm still never
  counts toward give-up and the local tier stays quiet.)
- **P6 — Ordering constraint (Wave 0.5 ⚠).** T6 (result reuse) lands
  **before** T8 (never-count 429/5xx). T8 removes the `max_failures` bound on
  a persistent write-phase fault; without T6 that would mean unbounded
  re-classification spend. The order in the task list is normative.

## Task list

| # | Task | Registry | Files touched (production) |
|---|---|---|---|
| T1 | Remove the eval web app | D10 | `evals/web_*.py`, `evals/run_web.py`, `evals/templates/`, `pyproject.toml`, `uv.lock`, `.claude/skills/tui-regression/SKILL.md` |
| T2 | Exact-address newsletter recipient match | D3 | `newsletter.py` |
| T3 | Release identity: SHA build-arg, startup log, tags | D11 | `Dockerfile`, `daemon.py` |
| T4 | Assessments schema doc + `schema_version` | D12 | `newsletter.py`, `scripts/migrate_assessments.py` |
| T5 | `write_sem` guards single writes | #33 | `daemon.py`, `labeler.py` |
| T6 | Classification result reuse across cycles | #29 | `daemon.py` |
| T7 | Proxy 403 is a human answer | D6 | `daemon.py` |
| T8 | Failure attribution: never-count + correlation + masquerade escalation | D5 (429/5xx + correlation corollaries) | `llm_client.py`, `proxy_client.py`, `daemon.py`, `config.toml` (comment) |
| T9 | Per-function halt | D5 (halt corollary), D19 | `daemon.py` (+ docstrings in `llm_client.py`, `newsletter.py`) |
| T10 | Keyword-free label reply raises | D5 (Rule-1 corollary) | `classifier.py` (+ comment touches in `llm_client.py`, `config.toml`) |
| T11 | `no-stories` only from successful zero-story extraction | D5/D20 | `newsletter.py` |
| T12 | Sink faults never counted, retried forever | D5 (sink corollary) | `daemon.py`, `newsletter.py` |
| T13 | `MAX_FAILURES` knob | D5 (knob corollary) | `daemon.py`, `config.toml` |
| T14 | Acceptance | — | — |

Ordering rationale: T1–T4 are independent, low-risk registry debts — land them
first so the failure-model cluster sits on a smaller, cleaner base (T1 alone
removes four runtime deps). T5 restructures the write path mechanically before
T6 builds on it; T6 must precede T8 (P6); T7 is small and benefits from T6
(a gated thread re-offers without re-classifying). T8 is the core redesign;
T9–T13 land the remaining corollaries on top of its shapes.

---

## T1 — D10: remove the eval web app

Pure removal, per the D10 entry plus recon deltas the entry does not name.

Checklist (all in one commit):

1. `git rm evals/web_app.py evals/web_auth.py evals/web_data.py
   evals/run_web.py` and the five files under `evals/templates/`.
2. pyproject.toml: drop `fastapi`, `jinja2`, `uvicorn` **and
   `python-multipart`** — the latter is web-only in fact (FastAPI form
   parsing, web_app.py:63) but unnamed in D10; the D10 entry is amended in
   this commit to name it (Review Charter: extending an entry updates it in
   the same change). The `multipart` strings in gmail_utils.py are MIME
   types, not the package — untouched.
3. Regenerate `uv.lock` (`uv sync --extra dev`) in the same commit — CI's
   `--locked` and the Dockerfile's `--frozen` both fail on a stale lock.
4. tests/test_eval_cli_docs.py: delete the `"evals.run_web": "### run_web"`
   entry from `_CLI_MODULES` (the test imports the module at run time via
   importlib — without this, the test errors with ModuleNotFoundError
   instead of failing a doc assertion).
5. evals/README-technical.md: delete the `### run_web` section (:118-123)
   **and** the web-UI sentence at :478 in the Chain-of-Thought section (the
   rest of that section describes capture/sidecars and stays).
6. evals/README.md: delete `## 5. Web UI — Interactive reporting and
   comparison` (:142-156) **and** the "plus a web UI for browsing results"
   clause in the intro (:3). Renumber later sections only if the file's own
   numbering demands it.
7. docs/decisions.md: flip D10 to `implemented (Wave 2)`, amending the entry
   per item 2 and noting the two doc edits found beyond its list.
8. `.claude/skills/tui-regression/SKILL.md:44` still pip-installs the web
   deps in its throwaway venv — trim the four packages from that line
   (mandatory: the acceptance grep below covers this file).

Do **not** touch: docs/plans/ mentions (frozen history, D8); `.env.example`
(no web vars — `EVAL_WEB_SECRET` is invisible to both env guards, verified);
root README.md / README-technical.md / CLAUDE.md (zero references);
Dockerfile (web code was never copied into the image; only the lock regen
matters). CoT capture/sidecars are unaffected (verified: `load_thinking_sidecar`
is web-only; the write path lives in run_eval.py / newsletter_run.py).

**Gate (P1 exception):** full suite green + `git grep -i -E
"fastapi|uvicorn|jinja2|run_web|web_app|web_auth|web_data"` returns hits
only in `docs/plans/` (frozen history, including this plan) and
`docs/decisions.md` (the D10 entry legitimately records what was removed).

## T2 — D3: exact-address newsletter recipient match

`is_newsletter` (newsletter.py:486-498) currently does a lowercased
**substring** test against the raw To/Cc header value — the convicted bug:
`newsletters@dm.org` matches `abcnewsletters@dm.org` and any address whose
display name merely contains the recipient. One definition, two production
consumers (daemon.py:409, evals/newsletter_harvest.py:145) — fixing the
function fixes both.

**RED tests** (tests/test_newsletter.py::TestIsNewsletter — existing fixtures
all use bare exact addresses, so none pin the substring behavior; new tests
are needed to convict it):

- `test_superstring_address_does_not_match`: To =
  `abcnewsletters@dm.org` → False (red today: substring matches).
- `test_display_name_containing_recipient_does_not_match`: To =
  `"newsletters@dm.org fan club" <other@example.org>` → False (red today).
- `test_bracketed_display_name_form_matches`: To =
  `Newsletter Desk <newsletters@dm.org>` → True (green today via substring —
  include it anyway: it pins the form the rewrite must keep working).
- `test_multiple_recipients_with_display_names_match`: comma-separated list
  where the recipient appears bracketed mid-list → True.

**Implementation:** parse each header value with `email.utils.getaddresses`
and compare casefolded addr-specs for equality with the casefolded
configured recipient. (`getaddresses` is a new name from the already-imported
`email.utils`; the repo has no To/Cc address-*list* parsing today —
classifier.py's `parse_sender` regex handles single From headers and is
untouched. `getaddresses` is chosen because it handles comma-separated lists,
display names, and quoting.) Keep: thread-level OR over messages, To and Cc,
case-insensitivity (test_case_insensitive pins it), first-header-only
`get_header` semantics (pre-existing, unchanged). The Gmail-side `to:` query
prefilters (daemon.py:928-929, newsletter_harvest.py:106) use Gmail's own
address matching and stay as prefilters; `is_newsletter` remains the sole
decider.

Same commit (P4): flip D3 to `implemented (Wave 2)` (drop the "convicted
bug" clause); remove README.md's current-behavior parenthetical in the
privacy section (:28-30, "Today the recipient match is a substring check …
pending, decision D3"). Existing tests untouched (recon-verified
exact-match-safe, including test_privacy.py's newsletter-bypass test, which
was written for this change).

## T3 — D11: release identity

Three parts, per the entry:

1. **Dockerfile:** `ARG GIT_SHA=unknown` + `ENV GIT_SHA=${GIT_SHA}` inserted
   after the code/config COPY block (lines 15-16) so a SHA change never busts
   the dependency layer. The default keeps un-arg'd builds working — the
   build is driven by the external agent-stack compose file this repo does
   not control. README-technical:352-357 quotes the HEALTHCHECK block
   verbatim; the insertion sits above it and must not alter the quoted lines.
2. **Startup log:** one line at the top of `run_daemon()` (daemon.py:816ff):
   `log.info("email-labeler starting — build %s", os.environ.get("GIT_SHA", "unknown"))`.
   Inside `run_daemon`, not `main()`: the test harness
   (`run_poll_cycles`, test_daemon.py:1014-1065) exercises `run_daemon`
   directly, and module import must stay logging-silent (daemon.py:64-77
   contract).
3. **Tags:** no automation. README-technical gets a two-line procedure note
   (with the Docker/build-arg note in README.md's "### Docker (via
   agent-stack)" section, :187-197): notable releases get
   `git tag deploy-YYYY-MM-DD <sha>` — the owner's manual action. No semver,
   no changelog (D11).

**RED test:** `test_startup_logs_build_sha` in test_daemon.py — caplog
assertion through the `run_poll_cycles` harness with
`monkeypatch.setenv("GIT_SHA", ...)`; red before the log line exists.
`tests/test_env_var_docs.py` auto-discovers the new `os.environ.get("GIT_SHA")`
read and fails until the README-technical env table documents it — that
guard firing and then passing is part of the same red/green arc. Env-table
row: optional, default `unknown`, "stamped by the image build; not an
operator knob". Not added to `.env.example` (optional vars are deliberately
unasserted there).

Same commit: flip D11 to `implemented (Wave 2)`.

## T4 — D12: assessments schema doc + `schema_version`

**Schema doc:** new `#### Assessment record schema` under README-technical's
"Newsletter settings", beside the existing migration (:247) and
write-before-label (:285) H4s. Contents (all recon-verified against
`write_assessment`, newsletter.py:254-299):

- Field table: `timestamp` (processed time, ISO-8601 UTC, always),
  `schema_version` (int, from this change), `message_id`, `thread_id`,
  **`from`** (the JSON key is the reserved word, not `sender`), `subject`,
  `send_date` (email-intrinsic, ISO-8601 UTC or null), `model` (or null),
  `overall_tier` (tier string or null), `stories[]` with
  `text` / `scores` (dict of dimension → 1|2|3, or null) / `average_score` /
  `tier` / `themes` / `quality_cot` / `theme_cot`. Two semantics that must be
  stated: a theme **absent** from `stories[].themes` means grade Absent
  (absence-by-omission, D14/D15), and `migrated_from: "pre-#53"` marks
  records converted by the migration script. `send_date`/`model` enter the
  doc per the Wave 0.5 feed (issue #35's fields).
- Version semantics: `schema_version: 1` = this documented shape, with one
  stated carve-out: records stamped v1 by the migration script (bearing
  `migrated_from`) may lack `send_date`/`model` entirely — those keys
  postdate the records being migrated, and the migration deliberately does
  not fabricate them. **Absence of `schema_version` means a pre-versioning
  record**, of which two shapes exist: post-#53 current-shape (possibly
  lacking `send_date`/`model`, which arrived without a bump) and pre-#53
  legacy (list themes / 1-5 scores — readable only after
  `scripts/migrate_assessments.py`). Files may mix versions (rescue-copy
  concatenation is expected; dedup is by `timestamp` per `thread_id`, D18) —
  readers keep `.get()` tolerance.
- Pointer to the dedup rule and the migration table, not restatements (D7).

**Code:** `write_assessment` adds `"schema_version": 1` to the record;
`migrate_record` stamps the same **only on records it converts** —
new-scheme pass-throughs stay byte-identical
(`test_new_scheme_record_passes_through_unchanged`,
tests/test_migrate_assessments.py:175, pins this, along with the
counts/no-op-file assertions at :191 and :228). All readers are
`.get()`-tolerant (recon-verified: review TUI, migration sniffing,
`count_records`) — no reader changes.

**RED tests:** extend `test_writes_jsonl_record`
(tests/test_newsletter.py:602) to assert `schema_version == 1` (red first);
a migration test asserting the stamp on converted output and its absence on
pass-throughs (red first). The tui-regression skill's synth_data.py is a
fourth hand-maintained shape copy — optional alignment, outside the suite;
note it, don't gate on it.

Same commit: flip D12 to `implemented (Wave 2)`.

## T5 — #33: `write_sem` guards single writes

Today five `daemon.py` sites hold `write_sem` across a whole LabelManager
method — each an N-message serial `modify_message` loop with a 300 s
human-approval write timeout per call — so the bound is "4 threads writing",
not "4 writes in flight". config.toml's authoritative comment (:26-30)
already describes per-write granularity; this task makes it true. Adopting
issue #33's preferred option: **LabelManager owns the semaphore.**

- `LabelManager.__init__` takes `write_sem: asyncio.Semaphore | None = None`;
  every `await self.proxy.modify_message(...)` (labeler.py:142, :158, :217)
  acquires it per call (`nullcontext` when None, preserving the existing
  tests' unbounded default).
- Remove the five daemon acquisition sites (daemon.py:243, :466, :499, :554,
  :560) and the `write_sem` parameters threaded through
  `process_single_thread` and `_give_up_if_stuck`; `run_daemon` passes the
  semaphore to the LabelManager constructor instead.

**RED test:** in tests/test_labeler.py — real LabelManager, mock proxy,
`Semaphore(1)`, two concurrent multi-message applies: assert interleaving is
per-message (the slot is released between messages). Red today (can't be:
LabelManager doesn't accept a semaphore — the constructor change itself is
test-driven). The daemon-level pin
`test_label_application_is_bounded_by_write_sem` (test_daemon.py:728-754)
tests the old plumbing; rework it to construct the real LabelManager with an
exhausted semaphore (its classify-completes-before-write-blocks assertion
stays meaningful and must stay true). Test call sites passing `write_sem=`
into `process_single_thread` are updated mechanically in the same commit.

Same commit: close-comment material for #33 (owner closes on merge);
config.toml comment now truthful — cite it in the commit message rather than
editing it.

## T6 — #29: classification result reuse across cycles

Today a transient write-phase fault discards the finished classification and
re-runs Stage 1 + Stage 2 (the scarce local GPU pass) every cycle while
writes fail — and since #26, five such cycles abandon a **fully classified**
thread to `agent/attempted`. Issue #29 offers in-cycle write retry or result
reuse; recon settles it: with `WRITE_TIMEOUT = 300s` and human approval in
the loop, in-cycle retries are expensive and approval-duplicating, and
transport faults get zero HTTP-layer retries anyway. **Result reuse**, cache
session-scoped like everything else:

- `ResultCache` (daemon.py, beside FailureTracker): `thread_id →
  (fingerprint, payload)`. Fingerprint = the sorted tuple of the thread's
  message ids — a new message changes the input, so a mismatched fingerprint
  drops the entry and classifies fresh (staleness answer). Pruned each cycle
  to the still-pending set, mirroring FailureTracker.prune; cleared on
  successful label write.
- Email payload: `(label, sender_type)` recorded after Stage 2 succeeds. On
  a later cycle with matching fingerprint: skip Stage 1/Stage 2, keep the
  fresh `get_existing_priority` no-downgrade check (it reads the fresh
  thread), go straight to the label write.
- Newsletter payload: `(best_tier, all_themes, story_results,
  assessment_written)` — the graded StoryResult list must be cached because
  a sink-fault retry rebuilds the JSONL record from it;
  `assessment_written` is set once `write_assessment` returns. Reuse skips
  re-extraction/re-grading; skips the JSONL append when `assessment_written`
  (for the same fingerprint the write never repeats; a fingerprint
  invalidation legitimately re-grades and re-appends, with D18's
  newest-timestamp dedup as the read-side semantics). A sink-fault retry
  therefore re-attempts **only** the write. Write-before-label ordering is
  preserved by construction.

**RED tests** (both fail today — recon confirmed no test injects a label-write
failure after successful classification):

- `test_write_failure_then_retry_does_not_reclassify` (email): first
  `process_single_thread` call classifies, `apply_classification` raises
  `ProxyUnavailableError`; second call (same thread data) succeeds. Assert
  the LLM mocks were called exactly once across both cycles and labels
  landed.
- `test_newsletter_write_failure_reuses_grading_and_does_not_reappend`:
  same shape on the newsletter path; assert one `classify_newsletter` call
  and exactly one JSONL record.
- `test_new_message_invalidates_cached_result`: second cycle's thread carries
  an extra message → LLM mocks called again.

Same commit (P4): update the README-technical write-before-label table's
"Labels fail after a successful write" row (it documents the re-grade +
second-record behavior this task removes) and README.md's Resilience bullet
if it restates the discard; rewrite the now-stale "#29" clause in
`_give_up_if_stuck`'s ProxyUnavailableError comment (daemon.py:252-254 —
this task implements exactly what that clause defers). #29 close-comment
material for the owner.

## T7 — D6: proxy 403 is a human answer

`ProxyForbiddenError` (a plain Exception, deliberately not a ProxyError —
that hierarchy must not change: `verify_labels_with_retry` depends on it to
treat a startup 403 as permanent, daemon.py:735-746) is caught nowhere in
the per-thread path today; a rejected write falls into `except Exception` →
traceback + strike → after five rejections the operator's "not now" becomes
`agent/attempted`. D6: log one clean line, count nothing, re-offer next
cycle. Implementing issue #28's Option A:

- New arm in `process_single_thread`, anywhere before `except Exception`:
  `except ProxyForbiddenError as exc:` → one INFO line
  ("write rejected/blocked by the proxy for thread X — re-offering next
  cycle"), `return False`. No strike, no marker, no traceback.
- Same treatment inside `_give_up_if_stuck`'s marker write (a 403 on the
  `mark_attempted` write currently lands in its generic `except Exception`
  with a full traceback): clean line, `return False` — the rejection blocks
  the marker, the thread stays give-up-eligible, and the marker re-offers
  next cycle. (A rejection still never *causes* the give-up — the strikes
  that reached the threshold came from elsewhere.)
- With T6 in place, the re-offer skips re-classification: a gated thread
  costs one write attempt per cycle, not two LLM calls.
- `verify_labels_with_retry`'s startup treatment of ProxyForbiddenError as
  permanent stays (a blocked op at startup is a config error, not a gated
  write).

**RED tests:** daemon-level 403 through `process_single_thread` (none exists
today, per recon): `test_rejected_write_defers_without_strike_or_traceback` —
`apply_classification` raises ProxyForbiddenError; assert return False, no
`record_failure` effect (drive it `max_failures` times and assert no
`mark_attempted`), no `log.exception` record. And
`test_rejected_marker_write_re_offers` for the `_give_up_if_stuck` arm.

Same commit: flip D6 to `implemented (Wave 2; issue #28 Option A)`; #28
close-comment material. README.md Resilience gets one sentence (a rejected
write is re-offered, never abandoned).

## T8 — D5 core: never-count 429/5xx, correlation attribution, masquerade escalation

The wave's centerpiece; the "small design" D5's correlation corollary calls
for. Three coordinated changes:

### 8a. Provider-shaped errors become typed and never count

- **llm_client:** the non-200 handler (llm_client.py:266-281) currently
  raises bare `RuntimeError` for everything non-balance — so an exhausted
  429/5xx is indistinguishable from a 400 and strikes via the daemon's
  RuntimeError arm. Change: after the balance check, `status == 429 or
  status >= 500` → raise `LLMUnavailableError(..., tier=self.tier)` (the
  provider can't serve anyone right now — same class the transport faults
  already use, carrying the tier the daemon's quiet-local handling needs).
  Other non-200s (400/401/403/404/422…) stay `RuntimeError`:
  request-specific, strike-candidates.
- **daemon:** the `ProxyUnavailableError` arm (daemon.py:591-610) stops
  calling `_give_up_if_stuck` — defer only, plus provider-shaped bookkeeping
  (8c). Its issue-#26 rationale comment is rewritten to the D5 rule, as is
  the FailureTracker docstring's "deterministic per-request proxy 5xx
  (issue #26)" clause (daemon.py:148-156). **This deliberately reverses
  issue #26**, which the D5 entry records; the reversal note stays in the
  entry.
- **Docstring/comment rewrites, same commit (P4):** llm_client's
  `LLMUnavailableError` class docstring "non-200 … is eligible for the
  daemon's give-up logic" framing (:58-61), `complete()`'s Raises section
  (:194-207), the `_BALANCE_SIGNATURE_STATUSES` comment's closing clause
  "fall back to per-thread give-up" (:103-108 — after 8a a 429-signaled
  out-of-funds is retried as unavailability, surfacing via the shared-cause
  ERROR/masquerade escalation, not give-up; the D19 rationale survives in
  new terms), and `LLMContentError`'s "the email pipeline's `except
  RuntimeError` give-up handler catches it unchanged" clause (:82-86 — the
  arm becomes a collector append under 8b). In proxy_client: the docstrings
  D5 says "become true" — `ProxyUnavailableError` (:50-57), `_send`
  (:183-189), `_handle_response` Raises (:137-139), the `ProxyError` class
  docstring's give-up split (:37-44), the non-JSON-2xx comment's "a
  persistent one is still bounded by the FailureTracker" clause (:175-180),
  and the `TRANSIENT_TRANSPORT_ERRORS` exclusion comment (:62-74) — most
  become true untouched; the give-up clauses need rewording. In config.toml:
  the `[llm.local]` comment's "turns every over-budget thread into an
  eventual give-up" clause softens to the correlation-era truth (co-failing
  same-signature threads share cause and are kept).

### 8b. Correlation decides strikes for thread-attributable candidates

Candidate classes (may strike): `TimeoutError`, `RuntimeError` (incl.
`LLMContentError`), bare `Exception`. Provider-shaped classes (never
strike): `ProxyUnavailableError`, `LLMUnavailableError`, the defensive
`httpx.ConnectError`.

Mechanics — cycle-level, per the roadmap's wording:

- `process_single_thread` gains a per-cycle collector parameter (the
  `local_deferrals` pattern) **which replaces its `failure_tracker`
  parameter** — the tracker becomes poll-loop-owned, fed only by the
  post-gather attribution step. Each candidate arm appends
  `(thread_id, ids_to_mark, signature)` — signature = the exception's class
  qualname — and returns False. Each provider-shaped arm appends a
  provider-shaped entry (8c). No inline `_give_up_if_stuck` calls remain.
- Post-gather, the poll loop runs the attribution sequence, strictly in
  this order: **attribute → strike → mark → summarize → cycle log.**
  A candidate failure **counts a strike iff its signature is unique among
  the cycle's candidate failures AND (the cycle contained other threads →
  at least one of them was handled successfully)**. Marking eligibility
  derives from *this cycle's collector entries* that just struck out —
  never from raw tracker counts (a stale at-threshold count left by a
  failed marker write must not mark a thread that later succeeded).
  Marking reuses `_give_up_if_stuck`'s guarded-write body (its
  ProxyUnavailable-during-marker handling, its log-after-write-lands
  discipline, and T7's Forbidden arm are preserved as post-T6/T7 text; the
  function loses its `record_failure` head, which attribution now owns) and
  performs `record_give_up` + count-clear beside the successful mark.
  `summarize_cycle` keeps success-clears; its docstring and the
  "Processed %d/%d (%d abandoned)" line's arithmetic are rewritten in the
  same commit — give-ups no longer return True from `process_single_thread`,
  so `given_up` stops being a subset of the handled count.

Adjudicated edges, recorded here and in the D5 entry when flipped:

- **Singleton cycles count.** A lone thread's candidate failure has no
  siblings to correlate against; bounded strikes to a findable
  `agent/attempted` is the honest fallback — and the poison-thread case
  *is* typically a singleton (everything else processed away). Residual: a
  code bug that fails the only pending thread accrues strikes; accepted.
- **Zero-success multi-thread cycles count no strikes.** Two threads
  failing with *different* signatures while nothing succeeds is still
  consistent with a shared cause surfacing through two code paths (Rule 2
  conditions thread-blame on siblings *succeeding*); all failures are
  treated shared-cause, one ERROR line says so, the backlog is kept.
- **N same-signature poison threads shield each other** (e.g. three
  oversized-transcript timeouts in one cycle): no strikes while they
  co-fail. Accepted per Rule 2's plain text; the shared-cause ERROR line is
  the loudness, the backlog stays.

### 8c. The masquerade case escalates on the heartbeat

The single-thread masquerade — provider-shaped errors on one thread while
siblings succeed (issue #26's poison-thread scenario) — retries forever,
never abandoned, but must not be silent:

- A `MasqueradeTracker` (shape of FailureTracker: per-thread counter,
  success-clears, pruned to pending): incremented for a thread whose cycle
  failure was provider-shaped **while no other thread failed
  provider-shaped in that cycle AND at least one sibling was handled
  successfully** — the entry's own definition ("provider-shaped errors,
  siblings succeeding"). A singleton cycle or a zero-success cycle neither
  increments nor resets: there is no correlation evidence either way, so a
  genuine short provider outage with one pending thread never false-alarms
  (the per-thread WARNING each cycle remains its visibility; the counter
  resumes counting when siblings reappear). **Local-tier
  `LLMUnavailableError` is excluded entirely**: the deliberately-offline
  MLX laptop makes "person threads defer while service siblings succeed"
  the *routine* local state (issue #24's quiet-deferral design) — tracking
  it would false-alarm every night the laptop is closed.
- At `max_failures` qualifying cycles the thread becomes a suspect, and the
  poll loop emits a distinct ERROR — "thread X has failed with
  provider-shaped errors across N cycles while siblings succeed; retrying
  forever per the failure model — investigate the thread or the provider
  route" — **repeated at most once per `status_interval`** while any
  suspect persists (`idle_report` only runs on idle cycles, so this is its
  own small throttled emitter called after the attribution sequence; the
  existing give-up ERROR's log-once-after-write discipline is untouched).

**RED tests** (new class in test_daemon.py, driving `process_single_thread`
plus the attribution function):

- `test_exhausted_429_raises_unavailable_not_runtime` (test_llm_client.py;
  red — today it's RuntimeError).
- `test_proxy_unavailable_never_counts_toward_give_up` — the reversal of
  `test_proxy_unavailable_is_give_up_eligible_per_thread` (:524), rewritten
  in place citing D5/#26.
- `test_same_signature_failures_in_one_cycle_count_no_strikes` — two threads
  raising the same RuntimeError → no counts, shared-cause ERROR logged.
- `test_unique_signature_failure_with_succeeding_siblings_strikes` — one
  ValueError among successes → count recorded; at threshold → attempted.
- `test_singleton_cycle_candidate_failure_strikes` and
  `test_zero_success_cycle_counts_no_strikes` — pin the adjudicated edges.
- `test_masquerade_suspect_escalates_on_heartbeat_and_is_never_abandoned` —
  provider-shaped failure with succeeding siblings for `max_failures`
  cycles → ERROR emitted, repeated after `status_interval`, thread never
  marked; variants assert no escalation for the local tier and no increment
  in singleton cycles.

Tests updated citing the registry (enumerated — this is the wave's largest
test rework):

- test_llm_client.py: `test_raises_on_http_error` (:197 — mocks a 500,
  asserts RuntimeError; retarget to LLMUnavailableError, or move its
  RuntimeError pin to a 400) and `test_429_quota_phrasing_stays_runtime_error`
  (:428 — becomes expect-LLMUnavailableError while **preserving its
  not-LLMBalanceError half**, which guards D19's 429-never-halts decision).
- test_daemon.py: `test_proxy_unavailable_is_give_up_eligible_per_thread`
  (:524, reversal); `test_already_at_max_priority_give_up_marks_all_thread_messages`
  (:653 — a second ProxyUnavailable-strikes pin, on the mark_processed
  write; its all-thread-ids assertion is preserved on a candidate-class
  failure instead); `test_gives_up_on_thread_after_repeated_failures`
  (:332) and `test_give_up_marks_all_thread_messages_not_just_query_stubs`
  (:626) rework to the post-gather attribution path;
  `test_give_up_write_transient_failure_logs_clean_warning` (:556) and
  `test_give_up_write_unexpected_failure_keeps_traceback` (:581) follow the
  marker write to its post-gather home (their clean-warning/traceback
  assertions survive); `test_proxy_4xx_is_give_up_eligible` (:603) reworks
  to the cycle-context path, preserving 4xx-ProxyError-is-a-candidate;
  `TestSummarizeCycle` (:867) updates for give-ups-return-False. All test
  call sites passing `failure_tracker=` into `process_single_thread`
  (≈10, e.g. :342, :368, :391, :411, :1758, :2030) update mechanically to
  the collector parameter.

Same commit: flip the two D5 corollaries (429/5xx; correlation/masquerade);
update README.md Resilience and README-technical Health Checking to the new
attribution story; #26 gets a for-the-record comment (owner action; it is
already closed).

## T9 — D5/D19: per-function halt

`DaemonHalt` is daemon-wide; a newsletter-provider balance fault stops email
triage too. `LLMBalanceError` carries no function provenance and the
newsletter client is constructed `tier="cloud"`, so tier cannot split the
functions (recon) — but the *call site* can:

- Halt state becomes a `FunctionHalts` object with two `DaemonHalt` slots
  (`email`, `newsletter`), each first-tripper-wins, restart-only reset. The
  `halt` parameter of `process_single_thread` carries it (one parameter,
  renamed or retyped — the existing halt-passing test call sites
  (`test_balance_error_trips_daemon_halt` :418,
  `test_tripped_halt_short_circuits_before_any_work` :436,
  `test_halt_tripped_mid_fetch_skips_classification` :453, and the
  `_out_of_funds_process` harness :1137) update mechanically and are named
  here for that reason).
- The newsletter branch of `process_single_thread` wraps its LLM work
  (`classify_newsletter` call) in `except LLMBalanceError` → trip
  newsletter, defer the thread. The outer arm (daemon.py:622-629) now serves
  only the email pipeline → trips email. When `[newsletter.llm]` is absent
  the two functions share one client, and a shared-provider fault trips both
  within a cycle or two as each hits its own request — correct: the fault
  disables both functions.
- Skip logic: the pre-fetch short-circuit (:377) fires only when **every
  enabled function** is halted (enabled: newsletter iff configured; email
  iff not NEWSLETTER_ONLY). The current post-fetch re-check (:404) is
  **removed, not repurposed** — it sits *above* newsletter detection and
  cannot know the thread's function; the function-aware checks live below
  routing: a newsletter thread defers (return False, no strike) right after
  detection when the newsletter halt is tripped; a non-newsletter thread
  defers beside the newsletter_only skip (:485-487) when the email halt is
  — before the max-priority check, so no email-function marker writes
  happen under an email halt either.
- Poll loop: full stand-down (:938-957) only when every enabled function is
  halted — message updated to name the halted function(s). A partial halt
  keeps polling; a per-cycle ERROR names the halted function and repeats the
  add-funds-and-restart instruction (the existing repeated-ERROR precedent).
  Healthcheck stays fresh in both states. While **only email** is halted
  (and newsletter is enabled), the poll query is narrowed with the
  NEWSLETTER_ONLY-style `to:{recipient}` clause (daemon.py:928-929 is the
  precedent) so the halted function's backlog doesn't burn a `get_thread`
  per thread per cycle or crowd newsletter threads out of the
  `max_results` page; halts are restart-reset, so the narrowing holds until
  restart. The mirror direction (newsletter halted, email running) accepts
  the residual fetch-and-skip churn — newsletter volume is a trickle and
  the query cannot express "not to:recipient" reliably.

**RED tests:** `test_newsletter_balance_fault_halts_newsletter_only` —
newsletter thread trips the halt; a sibling email thread in the same and the
next cycle still classifies (red today: daemon-wide halt short-circuits it).
`test_email_balance_fault_leaves_newsletter_running` — the mirror, including
the query-narrowing assertion. `test_both_functions_halted_stands_down` —
poll loop stand-down + repeated ERROR (rework of `TestOutOfFundsHalt`, which
pins the daemon-wide behavior and is updated citing D5/D19).

Same commit (P4): flip the halt corollary in D5; update D19's "(today
daemon-wide; per-function under D5, pending)" clause; remove CLAUDE.md's
**intro** pending-note about the daemon-wide halt (:17-19) — the Failure
Model section's general "several are `pending`" paragraph stays until T13
lands the last corollary; reword "the daemon halts" to "the affected
function halts" in llm_client's `LLMBalanceError` docstring (:112-122),
newsletter.py's `classify_newsletter` docstring (:548-554), and the
`_PIPELINE_WIDE_ERRORS` comment (newsletter.py:23-27); update
README-technical's "Out-of-funds halt" subsection and README.md Resilience.

## T10 — D5 Rule 1: keyword-free label reply raises

`parse_email_label` (classifier.py:149-169) silently defaults a keyword-free
Stage-2 reply to LOW_PRIORITY — which config maps to **archive**: a
completely unparseable answer commits label + processed-marker + archive.
The exact Rule-1 violation D5 names; completes the issue-#64 fail-loud arc
(llm_client already raises on empty/truncated replies; this closes the
complete-but-keyword-free gap).

- Raise `LLMContentError` (imported from llm_client — its semantics match
  exactly: unusable answer, request-specific, a strike candidate under T8's
  correlation: one thread's garbage reply strikes toward `agent/attempted`;
  a prompt bug failing every thread the same way correlates to
  shared-cause, no strikes, loud). Docstring rewritten ("Defaults to
  LOW_PRIORITY" dies); the `_EMAIL_LABEL_VALID` / `_SENDER_TYPE_VALID` dead
  constants can go in the same commit.
- **`parse_sender_type`'s SERVICE default is untouched** (D2, P5).
- evals degrade honestly: run_eval catches bare Exception into
  `result.error`, so keyword-free replies become error rows instead of
  silent LOW_PRIORITY predictions. (The reply itself still caches — it is a
  successful HTTP completion; the raise happens downstream in the parser,
  so no re-billing and the error reproduces from cache.)

**RED tests:** rewrite the five default-pinning tests in
`TestParseEmailLabel` (:172-179, :196-201, :223-225 — including the
whole-word-scan test whose asserted outcome is the default) to
`pytest.raises(LLMContentError)`, citing D5; red before the production
change. A daemon-level test asserting a keyword-free reply commits nothing
(no labels, no marker) and strikes as a candidate.

Same commit (P4): flip the corollary in D5; rewrite README.md's
Safe-defaults bullet (:303 — "Unrecognizable classification → LOW_PRIORITY
(archived, not deleted)" becomes the raise-and-defer truth; the SERVICE
half stays per D2); reword the present-tense LOW_PRIORITY-default clauses
in llm_client's content-guard comment (:296-304) and config.toml's
`[llm.local]` comment (:72) — after this task there is no default left to
"silently parse to".

## T11 — D5/D20: `no-stories` only from successful zero-story extraction

Recon found **two** false-`no-stories` paths, not one:

1. **Unparseable extraction output** — `parse_stories` returns `[]` for
   garbage exactly as for a genuine `NO_STORIES` reply (its docstring says
   so, newsletter.py:116). Fix: `[]` only for an explicit `NO_STORIES`
   token; any other input that yields zero parsed stories — including
   empty/whitespace input, which is unreachable in production because
   llm_client's content guard raises first, but the parser's contract
   should not lie — raises `LLMContentError`. The parser is shared with
   evals (evals/newsletter_run.py:209-215 → error rows — honest
   degradation, same as T10).
2. **All-grades-unparseable** — stories extracted, but every story's
   `parse_quality_scores` returns None → every tier None → `best_tier=None`
   → the no-stories label (labeler.py:207) plus a tier-less JSONL record.
   This is issue #30's remaining parse-to-None route. Fix in
   `classify_newsletter`: if stories exist and **every** story's scores are
   None, raise `LLMContentError`. Per-story isolation survives for the
   partial case: one story grading while others fail keeps the tolerant
   path — but note the guard's most common instance is a **single-story**
   newsletter whose only story fails to grade, which today commits a false
   no-stories and after this task raises.

Both raises propagate exactly like the implemented D20 exception path
(LLMContentError is `_PIPELINE_WIDE_ERRORS`); under T8 they are strike
candidates — a poison newsletter converges to a findable `agent/attempted`,
never to a false no-stories record.

**RED tests:** `test_garbage_extraction_raises_not_empty`
(reworks `test_garbage_input` :105, which pins the conflation — cite
D5/D20); `test_all_grades_unparseable_raises_instead_of_no_stories`
(classify_newsletter level, red today); a daemon-level test asserting no
labels and no JSONL record on either path.

Tests reworked in the same commit, citing D5/D20 (all currently use
single-story fixtures that the all-None guard would trip):

- `test_quality_failure_still_classifies_themes` (:483) and
  `test_non_transient_quality_error_stays_isolated` (:584) — convert to
  two-story fixtures (one story failing, one scoring) so they keep pinning
  the surviving per-story isolation; their old single-story shapes become
  the new raise tests.
- `test_empty_input` (:101) — becomes expect-raise per fix 1's parser
  contract.

Kept green untouched: `test_no_stories` (:82) and
`test_no_stories_with_whitespace` (:86), `test_no_stories_returns_empty`
(:477) — the genuine zero-story outcome — and `test_newsletter_no_stories`
(test_daemon.py:1799, mocks `classify_newsletter → []`, which now means
exactly the genuine case).

Same commit (P4): flip the corollary in D5 and D20's "parse-to-None path
pending" status; update README.md:53's labels-table note (it documents the
conflation honestly with a pending marker — the marker resolves).

## T12 — D5: sink faults never counted, retried forever

A newsletter sink `OSError` (write_assessment) is re-raised into the generic
Exception arm and **strikes** — five failing cycles of a read-only mount
abandon graded newsletters to `agent/attempted`. D5: sink faults are
shared-cause (disk), never counted, retried forever.

- Wrap the sink fault at its re-raise site (daemon.py:453-464): raise a
  dedicated `AssessmentSinkError` (new, in newsletter.py, wrapping the
  OSError) instead of letting the bare OSError walk the arms. A dedicated
  arm in `process_single_thread` (before the candidate arms; note the
  existing ERROR line with the resolved path already fires at :456) returns
  False — no strike, no marker, retried next cycle, forever. (A dedicated
  class avoids `except OSError` ordering hazards — TimeoutError subclasses
  OSError.)
- With T6, the retry re-attempts only the JSONL write — the grading is
  cached, so "retried forever" costs no LLM spend.
- The per-cycle ERROR (path, cause) is the loudness; the startup preflight
  already screams about bad sinks.

**RED test:** rework `test_sink_failure_leaves_thread_unlabeled_for_retry`
(:1995 — its `FailureTracker(max_failures=3)` comment encodes today's
counting): drive `max_failures`+1 cycles of sink failure and assert the
thread is **never** marked attempted and no strike accrues (red today).

Same commit (P4): flip the corollary in D5; update the **three** texts that
promise the old give-up ending — `sink_writability_warning`
(newsletter.py:463-467, "…and eventually marked agent/attempted"), the
write-before-label comment above the re-raise site (daemon.py:434-439, "a
persistent fault ends at the give-up path's findable agent/attempted"), and
README-technical's write-before-label "Sink write fails" row plus the
sink-preflight example block (:212-214) that quotes the same sentence.

## T13 — D5: `MAX_FAILURES` knob

`max_failures` is a hardcoded constructor default (5, daemon.py:159; bare
`FailureTracker()` at :910). Follow the WRITE_PARALLEL precedent exactly:
`[daemon] max_failures = 5` in config.toml with a rationale comment
(authoritative home, D7 — the comment notes it also sets T8's masquerade
threshold), `resolve_int_env("MAX_FAILURES", ...)` at construction, a row in
README-technical's env table (optional; not added to `.env.example`), and
the Daemon-settings key list.

**RED test:** `test_max_failures_env_override` — monkeypatched env +
config, assert the tracker's threshold; red before the wiring exists
(test_env_var_docs also goes red on the new read until the table row
lands — same arc as T3).

Same commit: this is the **last** pending corollary — flip it, rewrite the
D5 entry's "Corollaries, each `implementation pending (Wave 2)`" preamble
to past tense with per-corollary implementing commits, flip the entry's
**Status:** line itself ("model is the governing design now; corollaries
pending (Wave 2)" → "implemented (model Wave 0; corollaries Wave 2)"), and
resolve CLAUDE.md's Failure Model paragraph ("several are `pending`, and
until they land, code deviates…", :80-82) — true until this commit, false
after it.

## T14 — Acceptance

1. Full suite green (CI's three-zone TZ matrix on the PR is the
   authoritative run; locally at least `TZ=UTC` and one west-of-UTC zone) +
   `ruff check .` clean.
2. Red/green log complete in the execution record: every task's named RED
   test(s) listed with the observed pre-change failure; any test updated
   because it pinned reversed behavior listed with the registry entry it
   cited.
3. Registry coherent: D3, D6, D10, D11, D12 flipped; D5's corollary list
   fully resolved and its Status line flipped (T8/T9/T10/T11/T12/T13
   commits named); D19's daemon-wide note updated; D20 closed out. No other
   entry touched.
4. CLAUDE.md coherence pass: the intro's per-function-halt note went with
   T9, the Failure Model "several are pending" paragraph with T13 — verify
   no stale pending-language remains; README.md Resilience and
   README-technical (write-before-label, sink preflight, Health Checking,
   env table; CI note untouched) describe the new behavior only.
5. Issue hygiene (owner actions on merge, prepared as comment drafts in the
   PR): close #28 (T7), #29 (T6), #33 (T5); comment on #30 (parse-to-None
   remainder landed in T11; sub-problem (b) stays open, still constrained by
   D5); comment on #35 (send_date/model now schema-documented by T4; the TUI
   header work remains open); for-the-record comment on closed #26 (T8
   executed D5's reversal); #68 gets the wave-executed comment with the
   Wave 3 handoff (the findings-catalog sweep, the #39 gap-list regeneration
   now unblocked by T1, the docstring/altitude sweep).
6. Scope check: Wave 2 owes nothing else from the roadmap; anything
   discovered mid-wave but out of scope is filed as an issue or left for
   Wave 3's catalog walk, not absorbed silently.
7. Final commit: append the **Execution record** (deviations, red/green log,
   per-task commit SHAs) and flip this plan's status header
   `proposed → executed` (D8).
