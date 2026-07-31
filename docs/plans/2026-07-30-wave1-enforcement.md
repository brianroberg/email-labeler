# Wave 1 — Enforcement (privacy negative-form tests, CI, #67)

> **Status: executed 2026-07-30** (owner-approved with T4 included; see the
> Execution record at the end for the mutation log).
> Drafted after the Wave 0.5 issue triage; #67 is absorbed into this wave per
> the triage disposition on issue #68.

Third step of the 2026-07-30 clarity effort, after Wave 0 (docs foundation,
executed) and Wave 0.5 (issue triage, executed). Wave 1 adds **enforcement
only**: the D2 privacy negative-form tests, the D13 CI gate, and the #67
timezone fix the CI gate depends on for honesty. Behavior changes wait for
Wave 2; the findings-catalog sweep for Wave 3.

## Goal

Make the guarantees the docs now state mechanically enforceable:

1. **D2, tested.** The privacy posture — bodies of threads *classified as*
   person are processed only by the local LLM; before routing, the cloud sees
   only sender/subject/snippet — gets negative-form tests that fail if any
   change widens what the cloud receives, plus a shape guard so the metadata
   objects cannot quietly grow a body-carrying field.
2. **D13, running.** Every PR and push to main runs lint plus the full mocked
   suite — on a suite that is green in every timezone (#67), so a green
   check on a UTC runner means the same thing on the owner's EDT machine.

## Principles (binding on every task below)

- **P1 — No production behavior changes.** No production `.py` is edited; the
  only source edits are under `tests/`. `config.toml` is untouched. Mutations
  used to prove tests are transient: applied, observed red, reverted, never
  committed.
- **P2 — After-the-fact tests are mutation-proven.** Every T2/T4 test targets
  code that already exists, so per CLAUDE.md's testing rules each is proven
  by a named mutation (matrix in T2/T4); the observed red runs are recorded
  in the execution record.
- **P3 — Honest phrasing.** Test names and docstrings state the D2 guarantee
  in its best-effort form ("classified as person", "nothing beyond
  sender/subject/snippet") — never the absolute form D2 forecloses. The
  snippet is Gmail's body-derived preview of the latest message
  (daemon.py:520), so no test asserts "no body text reaches the cloud"; the
  assertions are scoped to text beyond the three metadata fields.
- **P4 — One commit per task**, in task order; suite run after every task.
- **P5 — Timezone honesty.** At T1 and at acceptance, the suite must be fully
  green under `TZ=UTC`, `TZ=America/New_York`, and `TZ=Asia/Tokyo`
  (single-TZ runs suffice in between). Baseline before T1: exactly one known
  failure west of UTC — issue #67's test.
- **P6 — Registry statuses flip in the implementing commit** (D2's in T2's
  commit, D13's in T3's), per the Review Charter's same-change rule.

## Task list

| # | Task | Files touched |
|---|---|---|
| T1 | #67: timezone-proof the review-TUI date fixture | `tests/test_newsletter_review.py` |
| T2 | D2 privacy negative-form tests + metadata-shape guard | `tests/test_privacy.py` (new), `README-technical.md` (coverage row), `docs/decisions.md` (D2 status) |
| T3 | D13 CI workflow | `.github/workflows/ci.yml` (new), `README-technical.md` (CI note), `docs/decisions.md` (D13 status) |
| T4 | `.env.example` ↔ env-table sync guard *(optional — strike in review if unwanted)* | `tests/test_env_example_docs.py` (new), `README-technical.md` (coverage row) |
| T5 | Acceptance | — |

---

## T1 — #67: timezone-proof `_dated_records`

The production behavior is intentional (`_local_date`,
newsletter_review/tui.py:115, renders the viewer's local date); the fixture
defeats it by pinning UTC-midnight instants and asserting the UTC date
string. Fix the fixture, not the code and not the clock — issue #67's own
analysis, adopted verbatim: replace the module-level `_dated_records()`
helper (tests/test_newsletter_review.py:1031-1037) with local-midday instants
converted to UTC:

```python
def _dated_records():
    """Records with distinct fixed send-dates (all in 2024) for sort/date tests.

    Instants are local midday converted to UTC; rendering converts back in
    the same process timezone, an identity round-trip, so each record shows
    the authored local date in every timezone (issue #67). Midday also keeps
    the UTC calendar date within a day of the authored date, clear of the
    date filter's mid-month cutoffs.
    """
    def utc_iso(y, m, d):
        return datetime(y, m, d, 12, 0).astimezone(timezone.utc).isoformat()
    return [
        _make_record(subject="Jan", send_date=utc_iso(2024, 1, 10)),
        _make_record(subject="Mar", send_date=utc_iso(2024, 3, 10)),
        _make_record(subject="Feb", send_date=utc_iso(2024, 2, 10)),
    ]
```

Constraints, from the issue and verified against current code:

- **Do not** assert via `_local_date(...)` (re-implements the code under
  test) and **do not** pin `TZ=UTC` (hides the west-of-UTC rendering path;
  the file's `_use_tz` helper at :29-44 stays for the tests that genuinely
  test TZ behavior). GitHub runners are UTC — pinning would make CI
  permanently blind to exactly the class of bug this is.
- `TestReviewAppDateFilter` shares the fixture (call sites :1054, :1063,
  :1075, :1082, :1091); its mid-month cutoffs stay correct under midday
  instants — by construction now, not by luck.
- `datetime`/`timezone` are already imported (line 8); no new imports.

**Proof (red/green on the fixture):** before the change,
`TZ=America/New_York uv run --extra dev pytest tests/test_newsletter_review.py`
fails exactly `test_default_sort_by_send_date_desc`; after it, the full suite
is green under all three P5 timezones. On merge, close #67 citing this
commit.

## T2 — `tests/test_privacy.py`: D2 negative-form tests

New module; docstring cites D2/D3 and states the guarantee in best-effort
form. Uses the existing patterns only — no new test infrastructure:
`AsyncMock` LLM clients injected into a **real** `EmailClassifier`
(constructor DI as in tests/test_classifier.py:281-288), payload inspection
via `complete.call_args` / `call_args_list` (idiom:
test_classifier.py:299-301), and for the daemon-level tests the
`process_single_thread` invocation shape of tests/test_daemon.py:167-176
with thread dicts authored locally in the module, shaped like
test_daemon.py's `mock_thread_response` fixture (:77-118) — conftest's
fixtures are single-message resources, not thread payloads.

**Mock contract.** Every `complete(...)` call is made with
`include_thinking=True` and unpacked as a `(text, cot)` tuple
(classifier.py:246-249, 263-265, 299), so **both** LLM mocks get an explicit
2-tuple `return_value` (or `side_effect`) in **every** test — including the
mock a test expects never to be called — so that a mutation's red lands on
the named assertion, not on a tuple-unpack `ValueError` from an unconfigured
`AsyncMock`.

**Sentinel discipline.** Fixture bodies carry **two** distinctive markers:
one within the first ~100 characters (catches truncated leaks, e.g. a future
`body[:200]` interpolation) and one after several hundred characters of
filler (catches anything a snippet-sized prefix would miss). The fixture's
`snippet` field is a separate distinctive string containing neither marker —
in tests we author the thread dict, so the snippet is whatever we set; the
filler mirrors real Gmail, where a snippet is a short body prefix. "Leaked"
means: either marker present in any argument or kwarg of any
`cloud_llm.complete(...)` call.

Tests and the mutation each is proven by (every mutation is applied
transiently, the named test observed red, then reverted — P1/P2):

| Test | Asserts | Proven by mutation |
|---|---|---|
| `test_person_thread_body_reaches_only_local_llm` (classifier-level) | Stage 1 cloud mock returns `PERSON`; `classify()` sends the marker to `local_llm.complete` and to no `cloud_llm.complete` call | **M1**: classifier.py:284 — make the tier expression always pick `self.cloud_llm` |
| `test_person_thread_end_to_end_daemon` (daemon-level; the load-bearing one) | real `EmailClassifier` + AsyncMock LLMs through `process_single_thread`: markers in the local payload; absent from every cloud payload; each Stage 1 cloud call is checked by **whole-call equality** — `args[0]` equals the injected config's `sender_classification.system` string, `args[1]` equals the rendered `user_template`, kwargs exactly `{"include_thinking": True}` | **M2**: daemon.py:520-527 — pass the full transcript as `snippet` when building `ThreadMetadata` (hoist the daemon.py:530 `format_thread_transcript` call above the metadata build for the transient mutation; also caught: **M1**) |
| `test_service_thread_body_goes_to_cloud_and_local_unused` | Stage 1 returns `SERVICE` → marker in cloud Stage 2a payload, `local_llm.complete` never called (pins the tier split's other arm — and documents, honestly, that service bodies go to the cloud) | **M3**: classifier.py:284 — always pick `self.local_llm` |
| `test_unparseable_stage1_defaults_to_service_route` | Stage 1 returns keywordless prose → routed to cloud Stage 2a. Docstring pins the adjudicated D2 default: availability first; do **not** "fix" to PERSON | **M4**: classifier.py:110-111 — default to `PERSON` |
| `test_vip_sender_skips_cloud_entirely` | sender in `VIP_SENDERS` → `cloud_llm.complete` never called at all; marker only in local payload | **M5**: classifier.py:239-240/252-253 — disable the VIP short-circuit |
| `test_newsletter_thread_bypasses_stage1_and_local` (daemon-level) | thread whose To **exactly equals** the configured recipient → newsletter classifier receives the transcript (cloud by design, D3); the email classifier and `local_llm` are never invoked. Fixture uses an exact-match address so the test survives Wave 2's D3 exact-match change unchanged | **M6**: daemon.py:408-409 — force the newsletter branch **False** so the newsletter thread falls through to the email pipeline |
| `test_person_thread_local_failure_never_falls_back_to_cloud` (daemon-level) | Stage 1 returns `PERSON`, then the local mock raises `httpx.ConnectError` → neither marker in any `cloud_llm.complete` call, and no labels applied (the failure defers per today's behavior). Pins the most plausible D2 regression: a well-meaning "resilience" fallback that routes person bodies to the cloud when the local tier is down | **M8**: classifier.py:284 area / daemon.py:537-542 — transiently wrap the local call in try/except that retries via `self.cloud_llm` |
| `test_metadata_shapes_cannot_carry_bodies` | `dataclasses.fields` name-sets are exactly `{thread_id, senders, subject, snippet}` (`ThreadMetadata`, classifier.py:39-46) and `{message_id, sender, subject, snippet}` (`EmailMetadata`, classifier.py:31-36). Docstring: body text travels only as the explicit `body` argument (classifier.py:273/309); growing these classes is a privacy-review event, and a body-carrying field is foreclosed (D2) | **M7a**: add `body: str = ""` to `ThreadMetadata`; **M7b**: same on `EmailMetadata` (each arm proven separately) |

Notes binding the implementation:

- The Stage 1 assertion in the end-to-end test is a *whole-call equality*
  check — system string, rendered `user_template`, and kwargs — for each
  unique sender, not a substring blacklist, so a widening of the Stage 1
  payload in any slot fails it, not just one carrying our sentinels. The
  templates are rendered from **the config injected into the classifier
  under test** (the DI fixture pattern carries its own copy);
  config.toml:116-118 is cited as the production home of that template, not
  as the test's data source.
- The multi-sender loop (classifier.py:258-268) is covered by giving the
  end-to-end fixture two senders: assert one Stage 1 cloud call per sender
  until the first `PERSON` (short-circuit), each metadata-only.
- No test asserts anything about the *content* of `snippet` reaching the
  cloud (P3 — it is body-derived by design), and no test touches the
  newsletter transcript's cloud path except to pin that it *is* the cloud
  path (D3 forecloses privacy findings there).
- Same commit: add the `tests/test_privacy.py` row to README-technical's
  "Test Coverage by Module" table, and flip D2's status to
  `implemented (docs, Wave 0; test enforcement, Wave 1)` (P6).

## T3 — `.github/workflows/ci.yml`: the D13 gate

Create with exactly this content (action versions confirmed current at
execution time; bump majors then if needed):

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.14"
      - run: uv sync --locked --extra dev
      - run: uv run --extra dev ruff check .
      - run: uv run --extra dev pytest tests/
```

Rationale (kept here, not restated elsewhere — the workflow file plus this
plan section is the record):

- Triggers per D13 and observed history: PRs *and* direct pushes to main
  (Wave 0 merged to main without a PR; both routes must gate).
- `--locked` strengthens D13's `uv sync`: CI fails if `uv.lock` is stale
  rather than silently re-resolving — the Dockerfile's `--frozen`
  no-silent-re-resolution discipline, plus a staleness check. Python pinned
  to 3.14 matching `requires-python >= 3.14` (pyproject.toml:5) and the
  Dockerfile's `python:3.14-slim`.
- The suite is fully mocked and offline (1292 tests + 110 subtests, ~1 min
  local); no secrets, no `.env` (gitignored and absent in CI; the
  import-time `load_dotenv()` calls are no-ops there). `permissions:
  contents: read` is least-privilege.
- **No `TZ` pinning** — T1 made the suite timezone-honest; pinning UTC would
  reintroduce the blindness #67 exposed.
- Meta-test check (from recon): nothing parses YAML; `.github/` contains no
  `.py`, so `test_tui_docs`'s rglob finds nothing; `test_env_var_docs` scans
  root-level `.py` only. Adding the workflow file trips no test.

Same commit: a new two-line `## Continuous Integration` section in
README-technical (operational reference, D7), as a pointer, not a
restatement: "GitHub Actions (`.github/workflows/ci.yml`, authoritative)
runs dependency sync, lint, and the full mocked suite on every PR and push
to main; no secrets are required." Flip D13's status to
`implemented (Wave 1)` (P6).

## T4 — `.env.example` sync guard *(optional)*

Wave 0 T4 noted `.env.example` is guarded by nothing; the roadmap carries it
as an optional Wave 1 candidate. Smallest useful guard, mirroring
tests/test_env_var_docs.py's style — new `tests/test_env_example_docs.py`:

- Parse `.env.example` lines matching `^#? ?([A-Z_]+)=`.
- **(a)** Every parsed var must appear in README-technical's
  "## Environment Variables" table (catches a stale or misspelled example).
- **(b)** Every var that table marks Required (today: `PROXY_API_KEY`,
  `CLOUD_LLM_URL`, `CLOUD_LLM_API_KEY`) must appear as an **active**
  (uncommented) line in `.env.example` (catches the example losing a var the
  daemon won't start without).

Deliberately *not* asserted: that every optional/override knob appears in
`.env.example` — whether the override knobs (`LOCAL_PARALLEL`,
`MAX_EMAILS_PER_CYCLE`, …) belong in the example is an owner call this wave
does not make.

**Proof:** **M9** — add a bogus `NOT_A_REAL_VAR=x` line to `.env.example` →
(a) red; **M10** — comment out the `PROXY_API_KEY=` line → (b) red; revert
both. Same commit: add coverage-table rows for **both** env-doc guards
(`test_env_var_docs.py`, today absent from the table, and the new
`test_env_example_docs.py`) so the table is not selectively inconsistent —
the rest of the meta-test backfill stays with Wave 3.

## T5 — Acceptance

1. Full suite green under all three P5 timezones; `ruff check .` clean.
2. Mutation log complete in the execution record: M1–M8 including both M7
   arms (and M9–M10 if T4 kept), each showing the named test red on the
   named assertion, then reverted to green.
3. `git diff main -- '*.py'` shows changes only under `tests/`  (P1).
4. Push the branch, open the PR: the `pull_request` trigger runs and is
   green — the workflow's first live run *is* the acceptance evidence for
   T3. After merge, confirm one green `push` run on main.
5. Registry coherent: D2 and D13 statuses flipped (P6); no other entry
   touched. Close #67 (citing T1's commit) and comment on #68 that Wave 1 is
   executed.
6. Scope check against the roadmap: Wave 1 owes nothing else — D3
   exact-match, D5 corollaries, D6, D10–D12 are Wave 2; the docstring/
   altitude sweep is Wave 3. The #67-absorption and the optional
   `.env.example` guard are the only triage-fed additions.
7. In the final commit: append an **Execution record** section to this plan
   (home of the mutation log and any accepted deviations, per the Wave 0
   precedent) and flip this plan's status header `proposed → executed`
   (D8).

## Execution record (2026-07-30)

Executed T1–T4 in order (T4 included by owner decision), one commit each,
full suite + ruff green after every task. Tri-timezone acceptance
(`TZ=UTC` / `America/New_York` / `Asia/Tokyo`): 1302 passed + 123 subtests
in each. `git diff main -- '*.py'` touches only `tests/` (P1 held).

Mutation log (each applied transiently, observed red, reverted — the
production tree was verified clean after the pass):

| Mutation | Applied | Red observed on |
|---|---|---|
| M1 tier expression → always cloud | classifier.py `classify_email` | `test_person_thread_body_reaches_only_local_llm` (+ 3 route-dependent siblings) |
| M2 snippet → full transcript | daemon.py metadata build | `test_person_thread_end_to_end_daemon` (whole-call equality) + local-failure leak sweep |
| M3 tier expression → always local | classifier.py `classify_email` | both `TestServiceRouting` tests |
| M4 unparseable default → PERSON | classifier.py `parse_sender_type` | `test_unparseable_stage1_defaults_to_service_route` |
| M5 VIP short-circuit disabled | classifier.py `classify_sender` | `test_vip_sender_skips_cloud_entirely` |
| M6 newsletter branch forced False | daemon.py newsletter check | `test_newsletter_thread_bypasses_stage1_and_local` (on the cloud-count assertion) |
| M7a `body` field on ThreadMetadata | classifier.py | `test_metadata_shapes_cannot_carry_bodies` |
| M7b `body` field on EmailMetadata | classifier.py | `test_metadata_shapes_cannot_carry_bodies` (other arm) |
| M8 cloud fallback on local failure | classifier.py `classify_email` | `test_person_thread_local_failure_never_falls_back_to_cloud` (on the leak sweep) |
| M9 bogus var in .env.example | .env.example | `test_env_example_vars_are_documented` (subtest `NOT_A_REAL_VAR`) |
| M10 `PROXY_API_KEY` commented out | .env.example | `test_required_vars_present_and_active` (subtest `PROXY_API_KEY`) |

Accepted implementation refinements (within plan scope): the local-failure
test's leak assertion is ordered before the result assertion (and its cloud
mock carries a spare tuple) so M8's red lands on the leak sweep, not on an
incidental assertion; same reordering in the newsletter test for M6.

CI evidence: recorded on the PR (pull_request run) and on main after merge
(push run) — see the PR thread.
