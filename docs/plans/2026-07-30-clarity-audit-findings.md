# Phase 1 Clarity Audit — Findings Catalog (2026-07-30)

> **Status: frozen audit record (2026-07-30).** Point-in-time snapshot: all
> file:line references and quoted text date from commit d30b55c (2026-07-30)
> and are not maintained. This is the Wave 3 worklist — each finding gets a
> deliberate disposition (resolved-by-wave-N / fix-now / accept-with-registry-
> entry / decline-with-reason) once Waves 0–2 land. See
> 2026-07-30-wave0-clarity-foundation.md for the wave plan and
> docs/decisions.md (created by Wave 0) for the decisions many of these
> findings resolve into.


Ten examination dimensions, every finding adversarially verified against the files.
Verdicts: **confirmed** = claim and evidence hold as stated; **adjusted** = issue real, claim corrected per the verifier note. 0 findings were rejected.


## Stated aims & project identity

**Dimension summary:** The project has no single authoritative statement of purpose — every top-level statement (README.md:3, CLAUDE.md:3, pyproject.toml:4) describes only the email-labeling product, while the newsletter-grading product (its own module, review TUI, five-module eval harness, dedicated LLM config, and durable analytics corpus) states its aims only inside a mode paragraph and a February design doc that current code has since contradicted on at least four owner-decided points. The docs disagree on whether newsletter grading is an always-on branch or a flag-gated mode, and that ambiguity has already propagated a false claim into a later plan doc. Plan/design docs are presented as current truth with no historical/superseded markers and no completion status, and no non-goals are stated anywhere — together a plausible root cause of the fix-uncovers-new-problems symptom: every change to shared infrastructure is silently adjudicated against two products' unstated and partly contradictory intents.

### 1. [HIGH | gap | confirmed] No unified purpose statement covering both products

**Locations:** `README.md:1-3`; `CLAUDE.md:3`; `pyproject.toml:4`; `CLAUDE.md:52`

**Evidence:** Every top-level purpose statement describes only email triage: README.md:3 "A background daemon that continuously polls Gmail for unclassified emails, classifies them using a two-tier LLM system, and applies labels autonomously"; CLAUDE.md:3 near-identical; pyproject.toml:4 description = "Background daemon that classifies Gmail emails using a two-tier LLM system"; repo/package name is "email-labeler". The newsletter product's aim ("grades ministry newsletter stories on writing quality and thematic alignment", CLAUDE.md:52) appears only inside a mode-flag paragraph, yet that product owns newsletter.py, newsletter_review/, five newsletter_* eval modules, its own [newsletter.llm] config/env vars, and the durable JSONL corpus that Design Decision 6 (CLAUDE.md:114) exists to protect.

**Why it matters:** A maintainer fixing shared infrastructure (daemon loop, labeler, give-up path, LLM client) must decide which product's values win — triage safety vs corpus durability — but no document says the project IS two products or ranks their priorities. Fixes made under the 'email labeler' identity keep colliding with the newsletter product's unstated requirements, which surfaces as new problems on review.

**Verifier note:** Verified verbatim: every purpose statement (README.md:3, CLAUDE.md:3, pyproject.toml:4, name "email-labeler") describes only email triage, while the newsletter product's aim appears solely inside CLAUDE.md:52's NEWSLETTER_ONLY paragraph despite owning newsletter.py, newsletter_review/, five evals/newsletter_* modules, [newsletter.llm] config, and the JSONL corpus Design Decision 6 (CLAUDE.md:114) protects — no document names the two-product identity or ranks triage safety vs corpus durability.

### 2. [HIGH | ambiguity | confirmed] Newsletter pipeline: mode vs always-on branch

**Locations:** `CLAUDE.md:52`; `CLAUDE.md:138`; `README-technical.md:74`; `daemon.py:850-874`; `daemon.py:407-408`; `README.md:65-68`; `docs/plans/2026-07-01-newsletter-eval-plan.md:5`

**Evidence:** CLAUDE.md:52: "When `NEWSLETTER_ONLY=1`, the daemon switches to a newsletter-specific pipeline"; CLAUDE.md:138: "daemon runs newsletter classification pipeline instead of email labeling". But the code enables grading whenever [newsletter] exists in config.toml regardless of the flag (daemon.py:855 `if nl_config:` builds the classifier; daemon.py:407-408 routes any To/Cc-matching thread to grading), and README-technical.md:74 says the flag merely "skip[s] non-newsletter threads. Useful for testing newsletter classification in isolation." README.md:65-68 shows the newsletter branch inside the normal poll loop with no flag. The misreading already propagated: docs/plans/2026-07-01-newsletter-eval-plan.md:5 asserts the pipeline is "active under `NEWSLETTER_ONLY=1`" — false per the code.

**Why it matters:** A maintainer cannot answer the basic operational question 'does normal (non-flag) production grade newsletters and spend cloud tokens/labels on them?' CLAUDE.md implies no, code says yes. Any fix reasoned from CLAUDE.md's mode framing (e.g. gating newsletter wiring behind the env var) would change production behavior while appearing to match documented intent — a documented instance of a downstream doc already inheriting the wrong reading.

**Verifier note:** Verified: CLAUDE.md:52 ("switches to") and CLAUDE.md:138 ("instead of email labeling") frame grading as flag-gated, but daemon.py:851-873 builds the classifier whenever [newsletter] exists in config.toml and daemon.py:407-419 grades any To/Cc-matching thread unconditionally — the flag is read separately (daemon.py:886) and only skips non-newsletter threads (daemon.py:485-487), exactly as README-technical.md:74 and README.md's diagram (:65-68) say; the misreading already propagated verbatim into docs/plans/2026-07-01-newsletter-eval-plan.md:5 ("active under `NEWSLETTER_ONLY=1`").

### 3. [HIGH | tension | adjusted] Privacy invariant unqualified vs cloud newsletter bodies

**Locations:** `CLAUDE.md:15`; `README.md:9-19`; `README.md:65-72`; `daemon.py:407-419`; `newsletter.py:1-5`

**Evidence:** CLAUDE.md:15: "Person email bodies NEVER leave the local network. Cloud LLM only sees metadata (sender, subject, snippet) for Stage 1 classification." README.md:9: "person email bodies never leave the local network." Yet the newsletter branch (daemon.py:407-419) sends the full transcript to the cloud LLM keyed only on a To/Cc header match, before any person/service determination — and newsletters are authored by persons (staff senders like john@dm.org per the design doc). The only reconciliation lives in a code docstring, newsletter.py:4-5: "All LLM calls use the cloud endpoint (newsletter content is not privacy-sensitive)" — neither privacy section states the carve-out or its principle. Looks resolvable: the invariant needs an explicit scope rule (broadcast/organizational content exempt), but none is stated.

**Why it matters:** The privacy invariant is the project's first and strongest stated aim. Without a stated scope rule, a maintainer cannot judge whether a change to routing order or a new header-based branch honors or violates the aim, and any fix touching the pre-Stage-1 path invites a privacy objection on review — or worse, a 'fix' that reroutes newsletters through Stage 1 to 'restore' the invariant.

**Verifier note:** Tension real but one supporting claim needs correction: README.md:16, inside the Privacy Model section, already states the principle — "service emails (receipts, newsletters, notifications) contain no personal correspondence" — so the reconciliation is not only in newsletter.py:4-5. However it is scoped to Stage-2a service-classified mail; neither privacy section covers the pre-Stage-1 To/Cc branch (daemon.py:407-419) that ships full bodies of person-authored staff newsletters to the cloud with no person/service determination, and CLAUDE.md:15's invariant remains fully unqualified. Kind (tension) and severity stand.

### 4. [HIGH | contradiction | adjusted] 2026-02-19 design doc contradicts current behavior, unmarked

**Locations:** `docs/plans/2026-02-19-newsletter-classification-design.md:129`; `docs/plans/2026-02-19-newsletter-classification-design.md:35-51`; `docs/plans/2026-02-19-newsletter-classification-design.md:86-88`; `docs/plans/2026-02-19-newsletter-classification-design.md:127`; `CLAUDE.md:114`; `daemon.py:428-452`; `docs/plans/2026-07-08-phase1-decisions.md:14-48`

**Evidence:** The design doc reads in the present tense with no superseded marker, and contradicts current owner-decided behavior at least four times. (1) Error table row :129 "JSONL write fails | Log error, still apply Gmail labels" is the exact inverse of CLAUDE.md:114 ("the record is written first: a sink fault ... leaves the thread unprocessed and retried rather than labeled-but-lost") and daemon.py:429-438 ("Persist the assessment BEFORE the labels commit ... Writing afterwards (and swallowing the error) turned any sink fault ... into permanent silent data loss"). (2) :35 "scored on four dimensions (1-5 scale)" and :44-51 tier thresholds "Average >= 4.0" vs the shipped Poor/OK/Good scheme with >=2.75 bands (phase1-decisions:16-28). (3) :86 theme labels as "union across all stories" vs Emphasized-only labeling (phase1-decisions:30-33, CLAUDE.md:71). (4) :127 "Quality assessment fails for a story | That story gets no scores; other stories still processed" vs the #30 decision "Stories-exist-but-every-grade-errored is a failure, never a committed outcome" (phase1-decisions:39-41).

**Why it matters:** docs/plans is where an agent or maintainer looks for design intent. Following the JSONL-write row would literally re-introduce the silent-data-loss bug Design Decision 6 fixed; following the tier thresholds or theme-union rules would re-break the #53 scheme. Stale intent presented as current truth is a direct mechanism for the fix-creates-new-problems symptom.

**Verifier note:** Contradictions (1)-(3) confirmed verbatim (design :129 JSONL-write row inverts CLAUDE.md:114/daemon.py:430-439; :35 + :44-51 1-5 scale and >=4.0 bands vs shipped Poor/OK/Good >=2.75; :86 union theme labels vs Emphasized-only), and the doc carries no superseded marker. Corrections: the quality-fail row is at design.md:126 (not :127), and it is only partially contradicted — per-story parse-level isolation still matches the design (phase1-decisions:42-43); the contradiction holds for pipeline-wide errors and the every-grade-errored case (newsletter.py:23-27, phase1-decisions:39-41). Kind and severity stand.

### 5. [MEDIUM | contradiction | adjusted] 2026-02-20 TUI docs describe nonexistent modules

**Locations:** `docs/plans/2026-02-20-newsletter-tui-design.md:13-16`; `docs/plans/2026-02-20-newsletter-tui-design.md:74-80`; `docs/plans/2026-02-20-newsletter-tui-design.md:89-91`; `docs/plans/2026-02-20-newsletter-tui-plan.md:7`; `CLAUDE.md:90`

**Evidence:** Design :14-16 specifies "`tui_data.py` — Pure data layer ... `tui.py` — Textual App ... Entry point via `[project.scripts]`" and :91 "`tui = \"tui:main\"`"; the plan (:7) repeats it. Reality: no tui_data.py exists anywhere (find returns nothing), the app is newsletter_review/tui.py launched via `python -m newsletter_review`, and pyproject.toml has no [project.scripts] section. Keybindings also diverged: design :78-79 "`t` — cycle tier filter ... `h` — cycle theme filter" vs CLAUDE.md:90 "`f` opens the filter menu (`t` tier → ... `h` theme → ...)". No historical/superseded marker on either doc.

**Why it matters:** A maintainer or agent consulting these docs targets modules and entry points that don't exist, or 'restores' an abandoned filter UX. Cheap to mislabel as drift, but it erodes trust in docs/plans generally — making it unclear which plan docs are still binding (see the roadmap, which IS treated as binding).

**Verifier note:** Core confirmed: no tui_data.py exists anywhere, pyproject.toml has no [project.scripts], the app is newsletter_review/tui.py via `python -m newsletter_review`, and design :78-79's t/h cycle bindings diverged from the shipped f-menu. Two corrections: the hotkeys line is CLAUDE.md:92 (not :90), and "no historical/superseded marker on either doc" overstates — docs/plans/2026-07-03-tui-framework-evaluation.md:18-20 explicitly flags the 02-20 design as historical ("originally designed...in Textual, but the implementation shipped on stdlib curses"), though that marker sits in a sibling doc, not on either 02-20 doc, and covers only the framework fork, not the dead module names/entry point.

### 6. [MEDIUM | gap | adjusted] Plan/decision docs carry no completion status

**Locations:** `docs/plans/2026-07-08-issue-roadmap.md:5-7`; `docs/plans/2026-07-08-issue-roadmap.md:33`; `docs/plans/2026-07-09-phase2-decisions.md:30-33`; `scripts/migrate_assessments.py:1`

**Evidence:** Roadmap :6 declares itself "the *forward* plan" and :33 lists #53 as open P0 work — but #53 shipped long ago (scripts/migrate_assessments.py:1 "Migrate pre-#53 records ... to the current scheme"; CLAUDE.md:71 documents the shipped Emphasized-only labeling). phase2-decisions :32 says of #28 "**Implementation is Phase 3** (per the roadmap)", and no phase-3 decisions doc exists to record whether that happened. Nothing in docs/plans distinguishes live intent from completed or superseded work.

**Why it matters:** A maintainer reading the roadmap cannot tell which items are still intent, which shipped, and which decisions were revised in flight — so priorities get re-derived from stale snapshots, done work gets re-planned, and the #28 Option-A semantics may or may not be in force (nobody can tell from docs alone).

**Verifier note:** The gap is real but the blanket claim overreaches: phase1-decisions marks sub-items "DONE" (:71-82, :86-89) and phase2-decisions marks tasks "(shipped)" (:36, :66, :82), so decision docs do carry per-item status. The confirmed core: the roadmap (:6 "the *forward* plan") still lists shipped #53 as open P0 (:33), nothing back-propagates completion to it, no phase-3 doc exists, and the #28 Option-A question is live — code check confirms ProxyForbiddenError is never caught in daemon per-thread paths (mentioned only in a docstring, daemon.py:737), so the deferred implementation indeed never happened and docs cannot tell you that.

### 7. [MEDIUM | gap | confirmed] No non-goals stated anywhere

**Locations:** `README.md:1-317`; `CLAUDE.md:1-15`; `docs/plans/2026-07-08-issue-roadmap.md:42`

**Evidence:** A repo-wide grep for "non-goal|out of scope|NOT a goal" hits only the roadmap's per-issue note (:42 "the owner already flagged this as out of scope for #53 itself"). No document bounds the project's scope: nothing states whether multi-account support, label auto-creation, cloud fallback for person mail, org-agnostic configurability, or generalizing the newsletter rubric are deliberately excluded. Scope limits that do exist are phrased as mechanism, not intent (README.md:19 "They are never sent to the cloud as a fallback"; README.md:133 "The api-proxy blocks programmatic label creation" — an external constraint, not a stated non-goal).

**Why it matters:** Without stated non-goals, every fix's blast radius is negotiable: a reviewer can always reasonably ask "shouldn't this also handle X?" and be neither confirmably right nor wrong. That is the classic generator of the problem-per-fix churn the owner reports.

**Verifier note:** Verified by repo-wide grep: the only "out of scope" phrasing anywhere is the roadmap's per-issue note (2026-07-08-issue-roadmap.md:42), and the closest scope limits are mechanism-phrased (README.md:19 no-cloud-fallback, README.md:133 proxy blocks label creation) — no document bounds what the project deliberately excludes.

### 8. [LOW | tension | confirmed] Generic-product framing vs single-owner org-specific tool

**Locations:** `README.md:88-127`; `config.toml:157`; `config.toml:244-250`; `docs/runbook-agent-attempted-recovery.md:3-5`

**Evidence:** README.md frames the project as installable software for anyone ("A cloud LLM endpoint (any OpenAI-compatible chat completion API)", README.md:93; cp .env.example setup flow), while config.toml:157 hardcodes `recipient = "newsletters@dm.org"` and :244-250 embeds one specific organization's "Ends Statement" themes into version-controlled prompts. The runbook (:3-5) is explicitly "Owner-run, manual ... no step should be run by an agent with Gmail write access" — a single-owner operating model no other doc states. Looks inherent unless the single-deployment assumption is written down.

**Why it matters:** A maintainer cannot decide whether a change must stay org-agnostic (per the README's generic framing) or may hardcode dm.org specifics (per config.toml's precedent). Both patterns exist in the codebase, so whichever a fix chooses, it looks inconsistent with half the project.

**Verifier note:** Verified: README.md:88-127 frames an installable product ("any OpenAI-compatible chat completion API" :93, cp .env.example :116) while config.toml:157 hardcodes recipient="newsletters@dm.org" and :244-256 embeds one organization's Ends Statement into version-controlled prompts, and the runbook (:3-5) assumes an owner-run single-operator model no other doc states — no document records the single-deployment assumption, so org-agnostic vs dm.org-specific remains undecidable for new changes.

## The privacy invariant

**Dimension summary:** The invariant is stated prominently and near-identically in four places (CLAUDE.md, README.md, classifier.py, daemon.py), but it is stated as an absolute guarantee while being implemented as a single best-effort LLM routing decision whose failure mode the project itself quantifies as a "privacy violation rate." Around that core, the boundary is fuzzy on every edge: the SERVICE parse-default is documented as "safe" although it is the privacy-worst-case choice; the newsletter pipeline routes full bodies to the cloud on a To/Cc substring match before sender classification with no documented carve-out; and the docs sanction pointing the "local" endpoint at a public API (Novita.ai) with no acknowledgment that this voids the invariant. No test asserts the negative form of the invariant (person body never reaches the cloud client), so the flagship constraint exists only as prose plus one routing line — exactly the shape that lets a well-intentioned fix silently create a new violation.

### 9. [HIGH | contradiction | confirmed] Absolute invariant vs SERVICE-default and measured violation rate

**Locations:** `README.md:9`; `CLAUDE.md:15`; `CLAUDE.md:110`; `README.md:292`; `classifier.py:110-116`; `evals/schemas.py:74`; `evals/run_eval.py:342-344`; `evals/report.py:280-281`

**Evidence:** README.md:9 claims "Email labeler enforces a strict privacy invariant: **person email bodies never leave the local network.**" and CLAUDE.md:15 says bodies "NEVER leave". But CLAUDE.md:110 documents the parse fallback as safe: "Unknown sender type → SERVICE (body goes to cloud, safe for non-person)", and classifier.py:110-111 implements it (`result = SenderType.SERVICE` on unparseable Stage 1 output) — for an actual person sender this routes the full body to the cloud. The eval suite openly treats this as an expected, countable outcome: evals/schemas.py:74 "privacy_violation: bool = False  # True if expected=person, predicted=service" and evals/report.py prints "[PRIVACY VIOLATION]" per thread plus a "privacy_violation_rate". An absolute "NEVER"/"enforces" and a measured nonzero violation rate driven by a deliberate cloud-ward default cannot both be true; the real guarantee is "bodies of threads *classified as* person stay local".

**Why it matters:** Every routing/parsing fix forces the maintainer to answer: is a person→service misroute a bug to be prevented at all costs, or an accepted error mode to be minimized? The docs answer both ways at once. That is how a fix creates new problems: someone hardening "NEVER" (e.g. defaulting unknown to PERSON, or blocking on ambiguity) breaks the "safe default = SERVICE" resilience story, while someone extending the safe-default pattern erodes the flagship constraint — and neither can cite the docs as authority.

**Verifier note:** All citations verified: README.md:9/CLAUDE.md:15 state an absolute "NEVER"/"enforces", while CLAUDE.md:110 and classifier.py:110-111 deliberately default unparseable Stage 1 output to SERVICE (body cloud-ward), and the project's own eval formalizes person→service misroutes as an expected, counted outcome (schemas.py:74, run_eval.py:342-344, report.py:143-150 privacy_violation_rate, report.py:280-281 "[PRIVACY VIOLATION]"); no doc sentence anywhere scopes the invariant to "classified-as-person", so the absolute guarantee and the measured violation mode are irreconcilable as written.

### 10. [HIGH | tension | confirmed] Newsletter To/Cc route bypasses Stage 1 privacy routing

**Locations:** `daemon.py:407-419`; `newsletter.py:486-498`; `newsletter.py:4-5`; `CLAUDE.md:15`; `README.md:63-68`

**Evidence:** daemon.py routes to the newsletter pipeline BEFORE any sender classification: at daemon.py:409 `if is_newsletter(messages, newsletter_recipient):` the full transcript goes to the cloud (`await newsletter_classifier.classify_newsletter(transcript)`, daemon.py:419) — Stage 1 person/service routing never runs. is_newsletter (newsletter.py:491-497) is a case-insensitive *substring* match on To/Cc of ANY message in the thread (`if target in value: return True`), so a person's reply in a thread that merely Cc's newsletters@dm.org — or an address that contains the recipient as a substring (e.g. "my-newsletters@dm.org") — sends person-written body text to the cloud. The only reconciliation with the invariant is an assumption stated by fiat in newsletter.py:4-5: "All LLM calls use the cloud endpoint (newsletter content is not privacy-sensitive)". CLAUDE.md's Privacy Invariant section (line 15) states "Person email bodies NEVER leave the local network" with no newsletter carve-out, and tests/test_newsletter.py:720-794 never test the substring false-positive.

**Why it matters:** Looks resolvable (document the carve-out precisely; tighten the match to an exact address), but today a maintainer cannot answer "which emails are allowed to go to the cloud in full?" from the privacy docs — the actual answer is "anything whose To/Cc contains a configured substring, sender identity irrelevant", which contradicts the mental model the invariant builds. Any fix touching thread routing, the recipient config, or is_newsletter can silently widen the cloud path without tripping any documented rule or test.

**Verifier note:** Verified: daemon.py:408-419 routes the full thread transcript to the cloud before any sender classification; is_newsletter (newsletter.py:491-497) is a case-insensitive substring match (`if target in value`) over To/Cc of ANY message, so "my-newsletters@dm.org" or a person's reply in a Cc'd thread sends person-written body text cloud-ward; the only reconciliation is the fiat comment at newsletter.py:4-5, CLAUDE.md:15 has no newsletter carve-out, and TestIsNewsletter (tests/test_newsletter.py:718-794) never tests the substring false-positive.

### 11. [HIGH | contradiction | adjusted] MLX public-API stand-in (Novita) voids the local-only rule

**Locations:** `CLAUDE.md:137`; `README-technical.md:70`; `evals/run_eval.py:573-574`; `daemon.py:832-841`; `CLAUDE.md:15`

**Evidence:** CLAUDE.md:137 documents "MLX_API_KEY — Local LLM API key (empty for real MLX, set for public API stand-ins like Novita.ai)" and README-technical.md:70 repeats it: "Empty string for real MLX; set for public API stand-ins like Novita.ai." Both daemon (daemon.py:832-841 builds the local client from MLX_URL/MLX_API_KEY) and evals (evals/run_eval.py:573-574) will then send person email bodies to that public endpoint — the exact traffic CLAUDE.md:15 says "NEVER leave[s] the local network" and README.md:17 says "never leaves the local network". No document scopes the stand-in to non-person data, to evals only, or even flags it as an invariant exception; it is presented as a routine configuration alongside "real MLX".

**Why it matters:** This is a sanctioned configuration that falsifies the project's flagship constraint, documented in the same file that states the constraint, 120 lines apart. A maintainer (or agent) reading the env-var table has no signal that setting MLX_URL to Novita is privacy-consequential — so a "fix" like standing up a cloud stand-in while the laptop is being replaced looks fully supported. Either the invariant needs an explicit exception ('unless you opt in via a public stand-in') or the stand-in doc needs a hard warning; as written both statements cannot hold.

**Verifier note:** Evidence verified (CLAUDE.md:137, README-technical.md:70, daemon.py:832-841, run_eval.py:572-580; no scoping text anywhere per repo-wide grep), but kind should be tension, not contradiction: the env-var doc sanctions a configuration whose use with person mail would falsify CLAUDE.md:15, yet it never asserts person bodies go to the stand-in — a clarification scoping stand-ins to eval/testing or non-person traffic would reconcile the texts. The real, high-severity issue stands: a privacy-consequential option documented 120 lines from the invariant with zero warning.

### 12. [MEDIUM | ambiguity | confirmed] Snippet is body-derived but classed as metadata

**Locations:** `CLAUDE.md:15`; `README.md:13`; `classifier.py:233-234`; `daemon.py:520`; `config.toml:116-118`

**Evidence:** CLAUDE.md:15: "Cloud LLM only sees metadata (sender, subject, snippet)". README.md:13: "receives only email metadata — sender, subject line, and Gmail snippet". classifier.py:234: "Only metadata (sender, subject, snippet) is sent — never the body." But the Gmail snippet IS the opening text of the body: daemon.py:520 takes `messages[-1].get("snippet", "")` (the latest — possibly person-written — message) and config.toml:118 sends it cloud-ward as `Preview: {snippet}` in Stage 1. So for every person email, the first ~100-200 characters of body content leave the local network by design, and nothing anywhere defines why that is "metadata" rather than "body", or where the line sits.

**Why it matters:** "Exactly what is protected" is answered only by implication. A maintainer deciding whether a longer preview, a locally computed body prefix, or the subject of a person email may be sent to the cloud has no stated principle to apply — the current rule ("whatever Gmail calls a snippet is fine") is an accident of the API, not a documented privacy decision, and a plausible "improve Stage 1 accuracy by sending more context" fix would violate the spirit of the invariant while matching its letter.

**Verifier note:** Verified: CLAUDE.md:15, README.md:13, and classifier.py's docstring ("Only metadata (sender, subject, snippet) is sent — never the body." — actually at classifier.py:232, cited 233-234) all class the snippet as metadata, while daemon.py:520 takes the latest (possibly person-written) message's Gmail snippet and config.toml:118 sends it cloud-ward as "Preview: {snippet}"; the Gmail snippet is the body's opening text and no document defines where the metadata/body line sits.

### 13. [MEDIUM | gap | adjusted] No test asserts the invariant's negative form

**Locations:** `tests/test_classifier.py:319-327`; `tests/test_classifier.py:291-301`; `tests/test_daemon.py:149-812`; `CLAUDE.md:15`

**Evidence:** The only privacy-adjacent tests are positive routing checks: test_person_routes_to_local_llm (tests/test_classifier.py:319-327) asserts `mock_local_llm.complete.assert_called_once()` but never asserts the cloud mock was NOT called with the body; test_routes_to_cloud_llm (291-301) checks metadata appears in the Stage 1 prompt but not that the body is absent from it. `grep -ri privacy tests/` matches only tests/test_eval_report.py (eval metric math). test_daemon.py's TestProcessSingleThread has no case asserting a person thread's transcript never reaches the cloud client, and no case covering the newsletter route intercepting a person-sender thread. The invariant is enforced by essentially one line (classifier.py:284 `llm = self.local_llm if sender_type == SenderType.PERSON else self.cloud_llm`) plus daemon routing order, with no test that would fail if either changed.

**Why it matters:** This project's stated method is strict red/green TDD ("No production code without a failing test first"), which makes the test suite the de facto design record — yet its flagship constraint has no executable form. A refactor of process_single_thread or classify_email that also consults the cloud LLM for person threads (e.g. a "second opinion" or fallback fix) passes the entire suite. Given that the newsletter change already inserted a new cloud path in front of the person routing with no privacy test, the gap has demonstrably let boundary-moving changes land unexamined.

**Verifier note:** Gap is real but narrower than claimed: test_vip_full_pipeline (tests/test_classifier.py:490-502) DOES assert mock_cloud_llm.complete.assert_not_called() on a full person(VIP)+body pipeline, and flipping classifier.py:284 would fail test_person_routes_to_local_llm's positive local assertion — so "no test would fail if either changed" is overstated. Also `grep -ri privacy tests/` matches test_eval_schemas.py too, and test_daemon.py:1641-1681 does cover the newsletter route intercepting a person-sender thread (From "John Staff <john@dm.org>", classify_sender.assert_not_called) — as routing, not privacy. The surviving gap: the ordinary non-VIP person path has no cloud-never-receives-the-body assertion, and daemon-level cloud paths are structurally invisible to tests because test_daemon.py mocks the whole classifier.

### 14. [MEDIUM | ambiguity | confirmed] NEWSLETTER_ONLY docs imply flag-gated cloud pipeline; wiring is always-on

**Locations:** `CLAUDE.md:52`; `CLAUDE.md:138`; `daemon.py:851-876`; `daemon.py:408-419`; `README.md:63-68`

**Evidence:** CLAUDE.md:52: "When `NEWSLETTER_ONLY=1`, the daemon switches to a newsletter-specific pipeline" and CLAUDE.md:138: "daemon runs newsletter classification pipeline instead of email labeling" — both read as if newsletter grading (a full-body cloud path) exists only under the flag. But daemon.py:851-876 wires newsletter_classifier whenever config.toml has a [newsletter] section (the shipped config always does), and process_single_thread's route (daemon.py:408: `if newsletter_classifier and newsletter_recipient:`) runs in normal mode too; the flag only *skips non-newsletter threads* (daemon.py:485-487). README.md's diagram (lines 63-68) shows the newsletter check inside the ordinary poll loop, agreeing with the code and disagreeing with CLAUDE.md's framing.

**Why it matters:** An operator who has not set NEWSLETTER_ONLY can reasonably believe, from CLAUDE.md, that no full-body newsletter traffic goes to the cloud — while in fact every inbox thread touching newsletters@dm.org does. It is genuinely unclear whether mixed-mode grading is intended design or an accident of wiring, so a maintainer cannot decide whether a fix should gate the newsletter route on the flag (tightening privacy exposure) or preserve mixed-mode (matching README) without guessing at intent.

**Verifier note:** Verified: daemon.py:851-876 wires newsletter_classifier whenever config.toml has [newsletter] (the shipped config does, per test_config_has_newsletter_section), the route at daemon.py:408 runs in normal mode, and the flag only skips non-newsletter threads (daemon.py:485-487); CLAUDE.md:52/138 frame the pipeline as existing only under the flag while README.md:63-68 and README-technical.md:74 ("skip non-newsletter threads") agree with the code — the docs genuinely answer the mixed-mode question both ways.

### 15. [LOW | ambiguity | confirmed] "Local network" undefined; blessed setup is a Tailscale overlay

**Locations:** `.env.example:27-28`; `README.md:17`; `CLAUDE.md:15`

**Evidence:** .env.example:27-28 documents the canonical local endpoint as "# Local LLM Configuration (MLX/Qwen3.6 via Tailscale)" / "MLX_URL=http://macbook:8080/...". README.md:17 promises the body is sent "to a local MLX/Qwen3.6 instance running on the same network. The body never leaves the local network." Tailscale is a WireGuard overlay whose traffic can transit the public internet (encrypted, possibly via DERP relays), so even the reference deployment is only "local" under an unstated definition (same tailnet / same administrative control), not "same LAN". No document defines the boundary the invariant protects.

**Why it matters:** The invariant's noun is undefined. Whether moving the MLX box to a friend's house, a colo, or a rented GPU host on the same tailnet keeps the invariant is unanswerable from the docs — and the Novita stand-in shows the boundary already sliding from "LAN" toward "any endpoint I configure". A crisp definition (e.g. "machines under the owner's administrative control") would make each future endpoint decision mechanical instead of debatable.

**Verifier note:** Verified: .env.example:27-28 names Tailscale as the canonical MLX path, README-technical.md:334 further documents reaching the local server "over Tailscale", and no document defines the boundary README.md:17/CLAUDE.md:15 promise — the reference deployment is "local" only under an unstated same-tailnet/same-owner definition.

### 16. [LOW | gap | confirmed] Eval artifacts' privacy posture undocumented

**Locations:** `.gitignore:25-29`; `evals/schemas.py:13-17`; `evals/llm_cache.py:158-164`; `evals/web_auth.py:13-16`; `evals/run_web.py:15`; `evals/README.md:1-30`

**Evidence:** The golden set stores full person email bodies on disk (evals/schemas.py:14 `messages: list[dict]  # Raw Gmail message resources`), the LLM cache persists responses/thinking that can quote them (evals/llm_cache.py:158-164), and .cot.jsonl sidecars capture local-model reasoning about person bodies. The only privacy control is a .gitignore comment: "# Classification quality data (contains email content, keep local)" (.gitignore:25). The eval web app can serve this content beyond localhost (run_web.py:15 `--host` flag) with auth that silently disables itself: web_auth.py:16 "return True  # auth disabled when no secret configured". Neither evals/README.md nor evals/README-technical.md mentions privacy at all (grep for privacy/bodies returns nothing except the metric name).

**Why it matters:** The invariant governs the daemon's network traffic but is silent about the at-rest and re-transmission fate of the same person bodies once evals copy them. A maintainer wiring the eval web UI to a NAS, sharing a cache file to reproduce a run, or running run_eval against a remote endpoint has no stated rule to consult — the closest thing to policy is a comment in .gitignore. Stating the eval-data posture once would prevent each future eval feature from re-deciding it implicitly.

**Verifier note:** Verified: full person bodies persist in the golden set (schemas.py:14 raw Gmail resources), the cache stores responses+thinking (llm_cache.py:158-164), the web UI can bind non-localhost (run_web.py:15) with auth that self-disables absent EVAL_WEB_SECRET (web_auth.py:15-16), and the only stated policy is the .gitignore:25 comment — neither evals/README.md nor evals/README-technical.md mentions privacy at all.

## Email vs newsletter dual pipeline

**Dimension summary:** The newsletter pipeline is not a cleanly separated mode but a branch bolted into the email pipeline: it activates on [newsletter] config presence, while NEWSLETTER_ONLY merely suppresses the email path and narrows the query — yet the three docs describe the flag three different ways (production mode switch, "classify only newsletters", testing utility), so the shipped default (both pipelines active in one daemon) is never named as a supported mode anywhere. Shared machinery designed for the email pipeline — the out-of-funds halt, cloud_parallel semaphore, the privacy invariant, marker-label discipline — carries single-pipeline assumptions that silently degrade when the newsletter path runs (per-pipeline providers, whole-grading semaphore holds, cloud-bound person replies inside newsletter threads, an unmarked skip branch). This "two switches, three descriptions, one shared substrate" structure is a plausible root cause of the fix-creates-new-problems symptom: any fix scoped to one pipeline's mental model has undocumented consequences in the other.

### 17. [HIGH | ambiguity | confirmed] NEWSLETTER_ONLY is a filter, not a mode switch

**Locations:** `CLAUDE.md:52`; `CLAUDE.md:138`; `README.md:65`; `README.md:183-186`; `README-technical.md:74`; `daemon.py:407-409`; `daemon.py:484-487`; `config.toml:156-158`

**Evidence:** CLAUDE.md:138 says the env var makes the "daemon run[s] newsletter classification pipeline instead of email labeling" and CLAUDE.md:52 says the daemon "switches to a newsletter-specific pipeline". But daemon.py:407-408 routes to the newsletter pipeline whenever `if newsletter_classifier and newsletter_recipient:` — gated on config presence, not on the flag — and config.toml:156-158 ships `[newsletter]` with `recipient = "newsletters@dm.org"` checked in, so the default email-mode daemon grades newsletters too (README.md:65's diagram shows "Newsletter? (To: check)" inside the normal poll loop, consistent with code). The flag's only effects are the skip at daemon.py:484-487 and the query narrowing at 928-930. README-technical.md:74 calls it "Useful for testing newsletter classification in isolation."

**Why it matters:** A maintainer cannot answer the most basic question about the design: is hybrid operation (email labeling + newsletter grading in one daemon) an intended mode, an accident of the config being checked in, or a bug? CLAUDE.md's "instead of" implies newsletters are untouched in email mode — a fix reasoned from that model (e.g. changing newsletter labeling, retries, or the sink) unexpectedly changes the behavior of the production email daemon, which is exactly the fix-creates-new-problems symptom.

**Verifier note:** Verified end to end: daemon.py:408 routes to the newsletter pipeline on `if newsletter_classifier and newsletter_recipient:` (config presence, never the flag), config.toml:156-158 ships [newsletter] enabled, so the default email-mode daemon grades newsletters; the flag's only effects are the skip (daemon.py:485-487) and query narrowing (928-930) — CLAUDE.md:52's "switches to" and :138's "instead of email labeling" describe a mode switch the code does not implement, while README-technical.md:74 correctly describes a filter.

### 18. [HIGH | tension | confirmed] Two overlapping pipeline switches

**Locations:** `daemon.py:851-855`; `daemon.py:886-888`; `labeler.py:71-78`; `tests/test_daemon.py:1049-1056`; `CLAUDE.md:138`

**Evidence:** Pipeline enablement is config presence: daemon.py:851-855 `nl_config = config.get("newsletter") ... if nl_config:` builds the classifier, and labeler.py:72-78 requires all newsletter Gmail labels iff `[newsletter.labels]` exists ("# Newsletter labels (if configured)"). The env var read at daemon.py:886 only disables the email path. The test harness itself uses the other switch: test_daemon.py:1056 `config.pop("newsletter", None)  # keep the loop on the plain email pipeline`, while its comment at 1051 says "NEWSLETTER_ONLY stays unset so the loop itself remains on the plain email pipeline" — conflating both switches in two adjacent lines. No doc states which switch controls what.

**Why it matters:** "How do I turn newsletter grading off?" has no documented answer, and the two switches have different blast radii: removing [newsletter] also changes startup label verification and the sink preflight, while unsetting NEWSLETTER_ONLY changes nothing about newsletters. This looks resolvable (document config-presence as the enable, env var as the email-path suppressor), but until stated, fixes toggling one switch get surprised by the other.

**Verifier note:** Verified: enablement is config presence (daemon.py:851-855 builds the classifier iff [newsletter] exists; labeler.py:72-78 requires all newsletter Gmail labels iff [newsletter.labels] exists) while the env var read at daemon.py:886 only suppresses the email path — and the test harness itself needs both switches in adjacent lines (test_daemon.py:1051 comment on NEWSLETTER_ONLY, :1056 `config.pop("newsletter", None)` to actually keep the loop on the email pipeline); no doc states config-presence is the enable, and CLAUDE.md:138 actively misattributes it to the env var.

### 19. [MEDIUM | tension | confirmed] Newsletter-only skip never marks threads

**Locations:** `daemon.py:484-487`; `daemon.py:493-498`; `daemon.py:886-888`; `daemon.py:928-930`; `tests/test_daemon.py:1883-1886`

**Evidence:** daemon.py:484-487: `if newsletter_only: log.debug("Skipping non-newsletter thread %s ..."); return False` — no mark_processed, no failure count, so the thread re-matches gmail_query every cycle. This contradicts the principle stated six lines later (daemon.py:496-497): without marking, "the thread has no agent/processed label and re-matches every cycle forever, costing a full get_thread round-trip per thread per poll". The narrowing that makes the branch mostly-dead only applies `if newsletter_only and newsletter_recipient` (daemon.py:928), so NEWSLETTER_ONLY=1 with no/empty [newsletter] config yields a daemon that fetches every inbox thread every cycle forever while logging only "Newsletter-only mode: non-newsletter threads will be skipped" (daemon.py:888). test_daemon.py:1883-1886 asserts the skip returns False and calls nothing — the loop behavior is untested and unstated.

**Why it matters:** The codebase's own hardest-won rule (every surfaced thread must end marked or give-up-bounded — the reason agent/attempted exists) is silently violated by the newsletter-only branch. A maintainer touching the skip, the query narrowing, or is_newsletter cannot tell whether the unbounded re-fetch is accepted, relied upon, or an oversight, and the degenerate config state (flag on, config absent) is undefined.

**Verifier note:** Verified: daemon.py:485-487 returns False with no mark_processed and no failure count (only exception arms reach _give_up_if_stuck), contradicting the mark-or-loop-forever principle written six lines later at 493-498; the narrowing at 928 requires `newsletter_only and newsletter_recipient`, so NEWSLETTER_ONLY=1 with [newsletter] absent (classifier None, recipient "") re-fetches every inbox thread every cycle forever with only a per-thread DEBUG line, and test_daemon.py:1883-1886 asserts only the skip's return value, not the loop consequence.

### 20. [MEDIUM | ambiguity | confirmed] Two definitions of newsletter membership

**Locations:** `daemon.py:928-930`; `newsletter.py:486-498`; `CLAUDE.md:55`; `README.md:65`

**Evidence:** The query narrowing appends `to:` only — daemon.py:929 `gmail_query += f" to:{newsletter_recipient}"` — while the code-level test checks To AND Cc: newsletter.py:494 `for header_name in ("To", "Cc")` with substring match over "any message in a thread". The docs split the same way: CLAUDE.md:55 "find unprocessed newsletters (To/Cc matches config recipient)", README.md:65 "Newsletter? (To: check)". If Gmail's `to:` operator does not match Cc, a Cc-only newsletter is graded in hybrid mode but never surfaces in NEWSLETTER_ONLY mode — the mode flag changes which mail counts as a newsletter.

**Why it matters:** There is no single authoritative definition of "a newsletter thread": one lives in the Gmail query, one in is_newsletter, and the two docs each describe a different one. Anyone fixing newsletter detection (or the recipient matching, which is also a bare substring) must guess which definition is intended and can silently change corpus membership differently per mode.

**Verifier note:** Verified: daemon.py:929 narrows with `to:` only while newsletter.py:494 checks both ("To", "Cc") via case-insensitive substring over any message in the thread, and the docs split identically — CLAUDE.md:55 "To/Cc matches config recipient" vs README.md:65 "Newsletter? (To: check)" — so a Cc-only newsletter is graded in hybrid mode but never surfaces under the NEWSLETTER_ONLY-narrowed query (Gmail's to: operator has a separate cc: counterpart), leaving no single authoritative membership definition.

### 21. [HIGH | tension | confirmed] Privacy invariant unscoped for newsletter threads

**Locations:** `CLAUDE.md:15`; `daemon.py:6`; `classifier.py:6`; `README.md:9`; `newsletter.py:4-5`; `daemon.py:416-419`; `newsletter.py:486-498`

**Evidence:** The invariant is stated absolutely: CLAUDE.md:15 "Person email bodies NEVER leave the local network", repeated in daemon.py:6 and classifier.py:6. The newsletter path sends the FULL thread transcript to the cloud with no sender-type check: daemon.py:416-419 `transcript = format_thread_transcript(messages, ...)` then `await newsletter_classifier.classify_newsletter(transcript)` under cloud_sem. Routing needs only one message in the thread to To/Cc-match the recipient (newsletter.py:487 "Check if any message in a thread was sent to the newsletter address"), so person replies elsewhere in that thread go to the cloud too. The only justification is a parenthetical in newsletter.py:5: "(newsletter content is not privacy-sensitive)" — a scoping the invariant's own documentation never mentions.

**Why it matters:** The project's number-one stated invariant and its number-one exception live in different files and never reference each other. A maintainer hardening the privacy path (or widening is_newsletter, or extending the newsletter transcript) has no way to know whether cloud-bound person replies inside newsletter threads are an accepted trade-off or a latent violation — and either fix direction looks wrong from one of the two documents.

**Verifier note:** Verified: the invariant is stated absolutely in CLAUDE.md:15, README.md:9, daemon.py:6 and classifier.py:6, yet daemon.py:416-419 sends format_thread_transcript of ALL messages to the cloud with no sender-type check whenever any single message To/Cc-matches the recipient (newsletter.py:487-498), so person replies inside a newsletter thread go to the cloud; the sole scoping is the parenthetical at newsletter.py:5 "(newsletter content is not privacy-sensitive)", which none of the invariant's own statements reference.

### 22. [MEDIUM | tension | confirmed] Daemon-wide halt vs per-pipeline providers

**Locations:** `daemon.py:197-206`; `daemon.py:622-629`; `daemon.py:867`; `llm_client.py:112-122`; `newsletter.py:27`; `newsletter.py:552-554`; `CLAUDE.md:113`; `README-technical.md:66-67`

**Evidence:** The halt's rationale assumes one provider: daemon.py:200-201 "an out-of-funds provider fails EVERY request", CLAUDE.md:113 "account-wide, not a poison thread: the daemon halts polling entirely". But llm_client.py:115-116 states the accurate scope — "every subsequent request to the same provider" — and the design explicitly supports a second provider for newsletters (README-technical.md:66: "Set when that model needs a different provider than the cloud classifier — e.g. a Claude model via Anthropic's..."). An LLMBalanceError from the newsletter provider propagates (newsletter.py:27, 552-554) and trips the shared DaemonHalt (daemon.py:627-628), halting email classification on the still-funded cloud provider — and vice versa. The newsletter client is also constructed with `tier="cloud"` (daemon.py:867), so tier-based diagnostics conflate it with the email cloud LLM.

**Why it matters:** In the shipped hybrid configuration the halt's premise ("every request fails") is false — half the daemon's requests would succeed. A maintainer cannot tell whether halting both pipelines on one provider's balance fault is a deliberate simplification or an unexamined leftover from the single-provider era, so any fix to halt behavior risks re-litigating an undocumented decision; the "add funds to the provider account" instruction also doesn't say which account in a two-provider deployment.

**Verifier note:** Verified: daemon.py:200-201 and CLAUDE.md:113 frame the halt as "fails EVERY request"/account-wide while llm_client.py:115-116 correctly scopes it to "the same provider"; with the documented two-provider setup (README-technical.md:66-67, NEWSLETTER_LLM_URL) an LLMBalanceError propagates from the newsletter pipeline (newsletter.py:27, 552-554) and trips the shared halt (daemon.py:627-628), stopping the still-funded email pipeline — and daemon.py:867 tags the newsletter client tier="cloud", conflating it with the email cloud LLM in diagnostics.

### 23. [MEDIUM | tension | confirmed] cloud_parallel semantics diverge across pipelines

**Locations:** `daemon.py:418-419`; `daemon.py:533-534`; `daemon.py:541-542`; `newsletter.py:556-591`; `config.toml:12`; `README-technical.md:94`

**Evidence:** README-technical.md:94 defines it as "cloud_parallel = 2             # Max concurrent cloud LLM requests". The email pipeline matches that: cloud_sem is acquired per LLM call (daemon.py:533, 541). The newsletter path holds one acquisition across the entire grading — daemon.py:418-419 `async with cloud_sem: story_results = await newsletter_classifier.classify_newsletter(transcript)` — which internally issues 1 + 2×stories sequential LLM calls (newsletter.py:556-591). So one long newsletter occupies a cloud slot for its whole multi-call, multi-minute grading, and when NEWSLETTER_LLM_URL points at a different provider, cloud_parallel is gating that other provider's traffic under the email knob.

**Why it matters:** The knob's documented meaning is only true for one of the two pipelines. In hybrid mode a couple of newsletters can starve email classification for minutes with no doc acknowledging it; a maintainer tuning concurrency (or fixing a rate-limit issue) has to reverse-engineer that "concurrent cloud requests" means different units in each pipeline.

**Verifier note:** Verified: README-technical.md:94 defines cloud_parallel as "Max concurrent cloud LLM requests" and the email path matches (per-call acquisition at daemon.py:533, 541), but daemon.py:418-419 holds one cloud_sem slot across the entire classify_newsletter, which issues 1 + 2×stories sequential LLM calls (newsletter.py:556-591) — so with cloud_parallel=2 (config.toml:12) two newsletters occupy both slots for their whole multi-call grading, starving email Stage 1/2a, and when NEWSLETTER_LLM_URL points elsewhere the knob gates a different provider's traffic.

### 24. [LOW | tension | confirmed] Newsletter LLM fallback recreates documented failure

**Locations:** `daemon.py:869-870`; `config.toml:160-174`; `config.toml:61`

**Evidence:** daemon.py:869-870: if `[newsletter]` exists without `[newsletter.llm]`, `nl_llm = cloud_llm` — the email cloud client with `max_tokens = 1024` (config.toml:61). But config.toml:162-168 documents exactly why that budget is fatal for newsletters: "story_extraction re-emits the FULL text of every story ... 1024 truncated multi-story newsletters mid-story ... an undersized budget here would abandon long newsletters to agent/attempted rather than mis-grade them." The fallback state emits no warning and is undocumented.

**Why it matters:** The code offers a configuration state whose behavior the config file itself describes as a bug outcome. A maintainer simplifying config (or a deployment that trims [newsletter.llm]) silently re-enables the issue-#64-class failure the sizing comments exist to prevent — with the only symptom being newsletters drifting into agent/attempted.

**Verifier note:** Verified: daemon.py:869-870 silently falls back to `nl_llm = cloud_llm` (max_tokens=1024, config.toml:61) when [newsletter] exists without [newsletter.llm], with no log in that branch, while config.toml:162-168 documents exactly that budget as the failure mode — "1024 truncated multi-story newsletters mid-story" and "an undersized budget here would abandon long newsletters to agent/attempted" (LLMContentError on finish_reason length is give-up-eligible).

### 25. [MEDIUM | ambiguity | adjusted] no-stories label conflates absence with parse failure

**Locations:** `newsletter.py:565-570`; `daemon.py:421-426`; `daemon.py:478`; `labeler.py:203-207`; `README.md:42`

**Evidence:** README.md:42 defines `agent/newsletter/no-stories` as "No extractable stories found". But a story whose quality reply fails to parse gets scores=None silently — newsletter.py:566 `if scores:` has no else branch and no log (the warnings at 578/587 fire only on exceptions) — leaving `tier=None`. If every story parses to None, daemon.py:421-426 leaves `best_tier=None` and labeler.py:203-207 applies the no_stories label plus agent/processed, permanently. daemon.py:478 logs it as "no-stories" too.

**Why it matters:** Two very different conditions — a genuinely story-less newsletter and a quality-prompt format drift — produce the identical permanent label and log line, with no warning distinguishing them. A maintainer changing the quality prompt or parser could silently convert the whole stream to no-stories; and anyone triaging no-stories mail is told a falsehood by the documented label meaning.

**Verifier note:** The label conflation is real but the "identical log line" claim is refuted: daemon.py:473-478 logs "Newsletter thread %s: %d stories, tier=no-stories", so an all-parse-failure thread logs a nonzero story count, distinguishable from a genuine no-stories thread's "0 stories" (and the JSONL record keeps the stories with scores=null). What holds: newsletter.py:567 `if scores:` (finding cited 566; it is 567) has no else branch and no log, warnings at 578/587 fire only on exceptions, all-None tiers yield best_tier=None (daemon.py:421-426) so labeler.py:203-207 permanently applies no_stories + agent/processed, and README.md:42's "No extractable stories found" is then false — keep as ambiguity, scoped to the Gmail label and the missing parse-failure warning rather than the log line.

### 26. [LOW | tension | confirmed] Newsletter archive hard-coded, priority check bypassed

**Locations:** `labeler.py:216-221`; `labeler.py:134-135`; `config.toml:43-46`; `daemon.py:407-408`; `daemon.py:489-491`; `README.md:25-43`

**Evidence:** The email pipeline's inbox actions are config-driven — labeler.py:134 `action = self.labels_config["actions"][config_key]` from `[labels.actions]` (config.toml:43-46) and documented in README.md:25-29's Action column. apply_newsletter_classification archives unconditionally — labeler.py:220 `remove_label_ids=["INBOX"]` hard-coded — and README.md:33-43's newsletter label table has no Action column and never says newsletters are archived. The newsletter branch (daemon.py:407-408) also runs before the existing-priority/no-downgrade check (daemon.py:489-491), so a newsletter thread carrying e.g. agent/needs-response is archived without the downgrade guard the email path enforces.

**Why it matters:** Whether a mode's inbox action is policy (configurable) or invariant (hard-coded) differs between the pipelines with no statement of intent, and the no-downgrade rule — a named design rule — silently doesn't apply to one pipeline. A fix generalizing actions or the priority guard has to guess whether the newsletter divergence is deliberate.

**Verifier note:** Verified: apply_newsletter_classification hard-codes remove_label_ids=["INBOX"] (labeler.py:220) while email actions are config-driven ([labels.actions], labeler.py:134); README.md's newsletter table (33-43) has Label|Purpose columns only and a grep finds no doc stating newsletters are archived; and the newsletter branch (daemon.py:408) runs before the existing-priority check (daemon.py:489-491), so a thread already labeled agent/needs-response is archived without the no-downgrade guard the email path enforces.

### 27. [MEDIUM | gap | confirmed] Simultaneous dual-daemon operation undefined

**Locations:** `README.md:183-186`; `newsletter.py:296-299`; `CLAUDE.md:52`

**Evidence:** README.md:183-186 offers `NEWSLETTER_ONLY=1 docker compose up email-labeler` "To classify only newsletters (skipping all other emails)" — but since the default mode already grades newsletters (daemon.py:407-408), no document says whether a NEWSLETTER_ONLY instance is meant to replace the normal daemon, run alongside it, or exist only for testing. Two concurrent daemons would race on the same threads (duplicate grading spend, racing Gmail label writes) and append to the same JSONL sink from two processes with no locking (newsletter.py:298 `with open(path, "a") as f:`). No doc anywhere addresses concurrent operation.

**Why it matters:** The audit question "may the two modes ever run simultaneously and what happens if they do" has no answer in the repo. An operator following README.md's newsletter-only command while the normal service keeps running gets double-processing whose safety depends on unstated properties (timestamp dedup, Gmail label idempotence); a maintainer cannot tell which interleavings the design must tolerate.

**Verifier note:** Verified: README.md:183-186 offers the NEWSLETTER_ONLY=1 command with no statement of whether it replaces or accompanies the normal daemon (which already grades newsletters per daemon.py:408), newsletter.py:298 appends to the shared JSONL with plain `open(path, "a")` and no locking, and greps for simultaneous/concurrent/alongside operation across README.md, README-technical.md, CLAUDE.md and docs/ find nothing — the safety of two concurrent instances rests entirely on unstated properties (timestamp dedup, label-write idempotence).

### 28. [LOW | gap | confirmed] Per-mode requirements undocumented

**Locations:** `README-technical.md:60-78`; `daemon.py:821-848`; `labeler.py:59-78`

**Evidence:** README-technical.md:64-65 marks CLOUD_LLM_URL / CLOUD_LLM_API_KEY "Required: Yes" unconditionally, and daemon.py:822-847 always constructs both email LLM clients and the EmailClassifier — yet in NEWSLETTER_ONLY mode with NEWSLETTER_LLM_URL set, neither email client is ever called, and MLX_* is idle. Conversely, an email-only deployment still must pre-create all 11 newsletter Gmail labels because the shipped config carries [newsletter.labels] (labeler.py:72-78) or the daemon exits. Nothing documents which env vars, config sections, or Gmail labels each mode actually requires.

**Why it matters:** Every deployment must satisfy the union of both modes' prerequisites without being told so; a maintainer standing up a newsletter-only (or email-only) instance discovers the cross-mode requirements only via startup failures, and a fix that makes any requirement genuinely conditional will surprise deployments relying on the accidental union.

**Verifier note:** Verified: README-technical.md:64-65 marks CLOUD_LLM_URL/CLOUD_LLM_API_KEY "Required: Yes" unconditionally and daemon.py:822-847 always constructs both email LLM clients + EmailClassifier though a NEWSLETTER_ONLY instance with NEWSLETTER_LLM_URL set never calls them; conversely labeler.py:72-78 makes all 11 shipped newsletter labels (config.toml:184-197) startup-fatal via sys.exit(1) at daemon.py:917-921 even for an email-only deployment, README.md:137-156 lists all labels as one undifferentiated required set, and no doc conditions any requirement on mode.

## Key Design Decisions vs implementation

**Dimension summary:** The seven Key Design Decisions are individually well-implemented — decisions 1, 3, 5, 6, and 7 match the code closely, and 5/6 are unusually well-reasoned in both docs and comments. The clarity problems are almost all at the seams: decision 2's "safe defaults" now quietly conflicts with both the flagship privacy invariant (unknown sender → full body to cloud) and the issue-#64 fail-loud philosophy (some unusable LLM outputs raise, others still silently default), with no stated rule for which philosophy governs a new failure mode. The most fix-ripple-prone asset — the transient/permanent/account-wide error taxonomy that decides retry vs give-up vs halt — is not a listed design decision at all; it lives in an order-sensitive except-chain and scattered comments, and one of its own docstring claims ("an endpoint-wide outage never lands here") is falsified by exhausted-5xx/429 storms. Several cross-decision interactions (heartbeat freshness vs long bounded-concurrency cycles, one provider's balance halting all three providers' pipelines, no-web-server's unstated scope next to the eval FastAPI app) are unspecified, so a maintainer fixing one decision has no way to know which neighboring guarantees the fix silently bends.

### 29. [HIGH | tension | confirmed] Safe-default SERVICE vs privacy invariant

**Locations:** `CLAUDE.md:15`; `CLAUDE.md:110`; `classifier.py:107-116`; `README.md:9-17`; `daemon.py:537-542`

**Evidence:** CLAUDE.md:15 states the invariant absolutely: "Person email bodies NEVER leave the local network." Decision 2 (CLAUDE.md:110) states: "Unknown sender type → SERVICE (body goes to cloud, safe for non-person)." parse_sender_type (classifier.py:110-111) defaults to SenderType.SERVICE when the LLM output is unparseable, and daemon.py:541-542 then sends the full transcript to the cloud LLM. README.md:16 justifies cloud routing with "This is safe because service emails ... contain no personal correspondence" — which assumes Stage 1 was right; the default is applied precisely when we do NOT know. No doc acknowledges that a misclassified or unparseable-output person sender results in a person body going to the cloud.

**Why it matters:** The invariant is written as unconditional but the mechanism only guarantees it conditional on Stage 1 correctness, with the failure default leaning toward the cloud. A maintainer changing the Stage 1 prompt, parser, or fallback (e.g. "fix: be more lenient when parsing sender type") cannot tell whether they must fail toward PERSON/local (privacy-first) or SERVICE/cloud (availability-first, since decision 3 skips person mail when MLX is down) — the two goals pull opposite ways and the accepted residual risk is stated nowhere. This tension is inherent (a metadata-only classifier can always be wrong) and needs the residual to be explicitly accepted in writing.

**Verifier note:** The invariant is stated unconditionally (CLAUDE.md:15, README.md:9) but is only guaranteed conditional on Stage 1 being right, and the failure default (classifier.py:110-111 → SERVICE → daemon.py:540-542 full transcript to cloud) leans cloud-ward; no doc anywhere acknowledges the residual or states which way parser/prompt fixes must fail.

### 30. [HIGH | tension | confirmed] Safe-default LOW_PRIORITY vs issue-64 fail-loud

**Locations:** `CLAUDE.md:110`; `classifier.py:162-169`; `llm_client.py:288-311`; `config.toml:70-75`

**Evidence:** Decision 2 (CLAUDE.md:110): "Unknown email label → LOW_PRIORITY (archived, not deleted)." parse_email_label still does this for non-empty unparseable output (classifier.py:164: "result = EmailLabel.LOW_PRIORITY" with only a log.warning). But the #64 fix declares silent defaulting a bug for adjacent shapes: llm_client.py:298-300 "Both empty shapes would otherwise parse to a default SERVICE / LOW_PRIORITY label and silently mislabel the email", and raises LLMContentError on empty/think-only/length-truncated responses; config.toml:72-73 "it can no longer silently parse to the default LOW_PRIORITY → archive". So a complete-but-keyword-free reply is silently archived, while a truncated reply is a raise → give-up → agent/attempted.

**Why it matters:** Two contradictory philosophies now coexist for "the model didn't give a usable label": default-and-archive (decision 2) versus raise-and-give-up (issue #64), with the boundary defined only implicitly by which code path detects the problem. The next parse-failure mode discovered (e.g. the label word appearing only inside reasoning prose) has no rule saying which side it belongs on, so any fix will be judged inconsistent against one of the two philosophies — exactly the fix-spawns-review-findings symptom. Resolvable: state the boundary (e.g. "defaults apply only to well-formed answers with an unknown token; unusable responses always raise") in decision 2.

**Verifier note:** classifier.py:163-168 still silently defaults a non-empty keyword-free reply to LOW_PRIORITY→archive while llm_client.py:288-333 raises LLMContentError on empty/think-only/length responses explicitly because silent defaulting 'would... silently mislabel the email' — the default-vs-raise boundary exists only implicitly in which code path detects the problem, never as a stated rule in decision 2.

### 31. [HIGH | gap | adjusted] Give-up error taxonomy only in code comments

**Locations:** `daemon.py:574-635`; `llm_client.py:80-87`; `llm_client.py:117-122`; `newsletter.py:23-27`; `CLAUDE.md:107-115`

**Evidence:** The design that decides retry-next-cycle vs count-toward-give-up vs halt is encoded solely in an ordered except-chain (daemon.py:574-635) plus class-hierarchy tricks: LLMContentError "subclasses RuntimeError so the email pipeline's except RuntimeError give-up handler catches it unchanged" (llm_client.py:83-85); LLMBalanceError "the daemon must catch it *before* its except RuntimeError arm" (llm_client.py:121-122) and daemon.py:623-624 "must precede the RuntimeError arm, which it subclasses"; newsletter.py:23-27 defines _PIPELINE_WIDE_ERRORS to re-raise the same trio. CLAUDE.md's Key Design Decisions list covers only the halt arm (decision 5); the overall taxonomy (LLMUnavailable never counts, ProxyUnavailable/Timeout/RuntimeError count, ordering constraints) is not a listed decision anywhere.

**Why it matters:** This taxonomy is the highest-leverage, most fragile design decision in the codebase: introducing or re-parenting any exception type silently changes whether threads are retried forever, burned into agent/attempted, or halt the daemon (issue #64 did exactly this by making LLMContentError a RuntimeError). A maintainer adding a new error type has no design-level statement to check against — only long comments spread across four files — so every error-handling fix must reconstruct the intent from scratch, and reviewers keep finding taxonomy violations after the fact.

**Verifier note:** Corrected claim: the decisions list covers TWO arms, not one — decision 5 (halt) and decision 3 (local-outage retry-next-cycle, 'skipped (retried next cycle)'), and the Labels section names agent/attempted's give-up role; but the counting taxonomy itself (LLMUnavailable never counts for any tier, ProxyUnavailable/Timeout/RuntimeError count, the LLMBalanceError-before-RuntimeError ordering constraint, LLMContentError's deliberate RuntimeError parentage, newsletter's _PIPELINE_WIDE_ERRORS re-raise trio) lives solely in comments across daemon.py:574-635, llm_client.py:83-85/119-121, newsletter.py:23-27 with no design-level statement to check changes against. Gap real; severity high stands.

### 32. [MEDIUM | contradiction | confirmed] FailureTracker endpoint-wide claim vs 5xx/429 storms

**Locations:** `daemon.py:148-151`; `retry.py:17-19`; `retry.py:62-73`; `llm_client.py:266-281`; `daemon.py:630-632`

**Evidence:** FailureTracker's docstring claims "An endpoint-wide outage never lands here" (daemon.py:148), citing ConnectError→LLMUnavailableError and cycle-level proxy failure as the reasons. But an endpoint-wide LLM outage that answers HTTP 503 (or a sustained 429 throttle) exhausts retry_with_backoff (retry.py:62-73 returns "the last retryable failure"), llm_client.py:277 then raises a plain RuntimeError ("LLM request failed with status ..."), and daemon.py:630-632's RuntimeError arm counts it toward give-up — so after max_failures cycles the entire backlog is marked agent/attempted by an outage that is unambiguously endpoint-wide, not thread-specific.

**Why it matters:** The docstring's blanket claim and the RuntimeError arm's behavior cannot both be true; only connection-level outages are actually exempt from give-up. A maintainer trusting the docstring would (a) not expect a provider 5xx incident to mass-abandon threads, and (b) when fixing that, not know whether counting HTTP-level storms was a deliberate accepted residual (as it explicitly is for the proxy, daemon.py:601-607) or an oversight. Either the claim or the behavior needs to change; a comment cannot reconcile "never" with "does".

**Verifier note:** daemon.py:148's 'An endpoint-wide outage never lands here' is disproven by the code path it doesn't cover: a sustained LLM 503/429 exhausts retry_with_backoff (retry.py:62-73 returns the last failure), llm_client.py:277-281 raises plain RuntimeError, and daemon.py:630-632 counts it toward give-up — so an HTTP-answering endpoint-wide outage mass-abandons the backlog to agent/attempted after max_failures cycles, with the residual explicitly accepted for the proxy (daemon.py:601-607) but affirmatively denied for the LLM.

### 33. [MEDIUM | ambiguity | confirmed] MLX 'down' vs slow boundary undefined

**Locations:** `CLAUDE.md:111`; `llm_client.py:14-18`; `llm_client.py:248-255`; `daemon.py:616-621`; `config.toml:83`

**Evidence:** Decision 3 (CLAUDE.md:111): "If local MLX is down, person emails are skipped (retried next cycle). Privacy invariant preserved." In code, "down" means connect-level only: read/write timeouts raise TimeoutError, documented as "request-specific (e.g. a transcript too large to prefill within the timeout)" (llm_client.py:250-252), and daemon.py:616-621 makes them give-up-eligible → agent/attempted. Yet llm_client.py:15-18 itself acknowledges the opposite for probe(): "a server that loads models on demand ... a cold load of a large model routinely exceeds 10s, and timing out would wrongly report it unreachable" — the same cold-load/slow-server condition in complete() is a read timeout attributed to the request, and counts toward abandoning the thread.

**Why it matters:** Whether a slow-but-up local server counts as "down" (retry forever, decision 3) or as a per-thread fault (give up after 5 strikes) determines whether person emails can be abandoned to agent/attempted during warm-up or load spikes — a state decision 3 implies cannot happen ("skipped, retried next cycle"). A maintainer tuning the 180s local timeout (config.toml:83) or fixing a cold-start complaint has no stated definition of the down/slow boundary to preserve, so any fix moves threads across the retry/give-up line unreviewed.

**Verifier note:** 'Down' in code means connect-level only: llm_client.py:248-255 attributes read timeouts to the request and daemon.py:616-621 makes them give-up-eligible, while llm_client.py:14-18 itself concedes for probe() that a slow-but-up server (cold model load) is routinely mistaken for unreachable — so a local server that stays slow across max_failures cycles abandons person threads to agent/attempted, a state decision 3's 'skipped (retried next cycle)' implies cannot happen, and no doc defines the down/slow boundary.

### 34. [MEDIUM | ambiguity | adjusted] No-web-server decision scope and rationale unstated

**Locations:** `CLAUDE.md:112`; `evals/web_app.py:1-26`; `evals/run_web.py:1-20`; `evals/README.md:142-151`

**Evidence:** Decision 4 (CLAUDE.md:112) reads absolutely: "**No web server**: Pure asyncio daemon. Health check via file timestamp + Docker HEALTHCHECK." But the repo ships a FastAPI/uvicorn web server: evals/web_app.py ("app = FastAPI(title=\"Email Labeler Eval Suite\")") launched by evals/run_web.py, documented in evals/README.md:142-151 ("## 5. Web UI — Interactive reporting and comparison ... Navigate to `http://localhost:5000`"). Neither CLAUDE.md nor the READMEs state the decision's scope (daemon-process-only?) or its rationale (attack surface on the Gmail-write host? simplicity? container footprint?).

**Why it matters:** Without the rationale, the decision cannot guide new work: is a status/metrics HTTP endpoint on the daemon forbidden (breaks "no web server") or fine (the evals precedent shows web servers are acceptable in this repo)? Is the constraint about the daemon container, the host, or credential-bearing processes? A maintainer adding observability — a natural follow-up to the halted-vs-hung healthcheck subtleties — has to guess the design intent, and either answer can be attacked in review.

**Verifier note:** Corrected claim: the scope is reasonably supplied by the decision's own wording — 'Pure asyncio daemon' scopes 'No web server' to the daemon process, so the evals FastAPI/uvicorn server (evals/web_app.py:24, evals/run_web.py, a dev-side tool) is not a genuine scope conflict; the surviving gap is only the missing rationale, which leaves 'may the daemon grow an HTTP status/metrics endpoint, and why not?' unanswerable. Kind ambiguity stands but narrower; severity low, not medium.

### 35. [MEDIUM | ambiguity | confirmed] Halt is daemon-wide across independent providers

**Locations:** `CLAUDE.md:113`; `daemon.py:197-217`; `daemon.py:622-629`; `daemon.py:855-871`; `newsletter.py:542-555`; `README-technical.md:377-391`

**Evidence:** Decision 5 calls the fault "account-wide" (CLAUDE.md:113) and README-technical.md:380-385 says "the fault is account-wide, so per-thread retries would only burn the backlog". But the daemon runs up to three LLMClient instances against potentially three different provider accounts (cloud, local MLX "or public API stand-ins like Novita.ai", and the newsletter endpoint — CLAUDE.md's NEWSLETTER_LLM_URL doc says "Set when the newsletter model needs a different provider"). DaemonHalt (daemon.py:197-217) has no provider dimension, and any LLMBalanceError — including one propagated from the newsletter pipeline (newsletter.py:552-555) — trips it (daemon.py:627-628), halting ALL polling. So an exhausted Anthropic newsletter account halts email classification running on a fully funded cloud account, and vice versa. The newsletter client is even constructed with tier="cloud" (daemon.py:867) despite possibly being a different provider.

**Why it matters:** "Account-wide" justifies halting requests to THAT account, but the implementation escalates it to daemon-wide across unrelated accounts, and no doc says whether that over-reach is a deliberate simplification or a leftover from single-provider days. A maintainer fielding the complaint "newsletter credit ran out and my email stopped being labeled" cannot tell if scoping the halt per-provider would be a fix or a design violation; decision 5's stated rationale ("fails EVERY request") is simply untrue for the other providers' requests.

**Verifier note:** Verified end to end: DaemonHalt (daemon.py:197-217) carries no provider dimension, any LLMBalanceError — including one propagated from the newsletter pipeline (newsletter.py:552-555) — trips it (daemon.py:622-628) and halts ALL polling, while CLAUDE.md:133 explicitly supports the newsletter endpoint being a different provider (and daemon.py:867 mislabels that client tier="cloud"); decision 5's rationale 'fails EVERY request' (daemon.py:200, 939) is untrue for the other providers' requests, and no doc says the cross-provider over-reach is deliberate.

### 36. [MEDIUM | tension | confirmed] Healthcheck freshness vs long busy cycles

**Locations:** `CLAUDE.md:112`; `CLAUDE.md:115`; `Dockerfile:18-20`; `daemon.py:983-1020`; `config.toml:81-83`; `retry.py:59-91`

**Evidence:** The heartbeat is written only after a full cycle's gather completes (daemon.py:1020, after the gather at 983-1007), and Docker requires it fresher than 180s (Dockerfile:19-20: "... -lt 180"). Decision 7 serializes local work (local_parallel=1) with a 180s local timeout (config.toml:83) and up to 10 threads/cycle, and retry_with_backoff can sleep ~62s per request while holding a semaphore slot (retry.py:79, MAX_RETRIES=5, BASE_DELAY=2.0) — so a single legitimate busy cycle (e.g. several slow person threads, or a 429 storm) can run many minutes with a stale heartbeat, flagging the container unhealthy. Meanwhile decision 5 invests specifically in the opposite guarantee for the halted state: "keeps the healthcheck heartbeat fresh (halted by design, not hung)" (CLAUDE.md:113, daemon.py:949-956).

**Why it matters:** Decisions 4 (health = file timestamp), 7 (bounded concurrency serializes a cycle), and the retry/timeout budgets jointly determine whether a busy daemon reads as hung, but no document specifies the invariant "a healthy cycle always finishes within the freshness window" or acknowledges it can be violated. A fix that raises max_tokens/timeout/MAX_EMAILS_PER_CYCLE (all knobs recently touched by #64) can silently push healthy cycles past 180s — a new problem created by an unrelated fix, discovered only in review or production. Looks resolvable (mid-cycle heartbeats or a stated budget), but today the pieces conflict.

**Verifier note:** The heartbeat is written only after the cycle's gather (daemon.py:1020) yet Docker demands <180s freshness (Dockerfile:19-20) while a single local request may legitimately run up to the 180s timeout (config.toml:83) serialized under local_parallel=1 across up to 10 threads, plus ~62s of in-semaphore retry sleeps (retry.py MAX_RETRIES=5/BASE_DELAY=2.0) — so a healthy busy cycle can flag the container unhealthy, and the only freshness-during-long-state investment is for the halted state (daemon.py:949-956); no doc states a cycle-duration budget.

### 37. [MEDIUM | gap | confirmed] No-downgrade rule and VIP routing missing from decisions

**Locations:** `labeler.py:25-30`; `daemon.py:544-556`; `classifier.py:197-211`; `classifier.py:300-304`; `CLAUDE.md:107-115`; `CLAUDE.md:127-141`

**Evidence:** Two behavior-defining decisions exist only in code: (1) the no-downgrade rule — labeler.py:26 "# Priority ordering: higher index = higher priority. Never downgrade." enforced at daemon.py:544-556 ("skipping downgrade" then mark_processed, which permanently removes the thread from the query); (2) VIP senders — classifier.py:199 reads a VIP_SENDERS env var, VIPs are unconditionally PERSON with no LLM call (classifier.py:239-240, 252-253), restricted to NEEDS_RESPONSE/FYI, and clamped to FYI on violation (classifier.py:302-304). Neither appears in CLAUDE.md's Key Design Decisions (107-115), and VIP_SENDERS/USER_NAME are absent from CLAUDE.md's Environment Variables list (127-141) though README-technical.md:71-72 documents them.

**Why it matters:** Both are exactly the kind of implicit invariant a fix trips over: a re-classification feature that ignores no-downgrade breaks a deliberate one-way ratchet (and mark_processed makes the skip permanent — an unstated corollary); a Stage-1 change that routes VIPs through the LLM breaks the always-PERSON/never-low-priority guarantee. An agent following CLAUDE.md (whose env-var list reads as authoritative) would not know VIP_SENDERS exists at all, so tests or fixes written from the documented model will be wrong in review.

**Verifier note:** Both invariants exist only in code: labeler.py:25-30's never-downgrade ordering enforced at daemon.py:544-556 with a permanent mark_processed on skip, and VIP_SENDERS (classifier.py:198-199) short-circuiting to PERSON with no LLM call (239-240, 252-253) plus the FYI clamp (301-303); grep confirms zero mentions of VIP or downgrade in CLAUDE.md (including its env-var list, 127-141) or README.md, while README-technical.md:71-72 documents both env vars.

### 38. [LOW | gap | confirmed] Decision 7 omits fetch/write semaphores and retry-hold

**Locations:** `CLAUDE.md:115`; `daemon.py:890-899`; `config.toml:22-31`; `llm_client.py:237`; `daemon.py:533-542`

**Evidence:** Decision 7 (CLAUDE.md:115) describes bounding by "the `cloud_parallel`/`local_parallel` semaphores", but the implementation has four (daemon.py:890-895 adds fetch_sem and write_sem; config.toml:22-31 documents their sizing rationale). Two related unstated facts: retry backoff runs INSIDE the semaphore (daemon.py:533-534 `async with cloud_sem:` wraps classify_sender, whose complete() call at llm_client.py:237 sleeps through retry_with_backoff), so a throttled request occupies a concurrency slot for its full ~62s+ of backoff; and the env-override surface is inconsistent — LOCAL_PARALLEL/WRITE_PARALLEL/MAX_EMAILS_PER_CYCLE exist but there is no CLOUD_PARALLEL or FETCH_PARALLEL, with no stated reason.

**Why it matters:** Someone tuning concurrency from decision 7's description alone will reason about two semaphores when four interact, and will not anticipate that retry sleeps make effective parallelism collapse under throttling (which also lengthens cycles — see the healthcheck finding). The asymmetric env overrides leave a maintainer unable to tell whether adding FETCH_PARALLEL would fill an accidental hole or violate a deliberate keep-it-in-config choice.

**Verifier note:** Decision 7 names two semaphores but daemon.py:890-895 creates four (fetch_sem/write_sem, sizing rationale only in config.toml:22-31; fetch_parallel absent from CLAUDE.md entirely); retry backoff sleeps INSIDE the held semaphore (daemon.py:533-534 wraps classify_sender → llm_client.py:237 retry_with_backoff), and the env-override surface is asymmetric with no stated reason (LOCAL_PARALLEL/WRITE_PARALLEL/MAX_EMAILS_PER_CYCLE exist; CLOUD_PARALLEL/FETCH_PARALLEL do not — daemon.py:890, 893 read config only).

### 39. [LOW | gap | confirmed] Persistent sink fault ends at attempted, not retry

**Locations:** `CLAUDE.md:114`; `daemon.py:436-439`; `daemon.py:453-464`; `daemon.py:633-635`; `newsletter.py:464-467`

**Evidence:** Decision 6 (CLAUDE.md:114) says a sink fault "leaves the thread unprocessed and retried rather than labeled-but-lost". The code adds a terminal state the decision text omits: the OSError is re-raised (daemon.py:464), caught by the generic Exception arm (633-635), counted, and "a persistent fault ends at the give-up path's findable agent/attempted" (daemon.py:438-439); newsletter.py:464-466's warning says the same ("eventually marked agent/attempted rather than graded into a record that cannot be saved"). At that point the thread is excluded from gmail_query with NO assessment record and NO tier labels — lost to the pipeline unless the owner runs the manual runbook.

**Why it matters:** The decision's stated dichotomy (retried vs labeled-but-lost) hides the actual third outcome (attempted-and-ungraded after 5 cycles). A maintainer reasoning from CLAUDE.md would believe a broken sink can never lose gradings, and might e.g. weaken the startup preflight or lengthen give-up thresholds without realizing they are the only guards on that path. Low because the code comments do disclose it — but the decision text is where an agent looks first.

**Verifier note:** Decision 6's dichotomy ('retried rather than labeled-but-lost') omits the actual terminal state: the OSError re-raised at daemon.py:464 falls to the generic Exception arm (633-635) and counts toward give-up, so a persistent sink fault ends at agent/attempted with no assessment record and no tier labels — disclosed only in code comments (daemon.py:437-439, newsletter.py:463-467), not in the decision text an agent reads first.

### 40. [LOW | ambiguity | confirmed] Snippet is body-derived metadata

**Locations:** `CLAUDE.md:15`; `README.md:13`; `daemon.py:520`; `classifier.py:241-249`

**Evidence:** CLAUDE.md:15: "Person email bodies NEVER leave the local network. Cloud LLM only sees metadata (sender, subject, snippet)". A Gmail snippet is the opening text of the message body; daemon.py:520 takes `messages[-1].get("snippet", "")` and classifier.py:241-249 sends it to the cloud for every email, including known-person ones. Both docs classify the snippet as "metadata" by fiat; neither defines how much body-derived preview text may go to the cloud before the invariant is violated.

**Why it matters:** The privacy invariant's key term — "body" — is undefined at its boundary. A plausible accuracy fix ("give Stage 1 the first 500 chars, snippets are too short") either violates or honors the invariant depending on an unwritten rule, so the decision can only be litigated in review rather than checked against the docs. Low because the snippet exposure itself is disclosed; the gap is the missing definition.

**Verifier note:** Verified: daemon.py:520 takes the latest message's Gmail snippet (opening body text) and classifier.py:241-249 sends it to the cloud for every non-VIP sender — including senders repeatedly classified person before, since nothing is cached — while CLAUDE.md:15 and README.md:13 classify it as 'metadata' by fiat; no doc defines how much body-derived preview text may go cloud-ward before the invariant is violated. (Minor: VIP senders skip the LLM call, the sole exception to 'every email' — does not affect the missing-definition point.)

## Test suite as behavior spec

**Dimension summary:** The suite is unusually intent-rich: most tests carry docstrings that state the design rationale (often citing issue/review numbers), boundary values match the documented tier thresholds, and nearly every CLAUDE.md behavior has a named regression test — halt (TestOutOfFundsHalt, test_balance_error_trips_daemon_halt), record-before-labels (test_record_is_on_disk_before_labels_are_applied), dedup-by-timestamp (test_newest_wins_by_timestamp_not_by_file_position), give-up→agent/attempted (test_gives_up_on_thread_after_repeated_failures, TestMarkAttempted), safe defaults (test_unknown_defaults_to_service/low_priority), and MLX degradation (test_person_thread_returns_false_when_mlx_unreachable). The headline miss is that the project's #1 documented invariant — person email bodies never reach the cloud LLM — has no test that would fail if it regressed: routing is asserted only as mock call counts, and no assertion anywhere checks that body/transcript text is absent from a cloud call. The other systematic weakness is mock-bounded seams: run_daemon's wiring of the optional safety collaborators (failure_tracker, fetch/write sems) and the daemon↔EmailClassifier contract are each tested only with the far side mocked, so a wiring or signature fix can pass the whole suite while breaking production — a direct generator of the "fix reviewed → new problems found" symptom. Finally, the doc meta-tests enforce presence (not accuracy) and only in README-technical.md, leaving the agent-facing CLAUDE.md duplicates to drift, and a handful of mirrored-constant tests pin tunables without rationale, creating fix-breaks-test friction.

### 41. [HIGH | gap | confirmed] Privacy invariant has no failing test

**Locations:** `daemon.py:520-542`; `tests/test_classifier.py:292-301`; `tests/test_classifier.py:320-336`; `tests/test_daemon.py:45-55`; `tests/test_daemon.py:149-184`

**Evidence:** CLAUDE.md's top invariant: "Person email bodies NEVER leave the local network. Cloud LLM only sees metadata (sender, subject, snippet)". The tests encode only routing-by-call-count: test_person_routes_to_local_llm asserts `mock_local_llm.complete.assert_called_once()`; test_routes_to_cloud_llm asserts the stage-1 prompt CONTAINS sender/subject (`assert "John Doe <john@example.com>" in call_args.args[1]`) but never that body text is ABSENT. A grep for body-absence assertions (`body not in` / `transcript not in` against a cloud mock) finds zero hits in tests/. In daemon.py the boundary is one line — `snippet = messages[-1].get("snippet", "")` (daemon.py:520) built right next to `transcript = format_thread_transcript(...)` (daemon.py:530) — and daemon tests use a fully AsyncMock classifier, so a change passing the transcript into ThreadMetadata.snippet (a plausible "improve stage-1 accuracy" fix) would leak person bodies to the cloud with a green suite.

**Why it matters:** A maintainer touching stage-1 prompting or the metadata build cannot answer "which test protects the privacy guarantee?" — the answer is none. The invariant is enforced only by convention (EmailMetadata/ThreadMetadata happen to lack a body field), so any fix that widens what stage 1 sees looks safe to both the author and CI, and the violation only surfaces in later review — the exact fix-creates-problems symptom, on the system's most important property.

**Verifier note:** Verified at every citation: the system's top invariant is enforced only by ThreadMetadata/EmailMetadata lacking a body field (classifier.py:32-46) — no test in tests/ asserts body/transcript ABSENCE from any cloud-LLM call (grep confirms zero hits), stage-1 tests assert only presence of sender/subject (test_classifier.py:300-301), and daemon tests mock the classifier entirely, so stuffing the transcript into ThreadMetadata.snippet at daemon.py:520-527 would leak person bodies to the cloud with a fully green suite. (test_eval_report's 'privacy violation metrics' is eval-metrics arithmetic, not a guard on daemon code.)

### 42. [MEDIUM | gap | confirmed] run_daemon wiring of optional collaborators untested

**Locations:** `tests/test_daemon.py:1062-1073`; `daemon.py:326-341`; `daemon.py:984-1004`

**Evidence:** Every safety mechanism of process_single_thread is an optional kwarg defaulting to None (`failure_tracker: "FailureTracker | None" = None`, likewise fetch_sem, write_sem, local_deferrals; daemon.py:237 `if failure_tracker is None: return False` silently disables give-up). The only tests that drive run_daemon (run_poll_cycles helper) replace the seam entirely: `monkeypatch.setattr(daemon, "process_single_thread", ... AsyncMock(return_value=True))` plus `monkeypatch.setattr(daemon, "EmailClassifier", MagicMock())` etc. (tests/test_daemon.py:1065-1073). So deleting `failure_tracker=failure_tracker,` from the gather call at daemon.py:998 — disabling the whole agent/attempted give-up path in production — fails no test. (halt wiring is indirectly covered because _out_of_funds_process reads kwargs["halt"]; the others are not.)

**Why it matters:** The per-thread behavior tests all pass the collaborator explicitly, so the suite proves "give-up works WHEN a tracker is supplied" but never "the daemon supplies one". Combined with None-means-silently-off defaults, any refactor of the poll loop can sever a documented behavior (bounded writes, give-up, deferral counting) invisibly — problems that only appear in review or production after a fix.

**Verifier note:** Verified: every safety collaborator is a None-default kwarg whose absence silently disables the feature (daemon.py:327-335; :237-238 'if failure_tracker is None: return False'), and the only tests driving run_daemon (run_poll_cycles) monkeypatch process_single_thread and EmailClassifier away (test_daemon.py:1065-1073) — only kwargs['halt'] is ever read by a process mock (line 1162), so deleting failure_tracker=failure_tracker (or fetch_sem/write_sem/local_deferrals) from the gather at daemon.py:998-1001 fails no test.

### 43. [MEDIUM | gap | confirmed] local_sem gating and local_parallel=1 unguarded

**Locations:** `daemon.py:533-542`; `tests/test_daemon.py:712-777`; `tests/test_daemon.py:1458-1463`; `config.toml:21`

**Evidence:** CLAUDE.md design decision 7: "`local_parallel` defaults to **1** ... concurrent requests can exceed the GPU's Metal working set and OOM-crash the local server." fetch_sem and write_sem each have an exhausted-semaphore test proving the gate exists (test_get_thread_is_bounded_by_fetch_sem, test_label_application_is_bounded_by_write_sem, tests/test_daemon.py:712-777), but the `async with local_sem:` / `async with cloud_sem:` blocks around classification (daemon.py:533-542) have no analogous test — removing them passes the suite. The config guard asserts only `config["daemon"]["local_parallel"] >= 1` (tests/test_daemon.py:1463), so bumping config.toml's `local_parallel = 1` to 8 also passes, despite the documented OOM rationale for 1.

**Why it matters:** The suite pins some concurrency bounds hard and leaves the one with the documented crash consequence (local KV-cache OOM) untested, so a maintainer cannot tell from the tests which bounds are load-bearing. A cleanup that restructures the classify block can drop the local gate with no red test, and the failure mode (GPU OOM on long transcripts) only appears in production.

**Verifier note:** Verified: fetch_sem and write_sem each get an exhausted-Semaphore(0) test proving the gate (test_daemon.py:723, :763), but no test exhausts local_sem or cloud_sem around the classify block (daemon.py:533-542) — removing those 'async with' lines passes the suite — and the config guard asserts only local_parallel >= 1 (test_daemon.py:1463), so the documented OOM-critical value of 1 (config.toml:13-21) is unpinned.

### 44. [MEDIUM | gap | confirmed] gmail_query marker exclusions unpinned

**Locations:** `config.toml:5`; `tests/test_daemon.py:1419-1424`; `tests/test_daemon.py:258-269`; `tests/test_daemon.py:1504-1527`

**Evidence:** Many give-up/mark tests justify themselves by the query: "mark_processed must cover the FULL thread ... otherwise the unmarked sibling keeps re-matching the query" (tests/test_daemon.py:265-269). But the query itself — `gmail_query = "in:inbox -label:agent/processed -label:agent/attempted"` (config.toml:5) — is only checked for existence: `assert "gmail_query" in config["daemon"]` (tests/test_daemon.py:1423). Meanwhile TestLoadConfig pins other config values aggressively with rationale (`assert config["llm"]["local"]["max_tokens"] >= 2048`, the exact chat_template_kwargs nesting "is load-bearing", tests/test_daemon.py:1476-1489).

**Why it matters:** The entire processed/attempted machinery is effective only if the query excludes both markers, yet an edit to that string is the one config change no test notices. The inconsistent pinning policy (max_tokens and extra_body nesting are contract; the query is apparently tunable) leaves a maintainer unable to answer "which config values may I change in a fix without breaking design intent?"

**Verifier note:** Verified: the query string that makes the whole processed/attempted machinery effective (config.toml:5) is checked only for key existence (test_daemon.py:1423 'assert "gmail_query" in config["daemon"]') while sibling config values are pinned hard with rationale (max_tokens floors at 1504-1527, load-bearing chat_template_kwargs nesting at 1476-1489), and multiple test docstrings (e.g. 265-269) justify behavior by the query's exclusions — an edit to the string itself reddens nothing.

### 45. [MEDIUM | tension | confirmed] Meta-tests guard README-technical only; CLAUDE.md drifts

**Locations:** `tests/test_env_var_docs.py:13-14`; `tests/test_env_var_docs.py:51-64`; `CLAUDE.md:127-143`; `classifier.py:198`; `README-technical.md:72`; `CLAUDE.md:172-186`

**Evidence:** test_env_var_docs targets one file: `README_PATH = ROOT / "README-technical.md"`. The same env-var list is duplicated in CLAUDE.md (lines 127-143) with no test, and it has already drifted: `VIP_SENDERS` is read by the daemon (classifier.py:198 `os.environ.get("VIP_SENDERS", "")`), documented in README-technical.md:72, and absent from CLAUDE.md's Environment Variables section (the VIP feature is absent from CLAUDE.md entirely). Likewise CLAUDE.md:172 introduces "Test files mirror source files:" and lists ~13 of the 30 test files (omitting e.g. test_newsletter.py, test_retry.py, test_llm_cache.py), while README-technical.md:434-449 carries the fuller table.

**Why it matters:** CLAUDE.md is the agent-facing spec loaded into every session with "instructions OVERRIDE any default behavior", yet it is the unguarded copy. An agent making a fix works from a stale map of features (no VIP path) and of the suite (no idea test_newsletter.py exists), so fixes get made and tested against an incomplete model of intent — then review against the real code surfaces the missed interactions. Looks resolvable (point the meta-tests at both files, or de-duplicate).

**Verifier note:** Verified: test_env_var_docs.py:14 targets only README-technical.md; CLAUDE.md's duplicated env-var list (127-141) omits VIP_SENDERS — and 'vip' appears nowhere in CLAUDE.md — despite classifier.py:198 reading it and README-technical.md:72 documenting it; CLAUDE.md:172-185 lists 13 test files versus 32 actually on disk (test_newsletter.py, test_retry.py, test_llm_cache.py all exist, all omitted), while README-technical.md's Test Coverage table carries the fuller list. Only correction: the on-disk test-file count is 32, not 30 — which strengthens the drift claim.

### 46. [MEDIUM | gap | confirmed] No email-path integration test through real EmailClassifier

**Locations:** `tests/test_daemon.py:45-55`; `tests/test_daemon.py:1067`; `tests/test_daemon.py:1733-1788`; `daemon.py:533-542`

**Evidence:** Every email-pipeline daemon test uses `classifier = AsyncMock()` (fixture, tests/test_daemon.py:46-55), and the poll-loop helper even patches the class: `monkeypatch.setattr(daemon, "EmailClassifier", MagicMock())` (tests/test_daemon.py:1067). The daemon calls `classifier.classify(metadata, transcript, sender_type, sender_raw)` (daemon.py:539); an AsyncMock accepts any signature. The newsletter path, by contrast, has exactly one integration test through the real classifier (test_content_error_routes_to_give_up builds `NewsletterClassifier(cloud_llm=fake_llm, ...)`, tests/test_daemon.py:1765); the email path has zero.

**Why it matters:** If a fix changes EmailClassifier.classify's signature or return contract and updates test_classifier.py to match, the daemon-side tests stay green while production breaks at the seam. Both halves of the pipeline are individually well-tested but never proven to fit together — a classic source of fixes that pass CI and fail review.

**Verifier note:** Verified: every daemon email-path test uses an AsyncMock classifier (test_daemon.py:46-55) and the poll-loop helper patches the class itself (1067); a real EmailClassifier is constructed only in test_classifier.py:284 (unit tests, never through process_single_thread), so the daemon.py:539 call signature/return contract is proven on neither side of the seam — while the newsletter path has exactly one such integration test (real NewsletterClassifier at test_daemon.py:1765).

### 47. [LOW | other | confirmed] Doc meta-tests verify presence, not accuracy

**Locations:** `tests/test_env_var_docs.py:59-64`; `tests/test_eval_cli_docs.py:72-74`; `tests/test_tui_docs.py:278-291`; `tests/test_tui_docs.py:205-265`

**Evidence:** The guarantees are token-presence in a section: `assert var in env_section` (test_env_var_docs.py:61), documented flags = `re.findall(r"`(--[\w-]+)`", section_text)` (test_eval_cli_docs.py:74), and a launch command appearing "in a code block" (test_tui_docs.py:287-291). Notably the pattern is executed unusually well — expectations are derived from disk and the scan itself is unit-tested ("a hole that silently drops a real TUI ... fails here instead of passing vacuously there", test_tui_docs.py:207-210). But nothing checks that a documented default, description, or behavior claim is TRUE: README-technical.md:77's "default `4`" for WRITE_PARALLEL, or any flag description, can be wrong indefinitely.

**Why it matters:** These tests make it structurally impossible to ADD an undocumented flag/env var/TUI, which is real value — but they also confer false confidence: 'docs tests pass' reads as 'docs are right', when only 'docs mention it' is guaranteed. They also shape doc style toward token-satisfying table rows (the minimum that makes the regex pass), so prose accuracy erodes without any red test, and maintainers inherit stale semantics with green docs-checks.

**Verifier note:** Verified: the guarantees are token-presence only — 'assert var in env_section' (test_env_var_docs.py:61), backtick-flag regex (test_eval_cli_docs.py:74), launch command in a code block (test_tui_docs.py:287-291) — with the scan itself well unit-tested (205-265), but no assertion ties any documented default or description to code (README-technical.md:77's 'default 4' for WRITE_PARALLEL could be wrong indefinitely with green checks).

### 48. [LOW | other | confirmed] Mirrored-constant tests pin tunables without rationale

**Locations:** `tests/test_retry.py:168-171`; `tests/test_labeler.py:153-157`

**Evidence:** test_default_constants restates the source verbatim: `assert MAX_RETRIES == 5; assert BASE_DELAY == 2.0; assert RETRYABLE_STATUS_CODES == {429, 502, 503, 504}` — a tautology that tests nothing behavioral (the retryable-status BEHAVIOR is already covered by test_retries_on_502 etc.). test_labeler pins absolute ranks where only ordering is intent: `assert _get_priority(EmailLabel.NEEDS_RESPONSE) == 2` / `== 0`, alongside the ordering test that already encodes the intent (tests/test_labeler.py:149-151). Contrast the well-formed config guards which pin floors with documented reasons ("2048 is the measured floor; shipped value 4096", tests/test_daemon.py:1513).

**Why it matters:** Retuning a constant (raise MAX_RETRIES, insert a priority level) breaks tests that carry no statement of why the old value mattered, so the fixer must guess whether the test encodes intent or just mirrors the code — and usually edits the test to match, which trains everyone to treat red tests as noise. This is a direct, if small, contributor to the every-fix-breaks-tests symptom.

**Verifier note:** Verified: test_retry.py:168-171 restates MAX_RETRIES/BASE_DELAY/RETRYABLE_STATUS_CODES verbatim while the retryable behavior is separately covered (test_retries_on_429/502/503/504 at lines 69-108), and test_labeler.py:154/157 pin absolute ranks (==2, ==0) that nothing in production depends on (daemon compares priorities only relatively) beside the ordering test at 149-151 that already encodes the intent — in contrast to the rationale-bearing config floors at test_daemon.py:1504-1527.

### 49. [LOW | ambiguity | confirmed] Stale priority numbers in daemon test comments

**Locations:** `tests/test_daemon.py:231-232`; `tests/test_daemon.py:314-315`; `tests/test_labeler.py:153-157`; `labeler.py:25-35`

**Evidence:** Comments in the downgrade/upgrade tests say "Existing priority = FYI (2), new = LOW_PRIORITY (1) -> skip" and "Existing priority = FYI (2), new = NEEDS_RESPONSE (3) -> upgrade", but the actual ranks are LOW_PRIORITY=0, FYI=1, NEEDS_RESPONSE=2 (`assert _get_priority(EmailLabel.NEEDS_RESPONSE) == 2`, tests/test_labeler.py:154; `_PRIORITY_ORDER.index(label)`, labeler.py:35).

**Why it matters:** A maintainer reading these comments while modifying the no-downgrade logic infers a 1-based rank scheme that doesn't exist; code that compares against literal rank values written from the comments would be off by one. Small, but exactly the kind of stale in-test documentation that makes a 'simple' fix produce a wrong follow-on change.

**Verifier note:** Verified: test_daemon.py:231 says 'FYI (2), new = LOW_PRIORITY (1)' and :314 says 'FYI (2), new = NEEDS_RESPONSE (3)', but the actual ranks are LOW_PRIORITY=0, FYI=1, NEEDS_RESPONSE=2 (labeler.py:26-30, pinned by test_labeler.py:154,157) — the comments describe a 1-based scheme that has never matched the code the tests exercise.

### 50. [LOW | gap | confirmed] Truncation test never checks oldest-first

**Locations:** `tests/test_daemon.py:142-146`

**Evidence:** test_truncates_oldest_first's only assertion is disjunctive: `assert "[Earlier messages truncated]" in transcript or len(transcript) <= 100`. It never asserts WHICH messages survive, so a regression that truncates newest-first (dropping the latest replies instead of the earliest, while still emitting the marker) passes; the test name promises a behavior the body does not encode.

**Why it matters:** Truncation direction is real intent (classification should see the newest messages of a long thread), and the only artifact recording that intent is a test name whose assertion doesn't enforce it. A refactor of format_thread_transcript can invert the behavior with a green suite, and the next reader can't tell whether oldest-first was ever actually required.

**Verifier note:** Verified: test_truncates_oldest_first's sole assertion is the disjunction '"[Earlier messages truncated]" in transcript or len(transcript) <= 100' (test_daemon.py:146) — it never asserts which message survives, so inverting daemon.py:307-315 to drop newest-first (or even to truncate arbitrarily under 100 chars with no marker) passes; the behavior lives only in the test name.

### 51. [LOW | gap | confirmed] Config label guard omits agent/attempted

**Locations:** `tests/test_daemon.py:1426-1436`; `labeler.py:58-67`; `config.toml:39`

**Evidence:** test_config_has_all_labels enumerates six required keys — "needs_response", "fyi", "low_priority", "processed", "personal", "non_personal" — but labeler.verify_labels requires seven, including "attempted" (labeler.py:64), which config.toml:39 provides. If a config edit dropped the `attempted` key, this guard test stays green and the daemon fails at runtime (KeyError in verify_labels).

**Why it matters:** The guard test lagged the feature it guards (agent/attempted was added later, per the give-up work), showing that the config-contract tests are hand-maintained lists that silently go stale as features land — a maintainer trusting test_config_has_all_labels as 'the required-labels contract' gets an incomplete answer.

**Verifier note:** Verified: test_config_has_all_labels (test_daemon.py:1428-1435) checks six keys and omits 'attempted', while labeler.verify_labels requires it (labeler.py:64, config.toml:39 supplies it); no test reads config['labels']['attempted'] from the real config (test_labeler builds literal label dicts), so deleting the key from config.toml passes the suite and KeyErrors in verify_labels at daemon startup.

## Cross-document consistency

**Dimension summary:** The doc set has a clearly stated role scheme (CLAUDE.md:5-11: README=human overview, README-technical=agent reference, plans=unstated) but does not respect it in practice: the same operational facts are restated in 3-4 places, and several duplicates have already drifted — most seriously, README-technical's canonical LLM-settings block shows a different cloud model, max_tokens, and temperature than config.toml, and is contradicted by its own issue-#64 note 200 lines later. docs/plans/ and docs/newsletter-label-ux-redesign.md have no stated historical-vs-living status; the UX-redesign doc explicitly claims to describe "the way it works now" while documenting an auto-seed feature that issue #59 removed, and two Feb-2026 plan docs open with imperative "For Claude: implement this plan" directives over fully superseded designs. This combination — unmarked stale design records plus multi-copy fact duplication with only partial test-guarding — is a direct mechanism for the reported symptom: any fix that updates one copy of a fact (as the #64 fix did) leaves contradicting copies behind, which the next review then surfaces as "new problems".

### 52. [HIGH | contradiction | confirmed] LLM-settings block drifted from config.toml

**Locations:** `README-technical.md:126-138`; `README-technical.md:349`; `config.toml:49`; `config.toml:61-62`; `config.toml:81-82`

**Evidence:** README-technical's 'LLM settings' block shows `model = "deepseek/deepseek-v3.2"`, `max_tokens = 8096`, `temperature = 0.2` for [llm.cloud] and `max_tokens = 8096`, `temperature = 0.2` for [llm.local]. Actual config.toml has `model = "zai-org/glm-5"`, `max_tokens = 1024`, `temperature = 0` (cloud) and `max_tokens = 4096`, `temperature = 0` (local). README-technical even contradicts itself at line 349: "Note that `[llm.local] max_tokens` was raised 1024 → 4096 (issue #64)" — while its own snippet 200 lines earlier says 8096. The cloud 1024 value was a deliberate #64-era decision (config.toml:52-61 comment; commit 28813bd "Keep cloud max_tokens at 1024").

**Why it matters:** This is the reference doc agents are told to consult, and the drifted numbers are exactly the parameters the #64 fix tuned (budget vs. thinking coupling, KV-cache sizing). A maintainer asking 'what is the intended token budget / temperature?' gets three different answers depending on which line they read, so a 'fix' calibrated against the doc values will contradict the config's carefully-reasoned comments — the precise fix-creates-new-problems pattern reported.

**Verifier note:** README-technical.md:127-137 shows deepseek/deepseek-v3.2, max_tokens 8096, temperature 0.2 for both tiers while config.toml:49/61-62/81-82 ships zai-org/glm-5, 1024/0 (cloud) and 4096/0 (local) — and README-technical:349 itself states the current local value is 4096 ('raised 1024 → 4096 (issue #64)'), irreconcilably contradicting its own snippet 200 lines earlier; commit 28813bd confirms cloud 1024 was deliberate.

### 53. [HIGH | contradiction | confirmed] UX-redesign doc documents auto-seed removed by #59

**Locations:** `docs/newsletter-label-ux-redesign.md:4`; `docs/newsletter-label-ux-redesign.md:29-34`; `docs/newsletter-label-ux-redesign.md:68`; `docs/newsletter-label-ux-redesign.md:97-99`; `docs/newsletter-label-ux-redesign.md:139`; `evals/README-technical.md:188-190`; `evals/README.md:182-183`

**Evidence:** The redesign doc states its role in the present tense: "This documents *why* the tool works the way it does now" (line 4), then records "Two decisions the user made: 1. **Auto-seed on open.** ... the tool automatically runs the production extractor (a live LLM call ...)", lists a browse-mode hotkey "`r` re-seeds (confirm if non-empty)", and an "**Auto-seed safety**" implementation note. But evals/README-technical.md:188-190 says "issue #59 removed the Phase-A auto-seeding and its `--edit`/`--config` flags — the seed never sped curation up, so manual-only became the only mode", and evals/README.md agrees. No annotation in the redesign doc marks decision 1 as reversed.

**Why it matters:** A maintainer consulting the design record — the document that claims to hold current design intent for this TUI — will treat auto-seeding as an owner-made product decision to preserve, and could 'fix' a regression by restoring behavior that was deliberately removed. Owner decisions recorded and later reversed without annotation make it impossible to answer 'is this behavior intended?' from the docs.

**Verifier note:** docs/newsletter-label-ux-redesign.md:4 claims to document 'why the tool works the way it does now' and records auto-seed as owner decision #1 (lines 29-33), an 'r' re-seed hotkey (line 68), an 'Auto-seed safety' note (97-99), and 'open → auto-seed' (139), while evals/README-technical.md:188-190 and evals/README.md:182-183 state issue #59 removed auto-seeding entirely; no annotation in the redesign doc marks the reversal.

### 54. [HIGH | gap | confirmed] docs/plans status unstated; stale plans carry live agent directives

**Locations:** `docs/plans/2026-02-19-newsletter-classification-plan.md:3`; `docs/plans/2026-02-20-newsletter-tui-plan.md:3`; `docs/plans/2026-07-03-tui-framework-evaluation.md:18-20`; `README-technical.md:37-40`; `CLAUDE.md:6-11`

**Evidence:** Nothing anywhere states whether docs/plans/ are historical artifacts or living docs — CLAUDE.md's Documentation list and README-technical's project-structure tree list only `docs/runbook-agent-attempted-recovery.md`, omitting docs/plans/ and docs/newsletter-label-ux-redesign.md entirely. Both Feb-2026 plans open with an imperative: "> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task." while containing fully superseded designs (TITLE:/TEXT: extraction format ×14 occurrences, 1-5 dimension scales, a flat `tui_data.py`/`tui.py` architecture that never shipped). The framework-eval doc itself records prior silent divergence: "docs/plans/2026-02-20-newsletter-tui-design.md originally designed the newsletter review TUI **in Textual**, but the implementation shipped on stdlib `curses`."

**Why it matters:** An agent or human that discovers these plans has no signal for 'executed and superseded' vs 'pending intent' — and one plan explicitly instructs an agent to implement it. Executing or consulting them would reintroduce the pre-#53 rubric and pre-title-removal prompt formats, directly manufacturing the new-problems-per-fix symptom. A one-line status header (or a stated convention that plans are frozen history) is missing.

**Verifier note:** Both Feb-2026 plans open at line 3 with 'For Claude: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task' while containing superseded designs (TITLE: appears 14 times in the classification plan; tui_data.py/tui.py never shipped); no README, CLAUDE.md, or convention anywhere references docs/plans/ or marks any plan as historical, and the framework-eval doc (lines 18-20) records prior silent divergence.

### 55. [MEDIUM | contradiction | confirmed] Feb design contains unannotated reversed decisions

**Locations:** `docs/plans/2026-02-19-newsletter-classification-design.md:129`; `docs/plans/2026-02-19-newsletter-classification-design.md:86`; `docs/plans/2026-02-19-newsletter-classification-design.md:44-52`; `CLAUDE.md:114`; `docs/plans/2026-07-08-phase1-decisions.md:30-33`

**Evidence:** The newsletter design doc's error-handling table says "JSONL write fails | Log error, still apply Gmail labels" — the exact behavior CLAUDE.md Design Decision 6 now calls a data-loss bug and inverts ("the record is written first: a sink fault ... leaves the thread unprocessed and retried rather than labeled-but-lost"). It also specifies "One or more theme labels (union across all stories)", reversed by #53's Emphasized-only rule (phase1-decisions: "apply ... **only when the theme is Emphasized** ... Deliberately changes the existing label's meaning"), and 1-5 scoring with tier bands ≥4.0/≥3.0/≥2.0, replaced by Poor/OK/Good and ≥2.75/≥2.25/≥1.75. None of these reversals is annotated in the design doc.

**Why it matters:** Each reversal was itself a fix (issue #30-adjacent write-ordering, #53 rubric). Because the original decisions still stand unmarked in a 'design' document, any maintainer reconciling code against design will find the code 'wrong' — the question 'which document wins?' has no recorded answer, so fixes keep re-litigating settled decisions.

**Verifier note:** docs/plans/2026-02-19-newsletter-classification-design.md:129 ('JSONL write fails | Log error, still apply Gmail labels'), :86 ('One or more theme labels (union across all stories)'), and :44-52 (1-5 scores, tier bands >=4.0/3.0/2.0) all stand unannotated while CLAUDE.md:114 inverts the write ordering and phase1-decisions.md:30-33 inverts theme labeling (Emphasized-only) and the rubric (Poor/OK/Good, >=2.75/2.25/1.75).

### 56. [MEDIUM | contradiction | adjusted] NEWSLETTER_ONLY pipeline gating mis-stated in CLAUDE.md

**Locations:** `CLAUDE.md:52`; `CLAUDE.md:138`; `README-technical.md:74`; `daemon.py:850-874`; `daemon.py:407-409`; `README.md:63-68`

**Evidence:** CLAUDE.md:52: "When `NEWSLETTER_ONLY=1`, the daemon switches to a newsletter-specific pipeline" and :138 "daemon runs newsletter classification pipeline instead of email labeling". The code enables the newsletter pipeline whenever `[newsletter]` is configured, regardless of the flag (daemon.py:851-874 builds `newsletter_classifier` from `config.get("newsletter")`; daemon.py:408-409 routes any matching thread), and NEWSLETTER_ONLY only narrows the Gmail query / skips non-newsletter threads — as README-technical:74 correctly says: "skip non-newsletter threads. Useful for testing newsletter classification in isolation." README.md's architecture diagram also shows the newsletter branch inside the normal poll loop.

**Why it matters:** CLAUDE.md is loaded into every agent session as authoritative instruction. An agent believing newsletter grading is inert unless NEWSLETTER_ONLY=1 will judge changes to the newsletter path 'safe' for normal deployments (or fail to consider newsletters when fixing the ordinary email path), producing exactly the class of surprise regressions the owner reports. Which behavior is intended is answerable only by reading daemon.py.

**Verifier note:** The code facts hold exactly as cited (daemon.py:851-874 enables the newsletter pipeline whenever [newsletter] is configured; :886-888/:408-409 show the flag only skips non-newsletter threads), but the kind should be misleading-implication/ambiguity, not contradiction: CLAUDE.md:52 and :138 are literally true of flag=1 behavior and never assert the pipeline runs ONLY then, so a one-sentence clarification (newsletter grading is config-gated, the flag merely narrows the queue) reconciles all documents; severity medium stands.

### 57. [MEDIUM | tension | confirmed] Facts duplicated 3-4x across docs with partial test-guarding

**Locations:** `config.toml:84-98`; `README-technical.md:144-171`; `evals/README.md:299-319`; `evals/README-technical.md:69-106`; `CLAUDE.md:115`; `README.md:189-221`; `README-technical.md:192-228`; `README.md:259-281`; `README-technical.md:260-296`

**Evidence:** The thinking-disable dialect story (Ollama `reasoning_effort="none"` vs `chat_template_kwargs.enable_thinking`) is told in full in four places: config.toml comments, README-technical 'Extra request body fields', evals/README 'Thinking on/off A/B', evals/README-technical's A/B section. KV-cache/local_parallel guidance appears in config.toml:13-20, README-technical:101-111, README-technical:342-359, and CLAUDE.md:115. The Docker volume-mount/sink story is in README.md:189-221 and README-technical:192-228; the pre-#53 migration how-to is in README.md:259-281 and README-technical:260-296. Only some copies are test-guarded (test_env_var_docs/test_eval_cli_docs cover README-technical only). Finding 'LLM-settings block drifted' is a duplicate of config.toml that has already drifted.

**Why it matters:** Every fix that touches one of these facts must find and update 3-4 prose copies with no mechanical check on most of them; the #64 fix demonstrably missed one. Duplication is the engine of the reported symptom: it makes 'where is the canonical statement of this fact?' unanswerable, so reviews of any fix keep discovering un-updated copies. Looks resolvable (designate one home per fact + cross-links), not inherent.

**Verifier note:** All cited copies verified: the thinking-disable dialect story appears in config.toml:84-103, README-technical:140-171, evals/README.md (~299-319), and evals/README-technical.md:69-106; KV-cache guidance in config.toml:13-20, README-technical:101-111 and :342-355, CLAUDE.md:115; volume-mount and pre-#53 migration each told twice — and test_env_var_docs.py/test_eval_cli_docs.py mechanically guard only the two README-technical files, leaving the human READMEs and config comments unguarded (the drifted LLM-settings block proves it).

### 58. [MEDIUM | gap | adjusted] Project-structure and test listings omit many modules

**Locations:** `CLAUDE.md:27-38`; `README-technical.md:5-56`; `README-technical.md:430-449`; `CLAUDE.md:143-186`; `docs/plans/2026-07-08-issue-roadmap.md:133`

**Evidence:** CLAUDE.md's Project Structure omits `newsletter.py` (the module implementing the doc's whole 'Newsletter Classification' section), `config_utils.py`, `retry.py`, the entire `evals/` tree, `scripts/eval_model.py`, and `scripts/smoke_concurrency.py`. README-technical's tree omits `retry.py`, `evals/llm_cache.py`, `evals/edit_tui.py`, `evals/run_web.py`, `evals/web_app.py`, `evals/web_auth.py`, `evals/web_data.py`, `evals/templates/` (the Web UI that evals/README.md §5 documents), `docs/plans/`, and `docs/newsletter-label-ux-redesign.md`; its tests/ listing shows 9 of 34 test files and the 'Test Coverage by Module' table 16 of 34 (no rows for test_retry, test_tui_common, test_proxy_client, test_gmail_utils, test_llm_cache, test_eval_review, test_eval_run, test_eval_edit_tui, test_eval_model, test_smoke_concurrency, and the doc-sync tests). The roadmap already knows: #39 "Generalize the doc-completeness guard to the whole repo + backfill missing modules/tests" is filed but deferred to Phase 4.

**Why it matters:** The structure lists are how a maintainer or agent decides where behavior lives and what is covered; a list that silently omits `retry.py` and `newsletter.py` invites re-implementing existing behavior or missing an affected module when fixing. Partial rosters presented without an 'incomplete' marker read as complete.

**Verifier note:** Every named omission verified on disk (CLAUDE.md omits newsletter.py, config_utils.py, retry.py, evals/, scripts/eval_model.py, smoke_concurrency.py; README-technical's tree omits retry.py, evals/llm_cache.py, edit_tui.py, run_web.py, web_app.py, web_auth.py, web_data.py, templates/, docs/plans/, and the ux-redesign doc; roadmap #39 filed at line 133) — but the counts need correction: tests/ holds 32 test_*.py files, not 34, so the tree lists 9 of 32 and the coverage table 16 of 32.

### 59. [MEDIUM | gap | confirmed] VIP-sender feature absent from CLAUDE.md

**Locations:** `CLAUDE.md:127-141`; `README-technical.md:71-73`; `config.toml:120-124`; `classifier.py:197-211`; `classifier.py:239-252`

**Evidence:** README-technical documents `USER_NAME`, `VIP_SENDERS` ("VIP threads skip the sender classification LLM call"), and `EMAIL_LABELER_API_KEY`; config.toml has a `[vip_senders]` section ("always classified as PERSON (skips sender LLM call) and restricted to VIP-only categories (skips low-priority)"), and classifier.py implements it. CLAUDE.md's Environment Variables list omits all three vars, and its Architecture/Key Design Decisions never mention the VIP routing path at all.

**Why it matters:** CLAUDE.md's two-tier classification narrative ('Stage 1 determines person vs service') is incomplete: a whole deterministic bypass exists that changes both routing and the allowed label set. An agent fixing classification behavior from CLAUDE.md's model of the system will not account for the VIP branch, so its fixes can break (or be broken by) a path it doesn't know exists.

**Verifier note:** classifier.py:197-211 and :239-253 implement the VIP bypass (PERSON with no LLM call, restricted categories), config.toml:120-124 and README-technical:71-73 document it (USER_NAME, VIP_SENDERS, EMAIL_LABELER_API_KEY), yet CLAUDE.md's Environment Variables list (:127-141) omits all three vars and its Architecture/Key Design Decisions never mention the VIP path.

### 60. [LOW | gap | confirmed] README dead anchor; no newsletter overview section

**Locations:** `README.md:33`; `README.md:43`; `README.md:183-188`

**Evidence:** README.md:33 links "see [Newsletter Classification](#newsletter-classification)" but README.md's headings are Privacy Model / Label Taxonomy / Architecture / Prerequisites / Setup / Running / Resilience / Evaluation Suite / Testing — no 'Newsletter Classification' section exists, so the anchor is dead. The human README's only newsletter-pipeline prose is the NEWSLETTER_ONLY docker command and the volume-mount warning; the pipeline's actual overview lives in CLAUDE.md:50-64 (agent instructions).

**Why it matters:** The stated doc scheme puts human-facing overviews in README.md, but the newsletter pipeline's overview sits only in CLAUDE.md — so a human maintainer follows a dead link, and content placed in the 'wrong' doc per the scheme is a standing invitation to add a second copy (see the duplication finding).

**Verifier note:** README.md:33 links to #newsletter-classification but the file's full heading list (Privacy Model, Label Taxonomy, Architecture, Prerequisites, Setup, Running, Resilience, Evaluation Suite, Testing) contains no 'Newsletter Classification' heading — the anchor is dead, and the human README's only newsletter-pipeline prose is the NEWSLETTER_ONLY docker command (:183-186) plus the volume-mount warning; the overview lives only in CLAUDE.md:50-64.

### 61. [LOW | ambiguity | confirmed] 'Newsletter' means three different things

**Locations:** `README.md:29`; `config.toml:132`; `config.toml:154`; `CLAUDE.md:50-52`; `README.md:16`

**Evidence:** README.md:29 defines `agent/low-priority` as "Routine notifications, newsletters, spam, unwanted" (unqualified 'newsletters'). config.toml's FYI rule says "curated editorial newsletters (not automated product updates or account alerts) are worth reading. If yes → FYI" while LOW_PRIORITY's examples list "marketing newsletters". Meanwhile the project's headline 'newsletter pipeline' (CLAUDE.md:50) means ministry newsletters sent to the configured recipient, which get neither label.

**Why it matters:** Three incompatible referents share one word across the docs and prompts. A maintainer tuning label criteria or reading eval disagreements ('this newsletter was labeled FYI — bug?') cannot tell which sense a given doc sentence intends, and prompt fixes for one sense can silently shift behavior for another.

**Verifier note:** Verified verbatim: README.md:29 sends unqualified 'newsletters' to low-priority/Archived, config.toml:132 routes 'curated editorial newsletters' to FYI while :154 lists 'marketing newsletters' under LOW_PRIORITY, and CLAUDE.md:50-52's headline pipeline means ministry newsletters to the configured recipient which get neither label — three referents, one word, with README.md:29's summary directly in tension with the FYI rule.

### 62. [LOW | ambiguity | confirmed] proxy_client/gmail_utils: 'copied from' vs 'shared with' email-agent

**Locations:** `CLAUDE.md:33-34`; `README-technical.md:14-15`

**Evidence:** CLAUDE.md: "`proxy_client.py` — Gmail API proxy client (copied from email-agent)"; README-technical: "proxy_client.py     Gmail API proxy client (shared with email-agent)". Same for gmail_utils.py.

**Why it matters:** 'Copied' implies an independent fork that may be edited freely; 'shared' implies a sync obligation with a sibling repo. A maintainer fixing a bug in these files cannot tell from the docs whether the fix must be mirrored in email-agent or whether upstream changes should be pulled in — divergence here is a quiet source of cross-repo surprises.

**Verifier note:** Verbatim divergence verified: CLAUDE.md:33-34 says '(copied from email-agent)' for both files while README-technical.md:14-15 says '(shared with email-agent)' — fork-freely vs. sync-obligation semantics, with no doc resolving which relationship holds.

### 63. [LOW | gap | confirmed] Newsletter archiving (INBOX removal) undocumented in current docs

**Locations:** `labeler.py:216-221`; `CLAUDE.md:56-62`; `docs/plans/2026-02-19-newsletter-classification-design.md:88`

**Evidence:** `apply_newsletter_classification` removes the thread from the inbox (`remove_label_ids=["INBOX"]`, labeler.py:220). The only document stating this is the historical Feb design ("All newsletter emails are archived (removed from inbox) after labeling"); CLAUDE.md's current pipeline description ends at "Apply tier + theme labels via api-proxy → Gmail" and README.md's newsletter-label table says nothing about an archive action, though the email-label table has an explicit Action column.

**Why it matters:** Inbox-visible side effects are the user-facing contract. The fact that this behavior is documented only in an unmarked historical plan means a maintainer must read labeler.py to learn it — and a change to the give-up/recovery flow (the runbook depends on threads being 'still in the inbox') could interact with it unknowingly.

**Verifier note:** labeler.py:216-221 removes the INBOX label on every newsletter message; a grep for 'archiv' across README.md, README-technical.md, CLAUDE.md, both eval READMEs, and the runbook finds no mention of newsletter archiving anywhere — the only statement is the unmarked historical design doc line 88, and the runbook's recovery path (:50) explicitly depends on threads still being in the inbox.

### 64. [LOW | other | confirmed] Runbook embeds dated deployment-state facts

**Locations:** `docs/runbook-agent-attempted-recovery.md:27-31`; `CLAUDE.md:11`

**Evidence:** "The image deployed as of 2026-07-30 (built 2026-07-28) predates #65, so today the label still means only 'classification give-ups' — **the sweep is cleanest before the next image is deployed.**" The doc is checked in and describes a deployment state ('today') that the repo cannot verify and that inverts meaning after the next deploy; CLAUDE.md summarizes it as 'time-sensitive' but nothing marks the doc expired once the window closes.

**Why it matters:** The runbook does hedge ('After a post-#65 deploy, check a thread's To/Cc...'), but its headline claims will silently become false. A maintainer answering 'what does agent/attempted mean right now?' from this doc after the next deploy gets a stale answer with no staleness signal.

**Verifier note:** docs/runbook-agent-attempted-recovery.md:27-29 asserts 'The image deployed as of 2026-07-30 (built 2026-07-28) predates #65, so today the label still means only classification give-ups' — checked-in deployment-state claims that invert after the next deploy with no expiry marker; the doc's own hedge (:30-31) and CLAUDE.md's 'time-sensitive' note exist but nothing flags the doc stale once the window closes.

## Configuration & environment coherence

**Dimension summary:** config.toml itself is unusually well-commented (the post-#64 sizing comments are exemplary), but the layers around it have drifted out of sync with it: README-technical's LLM reference block still shows pre-#64 values, code fallback defaults in two call sites re-encode the exact sizing that #64 established as unsafe, and the three env-var inventories (CLAUDE.md, .env.example, README-technical) disagree — with a parity test that enforces only one of the three. Values are plumbed through three different mechanisms ({env.VAR} string substitution, dedicated int-override env vars, env-only settings) with an uneven and unexplained override surface, and several values (label names in gmail_query, the healthcheck path, VIP category defaults) are duplicated so that a single-value change must land in 2-4 places. This is precisely the shape that makes a correct-looking fix spawn new problems: the fix updates the copy the author knows about and the stale siblings silently reassert the old design.

### 65. [HIGH | contradiction | confirmed] Stale LLM settings block in README-technical

**Locations:** `README-technical.md:122-138`; `config.toml:48-63`; `config.toml:65-83`; `README-technical.md:188-190`

**Evidence:** README-technical's Configuration section ("All operational parameters are in config.toml") shows `model = "deepseek/deepseek-v3.2"`, `max_tokens = 8096`, `temperature = 0.2` for [llm.cloud] and `max_tokens = 8096`, `temperature = 0.2` for [llm.local]. Shipped config.toml has `model = "zai-org/glm-5"` (line 49), cloud `max_tokens = 1024` / `temperature = 0` (61-62), local `max_tokens = 4096` / `temperature = 0` (81-82) — with long comments explaining the sizes are deliberate #64 calibration ("the budget must fit the honest demand"). The adjacent daemon-settings block (README-technical.md:88-99) mirrors config.toml exactly, so the LLM block reads as equally authoritative but is stale. The [newsletter.llm] doc block (188-190) shows only `model`, omitting the sized max_tokens/timeout entirely.

**Why it matters:** A maintainer asking "what should max_tokens be?" gets two authoritative-looking answers that differ by 2-8x, on the exact parameter issue #64's fix was about (llm_client now raises LLMContentError on finish_reason "length", so budget sizing is load-bearing). Someone "fixing" a truncation issue by restoring the README's 8096, or reconciling docs to code in the wrong direction, silently undoes calibrated behavior — the classic fix-creates-new-problems path.

**Verifier note:** Verified verbatim: README-technical.md:126-138 documents model="deepseek/deepseek-v3.2", max_tokens=8096, temperature=0.2 for both [llm.cloud] and [llm.local], while shipped config.toml has zai-org/glm-5 (49), cloud 1024/0 (61-62), local 4096/0 (81-82) with comments declaring those sizes deliberate #64 calibration; the adjacent daemon-settings block (README-technical.md:88-99) mirrors config.toml exactly, so the stale LLM block reads as equally authoritative, and the [newsletter.llm] doc (188-190) omits the sized max_tokens=4096/timeout=180 entirely.

### 66. [HIGH | tension | confirmed] Fallback defaults preserve pre-#64 sizing

**Locations:** `daemon.py:863-865`; `evals/newsletter_run.py:644-646`; `config.toml:162-181`; `llm_client.py:133-135`

**Evidence:** daemon.py builds the newsletter client with `nl_llm_config.get("max_tokens", 1024)`, `.get("temperature", 0)`, `.get("timeout", 60)`; evals/newsletter_run.py:644-646 duplicates the identical fallbacks. config.toml's own comments say those exact values are the known-bad pre-#64 sizing: "4096, not 1024: ... 1024 truncated multi-story newsletters mid-story" (162-165) and "180, not 60: ... the old 60s bound only fit responses capped at 1024 tokens" (176-181). A third, different default set lives on LLMClient itself: `max_tokens: int = 8096, temperature: float = 0.2, timeout: int = 60` (8096 also looks like a typo for 8192).

**Why it matters:** Three layers of defaults (class default, call-site fallback, config.toml value) disagree with each other and with the documented design decision. Deleting or renaming a [newsletter.llm] key silently reverts to sizing the config file itself declares broken — and post-#64 that now means LLMContentError → repeated failures → newsletters burned to agent/attempted. Looks resolvable (make the keys required like [llm.cloud]/[llm.local], which use direct indexing at daemon.py:826-828 — an unexplained asymmetry in itself), but until then every fix to sizing must be mirrored in two fallback literals nobody documents.

**Verifier note:** Verified: daemon.py:863-865 and evals/newsletter_run.py:644-646 both fall back to max_tokens=1024/temperature=0/timeout=60 — the exact values config.toml:162-181 documents as the known-bad pre-#64 sizing — while LLMClient carries a third default set (8096/0.2/60, llm_client.py:133-135) and [llm.cloud]/[llm.local] use fallback-free direct indexing (daemon.py:826-828), so deleting a [newsletter.llm] key silently reverts to sizing the config file itself declares broken.

### 67. [MEDIUM | ambiguity | confirmed] NEWSLETTER_ONLY gating described inconsistently

**Locations:** `CLAUDE.md:52`; `daemon.py:851-874`; `daemon.py:407-409`; `daemon.py:484-487`; `README-technical.md:74`

**Evidence:** CLAUDE.md: "When `NEWSLETTER_ONLY=1`, the daemon switches to a newsletter-specific pipeline that grades ministry newsletter stories..." — implying the flag enables the pipeline. Code enables it unconditionally whenever [newsletter] exists in config: daemon.py:851-874 builds the classifier and logs "Newsletter classification enabled for: %s" in every mode, and process_single_thread routes any To/Cc-matching thread to grading (`if newsletter_classifier and newsletter_recipient: if is_newsletter(...)`) regardless of the flag. NEWSLETTER_ONLY only adds skipping non-newsletter threads (484-487) and query narrowing (928-929). README-technical.md:74 matches the code ("skip non-newsletter threads. Useful for testing newsletter classification in isolation").

**Why it matters:** A maintainer cannot tell whether grading newsletters during normal email-labeling runs is intended behavior or an accident of the config-presence check. Anyone who trusts CLAUDE.md (the agent-facing, override-all doc) would conclude removing NEWSLETTER_ONLY stops newsletter grading — it doesn't — and could "fix" the routing in either direction while believing they're preserving intent.

**Verifier note:** Verified: CLAUDE.md:52 (and :138 'instead of email labeling') implies the flag enables the newsletter pipeline, but daemon.py:851-874 enables it whenever [newsletter] exists in config and daemon.py:407-409 routes any To/Cc-matching thread to grading in every mode; NEWSLETTER_ONLY only adds the skip at 484-487 and the query narrowing at 928-929, and README-technical.md:74 documents the flag as skip-only — so CLAUDE.md alone never tells a reader that newsletter grading also runs during normal email-labeling.

### 68. [MEDIUM | gap | confirmed] Env-var inventory drift across three sources

**Locations:** `tests/test_env_var_docs.py:13-14`; `CLAUDE.md:127-141`; `.env.example:1-35`; `classifier.py:198`; `proxy_client.py:21`; `config.toml:131`

**Evidence:** test_env_var_docs.py enforces documentation only against `README_PATH = ROOT / "README-technical.md"`. CLAUDE.md's Environment Variables list omits USER_NAME (substituted into prompts at config.toml:131), VIP_SENDERS (classifier.py:198 — forces PERSON and blocks LOW_PRIORITY for those senders), and EMAIL_LABELER_API_KEY (proxy_client.py:21 fallback key). .env.example omits MLX_API_KEY (which README-technical.md:70 says to "set for public API stand-ins like Novita.ai"), VIP_SENDERS, EMAIL_LABELER_API_KEY, and all override vars (NEWSLETTER_ONLY, LOCAL_PARALLEL, MAX_EMAILS_PER_CYCLE, WRITE_PARALLEL). README-technical.md:72 also understates VIP_SENDERS: "VIP threads skip the sender classification LLM call" — omitting the category restriction (config.toml:122-123: "restricted to VIP-only categories (skips low-priority)").

**Why it matters:** The repo has the right idea — a parity test — but it blesses exactly one of three inventories, so the other two drift invisibly. An agent working from CLAUDE.md doesn't know VIP_SENDERS exists even though it changes classification outcomes; an operator working from .env.example never learns MLX_API_KEY exists. Any fix touching the sender-classification path can silently break the undocumented VIP behavior.

**Verifier note:** Verified in full: tests/test_env_var_docs.py:13-14 enforces parity only against README-technical.md; CLAUDE.md's env list (127-141) omits USER_NAME, VIP_SENDERS (classifier.py:198), and EMAIL_LABELER_API_KEY (proxy_client.py:21); .env.example (35 lines) omits MLX_API_KEY, VIP_SENDERS, EMAIL_LABELER_API_KEY, NEWSLETTER_ONLY, LOCAL_PARALLEL, MAX_EMAILS_PER_CYCLE, WRITE_PARALLEL; and README-technical.md:72 omits the VIP category restriction that config.toml:122-123 states ('restricted to VIP-only categories (skips low-priority)').

### 69. [MEDIUM | ambiguity | confirmed] Silent empty-string substitution for missing env vars

**Locations:** `config_utils.py:17-18`; `config_utils.py:29`; `config.toml:66`; `config.toml:131`; `README-technical.md:68-69`

**Evidence:** config_utils.py: "Missing environment variables are replaced with empty strings." So unset MLX_MODEL turns `model = "{env.MLX_MODEL}"` (config.toml:66) into `model = ""` — failing only later as a per-request 404 (README-technical.md:335: "MLX_MODEL must name the served model") — and unset USER_NAME leaves holes in the classification prompts ("Does this email specifically require {env.USER_NAME} personally to respond", config.toml:131). README-technical's table marks both MLX_MODEL and USER_NAME as Required: No with Default: — , yet the local tier cannot work without MLX_MODEL and the prompts degrade without USER_NAME. No startup check distinguishes the two cases.

**Why it matters:** It is genuinely unclear which {env.*} vars are optional-by-design versus effectively required, and the empty-string policy has no stated rationale. This clashes with the repo's otherwise strict-startup philosophy (labels verified with sys.exit, newsletter sink preflighted loudly): a maintainer adding validation doesn't know whether silently-empty is a contract someone relies on, and a maintainer debugging a 404 or a mangled prompt gets no signal pointing at the unset variable.

**Verifier note:** Verified: config_utils.py:18/29 substitutes empty strings for unset vars with no warning, so unset MLX_MODEL yields model="" (config.toml:66) failing only as a per-request 404 (README-technical.md:335: 'MLX_MODEL must name the served model') and unset USER_NAME leaves holes in prompts (config.toml:131); the README table marks both Required:No/Default:— (MLX_MODEL at :69, USER_NAME at :71), and run_daemon performs no startup check on either — clashing with the repo's otherwise strict startup (label verify sys.exit, sink preflight).

### 70. [MEDIUM | gap | confirmed] No startup validation of prompt templates

**Locations:** `config.toml:105-154`; `classifier.py:241-245`; `daemon.py:630-635`; `README.md:160`

**Evidence:** Prompts are Python format templates rendered per-request (`self.sender_config["user_template"].format(sender=..., subject=..., snippet=...)`). A typo'd placeholder (e.g. `{snipet}`) raises KeyError only at classify time, which lands in the daemon's generic `except Exception` → `_give_up_if_stuck` path (daemon.py:633-635) and burns every thread to agent/attempted, five strikes each. README.md:160 advertises "startup validation is strict" — but that strictness covers Gmail labels and the newsletter sink, not prompts or the [llm.*] blocks.

**Why it matters:** Prompt text is the most-edited part of config.toml (it is the whole point of the eval harness), yet it is the least-validated: a prompt fix that would fail instantly at startup instead fails per-thread with data-destroying consequences (backlog abandoned to agent/attempted — the exact cleanup the #64 runbook exists for). A maintainer has no documented answer to "what checks my prompt edit before it meets production mail?".

**Verifier note:** Verified: prompts (config.toml:105-154) are rendered per-request via .format() (classifier.py:241-245); a typo'd placeholder raises KeyError at classify time, caught by the generic except Exception → _give_up_if_stuck (daemon.py:633-635) with FailureTracker max_failures=5 (daemon.py:160), burning threads to agent/attempted five strikes each; README.md:160 advertises 'startup validation is strict' but run_daemon validates only Gmail labels and the newsletter sink, never a prompt render.

### 71. [MEDIUM | tension | confirmed] Uneven env-override surface across daemon knobs

**Locations:** `daemon.py:97-122`; `daemon.py:890-895`; `daemon.py:925-926`; `config.toml:2-31`; `README-technical.md:88-99`

**Evidence:** resolve_int_env exists because "{env.VAR} substitution only works for string config values, so numeric overrides are read here instead" (daemon.py:100-102). But only three of the numeric knobs get overrides — LOCAL_PARALLEL, WRITE_PARALLEL, MAX_EMAILS_PER_CYCLE — while `cloud_parallel`, `fetch_parallel`, `poll_interval_seconds`, `max_thread_chars` are toml-only (daemon.py:890, 893, 925, 980) with no stated reason. README-technical.md:94-97 annotates the overridable ones ("(override: LOCAL_PARALLEL)") but never says the others' non-overridability is intended. Meanwhile string values use a wholly different mechanism ({env.MLX_MODEL} substitution) and URLs/keys a third (env-only, never in toml — rationale at README-technical.md:79).

**Why it matters:** Resolvable tension, but today a maintainer adding an override for fetch_parallel can't tell whether its absence is a decision or an accident, and the "obvious" route — writing `{env.FETCH_PARALLEL}` in config.toml — fails non-obviously (substitution yields a string/empty, then Semaphore() crashes or misbehaves). Three plumbing mechanisms with per-knob membership decided nowhere is a steady generator of surprising fixes.

**Verifier note:** Verified: resolve_int_env (daemon.py:97-122) exists precisely because {env.VAR} substitution is string-only, yet only LOCAL_PARALLEL (891), WRITE_PARALLEL (894), and MAX_EMAILS_PER_CYCLE (926) get it, while cloud_parallel (890), fetch_parallel (893), poll_interval_seconds (925), and max_thread_chars (980) are toml-only with no stated rationale — three plumbing mechanisms (resolve_int_env, {env.*} substitution, env-only URLs/keys per README-technical.md:79) with per-knob membership decided nowhere.

### 72. [LOW | tension | adjusted] Divergent eval-versioning schemes for one config file

**Locations:** `evals/run_eval.py:642`; `evals/newsletter_run.py:107-116`; `config.toml:169-173`; `evals/schemas.py:142-166`

**Evidence:** The email eval stamps runs with `config_hash=hashlib.sha256(config_bytes).hexdigest()[:16]` — a hash of the whole config.toml file bytes — while the newsletter eval stamps `prompt_hash` over `config["newsletter"]["prompts"]` only. config.toml:169-173 documents the newsletter blind spot ("changing [max_tokens] ... prompt_hash (prompts only) stays identical, so pre/post-change runs are NOT distinguishable by hash"), but nothing documents the email side's inverse blind spot: any edit anywhere in config.toml — a newsletter prompt tweak, even a comment — changes every subsequent email-eval run's config_hash even though the email pipeline is untouched.

**Why it matters:** config.toml is shared by two pipelines whose eval harnesses version it by incompatible rules. A maintainer comparing email-eval runs across a newsletter-only edit sees hashes disagree and can't tell whether the email config actually changed; a maintainer unifying the schemes doesn't know which semantics is the intended design. Both linkages exist, so this is tension rather than a gap — and it looks resolvable.

**Verifier note:** The tension is real but the email-side mechanism is misstated: run_eval.py:633 hashes json.dumps(config, sort_keys=True) of the PARSED, env-substituted, CLI-overridden config dict — not 'the whole config.toml file bytes' — so a comment edit does NOT change config_hash (comments are dropped by the TOML parser), though any value change in any section (including a newsletter prompt tweak) and any change to a substituted env var (MLX_MODEL, USER_NAME) does; the core asymmetry stands (email eval versions the whole effective config, newsletter eval only [newsletter][prompts] per newsletter_run.py:107-116, with only the newsletter blind spot documented at config.toml:169-173), and the undocumented inverse blind spot should read 'any parsed-value or env-var change anywhere' rather than 'even a comment'.

### 73. [LOW | ambiguity | adjusted] NEWSLETTER_LLM_API_KEY half-set state silently inert

**Locations:** `daemon.py:139-142`; `README-technical.md:66-67`; `CLAUDE.md:135-136`; `.env.example:20-25`

**Evidence:** Code matches the documented atomicity (verified): `if override_url: return override_url, os.environ.get("NEWSLETTER_LLM_API_KEY", "")` — and the rationale is stated in daemon.py:135-137 ("pairing an override endpoint with the cloud provider's credential would authenticate against the wrong provider"). But the inverse half-set state is undefined-by-silence: NEWSLETTER_LLM_API_KEY set with NEWSLETTER_LLM_URL unset is silently ignored (the cloud key is used), with no warning. README-technical's table also lists Default: `CLOUD_LLM_API_KEY` in the same row that says the key "never" comes from the cloud key once URL is set — true only in the state where the variable itself is inert.

**Why it matters:** "Set both together" is stated, but nothing tells an operator who set only the key that their credential is being ignored — the failure surfaces as auth errors against the wrong provider or silent use of the wrong account. A maintainer adding a warning can't tell from the docs whether the silent-ignore is intended tolerance or an oversight.

**Verifier note:** All substantive claims verified (daemon.py:139-142 silently ignores NEWSLETTER_LLM_API_KEY when NEWSLETTER_LLM_URL is unset, with no warning anywhere; README-technical.md:67 lists Default: CLOUD_LLM_API_KEY in the same row as the 'never the cloud key' atomicity sentence), but the CLAUDE.md citation is wrong: the NEWSLETTER_LLM_URL/NEWSLETTER_LLM_API_KEY bullets are at CLAUDE.md:133-134 — lines 135-136 are MLX_URL/MLX_MODEL.

### 74. [LOW | ambiguity | confirmed] Newsletter match: to: query vs To/Cc code check

**Locations:** `daemon.py:928-930`; `newsletter.py:486-498`; `CLAUDE.md:56`

**Evidence:** In newsletter-only mode the Gmail query is narrowed with `gmail_query += f" to:{newsletter_recipient}"`, but the in-code filter `is_newsletter` "Checks both To and Cc headers, case-insensitive" (and CLAUDE.md:56 says "To/Cc matches config recipient"). The pipeline thus has two different definitions of "sent to the newsletter address": Gmail's `to:` operator server-side, and a To-or-Cc substring match client-side.

**Why it matters:** Whether a Cc-only newsletter should be graded is answered differently by the two filters — in normal mode it is graded (only is_newsletter runs), in newsletter-only mode it depends on Gmail's `to:` operator semantics. A maintainer can't tell which definition is the intent, so a fix to either filter risks silently changing which mail gets graded depending on mode.

**Verifier note:** Verified: newsletter-only mode narrows the query with Gmail's to: operator (daemon.py:928-930) while is_newsletter checks To OR Cc case-insensitively via substring (newsletter.py:486-498, docstring 'Checks both To and Cc headers'), matching CLAUDE.md's 'To/Cc matches config recipient' (actually line 55, cited as 56 — immaterial) — so a Cc-only newsletter is graded in normal mode but its fate in newsletter-only mode depends on Gmail's server-side to: semantics, two genuinely different definitions of 'sent to the newsletter address'.

### 75. [LOW | gap | confirmed] gmail_query duplicates label names from [labels]

**Locations:** `config.toml:5`; `config.toml:38-39`

**Evidence:** `gmail_query = "in:inbox -label:agent/processed -label:agent/attempted"` hardcodes the same label strings that [labels] defines as `processed = "agent/processed"` and `attempted = "agent/attempted"`. The attempted line's comment notes it is "excluded from gmail_query", but nothing states that the two places must be edited together, and no check catches divergence.

**Why it matters:** Renaming a label in [labels] without editing gmail_query makes processed threads re-match the query every cycle forever (or excludes the wrong label) — a silent behavior change from a seemingly-local config fix. The dependency is real and load-bearing (Design Decision 6 relies on agent/processed dropping threads out of gmail_query) but is documented only obliquely.

**Verifier note:** Verified: config.toml:5 hardcodes '-label:agent/processed -label:agent/attempted' as strings that [labels] independently defines at lines 38-39; the only documentation of the coupling is the oblique line-39 comment 'excluded from gmail_query', and no test enforces consistency (tests/test_daemon.py:1423 only asserts the key exists), so renaming a label in [labels] alone silently breaks the query exclusion Design Decision 6 depends on.

### 76. [LOW | gap | confirmed] healthcheck_file toml key vs Dockerfile hardcode

**Locations:** `config.toml:32`; `Dockerfile:19-20`; `README-technical.md:363-369`

**Evidence:** config.toml exposes `healthcheck_file = "/tmp/healthcheck"` as an operational parameter, but the Dockerfile HEALTHCHECK hardcodes the same path (`CMD test -f /tmp/healthcheck && ...`). Neither file cross-references the other; README-technical documents both without noting they must match.

**Why it matters:** Changing the toml value — which its presence as a config key invites — makes the daemon write one file while Docker checks another: the container goes permanently unhealthy (or, worse, checks a stale file) with no error anywhere. A configurable key whose only valid value is the hardcoded one is a trap; either the coupling should be documented or the key shouldn't exist.

**Verifier note:** Verified: config.toml:32 exposes healthcheck_file = "/tmp/healthcheck" as a config key (read at daemon.py:931) while the Dockerfile HEALTHCHECK (lines 18-20) hardcodes 'test -f /tmp/healthcheck'; neither file references the other and README-technical.md:363-369 documents both without stating they must match, so changing the inviting-looking toml value makes the container go permanently unhealthy with no error.

### 77. [LOW | ambiguity | confirmed] WRITE_TIMEOUT typeset as env var but is a constant

**Locations:** `proxy_client.py:92`; `CLAUDE.md:141`; `README-technical.md:77`; `config.toml:30`

**Evidence:** CLAUDE.md's Environment Variables section and README-technical's env-var table both say "writes may block on human approval (`WRITE_TIMEOUT`, 300s)" — monospace, in env-var context, indistinguishable from the real override vars around it. It is actually a class constant: `WRITE_TIMEOUT = 300.0  # seconds` in proxy_client.py, not readable from the environment.

**Why it matters:** An operator whose approval flow needs more than 300s will plausibly set WRITE_TIMEOUT=600 in .env and observe nothing change; a maintainer auditing env vars will search for a getenv that doesn't exist. Small, but it muddies the one inventory (the env-var list) that most needs to be precise.

**Verifier note:** Verified: WRITE_TIMEOUT is a class constant (proxy_client.py:92, used at 307/325/343; no os.environ read of it anywhere in the codebase), yet it appears monospaced inside the env-var inventories at CLAUDE.md:141 and README-technical.md:77 (and config.toml:30), indistinguishable from the real override vars in the same bullets — an operator setting WRITE_TIMEOUT=600 in .env would observe nothing change.

### 78. [LOW | other | confirmed] VIP config split across env and toml with duplicated default

**Locations:** `config.toml:120-124`; `classifier.py:197-200`; `README-technical.md:72`

**Evidence:** One feature, two homes: VIP addresses come only from the VIP_SENDERS env var while VIP categories come only from `[vip_senders] categories` in toml (the toml section's comment is the sole place the split is explained). The categories default is duplicated: config.toml sets `categories = ["NEEDS_RESPONSE", "FYI"]` and classifier.py independently hardcodes `vip_config.get("categories", ["NEEDS_RESPONSE", "FYI"])`.

**Why it matters:** A maintainer changing VIP behavior must know to look in two configuration systems, and the duplicated default means removing the toml section silently keeps the old behavior via the code literal — fine today, wrong the day either copy is edited alone. Combined with VIP_SENDERS being absent from CLAUDE.md and .env.example, the whole VIP feature is easy to break without knowing it exists.

**Verifier note:** Verified: VIP addresses come only from the VIP_SENDERS env var (classifier.py:198) while categories come only from [vip_senders] in toml, with the split explained solely in the config.toml:120-123 comment; the categories default ["NEEDS_RESPONSE", "FYI"] is duplicated between config.toml:124 and the code literal at classifier.py:200 (and again at evals/run_eval.py's vip_categories fallback), and VIP_SENDERS is absent from both CLAUDE.md's env list and .env.example.

## Failure-handling policy

**Dimension summary:** Failure handling in this repo is unusually well-commented site-by-site (every except arm in daemon.py carries a rationale, and the LLMBalanceError halt heuristic matches its docs exactly, with its brittleness explicitly acknowledged), but there is no single stated policy — the mapping from error class to response (retry-in-cycle, defer-uncounted, give-up-counted, halt, log-and-continue, crash) exists only as scattered per-arm comments, and those comments have drifted into outright contradiction: proxy_client's docstrings promise ProxyUnavailableError is never counted toward give-up while daemon.py counts it (the runbook records real "proxy-flap give-up" casualties), and llm_client's transient/permanent taxonomy classifies exhausted 429/5xx as request-specific-permanent while retry.py and proxy_client classify the same statuses as transient. The deeper #64-shaped exposure remains: FailureTracker's two-bucket model (poison thread vs endpoint outage) has no bucket for deterministic class-wide faults, and the boundary between the "safe default" parse philosophy and the post-#64 "never silently default" philosophy is nowhere stated — each newly discovered unusable-output shape gets ad-hoc assignment to one side, which is exactly the fix-creates-new-problems generator the owner reports. The newsletter pipeline adds its own divergences (per-story timeout swallowed and committed permanently; unparseable grading silently conflated with the "no extractable stories" label).

### 79. [HIGH | contradiction | confirmed] ProxyUnavailable docstring vs give-up

**Locations:** `proxy_client.py:53-56`; `proxy_client.py:150-154`; `daemon.py:591-610`; `docs/runbook-agent-attempted-recovery.md:23-24`

**Evidence:** proxy_client.py:53-54 (ProxyUnavailableError docstring): "Mirrors llm_client.LLMUnavailableError: a transient outage that should be deferred and retried next cycle, NOT counted toward giving up on a thread." proxy_client.py:152-154 (429/5xx arm): "next poll cycle is the right place to retry, not give-up." But daemon.py:595 says the opposite and is what runs: "Unlike LLMUnavailableError above, this is give-up-eligible (issue #26)", and daemon.py:610 routes it to _give_up_if_stuck. The runbook confirms real casualties of the daemon's reading: "Proxy-flap give-ups (~2026-07-09 era): threads abandoned during Gmail API proxy outages" (runbook:23-24). The daemon's stated justification — a fully endpoint-wide proxy outage is caught at list_messages before any thread is counted — does not hold for a flap that starts after list_messages succeeds: every thread in that cycle takes a strike.

**Why it matters:** A maintainer touching either layer gets opposite answers to "does a transient proxy fault count toward abandoning a thread?" depending on which file they read. Anyone who trusts the proxy_client docstrings (the natural place to look when handling the exception type) will write code or reviews that assume defer-only semantics and be surprised by agent/attempted fallout — precisely the class of fix-introduces-new-problems the owner reports, and one that has already abandoned real mail.

**Verifier note:** Irreconcilable as written: proxy_client.py:53-54 ('NOT counted toward giving up') and :152-154 ('next poll cycle is the right place to retry, not give-up') vs daemon.py:595 ('Unlike LLMUnavailableError above, this is give-up-eligible (issue #26)') routing every per-thread ProxyUnavailableError to _give_up_if_stuck at :610 — and runbook:23-24 records real proxy-flap casualties. Extra evidence: proxy_client.py:177-180 contradicts its own class docstring by conceding 'a *persistent* one is still bounded by the FailureTracker give-up path', and the _send docstring (:186) repeats the defer-only claim.

### 80. [HIGH | contradiction | confirmed] LLM 429/5xx taxonomy split

**Locations:** `llm_client.py:58-61`; `retry.py:3-4`; `retry.py:17`; `llm_client.py:277-281`; `daemon.py:630-632`; `proxy_client.py:149-159`; `CLAUDE.md:113`

**Evidence:** llm_client.py:58-60 (LLMUnavailableError docstring) states the taxonomy: "a non-200 response or a read/write timeout means the *request* is the problem (too-large input, bad payload) and is eligible for the daemon's give-up logic". But retry.py:3-4 classifies the same statuses as "transient HTTP errors (429 rate-limit, 502/503/504 server errors)" (RETRYABLE_STATUS_CODES, retry.py:17), and after retries exhaust, llm_client.py:277-281 raises generic RuntimeError, which daemon.py:630-632 counts toward give-up. Meanwhile proxy_client.py:150-151 classifies exhausted 429/5xx from the proxy as "Transient, endpoint-wide conditions that clear on their own → defer and retry rather than abandoning the thread". CLAUDE.md:113 also calls 429 "transient throttling". So a sustained cloud-provider throttle or 503 episode (transient by three of the four sources) burns threads toward agent/attempted at 5 strikes; the statements "transient, endpoint-wide" and "the request is the problem" cannot both be true of the same status codes.

**Why it matters:** The transient/permanent split is the load-bearing decision in this codebase (it decides retry-forever vs permanent abandonment), yet the two HTTP clients embed opposite classifications of identical status codes and the docstring that defines the rule is factually wrong for 429/5xx. Any fix that adds an error path must pick a side, and whichever side it picks will contradict half the existing comments — a direct generator of review churn and of a future incident where a weekend rate-limit sweeps the backlog into agent/attempted.

**Verifier note:** All quotes verified: llm_client.py:58-60 declares any non-200 'the *request* is the problem', yet retry.py:3-4/17 calls 429/502/503/504 transient, proxy_client.py:150-151 calls the same exhausted statuses 'transient, endpoint-wide', and CLAUDE.md:113 calls 429 'transient throttling'; behaviorally an exhausted cloud 429/5xx becomes generic RuntimeError (llm_client.py:277-281) which daemon.py:630-632 counts toward the 5-strike give-up, so a sustained throttle sweeps threads into agent/attempted.

### 81. [HIGH | tension | confirmed] Safe-default vs loud-failure boundary

**Locations:** `CLAUDE.md:110`; `classifier.py:95-96`; `classifier.py:149-150`; `classifier.py:160-169`; `llm_client.py:288-304`; `README-technical.md:146`

**Evidence:** CLAUDE.md:110 (Design Decision 2): "Safe defaults: Unknown sender type → SERVICE ... Unknown email label → LOW_PRIORITY (archived, not deleted)." classifier.py still implements it: "Defaults to LOW_PRIORITY (safe)" (classifier.py:150), logging only a WARNING (classifier.py:163-169). The #64 fix asserts the opposite philosophy in llm_client.py:298-300: empty/truncated shapes "would otherwise parse to a default SERVICE / LOW_PRIORITY label and silently mislabel the email", and commit 9acc63f ("Make a finish_reason of 'length' loud instead of silently parseable") calls the silent default the bug. The boundary that decides which unusable outputs raise LLMContentError (loud, give-up-eligible) versus silently default-and-archive (content present but no label keyword) is not stated anywhere — it emerges only from which layer happens to detect the shape.

**Why it matters:** This is the epicenter of #64, and the fix's shape (moving one output shape — finish_reason=length — across an undrawn line) shows the line itself was never drawn. The next unusable-output variant (e.g. a hallucinated label word inside prose, a non-English reply) forces the same ad-hoc decision again: is silent LOW_PRIORITY "safe" (decision 2) or "silent mislabeling" (#64 rationale)? Two live philosophies with no stated criterion is a reliable source of fixes that reviewers flag as introducing new inconsistencies.

**Verifier note:** Both philosophies verified live: classifier.py:95-96/149-150 still call silent SERVICE/LOW_PRIORITY defaults '(safe)' (WARNING-only at :163-169) per CLAUDE.md:110, while llm_client.py:298-304 and commit 9acc63f (verified: 'Make a finish_reason of "length" loud instead of silently parseable') call the same silent default 'silently mislabel'; no file states the criterion for which unusable output shapes raise LLMContentError vs default-and-archive — it emerges only from which layer detects the shape.

### 82. [HIGH | gap | adjusted] No bucket for class-wide faults

**Locations:** `daemon.py:145-157`; `docs/runbook-agent-attempted-recovery.md:8-17`; `CLAUDE.md:113`; `daemon.py:197-206`

**Evidence:** FailureTracker's docstring (daemon.py:145-157) defines a two-bucket world: "An endpoint-wide outage never lands here ... A *persistent per-thread* fault — a poison thread ... does accrue here and is given up." Issue #64 was neither: a deterministic config bug affecting an entire class of threads ("hard person threads ... took FailureTracker's 5 strikes, and were labeled agent/attempted", runbook:9-12), which the tracker cannot distinguish from N independent poison threads. The #61 balance-halt work (DaemonHalt, daemon.py:197-206; CLAUDE.md:113) carved out exactly one account-wide fault type, proving the taxonomy needed a third bucket — but there is no general policy, detection, or even documented consideration for correlated give-ups (e.g. many threads reaching 5 strikes with the same error in one window), and the #65/#66 fix addressed only the specific Ollama dialect bug plus louder diagnostics.

**Why it matters:** The runbook exists because this gap turned a config typo into permanent, unenumerable data loss. A maintainer asking "when N threads all give up with identical errors, is that N poison threads or one systemic fault?" finds no answer anywhere; the next deterministic class-wide fault (a prompt change, a provider API change, another dialect no-op) will replay #64 through the same unguarded path, and each such incident spawns another one-off carve-out like DaemonHalt.

**Verifier note:** Gap is real — FailureTracker's two-bucket docstring (daemon.py:145-157) cannot distinguish N poison threads from one deterministic class-wide fault, DaemonHalt (daemon.py:197-206) is a single one-off carve-out, and the next #64-style fault replays through the same path — but 'not even documented consideration for correlated give-ups' overstates: daemon.py:604-608 explicitly documents and accepts that 'a *sustained* partial outage can abandon the backlog' (to findable agent/attempted) for the proxy arm, and the cycle summary logs give-ups distinctly (daemon.py:1011-1015). No consideration exists for the LLM-content/config-bug class that caused #64.

### 83. [MEDIUM | contradiction | confirmed] Stale 'marked processed' comments

**Locations:** `daemon.py:907-909`; `tests/test_daemon.py:358`; `daemon.py:244`; `labeler.py:169-178`

**Evidence:** daemon.py:907-909, directly above the FailureTracker construction: "a thread that keeps failing for a thread-specific reason (not a transient outage) is marked processed after a few attempts." tests/test_daemon.py:358 docstring: "A thread that keeps failing is marked processed after max_failures, breaking the loop." The code does neither: _give_up_if_stuck calls mark_attempted (daemon.py:244), and the same test asserts "mark agent/attempted (not agent/processed)" (test_daemon.py:373-377). The processed→attempted split was the whole point of issue #23 (commit f61be9f "mark abandoned threads agent/attempted instead of agent/processed"), and CLAUDE.md:123 depends on the distinction ("kept distinct so abandoned threads stay findable").

**Why it matters:** These are exactly the comments a maintainer reads when deciding what give-up does. Someone extending the give-up path from the daemon.py:907 comment (or writing a new test from the test docstring) could plausibly reuse mark_processed, silently destroying the findability property the runbook's entire recovery procedure depends on. Stale statements of a since-changed policy are how a correct fix gets reviewed as wrong, or vice versa.

**Verifier note:** Verified verbatim: daemon.py:907-909 comment and tests/test_daemon.py:358 docstring both say 'marked processed', while the code marks agent/attempted (daemon.py:244 mark_attempted; the same test asserts mark_attempted called and mark_processed NOT called at :376-377); commit f61be9f ('mark abandoned threads agent/attempted instead of agent/processed') exists, and CLAUDE.md:123 and labeler.py:169-178 depend on the distinction the stale comments erase.

### 84. [MEDIUM | tension | confirmed] Timeout conflates size and slowness

**Locations:** `llm_client.py:248-255`; `daemon.py:616-621`; `README-technical.md:359`; `llm_client.py:14-18`; `CLAUDE.md:111`

**Evidence:** llm_client.py:249-252 classifies every read/write timeout as "request-specific (e.g. a transcript too large to prefill within the timeout)", making it give-up-eligible (daemon.py:616-621: "Request-specific slowness ... eligible for give-up so one huge thread can't be retried forever"). But the codebase itself documents endpoint-wide causes of the same timeout: "a cold load of a large model routinely exceeds 10s" (llm_client.py:16-17), and README-technical's memory section describes GPU thrash/prefill contention as "the most failure-prone part of the stack". CLAUDE.md:111 promises "If local MLX is down, person emails are skipped (retried next cycle). Privacy invariant preserved" — but an MLX server that is up-and-slow (thrashing, cold-loading, saturated) times out every thread, and five slow cycles sweep the entire person backlog into agent/attempted, bodies never classified. The endpoint-wide-slowness reading of a timeout is nowhere acknowledged; only the oversized-transcript reading is. (Looks resolvable — e.g. by stating which reading wins and why — but the tension is inherent to a single timeout signal.)

**Why it matters:** This is the same shape as #64 (a non-thread-specific condition consuming per-thread strikes) still live on another signal. A maintainer tuning timeouts or debugging an agent/attempted spike cannot tell from any doc whether timeout-equals-poison-thread is a considered decision or an oversight, so any fix here (e.g. reclassifying timeouts as deferrals) risks re-introducing the retried-forever problem the current policy explicitly guards against.

**Verifier note:** Verified: every read/write timeout becomes give-up-eligible TimeoutError (llm_client.py:248-255, daemon.py:616-621), documented only under the oversized-transcript reading (README-technical.md:359 even presents FailureTracker as a deliberate guard for prefill timeouts), while the codebase's own slow-endpoint evidence (cold loads >10s at llm_client.py:14-18, 'most failure-prone part of the stack' at README-technical.md:324) and CLAUDE.md:111's skip-and-retry promise for a down MLX are never reconciled with an up-but-slow server timing out every thread into agent/attempted.

### 85. [MEDIUM | tension | confirmed] Newsletter timeout swallowed per-story

**Locations:** `newsletter.py:27`; `newsletter.py:548-554`; `newsletter.py:571-578`; `newsletter.py:584-587`; `daemon.py:616-621`

**Evidence:** classify_newsletter's docstring promises transient faults propagate "rather than committing a permanently mis-graded newsletter (empty tier/themes) and marking it processed" (newsletter.py:551-553), and _PIPELINE_WIDE_ERRORS (newsletter.py:27) propagates LLMUnavailableError, LLMContentError, LLMBalanceError. TimeoutError is not in that tuple, so a per-story timeout falls to "except Exception: log.warning('Quality assessment failed for story...')" (newsletter.py:577-578) — the story is committed with scores=None, the newsletter is labeled and marked agent/processed, permanently. The email pipeline treats the identical exception as serious enough for the bounded give-up path (daemon.py:616-621), i.e. it gets 5 chances; a newsletter story's timeout gets zero chances and an immediate permanent commit. No comment states why timeout is pipeline-wide in one pipeline and story-isolated in the other.

**Why it matters:** One transient slow-response episode during grading silently produces exactly the outcome the write-before-label work (#65) was built to prevent: a permanently committed, never-re-graded, mis-graded newsletter. A maintainer extending _PIPELINE_WIDE_ERRORS has no stated criterion for membership ("affects every story" fits a timeout under provider slowness), so the next error type added will be another judgment call reviewers can dispute.

**Verifier note:** Verified: TimeoutError is absent from _PIPELINE_WIDE_ERRORS (newsletter.py:27), so a per-story timeout falls to 'except Exception: log.warning(...)' (newsletter.py:577-578/586-587) and the newsletter is committed and marked agent/processed permanently with scores=None — zero retries — while the email pipeline gives the identical exception 5 strikes (daemon.py:616-621); the membership comment ('errors that affect every story', newsletter.py:23-26) fits a slowness-driven timeout yet excludes it, and neither the docstring (:548-554) nor any comment mentions timeout.

### 86. [MEDIUM | ambiguity | adjusted] no-stories label conflation

**Locations:** `CLAUDE.md:70`; `newsletter.py:565-570`; `daemon.py:421-426`; `labeler.py:203-207`; `classifier.py:160-169`

**Evidence:** CLAUDE.md:70 defines the label: "agent/newsletter/no-stories — Newsletter contained no extractable stories." But the label is applied whenever best_tier is None (daemon.py:421-426 finds no story with a non-None tier; labeler.py:203-207 then adds no_stories), which also happens when stories WERE extracted but every quality response was unparseable: parse_quality_scores returning None just skips the "if scores:" block (newsletter.py:567-570) with no else branch and no log line at all — unlike the email pipeline, where a parse fallback at least logs "Unexpected email label output" (classifier.py:163-169). So "grading unparseable" is silently recorded and labeled as "no extractable stories", and the record is committed permanently (agent/processed applied).

**Why it matters:** An operator triaging no-stories-labeled newsletters cannot distinguish "genuinely storyless mailing" from "grader output format drifted" — the second is a prompt/model regression that needs action, and it is the only failure mode in the newsletter pipeline with literally zero log evidence. The #64 lesson (silent defaulting masks systemic faults) applies unaddressed here; a prompt tweak that breaks the SCORE format would present as a quiet rise in no-stories labels.

**Verifier note:** Core ambiguity confirmed: the label defined as 'no extractable stories' (CLAUDE.md:70) is applied whenever best_tier is None (daemon.py:421-426, labeler.py:203-207), including extracted-but-unparseable gradings, and parse_quality_scores returning None is silent at the parse site (newsletter.py:567-570 has no else/log, unlike classifier.py:163-169). But 'literally zero log evidence' is refuted: daemon.py:474-481 logs 'Newsletter thread X: N stories, tier=no-stories' at INFO — N>0 with no-stories is visible evidence — and the JSONL record retains the story texts and quality_cot with null scores, so the failure mode is diagnosable from the record even though the Gmail label conflates.

### 87. [MEDIUM | tension | confirmed] agent/attempted has no durable reason

**Locations:** `CLAUDE.md:114`; `docs/runbook-agent-attempted-recovery.md:14-17`; `docs/runbook-agent-attempted-recovery.md:33-35`; `daemon.py:269-273`

**Evidence:** Design Decision 6 (CLAUDE.md:114) establishes a durability principle for newsletters: "the newsletter JSONL is the only durable copy ... So the record is written first" before the destructive agent/processed label. The give-up path applies the equally destructive agent/attempted (drops the thread from gmail_query) with only an in-memory count and a log line (daemon.py:269-273); the runbook documents the consequence: "Production logs did not survive the container teardown ... **the label itself is the only durable trace**" (runbook:16-17) and "there is no per-thread record of *why* a given thread was given up on — only the label" (runbook:34-35). The asymmetry — why gradings deserve a durable pre-commit record but abandonment reasons do not — is not explained anywhere. (Looks resolvable; both principles can stand once the scope of decision 6 is stated.)

**Why it matters:** The #64 recovery was crippled by exactly this gap ("the affected-thread count cannot be reconstructed from logs"), and the runbook's triage step 1 must guess populations by heuristics (sender-looks-human, date era). A maintainer deciding where decision 6's write-before-destructive-label rule applies has no stated boundary, so the next destructive-marker feature will re-litigate it.

**Verifier note:** Verified: decision 6 (CLAUDE.md:114) mandates a durable record before the destructive agent/processed label, while the equally destructive agent/attempted is applied with only an in-memory count and a log line (daemon.py:269-273); runbook:16-17 ('the label itself is the only durable trace') and :33-35 ('no per-thread record of *why*... only the label') document the cost, and no file explains the asymmetry or states decision 6's scope.

### 88. [MEDIUM | tension | confirmed] Daemon-wide halt, per-provider fault

**Locations:** `daemon.py:622-629`; `daemon.py:938-948`; `CLAUDE.md:113`; `daemon.py:125-142`; `llm_client.py:112-122`

**Evidence:** CLAUDE.md:113 justifies the halt as "account-wide, not a poison thread: the daemon halts polling entirely", and DaemonHalt stops all processing (daemon.py:938-948: "An out-of-funds provider fails EVERY request"). But the deployment deliberately runs up to three distinct provider accounts — resolve_newsletter_llm_endpoint exists because "the newsletter model lives elsewhere" (daemon.py:129-137, Anthropic vs the DeepSeek cloud tier), and MLX_API_KEY supports paid local stand-ins ("Novita.ai", CLAUDE.md env-var docs). An LLMBalanceError from the newsletter tier's Claude account trips the same daemon-wide halt (daemon.py:626-628) and stops email classification on the separately funded cloud provider, where "every request" would in fact succeed. Neither the code comments nor the halt docs (README-technical.md:377-393) acknowledge that halt scope (daemon) exceeds fault scope (one provider account). (Resolvable: state whether the over-halt is accepted or scope it per-tier.)

**Why it matters:** The stated rationale ("fails EVERY request") is only true in single-provider deployments, but the config actively encourages multi-provider ones. A maintainer hit by an email-processing outage caused by the newsletter account running dry has no doc saying this is intended; and one deciding whether to scope the halt per-tier can't tell whether daemon-wide was a considered trade-off or an artifact of the single-provider era.

**Verifier note:** Verified and strengthened: the newsletter classifier is constructed whenever [newsletter] config exists (daemon.py:851-874), NOT gated on NEWSLETTER_ONLY, so the separately-funded newsletter (Anthropic), cloud (DeepSeek), and optionally paid local stand-in providers coexist in one daemon in the normal configured state; LLMBalanceError from any tier trips the tier-agnostic daemon-wide halt (daemon.py:622-628, 938-948 'fails EVERY request'), and neither CLAUDE.md:113 nor README-technical.md:377-393 acknowledges halt scope exceeding fault scope.

### 89. [MEDIUM | gap | confirmed] No unified failure-policy map

**Locations:** `daemon.py:574-635`; `CLAUDE.md:110-114`; `README-technical.md:377-393`; `proxy_client.py:36-79`; `llm_client.py:55-122`; `retry.py:17-20`

**Evidence:** The system has at least seven distinct failure responses — in-request HTTP retry (retry.py:17-20), defer-uncounted next cycle (daemon.py:574-590, 611-615), counted give-up → agent/attempted (daemon.py:591-635), daemon halt (daemon.py:622-629), cycle-level backoff (daemon.py:1031-1062), startup block-or-exit (daemon.py:710-754, 917-921), and newsletter per-story log-and-continue (newsletter.py:577-587) — but the error-class→response mapping is stated nowhere as one policy. CLAUDE.md's decisions 3/5/6 cover three fragments; README-technical covers the halt and the sink; everything else lives in per-arm comments across four files. The drift documented in the other findings (proxy_client docstrings vs daemon behavior; "marked processed" comments; the 429/5xx taxonomy split) is the measurable cost: the scattered copies have already diverged.

**Why it matters:** Every new error type (and every reviewed fix) must be slotted into this taxonomy, and today that requires reading and reconciling four files whose statements disagree. This is the most direct structural explanation for the owner's symptom: without a single normative table, each fix re-derives the policy from whichever local comments the author read, and reviewers checking against different comments find 'new problems'.

**Verifier note:** Verified: all seven distinct failure responses exist at the cited locations (retry.py:17-20 in-request retry; daemon.py:574-590/611-615 uncounted defer; :591-635 counted give-up; :622-629 halt; :1031-1062 cycle backoff; :710-754/917-921 startup block-or-exit; newsletter.py:577-587 per-story swallow), no error-policy section exists in any doc (README-technical headings checked; CLAUDE.md decisions 3/5/6 cover fragments), and the confirmed drift in findings 1, 2, and 5 is the measured cost of the scattering.

### 90. [LOW | gap | adjusted] Give-up threshold undocumented

**Locations:** `daemon.py:159`; `daemon.py:910`; `CLAUDE.md:123`; `README-technical.md:87-99`; `docs/runbook-agent-attempted-recovery.md:11-12`

**Evidence:** The give-up criterion is max_failures consecutive-ish failures, defaulting to 5, hardcoded in the constructor (daemon.py:159 "def __init__(self, max_failures: int = 5)") and taken as default at daemon.py:910 ("FailureTracker()"). CLAUDE.md:123 says only "applied on give-up (after repeated failures)"; the config reference (README-technical.md:87-99) lists every other operational knob but no give-up threshold, and it is not env-overridable like its peers (LOCAL_PARALLEL, MAX_EMAILS_PER_CYCLE, WRITE_PARALLEL). The number 5 is documented only incidentally, inside the incident runbook ("took FailureTracker's 5 strikes", runbook:11-12). Tests exercise thresholds 1-3, never the production default. Also unstated: what resets a count (success or give-up via summarize_cycle, pruning when a thread leaves the query, daemon restart) — i.e. what "repeated" actually means operationally.

**Why it matters:** An operator watching a thread fail cannot predict from the docs when it will be abandoned, and after #64 the threshold's tuning is consequential (5 strikes at 60s polls abandoned threads within minutes). A maintainer changing the default has no doc to update except an incident runbook — a sign the policy lives nowhere.

**Verifier note:** Core confirmed: max_failures=5 is hardcoded (daemon.py:159, default-constructed at :910), absent from the config reference (README-technical.md:87-99), CLAUDE.md ('after repeated failures', :123), and any env override (no MAX_FAILURES/GIVE_UP hits anywhere), documented only incidentally in the runbook; tests use max_failures 1-3, never 5. One sub-claim over-reaches: what resets a count IS stated in code docstrings (restart reset at daemon.py:154-156, pruning at :173-181, success/give-up clear at :655 and summarize_cycle's docstring) — it is unstated only in operator-facing docs.

### 91. [LOW | other | confirmed] Sink-fault log promises retry on give-up strike

**Locations:** `daemon.py:453-464`; `daemon.py:633-635`; `README-technical.md:310`

**Evidence:** On a newsletter sink OSError the daemon logs at ERROR "thread %s left unprocessed for retry (check the path exists...)" (daemon.py:456-463) and then re-raises; the exception lands in the generic "except Exception" arm (daemon.py:633-635), which logs the same failure a second time via log.exception AND takes a give-up strike. On the fifth occurrence the cycle's log therefore contains both "left unprocessed for retry" and "marked agent/attempted to break the retry loop" for the same thread — the first message is false on exactly the strike where it matters most, and the thread is dropped with no assessment ever written. README-technical.md:310 documents the chain correctly ("persistent fault ends at the give-up path's findable agent/attempted") but the operator-facing log does not.

**Why it matters:** The sink-fault path was built (#65) specifically so operators could trust the logs about what happened to a grading; an ERROR line that promises a retry that will not happen sends the operator to fix the mount believing no mail was lost, when the thread has just been permanently abandoned ungraded.

**Verifier note:** Verified: the OSError arm logs 'thread %s left unprocessed for retry' at ERROR then re-raises (daemon.py:456-464); no intervening arm catches OSError (TimeoutError is a subclass of OSError, not vice versa), so it lands in 'except Exception' (daemon.py:633-635) which logs the same failure again via log.exception AND takes a give-up strike — on the fifth occurrence the thread is marked agent/attempted with no assessment ever written, making the 'left unprocessed for retry' line false exactly when it matters; README-technical.md:310 documents the chain correctly but the log does not.

## Eval-harness fidelity to production

**Dimension summary:** Core fidelity is genuinely strong: both eval runners import and drive the real production classifiers (EmailClassifier, NewsletterClassifier), real parsers (parse_themes/classify_theme_line are single-sourced), the real config prompts, and the shared transcript formatter — so most prompt/parse changes are automatically reflected in evals. The clarity problems concentrate at the seams: the LLM cache's validity contract omits endpoint/backend identity while docs claim staleness cannot occur; thread-metadata assembly is a hand-mirrored copy of daemon code whose anchoring comment now points at unrelated code; run provenance (prompt_hash/config_hash) cannot distinguish code-version changes; and the inline (--report) and standalone report entry points have already drifted apart, producing different numbers for the same results file. Several metric definitions (Stage 2 accuracy under routing vs oracle, per-story vs newsletter-level tier) are production-relevant but never pinned down in docs, so a maintainer cannot tell whether an eval delta reflects the model, the harness, or a stale cache — exactly the soil in which fix-induced regressions go unnoticed.

### 92. [HIGH | gap | confirmed] Cache key omits endpoint identity

**Locations:** `evals/llm_cache.py:80-87`; `evals/README-technical.md:428-434`; `evals/run_eval.py:563-580`; `evals/newsletter_run.py:639-648`; `daemon.py:125-142`

**Evidence:** The cache key is `[self.inner.model, self.inner.temperature, self.inner.max_tokens, self.inner.extra_body, system_prompt, user_content]` (llm_cache.py:83-85) — base_url is not included, yet base_url comes from env (`CLOUD_LLM_URL`, `MLX_URL`, `NEWSLETTER_LLM_URL` with silent fallback to `CLOUD_LLM_URL`). README-technical.md:434 claims: "Changing any of these — the model name, temperature, inference parameters, `extra_body` config, or the prompt content — produces a different cache key, ensuring stale hits don't occur across configurations." Switching backend/provider while keeping the model name (e.g. the same HF id served by mlx_lm.server vs Ollama, or `claude-sonnet-4-6` via two different gateways under the NEWSLETTER_LLM_URL fallback) reuses the other backend's cached responses. Nothing records server identity or version in the cache entry or RunMeta.

**Why it matters:** The maintainer question "are these eval numbers from the endpoint I think I'm testing, or replayed from a different backend's old responses?" is unanswerable from the artifacts. The documented model-swap and backend-A/B workflows make same-name/different-backend realistic, and a stale hit is indistinguishable from a fresh result — an eval can 'confirm' a fix that was never actually exercised against the changed backend, feeding the fix-creates-new-problems loop.

**Verifier note:** Verified: llm_cache.py:82-87 keys on [model, temperature, max_tokens, extra_body, system_prompt, user_content] with no base_url, cache entries persist only "model", and neither RunMeta (schemas.py:141-166) nor NewsletterRunMeta (newsletter_schemas.py:304-330) records an endpoint URL — so README-technical.md:434's "ensuring stale hits don't occur across configurations" is false for a backend swap that keeps the model name (realistic via MLX_URL stand-ins and the NEWSLETTER_LLM_URL→CLOUD_LLM_URL silent fallback at daemon.py:139-142); endpoint identity surfaces only in stderr and per-row *error* strings, never in successful-run artifacts.

### 93. [HIGH | tension | confirmed] Eval metadata reconstruction duplicates daemon (stale ref)

**Locations:** `evals/run_eval.py:263-291`; `daemon.py:504-527`; `daemon.py:137-168`; `daemon.py:79-82`

**Evidence:** run_eval.reconstruct_thread_metadata's docstring says "Mirrors daemon.py:137-160 — sort by internalDate, extract unique senders, first-message subject, last-message snippet" — but daemon.py:137-160 is now resolve_newsletter_llm_endpoint/FailureTracker; the mirrored logic actually lives at daemon.py:504-527. The repo's own stated anti-drift principle is the opposite technique: DEFAULT_MAX_THREAD_CHARS is "Shared with the eval harness (evals/run_eval.py) so the two never drift" (daemon.py:80-82), and format_thread_transcript is imported. The copies already differ at an edge: daemon skips a no-sender thread ("Thread %s has no valid senders, skipping", daemon.py:515) while the eval falls back ("senders = golden.senders  # Fallback to harvested senders", run_eval.py:280).

**Why it matters:** A maintainer changing how the daemon assembles ThreadMetadata (sender dedup, snippet choice, subject source) has no reliable pointer to the eval copy — the anchoring comment points at unrelated code — so the eval silently starts measuring different inputs than production sees. This is a direct mechanism for "the fix looked fine in evals but behaved differently in production". The tension (import-and-share vs hand-mirror) is resolvable but currently applied inconsistently and documented misleadingly.

**Verifier note:** Verified: run_eval.py:266's "Mirrors daemon.py:137-160" now points at resolve_newsletter_llm_endpoint/FailureTracker; the mirrored logic lives at daemon.py:504-527, the repo's stated anti-drift technique is import-and-share (daemon.py:79-82 "so the two never drift", format_thread_transcript imported), and the copies already diverge — daemon skips a no-sender thread (daemon.py:514-516) while the eval falls back to golden.senders (run_eval.py:279-280).

### 94. [MEDIUM | contradiction | confirmed] Inline vs standalone report paths disagree

**Locations:** `evals/newsletter_run.py:686-716`; `evals/newsletter_report.py:109-119`; `evals/newsletter_report.py:1076-1077`; `evals/README-technical.md:352-357`; `evals/report.py:423`; `evals/newsletter_report.py:966-967`; `evals/run_eval.py:501-506`

**Evidence:** newsletter_report.compute_all_metrics takes `mode` because "it tells tier metrics whether errored rows count as quality failures (they don't in a themes-only run)" (newsletter_report.py:117-118), and README-technical.md:355-357 promises "a themes-only results file cannot resurrect a tier section full of fake errors." But newsletter_run.maybe_report calls `compute_all_metrics(story_preds, extraction_preds, match_threshold=match_threshold)` (newsletter_run.py:687-689, and again for the compare baseline at 709-711) with mode defaulting to "all", while the standalone CLI passes `mode=meta.mode` (newsletter_report.py:1076-1077) — so `newsletter_run --mode themes --report` on a run with an errored row shows tier errors that the standalone report on the same file omits. Same twin-drift pattern on the email side: report.print_trend globs `results_dir.glob("*.jsonl")` with no `.cot.jsonl` filter (report.py:423) while newsletter_report filters sidecars (newsletter_report.py:967), and run_eval.maybe_report warns on stage mismatch (run_eval.py:501-506) but `report --compare` never does.

**Why it matters:** Two entry points that claim to print "the report for this run" produce different numbers/sections for the identical results file. A maintainer verifying a fix via `--report` and a reviewer re-checking via `evals.newsletter_report` see conflicting error counts and cannot tell which is authoritative. The mode parameter was itself a fix (per README-technical) that was not threaded through the inline path — a concrete instance of the fix-spawns-new-problems symptom.

**Verifier note:** Verified: newsletter_run.maybe_report calls compute_all_metrics without mode (newsletter_run.py:687-689 and 709-711, defaulting mode="all" per newsletter_report.py:113) while the standalone CLI passes mode=meta.mode (newsletter_report.py:1076-1077); compute_tier_metrics with mode="all" counts errored rows as tier errors and print_report renders the section whenever tier["errors"] is nonzero, so a --mode themes --report run with an errored row shows tier errors the standalone report (and README-technical.md:352-357's promise) omits; the twin-drift sub-claims also check out (report.py:423 globs *.jsonl with no .cot.jsonl filter vs newsletter_report.py:966-967; run_eval.py:501-506 warns on stage mismatch, report.py --compare at 500-502 never warns — print_comparison only silently skips inapplicable sections).

### 95. [MEDIUM | gap | confirmed] Newsletter-level aggregation unmeasured

**Locations:** `daemon.py:421-428`; `evals/newsletter_run.py:224-279`; `evals/newsletter_report.py:372-401`; `evals/README.md:206-211`; `CLAUDE.md:59`

**Evidence:** Production's Gmail-visible outputs are newsletter-level: "# Determine overall tier (best story's tier)" (daemon.py:421) and aggregate_theme_grades merging "strongest grade per theme" (daemon.py:427-428). The eval scores only per-story units on fixed golden text (evaluate_story, newsletter_run.py:224-279; tier metrics are per-story, newsletter_report.py:372-401). README.md:206-211 documents the decoupling rationale ("quality and theme scoring is *decoupled* from it ... so a prompt tweak that only affects scoring is measured cleanly") but nowhere states the blind spots: (a) the composed pipeline (extraction feeding scoring) is never measured, so extraction drift's effect on production gradings is invisible in quality/theme runs; (b) best-of-stories overall tier and cross-story theme merging have no eval at all. CLAUDE.md:59 even describes the overall tier as computed "from the averaged dimension scores", which matches the per-story tier the eval measures, not the best-story aggregation production applies.

**Why it matters:** A maintainer reading "tier accuracy 85%" cannot tell whether that predicts the tier label a newsletter actually receives in Gmail — it does not, because story membership and the best-of aggregation are outside the metric. A fix that improves per-story scoring while worsening extraction (or vice versa) can pass evals and regress production, and the docs never warn that this composition is unmeasured.

**Verifier note:** Verified: production's Gmail-visible outputs are best-story tier (daemon.py:421-426) and strongest-grade theme merging (daemon.py:427-428), while the eval scores only per-story units on fixed golden text (newsletter_run.py:224-279; per-story tier metrics newsletter_report.py:372-401) and no eval mode composes extraction into scoring or measures newsletter-level accuracy; CLAUDE.md:59 indeed describes the overall tier as "from the averaged dimension scores", omitting the best-story aggregation. Nuance: aggregate_theme_grades has unit tests (tests/test_newsletter.py:342-356) proving the merge logic, but nothing measures aggregated outputs against ground truth, which is the claimed gap.

### 96. [MEDIUM | ambiguity | confirmed] Stage 2 accuracy meaning varies by mode

**Locations:** `evals/run_eval.py:347-356`; `evals/run_eval.py:359-374`; `evals/report.py:154-166`; `evals/report.py:421-456`; `evals/README.md:124-139`

**Evidence:** In stage2_only the eval feeds the oracle: "# Use expected sender type as input (skip Stage 1)" (run_eval.py:348-349). In full mode the label comes from routing by the *predicted* sender type, and `result.label_correct = result.predicted_label == golden.expected_label` is set unconditionally (run_eval.py:374) — so a mis-routed person thread's label (produced by the cloud model instead of the local one) still counts in "Stage 2 accuracy". report.py labels both simply "--- Stage 2: Label Classification ---" (report.py:255), and the trend table shows full and stage2_only runs in one column (report.py:432-456). Neither README defines which population/routing "Stage 2 accuracy" refers to.

**Why it matters:** When a full run's Stage 2 number moves after a fix, the maintainer cannot tell whether Stage 2 (the label prompt/model) changed or Stage 1 routing changed which model produced the labels — and comparing a stage2_only local-model run against a full run's Stage 2 column is an apples-to-oranges comparison the tooling renders without definition. Ambiguous metric semantics are a classic way a fix "regresses" a number for reasons unrelated to the fix.

**Verifier note:** Verified: stage2_only feeds the oracle sender type (run_eval.py:348-350) while full mode routes by the predicted type and sets label_correct unconditionally (run_eval.py:374), so a mis-routed person thread's cloud-produced label still counts in full-mode "Stage 2 accuracy"; report.py:255 labels both identically, the trend table pools both run types in one Stage 2 column (rows do carry a Stages cell, but the metric semantics are undefined), and neither README defines the full-mode population/routing — README.md:104-105 only glosses stage2_only as "uses expected sender type as input".

### 97. [MEDIUM | gap | confirmed] prompt_hash blind to code and taxonomy changes

**Locations:** `evals/newsletter_run.py:107-116`; `evals/README.md:244-246`; `evals/README-technical.md:428-434`; `newsletter.py:37-54`; `newsletter.py:190-203`; `evals/newsletter_schemas.py:304-330`; `evals/schemas.py:141-166`

**Evidence:** compute_prompt_hash hashes only `config["newsletter"]["prompts"]` (newsletter_run.py:114-116); README.md:245-246 says "Each run records a `prompt_hash`, so variants are self-identifying." README-technical.md:434 documents one blind spot (inference params: "the hash covers prompts only") but not the larger one: the theme taxonomy (`_VALID_THEMES`), grade tokens, and tier thresholds ("excellent >= 2.75...", newsletter.py:196-203) plus all parsers live in code, and neither NewsletterRunMeta nor RunMeta records any code version/git SHA — two runs with identical prompt_hash and config_hash can differ solely because parse_stories/parse_themes/compute_tier changed. README-technical.md:257-261 shows this class of event has already happened once ("This changed the extraction/quality/theme prompt_hash, so runs recorded before the change are no longer prompt-comparable") — that one happened to touch prompts; a parser-only fix would not be flagged at all. The email eval records no prompt-level hash at all, only whole-config config_hash — an undocumented asymmetry between the twin harnesses.

**Why it matters:** When a trend row's metric shifts between runs with the same prompt_hash, the maintainer's first question — "what changed?" — has no recorded answer if the change was code. Cross-run comparisons after a parser fix silently mix incomparable populations, so a fix's before/after evidence can be wrong without anyone noticing.

**Verifier note:** Verified: compute_prompt_hash hashes only config["newsletter"]["prompts"] (newsletter_run.py:114-116); _VALID_THEMES/grade tokens (newsletter.py:37-54), tier thresholds (newsletter.py:196-203), and all parsers live in code; neither NewsletterRunMeta nor RunMeta records any code version/git SHA, so a parser-only fix leaves two runs hash-identical yet incomparable — the class of event README-technical.md:257-261 shows already happened once. Minor nuance: both metas also record system prompts verbatim, so the email side's gap is specifically "no prompt-level hash" (its trend table falls back to config_hash), which is literally true as stated.

### 98. [MEDIUM | tension | confirmed] Thinking backfill semantics under-specified

**Locations:** `evals/llm_cache.py:101-136`; `evals/llm_cache.py:108-125`; `evals/README.md:122`; `evals/README-technical.md:451-459`

**Evidence:** On a hit with unknown thinking, the cache re-calls the LLM (`self.misses += 1 ... await self.inner.complete(...)`, llm_cache.py:109-114) and pairs the NEW call's thinking with the OLD cached response (`self._cache[key] = (response, thinking)`, llm_cache.py:129) — the sidecar CoT may not be the reasoning that produced the recorded answer; neither the code nor README-technical.md:451-459 states this pairing caveat. The guard comment says the cached response is "never discard[ed] ... over a thinking backfill" (llm_cache.py:121), but only `LLMContentError` is caught (llm_cache.py:115): an LLMUnavailableError/RuntimeError during backfill propagates and errors the row despite a perfectly usable cached response. Meanwhile README.md:122 promises "re-runs with the same config are instant" — a cache full of pre-#64 `""` entries instead triggers one paid re-fetch per entry, and fails entirely offline.

**Why it matters:** The maintainer cannot answer "when is a cached result reusable, and what exactly does the sidecar CoT correspond to?" from any single place — the intent ("never discard"), the code (discards on network error), and the docs ("instant") each tell a different story. Debugging a prompt from a backfilled CoT that never produced the cached answer is a direct path to a wrong fix.

**Verifier note:** Verified: on a backfill hit the new call's response is discarded and its thinking is paired with the OLD cached response (llm_cache.py:112-114 `_, thinking = ...`, 129), a caveat stated nowhere including README-technical.md:448-459; only LLMContentError is caught (llm_cache.py:115), so an LLMUnavailableError during backfill propagates and errors the row despite the "never discard it over a thinking backfill" comment (llm_cache.py:119-120); and README.md:122's "re-runs with the same config are instant" is false for a cache of unmarked pre-#64 "" entries, which each trigger a paid re-fetch (and produce errored rows offline) since evals always request include_thinking=True.

### 99. [MEDIUM | ambiguity | confirmed] VIP path diverges between eval stages and is unrecorded

**Locations:** `evals/run_eval.py:347-356`; `evals/run_eval.py:362-366`; `classifier.py:196-200`; `classifier.py:238-253`; `classifier.py:316-322`; `evals/README.md:13-16`; `evals/run_eval.py:637-659`

**Evidence:** Production always computes VIP (`vip = self._is_vip(metadata)`, classifier.py:318) and narrows the Stage 2 prompt for VIP senders; the eval's full mode mirrors this (run_eval.py:363-366) but stage2_only calls `classifier.classify_email(metadata, transcript, sender_type)` with vip defaulting False (run_eval.py:350) — so the canonical local-model configuration (`--local-only`) never exercises the VIP-narrowed prompt production would use for VIP person threads. The VIP list comes from env (`os.environ.get("VIP_SENDERS", ...)`, classifier.py:198), and evals share the daemon's env via the `.env` symlink (README.md:13-16), so in full mode a VIP sender short-circuits Stage 1 with raw "VIP" and no LLM call (classifier.py:239-240) — Stage 1 "accuracy" then partially measures the address list, not the model. RunMeta records vip_email_system_prompt (run_eval.py:659) but never the VIP address list, so runs from shells with/without VIP_SENDERS set are indistinguishable.

**Why it matters:** It is genuinely unclear whether the eval intends to measure the bare model or the system-with-VIP-shortcuts, and the answer currently differs by stage mode without documentation. Two machines (or one machine before/after editing .env) produce different Stage 1 populations and different Stage 2 prompts for the same golden set and the result files carry no trace of why.

**Verifier note:** Verified: production always computes VIP (classifier.py:318) and full-mode eval mirrors it (run_eval.py:363-366), but stage2_only — the canonical --local-only configuration per resolve_local_only (run_eval.py:38-63) — calls classify_email with vip defaulting False (run_eval.py:350), never exercising the VIP-narrowed prompt; VIP addresses come from env VIP_SENDERS (classifier.py:198), shared with evals because run_eval imports daemon which runs load_dotenv() (daemon.py:47) on the README.md:13-16 .env symlink; a VIP sender short-circuits Stage 1 with raw "VIP" and no LLM call (classifier.py:239-240, 252-253); RunMeta records vip_email_system_prompt but never the VIP address list.

### 100. [MEDIUM | tension | confirmed] Golden newsletter body frozen at harvest

**Locations:** `evals/newsletter_harvest.py:150`; `evals/README-technical.md:164-166`; `evals/run_eval.py:294-297`; `daemon.py:79-82`; `evals/newsletter_schemas.py:128-141`

**Evidence:** The two harnesses take opposite fidelity strategies: the email eval stores raw Gmail messages and rebuilds the transcript at run time through the imported production formatter ("Reconstruct thread transcript using daemon.format_thread_transcript()", run_eval.py:295) — explicitly "so the two never drift" (daemon.py:81) — while newsletter_harvest bakes the transcript once at harvest (`body = format_thread_transcript(messages, max_thread_chars)`, newsletter_harvest.py:150) and GoldenNewsletter stores only that string ("body: str  # raw body fed verbatim to extract_stories", newsletter_schemas.py:135) with no raw messages, so it cannot be regenerated. README-technical.md:165 calls the body "identical to production input", which is only true as of harvest time: a later change to format_thread_transcript or max_thread_chars changes production input but not the frozen golden bodies.

**Why it matters:** After a transcript-formatting fix, extraction evals keep silently measuring the old input format — the eval can stay green while production extraction now sees different text, or flag a 'regression' that is really input skew. Nothing documents which strategy is intended or that the newsletter golden set goes stale on formatter changes and cannot be refreshed without a full re-harvest and re-label.

**Verifier note:** Verified: newsletter_harvest.py:150 bakes body via format_thread_transcript at harvest time and GoldenNewsletter stores only that string with no raw messages (newsletter_schemas.py:128-141), while the email eval stores raw messages and rebuilds at run time through the imported production formatter (run_eval.py:294-297, per daemon.py:79-82's "never drift" principle) — so README-technical.md:164-166's "identical to production input" is true only as of harvest, and a formatter/max_thread_chars change silently skews extraction evals; nothing documents the staleness (note: the freeze is partly forced by design, since golden stories are verbatim slices of the frozen body, which makes the undocumented tradeoff more worth stating, not less).

### 101. [LOW | contradiction | confirmed] Eval clients omit tier despite same-as-daemon claim

**Locations:** `evals/run_eval.py:562-580`; `daemon.py:822-841`; `daemon.py:859-868`; `evals/newsletter_run.py:630-648`

**Evidence:** run_eval.py:562 says "# Build LLM clients (same as daemon.run_daemon())" and newsletter_run.build_classifier's docstring says "mirroring daemon.run_daemon()" (newsletter_run.py:636), but the daemon passes `tier="cloud"` / `tier="local"` (daemon.py:830, 840, 867) while both eval constructors omit it — eval-side LLMUnavailableError carries tier=None and error strings render "tier=-" via _provider().

**Why it matters:** Small on its own, but it makes the "same as daemon" comments literally false, and each such near-mirror is one more place a future constructor change (a new LLMClient parameter) gets applied in production and missed in evals — the pattern behind fidelity drift.

**Verifier note:** Verified: run_eval.py:562 says "Build LLM clients (same as daemon.run_daemon())" and newsletter_run.py:635 says "mirroring daemon.run_daemon()", yet the daemon passes tier="cloud"/"local" (daemon.py:830, 840, 867) while both eval constructors (run_eval.py:563-580, newsletter_run.py:640-648) omit it, so eval-side errors render "tier=-" via _provider (llm_client.py:150-153) — the same-as-daemon comments are literally false.

### 102. [LOW | ambiguity | confirmed] duration_seconds and cache provenance unrecorded

**Locations:** `evals/run_eval.py:384-390`; `evals/newsletter_run.py:169-177`; `evals/schemas.py:75`; `evals/run_eval.py:690-696`; `evals/schemas.py:141-166`

**Evidence:** duration_seconds means different things per run: "Use actual LLM time when caching (0 for pure cache hits); wall time otherwise" (run_eval.py:384), mirrored in newsletter_run._elapsed (newsletter_run.py:169-177) — but the results row and RunMeta carry no flag saying which semantics applied, and whether the run used the cache at all (hit/miss stats print only to stderr, run_eval.py:690-696, and are not persisted).

**Why it matters:** From a results file alone one cannot tell whether predictions were fresh LLM samples or cache replays, nor compare durations across runs — so a 'latency regression' or 'accuracy change' seen between two files may just be cache-on vs cache-off, and there is no recorded evidence to rule that out.

**Verifier note:** Verified: duration_seconds is miss-only LLM time when cached but wall time when not (run_eval.py:384-390, newsletter_run.py:169-177), no flag in PredictionResult (schemas.py:75) or either RunMeta says which semantics applied or whether the cache was used, and hit/miss stats go only to stderr (run_eval.py:690-696) — a results file cannot distinguish fresh samples from cache replays.

## Module boundaries & ownership

**Dimension summary:** Module boundaries in this repo are mostly well-intentioned but inconsistently declared and almost never enforced. The two "copied from email-agent" files have in fact forked hard (a new error taxonomy, retry layer, and body-cleaning pipeline exist only here) while README-technical still calls them "shared with email-agent" — no sync/ownership policy exists anywhere, which is the single most dangerous ambiguity for future fixes. The TUI layer has a stated convention ("shared widgets/screens live in tui_common.py") that practice contradicts: scroll/resize/wrap logic is copy-pasted across TUIs, including a deliberately divergent wrap_text twin, and shared vocabulary travels via underscore-private imports. The tui-regression skill harness states its division of labor vs the pytest suite clearly, but only inside .claude/ — it is invisible from every README, run by no CI, and its "add a scenario in the same change" rule is the one doc-sync obligation in the repo with no enforcing test, in a codebase that elsewhere teaches maintainers that tests catch doc rot.

### 103. [HIGH | contradiction | confirmed] copied-vs-shared email-agent files

**Locations:** `CLAUDE.md:33-34`; `README-technical.md:14-15`; `proxy_client.py:36-46`; `gmail_utils.py:33-96`

**Evidence:** CLAUDE.md:33-34 says proxy_client.py / gmail_utils.py are '(copied from email-agent)'; README-technical.md:14-15 says the same files are '(shared with email-agent)'. Reality: git log shows 12 local commits to proxy_client.py and 3 to gmail_utils.py since the initial 'Initial project setup: pyproject.toml, shared files, config' commit (f39e744), including the whole ProxyUnavailableError/TRANSIENT_TRANSPORT_ERRORS error taxonomy (issues #16/#26/#27/#32) and the entire body-cleaning pipeline (strip_html/shorten_urls/clean_body). Diffing against the actual email-agent checkout (/Users/robergb/bin/email-agent) shows 227 diff lines in proxy_client.py; the SAME exception name now means different things in the two repos — upstream ProxyError: 'Raised for other proxy errors (5xx, connection errors, etc.)' vs here: 'Raised for a request-specific proxy fault — a 4xx response other than 401/403'. email-agent's CLAUDE.md/README make no reciprocal sharing claim. No sync policy (frozen mirror / fork / keep-in-sync) is stated anywhere in either repo.

**Why it matters:** Every fix that touches these files forces an unanswerable question: is this file mine to edit, must the change be ported to email-agent, and is it ever safe to 'sync' from email-agent? A well-meaning sync in either direction would silently revert the daemon's give-up/retry error model or the body-cleaning pipeline — a textbook way for a fix to spawn a batch of new problems. 'Shared with email-agent' is now factually false; a maintainer trusting it will mis-scope any change to Gmail parsing or proxy error handling.

**Verifier note:** All evidence verified: CLAUDE.md:33-34 says '(copied from email-agent)' while README-technical.md:14-15 says '(shared with email-agent)'; diff vs /Users/robergb/bin/email-agent shows exactly 227 lines for proxy_client.py (103 for gmail_utils.py); ProxyError's docstring means different things in the two repos (upstream '5xx, connection errors, etc.' vs local '4xx other than 401/403'); ProxyUnavailableError/TRANSIENT_TRANSPORT_ERRORS and strip_html/shorten_urls/clean_body exist only in email-labeler; no sync policy or reciprocal claim exists in either repo. (Trivial nit: 11 post-initial local commits to proxy_client.py and 2 to gmail_utils.py — the 12/3 figures include the initial commit.)

### 104. [MEDIUM | tension | confirmed] wrap_text twins with divergent semantics

**Locations:** `newsletter_review/tui.py:192-203`; `evals/newsletter_label.py:689-704`; `evals/newsletter_label.py:646-686`

**Evidence:** Two same-named, same-signature functions: newsletter_review/tui.py:192 wrap_text uses textwrap.wrap (character count); evals/newsletter_label.py:689 wrap_text docstring says 'Mirrors ``newsletter_review.tui.wrap_text`` (``width <= 0`` disables wrapping and empty *text* yields ``[""]``) but wraps by display width, so emoji/wide characters are never clipped at the screen edge' (backed by display_width/_wrap_paragraph). The cross-reference is one-directional: newsletter_review's copy has no pointer to its twin, and the display-width fix was never applied to it, so the browser still clips emoji/CJK that the labeler handles.

**Why it matters:** Resolvable tension, but as it stands a maintainer fixing wrapping in one TUI has no signal (in one direction, none at all) that a behavioral twin exists; and anyone 'deduplicating' them into tui_common must silently change one TUI's rendering. The mirror-but-diverge pattern is exactly how a fix in one place leaves or creates a defect in its unadvertised twin.

**Verifier note:** Verified: newsletter_review/tui.py:192-203 wrap_text uses textwrap.wrap (character count) with a docstring that never mentions its twin; evals/newsletter_label.py:689-704 wrap_text's docstring quote matches verbatim and wraps by display_width/_wrap_paragraph (646-686). The cross-reference is one-directional exactly as claimed, and the display-width fix was never back-ported to the browser.

### 105. [MEDIUM | tension | confirmed] tui_common boundary unstated

**Locations:** `README-technical.md:405-408`; `tui_common.py:18-23`; `evals/edit_tui.py:203-222`; `newsletter_review/tui.py:441-460`; `evals/edit_tui.py:265-268`; `newsletter_review/tui.py:527-530`; `evals/newsletter_label.py:537-548`

**Evidence:** Convention (README-technical.md:405): '**Shared widgets/screens** live in `tui_common.py`'. In practice, edit_tui.DetailScreen and newsletter_review DetailScreen carry byte-identical scroll-action blocks (action_scroll_up/page_up/scroll_home...), and both apps' on_resize handlers repeat the comment verbatim: 'self.size is still the OLD size while this handler runs — use the event\'s.' tui_common.truncate claims it is 'Shared by the list-row renderers across the TUIs (issue #49)', yet newsletter_label's row renderer uses raw slicing instead ('sender = (newsletter.sender or "")[:24]') and its own story_excerpt with a '…' ellipsis. The skill doc also records a guard-idiom divergence: 'All four apps guard re-entrant push_screen with len(self.screen_stack) > 1 (review.py uses self.screen is not screen_stack[0])' (SKILL.md:104-106).

**Why it matters:** Looks resolvable, but there is no stated criterion for what graduates into tui_common versus what stays per-TUI, so shared behaviors accrete as copies. A fix to any of these behaviors (scroll, resize re-render, truncation, re-entrancy guard) lands in one copy and a review later 'discovers' the unfixed siblings — directly feeding the fix-creates-new-problems symptom.

**Verifier note:** Verified: README-technical.md:405 states shared widgets live in tui_common.py, yet edit_tui.py:206-222 and newsletter_review/tui.py:444-460 carry byte-identical six-method scroll blocks (tui_common has no scroll mixin — only PageListView's list paging), both on_resize handlers repeat the OLD-size comment verbatim (edit_tui.py:265-268, tui.py:527-530), tui_common.truncate:18-23 claims 'Shared by the list-row renderers across the TUIs' while newsletter_label.py:537 raw-slices '[:24]' and story_excerpt:541-548 uses its own '…' ellipsis, and SKILL.md (~line 104) records the review.py screen-stack-guard divergence. No graduation criterion is stated anywhere.

### 106. [LOW | ambiguity | confirmed] private names as cross-module contracts

**Locations:** `newsletter_review/tui.py:18`; `evals/newsletter_label.py:51`; `evals/edit_tui.py:17-19`; `evals/review.py:31-32`

**Evidence:** The read-only browser imports a private name from the production pipeline module: 'from newsletter import _SCORE_TOKENS' (newsletter_review/tui.py:18); the labeling TUI does too ('from newsletter import _SCORE_TOKENS, compute_tier', newsletter_label.py:51). edit_tui imports private key maps from a sibling tool: 'from evals.review import _LABEL_KEY_MAP, _SENDER_KEY_MAP, save_golden_set' with the comment 'Hotkey maps come from evals.review so the two tools stay in lock-step ... No cycle: review imports edit_tui lazily' (edit_tui.py:17-19). Nothing states that these underscore names are stable interfaces.

**Why it matters:** The single-source-of-truth intent is good, but the underscore convention says 'free to change' while three modules depend on exactly these names. A maintainer refactoring newsletter.py or review.py 'internals' has no documented way to know which privates are load-bearing across module boundaries — is _SCORE_TOKENS part of newsletter.py's public contract or not? The answer currently exists only in the importing files.

**Verifier note:** Verified: newsletter_review/tui.py:18 'from newsletter import _SCORE_TOKENS'; newsletter_label.py:51 'from newsletter import _SCORE_TOKENS, compute_tier'; edit_tui.py:17-19 imports _LABEL_KEY_MAP/_SENDER_KEY_MAP from evals.review (defined review.py:31-32) with the lock-step comment. newsletter.py:49's comment explains the rubric but says nothing about external importers — stability knowledge lives only in the importing files, as claimed.

### 107. [MEDIUM | gap | adjusted] tui-regression harness invisible and unenforced

**Locations:** `.claude/skills/tui-regression/SKILL.md:20-23`; `.claude/skills/tui-regression/SKILL.md:131-133`; `README-technical.md:395-428`; `README-technical.md:8-56`

**Evidence:** SKILL.md does state the division of responsibility ('complementary to the unit-level Pilot tests already in `tests/` (those check one behavior each; these run realistic multi-step workflows ...)' and 'a matching `tests/` unit test is the durable home for the regression'), and imposes a sync duty: 'When you add a binding or a workflow branch, add a scenario here in the same change and re-run `run_all.py`.' But no README, CLAUDE.md, or the README-technical structure tree mentions the harness at all (grep for 'tui-regression' outside .claude/ returns nothing); there is no CI (.github/workflows does not exist) despite run_all.py calling itself 'suitable for CI or a pre-merge gate'; and unlike the repo's other doc-sync duties (test_tui_docs.py, test_newsletter_eval_docs.py, test_env_var_docs.py), nothing enforces the scenario-per-binding rule.

**Why it matters:** The 68 scenarios encode every binding of all four TUIs — a second, hand-synced copy of the UI's behavior. A maintainer who follows only the repo's visible docs will change a binding, keep pytest green, and never run or update the harness; the drift then surfaces later as a wall of 'failures' someone must triage as driver-rot vs app bug. That deferred discovery is precisely the fixes-uncover-new-problems pattern, and the repo's own partial automation teaches maintainers to assume a test would have caught it.

**Verifier note:** The 'unenforced' half fully holds (no .github/ at all, grep for 'tui-regression' outside .claude/ returns nothing in README/CLAUDE.md/structure tree, run_all.py:8 self-describes as CI-suitable, and no test enforces the SKILL.md:131-133 scenario-per-binding duty), but 'invisible' is overstated: the skill is auto-surfaced to Claude Code agents in the available-skills listing with explicit triggers ('Use this after changing any TUI, tui_common, the ... schemas, or the extraction/parse path'), so agent-driven changes — this repo's primary workflow per its agent-oriented docs — do get a discovery signal. The durable gap is enforcement (no CI, no doc-sync test) plus the missing pointer in human-facing docs; severity medium stands.

### 108. [LOW | contradiction | confirmed] run_all claims model mocked

**Locations:** `.claude/skills/tui-regression/run_all.py:7-8`; `.claude/skills/tui-regression/SKILL.md:31`; `.claude/skills/tui-regression/SKILL.md:70-71`

**Evidence:** run_all.py docstring: 'then drives all four TUIs via Pilot with the model mocked.' SKILL.md, same directory: 'Manual-only since issue #59 removed LLM seeding — **no TUI calls a model**, so there is no model seam to mock' (line 31) and '2. **No model to mock.** None of the four TUIs calls an LLM' (line 70).

**Why it matters:** One of these is stale (the run_all docstring predates issue #59). Small, but a maintainer extending the harness may hunt for a mock seam that no longer exists, or reintroduce one 'to match the docs' — and it shows fixes here don't sweep the two-file blast radius even within a single directory.

**Verifier note:** Verified verbatim at exact lines: run_all.py:7 'with the model mocked' vs SKILL.md:31 'no TUI calls a model, so there is no model seam to mock' and SKILL.md:70 'No model to mock. None of the four TUIs calls an LLM (issue #59 removed…)' — plus SKILL.md line 20 'fully offline; no TUI calls a model'. Irreconcilable stale docstring in the same directory.

### 109. [MEDIUM | tension | confirmed] python floor: 3.14 pin vs 3.11 harness

**Locations:** `pyproject.toml:5`; `.claude/skills/tui-regression/SKILL.md:39-46`

**Evidence:** pyproject.toml: 'requires-python = ">=3.14"'. SKILL.md: 'The repo pins `requires-python >=3.14`, but the code runs fine on 3.11–3.13' and instructs building a throwaway /tmp venv via a hand-listed pip command ('httpx python-dotenv fastapi jinja2 uvicorn python-multipart "textual>=8.2.8" pytest pytest-asyncio pytest-subtests ruff') that duplicates the pyproject dependency list.

**Why it matters:** Resolvable, but today a maintainer cannot answer 'may I use 3.14-only syntax?': the packaging metadata says yes, while the only end-to-end TUI harness depends on 3.11–3.13 compatibility through an out-of-band venv recipe. Using a 3.14 feature is doc-legal yet silently breaks the harness; adding a dependency to pyproject silently breaks the hand-copied pip line. Both are deferred breakages that surface as 'new problems' on a later fix.

**Verifier note:** Verified: pyproject.toml:5 'requires-python = ">=3.14"'; SKILL.md:39-46 says 'the code runs fine on 3.11–3.13' and its /tmp-venv pip line hand-duplicates the pyproject dependency list exactly (all 7 runtime deps incl. 'textual>=8.2.8' pin, plus the 4 dev deps) — both drift hazards are real.

### 110. [MEDIUM | gap | adjusted] structure/coverage inventories selectively enforced

**Locations:** `README-technical.md:8-56`; `README-technical.md:430-449`; `tests/test_newsletter_eval_docs.py:3-6`; `CLAUDE.md:29-38`; `CLAUDE.md:173-185`

**Evidence:** test_newsletter_eval_docs.py describes README-technical.md as 'the project's structure/reference doc: it lists the project's modules under ``## Project Structure`` and maps each source module to its tests under ``## Test Coverage by Module``' — but the structure tree omits retry.py, evals/edit_tui.py, evals/llm_cache.py, evals/web_app.py, evals/run_web.py, evals/web_auth.py, evals/web_data.py, and evals/templates/, and the coverage table has 16 rows while tests/ holds 31 test modules (absent: test_eval_review.py, test_eval_edit_tui.py, test_proxy_client.py, test_gmail_utils.py, test_retry.py, test_tui_common.py, test_llm_cache.py, test_eval_model.py, test_smoke_concurrency.py, test_eval_run.py, test_eval_util.py, and the doc meta-tests). Enforcement exists only in islands: newsletter_* eval modules (test_newsletter_eval_docs), TUI launch docs (test_tui_docs), env vars, CLI flags. CLAUDE.md's Project Structure names exactly one of the three scripts (migrate_assessments.py; eval_model.py and smoke_concurrency.py appear only in evals/README.md:274-275 and README-technical.md:338/README.md:276) and omits newsletter.py — the entire newsletter pipeline module — from its module list.

**Why it matters:** All three scripts are in fact maintained tools with tests, but a reader cannot tell that from any single document, nor tell whether an omission from the tree/table is policy or rot. Because doc-completeness is test-enforced for some islands and not others, maintainers learn 'a test will fail if I forget the docs' — true exactly where it doesn't matter and false where it does — so every fix in the unenforced zones tends to leave a doc inconsistency for the next review to 'discover'.

**Verifier note:** Core confirmed with two detail corrections: tests/ holds 32 test modules (not 31) vs the 16-row coverage table (all 11 named absentees verified absent, plus 5 doc meta-tests; 16+11+5=32); the structure-tree omissions (retry.py, evals/edit_tui.py, llm_cache.py, web_app.py, run_web.py, web_auth.py, web_data.py, templates/) and CLAUDE.md's omission of newsletter.py are all verified, as is the test_newsletter_eval_docs.py:3-6 self-description and the island enforcement. Correction: eval_model.py and smoke_concurrency.py DO appear in README-technical.md's Project Structure tree (lines 42 and 44), so 'appear only in evals/README.md:274-275 and README-technical.md:338/README.md:276' is wrong (README.md:276 has no such mention either) — the script-invisibility sub-claim applies to CLAUDE.md's structure list only.

## Completeness critic — what the ten dimensions missed

**Overall assessment:** The ten dimensions covered the daemon's classification logic, privacy claims, config surface, error taxonomy, and doc drift thoroughly at the level of individual facts, but the audit has three systematic blind spots. First, whole subsystems escaped every dimension: the eval web application (the repo's only network service, serving real email content with auth off by default, contradicting the 'no web server' decision at the dependency level), the release/deployment layer (no tags, no versions, an untracked compose stack, issue numbers as de facto release identifiers), and the logging layer (doc-quoted log lines functioning as an unguarded operator contract). Second, format and process governance went unexamined: the assessments JSONL — declared the only durable copy of gradings — has no documented schema and no version field despite one breaking change already, and the mandated TDD/meta-test regime has no CI gate enforcing it. Third, and most consequential, the audit does not adequately explain the owner's stated symptom. The surviving findings (3-4x fact duplication, doc drift, unresolved policy tensions) account for part of the review-finding volume, but the dominant generators are unnamed: there is no registry of adjudicated tradeoffs and no review-scope charter, so every review pass — like the audit itself, which reported the same two tensions across three dimensions each — rediscovers deliberate decisions as findings; and the documentation is pinned at implementation altitude (exact log strings, literal counts, thresholds, issue numbers in docstrings), so nearly any code change falsifies prose somewhere and each falsification becomes a finding. Until those two are fixed, resolving the audit's 110 individual findings will reduce but not stop the per-fix finding churn.

### Missed: The eval results web application (evals/web_app.py, web_auth.py, web_data.py, run_web.py, evals/templates/) received zero findings from any dimension

**Why it matters:** The privacy dimension flagged eval artifacts' at-rest posture as a low-severity gap, and design-decisions flagged the 'no web server' wording — but nobody examined the actual web server sitting in the tree. It is the component that turns person-email bodies into an HTTP-served resource with auth off by default, which is the sharpest untested edge of the privacy invariant, and it directly contradicts a stated design decision at the dependency/image level.

**Evidence:** This is the only network service in the repo, and it serves harvested real email content (web_data.py loads golden context and chain-of-thought sidecars derived from actual Gmail threads). /Users/robergb/tools/email-labeler/evals/web_auth.py line 15-16: authentication is silently disabled whenever EVAL_WEB_SECRET is unset (`if not SECRET: return True`); lines 17-18: the session cookie stores the raw shared secret verbatim. EVAL_WEB_SECRET appears only in evals/README.md — it is absent from .env.example and from README-technical's env table, and tests/test_env_var_docs.py scans only root-level *.py so evals env vars escape the doc guard. Meanwhile CLAUDE.md Design Decision 4 says 'No web server: Pure asyncio daemon' while pyproject.toml lists fastapi, uvicorn, jinja2 (and textual) as main non-dev dependencies, which the Dockerfile's `uv sync --frozen --no-dev` installs into the production daemon image.

### Missed: Release, versioning, and deployment identity: there is no scheme at all, and the deployment config is out-of-repo

**Why it matters:** A time-sensitive runbook keyed to 'the first post-#65 image' is unactionable when nothing — no tag, no version, no image label — lets the owner determine which code a running container contains. For an audit whose theme is docs drifting from code, the inability to even name a deployed version is a foundational clarity gap, and the untracked compose stack means the most failure-prone configuration (the assessments volume mount) lives where no doc test or reviewer can see it.

**Evidence:** `git tag` returns nothing; pyproject.toml is frozen at version 0.1.0; there is no CHANGELOG. Releases are identified in normative docs by GitHub issue numbers — CLAUDE.md and docs/runbook-agent-attempted-recovery.md say the runbook is 'cleanest before the first post-#65 image is deployed'. README.md instructs `docker compose build email-labeler` and references host paths like /srv/stack/data/..., but no compose file is tracked anywhere in the repo, so the volumes, env wiring, and restart policy that Design Decision 6's sink preflight depends on are defined in an invisible external stack. The Dockerfile's `COPY *.py ./` glob is the de facto (unstated) definition of what counts as production code.

### Missed: The assessments JSONL record format is undocumented and unversioned despite being 'the only durable copy'

**Why it matters:** This format has already had one breaking change (#53) that required a migration script and a shape-sniffing error message, and Design Decision 6 declares it the sole durable record of every grading. A durable format with no canonical schema doc and no version marker guarantees the next evolution repeats the migration pain and risks silent misreads. The audit covered dedup ordering and the migration script's existence but never the format-governance question.

**Evidence:** The record is a raw dict literal in /Users/robergb/tools/email-labeler/newsletter.py (append_assessment, lines 273-294) with no schema-version field. The only field-by-field appearance of the format in any document is the stale plan docs/plans/2026-02-20-newsletter-tui-plan.md; README-technical documents only what the #53 migration alters, not the format itself. Consumers (newsletter_review, scripts/migrate_assessments.py, and the append docstring's own admission that 'old records lacking these keys are read with .get() fallbacks') each independently re-infer the schema, and old-vs-new scheme is detected by shape.

### Missed: Logging and observability conventions: doc-quoted log lines function as an operator contract with no guard, and no level policy exists

**Why it matters:** The project's operator interface is substantially made of specific log strings, which is exactly the doc-vs-code drift failure mode this audit targets, in a subsystem no dimension examined. A reworded log line silently breaks the runbook and README instructions with no meta-test to catch it, unlike CLI flags and env vars which do have doc guards.

**Evidence:** README.md:205, README-technical.md:214/220, and the recovery runbook quote exact log lines (down to a literal example count, '412 existing record(s)') and instruct the operator to look for them; the halt behavior is likewise specified via its ERROR line text. Issue #58 ('log observability') deliberately designed the idle heartbeat, reconnect line, and httpx quieting (daemon.py lines 56-76 with in-code rationale). Yet no test pins any doc-quoted line (grepping tests/ for 'Newsletter assessments append to' finds nothing), no document states what warrants ERROR vs WARNING, and llm_client.py — the module where every provider failure originates — contains zero log calls while daemon.py has ~47.

### Missed: No CI or merge gate: the entire TDD/meta-test regime is convention-only

**Why it matters:** tests-as-spec flagged missing tests but never that the whole suite is unenforced at merge time. The elaborate meta-test lattice only defends the docs if someone runs it; with agent-driven development the multi-pass adversarial reviews visible in history are structurally compensating for the absent gate, which connects directly to the owner's complaint below.

**Evidence:** There is no .github directory or any CI configuration in the repo. CLAUDE.md mandates red/green TDD, mutation-proving, and 'always run the full test suite before declaring any task complete', and the codebase invests heavily in doc meta-tests (test_env_var_docs, test_tui_docs, test_eval_cli_docs, etc.) — but nothing enforces that pytest or ruff ever runs on a PR. Enforcement rests entirely on agent discipline.

### Missed: Runtime coupling with the sibling email-agent system (shared .env, shared MLX server/model, shared proxy, shared label namespace) is uncharacterized

**Why it matters:** Two independently-evolved daemons sharing an env file, a single-model LLM server, a proxy, and a label namespace with no documented ownership or compatibility contract is a classic cross-system clarity hazard: a change in the other repo can alter this one's behavior with no signal in this repo's docs, tests, or review.

**Evidence:** The boundaries dimension covered file provenance ('copied-vs-shared email-agent files') but not runtime coupling: evals/README-technical.md notes the eval tools read 'the same env as the daemon (see the .env symlink note)' — the env file is shared across repos; MLX_MODEL is 'shared with email-agent so both services use the same model' (README-technical.md:69), so a model change in the other repo silently reconfigures this daemon; scripts/eval_model.py's docstring warns 'Stop the daemon first if your MLX server only serves one model at a time', revealing single-model server contention between systems; both systems ride the same api-proxy and the `agent/*` Gmail label namespace.

### Missed: Intra-module cohesion of daemon.py: a 1074-line monolith with a ~320-line do-everything function

**Why it matters:** Nearly every behavioral fix lands inside process_single_thread or its neighbors, so each fix has a wide blast radius through unrelated concerns (halt logic, privacy routing, sink ordering, marker labels) — a per-fix finding multiplier that a design-clarity audit of 'boundaries' should have surfaced.

**Evidence:** daemon.py contains env parsing, logging setup, config loading, FailureTracker, DaemonHalt, transcript formatting, cycle summarization, idle reporting, label-retry, sink preflight, and the run loop; process_single_thread spans lines 318-637 and interleaves both pipelines' routing, error taxonomy, and labeling in one function. The boundaries dimension examined cross-module and cross-repo seams but produced no finding about the main entry point's internal structure.

### Missed: The owner's core complaint is only partially explained: no accepted-tradeoffs registry or review-scope charter exists, and docs are pinned at implementation altitude — the two strongest finding-generators went unnamed

**Why it matters:** This is the question the audit was implicitly hired to answer. The surviving findings explain perhaps half the review churn (duplication, drift, unresolved tensions); without naming the missing decision registry, the absent review charter, the implementation-altitude doc style, and the no-CI compensation dynamic, the owner will fix individual drifts and still watch every future fix spawn a fresh crop of findings.

**Evidence:** The symptom is real and quantifiable: 18 commits are dedicated review-fix commits ('Fix 22 findings from adversarial review of the Textual migration', 'Fix 10 findings from adversarial review of the Phase 1 diff', 'Apply ultracode review fixes', 'Apply xhigh review fixes'), with PRs routinely carrying 2-3 review rounds. Two causes absent from all 110 findings: (1) Deliberate tradeoffs have no adjudicated registry — the audit itself demonstrates the mechanism by rediscovering the same safe-default-vs-privacy tension in three separate dimensions and the daemon-wide-halt tension in three; review agents do the same on every pass because nothing marks a tension as already-decided, and no CONTRIBUTING/review-policy document says which review passes run, what severity bar applies, or which docs are normative versus historical (stale plans carrying 'live agent directives' was flagged, but not the missing charter that would neutralize them). (2) Docs are written at implementation altitude — CLAUDE.md and the READMEs embed exact log strings, a literal example count ('412 existing record(s)'), tier thresholds, hotkey tables, and issue numbers even inside production docstrings (newsletter.py line 270: 'issue #35/#36') — so nearly any code change falsifies prose somewhere, and diff-reading reviewers convert each falsified sentence into a finding. docs-consistency flagged drift instances and 3-4x duplication, but never the altitude/style root cause that makes the drift inevitable rather than incidental.
