# Implementation roadmap — all open issues on `brianroberg/email-labeler`

## Context

All 28 open issues were triaged against `HEAD b0b7631` (see each issue's comment for
current-state verification). This document is the *forward* plan: it groups the issues
into coherent workstreams, sizes them, maps dependencies, and recommends an order of
attack. It stays at the epic/sequencing level — no code.

**The organizing insight for prioritization:** the owner has just begun curating the
first *newsletter* golden set (#53's framing) and is exercising the golden-set editors
(#52). That is the live frontier, so the plan front-loads the work that de-risks and
unblocks that curation, then the measurement infrastructure that everything else
depends on, then correctness, then polish. Two rules of thumb drive the ordering:

1. **Don't tune what you can't measure.** Prompt/accuracy work is gated by a trustworthy
   golden set. Grow/curate the set *before* investing in prompt changes.
2. **Change the rubric before you mass-label under it.** Schema/scoring changes that
   invalidate existing labels (#53) must land before heavy labeling, or the labels get
   redone.

Priority key: **P0** now / time-sensitive · **P1** next · **P2** later · **P3** opportunistic.
Effort: **S** (hours) · **M** (a day-ish) · **L** (multi-day / needs a spike).

---

## Epic A — Newsletter scoring model & golden-set curation *(the active frontier)*

Directly serves the curation the owner is doing right now.

| Issue | What | Priority | Effort | Depends on |
|---|---|---|---|---|
| **#53** | Switch theme scores binary→3-value (Absent/Present/Emphasized) and storytelling dimensions 1–5→3-value (Poor/OK/Good); update prompts, schema, tier derivation, labels | **P0** | M–L | — |
| **#52** | Add an *exclude* action to `evals.review --edit` (only unexclude exists today); keep parity with the newsletter labeler | **P0** | S | — |
| **#41** | Deferred newsletter-eval UX items. Triaged 2026-07-08: items **2** (cursor-direct selection) and **3** (`X` newsletter exclude) already done by the Textual redesign; item **7** (`is_available` 404 detail) done here; items **1** ($EDITOR/sub-line), **4** (`q` quits), **5** (unreviewed extraction metrics — schema change), **6** (`--verbose` trend) deferred with rationale in `docs/plans/2026-07-08-phase1-decisions.md` | **P1** | S–L (triage per sub-item) | — |
| **#35** | Newsletter review-TUI header: show sent-date *and* processed-date, separate email-intrinsic vs classification data, show model | **P1** | M | upstream capture |
| **#36** | Newsletter listing: show send-date column, default sort by date desc, date filters (30/90/365/Since…) | **P1** | M | upstream capture |

**Sequencing within A:**
- **#53 first** — it changes the scoring schema, so labeling done under the old scheme
  would need redoing. Land it before the golden set grows large. Expect follow-up
  prompt-tuning at the rubric boundaries (the owner already flagged this as out of scope
  for #53 itself — file as a follow-up).
- **#52 is a quick, high-leverage curation-friction fix** — do it early alongside #53.
- **#35 + #36 share one prerequisite:** the assessment record (`write_assessment`)
  doesn't persist the email's send-date or the classifier model. Do that upstream
  capture *once*, then both TUI issues become straightforward. Treat "persist send-date
  + model in the assessment JSONL" as the shared enabling step.
- **#41** is a grab-bag: re-scope it against the current (redesigned) TUI and cherry-pick
  the still-relevant sub-items; item 1 (character-level spans / `$EDITOR` handoff) is the
  only large one and is optional.

---

## Epic B — Newsletter classifier quality *(downstream of eval + golden set)*

Improves *what* the newsletter classifier decides. Gated by having the eval harness
(exists) plus a curated golden set (Epic A, in progress) to measure against.

| Issue | What | Priority | Effort | Notes |
|---|---|---|---|---|
| **#8** | Refine SCRIPTURE theme to distinguish "reading the Bible" from "correctly handling Scripture" | **P1–P2** | S–M | Clear precedent: sibling themes were tightened in `7759dea`; apply the same pattern |
| **#4** | Distinguish personal vs ministry stories (personal recorded but doesn't "count" unless included) | **P2** | M–L | Needs a new schema concept (per-story class + "counts" semantics) + prompt + eval coverage |
| **#6** | Recognize repeat/similar stories across newsletters | **P3** | L (spike first) | Speculative; likely needs embeddings + storage. Do a feasibility spike before committing |

**Sequencing:** #8 is the cheapest and has a direct precedent — do it first, once the
rubric change (#53) has settled so you're not tuning a prompt that's about to change
shape. #4 is a genuine feature (schema + semantics). #6 is research — gate it behind a
spike that answers cost/feasibility.

---

## Epic C — Email classification accuracy *(gated by the email golden set)*

| Issue | What | Priority | Effort | Depends on |
|---|---|---|---|---|
| **#13** | Grow the email golden set to ≥60 reviewed threads, balanced across labels/sender types. **Sub-tasks added:** (1) a read-only `--stats` mode on `evals.review` (composition dashboard — total / excluded / reviewed-unexcluded, plus sender×label crosstab); (2) a `--sender-type person\|service` filter on `evals.review` (it has `--filter-label` but no sender filter today) so the reviewed-pending backlog can be drained toward thin cells. Both are small, reuse `load_golden_set`/`SENDER_TYPES`/`LABELS`, and steer the curation | **P1** | M–L (human review) | — *(enables the rest)* |
| **#14** | Fix the stable under-classification (needs_response→fyi ×1, fyi→low_priority ×2) via prompt-criteria tuning. **⚠ These tallies predate the 252-thread expansion** — re-run `evals.run_eval` + `evals.report --verbose` against the current set before acting | **P1** | M | **#13** |
| **#12** | Reduce local latency: A/B a lean prompt (drop the reasoning scaffold) vs current; adopt if accuracy holds | **P1–P2** | M | **#13** |
| **#2** | Reconcile golden-set labels with whole-thread state (already-replied → FYI); optionally add an explicit prompt rule | **P2** | S–M | overlaps #13 |

**Sequencing:** **#13 is the keystone** — it unblocks #14 and #12 and makes every
accuracy number trustworthy. Fold #2's label-consistency pass into the #13 curation
effort (same activity, same data). Then #14 (correctness of the rare, high-stakes class)
before #12 (latency, which must not regress accuracy).

**Current status (measured 2026-07-08, after the owner's expansion pass):** 252 reviewed
& usable threads out of 401 harvested (2 excluded) — the ≥60 *count* target is met (~4×).
Label balance is fine (needs_response 47 / fyi 99 / low_priority 106); the earlier
"needs_response barely represented" worry is resolved. Full sender×label crosstab of the
scored set:

```
           needs_response   fyi   low_priority   total
    person             20    32              4      56
   service             27    67            102     196
    total              47    99            106     252
```

The remaining lever is **sender balance**: person is 56/252 (~22%) — up from the earlier
45, but still the thin side, and it's the privacy-critical, locally-served path
(#14/#12). The weakest cell (person/low_priority = 4) likely reflects a true low base
rate — don't force it. Growth from here is mostly a *reviewing* task: ~147
harvested-but-unreviewed threads (401 − 2 excluded − 252 reviewed) still sit on disk, so
the new `--stats` and `--sender-type` review filters are the tools to drain that backlog
toward person coverage, then re-baseline accuracy.

---

## Epic D — Daemon resilience & correctness

| Issue | What | Priority | Effort | Notes |
|---|---|---|---|---|
| **#30** | *(bug)* Newsletter content-less error swallowed → commits an empty "no-stories" grade; pollutes the assessment corpus | **P1** | M | Intersects Epic A — the corpus the owner is curating/reviewing |
| **#28** | *(bug/question)* Proxy 403 (human-rejected write) counts toward give-up → can silently abandon a gated thread | **P1 (decision) / P2 (impl)** | S + M | Needs the A/B/C product decision first; fix must also cover the marker-write 403 path |
| **#29** | *(enh)* Write-phase outage discards a completed classification, re-running the expensive local LLM | **P2** | M | Now bounded to ~max_failures cycles; still wasteful |
| **#33** | *(enh)* `write_sem` held across a whole per-message write loop (wrong granularity; 5 call sites incl. the new one from PR #38) | **P2** | S–M | Correctness fine; throughput/fairness |
| **#24** | Local-LLM outage logged as WARNING per-thread; should be INFO + one per-cycle summary (tier-aware) | **P2** | S | `is_available()` already exists, unused in the loop |

**Sequencing:** **#30 first** (it's a correctness bug *and* it corrupts the very corpus
Epic A is building — treat it as part of protecting the newsletter data). Make the
**#28 product decision** early (cheap, unblocks a real operational risk) even if the
implementation waits. **#29 + #33** both live on the write path — bundle them. **#24** is
an easy quality-of-life win that can slot in anytime.

---

## Epic E — Documentation

| Issue | What | Priority | Effort | Notes |
|---|---|---|---|---|
| **#15** | Human READMEs: add the give-up bullet to Resilience, add the new eval flags + thinking-A/B workflow to `evals/README.md`, reference `smoke_concurrency.py`, fix the stale `qwen3-14b` example | **P1** | S | Closes a real, small inconsistency |
| **#39** | Generalize the doc-completeness guard to the whole repo + backfill missing modules/tests; add a guarded project-structure outline to `README.md` | **P2** | L | Valuable guardrail, but bigger |

**Sequencing:** **#15 now** — it's cheap and the stale model example is actively
misleading. **#39 after the code churn settles** (Epics A–D add/rename modules; a
completeness guard is most useful — and least thrash-prone — once the module set is
stable).

---

## Epic F — TUI shared-infrastructure tech-debt *(from the Textual migration)*

All `tech-debt`, all "not blocking." Value = lower cost/risk of *future* TUI change.

| Issue | What | Priority | Effort | Layer |
|---|---|---|---|---|
| **#50** | Extract atomic `save_golden_set` JSONL write into a shared eval helper | **P2** | S | foundation |
| **#51** | Unify the dismiss-guard + "modal owns the keys" idioms into `tui_common` | **P2** | S | foundation |
| **#49** | Unify `wrap_text`/`truncate`/`excerpt` into `tui_common` (behavior-sensitive) | **P2** | M | foundation |
| **#48** | Extract shared `ScrollableDetailScreen` + list-resize helper | **P2** | M | foundation |
| **#45** | Extract a `mutation_flow` decorator for the `_begin_flow`/`_end_flow` boilerplate (prevents wedge bugs) | **P2–P3** | S | safety |
| **#46** | Avoid full row rebuilds / re-filtering on non-structural refreshes (perf) | **P3** | M | perf |
| **#47** | Bound undo memory via copy-on-write snapshots | **P3** | M | memory |

**Sequencing — the key judgment:** these touch the *same* TUI files as Epic A
(`newsletter_label.py`, `newsletter_review/tui.py`, `edit_tui.py`, `review.py`). To avoid
editing the same duplication twice, **fold the relevant consolidations into Epic A's TUI
work** rather than running Epic F as a separate campaign:
- Doing **#52 / #35 / #36 / #41** already means opening those files — extract the shared
  helpers (**#50, #51, #49, #48**) *while you're there*.
- **#45** (mutation_flow) pairs naturally with any `newsletter_label` flow change.
- **#46 / #47** are pure optimizations that only bite on large newsletters; lowest
  priority — do only if profiling or memory shows they matter.

---

## Recommended phased roadmap

**Phase 1 — Unblock & protect the active newsletter curation.**
#53 (rubric change, before mass-labeling) → #52 (exclude editing) → #30 (stop corrupting
the corpus) → the shared send-date+model capture, then #35 + #36 → triage #41. Opportunistically
fold in the TUI-consolidation tech-debt (#50/#51/#49/#48/#45) as those files are touched.

**Phase 2 — Measurement & decisions.**
#13 (grow the email golden set; fold in #2's consistency pass) — can run in parallel with
newsletter curation. #15 (cheap doc fixes). Make the #28 product decision. #24 (log-noise win).

**Phase 3 — Accuracy & resilience.**
After #13: #14 then #12 (email accuracy/latency). After the rubric settles: #8 then #4
(newsletter quality). Implement #28; bundle #29 + #33 (write path).

**Phase 4 — Hardening & long-tail.**
#39 (doc-completeness guard, once modules are stable). Remaining TUI perf/memory
debt (#46/#47) if warranted. Spikes for #6 (repeat stories) and the #7 staff-cross-ref
script — both standalone and lowest urgency.

---

## Notes

- **#52 and #53 are brand-new** (owner-filed today) and were verified but intentionally
  left un-commented during triage; they are fully incorporated above. #53's own scope is
  just the scheme switch — boundary-tuning of the new rubric is expected follow-up.
- **No issue is obsolete.** Two were already Complete and closed during triage (#5, #23);
  everything remaining is live.
- **Biggest force multipliers:** #13 (unblocks all email-accuracy work) and #53
  (prevents re-labeling the newsletter set). Do these early.
- **Cheapest high-value wins:** #52, #15, #24, #8, #50, #51.
- This roadmap is deliberately code-free; each epic would get its own implementation
  plan (with TDD per the repo's red/green rule) when picked up.
