# Phase 2 implementation — confirmed decisions

Companion to `docs/plans/2026-07-08-issue-roadmap.md` (Phase 2 — Measurement &
decisions) and sibling to `2026-07-08-phase1-decisions.md`. Implemented on branch
`claude/phase2-measurement`, red/green TDD per `CLAUDE.md`, one commit per task.

Phase 2 scope: **#13 (tooling sub-tasks) → #15 (doc fixes) → #28 (product
decision) → #24 (log-noise fix)**, with #2 folded into #13's curation activity.

## Confirmed product decision (owner, 2026-07-09)

### #28 — proxy 403 (rejected/blocked write) semantics: **Option A**

A `ProxyForbiddenError` (403) on a write means an operator rejected the gated
write in the approval UI (or the op is blocked). Chosen semantics: **re-surface,
never give up** —

- Catch `ProxyForbiddenError` distinctly in the per-thread path: log one clean
  line (no traceback), do **not** record a failure toward give-up, return False
  so the thread is re-offered next cycle. "Rejected" = "ask me again later";
  the human stays in control, and a rejection can never cause `agent/attempted`.
- The same policy must cover the **marker-write path**: a 403 on the
  `agent/attempted` write during `_give_up_if_stuck` is logged distinctly (no
  traceback spam) and the give-up is not recorded — current behavior minus noise.
- Accepted residual: a *permanently blocked* op re-asks every cycle (bounded:
  one log line + one pending approval per cycle). If that becomes annoying,
  revisit the deferred option — teaching the proxy to distinguish "operator
  rejected" from "permanently blocked" (separate status/body) and applying B/C
  to the latter — as a follow-up, since it needs cross-repo api-proxy changes.
- **Implementation is Phase 3** (per the roadmap). It must add the missing test
  exercising a 403 through `process_single_thread` (today only the HTTP layer's
  `test_403_not_retried` covers 403 at all).

## Task dispositions (this branch)

### #13 — golden-set growth tooling (both sub-tasks shipped)

- `evals.review --stats`: read-only composition dashboard —
  total / excluded / unreviewed-pending / reviewed-&-unexcluded partition plus a
  sender × label crosstab over the reviewed-&-unexcluded set (the population
  `run_eval` scores). Unknown sender/label values get their own row/column so
  totals never silently undercount. Pure `format_stats_summary()` for testability
  (mirrors `newsletter_run.format_load_summary` precedent).
- `evals.review --sender-type person|service`: mirrors `run_eval`'s flag name;
  applies in review mode (via `select_review_threads`) and in `--edit` mode,
  where — matching `--filter-label`'s existing semantics — an explicit filter
  replaces the reviewed-only default.
- The **count** target (≥60) is already met (252 reviewed & usable of 401 at
  last measurement); the remaining lever is **person coverage** (~22%). That is
  a human reviewing task: drain the ~147 pending threads with
  `evals.review --unreviewed-only --sender-type person`, watching `--stats`.
  Then re-baseline accuracy (feeds #14).

### #2 — already-replied consistency (folded into #13, no code)

- The golden-set reconciliation ("rate the thread as a whole; already-replied →
  FYI") is part of the owner's curation pass — same activity, same data — aided
  by `--show-labels` review and the new `--stats`/`--sender-type` tooling. The
  golden set is gitignored, so no repo change is possible here.
- The **optional prompt rule** ("a message that called for a reply no longer
  needs one if you've already replied") is **deferred to #14**: it's a
  prompt-criteria change and must be measured against the re-baselined set
  (roadmap rule 1: don't tune what you can't measure). The issue itself notes
  the CoT already applies this reasoning unprompted.

### #24 — local-LLM outage logging (shipped)

- Tier-awareness lives **on the exception**: `LLMClient(tier="cloud"|"local")`
  and `LLMUnavailableError.tier`, set at both raise sites (connect-class
  failures and mid-request drops). The daemon handler can't infer which endpoint
  failed (Stage 1 is cloud, Stage 2 is either), so provenance is attached where
  it's known. Eval/newsletter constructions stay tier-less → keep WARNING.
- Local-tier outage: per-thread at **DEBUG** + one per-cycle **INFO** summary
  (`log_local_deferrals`, fed by a per-cycle accumulator passed like
  `failure_tracker`). Cloud/tier-less: WARNING unchanged. Deferral semantics
  untouched (no give-up, no cloud fallback).
- The issue's `is_available()` pre-check suggestion was obsolete — Phase 1
  replaced it with `probe()`. A probe-based short-circuit of N doomed Stage-2b
  calls was **not** added (acceptance didn't need it; adjacent to #29's
  wasted-work concern — note it there if picked up).

### #15 — human-README gaps (shipped)

- README.md Resilience now documents: transient-vs-request-specific split,
  stuck-thread give-up (`agent/attempted`, distinct per-cycle abandonment
  summary), and the routine-local-outage summary line (from #24, same branch).
- evals/README.md gained the per-endpoint `--*-timeout` / `--*-extra-body` /
  `--*-no-think` examples and a "Thinking on/off A/B" Typical Workflow (incl.
  the GLM-native note and the cache-key independence point).
- `scripts/smoke_concurrency.py` referenced from README-technical's Local Model
  Serving section.
- Stale model example: `qwen/qwen3-14b` → **`qwen/qwen3.6-27b`** in README.md
  and `.env.example` ("MLX/Qwen3" mentions → "MLX/Qwen3.6"). ⚠ The exact HF id
  was inferred from issue #15's "Qwen3.6-27B actually in use" — owner should
  confirm it matches the id the server actually serves (`MLX_MODEL` must equal
  the served model name or mlx_lm.server 404s).
