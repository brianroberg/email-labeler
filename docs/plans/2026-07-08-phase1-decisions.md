# Phase 1 implementation — confirmed decisions

Companion to `docs/plans/2026-07-08-issue-roadmap.md`. Phase 1 is being implemented
**autonomously** (owner authorization, 2026-07-08): work through the whole roadmap
order, stopping only for genuine product decisions. Red/green TDD is mandatory
(see `CLAUDE.md`). Local commit per sub-project; no push.

Roadmap order: **#53 → #52 → #30 → assessment-capture → #35 → #36 → #41 (triage)**,
folding in TUI tech-debt (#50/#51/#49/#48/#45) opportunistically where those files
are already open.

## Confirmed product decisions (owner, 2026-07-08)

### #53 — newsletter scoring schemes

- **Storytelling dimensions**: 1–5 numeric → 3-value **Poor / OK / Good**.
- **Themes**: boolean → 3-value **Absent / Present / Emphasized**.
- **Tier derivation** (excellent/good/fair/poor): map Poor/OK/Good → 1/2/3, take the
  **mean of the four dimensions** (simple, concrete, personal, dynamic), then band:

  | tier | average |
  |---|---|
  | excellent | ≥ 2.75 |
  | good | 2.25 – 2.74 |
  | fair | 1.75 – 2.24 |
  | poor | < 1.75 |

  (all-OK = 2.0 → fair; one Poor among three Good ≈ 2.5 → good.)

- **Theme Gmail labels**: apply `agent/newsletter/theme/<name>` **only when the theme
  is Emphasized**. Present and Absent are still persisted in the assessment record and
  compared in the eval/report, but do NOT get a Gmail label. (Deliberately changes the
  existing label's meaning from "present" to "emphasized".)
- Scope is the scheme switch only. Boundary prompt-tuning of the new rubric is expected
  follow-up, filed separately — not part of #53.

### #30 — content-less error handling

- Stories-exist-but-every-grade-errored is a **failure**, never a committed outcome.
  Raise a dedicated content-less exception from the `complete()` guard; re-raise it from
  `classify_newsletter` alongside `LLMUnavailableError` so it reaches the daemon give-up
  path (retry, then `agent/attempted`). Per-story parse-level `RuntimeError` stays
  isolated (`test_non_transient_quality_error_stays_isolated` must stay green).
- Genuine NO_STORIES (extraction returns zero stories) remains a valid committed
  `no-stories` outcome — must stay distinct from the failure path.
- **Sub-problem (b)** of #30 (non-convergence / wasted re-grading on a flaky endpoint):
  **deferred** — it's a separate enhancement (adjacent to #29), medium/low severity, and
  the confirmed correctness bug (a) is the P1 target. Noted as follow-up.

## Representation decisions (autonomous; from the understand-phase map)

- **Dimension scores** stored internally as **ints 1/2/3** (Poor/OK/Good), not string
  tokens — preserves `average_score`, the mean-based tier, and eval MAE math with minimal
  churn. Prompt emits `POOR|OK|GOOD` tokens; `parse_quality_scores` maps tokens→ints (a
  digit or unknown token is a parse failure — the old 1–5 clamp / `clamped_dimensions`
  feature becomes obsolete). The read-only review **detail** view renders ints→labels
  (Poor/OK/Good); the labeling TUI's compact story strip keeps the dense `2/3/1/2` int
  form (space-constrained editor grid). `_DIMENSIONS` unchanged.
- **Themes** stored as **`dict[str,str]`** mapping theme→`"present"`/`"emphasized"`,
  Absent omitted. `parse_themes`, `StoryResult.themes`, `expected_themes`/
  `predicted_themes`, the assessment JSONL, and the golden set all take this shape.
  Daemon cross-story aggregation = max grade per theme (emphasized > present > absent).
  **No backward-compat to the old scheme** (owner decision): the readers assume the
  new dict/1–3 shapes (no list→present coercion, no 1–5 fallback) and the golden set /
  assessment JSONL are regenerated under the new scheme. The parsers still *reject* old-
  format LLM output (a bare theme name or a digit score → parse failure), which is
  robustness against a misbehaving model, not data backward-compat.
- **Gmail theme label** (owner decision: Emphasized-only) → the daemon applies
  `agent/newsletter/theme/<name>` only for themes whose aggregated grade is `emphasized`.
- **Report metrics** (post-#53 follow-up):
  - **(b) within-1: DONE** — dropped from the dimension table + metrics bundle; on a 1–3
    scale the only non-within-1 pair is a Poor↔Good confusion, so it was ~always 1.0 and
    added nothing over MAE (now 0–2) + exact-match.
  - **(a) theme metric: DONE — chose A′ (hybrid).** The **primary** metric
    (`metrics["themes"]`) is now **Emphasized-positive** (per-theme + micro/macro F1 +
    exact-set-match) — exactly what earns a Gmail label, so it is production-aligned; a
    **secondary** `metrics["themes_detection"]` micro-F1 (positive = ≥Present) preserves
    the "did we detect the theme at all" signal (owner: present detection is
    secondary-but-not-nothing). Pure A was rejected because it makes Present detection
    entirely invisible. **Option C** (full 3-class per-theme with the downgrade-vs-miss
    confusion) is deferred to a filed issue — take it up if A′'s two aggregate F1s stop
    being enough to explain *how* emphasized detection fails during prompt-tuning.

## #41 triage outcome (verified against HEAD)

- **Item 2** (cursor-direct story-row selection) — **DONE**, close.
- **Item 3** (newsletter-level `X` exclude) — **DONE** (commit c4e3487; roadmap omitted
  this), close. It is the reference impl to mirror for #52.
- **Item 7** (`is_available` returns a bare bool) — **PICK UP**, bundled with #30 (same
  file, `llm_client.py`; both make LLM failures self-describing). Add status detail; gate
  the preflight 404 hint on an actual 404.
- **Item 1** ($EDITOR / sub-line spans) — **DEFER** (large; off the active path; prefer a
  Textual `TextArea` modal if ever taken up).
- **Item 4** (`q` quits app from detail) — **DEFER** (cross-TUI design opinion; Esc
  already backs out; auto-save makes it lossless).
- **Item 5** (unreviewed pollute extraction metrics) — **DEFER** (schema change on
  `ExtractionPrediction`; mitigated by a stderr warning today; risks double-migration
  amid the #53 golden-set churn — bundle with dedicated measurement work).
- **Item 6** (`--verbose` in trend mode) — **DEFER** (lowest value; mitigated).

Roadmap line 35 should be updated to acknowledge item 3 is done.

## TUI tech-debt fold-ins (only while the file is already open)

- **#50** `atomic_write_jsonl(records, path)` → `evals/__init__.py`; `review.save_golden_set`
  and `newsletter_label.save_golden_set` become thin delegating wrappers (both names kept
  so `from evals.review import save_golden_set` and test imports don't break). Fold in
  during #52/#53.
- **#49 `_truncate`** → `tui_common.truncate()` (edit_tui + newsletter_review copies are
  byte-identical). Fold in during #52/#35/#36.
- **#48 scroll base** (`ScrollableDetailScreen`) — only if it drops in cleanly with
  **per-subclass BINDINGS** (Textual merges BINDINGS across the MRO; a shared base would
  otherwise leak newsletter_review's `ctrl+b/ctrl+f` aliases into edit_tui). Otherwise skip.
- **SKIP**: #49 `wrap_text` (divergent whitespace handling), #48 list-resize (divergent
  bodies), #51 (idiom already unified in `tui_common`), #45 (single-file, `@work`-fragile,
  concurrency-test-pinned).
