# Newsletter-Classification Evaluation System

## Context

The daemon's newsletter pipeline (`newsletter.py`, active under `NEWSLETTER_ONLY=1`) grades ministry-newsletter stories: it **extracts stories** from a body, **scores** each on 4 quality dimensions (simple/concrete/personal/dynamic, 1–5) → a **tier** (excellent/good/fair/poor), and tags **themes** (scripture/christlikeness/church/vocation-family/disciple-making). Today there is **no way to measure whether this grading is any good** — assessments are written to JSONL and browsed read-only via the `newsletter_review/` TUI, but never scored against ground truth.

A mature eval harness already exists for the *email* classifier (`evals/`: harvest → review → run_eval → report, with a disk LLM cache, golden set, chain-of-thought sidecars, metrics, comparison). This plan builds the parallel harness for **newsletter** grading so we can **iterate on the prompts** (the stated purpose) and measure the effect of each change.

**Decisions (from the user):** evaluate all three outputs (tier/dimension scores, themes, extraction); build a **human labeling tool** for ground truth (no auto-labels exist — quality is subjective); use **fixed golden stories** so quality/theme scoring is decoupled from extraction variability; optimize the tooling for **prompt iteration** (per-run comparison + cache reuse).

**Key reconciliation** of "evaluate extraction" + "fixed golden stories": one golden set holds both. Extraction is scored at the **newsletter-body level** (feed raw body → predicted stories → match against the newsletter's golden story list). Quality + themes are scored on the **fixed golden `(title, text)`** inputs, independent of what extraction produced.

## Approach

Mirror the email eval's 4-stage shape with newsletter-specific modules under `evals/` (prefixed `newsletter_` to avoid clashing with the top-level read-only `newsletter_review/` package). Reuse everything reusable — the LLM cache, metric primitives, curses scaffolding, transcript builder — and only write what's genuinely new.

### New modules (all under `evals/`)

| File | Responsibility | Entry point | Key flags |
|---|---|---|---|
| `newsletter_schemas.py` | `GoldenStory`, `GoldenNewsletter`, `NewsletterRunMeta`, `NewsletterPrediction`, `NewsletterThinkingEntry` dataclasses with `to_dict`/`from_dict` (missing-key tolerant) | — | — |
| `newsletter_harvest.py` | Pull candidate newsletters from Gmail, seed golden set (no ground truth) | `python -m evals.newsletter_harvest` | `--output`, `--max-threads`, `--recipient`, `--config`, `--proxy-url` |
| `newsletter_label.py` | Curses/CLI tool: curate story list, then per-story scores + themes | `python -m evals.newsletter_label` | `--golden-set`, `--edit`, `--unreviewed-only` |
| `newsletter_run.py` | Replay golden set through real `NewsletterClassifier` (cached); write results + cot + meta | `python -m evals.newsletter_run` | `--golden-set`, `--config`, `--mode {extraction,quality,themes,all}`, `--tag`, `--no-cache`, `--parallelism`, `--prompts`, `--model`, `--report`, `--compare-to`, `--skip-preflight` |
| `newsletter_report.py` | Tier/dimension/theme/extraction metrics + two-run comparison + trend | `python -m evals.newsletter_report` | `--results`, `--compare`, `--results-dir`, `--verbose`, `--format {table,json}`, `--match-threshold` |

**Reuse (import, do not fork):** `evals.llm_cache.CachedLLMClient`, `evals.__init__.format_network_error`, `evals.report.{compute_confusion_matrix, compute_precision_recall_f1, compute_accuracy, format_pct, format_table_row, format_confusion_matrix, format_per_class_table, print_trend}`, `newsletter.{NewsletterClassifier, parse_stories, compute_tier, is_newsletter, NewsletterTier}`, `daemon.{resolve_newsletter_llm_endpoint, format_thread_transcript}`, `gmail_utils.get_header`, the `GmailProxyClient` + preflight patterns from `evals/harvest.py`/`evals/run_eval.py`, and the curses/atomic-save scaffolding from `evals/review.py` + `evals/edit_tui.py`.

### Golden-set schema (`evals/newsletter_schemas.py`)

One `GoldenNewsletter` per JSONL line in **`evals/newsletter_golden_set.jsonl`**; it owns its list of `GoldenStory`. Nesting stories mirrors how `newsletter.write_assessment()` already nests `stories:[...]`.

```python
@dataclass
class GoldenStory:
    story_id: str                                   # stable, f"{thread_id}:{index}"
    title: str
    text: str
    expected_scores: dict[str, int] | None = None   # simple/concrete/personal/dynamic, 1-5
    expected_tier: str | None = None                # derived from scores via compute_tier
    expected_themes: list[str] = field(default_factory=list)
    reviewed: bool = False
    notes: str = ""
    excluded: bool = False                          # drop from quality/theme scoring, keep as extraction truth

@dataclass
class GoldenNewsletter:
    thread_id: str; message_id: str; sender: str; subject: str
    body: str                                       # raw body fed verbatim to extract_stories
    stories: list[GoldenStory]
    source: str = "harvested"; harvested_at: str = ""
    reviewed: bool = False                          # story list confirmed = authoritative extraction truth
    notes: str = ""; excluded: bool = False
```

- **Extraction ground truth** = the confirmed `stories[*].(title,text)` of a `reviewed` newsletter.
- **Quality/theme ground truth** = each reviewed story's `expected_scores`/`expected_tier`/`expected_themes`.

### Stage details

**Harvest** — follows `evals/harvest.py`: `GmailProxyClient` + network-error handling; query by newsletter recipient (`to:<recipient>`, default from `config["newsletter"]["recipient"]`, mirroring `daemon.py`'s `gmail_query += f" to:{...}"`); per thread run `is_newsletter(...)` as the guard, build `body` with `daemon.format_thread_transcript(...)` (identical to production input at `daemon.py:348`), capture message_id/sender/subject. **No ground truth inferred** — seed `stories=[]`, `reviewed=False`. Dedup by `thread_id`, always-append (never truncate).

**Label** — reuses `review.py`/`edit_tui.py` patterns; two phases per newsletter:
- *Phase A — curate stories / set boundaries (extraction truth):* **Seed candidates by running the production `parse_stories` extractor once on the body** (a cheap one-time LLM call, cached), so the story list starts pre-populated. The reviewer then confirms/fixes every boundary: `[s]` split an over-merged candidate, `[m]` merge two candidates, `[e]` edit a title/text span, `[a]` add a missed story, `[d]` delete a non-story. `[Enter]` confirms the list → `newsletter.reviewed=True`; each confirmed story gets a stable `story_id`. The confirmed spans (not the raw seed) are what's stored, so the human always has the final say — but see risk #7 (mild extraction-recall bias from seeding with the system under test; mitigated by mandatory per-boundary confirmation, and `NewsletterRunMeta`/report note the seed source).
- *Phase B — per-story labels:* step the 4 dimensions (hotkeys `1`–`5`), then multi-select themes (`1`–`5` toggling the five themes, matching `newsletter_review`'s convention); `expected_tier` auto-derived via `compute_tier` on save; `[n]` notes, `[z]` undo, `story.reviewed=True`.
- Atomic temp-file+rename save; queue selection excludes `excluded`, honors `--unreviewed-only`. Because boundaries are fuzzy-matched (≥0.6) at eval time, they need to be recognizably-the-same-story, not character-perfect.

**Run** — like `run_eval.py`: load config (`tomllib` + env substitution), resolve endpoint via `resolve_newsletter_llm_endpoint()`, build one `LLMClient` from `[newsletter.llm]`, wrap in `CachedLLMClient(evals/cache/llm_cache.jsonl)` unless `--no-cache`, preflight via `is_available()`. **No cache changes needed** — `NewsletterClassifier` only calls `complete(system, user, include_thinking=True)`, and the cache key `[model, temperature, max_tokens, extra_body, system_prompt, user_content]` already **separates prompt variants by content**. Two modes under `--mode all`:
- *Extraction:* `extract_stories(body)` → predicted list; store predicted + golden story sets per newsletter.
- *Quality+theme (fixed golden stories):* per reviewed, non-excluded `GoldenStory`, `assess_quality(title,text)` and `classify_themes(title,text)`; derive predicted tier via `compute_tier`.

  Concurrency via `asyncio.Semaphore(parallelism)`. Outputs under `evals/newsletter_results/`: timestamped `<ts>_<mode>_<tag>_<runid8>.jsonl` (`NewsletterRunMeta` first line, then `NewsletterPrediction` rows) + `.cot.jsonl` sidecar (non-empty only). **`NewsletterRunMeta` records a `prompt_hash`** = `sha256(json.dumps(config["newsletter"]["prompts"], sort_keys=True))[:16]` plus model/temperature/max_tokens/extra_body/tag/counts + the three system prompts verbatim — so prompt A/Bs are self-identifying and comparable. `--prompts alt.toml` deep-merges alternate `[newsletter.prompts.*]`; `--model` overrides the model. Cache reuses unchanged stages, cleanly separates changed ones.

**Report** — reuses `report.py` helpers with new class lists + a few new computations:
- *Tier (4-class):* accuracy + confusion matrix + P/R/F1 over `["excellent","good","fair","poor"]` (reuse existing helpers). Parse-failure rows (`scores is None`) counted as errors, excluded from the matrix.
- *Per-dimension:* MAE and exact-match (and within-1) agreement per dimension — new `compute_dimension_mae` / `compute_dimension_exact_match`.
- *Themes (multi-label):* per-theme P/R/F1 (each theme an independent binary label), micro-F1, macro-F1, exact-set-match — new `compute_multilabel_metrics`.
- *Extraction:* per newsletter, greedy one-to-one match of predicted→golden via `difflib.SequenceMatcher` ratio on normalized title-or-text, default threshold `0.6` (`--match-threshold`); precision=matched/predicted, recall=matched/golden; micro + macro aggregate — new `match_stories` (**most test-worthy function**).
- Two-run `--compare` reuses the Run A/Run B/Delta table (tier acc/F1, per-dimension MAE with decrease=improvement, theme micro/macro F1, extraction F1); header prints both `prompt_hash`+`tag`. `--verbose` lists per-story flips and per-newsletter extraction diffs. `--results-dir` trend table.

### Critical files to create
- `evals/newsletter_schemas.py`, `evals/newsletter_harvest.py`, `evals/newsletter_label.py`, `evals/newsletter_run.py`, `evals/newsletter_report.py`
- Tests: `tests/test_eval_newsletter_{schemas,harvest,label,run,report,cli_docs}.py`

### Docs to update
- `evals/README.md` — human "Newsletter evaluation" section (4 stages, fixed-golden-stories concept, workflows: build golden set / iterate a prompt variant / compare two runs).
- `evals/README-technical.md` — `### newsletter_harvest|label|run|report` sections listing **every** backticked CLI flag (required by the CLI-docs sync test); note the shared cache + content-hash prompt separation + `prompt_hash`.
- `CLAUDE.md` — "Newsletter Classification" gets a short "Newsletter evaluation" subsection clarifying the distinction from read-only `newsletter_review/`; add new test files to the Testing list.

## TDD plan (strict red/green per CLAUDE.md — failing test first, watch it fail, then minimal code)

Write tests first, one behavior at a time:
- **schemas:** round-trip + defaults for `GoldenStory`/`GoldenNewsletter` (nested stories survive `json.dumps`); `NewsletterRunMeta` round-trip incl. `prompt_hash`/model; backward-compat missing keys.
- **harvest:** `is_newsletter` filter accepts/rejects; harvested newsletter has empty `stories` + `reviewed=False`; `body` equals `format_thread_transcript` output; dedup drops seen `thread_id` and tolerates malformed lines.
- **label:** add-story appends with stable `story_id`; confirm sets `newsletter.reviewed`; scoring 4 dims + themes sets `story.reviewed` and auto-derives tier; undo restores snapshot; exclude skips in queue selector. (Test the pure functions, not curses.)
- **report:** `match_stories` (exact, fuzzy≥threshold, below-threshold no-match, one-to-one greedy, extra→precision drop, missing→recall drop); tier confusion + P/R/F1; per-dimension MAE + exact-match; theme per-theme/micro/macro F1 + exact-set-match; parse-failure counted as error; comparison delta sign correct.
- **run:** `prompt_hash` changes iff a prompt string changes (mutation-style); second identical run records cache hits (fake inner client); extraction vs quality/theme modes record the right fields; `--prompts` override changes `prompt_hash`.
- **cli_docs:** mirror `tests/test_eval_cli_docs.py` — assert every new `--flag` appears in `README-technical.md`.

Run the full suite (`uv run --extra dev pytest tests/ -v`) before declaring done.

## Verification (end-to-end)

1. `uv run --extra dev pytest tests/ -v` — all green.
2. **Harvest** a few newsletters into `evals/newsletter_golden_set.jsonl` (against the api-proxy).
3. **Label** them — curate stories, assign scores + themes for a small golden set.
4. **Run** `python -m evals.newsletter_run --mode all --tag baseline`, then `--report`.
5. **Iterate:** edit a `[newsletter.prompts.*]` block (or `--prompts alt.toml`), re-run with `--tag variant`; confirm the cache reuses unchanged stages and `prompt_hash` differs.
6. **Compare:** `python -m evals.newsletter_report --compare <baseline>.jsonl <variant>.jsonl --verbose` — tier/dimension/theme/extraction deltas render, prompt hashes shown.

## Notes / defaults chosen (revisitable in implementation)
- **Tier is derived** from dimension scores via `compute_tier`, not hand-labeled separately (avoids conflict; add `--label-tier` later if holistic judgment is wanted).
- **`excluded` story** = "keep as extraction truth but skip quality/theme scoring"; a *wrongly extracted* candidate is **deleted** in Phase A, not excluded.
- **Extraction matcher**: `SequenceMatcher` ratio ≥ 0.6, one-to-one greedy, `--match-threshold` configurable; `--verbose` shows unmatched pairs to calibrate. Merges/splits are a known imperfection surfaced in verbose output.
- Small hand-labeled sets make 1–5 MAE noisy — report within-1 agreement alongside exact-match, keep `golden_set_count` in the header.
- **Risk #7 — extraction-recall bias from seeding.** Phase A seeds boundaries from the production `parse_stories`, so the extraction ground truth is influenced by the system under test and recall may read slightly high. Mitigations: the reviewer must actively confirm/split/merge every boundary (rubber-stamping is the failure mode to watch); record the seed source (`seeded_from: "parse_stories"`) in the golden set and surface it in the report header so results aren't read as fully-independent. If bias proves material, switch specific newsletters to a stronger seed model or hand-label them (the schema is identical either way).
