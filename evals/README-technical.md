# Evaluation Suite — Technical Reference

Complete CLI flag references, cache internals, and chain-of-thought capture details for the evaluation suite. For an overview and common workflows, see [README.md](README.md).

## CLI Reference

### harvest

| Flag | Description |
|---|---|
| `--output` | Output JSONL path (default: `evals/golden_set.jsonl`) |
| `--max-threads` | Max threads to fetch (default: `200`) |
| `--sender-type` | Filter: `person` or `service` |
| `--label` | Filter by config **key**: `needs_response`, `fyi`, `low_priority` (not the Gmail label name — the key is mapped via `[labels]` in config.toml, e.g. `needs_response` → `agent/needs-response`). ANDed into the Gmail query (e.g. `label:agent/processed label:agent/needs-response`) so the fetch returns a dense pool of matching threads — useful for boosting a rare class like `needs_response` in the golden set |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

Harvest always appends to `--output`, deduplicating by thread ID. There is no overwrite mode: the golden set also stores manual review state (confirmed labels, exclusions, notes), so harvest never truncates it. To rebuild from scratch, delete the file manually.

### review

| Flag | Description |
|---|---|
| `--golden-set` | Path to golden set JSONL (default: `evals/golden_set.jsonl`) |
| `--show-labels` | Show existing labels (default is blind mode) |
| `--edit` | Curses TUI for editing reviewed threads (auto-saves on each change) |
| `--stage` | Review only stage 1 (sender) or stage 2 (label) |
| `--unreviewed-only` | Show only threads not yet reviewed |
| `--filter-label` | Show only threads with this label |
| `--start-at` | Start at thread index (0-based) |

Review hotkeys: `p`/`s` sender (person/service); `r`/`f`/`l` label (needs_response/fyi/low_priority); `n` notes; `z` undo; `k` skip; `e` exclude; `q` quit. **Skip** (`k`) leaves the thread unreviewed so it reappears later. **Exclude** (`e`) sets `excluded=True` (also marks reviewed): excluded threads are dropped from the review queue here and from `run_eval` entirely. Un-exclude from the `--edit` TUI detail view with `e`. The `excluded` field is persisted in the golden set JSONL; legacy records using the old `skipped` key are still read as excluded.

### run_eval

| Flag | Description |
|---|---|
| `--golden-set` | Path to golden set JSONL (default: `evals/golden_set.jsonl`) |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--output-dir` | Output directory for results (default: `evals/results/`) |
| `--stages` | `full`, `stage1_only`, or `stage2_only` (default: `full`) |
| `--parallelism` | Concurrent evaluations (default: `cloud_parallel` from config) |
| `--include-unreviewed` | Also evaluate threads not yet reviewed (default: reviewed only) |
| `--dry-run` | Show what would be evaluated without calling LLMs |
| `--tag` | Tag for the results filename (e.g. `new-prompts`) |
| `--no-cache` | Disable LLM response cache (default: cache enabled) |
| `--sender-type` | Only evaluate threads with this expected sender type (`person` or `service`) |
| `--max-threads` | Max threads to evaluate (default: all) |
| `--local-only` | Shortcut for `--stages stage2_only --sender-type person` — evaluate only the local classifier (no cloud creds needed). Errors if combined with a conflicting `--stages`/`--sender-type` |
| `--skip-preflight` | Skip the endpoint reachability check that runs before evaluation |
| `--preflight-timeout` | Seconds to wait for the pre-run endpoint check, overriding both probes (default: each endpoint's own request timeout — generous for a local cold-load, fast-failing for the cloud). Raise it for a server that cold-loads a large model on demand |
| `--report` | Print a metrics report (accuracy, confusion matrix, per-class P/R/F1) for this run after it completes |
| `--compare-to` | After the run, print a side-by-side comparison against this prior results JSONL file |
| `--cloud-model` | Override cloud LLM model name from config |
| `--local-model` | Override local LLM model name from config |
| `--cloud-temperature` | Override cloud LLM temperature (alias: `--cloud-temp`) |
| `--local-temperature` | Override local LLM temperature (alias: `--local-temp`) |
| `--cloud-max-tokens` | Override cloud LLM max tokens from config |
| `--local-max-tokens` | Override local LLM max tokens from config |
| `--cloud-timeout` | Override cloud LLM request timeout in seconds (for slow prefills on long inputs) |
| `--local-timeout` | Override local LLM request timeout in seconds (for slow prefills on long inputs) |
| `--cloud-extra-body` | JSON object merged into every cloud LLM request body (e.g. `'{"top_p": 0.9}'`) |
| `--local-extra-body` | JSON object merged into every local LLM request body (e.g. `'{"chat_template_kwargs": {"enable_thinking": false}}'`) |
| `--cloud-no-think` | Disable thinking for the cloud LLM (shortcut for `--cloud-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`) |
| `--local-no-think` | Disable thinking for the local LLM (shortcut for `--local-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`) |

#### A/B testing thinking on vs. off

The local model (person-body label classification) is a reasoning model by
default. To measure the accuracy impact of disabling thinking before changing
`config.toml`, run two evaluations and compare:

```bash
# Baseline (thinking on, from config) — person threads only
uv run python -m evals.run_eval --stages stage2_only --sender-type person --tag think-on

# Thinking off
uv run python -m evals.run_eval --stages stage2_only --sender-type person \
    --local-no-think --tag think-off

uv run python -m evals.report --compare evals/results/<think-on>.jsonl evals/results/<think-off>.jsonl
```

Because `extra_body` is part of the cache key, the two runs are cached
independently — no `--no-cache` needed. The exact disable payload is
server-specific; `--local-no-think` uses the `chat_template_kwargs` form
(Qwen3 / LM Studio). If your server expects a top-level flag instead, pass it
explicitly, e.g. `--local-extra-body '{"enable_thinking": false}'`.

### report

| Flag | Description |
|---|---|
| `--results` | Path to a single results JSONL file |
| `--compare` | Two result file paths for side-by-side comparison |
| `--results-dir` | Directory of results for trend view |
| `--verbose` | Show per-thread disagreements |
| `--format` | `table` (default) or `json` |

### run_web

| Flag | Description |
|---|---|
| `--host` | Host to bind to (default: `127.0.0.1`) |
| `--port` | Port to bind to (default: `5000`) |

## Newsletter Evaluation CLI Reference

The newsletter eval harness mirrors the email eval's 4-stage shape
(`newsletter_harvest → newsletter_label → newsletter_run → newsletter_report`) but
scores the newsletter grading pipeline (story extraction, per-story quality
scores/tier, and themes). Its modules live under `evals/` prefixed `newsletter_`
to avoid colliding with the top-level read-only `newsletter_review/` package.

**Path defaults are CWD-relative** (except `--config`): the default `--output` /
`--golden-set` / `--output-dir` strings (`evals/newsletter_golden_set.jsonl`,
`evals/newsletter_results/`) are resolved against the current working directory,
so the four tools only chain on their defaults when run from the repo root. The
one exception is `--config`, whose default is the **repo-root** `config.toml`
regardless of CWD.

**Environment variables.** The tools run outside Docker and read the same env as
the daemon (see the `.env` symlink note in [README.md](README.md)):

| Variable | Used by | Purpose |
|---|---|---|
| `NEWSLETTER_LLM_URL` / `NEWSLETTER_LLM_API_KEY` | `newsletter_label` (Phase-A seeding), `newsletter_run` | The `[newsletter.llm]` endpoint. The override is atomic: once the URL is set, the key comes only from `NEWSLETTER_LLM_API_KEY` |
| `CLOUD_LLM_URL` / `CLOUD_LLM_API_KEY` | same | Fallback endpoint when `NEWSLETTER_LLM_URL` is unset |
| `PROXY_URL` / `PROXY_API_KEY` | `newsletter_harvest` | Gmail api-proxy address + auth. `PROXY_API_KEY` must be non-empty; harvest exits with a one-line error when it is missing |

`newsletter_label` (without `--edit`) and `newsletter_run` fail fast before doing
any work when no LLM endpoint is configured, and their error messages name the
resolved URL and which env var supplied it.

### newsletter_harvest

| Flag | Description |
|---|---|
| `--output` | Output golden-set JSONL path (default: `evals/newsletter_golden_set.jsonl`, CWD-relative — run from the repo root so it chains with the other tools) |
| `--max-threads` | Max threads to fetch (default: `50`) |
| `--recipient` | Newsletter recipient to query (`to:<recipient>`); default from `config["newsletter"]["recipient"]` |
| `--config` | Path to config.toml (default: the repo-root `config.toml`, regardless of CWD) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

Harvest guards each candidate with `is_newsletter(...)`, builds `body` via
`daemon.format_thread_transcript(...)` (identical to production input), and seeds
each `GoldenNewsletter` with `stories=[]` and `reviewed=False` — **no ground truth
is inferred**. It always appends, deduplicating by `thread_id`, and tolerates
malformed existing lines. To rebuild from scratch, delete the file manually.

A successful run ends with a next-step hint pointing at
`uv run python -m evals.newsletter_label --golden-set <output>`. Per-request
httpx/httpcore INFO logs are silenced (public helper
`evals.newsletter_harvest.quiet_http_logging()`), and a missing `PROXY_API_KEY`
exits 1 with a one-line message instead of a traceback.

### newsletter_label

| Flag | Description |
|---|---|
| `--golden-set` | Path to newsletter golden set JSONL (default: `evals/newsletter_golden_set.jsonl`) |
| `--edit` | Disable LLM seeding — manual curation only. Same curses TUI, but `Space` is inert and no LLM endpoint is needed |
| `--unreviewed-only` | Show only newsletters not yet reviewed |
| `--config` | Path to config.toml (default: the repo-root `config.toml`, regardless of CWD) |

Both modes open the same curses TUI; `--edit` only removes the Phase-A LLM
seeding (it does **not** filter to reviewed newsletters — combine with
`--unreviewed-only` or not as needed). Without `--edit`, a missing
`NEWSLETTER_LLM_URL`/`CLOUD_LLM_URL` exits immediately at startup with an
actionable message instead of failing later inside the TUI when `Space` is
pressed.

Two phases per newsletter. **Phase A** curates the story list (extraction
truth). Press `Space` to seed candidate stories by running the production
`parse_stories` extractor over a **fresh, uncached** LLM extraction of the body
(every press re-runs the call). Re-seeding over a non-empty list asks
`Replace N stories (M labeled) with a fresh seed? y/N`; a `Seeding…` indicator
shows during the call, and the outcome is reported on the status line
(`Seeded N stories.` / `Extractor returned NO_STORIES.` / `Extractor output had
no parseable story blocks.`). Under `--edit`, `Space` shows an explanatory
"seeding disabled" message. The seed is a deletable starting point; the reviewer
builds the authoritative list by marking body segments — move the cursor over the
rendered body (body lines already covered by a story are **dimmed**, so omitted
paragraphs stand out), press `s`/`e` to set the selection start/end line and
`Enter` to make a story from that inclusive span — plus `a`dd/`E`dit/`d`elete.
Deleting a story that carries labels asks `y/N` confirmation, and deletes echo
what was removed. Confirming (`c`) sets `newsletter.reviewed=True`, assigns each
story a stable `story_id`, and warns how many stories are still unlabeled.
Pressing `k` skips the newsletter without marking it reviewed, so it resurfaces
in a later pass (and the list view jumps straight to the next queued newsletter).
**Phase B** labels each story: the 4 dimensions (simple, concrete, personal,
dynamic; `1`–`5`), multi-select themes, notes, undo; `expected_tier` is
auto-derived via `compute_tier` on save and `story.reviewed` is set. Saves are
atomic (temp-file + rename). `excluded` stories are kept as extraction truth but
skipped from quality/theme scoring.

TUI details:

- **Undo** (`z`) is a multi-level stack (up to 100 snapshots per newsletter). A
  snapshot is pushed only when a mutation actually happens — cancelled prompts
  never consume an undo level.
- **Esc** cancels any prompt (title/text/notes/story #), clears an active
  `s`/`e` selection before acting as "back", and `ESCDELAY` defaults to 25 ms so
  it feels instant.
- **Prompts prefill current values**: `E` prefills the title and (single-line)
  text with blank=keep semantics (multi-line text remains blank=keep — body-span
  selection is the intended paragraph-level repair path); `n` prefills existing
  notes; `l` shows current scores ("now X") and prefills current themes. Prompt
  input scrolls horizontally (no width truncation) and rejects control
  characters.
- **Detail view**: labeled stories show `scores: a/b/c/d`; a `row X/Y` position
  indicator sits on the status line; the help footer wraps on narrow terminals
  and lists `Space:seed` and `PgUp/PgDn`.
- **List view**: columns are `R Lbl Sender Subject` where `Lbl` is
  labeled/total stories; the list supports `PgUp`/`PgDn`.

`seed_from_extractor()` returns `(stories, raw)` — the raw extractor output is
kept so the caller can distinguish a `NO_STORIES` verdict from unparseable
output.

### newsletter_run

| Flag | Description |
|---|---|
| `--golden-set` | Path to newsletter golden set JSONL (default: `evals/newsletter_golden_set.jsonl`) |
| `--config` | Path to config.toml (default: the repo-root `config.toml`, regardless of CWD) |
| `--output-dir` | Output directory for results (default: `evals/newsletter_results/`) |
| `--mode` | Which outputs to evaluate: `extraction`, `quality`, `themes`, or `all` (default: `all`). `quality` and `themes` fire **only their own** LLM calls; the skipped side's fields are `null` in the results rows |
| `--tag` | Tag for the results filename (e.g. `baseline`) |
| `--no-cache` | Disable the LLM response cache (default: cache enabled) |
| `--parallelism` | Concurrent evaluations (default: `1`) |
| `--include-unreviewed` | Also evaluate newsletters not yet reviewed (default: reviewed only) |
| `--prompts` | Path to a TOML file whose `[newsletter.prompts.*]` blocks are deep-merged over the base config (changes `prompt_hash`) |
| `--model` | Override the newsletter LLM model name from config |
| `--report` | Print a metrics report for this run after it completes |
| `--compare-to` | After the run, print a comparison against this prior results JSONL file. No longer always-verbose: per-story detail follows `--verbose` |
| `--verbose` | With `--report`/`--compare-to`: per-story diffs, parse-failure raws, extraction detail |
| `--match-threshold` | With `--report`/`--compare-to`: `SequenceMatcher` ratio threshold forwarded to the embedded report/comparison (default: `0.6`) |
| `--skip-preflight` | Skip the endpoint reachability check before evaluating |

Run replays the golden set through the real `NewsletterClassifier` using the
`[newsletter.llm]` endpoint (resolved via `resolve_newsletter_llm_endpoint()`).
Extraction mode feeds each `body` through `extract_stories`; quality/theme mode
scores the **fixed golden `(title, text)`** of every reviewed, non-excluded story
(independent of what extraction produced) and derives the predicted tier via
`compute_tier`. Story modes skip stories with `reviewed=False` (Phase-A-confirmed
but never Phase-B-labeled) and print a skip count; `meta.story_count` counts only
labeled, non-excluded stories. In a results row, `scores_raw`/`themes_raw = null`
means that call was **never attempted** (mode skipped it), while a captured raw
string with a `null` parsed field means **attempted but unparsed**. Outputs are
timestamped `<ts>_<mode>_<tag>_<runid8>.jsonl` (`NewsletterRunMeta` first line,
then `StoryPrediction` rows in quality/theme mode and `ExtractionPrediction` rows
in extraction mode) plus a `.cot.jsonl` sidecar of `NewsletterThinkingEntry` rows
(written only when at least one entry captured thinking). The sidecar holds two
row shapes: story-level rows (`story_id` + `quality_cot`/`theme_cot`) and
newsletter-level extraction rows (`thread_id` + `extraction_cot`) capturing the
extractor's segmentation reasoning per newsletter — `story_id` is optional in the
schema.

Run output: the golden-set load prints a kept/unreviewed/excluded breakdown (with
actionable hints when nothing is evaluable); `[k/N]` progress lines print during
evaluation; the end-of-run summary line is
`Rows: N (X errors, Y quality parse failures, Z theme parse failures)` plus
optional clamped-score and dropped-theme-token lines; a
`Chain-of-thought written to <path>.cot.jsonl` line appears when the sidecar is
written; httpx request logging is quieted to WARNING. Missing/malformed config,
`--prompts`, and golden-set files exit 1 with one-line errors, and preflight and
per-row LLM errors name the resolved endpoint URL and its source env var.

### newsletter_report

| Flag | Description |
|---|---|
| `--results` | Path to a single results JSONL file |
| `--compare` | Two result file paths for side-by-side comparison. `.cot.jsonl` sidecars matched by a glob are ignored (with a stderr note), so `*<tag>*.jsonl` globs work |
| `--results-dir` | Directory of results for trend view (`.cot.jsonl` sidecars skipped) |
| `--verbose` | Show per-story flips and per-newsletter extraction diffs (no effect with `--results-dir`; a stderr note says so) |
| `--format` | `table` (default) or `json`. JSON works in all three modes: single run (`{meta, metrics}`), `--compare` (`{"run_a": {meta, metrics}, "run_b": {...}}`), and `--results-dir` (array of summary rows: file, run_id, timestamp, mode, tag, prompt_hash, tier_accuracy, theme_micro_f1, mae_simple, extraction_micro_f1 — `null` for absent sections) |
| `--match-threshold` | `SequenceMatcher` ratio threshold for the extraction one-to-one match (default: `0.6`). Applies to **text** similarity (see below) |

Report computes tier accuracy + confusion matrix + P/R/F1 (excellent/good/fair/
poor), per-dimension MAE and exact/within-1 agreement, per-theme + micro/macro
multi-label F1 + exact-set-match, and extraction precision/recall/F1 from a greedy
one-to-one `SequenceMatcher` match.

Metric semantics:

- **Extraction matching compares story text** (whitespace-collapsed, lowercased),
  falling back to title only when a story has no text — golden text is the
  curated ground truth, titles are model-invented wording. `--match-threshold`
  applies to that text similarity.
- A newsletter with 0 predicted and 0 golden stories scores
  precision = recall = 1.0 (correct abstention), not 0.0.
- **Theme metrics** include a story iff its own theme call succeeded (row has no
  error and `themes_raw` is present, or — for legacy rows without `themes_raw` —
  a non-empty parsed prediction). A quality-parse failure does not drop a story
  from theme P/R/F1 or exact-set match.
- **Tier metrics** count a row as a parse error only when quality was actually
  attempted (an error, or `scores_raw` captured) and `predicted_scores` is
  `None`. Rows where quality was never attempted (`--mode themes`:
  `error=None, scores_raw=None`) are skipped entirely — a themes-only results
  file does not render a tier section full of fake errors.
- Aggregates (theme micro/macro F1, exact-set match, extraction micro/macro
  P/R/F1) are `None`/`N/A` when a run has zero scored rows, never a fake `0.0%`.
  Single-run reports omit the Quality Dimensions and Themes sections entirely
  when the run contains no story predictions (mirroring the tier section); JSON
  output serializes these as `null`.

Single-run report output: the header shows `Golden set:`; the tier section
pluralizes correctly and lists `Failed stories (quality parse/network): <ids>`;
the themes section shows a `Parse anomalies: N` line when theme responses
contained invalid tokens or unparseable output; the extraction section shows
`(<N> newsletters, match threshold <t>)`.

`--verbose` detail: Story Disagreements lines append story titles (best-effort
lookup from the run's recorded `golden_set_path`; degrades to bare ids if the
file moved) and include stories whose quality parse failed but whose themes
disagree; a `Parse/Network Failures` section prints each failed story with its
raw quality response (`scores_raw`) or error; a `Theme Parse Anomalies` section
prints `themes_raw` for stories whose theme output was invalid
(`invalid tokens dropped: ...`) or unparseable (`parsed to []`); Extraction Diffs
honor `--match-threshold` (header says `threshold <t>`) and, for any newsletter
that isn't a perfect match, list matched pairs with their `SequenceMatcher` ratio
plus each unmatched predicted / unmatched golden story by title (text excerpt
when untitled).

`--compare` prints a Run A/Run B/Delta table; the header shows each run's
`prompt_hash` + `tag` + `mode=...`, and a WARNING line prints when the two runs'
modes differ. Sections absent from one run render `N/A`, never `-100%` deltas.
`--results-dir` renders a trend table with Run ID, Timestamp, Mode, Tag, Prompt
(`prompt_hash[:8]`), TierAcc, ThemeF1, MAE(simple), and ExtractF1 columns; rows
are sorted chronologically by `run_meta.timestamp` (not filename), and the Tag
column shows `-` when a run has no tag.

Nonexistent or meta-less results files produce a one-line `Error: ...` on stderr
and exit code 1 (all three modes) — and when the offending path is a
`.cot.jsonl`, a hint points at the main results file. Public helpers:
`match_stories_detailed()`, `theme_parse_anomalies()`, `load_story_titles()`,
`build_trend_rows()`, `comparison_as_json()`.

### Newsletter shared cache & prompt separation

The newsletter runner reuses the **same** disk cache as the email eval —
`evals/cache/llm_cache.jsonl` via `CachedLLMClient`. The
cache key `[model, temperature, max_tokens, extra_body, system_prompt,
user_content]` already **separates prompt variants by content**: editing a
`[newsletter.prompts.*]` string changes the `system_prompt` (and thus the key), so
A/B runs are cached independently and unchanged stages are reused. Each run's
`NewsletterRunMeta` records a **`prompt_hash`** =
`sha256(json.dumps(config["newsletter"]["prompts"], sort_keys=True))[:16]` plus the
model/temperature/max_tokens/extra_body/tag/counts and the three system prompts
verbatim, so prompt A/Bs are self-identifying and comparable. `--prompts` (and
`--model`) are applied **before** `prompt_hash` is computed, so an override yields a
distinct hash.

## LLM Response Cache

The eval suite includes a disk-backed LLM response cache that avoids redundant LLM calls across repeated evaluation runs. This is especially useful during prompt A/B testing or model swaps — once a thread has been evaluated with a given configuration, re-running produces instant results from cache.

**Location:** `evals/cache/llm_cache.jsonl`

**Cache key:** A SHA-256 hash of a JSON array containing these fields (from `llm_cache.py`):

```
[model, temperature, max_tokens, extra_body, system_prompt, user_content]
```

Changing any of these — the model name, temperature, inference parameters, `extra_body` config, or the prompt content — produces a different cache key, ensuring stale hits don't occur across configurations.

**Behavior:**

- On startup, the entire cache file is loaded into memory.
- Cache hits return instantly without contacting the LLM. An entry whose stored
  `thinking` is `""` means the model was called and emitted no `<think>` block —
  it is a normal hit, not re-fetched.
- Legacy entries written before thinking capture (no `thinking` key at all) are
  backfilled with one LLM call the first time chain-of-thought is requested for
  them, then cached permanently.
- Cache misses call the real LLM, then store the response in memory.
- New entries are appended to disk when `flush()` is called (at the end of each run).
- Hit/miss statistics are printed at the end of every evaluation run.

**Disabling the cache:**

Pass `--no-cache` to `run_eval` to bypass the cache entirely and call the LLM for every thread:

```bash
uv run python -m evals.run_eval --no-cache
```

## Chain-of-Thought Capture

When running evaluations with the LLM cache enabled (the default), chain-of-thought content from `<think>...</think>` blocks is automatically captured and stored in sidecar files alongside results.

**Sidecar format:** `evals/results/<run>.cot.jsonl` — one JSON line per thread with `stage1_thinking` and `stage2_thinking` fields.

Chain-of-thought is viewable in the web UI on the run detail page.
