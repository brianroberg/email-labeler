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

### newsletter_harvest

| Flag | Description |
|---|---|
| `--output` | Output golden-set JSONL path (default: `evals/newsletter_golden_set.jsonl`) |
| `--max-threads` | Max threads to fetch (default: `50`) |
| `--recipient` | Newsletter recipient to query (`to:<recipient>`); default from `config["newsletter"]["recipient"]` |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

Harvest guards each candidate with `is_newsletter(...)`, builds `body` via
`daemon.format_thread_transcript(...)` (identical to production input), and seeds
each `GoldenNewsletter` with `stories=[]` and `reviewed=False` — **no ground truth
is inferred**. It always appends, deduplicating by `thread_id`, and tolerates
malformed existing lines. To rebuild from scratch, delete the file manually.

### newsletter_label

| Flag | Description |
|---|---|
| `--golden-set` | Path to newsletter golden set JSONL (default: `evals/newsletter_golden_set.jsonl`) |
| `--edit` | Curses TUI for editing already-reviewed newsletters |
| `--unreviewed-only` | Show only newsletters not yet reviewed |
| `--config` | Path to config.toml (default: `./config.toml`) |

Two phases per newsletter. **Phase A** curates the story list (extraction truth):
the list is seeded once by running the production `parse_stories` extractor (a
cached LLM call) as a deletable starting point, then the reviewer builds the
authoritative list by marking body segments — move the cursor over the rendered
body, press `s`/`e` to set the selection start/end line and `Enter` to make a
story from that inclusive span — plus `a`dd/`E`dit/`d`elete. Confirming sets
`newsletter.reviewed=True` and assigns each story a stable `story_id`. Pressing
`k` skips the newsletter without marking it reviewed, so it resurfaces in a later
pass (and the list view jumps straight to the next queued newsletter).
**Phase B** labels each story: the 4 dimensions (simple,
concrete, personal, dynamic; `1`–`5`), multi-select themes, notes, undo;
`expected_tier` is auto-derived via `compute_tier` on save and `story.reviewed`
is set. Saves are atomic (temp-file + rename). `excluded` stories are kept as
extraction truth but skipped from quality/theme scoring.

### newsletter_run

| Flag | Description |
|---|---|
| `--golden-set` | Path to newsletter golden set JSONL (default: `evals/newsletter_golden_set.jsonl`) |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--output-dir` | Output directory for results (default: `evals/newsletter_results/`) |
| `--mode` | Which outputs to evaluate: `extraction`, `quality`, `themes`, or `all` (default: `all`) |
| `--tag` | Tag for the results filename (e.g. `baseline`) |
| `--no-cache` | Disable the LLM response cache (default: cache enabled) |
| `--parallelism` | Concurrent evaluations (default: `1`) |
| `--include-unreviewed` | Also evaluate newsletters not yet reviewed (default: reviewed only) |
| `--prompts` | Path to a TOML file whose `[newsletter.prompts.*]` blocks are deep-merged over the base config (changes `prompt_hash`) |
| `--model` | Override the newsletter LLM model name from config |
| `--report` | Print a metrics report for this run after it completes |
| `--compare-to` | After the run, print a comparison against this prior results JSONL file |
| `--skip-preflight` | Skip the endpoint reachability check before evaluating |

Run replays the golden set through the real `NewsletterClassifier` using the
`[newsletter.llm]` endpoint (resolved via `resolve_newsletter_llm_endpoint()`).
Extraction mode feeds each `body` through `extract_stories`; quality/theme mode
scores the **fixed golden `(title, text)`** of every reviewed, non-excluded story
(independent of what extraction produced) and derives the predicted tier via
`compute_tier`. Outputs are timestamped `<ts>_<mode>_<tag>_<runid8>.jsonl`
(`NewsletterRunMeta` first line, then `StoryPrediction` rows in quality/theme
mode and `ExtractionPrediction` rows in extraction mode) plus a non-empty
`.cot.jsonl` sidecar of `NewsletterThinkingEntry` rows.

### newsletter_report

| Flag | Description |
|---|---|
| `--results` | Path to a single results JSONL file |
| `--compare` | Two result file paths for side-by-side comparison |
| `--results-dir` | Directory of results for trend view |
| `--verbose` | Show per-story flips and per-newsletter extraction diffs |
| `--format` | `table` (default) or `json` |
| `--match-threshold` | `SequenceMatcher` ratio threshold for the extraction one-to-one match (default: `0.6`) |

Report computes tier accuracy + confusion matrix + P/R/F1 (excellent/good/fair/
poor), per-dimension MAE and exact/within-1 agreement, per-theme + micro/macro
multi-label F1 + exact-set-match, and extraction precision/recall/F1 from a greedy
one-to-one `SequenceMatcher` match. `--compare` prints a Run A/Run B/Delta table
and shows each run's `prompt_hash` + `tag` in the header; `--results-dir` renders a
trend table. Parse-failure rows (`scores is None`) are counted as errors and
excluded from the tier matrix.

### Newsletter shared cache & prompt separation

The newsletter runner reuses the **same** disk cache as the email eval —
`evals/cache/llm_cache.jsonl` via `CachedLLMClient` — with no cache changes. The
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
- Cache hits return instantly without contacting the LLM.
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
