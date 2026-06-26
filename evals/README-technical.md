# Evaluation Suite — Technical Reference

Complete CLI flag references, cache internals, and chain-of-thought capture details for the evaluation suite. For an overview and common workflows, see [README.md](README.md).

## CLI Reference

### harvest

| Flag | Description |
|---|---|
| `--output` | Output JSONL path (default: `evals/golden_set.jsonl`) |
| `--max-threads` | Max threads to fetch (default: `200`) |
| `--append` | Append to existing file, deduplicating by thread ID |
| `--sender-type` | Filter: `person` or `service` |
| `--label` | Filter: `needs_response`, `fyi`, `low_priority`. ANDed into the Gmail query (e.g. `label:agent/processed label:agent/needs-response`) so the fetch returns a dense pool of matching threads — useful for boosting a rare class like `needs_response` in the golden set |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

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
