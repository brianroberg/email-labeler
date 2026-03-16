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
| `--label` | Filter: `needs_response`, `fyi`, `low_priority` |
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
