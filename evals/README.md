# Evaluation Suite

The `evals/` directory provides a 4-stage pipeline for measuring classification accuracy:

```
harvest → review → run_eval → report
```

The eval tools run outside Docker and need access to the same environment variables as the daemon. If you haven't already, symlink to `agent-stack/.env`:

```bash
ln -s ../agent-stack/.env .env
```

Since the symlinked `.env` may contain Docker-internal hostnames (e.g. `PROXY_URL=http://api-proxy:8000`), use `--proxy-url` to point at the proxy's host-accessible address:

```bash
uv run python -m evals.harvest --proxy-url http://localhost:8000 --max-threads 200
```

### 1. Harvest — Build a golden set from production data

Pulls threads already labeled by the daemon, infers ground truth from their Gmail labels, and exports to JSONL.

```bash
# Harvest up to 200 processed threads
uv run python -m evals.harvest --proxy-url http://localhost:8000 --max-threads 200

# Append new threads (deduplicates automatically)
uv run python -m evals.harvest --proxy-url http://localhost:8000 --append

# Filter by sender type or label
uv run python -m evals.harvest --proxy-url http://localhost:8000 --sender-type person
uv run python -m evals.harvest --proxy-url http://localhost:8000 --label needs_response
```

| Flag | Description |
|---|---|
| `--output` | Output JSONL path (default: `evals/golden_set.jsonl`) |
| `--max-threads` | Max threads to fetch (default: `200`) |
| `--append` | Append to existing file, deduplicating by thread ID |
| `--sender-type` | Filter: `person` or `service` |
| `--label` | Filter: `needs_response`, `fyi`, `low_priority`, `unwanted` |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--proxy-url` | API proxy URL (overrides `PROXY_URL` env var) |

### 2. Review — Manually verify ground truth labels

Interactive CLI for reviewing and correcting labels in the golden set. Saves atomically after each session.

```bash
# Review all threads (blind mode by default)
uv run python -m evals.review

# Show existing labels while reviewing
uv run python -m evals.review --show-labels

# Review only sender classification (stage 1)
uv run python -m evals.review --stage 1

# Review only label classification (stage 2)
uv run python -m evals.review --stage 2

# Only review unreviewed threads
uv run python -m evals.review --unreviewed-only

# Filter to a specific label
uv run python -m evals.review --filter-label needs_response

# Resume from thread index 5
uv run python -m evals.review --start-at 5
```

| Flag | Description |
|---|---|
| `--golden-set` | Path to golden set JSONL (default: `evals/golden_set.jsonl`) |
| `--show-labels` | Show existing labels (default is blind mode) |
| `--stage` | Review only stage 1 (sender) or stage 2 (label) |
| `--unreviewed-only` | Show only threads not yet reviewed |
| `--filter-label` | Show only threads with this label |
| `--start-at` | Start at thread index (0-based) |

### 3. Run — Replay golden set through the classifier

Sends each golden thread through the real `EmailClassifier` with live LLM endpoints. Results are written to timestamped JSONL files in `evals/results/`.

```bash
# Full evaluation (both stages)
uv run python -m evals.run_eval

# Evaluate only Stage 1 (sender classification)
uv run python -m evals.run_eval --stages stage1_only

# Evaluate only Stage 2 (uses expected sender type as input)
uv run python -m evals.run_eval --stages stage2_only

# Use alternate config and tag the run
uv run python -m evals.run_eval --config config_v2.toml --tag new-prompts

# Override a single setting without editing config.toml
uv run python -m evals.run_eval --local-model qwen/qwen3-14b --tag qwen3-14b

# Include unreviewed threads (default is reviewed-only)
uv run python -m evals.run_eval --include-unreviewed

# Dry run — show what would be evaluated
uv run python -m evals.run_eval --dry-run
```

| Flag | Description |
|---|---|
| `--golden-set` | Path to golden set JSONL (default: `evals/golden_set.jsonl`) |
| `--config` | Path to config.toml (default: `./config.toml`) |
| `--output-dir` | Output directory for results (default: `evals/results/`) |
| `--stages` | `full`, `stage1_only`, or `stage2_only` (default: `full`) |
| `--parallelism` | Concurrent evaluations (default: `3`) |
| `--include-unreviewed` | Also evaluate threads not yet reviewed (default: reviewed only) |
| `--dry-run` | Show what would be evaluated without calling LLMs |
| `--tag` | Tag for the results filename (e.g. `new-prompts`) |
| `--no-cache` | Disable LLM response cache (default: cache enabled) |
| `--sender-type` | Only evaluate threads with this expected sender type (`person` or `service`) |
| `--cloud-model` | Override cloud LLM model name from config |
| `--local-model` | Override local LLM model name from config |
| `--cloud-temperature` | Override cloud LLM temperature (alias: `--cloud-temp`) |
| `--local-temperature` | Override local LLM temperature (alias: `--local-temp`) |
| `--cloud-max-tokens` | Override cloud LLM max tokens from config |
| `--local-max-tokens` | Override local LLM max tokens from config |

### 4. Report — Compute metrics and compare runs

Generates accuracy, confusion matrix, per-class precision/recall/F1, and privacy violation reports.

```bash
# Single run report
uv run python -m evals.report --results evals/results/run.jsonl

# Verbose — show per-thread disagreements
uv run python -m evals.report --results evals/results/run.jsonl --verbose

# JSON output for programmatic use
uv run python -m evals.report --results evals/results/run.jsonl --format json

# Compare two runs side by side
uv run python -m evals.report --compare evals/results/run_a.jsonl evals/results/run_b.jsonl

# Trend view across all runs
uv run python -m evals.report --results-dir evals/results/
```

| Flag | Description |
|---|---|
| `--results` | Path to a single results JSONL file |
| `--compare` | Two result file paths for side-by-side comparison |
| `--results-dir` | Directory of results for trend view |
| `--verbose` | Show per-thread disagreements |
| `--format` | `table` (default) or `json` |

### 5. Web UI — Interactive reporting and comparison

Launch a local web server to browse runs, view metrics, compare results, and inspect chain-of-thought reasoning.

```bash
# Set auth secret (required unless you want no auth)
export EVAL_WEB_SECRET="your-secret-here"

# Launch web UI
uv run python -m evals.run_web

# Custom port
uv run python -m evals.run_web --port 8080
```

Navigate to `http://localhost:5000` in your browser. Features:

- **Run list**: Filter by model, stages, tag. Click any run to view details.
- **Run detail**: Metrics, confusion matrices, per-class P/R/F1, per-thread results with duration and chain-of-thought.
- **Compare**: Select a baseline run and one or more comparison runs to see side-by-side accuracy deltas.

| Flag | Description |
|---|---|
| `--host` | Host to bind to (default: `127.0.0.1`) |
| `--port` | Port to bind to (default: `5000`) |

## Chain-of-Thought Capture

When running evaluations with the LLM cache enabled (the default), chain-of-thought content from `<think>...</think>` blocks is automatically captured and stored in sidecar files alongside results.

**Sidecar format:** `evals/results/<run>.cot.jsonl` — one JSON line per thread with `stage1_thinking` and `stage2_thinking` fields.

Chain-of-thought is viewable in the web UI on the run detail page.

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

## Typical Workflows

**Prompt A/B test:**

```bash
# Run baseline
uv run python -m evals.run_eval --tag baseline
# Edit prompts in config.toml, then re-run
uv run python -m evals.run_eval --tag new-prompts
# Compare
uv run python -m evals.report --compare evals/results/*baseline*.jsonl evals/results/*new-prompts*.jsonl
```

**Model swap:**

```bash
# Run with current model
uv run python -m evals.run_eval --tag deepseek-v3
# Change [llm.cloud] model in config.toml, then re-run
uv run python -m evals.run_eval --config config_v2.toml --tag gpt-4o
# Compare
uv run python -m evals.report --compare evals/results/*deepseek*.jsonl evals/results/*gpt-4o*.jsonl
```

**Ongoing monitoring:**

```bash
# Periodically harvest new production data
uv run python -m evals.harvest --proxy-url http://localhost:8000 --append
# Review new threads
uv run python -m evals.review --unreviewed-only
# Re-evaluate and check trends
uv run python -m evals.run_eval --tag weekly
uv run python -m evals.report --results-dir evals/results/
```
