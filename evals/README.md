# Evaluation Suite

A 4-stage pipeline for measuring classification accuracy against human-reviewed threads, plus a web UI for browsing results.

```
harvest → review → run_eval → report
```

> For complete CLI flag references, cache internals, and chain-of-thought capture details, see [README-technical.md](README-technical.md).

## Setup

The eval tools run outside Docker and need access to the same environment variables as the daemon. If you haven't already, symlink to `agent-stack/.env`:

```bash
ln -s ../agent-stack/.env .env
```

Since the symlinked `.env` may contain Docker-internal hostnames (e.g. `PROXY_URL=http://api-proxy:8000`), use `--proxy-url` to point at the proxy's host-accessible address:

```bash
uv run python -m evals.harvest --proxy-url http://localhost:8000 --max-threads 200
```

**Stop the daemon before running evals** if your local MLX server only supports one model at a time. The daemon and eval runner may request different models, causing the server to swap models mid-request and return errors.

## 1. Harvest — Build a golden set from production data

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

## 2. Review — Manually verify ground truth labels

Interactive CLI for reviewing and correcting labels in the golden set. Saves atomically after each session. Press `z` at any prompt to undo the last classification — undo works as a stack, walking back through previous decisions.

```bash
# Review all threads (blind mode by default)
uv run python -m evals.review

# Show existing labels while reviewing
uv run python -m evals.review --show-labels

# Curses TUI for editing reviewed threads
uv run python -m evals.review --edit

# Review only sender classification (stage 1) or label classification (stage 2)
uv run python -m evals.review --stage 1
uv run python -m evals.review --stage 2

# Only review unreviewed threads
uv run python -m evals.review --unreviewed-only
```

## 3. Run — Replay golden set through the classifier

Sends each golden thread through the real `EmailClassifier` with live LLM endpoints. Results are written to timestamped JSONL files in `evals/results/`.

```bash
# Full evaluation (both stages)
uv run python -m evals.run_eval

# Evaluate only Stage 1 (sender classification)
uv run python -m evals.run_eval --stages stage1_only

# Evaluate only Stage 2 (uses expected sender type as input)
uv run python -m evals.run_eval --stages stage2_only

# Override model without editing config.toml
uv run python -m evals.run_eval --local-model qwen/qwen3-14b --tag qwen3-14b

# Dry run — show what would be evaluated
uv run python -m evals.run_eval --dry-run
```

Results include an LLM response cache so re-runs with the same config are instant. Pass `--no-cache` to bypass it.

## 4. Report — Compute metrics and compare runs

Generates accuracy, confusion matrix, per-class precision/recall/F1, and privacy violation reports.

```bash
# Single run report
uv run python -m evals.report --results evals/results/run.jsonl

# Verbose — show per-thread disagreements
uv run python -m evals.report --results evals/results/run.jsonl --verbose

# Compare two runs side by side
uv run python -m evals.report --compare evals/results/run_a.jsonl evals/results/run_b.jsonl

# Trend view across all runs
uv run python -m evals.report --results-dir evals/results/
```

## 5. Web UI — Interactive reporting and comparison

Launch a local web server to browse runs, view metrics, compare results, and inspect chain-of-thought reasoning.

```bash
export EVAL_WEB_SECRET="your-secret-here"
uv run python -m evals.run_web
```

Navigate to `http://localhost:5000`. Features:

- **Run list**: Filter by model, stages, tag. Click any run to view details.
- **Run detail**: Metrics, confusion matrices, per-class P/R/F1, per-thread results with chain-of-thought.
- **Compare**: Select a baseline and comparison runs to see side-by-side accuracy deltas.

## Typical Workflows

**Prompt A/B test:**

```bash
uv run python -m evals.run_eval --tag baseline
# Edit prompts in config.toml, then:
uv run python -m evals.run_eval --tag new-prompts
uv run python -m evals.report --compare evals/results/*baseline*.jsonl evals/results/*new-prompts*.jsonl
```

**Model swap:**

```bash
uv run python -m evals.run_eval --tag deepseek-v3
uv run python -m evals.run_eval --config config_v2.toml --tag gpt-4o
uv run python -m evals.report --compare evals/results/*deepseek*.jsonl evals/results/*gpt-4o*.jsonl
```

**Ongoing monitoring:**

```bash
uv run python -m evals.harvest --proxy-url http://localhost:8000 --append
uv run python -m evals.review --unreviewed-only
uv run python -m evals.run_eval --tag weekly
uv run python -m evals.report --results-dir evals/results/
```
