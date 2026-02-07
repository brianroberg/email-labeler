# Email Labeler

A background daemon that continuously polls Gmail for unclassified emails, classifies them using a two-tier LLM system, and applies labels autonomously.

## Privacy Model

Email labeler enforces a strict privacy invariant: **person email bodies never leave the local network.**

The system uses two classification stages to achieve this:

1. **Stage 1 (Cloud LLM)** receives only email metadata — sender, subject line, and Gmail snippet — and determines whether the sender is a real person or an automated service. The full email body is never sent to the cloud in this stage.

2. **Stage 2** routes based on the Stage 1 result:
   - **Service emails** have their full body sent to the cloud LLM for classification. This is safe because service emails (receipts, newsletters, notifications) contain no personal correspondence.
   - **Person emails** have their full body sent to a local MLX/Qwen3 instance running on the same network. The body never leaves the local network.

If the local LLM is unavailable, person emails are silently skipped and retried on the next poll cycle. They are never sent to the cloud as a fallback.

## Label Taxonomy

The daemon classifies emails into four categories and applies the corresponding Gmail label:

| Label | Meaning | Action |
|---|---|---|
| `agent/needs-response` | Requires a reply or action from you | Stays in inbox |
| `agent/fyi` | Worth reading, no action needed | Stays in inbox |
| `agent/low-priority` | Routine notifications, newsletters | Archived |
| `agent/unwanted` | Spam, unsolicited marketing | Archived |

Additional labels are used as markers:

| Label | Purpose |
|---|---|
| `agent/processed` | Applied to every classified email. Used by the poll query to skip already-processed messages. |
| `agent/personal` | Applied when Stage 1 classified the sender as a real person. Indicates the email body was processed by the local LLM only. |
| `agent/non-personal` | Applied when Stage 1 classified the sender as an automated service. Indicates the email body was processed by the cloud LLM. |
| `agent/would-have-deleted` | Applied alongside `agent/unwanted`. Marks emails that the daemon would delete if it had permission, without actually deleting anything. |

## Architecture

```
                          +-----------+
                          | Gmail API |
                          +-----+-----+
                                |
                          +-----+-----+
                          | api-proxy |  (human-in-the-loop controls)
                          +-----+-----+
                                |
              +-----------------+-----------------+
              |           email-labeler            |
              |                                    |
              |  Poll loop (every 60s)             |
              |    │                               |
              |    ├─ list unprocessed emails       |
              |    │                               |
              |    ├─ For each email:              |
              |    │   ├─ Stage 1 ──► Cloud LLM   |
              |    │   │  (metadata only)          |
              |    │   │    └─► PERSON or SERVICE  |
              |    │   │                           |
              |    │   ├─ Stage 2a (service)       |
              |    │   │    └─► Cloud LLM          |
              |    │   │         (full body)       |
              |    │   │                           |
              |    │   ├─ Stage 2b (person)        |
              |    │   │    └─► Local MLX          |
              |    │   │         (full body)       |
              |    │   │                           |
              |    │   └─ Apply label + action     |
              |    │                               |
              |    └─ Write healthcheck file       |
              +------------------------------------+
```

## Project Structure

```
email-labeler/
├── daemon.py           Main entry point: polling loop and orchestration
├── classifier.py       Two-tier classification logic and LLM output parsing
├── labeler.py          Gmail label verification and application
├── llm_client.py       LLM abstraction for cloud and local endpoints
├── proxy_client.py     Gmail API proxy client (shared with email-agent)
├── gmail_utils.py      Email header and body parsing (shared with email-agent)
├── config_utils.py     Config loading and env var substitution
├── config.toml         Label definitions, prompts, and operational parameters
├── pyproject.toml      Python project metadata and dependencies
├── Dockerfile          Container image definition
├── .env.example        Environment variable template
├── evals/              Classification evaluation suite
│   ├── schemas.py      Dataclasses: GoldenThread, PredictionResult, RunMeta
│   ├── harvest.py      Pull processed threads → golden set JSONL
│   ├── review.py       Interactive CLI for label review and correction
│   ├── run_eval.py     Replay golden set through real classifier
│   ├── report.py       Metrics: accuracy, confusion matrix, P/R/F1, privacy
│   └── results/        Timestamped result files from evaluation runs
└── tests/
    ├── conftest.py          Shared fixtures and sample Gmail data
    ├── test_llm_client.py   LLM client tests
    ├── test_classifier.py   Classifier tests
    ├── test_labeler.py      Label manager tests
    ├── test_daemon.py       Daemon orchestration tests
    ├── test_config_utils.py Config loading tests
    ├── test_eval_schemas.py Golden set and result serialization tests
    ├── test_eval_harvest.py Ground truth inference and deduplication tests
    └── test_eval_report.py  Metrics computation and report formatting tests
```

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to an [api-proxy](../api-proxy) instance with a valid API key
- A cloud LLM endpoint (any OpenAI-compatible chat completion API)
- A local MLX LLM endpoint for person email classification (optional but recommended)
- All eight Gmail labels created manually (see [Label Setup](#label-setup))

## Setup

### 1. Install dependencies

```bash
uv sync --extra dev
```

### 2. Create environment file

If running as part of the `agent-stack` setup (recommended), symlink to the shared `.env` to avoid maintaining variables in two places:

```bash
ln -s ../agent-stack/.env .env
```

Otherwise, copy the example and fill in your values:

```bash
cp .env.example .env
```

```env
# Required: api-proxy authentication
PROXY_API_KEY=aproxy_your_key_here

# Required: Cloud LLM (any OpenAI-compatible chat completion endpoint)
CLOUD_LLM_URL=https://your-llm-provider.com/v1/chat/completions
CLOUD_LLM_API_KEY=your_api_key_here

# Recommended: Local LLM (MLX/Qwen3 for person email privacy)
MLX_URL=http://macbook:8080/v1/chat/completions

# Optional: Override proxy URL (defaults to http://host.docker.internal:8000)
# PROXY_URL=http://api-proxy:8000
```

### 3. Label Setup

The api-proxy blocks programmatic label creation, so all eight labels must be created manually in Gmail before the daemon starts.

In Gmail, go to **Settings > Labels > Create new label** and create each of these:

```
agent/needs-response
agent/fyi
agent/low-priority
agent/unwanted
agent/processed
agent/would-have-deleted
agent/personal
agent/non-personal
```

Gmail will treat the `/` as a label hierarchy separator, nesting them under an `agent` parent. The daemon verifies all labels exist on startup and exits with an error if any are missing.

## Running

### Local development

```bash
uv run python daemon.py
```

The daemon will:
1. Verify all eight Gmail labels exist (exits if any are missing)
2. Enter the poll loop, querying Gmail every 60 seconds
3. Classify and label each unprocessed email
4. Write a healthcheck timestamp to `/tmp/healthcheck`

### Docker

Build and run with Docker Compose from the `agent-stack` directory:

```bash
docker compose build email-labeler
docker compose up email-labeler
```

The Docker Compose configuration also starts a Tailscale sidecar (`email-labeler-tailscale`) that shares the network namespace with the daemon container. This allows the daemon to reach the local MLX server via Tailscale hostname.

Required environment variables in `agent-stack/.env`:

```env
EMAIL_LABELER_API_KEY=aproxy_your_key_here
CLOUD_LLM_URL=https://your-llm-provider.com/v1/chat/completions
CLOUD_LLM_API_KEY=your_api_key_here
MLX_URL=http://macbook:8080/v1/chat/completions
TS_AUTHKEY=tskey-auth-...  # for Tailscale sidecar
```

### Health checking

The daemon writes a timestamp to `/tmp/healthcheck` after each successful poll cycle. The Dockerfile includes a `HEALTHCHECK` instruction that verifies this file was updated within the last 180 seconds:

```dockerfile
HEALTHCHECK --interval=120s --timeout=5s --retries=3 \
    CMD test -f /tmp/healthcheck && \
        test $(($(date +%s) - $(stat -c %Y /tmp/healthcheck))) -lt 180
```

Check container health with:

```bash
docker inspect --format='{{.State.Health.Status}}' agent-stack-email-labeler-1
```

## Evaluation Suite

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
uv run python -m evals.harvest --output evals/golden_set.jsonl --max-threads 200

# Append new threads (deduplicates automatically)
uv run python -m evals.harvest --output evals/golden_set.jsonl --append

# Filter by sender type or label
uv run python -m evals.harvest --output evals/golden_set.jsonl --sender-type person
uv run python -m evals.harvest --output evals/golden_set.jsonl --label needs_response
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
# Review all threads
uv run python -m evals.review

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

# Only evaluate reviewed threads, with higher parallelism
uv run python -m evals.run_eval --reviewed-only --parallelism 5

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
| `--reviewed-only` | Only evaluate threads marked as reviewed |
| `--dry-run` | Show what would be evaluated without calling LLMs |
| `--tag` | Tag for the results filename (e.g. `new-prompts`) |

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

### Typical Workflows

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
uv run python -m evals.harvest --append
# Review new threads
uv run python -m evals.review --unreviewed-only
# Re-evaluate and check trends
uv run python -m evals.run_eval --reviewed-only --tag weekly
uv run python -m evals.report --results-dir evals/results/
```

## Testing

All tests use mocks and require no external services.

```bash
# Run all 66 tests
uv run --extra dev pytest tests/ -v

# Run a specific test file
uv run --extra dev pytest tests/test_classifier.py -v

# Run with short output
uv run --extra dev pytest tests/
```

### Test coverage by module

| Test file | Module | What's covered |
|---|---|---|
| `test_llm_client.py` | `llm_client.py` | Request format, auth headers, `<think>` tag stripping, error handling, availability checks |
| `test_classifier.py` | `classifier.py` | `parse_sender` formats, `parse_sender_type` edge cases and defaults, `parse_email_label` edge cases and defaults, cloud/local routing, full pipeline |
| `test_labeler.py` | `labeler.py` | Label verification (all present, partial, none), label ID mapping, inbox/archive actions, extra labels for unwanted, single API call per email |
| `test_daemon.py` | `daemon.py` | Service email path, person email path, MLX-unavailable skip, error isolation, config loading |
| `test_config_utils.py` | `config_utils.py` | Config loading, `{env.VAR}` substitution |
| `test_eval_schemas.py` | `evals/schemas.py` | GoldenThread/PredictionResult/RunMeta serialization round-trips |
| `test_eval_harvest.py` | `evals/harvest.py` | Ground truth inference from labels, deduplication |
| `test_eval_report.py` | `evals/report.py` | Confusion matrix, precision/recall/F1, accuracy, privacy violation metrics |

### Linting

```bash
uv run --extra dev ruff check .
```

## Configuration

All operational parameters are in `config.toml`. The daemon reads this file on startup.

### Daemon settings

```toml
[daemon]
poll_interval_seconds = 60     # How often to poll Gmail
max_emails_per_cycle = 10      # Max emails to process per poll
gmail_query = "in:inbox -label:agent/processed"  # Gmail search query
healthcheck_file = "/tmp/healthcheck"            # Healthcheck timestamp path
```

### LLM settings

The cloud LLM model is configured here, not in `.env`. The URL and API key come from
`.env`, but the model name and inference parameters live in `config.toml` so they stay
in version control.

```toml
[llm.cloud]
model = "deepseek/deepseek-v3.2"   # must match your provider's model ID
max_tokens = 8096
temperature = 0.2
timeout = 60

[llm.local]
model = "mlx-community/Qwen3-14B-4bit"
max_tokens = 8096
temperature = 0.2
timeout = 120       # Local LLM gets more time (runs on consumer hardware)
```

### Prompt templates

The `[prompts.sender_classification]` and `[prompts.email_classification]` sections contain the system prompts and user message templates used for each classification stage. Templates use Python format strings with `{sender}`, `{subject}`, `{snippet}`, and `{body}` placeholders.

## Resilience

The daemon is designed to run unattended and recover from transient failures:

- **Exponential backoff**: If a poll cycle fails, the sleep interval doubles (up to 10x the base interval), then resets on the next successful cycle.
- **Per-email error isolation**: If one email fails to classify, the error is logged and the loop continues with the next email.
- **MLX graceful degradation**: If the local MLX server is unreachable, person emails are skipped (retried next cycle) while service emails continue to be classified via the cloud LLM.
- **Safe defaults**: If the LLM returns an unrecognizable sender type, it defaults to SERVICE (body goes to cloud, safe for non-person content). If the LLM returns an unrecognizable classification, it defaults to LOW_PRIORITY (archived but not marked as unwanted).
- **Startup validation**: The daemon verifies all required Gmail labels exist before entering the poll loop, preventing silent misclassification from misconfigured labels.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PROXY_API_KEY` | Yes | — | API key for the api-proxy server |
| `PROXY_URL` | No | `http://host.docker.internal:8000` | URL of the api-proxy server |
| `CLOUD_LLM_URL` | Yes | — | Cloud LLM chat completion endpoint (any OpenAI-compatible API) |
| `CLOUD_LLM_API_KEY` | Yes | — | API key for the cloud LLM |
| `MLX_URL` | No | — | Local MLX LLM chat completion endpoint. If unset or unreachable, person emails are skipped. |

Note: The cloud LLM **model name** is configured in `config.toml` under `[llm.cloud]`, not in `.env`. This keeps secrets (keys, URLs) in `.env` while operational parameters (model, temperature, prompts) stay in version-controlled `config.toml`.
