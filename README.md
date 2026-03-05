# Email Labeler

A background daemon that continuously polls Gmail for unclassified emails, classifies them using a two-tier LLM system, and applies labels autonomously.

> For detailed configuration reference, environment variables, project structure, and test coverage, see [README-technical.md](README-technical.md).

## Privacy Model

Email labeler enforces a strict privacy invariant: **person email bodies never leave the local network.**

The system uses two classification stages to achieve this:

1. **Stage 1 (Cloud LLM)** receives only email metadata — sender, subject line, and Gmail snippet — and determines whether the sender is a real person or an automated service. The full email body is never sent to the cloud in this stage.

2. **Stage 2** routes based on the Stage 1 result:
   - **Service emails** have their full body sent to the cloud LLM for classification. This is safe because service emails (receipts, newsletters, notifications) contain no personal correspondence.
   - **Person emails** have their full body sent to a local MLX/Qwen3 instance running on the same network. The body never leaves the local network.

If the local LLM is unavailable, person emails are silently skipped and retried on the next poll cycle. They are never sent to the cloud as a fallback.

## Label Taxonomy

The daemon classifies emails into three categories and applies the corresponding Gmail label:

| Label | Meaning | Action |
|---|---|---|
| `agent/needs-response` | Requires a reply or action from you | Stays in inbox |
| `agent/fyi` | Worth reading, no action needed | Stays in inbox |
| `agent/low-priority` | Routine notifications, newsletters, spam, unwanted | Archived |

Additional marker labels (`agent/processed`, `agent/personal`, `agent/non-personal`) track processing state and routing decisions. The newsletter pipeline adds its own labels under `agent/newsletter/`.

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

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Access to an [api-proxy](../api-proxy) instance with a valid API key
- A cloud LLM endpoint (any OpenAI-compatible chat completion API)
- A local MLX LLM endpoint for person email classification (optional but recommended)
- All Gmail labels created manually (see [Label Setup](#label-setup))

## Setup

### 1. Install dependencies

```bash
uv sync --extra dev
```

### 2. Create environment file

If running as part of the `agent-stack` setup (recommended), symlink to the shared `.env`:

```bash
ln -s ../agent-stack/.env .env
```

Otherwise, copy the example and fill in your values:

```bash
cp .env.example .env
```

At minimum you need:

```env
PROXY_API_KEY=aproxy_your_key_here
CLOUD_LLM_URL=https://your-llm-provider.com/v1/chat/completions
CLOUD_LLM_API_KEY=your_api_key_here
MLX_URL=http://macbook:8080/v1/chat/completions
MLX_MODEL=qwen/qwen3-14b
```

See [README-technical.md](README-technical.md#environment-variables) for the full variable reference.

### 3. Label Setup

The api-proxy blocks programmatic label creation, so all labels must be created manually in Gmail before the daemon starts.

In Gmail, go to **Settings > Labels > Create new label** and create each of these:

```
agent/needs-response
agent/fyi
agent/low-priority
agent/processed
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
1. Verify all Gmail labels exist (exits if any are missing)
2. Enter the poll loop, querying Gmail every 60 seconds
3. Classify and label each unprocessed email
4. Write a healthcheck timestamp to `/tmp/healthcheck`

### Docker (via agent-stack)

```bash
docker compose build email-labeler
docker compose up email-labeler
```

To classify only newsletters (skipping all other emails):

```bash
NEWSLETTER_ONLY=1 docker compose up email-labeler
```

The Docker Compose configuration also starts a Tailscale sidecar that shares the network namespace, allowing the daemon to reach the local MLX server via Tailscale hostname.

## Resilience

The daemon is designed to run unattended and recover from transient failures:

- **Exponential backoff**: If a poll cycle fails, the sleep interval doubles (up to 10x the base interval), then resets on the next successful cycle.
- **Per-email error isolation**: If one email fails to classify, the error is logged and the loop continues with the next email.
- **MLX graceful degradation**: If the local MLX server is unreachable, person emails are skipped (retried next cycle) while service emails continue to be classified via the cloud LLM.
- **Safe defaults**: Unrecognizable sender type → SERVICE (safe). Unrecognizable classification → LOW_PRIORITY (archived, not deleted).
- **Startup validation**: The daemon verifies all required Gmail labels exist before entering the poll loop.

## Evaluation Suite

The `evals/` directory provides a 4-stage pipeline (`harvest → review → run_eval → report`) for measuring classification accuracy against a golden set of human-reviewed threads.

See [`evals/README.md`](evals/README.md) for full documentation and CLI reference.

## Testing

All tests use mocks and require no external services.

```bash
# Run all tests
uv run --extra dev pytest tests/ -v

# Run a specific test file
uv run --extra dev pytest tests/test_classifier.py -v

# Lint
uv run --extra dev ruff check .
```

See [README-technical.md](README-technical.md#test-coverage-by-module) for per-module coverage details.
