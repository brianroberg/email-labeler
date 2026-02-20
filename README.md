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
| `agent/low-priority` | Routine notifications, newsletters, spam, unwanted | Archived |

Additional labels are used as markers:

| Label | Purpose |
|---|---|
| `agent/processed` | Applied to every classified email. Used by the poll query to skip already-processed messages. |
| `agent/personal` | Applied when Stage 1 classified the sender as a real person. Indicates the email body was processed by the local LLM only. |
| `agent/non-personal` | Applied when Stage 1 classified the sender as an automated service. Indicates the email body was processed by the cloud LLM. |

Newsletter labels (see [Newsletter Classification](#newsletter-classification)):

| Label | Purpose |
|---|---|
| `agent/newsletter` | Marker applied to all newsletter emails |
| `agent/newsletter/excellent` | Best story scored >= 4.0 average |
| `agent/newsletter/good` | Best story scored >= 3.0 average |
| `agent/newsletter/fair` | Best story scored >= 2.0 average |
| `agent/newsletter/poor` | Best story scored < 2.0 average |
| `agent/newsletter/no-stories` | No extractable stories found |
| `agent/newsletter/theme/*` | Theme tags: `scripture`, `christlikeness`, `church`, `vocation-family`, `disciple-making` |

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
              |    ├─ For each thread:             |
              |    │   │                           |
              |    │   ├─ Newsletter? (To: check)  |
              |    │   │   YES ──► Cloud LLM ×3   |
              |    │   │   (extract, score, theme) |
              |    │   │   └─► label + JSONL       |
              |    │   │                           |
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

## Newsletter Classification

Emails sent to a configured newsletter recipient address (e.g. `newsletters@dm.org`) are detected by a deterministic `To:`/`Cc:` header check and routed to a dedicated classification pipeline instead of the standard priority classification.

### Pipeline

The pipeline makes 1 + 2N cloud LLM calls for a newsletter containing N stories:

1. **Story Extraction** (1 call) — The cloud LLM receives the full newsletter body and extracts individual stories, each with a title and text. Non-story content (headers, footers, donation appeals, event calendars) is skipped. If no stories are found, the email is labeled `agent/newsletter/no-stories`.

2. **Quality Assessment** (1 call per story) — Each story is scored on four dimensions (1–5 scale):

   | Dimension | High score means... |
   |---|---|
   | Simple | Focuses on one key idea; no tangents |
   | Concrete | Narrates particular events, people, places |
   | Personal | Centers around one or a few people |
   | Dynamic | Shows transformation; a before and after |

   The average score determines a quality tier: **excellent** (>= 4.0), **good** (>= 3.0), **fair** (>= 2.0), **poor** (< 2.0). The LLM's chain-of-thought reasoning is captured for later review.

3. **Theme Classification** (1 call per story) — Each story is tagged with themes from the organization's Ends Statement: `scripture`, `christlikeness`, `church`, `vocation_family`, `disciple_making`. Multiple themes per story are allowed. Chain-of-thought reasoning is captured.

### Output

Each classified newsletter is:
- **Labeled in Gmail** with `agent/newsletter`, a quality tier label (from the best story), and theme labels (union across all stories), then archived.
- **Recorded to JSONL** at `data/newsletter_assessments.jsonl` with per-story scores, themes, and chain-of-thought reasoning.

### Configuration

Newsletter classification is configured in `config.toml`:

```toml
[newsletter]
recipient = "newsletters@dm.org"
output_file = "data/newsletter_assessments.jsonl"
```

Set `NEWSLETTER_ONLY=1` in `.env` to skip non-newsletter threads (useful for testing newsletter classification in isolation).

## Newsletter Assessment TUI

A Textual-based terminal UI for browsing and filtering the `newsletter_assessments.jsonl` data.

```
uv run python -m tui [path/to/assessments.jsonl]
```

Defaults to `data/newsletter_assessments.jsonl` if no path is given.

### Layout

Three-panel vertical layout:

- **Newsletter list** — Filterable table of newsletters showing subject, sender, tier, and date
- **Story list** — Stories from the selected newsletter with tier, average score, and themes
- **Detail panel** — Full detail for the selected story: dimension scores, themes, quality chain-of-thought, theme chain-of-thought, and story text

### Key Bindings

| Key | Action |
|---|---|
| `↑` / `↓` | Navigate within a list |
| `Tab` | Cycle focus between newsletter list, story list, and detail panel |
| `t` | Cycle tier filter: All → excellent → good → fair → poor → All |
| `h` | Cycle theme filter: All → scripture → christlikeness → church → vocation_family → disciple_making → All |
| `q` | Quit |

## Project Structure

```
email-labeler/
├── daemon.py           Main entry point: polling loop and orchestration
├── classifier.py       Two-tier classification logic and LLM output parsing
├── labeler.py          Gmail label verification and application
├── newsletter.py       Newsletter story extraction, quality scoring, theme tagging
├── llm_client.py       LLM abstraction for cloud and local endpoints
├── proxy_client.py     Gmail API proxy client (shared with email-agent)
├── gmail_utils.py      Email header and body parsing (shared with email-agent)
├── config_utils.py     Config loading and env var substitution
├── tui.py              Newsletter assessment TUI (Textual app)
├── tui_data.py         TUI data loading, filtering, and formatting
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
    ├── test_newsletter.py   Newsletter pipeline tests
    ├── test_tui_data.py     TUI data loading and filtering tests
    ├── test_tui.py          TUI app behavior tests (Textual pilot)
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
- All Gmail labels created manually (see [Label Setup](#label-setup))

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
MLX_MODEL=qwen/qwen3-14b

# Optional: Override proxy URL (defaults to http://host.docker.internal:8000)
# PROXY_URL=http://api-proxy:8000
```

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
agent/newsletter
agent/newsletter/excellent
agent/newsletter/good
agent/newsletter/fair
agent/newsletter/poor
agent/newsletter/no-stories
agent/newsletter/theme/scripture
agent/newsletter/theme/christlikeness
agent/newsletter/theme/church
agent/newsletter/theme/vocation-family
agent/newsletter/theme/disciple-making
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
MLX_MODEL=qwen/qwen3-14b
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

The `evals/` directory provides a 4-stage pipeline (`harvest → review → run_eval → report`) for measuring classification accuracy against a golden set of human-reviewed threads. It includes an LLM response cache for fast re-runs and supports prompt A/B testing, model comparison, and ongoing monitoring workflows.

See [`evals/README.md`](evals/README.md) for full documentation and CLI reference.

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
| `test_labeler.py` | `labeler.py` | Label verification (all present, partial, none), label ID mapping, inbox/archive actions, single API call per email |
| `test_daemon.py` | `daemon.py` | Service email path, person email path, MLX-unavailable skip, error isolation, config loading |
| `test_config_utils.py` | `config_utils.py` | Config loading, `{env.VAR}` substitution |
| `test_eval_schemas.py` | `evals/schemas.py` | GoldenThread/PredictionResult/RunMeta serialization round-trips |
| `test_eval_harvest.py` | `evals/harvest.py` | Ground truth inference from labels, deduplication |
| `test_newsletter.py` | `newsletter.py` | Story parsing, quality score parsing, theme parsing, tier computation, JSONL writing, full pipeline, newsletter detection |
| `test_tui_data.py` | `tui_data.py` | JSONL loading, story field parsing, null handling, malformed lines, tier/theme filtering, detail formatting |
| `test_tui.py` | `tui.py` | App launch, newsletter table rendering, drill-down navigation, CoT display, tier/theme filter cycling |
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
model = "{env.MLX_MODEL}"              # set MLX_MODEL in .env (shared with email-agent)
max_tokens = 8096
temperature = 0.2
timeout = 120       # Local LLM gets more time (runs on consumer hardware)
```

### Extra request body fields

Each `[llm.*]` section supports an optional `extra_body` table. Any key-value pairs defined here are merged into every API request body sent to that endpoint. This is useful for provider-specific parameters that aren't part of the standard OpenAI chat completion format.

**Disabling thinking for reasoning models** — Models like Qwen3, DeepSeek-R1, and GLM-4.5 generate chain-of-thought reasoning in `<think>` tags before answering. While the daemon already strips these tags from responses, you can disable thinking entirely to save tokens and reduce latency.

For providers that accept a top-level `enable_thinking` flag (Novita.ai, many OpenAI-compatible APIs):

```toml
[llm.local.extra_body]
enable_thinking = false
```

For LM Studio with models that use `chat_template_kwargs` (e.g. Qwen3):

```toml
[llm.local.extra_body.chat_template_kwargs]
enable_thinking = false
```

You can put any provider-specific fields in `extra_body` — it is not limited to thinking controls. For example:

```toml
[llm.cloud.extra_body]
top_p = 0.9
frequency_penalty = 0.5
```

### Prompt templates

The `[prompts.sender_classification]` and `[prompts.email_classification]` sections contain the system prompts and user message templates used for each classification stage. Templates use Python format strings with `{sender}`, `{subject}`, `{snippet}`, and `{body}` placeholders.

## Resilience

The daemon is designed to run unattended and recover from transient failures:

- **Exponential backoff**: If a poll cycle fails, the sleep interval doubles (up to 10x the base interval), then resets on the next successful cycle.
- **Per-email error isolation**: If one email fails to classify, the error is logged and the loop continues with the next email.
- **MLX graceful degradation**: If the local MLX server is unreachable, person emails are skipped (retried next cycle) while service emails continue to be classified via the cloud LLM.
- **Safe defaults**: If the LLM returns an unrecognizable sender type, it defaults to SERVICE (body goes to cloud, safe for non-person content). If the LLM returns an unrecognizable classification, it defaults to LOW_PRIORITY (archived).
- **Startup validation**: The daemon verifies all required Gmail labels exist before entering the poll loop, preventing silent misclassification from misconfigured labels.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PROXY_API_KEY` | Yes | — | API key for the api-proxy server |
| `PROXY_URL` | No | `http://host.docker.internal:8000` | URL of the api-proxy server |
| `CLOUD_LLM_URL` | Yes | — | Cloud LLM chat completion endpoint (any OpenAI-compatible API) |
| `CLOUD_LLM_API_KEY` | Yes | — | API key for the cloud LLM |
| `MLX_URL` | No | — | Local MLX LLM chat completion endpoint. If unset or unreachable, person emails are skipped. |
| `MLX_MODEL` | No | — | Local LLM model name. Shared with email-agent so both services use the same model. Referenced in `config.toml` as `{env.MLX_MODEL}`. |
| `USER_NAME` | No | — | User's display name, substituted into classification prompts via `{env.USER_NAME}` in `config.toml`. |
| `VIP_SENDERS` | No | — | Comma-separated email addresses of VIP senders. VIP threads skip the sender classification LLM call. |
| `EMAIL_LABELER_API_KEY` | No | — | Fallback API key for the api-proxy server, used when `PROXY_API_KEY` is not set. |
| `NEWSLETTER_ONLY` | No | — | Set to `1`, `true`, or `yes` to skip non-newsletter threads. Useful for testing newsletter classification in isolation. |

Note: The cloud LLM **model name** is configured in `config.toml` under `[llm.cloud]`, not in `.env`. The local LLM **model name** is set via the `MLX_MODEL` environment variable (shared with email-agent) and referenced in `config.toml` as `{env.MLX_MODEL}`. This keeps secrets (keys, URLs) in `.env` while operational parameters (temperature, prompts) stay in version-controlled `config.toml`.
