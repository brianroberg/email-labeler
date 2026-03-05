# Email Labeler — Technical Reference

Detailed configuration, environment variables, project structure, and test coverage for the email-labeler daemon. For an overview of the system, see [README.md](README.md).

## Project Structure

```
email-labeler/
├── daemon.py           Main entry point: polling loop and orchestration
├── classifier.py       Two-tier classification logic and LLM output parsing
├── labeler.py          Gmail label verification and application
├── newsletter.py       Newsletter story extraction, quality scoring, and theme classification
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
    ├── test_newsletter.py   Newsletter pipeline tests
    ├── test_eval_schemas.py Golden set and result serialization tests
    ├── test_eval_harvest.py Ground truth inference and deduplication tests
    └── test_eval_report.py  Metrics computation and report formatting tests
```

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

### Label configuration

All label names and their inbox/archive actions are defined in `config.toml` under `[labels]` and `[labels.actions]`. Newsletter labels live under `[newsletter.labels]`.

## Health Checking

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

## Test Coverage by Module

| Test file | Module | What's covered |
|---|---|---|
| `test_llm_client.py` | `llm_client.py` | Request format, auth headers, `<think>` tag stripping, error handling, availability checks |
| `test_classifier.py` | `classifier.py` | `parse_sender` formats, `parse_sender_type` edge cases and defaults, `parse_email_label` edge cases and defaults, cloud/local routing, full pipeline |
| `test_labeler.py` | `labeler.py` | Label verification (all present, partial, none), label ID mapping, inbox/archive actions, single API call per email |
| `test_daemon.py` | `daemon.py` | Service email path, person email path, MLX-unavailable skip, error isolation, config loading |
| `test_config_utils.py` | `config_utils.py` | Config loading, `{env.VAR}` substitution |
| `test_newsletter.py` | `newsletter.py` | Newsletter story extraction, quality scoring, theme classification |
| `test_eval_schemas.py` | `evals/schemas.py` | GoldenThread/PredictionResult/RunMeta serialization round-trips |
| `test_eval_harvest.py` | `evals/harvest.py` | Ground truth inference from labels, deduplication |
| `test_eval_report.py` | `evals/report.py` | Confusion matrix, precision/recall/F1, accuracy, privacy violation metrics |
