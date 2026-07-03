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
├── tui_common.py       Shared Textual widgets/screens for the TUIs
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
│   ├── newsletter_schemas.py  Newsletter golden-set and result dataclasses
│   ├── newsletter_harvest.py  Pull candidate newsletters → golden set (unlabeled)
│   ├── newsletter_label.py    Hand-label story quality scores and themes
│   ├── newsletter_run.py      Replay golden stories through the newsletter classifier
│   ├── newsletter_report.py   Newsletter tier/dimension/theme/extraction metrics
│   └── results/        Timestamped result files from evaluation runs
├── newsletter_review/  Textual TUI for browsing newsletter assessments
│   ├── __main__.py     CLI entry point (python -m newsletter_review)
│   └── tui.py          Pure data helpers + Textual app
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
| `NEWSLETTER_LLM_URL` | No | `CLOUD_LLM_URL` | Endpoint for the newsletter grading LLM (`[newsletter.llm]` in `config.toml`). Set when that model needs a different provider than the cloud classifier — e.g. a Claude model via Anthropic's OpenAI-compatible endpoint. |
| `NEWSLETTER_LLM_API_KEY` | No | `CLOUD_LLM_API_KEY` | API key for the newsletter LLM endpoint. The override is atomic: once `NEWSLETTER_LLM_URL` is set, the key comes only from this var (never the cloud key), so set both together. |
| `MLX_URL` | No | — | Local MLX LLM chat completion endpoint. If unset or unreachable, person emails are skipped. |
| `MLX_MODEL` | No | — | Local LLM model name. Shared with email-agent so both services use the same model. Referenced in `config.toml` as `{env.MLX_MODEL}`. |
| `MLX_API_KEY` | No | — | Local LLM API key. Empty string for real MLX; set for public API stand-ins like Novita.ai. |
| `USER_NAME` | No | — | User's display name, substituted into classification prompts via `{env.USER_NAME}` in `config.toml`. |
| `VIP_SENDERS` | No | — | Comma-separated email addresses of VIP senders. VIP threads skip the sender classification LLM call. |
| `EMAIL_LABELER_API_KEY` | No | — | Fallback API key for the api-proxy server, used when `PROXY_API_KEY` is not set. |
| `NEWSLETTER_ONLY` | No | — | Set to `1`, `true`, or `yes` to skip non-newsletter threads. Useful for testing newsletter classification in isolation. |
| `LOCAL_PARALLEL` | No | `1` (from `config.toml`) | Max concurrent local MLX requests, overriding `local_parallel` in `config.toml`. Modern MLX servers batch these (shared weights), so concurrency mostly costs KV cache. Keep ≤ 8 — mlx-lm has a KV-cache cross-contamination bug at 16+. |
| `MAX_EMAILS_PER_CYCLE` | No | `10` (from `config.toml`) | Max threads processed per poll cycle, overriding `max_emails_per_cycle` in `config.toml`. Raise temporarily to drain a large backlog faster. |
| `WRITE_PARALLEL` | No | `4` (from `config.toml`) | Max concurrent label-application writes (`modify_message`), overriding `write_parallel` in `config.toml`. Bounds the proxy-write burst when `max_emails_per_cycle` is large. Sized separately from reads because writes may block on human approval (`WRITE_TIMEOUT`, 300s). |

Note: The cloud LLM **model name** is configured in `config.toml` under `[llm.cloud]`, not in `.env`. The local LLM **model name** is set via the `MLX_MODEL` environment variable (shared with email-agent) and referenced in `config.toml` as `{env.MLX_MODEL}`. This keeps secrets (keys, URLs) in `.env` while operational parameters (temperature, prompts) stay in version-controlled `config.toml`.

## Configuration

All operational parameters are in `config.toml`. The daemon reads this file on startup.

### Daemon settings

```toml
[daemon]
poll_interval_seconds = 60     # How often to poll Gmail
max_emails_per_cycle = 10      # Max threads per poll (override: MAX_EMAILS_PER_CYCLE)
gmail_query = "in:inbox -label:agent/processed -label:agent/attempted"  # Gmail search query
max_thread_chars = 16000       # Cap on transcript chars sent to the classifier
cloud_parallel = 2             # Max concurrent cloud LLM requests
local_parallel = 1             # Max concurrent local MLX requests (override: LOCAL_PARALLEL)
fetch_parallel = 4             # Max concurrent Gmail thread fetches (get_thread)
write_parallel = 4             # Max concurrent label-application writes (override: WRITE_PARALLEL)
healthcheck_file = "/tmp/healthcheck"            # Healthcheck timestamp path
```

Threads found in a poll cycle are processed concurrently, bounded by the
`cloud_parallel` and `local_parallel` semaphores. **`local_parallel` defaults to 1**:
modern MLX servers do batch concurrent requests (weights loaded once, shared), but
each concurrent request still needs its own KV cache, and long email transcripts
make those caches multi-GB. On a memory-constrained Mac, a few concurrent
long-transcript requests can exceed the GPU's Metal working set (~75% of unified
memory) and OOM-crash the server. Raise `local_parallel` (via `LOCAL_PARALLEL`)
only once you've confirmed the model plus N concurrent KV caches fit the GPU
working set — tune the serving side too (`--prompt-cache-size`, and on macOS
`sudo sysctl iogpu.wired_limit_mb=...`). Keep it ≤ 8 regardless (mlx-lm KV-cache
cross-contamination bug at 16+). See [Local Model Serving & Memory](#local-model-serving--memory).

`max_thread_chars` caps the transcript fed to the classifier. It's deliberately
modest: the local model prefills the entire transcript before answering, so a
50k-char thread can take minutes and exceed the local request timeout — which, on
a stateless daemon, means the thread errors and is retried every cycle forever.
`max_emails_per_cycle`, `local_parallel`, and (via config) the LLM `timeout` are
the levers that keep a single huge thread from stalling the loop.

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
timeout = 180       # Local LLM gets more time (runs on consumer hardware)
```

### Extra request body fields

Each `[llm.*]` section supports an optional `extra_body` table. Any key-value pairs defined here are merged into every API request body sent to that endpoint. This is useful for provider-specific parameters that aren't part of the standard OpenAI chat completion format.

**Disabling thinking for reasoning models** — Models like Qwen3, DeepSeek-R1, and GLM-4.5 generate chain-of-thought reasoning in `<think>` tags before answering. While the daemon already strips these tags from responses, you can disable thinking entirely to save tokens and reduce latency.

The shipped `config.toml` **disables native thinking on the local person-email classifier** (`[llm.local.extra_body.chat_template_kwargs]` → `enable_thinking = false`). A paired `stage2_only --sender-type person` eval (n=20) found native thinking strictly worse here: the model spent its `max_tokens` budget reasoning in the `<think>` channel and emitted no label on some threads (a `KeyError: 'content'` failure), while think-off was ≥ think-on on every thread (85% vs 78% accuracy, 0 vs 2 errors). The classification prompt already drives step-by-step reasoning into the *content* channel, so disabling native thinking preserves reasoning quality without the budget-split failure. `mlx_lm.server` honors this request-level `chat_template_kwargs` form; a top-level `enable_thinking` is ignored by that server.

For providers that accept a top-level `enable_thinking` flag instead (Novita.ai, many OpenAI-compatible APIs):

```toml
[llm.local.extra_body]
enable_thinking = false
```

For LM Studio and `mlx_lm.server` with models that use `chat_template_kwargs` (e.g. Qwen3) — the shipped default:

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

## Local Model Serving & Memory

Person-email bodies are classified by a local MLX model (`MLX_URL`), typically
`mlx_lm.server`. On a memory-constrained Mac this is the most failure-prone part
of the stack, and the failures are non-obvious, so they're documented here.

### Starting the server

```bash
mlx_lm.server --model <mlx-model> --host 0.0.0.0 --port 8080 --temp 0 \
  --decode-concurrency 8 --prompt-cache-size 2
```

- `--host 0.0.0.0` — required to reach the server from another machine (e.g. over Tailscale). The default `127.0.0.1` accepts only localhost; a remote connection to a localhost-bound server is **reset**, surfacing in the daemon as a connection error.
- The request's `model` field must match the loaded `--model`, or mlx_lm.server returns `404 Not Found`. So `MLX_MODEL` must name the served model.
- `--prompt-cache-size N` bounds how many past request KV caches are retained for reuse. The default (10) accumulates several GB of KV across distinct emails (which share no prefix) and can exhaust GPU memory; **2 is recommended** for this workload.
- `--decode-concurrency` sizes the continuous-batching slots; it only matters when the daemon sends concurrent requests (`local_parallel` > 1).

### The GPU memory ceiling (why it OOM-crashes)

Apple Silicon caps the GPU's Metal working set at roughly **75% of unified memory** (~48 GB on a 64 GB Mac). Model weights + every live KV cache + prefill activation buffers must fit under that ceiling, not the full RAM. Exceeding it aborts the server:

```
[METAL] Command buffer execution failed: Insufficient Memory
  (kIOGPUCommandBufferCallbackErrorOutOfMemory)  ... SIGABRT
```

A 27B model at 8-bit is ~34 GB, leaving only ~14 GB of headroom. Long transcripts have multi-GB KV caches, so a few concurrent long-transcript requests (or a large retained prompt cache) blow the ceiling. Mitigations, in order of preference:

1. **Free system RAM** — leaked/idle processes shrink what the GPU can wire.
2. **`local_parallel = 1`** (the default) — one live KV cache at a time.
3. **`--prompt-cache-size 2`** — stop the retained cache piling up.
4. **`max_thread_chars`** — cap prefill size (also bounds latency).
5. **Raise the ceiling**: `sudo sysctl iogpu.wired_limit_mb=53248` (~52 GB on a 64 GB Mac; resets on reboot; don't starve macOS).

### Prefill latency and the request timeout

The model prefills the whole transcript before emitting a label, at ~100–200 tokens/sec on consumer hardware. A 50k-char (~17k-token) thread can take minutes — longer than `[llm.local] timeout` (default 180s) — so the client times out and the thread errors. On the stateless daemon that thread would otherwise be retried every cycle forever; **`max_thread_chars`** (cap the input) and the **`FailureTracker`** (give up after repeated failures, see `daemon.py`) are the two guards.

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

## TUI Conventions (Textual)

All interactive terminal UIs use [Textual](https://textual.textualize.io/) (a runtime
dependency; framework choice evaluated in issue #40, migration tracked in issue #43).
Conventions shared by every TUI:

- **Pure-helper split**: data transforms (filtering, row/detail formatting, state
  transitions, persistence) live in pure functions with direct unit tests; only the
  widget/screen layer is Textual-specific. UI behavior is tested with Textual's
  `Pilot` driver — real key presses in, widget state and rendered content out.
- **Shared widgets/screens** live in `tui_common.py` (e.g. `KeyMenuScreen`, the
  single-keypress menu that replaces the curses "press one key, anything else
  cancels" prompt idiom, and its `CANCEL` sentinel — distinct from a chosen value
  of `None`, which means "clear").
- **`markup=False` for all record-derived text**: bracketed content like
  `[f]ilter` or user text is otherwise parsed as Rich markup and silently
  swallowed (found via pty smoke test during the issue #40 spike).
- **Pilot test patterns**: `async with app.run_test(size=(100, 30)) as pilot:`,
  drive with `await pilot.press(...)`, assert on widget state
  (`app.query_one(...)`) and rendered content (`str(widget.render())`).
  `asyncio_mode = "auto"` is set, so Pilot tests are plain `async def` tests.
- **Async**: Textual apps are asyncio-native. Long-running work inside a UI
  (e.g. LLM calls) must be awaited or run in a Textual worker — never
  `asyncio.run()` inside an app, which raises `RuntimeError` in a running loop.

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
| `test_eval_newsletter_schemas.py` | `evals/newsletter_schemas.py` | Golden-set/result dataclass round-trips, missing-key tolerance |
| `test_eval_newsletter_harvest.py` | `evals/newsletter_harvest.py` | Newsletter filtering, body build, dedup, no ground-truth inference |
| `test_eval_newsletter_label.py` | `evals/newsletter_label.py` | Story curation + per-story scoring/theme pure functions, tier derivation, undo + Pilot UI tests (seed guard, undo stack, delete/label flows, selection, skip-through, autosave) |
| `test_eval_newsletter_run.py` | `evals/newsletter_run.py` | `prompt_hash`, cache reuse, extraction vs quality/theme modes |
| `test_eval_newsletter_report.py` | `evals/newsletter_report.py` | `match_stories`, tier/dimension/theme metrics, comparison deltas |
| `test_newsletter_review.py` | `newsletter_review/tui.py` | Pure helpers (loading, filtering, formatting) + Pilot UI tests (navigation, drill-down, tier/theme/sender filters, quit) |
