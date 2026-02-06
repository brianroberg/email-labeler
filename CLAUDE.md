# Email Labeler

Background daemon that continuously polls Gmail for unclassified emails and applies labels using a two-tier LLM classification system.

## Privacy Invariant

Person email bodies NEVER leave the local network. Cloud LLM only sees metadata (sender, subject, snippet) for Stage 1 classification. Person email bodies are processed by local MLX/Qwen3 only.

## Package Management

Uses `uv` for dependency management. No pip.

```bash
uv sync --extra dev          # Install all deps including dev
uv run --extra dev pytest    # Run tests
uv run --extra dev ruff check .  # Lint
```

## Project Structure

- `daemon.py` — Main entry point: polling loop + orchestration
- `classifier.py` — Two-tier classification logic + parsing
- `labeler.py` — Label verification + application
- `llm_client.py` — LLM abstraction (cloud + local endpoints)
- `proxy_client.py` — Gmail API proxy client (copied from email-agent)
- `gmail_utils.py` — Header/body parsing (copied from email-agent)
- `config.toml` — Label definitions, prompts, operational params

## Architecture

```
Poll loop → find unprocessed emails
  → Stage 1: sender+subject+snippet → Cloud LLM → person or service?
  → Stage 2a (service): full body → Cloud LLM → classify
  → Stage 2b (person): full body → Local MLX → classify
  → Apply label + action via api-proxy → Gmail
```

## Key Design Decisions

1. **Two-tier classification**: Stage 1 (cloud) determines person vs service. Stage 2 routes person bodies to local LLM only.
2. **Safe defaults**: Unknown sender type → SERVICE (body goes to cloud, safe for non-person). Unknown email label → LOW_PRIORITY (archived, not deleted).
3. **MLX degradation**: If local MLX is down, person emails are skipped (retried next cycle). Privacy invariant preserved.
4. **No web server**: Pure asyncio daemon. Health check via file timestamp + Docker HEALTHCHECK.
5. **Sequential processing**: Emails processed one at a time per poll cycle.

## Labels (must be pre-created in Gmail)

- `agent/needs-response` — Leave in inbox
- `agent/fyi` — Leave in inbox
- `agent/low-priority` — Archive
- `agent/unwanted` — Archive + apply `agent/would-have-deleted`
- `agent/processed` — Marker (always applied)
- `agent/personal` — Sender classified as person (body processed locally)
- `agent/non-personal` — Sender classified as service (body processed via cloud)
- `agent/would-have-deleted` — Extra marker for unwanted

## Environment Variables

- `PROXY_URL` — API proxy URL (default: `http://host.docker.internal:8000`)
- `PROXY_API_KEY` — API proxy authentication key
- `CLOUD_LLM_URL` — Cloud LLM endpoint (any OpenAI-compatible API)
- `CLOUD_LLM_API_KEY` — Cloud LLM API key
- `MLX_URL` — Local MLX LLM endpoint

## Testing

```bash
uv run --extra dev pytest tests/ -v
```

All tests use mocks — no external services needed. Test files mirror source files:
- `test_llm_client.py` — LLM client request format, auth, think-tag stripping
- `test_classifier.py` — Parsing functions + classification routing
- `test_labeler.py` — Label verification + application actions
- `test_daemon.py` — Processing pipeline + config loading
