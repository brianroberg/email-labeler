# Email Labeler

Background daemon that continuously polls Gmail for unclassified emails and applies labels using a two-tier LLM classification system.

## Documentation

- `README.md` — Human-oriented overview: privacy model, architecture, setup instructions, running commands
- `README-technical.md` — Agent/reference: project structure, config.toml reference, environment variables, test coverage
- `evals/README.md` — Human-oriented eval suite guide: pipeline stages, common workflows, key commands
- `evals/README-technical.md` — Agent/reference: complete CLI flags for all eval tools, LLM cache internals, chain-of-thought capture format

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
- `newsletter_review/` — Curses TUI for browsing newsletter assessment results (`python -m newsletter_review`)

## Architecture

```
Poll loop → find unprocessed emails
  → Stage 1: sender+subject+snippet → Cloud LLM → person or service?
  → Stage 2a (service): full body → Cloud LLM → classify
  → Stage 2b (person): full body → Local MLX → classify
  → Apply label + action via api-proxy → Gmail
```

## Newsletter Classification

When `NEWSLETTER_ONLY=1`, the daemon switches to a newsletter-specific pipeline that grades ministry newsletter stories on writing quality and thematic alignment.

```
Poll loop → find unprocessed newsletters (To/Cc matches config recipient)
  → Extract individual stories from newsletter body (Cloud LLM)
  → Score each story on 4 quality dimensions: simple, concrete, personal, dynamic (Cloud LLM)
  → Classify each story against Ends Statement themes (Cloud LLM)
  → Compute overall tier (excellent/good/fair/poor) from averaged scores
  → Apply tier + theme labels via api-proxy → Gmail
  → Append assessment record to JSONL file
```

Newsletter uses its own `[newsletter.llm]` config (currently Sonnet 4.6) independent of the email classification LLM settings.

### Newsletter Labels (must be pre-created in Gmail)

- `agent/newsletter` — Marker (always applied)
- `agent/newsletter/excellent|good|fair|poor` — Overall quality tier
- `agent/newsletter/no-stories` — Newsletter contained no extractable stories
- `agent/newsletter/theme/*` — Per-story theme labels (scripture, christlikeness, church, vocation-family, disciple-making)

### Newsletter Review TUI

```bash
python -m newsletter_review                          # Browse all assessments
python -m newsletter_review --tier poor              # Filter by tier
python -m newsletter_review --theme scripture        # Filter by theme
python -m newsletter_review --sender dm.org          # Filter by sender
python -m newsletter_review --file path/to/file.jsonl  # Custom JSONL path
```

Hotkeys: `e/g/f/p` filter by tier, `1-5` filter by theme, `s` filter by sender, `c` clear filters, `q` quit.

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
- `agent/processed` — Marker (always applied)
- `agent/personal` — Sender classified as person (body processed locally)
- `agent/non-personal` — Sender classified as service (body processed via cloud)

## Environment Variables

- `PROXY_URL` — API proxy URL (default: `http://host.docker.internal:8000`)
- `PROXY_API_KEY` — API proxy authentication key
- `CLOUD_LLM_URL` — Cloud LLM endpoint (any OpenAI-compatible API)
- `CLOUD_LLM_API_KEY` — Cloud LLM API key
- `MLX_URL` — Local MLX LLM endpoint
- `MLX_MODEL` — Local LLM model name (shared with email-agent, referenced in config.toml as `{env.MLX_MODEL}`)
- `MLX_API_KEY` — Local LLM API key (empty for real MLX, set for public API stand-ins like Novita.ai)
- `NEWSLETTER_ONLY` — When `1`/`true`/`yes`, daemon runs newsletter classification pipeline instead of email labeling

## Testing

**Always run the full test suite before declaring any task complete.**

```bash
uv run --extra dev pytest tests/ -v
```

All tests use mocks — no external services needed. Test files mirror source files:
- `test_llm_client.py` — LLM client request format, auth, think-tag stripping
- `test_classifier.py` — Parsing functions + classification routing
- `test_labeler.py` — Label verification + application actions
- `test_daemon.py` — Processing pipeline + config loading
- `test_newsletter_review.py` — TUI data loading, filtering, formatting, detail view
