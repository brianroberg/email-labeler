# Email Labeler

Background daemon that continuously polls Gmail for unclassified emails and applies labels using a two-tier LLM classification system.

## Documentation

- `README.md` — Human-oriented overview: privacy model, architecture, setup instructions, running commands
- `README-technical.md` — Agent/reference: project structure, config.toml reference, environment variables, test coverage
- `evals/README.md` — Human-oriented eval suite guide: pipeline stages, common workflows, key commands
- `evals/README-technical.md` — Agent/reference: complete CLI flags for all eval tools, LLM cache internals, chain-of-thought capture format

## Privacy Invariant

Person email bodies NEVER leave the local network. Cloud LLM only sees metadata (sender, subject, snippet) for Stage 1 classification. Person email bodies are processed by local MLX/Qwen3.6 only.

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
- `newsletter_review/` — Textual TUI for browsing newsletter assessment results (`python -m newsletter_review`)
- `tui_common.py` — Shared Textual widgets/screens for all TUIs (see README-technical "TUI Conventions")

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
  → Score each story on 4 quality dimensions (simple, concrete, personal, dynamic) as Poor/OK/Good (Cloud LLM)
  → Grade each story against Ends Statement themes as Absent/Present/Emphasized (Cloud LLM)
  → Compute overall tier (excellent/good/fair/poor) from the averaged dimension scores (Poor/OK/Good → 1/2/3; excellent ≥ 2.75, good ≥ 2.25, fair ≥ 1.75, else poor)
  → Apply tier + theme labels via api-proxy → Gmail
  → Append assessment record to JSONL file
```

Newsletter uses its own `[newsletter.llm]` config (currently Sonnet 4.6) independent of the email classification LLM settings.

### Newsletter Labels (must be pre-created in Gmail)

- `agent/newsletter` — Marker (always applied)
- `agent/newsletter/excellent|good|fair|poor` — Overall quality tier
- `agent/newsletter/no-stories` — Newsletter contained no extractable stories
- `agent/newsletter/theme/*` — Per-story theme labels (scripture, christlikeness, church, vocation-family, disciple-making); applied **only when the theme is graded Emphasized** (merely-Present themes are recorded in the assessment JSONL but not labeled; Absent themes are omitted from the record entirely)

### Newsletter Review TUI

```bash
python -m newsletter_review                          # Browse all assessments
python -m newsletter_review --tier poor              # Filter by tier
python -m newsletter_review --theme scripture        # Filter by theme
python -m newsletter_review --sender dm.org          # Filter by sender
python -m newsletter_review --since 2026-01-01       # Filter to sends on/after a local date (YYYY-MM-DD)
python -m newsletter_review --file path/to/file.jsonl  # Custom JSONL path
```

The listing shows a send-date column and is sorted by send-date descending (newest
first; records with no send-date sort last).

Hotkeys: `f` opens the filter menu (`t` tier → `e/g/f/p/c`, `h` theme → `s/c/h/v/d/x`, `s` sender text input, `d` date → `3`=past 30d / `9`=past 90d / `y`=past 365d / `s`=since YYYY-MM-DD / `x`=clear), `Enter` opens detail, `Esc` back, `q` quit.

### Newsletter evaluation

Distinct from the read-only `newsletter_review/` package (which only *browses*
production assessments), the newsletter eval harness under `evals/` *measures* the
grading pipeline against hand-labeled ground truth so the prompts can be iterated.
It mirrors the email eval's 4 stages with `newsletter_`-prefixed modules:
`newsletter_harvest → newsletter_label → newsletter_run → newsletter_report`
(schemas in `evals/newsletter_schemas.py`). Quality/theme scoring uses **fixed
golden stories** so it is decoupled from extraction variability; the shared cache
(`evals/cache/llm_cache.jsonl`) and each run's `prompt_hash` make prompt A/Bs
cheap and self-identifying. See `evals/README.md` (workflows) and
`evals/README-technical.md` (every CLI flag, cache/prompt_hash details).

## Key Design Decisions

1. **Two-tier classification**: Stage 1 (cloud) determines person vs service. Stage 2 routes person bodies to local LLM only.
2. **Safe defaults**: Unknown sender type → SERVICE (body goes to cloud, safe for non-person). Unknown email label → LOW_PRIORITY (archived, not deleted).
3. **MLX degradation**: If local MLX is down, person emails are skipped (retried next cycle). Privacy invariant preserved.
4. **No web server**: Pure asyncio daemon. Health check via file timestamp + Docker HEALTHCHECK.
5. **Bounded-concurrency processing**: Threads in a poll cycle are processed concurrently, bounded by the `cloud_parallel`/`local_parallel` semaphores. `local_parallel` defaults to **1** (env override: `LOCAL_PARALLEL`): each concurrent local request needs its own KV cache, and long transcripts make those multi-GB, so concurrent requests can exceed the GPU's Metal working set and OOM-crash the local server. Raise it only after confirming the model + N KV caches fit; keep ≤ 8. See README-technical "Local Model Serving & Memory".

## Labels (must be pre-created in Gmail)

- `agent/needs-response` — Leave in inbox
- `agent/fyi` — Leave in inbox
- `agent/low-priority` — Archive
- `agent/processed` — Marker (applied on success / already-handled)
- `agent/attempted` — Marker applied on give-up (after repeated failures); excluded from `gmail_query` like `agent/processed`, but kept distinct so abandoned threads stay findable
- `agent/personal` — Sender classified as person (body processed locally)
- `agent/non-personal` — Sender classified as service (body processed via cloud)

## Environment Variables

- `PROXY_URL` — API proxy URL (default: `http://host.docker.internal:8000`)
- `PROXY_API_KEY` — API proxy authentication key
- `CLOUD_LLM_URL` — Cloud LLM endpoint (any OpenAI-compatible API)
- `CLOUD_LLM_API_KEY` — Cloud LLM API key
- `NEWSLETTER_LLM_URL` — Endpoint for the newsletter grading LLM (`[newsletter.llm]`); defaults to `CLOUD_LLM_URL`. Set when the newsletter model needs a different provider (e.g. a Claude model via Anthropic's OpenAI-compatible endpoint)
- `NEWSLETTER_LLM_API_KEY` — API key for `NEWSLETTER_LLM_URL`. The override is atomic: once `NEWSLETTER_LLM_URL` is set the key comes only from this var (never the cloud key), so set both together
- `MLX_URL` — Local MLX LLM endpoint
- `MLX_MODEL` — Local LLM model name (shared with email-agent, referenced in config.toml as `{env.MLX_MODEL}`)
- `MLX_API_KEY` — Local LLM API key (empty for real MLX, set for public API stand-ins like Novita.ai)
- `NEWSLETTER_ONLY` — When `1`/`true`/`yes`, daemon runs newsletter classification pipeline instead of email labeling
- `LOCAL_PARALLEL` — Override `local_parallel` (max concurrent local MLX requests; default 1, keep ≤ 8)
- `MAX_EMAILS_PER_CYCLE` — Override `max_emails_per_cycle` (max threads per poll cycle; default 10)
- `WRITE_PARALLEL` — Override `write_parallel` (max concurrent label-application writes; default 4). Sized separately from reads because writes may block on human approval (`WRITE_TIMEOUT`, 300s)

## Testing

**Always build using red/green TDD.** Every behavior change — new feature, bug fix,
or refactor — starts with a failing test:

1. **RED** — Write one minimal test for the new behavior. Run it and *watch it fail
   for the expected reason* (behavior missing, not a typo/import error). A test you
   never saw fail proves nothing.
2. **GREEN** — Write the minimal production code to make it pass. Run the test; confirm
   it passes and the rest of the suite stays green.
3. **REFACTOR** — Clean up with the tests green. Add no behavior without a new failing
   test first.

**No production code without a failing test first.** If you wrote code before the test,
delete it and start over from the test.

**Proving after-the-fact tests with mutation.** When a test is written *after* the code
(e.g. covering a fix already applied, or legacy code), it can't be red/green-verified —
so prove it instead: deliberately break the production behavior it targets (a "mutation")
and confirm the test fails, then revert. A test that stays green when you break the code
under it is not testing that code. Prefer the mutation that reproduces the exact bug the
test guards against.

**Always run the full test suite before declaring any task complete.**

```bash
uv run --extra dev pytest tests/ -v
```

All tests use mocks — no external services needed. Test files mirror source files:
- `test_llm_client.py` — LLM client request format, auth, think-tag stripping
- `test_classifier.py` — Parsing functions + classification routing
- `test_labeler.py` — Label verification + application actions
- `test_daemon.py` — Processing pipeline + config loading
- `test_newsletter_review.py` — TUI data loading, filtering, formatting + Pilot UI tests (navigation, drill-down, filters, quit)
- `test_eval_newsletter_schemas.py` — Golden-set dataclass round-trip + missing-key tolerance
- `test_eval_newsletter_harvest.py` — Newsletter harvest filtering, body build, dedup
- `test_eval_newsletter_label.py` — Story curation + per-story scoring/theme pure functions + Pilot UI tests (seeding, undo, labeling, skip-through)
- `test_eval_newsletter_run.py` — `prompt_hash`, cache reuse, extraction vs quality/theme modes
- `test_eval_newsletter_report.py` — `match_stories`, tier/dimension/theme metrics, comparison deltas
- `test_eval_newsletter_cli_docs.py` — Every newsletter eval `--flag` documented in `README-technical.md`
- `test_newsletter_eval_docs.py` — Newsletter eval modules/tests listed in `README-technical.md` structure + coverage
