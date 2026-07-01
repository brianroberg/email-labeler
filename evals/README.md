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

Pulls threads already labeled by the daemon, infers ground truth from their Gmail labels, and appends to JSONL.

Harvest always **appends** to the output file, deduplicating by thread ID. It never overwrites an existing golden set — that file also holds your manual review state (confirmed labels, exclusions, notes). To start fresh, delete the file manually.

```bash
# Harvest up to 200 processed threads (appends new ones, dedupes automatically)
uv run python -m evals.harvest --proxy-url http://localhost:8000 --max-threads 200

# Filter by sender type or label
uv run python -m evals.harvest --proxy-url http://localhost:8000 --sender-type person
uv run python -m evals.harvest --proxy-url http://localhost:8000 --label needs_response
```

`--label` takes the config **key** (`needs_response`, `fyi`, `low_priority`) — not the Gmail label name. Harvest translates the key to the actual Gmail label via the `[labels]` section of `config.toml` (e.g. `needs_response` → `agent/needs-response`), so pass `--label needs_response`, not `--label agent/needs-response`.

`--label` is ANDed into the Gmail query, so the fetch targets matching threads
directly rather than filtering the recent processed window after the fact. This
is the way to boost a rare class (e.g. `needs_response`) in the golden set:
harvested threads still carry their inferred labels into `evals.review` for
manual confirmation, so the manual classification step is not bypassed.

## 2. Review — Manually verify ground truth labels

Interactive CLI for reviewing and correcting labels in the golden set. Saves atomically after each session. Press `z` at any prompt to undo the last classification — undo works as a stack, walking back through previous decisions.

**Setting an email aside.** Some threads aren't useful as test cases. Two options:

- **Skip** (`k`) — render no judgment. The thread is left unreviewed and resurfaces in a later review session.
- **Exclude** (`e`) — permanently set the thread aside. It is dropped from the review queue and never evaluated. Excluded threads are flagged with an `X` in the `--edit` list view; to bring one back, open `--edit`, select it, and press `e` to un-exclude.

Hotkeys: sender type `p`/`s` (person/service); label `r`/`f`/`l` (needs_response/fyi/low_priority); `n` notes; `z` undo; `k` skip; `e` exclude; `q` quit.

```bash
# Review all threads (blind mode by default)
uv run python -m evals.review

# Show existing labels while reviewing
uv run python -m evals.review --show-labels

# Curses TUI for editing reviewed threads (also where you un-exclude)
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

## Newsletter evaluation

The daemon's newsletter pipeline (active under `NEWSLETTER_ONLY=1`) grades
ministry-newsletter stories: it **extracts** stories from a body, **scores** each
on 4 quality dimensions (simple/concrete/personal/dynamic, 1–5) → a **tier**
(excellent/good/fair/poor), and tags **themes** (scripture/christlikeness/church/
vocation-family/disciple-making). This harness measures whether that grading is any
good, so you can **iterate on the prompts** and see the effect of each change.

It mirrors the email eval's 4 stages, with newsletter-specific modules prefixed
`newsletter_` (distinct from the top-level read-only `newsletter_review/` TUI):

```
newsletter_harvest → newsletter_label → newsletter_run → newsletter_report
```

- **harvest** — Pull candidate newsletters from Gmail into
  `evals/newsletter_golden_set.jsonl`. Each is guarded by `is_newsletter(...)` and
  its `body` is built exactly like production input. **No ground truth is inferred** —
  newsletters land unlabeled with an empty story list.
- **label** — A curses/CLI tool to build ground truth by hand (quality is
  subjective, so there are no auto-labels). Phase A curates the story list
  (seeded from the production extractor as a starting point, then the reviewer
  marks body segments — `s`/`e` to set the span, `Enter` to make a story — plus
  add/edit/delete, or `k` to skip the newsletter for a later pass); Phase B
  assigns per-story dimension scores + themes. The tier is auto-derived from scores.
- **run** — Replay the golden set through the real classifier (LLM-cached) and
  write timestamped results.
- **report** — Compute tier/dimension/theme/extraction metrics, compare two runs,
  or show a trend across runs.

**Fixed golden stories.** Extraction is inherently variable, so quality and theme
scoring is *decoupled* from it: one golden set holds both. Extraction is scored at
the newsletter-body level (raw body → predicted stories → matched against the
newsletter's golden story list). Quality + themes are scored on the **fixed golden
`(title, text)`** of each reviewed story, independent of what extraction produced —
so a prompt tweak that only affects scoring is measured cleanly.

```bash
# 1. Harvest a few newsletters (no ground truth yet)
uv run python -m evals.newsletter_harvest --proxy-url http://localhost:8000 --max-threads 50

# 2. Label them: curate stories, then score dimensions + themes
uv run python -m evals.newsletter_label
uv run python -m evals.newsletter_label --unreviewed-only   # only unlabeled
uv run python -m evals.newsletter_label --edit              # revisit reviewed ones

# 3. Run the whole pipeline and print a report
uv run python -m evals.newsletter_run --mode all --tag baseline --report

# 4. Report on a single run, or a trend
uv run python -m evals.newsletter_report --results evals/newsletter_results/<run>.jsonl
uv run python -m evals.newsletter_report --results-dir evals/newsletter_results/
```

**Build a golden set:**

```bash
uv run python -m evals.newsletter_harvest --proxy-url http://localhost:8000
uv run python -m evals.newsletter_label --unreviewed-only
```

**Iterate a prompt variant:**

```bash
uv run python -m evals.newsletter_run --tag baseline
# Edit a [newsletter.prompts.*] block in config.toml (or use --prompts alt.toml), then:
uv run python -m evals.newsletter_run --prompts alt.toml --tag variant
```

The shared cache (`evals/cache/llm_cache.jsonl`) is keyed by prompt content, so
unchanged stages are reused and only the changed prompt is re-run. Each run records
a `prompt_hash`, so variants are self-identifying.

**Compare two runs:**

```bash
uv run python -m evals.newsletter_report \
    --compare evals/newsletter_results/*baseline*.jsonl evals/newsletter_results/*variant*.jsonl --verbose
```

The comparison renders tier/dimension/theme/extraction deltas and prints each run's
`prompt_hash` + tag; `--verbose` lists per-story flips and per-newsletter extraction
diffs.

## Typical Workflows

**Evaluate a new local model (fast path):**

The local model is only used for Stage 2 on person bodies, so `--local-only`
(= `--stages stage2_only --sender-type person`) isolates it — and needs no cloud
credentials. The one-command wrapper sets the model, auto-tags the run by the
model name, preflights the endpoint, prints a report, and optionally compares to
a prior run:

```bash
# Prereq: mlx_lm.server is already serving <hf-id>, and MLX_URL points at it.
python scripts/eval_model.py qwen/qwen3-14b                 # run + report
python scripts/eval_model.py qwen/qwen3-14b qwen3-8b        # ...also compare vs the newest qwen3-8b run
```

Equivalent explicit form (the wrapper just chains these):

```bash
uv run python -m evals.run_eval --local-only --local-model qwen/qwen3-14b --report
```

`--local-only` errors if the local endpoint is unreachable or `MLX_MODEL` doesn't
match the served model (a 404), instead of producing a whole run of per-thread
errors. The check waits up to the local model's request timeout, so a server that
loads the model on demand has time to cold-load it; tune with `--preflight-timeout
SECONDS`, or skip it entirely with `--skip-preflight`.

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
uv run python -m evals.harvest --proxy-url http://localhost:8000
uv run python -m evals.review --unreviewed-only
uv run python -m evals.run_eval --tag weekly
uv run python -m evals.report --results-dir evals/results/
```
