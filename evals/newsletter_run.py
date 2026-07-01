"""Replay the newsletter golden set through the real NewsletterClassifier.

Usage:
    python -m evals.newsletter_run
    python -m evals.newsletter_run --golden-set evals/newsletter_golden_set.jsonl --mode all
    python -m evals.newsletter_run --prompts alt.toml --tag variant
    python -m evals.newsletter_run --mode quality --no-cache

Mirrors evals/run_eval.py: load config (tomllib + env substitution), resolve the
newsletter endpoint, build one LLMClient from [newsletter.llm], wrap in the disk
LLM cache unless --no-cache, preflight the endpoint, then replay the golden set.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config_utils import substitute_env_vars
from daemon import resolve_newsletter_llm_endpoint
from evals import format_network_error
from evals.llm_cache import CachedLLMClient
from evals.newsletter_schemas import (
    ExtractionPrediction,
    GoldenNewsletter,
    GoldenStory,
    NewsletterRunMeta,
    NewsletterThinkingEntry,
    StoryPrediction,
)
from llm_client import LLMClient
from newsletter import NewsletterClassifier, compute_tier


def load_golden_set(path: Path, reviewed_only: bool = True) -> list[GoldenNewsletter]:
    """Load newsletters from the golden set JSONL.

    Drops ``excluded`` newsletters unconditionally; drops unreviewed ones unless
    ``reviewed_only`` is False. Blank lines are skipped.
    """
    newsletters = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gn = GoldenNewsletter.from_dict(json.loads(line))
            if gn.excluded:
                continue
            if reviewed_only and not gn.reviewed:
                continue
            newsletters.append(gn)
    return newsletters


def compute_prompt_hash(config: dict) -> str:
    """Content-hash the newsletter prompt block so prompt A/Bs are self-identifying.

    Hashes config["newsletter"]["prompts"] deterministically (sorted keys) — any
    change to a prompt string (system or user_template) changes the hash, and an
    identical prompt block always hashes the same.
    """
    prompts = config["newsletter"]["prompts"]
    raw = json.dumps(prompts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class _RawCapture:
    """Wraps an LLM client to record the raw (non-thinking) response of each call.

    The NewsletterClassifier parses scores/themes internally and only returns the
    parsed result + chain-of-thought, discarding the raw model text. To store the
    raw text in a StoryPrediction (for debugging a parse failure) without a second
    LLM call, temporarily route the classifier's client through this recorder — it
    delegates every call to the inner client and keeps the last response.
    """

    def __init__(self, inner):
        self.inner = inner
        self.last_raw = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def complete(self, system_prompt, user_content, include_thinking=False):
        result = await self.inner.complete(system_prompt, user_content, include_thinking=True)
        response, thinking = result
        self.last_raw = response
        if include_thinking:
            return response, thinking
        return response


def _elapsed(llm, start: float) -> float:
    """Actual LLM call time when caching (0 for pure cache hits), else wall time.

    Mirrors run_eval: a CachedLLMClient exposes take_llm_seconds() (miss-only
    time); an un-cached raw client falls back to wall-clock.
    """
    if hasattr(llm, "take_llm_seconds"):
        return round(llm.take_llm_seconds(), 3)
    return round(time.monotonic() - start, 3)


async def evaluate_extraction(
    newsletter: GoldenNewsletter, classifier: NewsletterClassifier,
) -> ExtractionPrediction:
    """Run extract_stories on the raw body; compare predicted vs golden story list.

    Extraction is scored at the newsletter-body level: the raw body goes in, and
    the predicted [{title,text}] list is matched (in the report) against the
    newsletter's confirmed golden stories.
    """
    pred = ExtractionPrediction(
        thread_id=newsletter.thread_id,
        golden_stories=[
            {"story_id": s.story_id, "title": s.title, "text": s.text}
            for s in newsletter.stories
        ],
    )
    start = time.monotonic()
    try:
        stories = await classifier.extract_stories(newsletter.body)
        pred.predicted_stories = [{"title": t, "text": x} for t, x in stories]
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        pred.error = format_network_error(exc, "newsletter LLM")
    except Exception as exc:
        pred.error = str(exc)
    pred.duration_seconds = _elapsed(classifier.cloud_llm, start)
    return pred


async def evaluate_story(
    story: GoldenStory, thread_id: str, classifier: NewsletterClassifier,
) -> tuple[StoryPrediction, NewsletterThinkingEntry]:
    """Score one fixed golden story on quality + themes; derive tier from scores.

    Quality and themes are evaluated on the fixed golden (title, text), decoupled
    from extraction variability. predicted_tier is derived via compute_tier only
    when the scores parsed; a parse failure leaves scores/tier None (an error in
    the report, not a mis-tier).
    """
    pred = StoryPrediction(
        story_id=story.story_id,
        thread_id=thread_id,
        expected_scores=story.expected_scores,
        expected_tier=story.expected_tier,
        expected_themes=list(story.expected_themes),
    )
    thinking = NewsletterThinkingEntry(story_id=story.story_id)

    real_client = classifier.cloud_llm
    capture = _RawCapture(real_client)
    # Shallow-copy the classifier so swapping the client is concurrency-safe: under
    # parallelism>1, mutating the shared classifier's cloud_llm would race between
    # tasks. The copy shares the (immutable) config dicts but owns its client ref.
    local_classifier = copy.copy(classifier)
    local_classifier.cloud_llm = capture
    start = time.monotonic()
    try:
        capture.last_raw = None
        scores, quality_cot = await local_classifier.assess_quality(story.title, story.text)
        pred.scores_raw = capture.last_raw
        pred.predicted_scores = scores
        thinking.quality_cot = quality_cot
        if scores:
            pred.predicted_tier = compute_tier(scores).value

        capture.last_raw = None
        themes, theme_cot = await local_classifier.classify_themes(story.title, story.text)
        pred.themes_raw = capture.last_raw
        pred.predicted_themes = themes
        thinking.theme_cot = theme_cot
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        pred.error = format_network_error(exc, "newsletter LLM")
    except Exception as exc:
        pred.error = str(exc)
    pred.duration_seconds = _elapsed(real_client, start)
    return pred, thinking


VALID_MODES = ("extraction", "quality", "themes", "all")


async def run_evaluation(
    golden_set: list[GoldenNewsletter],
    classifier: NewsletterClassifier,
    mode: str,
    parallelism: int = 1,
) -> tuple[list, list[NewsletterThinkingEntry]]:
    """Replay the golden set through the classifier under *mode*.

    Returns (rows, thinking_entries). ``rows`` mixes ExtractionPrediction (one per
    reviewed newsletter, extraction/all modes) and StoryPrediction (one per
    reviewed, non-excluded golden story, quality/themes/all modes). Concurrency is
    bounded by an asyncio.Semaphore(parallelism); the shared classifier's client is
    never mutated (evaluate_story shallow-copies it), so tasks don't race.
    """
    semaphore = asyncio.Semaphore(parallelism)
    do_extraction = mode in ("extraction", "all")
    do_stories = mode in ("quality", "themes", "all")

    async def run_extraction(nl: GoldenNewsletter):
        async with semaphore:
            return await evaluate_extraction(nl, classifier)

    async def run_story(story: GoldenStory, thread_id: str):
        async with semaphore:
            return await evaluate_story(story, thread_id, classifier)

    extraction_tasks = []
    if do_extraction:
        extraction_tasks = [run_extraction(nl) for nl in golden_set]

    story_tasks = []
    if do_stories:
        for nl in golden_set:
            for story in nl.stories:
                if story.excluded:
                    continue
                story_tasks.append(run_story(story, nl.thread_id))

    extraction_rows = await asyncio.gather(*extraction_tasks)
    story_pairs = await asyncio.gather(*story_tasks)

    rows = list(extraction_rows)
    thinking = []
    for pred, entry in story_pairs:
        rows.append(pred)
        thinking.append(entry)
    return rows, thinking


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in place (override wins on leaves)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def build_meta(
    config: dict,
    config_path: str,
    golden_path: str,
    golden_set: list[GoldenNewsletter],
    mode: str,
    tag: str,
    parallelism: int,
) -> "NewsletterRunMeta":
    """Assemble the NewsletterRunMeta first-line record for a results file.

    Records prompt_hash (content-hash of the merged prompt block) + the three
    system prompts verbatim + model/temperature/max_tokens/extra_body/counts/tag/
    mode so prompt A/Bs are self-identifying and comparable. seeded_from is pulled
    from the golden set (first newsletter that carries it) so the report can flag
    extraction-recall bias from Phase-A seeding.
    """
    llm_cfg = config["newsletter"]["llm"]
    prompts = config["newsletter"]["prompts"]
    config_bytes = json.dumps(config, sort_keys=True).encode()
    story_count = sum(
        1 for nl in golden_set for s in nl.stories if not s.excluded
    )
    seeded_from = next((nl.seeded_from for nl in golden_set if nl.seeded_from), "")
    return NewsletterRunMeta(
        run_id=uuid.uuid4().hex,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=hashlib.sha256(config_bytes).hexdigest()[:16],
        config_path=config_path,
        newsletter_model=llm_cfg["model"],
        golden_set_path=golden_path,
        golden_set_count=len(golden_set),
        story_count=story_count,
        mode=mode,
        prompt_hash=compute_prompt_hash(config),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 0),
        extra_body=llm_cfg.get("extra_body"),
        parallelism=parallelism,
        tag=tag,
        seeded_from=seeded_from,
        extraction_system_prompt=prompts["story_extraction"]["system"],
        quality_system_prompt=prompts["quality_assessment"]["system"],
        theme_system_prompt=prompts["theme_classification"]["system"],
    )


def build_output_path(output_dir: Path, mode: str, tag: str, run_id: str) -> Path:
    """Build the timestamped results file path: <ts>_<mode>_<tag>_<runid8>.jsonl."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, mode]
    if tag:
        parts.append(tag)
    parts.append(run_id[:8])
    return output_dir / ("_".join(parts) + ".jsonl")


def write_results(path: Path, meta: "NewsletterRunMeta", rows: list) -> None:
    """Write the run meta (first line) then each prediction row to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def write_thinking_sidecar(
    results_path: Path, thinking_entries: list[NewsletterThinkingEntry],
) -> None:
    """Write non-empty chain-of-thought entries to the .cot.jsonl sidecar.

    Only entries with at least one non-empty cot field are written; if none
    qualify, no sidecar file is created.
    """
    entries = [t for t in thinking_entries if t.quality_cot or t.theme_cot]
    if not entries:
        return
    sidecar_path = results_path.with_suffix(".cot.jsonl")
    with open(sidecar_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")


def merge_prompts_override(config: dict, override: dict) -> None:
    """Deep-merge only [newsletter.prompts.*] from *override* onto *config*, in place.

    A --prompts alt.toml may declare other sections, but only its newsletter prompt
    blocks are applied (everything else — model, recipient — is ignored) so a prompt
    variant file can't silently change the model or other run parameters. Must run
    BEFORE compute_prompt_hash so the recorded hash reflects the merged prompts.
    """
    prompts_override = override.get("newsletter", {}).get("prompts")
    if not prompts_override:
        return
    base_prompts = config.setdefault("newsletter", {}).setdefault("prompts", {})
    _deep_merge(base_prompts, prompts_override)


def build_classifier(config: dict, no_cache: bool) -> tuple[NewsletterClassifier, object]:
    """Build the LLM client (cached unless *no_cache*) and the classifier.

    Resolves the newsletter endpoint via daemon.resolve_newsletter_llm_endpoint()
    (falls back to the cloud endpoint) and builds one LLMClient from
    [newsletter.llm], mirroring daemon.run_daemon(). Returns (classifier, llm) so
    the caller can flush()/report cache stats on the wrapping client.
    """
    llm_cfg = config["newsletter"]["llm"]
    base_url, api_key = resolve_newsletter_llm_endpoint()
    inner = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=llm_cfg["model"],
        max_tokens=llm_cfg.get("max_tokens", 1024),
        temperature=llm_cfg.get("temperature", 0),
        timeout=llm_cfg.get("timeout", 60),
        extra_body=llm_cfg.get("extra_body"),
    )
    if no_cache:
        llm = inner
    else:
        cache_path = Path(__file__).parent / "cache" / "llm_cache.jsonl"
        llm = CachedLLMClient(inner, cache_path)
    classifier = NewsletterClassifier(cloud_llm=llm, config=config)
    return classifier, llm


def _split_rows(rows: list) -> tuple[list[StoryPrediction], list[ExtractionPrediction]]:
    """Split a mixed prediction row list into (story_preds, extraction_preds)."""
    story = [r for r in rows if isinstance(r, StoryPrediction)]
    extraction = [r for r in rows if isinstance(r, ExtractionPrediction)]
    return story, extraction


def maybe_report(meta, rows, report_enabled: bool, compare_to: str | None) -> None:
    """Print a metrics report (and optional comparison) for a just-finished run.

    A no-op unless --report or --compare-to was passed. Defers to
    evals.newsletter_report; prints a hint if it isn't importable so a missing
    report module never crashes a completed run.
    """
    if not report_enabled and not compare_to:
        return
    try:
        from evals import newsletter_report
    except ImportError:
        print("Note: evals.newsletter_report is not available; skipping --report.",
              file=sys.stderr)
        return

    story_preds, extraction_preds = _split_rows(rows)
    metrics = newsletter_report.compute_all_metrics(story_preds, extraction_preds)
    if report_enabled:
        newsletter_report.print_report(
            meta, metrics,
            story_results=story_preds, extraction_results=extraction_preds,
        )
    if compare_to:
        compare_path = Path(compare_to)
        if not compare_path.exists():
            print(f"Error: --compare-to file not found: {compare_path}", file=sys.stderr)
            return
        meta_b, story_b, extraction_b = newsletter_report.load_results(compare_path)
        metrics_b = newsletter_report.compute_all_metrics(story_b, extraction_b)
        # Prior run is Run A, the just-finished run is Run B (chronological order).
        newsletter_report.print_comparison(
            meta_b, metrics_b, meta, metrics,
            verbose=True, story1=story_b, story2=story_preds,
        )


async def main(args: argparse.Namespace) -> None:
    # Load config (tomllib + env substitution), mirroring run_eval.
    config_path = (
        Path(args.config) if args.config
        else Path(__file__).parent.parent / "config.toml"
    )
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    config = substitute_env_vars(config)

    # --prompts deep-merges alternate [newsletter.prompts.*] BEFORE prompt_hash and
    # classifier are computed, so the run measures (and records the hash of) the
    # merged prompts. --model overrides the newsletter model.
    if args.prompts:
        with open(args.prompts, "rb") as f:
            override = tomllib.load(f)
        override = substitute_env_vars(override)
        merge_prompts_override(config, override)
    if args.model:
        config["newsletter"]["llm"]["model"] = args.model

    parallelism = args.parallelism if args.parallelism is not None else 1

    # Load golden set
    golden_path = Path(args.golden_set)
    golden_set = load_golden_set(golden_path, reviewed_only=not args.include_unreviewed)
    if not golden_set:
        print("No newsletters to evaluate.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(golden_set)} newsletters from {golden_path}", file=sys.stderr)

    classifier, llm = build_classifier(config, args.no_cache)

    # Preflight: probe the newsletter endpoint once and fail fast on an
    # unreachable / name-mismatched one (skippable via --skip-preflight).
    if not args.skip_preflight:
        if not await llm.is_available():
            base = config["newsletter"]["llm"]
            print(
                f"Error: newsletter LLM '{base['model']}' not reachable "
                "(a model-name mismatch with the served model returns 404)",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        print(f"Running newsletter evaluation (mode={args.mode}, "
              f"parallelism={parallelism})...", file=sys.stderr)
        rows, thinking_entries = await run_evaluation(
            golden_set=golden_set,
            classifier=classifier,
            mode=args.mode,
            parallelism=parallelism,
        )

        meta = build_meta(
            config=config,
            config_path=str(config_path),
            golden_path=str(golden_path),
            golden_set=golden_set,
            mode=args.mode,
            tag=args.tag or "",
            parallelism=parallelism,
        )

        output_dir = Path(args.output_dir)
        output_path = build_output_path(output_dir, args.mode, args.tag or "", meta.run_id)
        write_results(output_path, meta, rows)
        write_thinking_sidecar(output_path, thinking_entries)

        maybe_report(meta, rows, args.report, args.compare_to)

        errors = sum(1 for r in rows if getattr(r, "error", None))
        print(f"\nResults written to {output_path}", file=sys.stderr)
        print(f"  Rows: {len(rows)} ({errors} errors)", file=sys.stderr)
        if not args.no_cache and hasattr(llm, "hits"):
            total = llm.hits + llm.misses
            if total:
                print(f"  Cache: {llm.hits}/{total} hits ({llm.hits / total:.1%})",
                      file=sys.stderr)
    finally:
        if not args.no_cache and hasattr(llm, "flush"):
            llm.flush()


def cli():
    parser = argparse.ArgumentParser(description="Run newsletter classification evaluation")
    parser.add_argument("--golden-set", default="evals/newsletter_golden_set.jsonl",
                        help="Path to newsletter golden set JSONL")
    parser.add_argument("--config", help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--output-dir", default="evals/newsletter_results/",
                        help="Output directory for results")
    parser.add_argument("--mode", choices=VALID_MODES, default="all",
                        help="Which outputs to evaluate")
    parser.add_argument("--tag", help="Tag for the results file name")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response cache")
    parser.add_argument("--parallelism", type=int, default=None,
                        help="Concurrent evaluations (default: 1)")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Also evaluate newsletters not yet reviewed (default: reviewed only)")
    parser.add_argument("--prompts", metavar="TOML",
                        help="Deep-merge [newsletter.prompts.*] from this TOML over the base config")
    parser.add_argument("--model", help="Override the newsletter LLM model name")
    parser.add_argument("--report", action="store_true",
                        help="Print a metrics report for this run after it completes")
    parser.add_argument("--compare-to", metavar="PATH",
                        help="After the run, print a comparison against this prior results file")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the endpoint reachability check before evaluating")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
