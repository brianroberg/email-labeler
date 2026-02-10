"""Replay golden set through the real EmailClassifier with real LLM endpoints.

Usage:
    python -m evals.run_eval
    python -m evals.run_eval --golden-set evals/golden_set.jsonl --stages full
    python -m evals.run_eval --config config_v2.toml --tag new-prompts
    python -m evals.run_eval --stages stage1_only --include-unreviewed
    python -m evals.run_eval --dry-run
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from classifier import EmailClassifier, SenderType, ThreadMetadata
from config_utils import substitute_env_vars
from daemon import format_thread_transcript
from evals import format_network_error
from evals.llm_cache import CachedLLMClient
from evals.schemas import GoldenThread, PredictionResult, RunMeta
from gmail_utils import get_header
from llm_client import LLMClient

VALID_STAGES = ("full", "stage1_only", "stage2_only")


def load_golden_set(path: Path, reviewed_only: bool = False) -> list[GoldenThread]:
    """Load golden set from JSONL."""
    threads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt = GoldenThread.from_dict(json.loads(line))
                if gt.skipped:
                    continue
                if reviewed_only and not gt.reviewed:
                    continue
                threads.append(gt)
    return threads


def reconstruct_thread_metadata(golden: GoldenThread) -> ThreadMetadata:
    """Reconstruct ThreadMetadata from a golden set entry.

    Mirrors daemon.py:137-160 — sort by internalDate, extract unique senders,
    first-message subject, last-message snippet.
    """
    messages = sorted(golden.messages, key=lambda m: int(m.get("internalDate", "0")))

    senders = []
    seen: set[str] = set()
    for msg in messages:
        sender = get_header(msg["payload"]["headers"], "From")
        if sender and sender not in seen:
            senders.append(sender)
            seen.add(sender)

    if not senders:
        senders = golden.senders  # Fallback to harvested senders

    first_headers = messages[0]["payload"]["headers"] if messages else []
    subject = get_header(first_headers, "Subject") if first_headers else golden.subject
    snippet = messages[-1].get("snippet", "") if messages else golden.snippet

    return ThreadMetadata(
        thread_id=golden.thread_id,
        senders=senders,
        subject=subject,
        snippet=snippet,
    )


def reconstruct_transcript(golden: GoldenThread, max_chars: int) -> str:
    """Reconstruct thread transcript using daemon.format_thread_transcript()."""
    messages = sorted(golden.messages, key=lambda m: int(m.get("internalDate", "0")))
    return format_thread_transcript(messages, max_chars)


async def evaluate_single(
    golden: GoldenThread,
    classifier: EmailClassifier,
    stages: str,
    max_thread_chars: int,
) -> PredictionResult:
    """Evaluate a single golden thread.

    Args:
        golden: Golden set entry with ground truth.
        classifier: Real EmailClassifier instance.
        stages: "full", "stage1_only", or "stage2_only".
        max_thread_chars: Max chars for thread transcript.

    Returns:
        PredictionResult with predictions and correctness flags.
    """
    result = PredictionResult(
        thread_id=golden.thread_id,
        expected_sender_type=golden.expected_sender_type,
        expected_label=golden.expected_label,
    )

    metadata = reconstruct_thread_metadata(golden)
    transcript = reconstruct_transcript(golden, max_thread_chars)

    start = time.monotonic()
    try:
        if stages == "stage1_only":
            sender_type, sender_raw = await classifier.classify_sender(metadata)
            result.predicted_sender_type = sender_type.value
            result.predicted_sender_type_raw = sender_raw
            result.sender_type_correct = result.predicted_sender_type == golden.expected_sender_type
            result.privacy_violation = (
                golden.expected_sender_type == "person" and result.predicted_sender_type == "service"
            )

        elif stages == "stage2_only":
            # Use expected sender type as input (skip Stage 1)
            sender_type = SenderType(golden.expected_sender_type)
            label, label_raw = await classifier.classify_email(metadata, transcript, sender_type)
            result.predicted_sender_type = golden.expected_sender_type  # Passed through
            result.predicted_label = label.value
            result.predicted_label_raw = label_raw
            result.sender_type_correct = True  # By definition
            result.label_correct = result.predicted_label == golden.expected_label

        else:  # full
            classification = await classifier.classify(metadata, transcript)
            result.predicted_sender_type = classification.sender_type.value
            result.predicted_sender_type_raw = classification.sender_type_raw
            result.predicted_label = classification.label.value
            result.predicted_label_raw = classification.label_raw
            result.sender_type_correct = result.predicted_sender_type == golden.expected_sender_type
            result.label_correct = result.predicted_label == golden.expected_label
            result.privacy_violation = (
                golden.expected_sender_type == "person" and result.predicted_sender_type == "service"
            )

    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        result.error = format_network_error(exc, "LLM endpoint")
    except Exception as exc:
        result.error = str(exc)

    # Use actual LLM time when caching (0 for pure cache hits); wall time otherwise
    cloud = classifier.cloud_llm
    local = classifier.local_llm
    if hasattr(cloud, "take_llm_seconds"):
        result.duration_seconds = round(
            cloud.take_llm_seconds() + local.take_llm_seconds(), 3
        )
    else:
        result.duration_seconds = round(time.monotonic() - start, 3)
    return result


async def run_evaluation(
    golden_set: list[GoldenThread],
    classifier: EmailClassifier,
    stages: str,
    max_thread_chars: int,
    parallelism: int = 1,
) -> list[PredictionResult]:
    """Run evaluation across all golden threads.

    Args:
        golden_set: List of golden set entries.
        classifier: Real EmailClassifier instance.
        stages: Which stages to run.
        max_thread_chars: Max chars for transcript.
        parallelism: Number of concurrent evaluations.

    Returns:
        List of PredictionResult objects.
    """
    semaphore = asyncio.Semaphore(parallelism)
    results: list[PredictionResult] = []
    total = len(golden_set)

    async def eval_with_semaphore(golden: GoldenThread, idx: int) -> PredictionResult:
        async with semaphore:
            r = await evaluate_single(golden, classifier, stages, max_thread_chars)
            status = "error" if r.error else "ok"
            print(f"  [{idx + 1}/{total}] {golden.thread_id[:12]}... {status}", file=sys.stderr)
            return r

    tasks = [eval_with_semaphore(gt, i) for i, gt in enumerate(golden_set)]
    results = await asyncio.gather(*tasks)
    return list(results)


def build_output_path(output_dir: Path, stages: str, tag: str, run_id: str) -> Path:
    """Build output file path from run parameters."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, stages]
    if tag:
        parts.append(tag)
    parts.append(run_id[:8])
    filename = "_".join(parts) + ".jsonl"
    return output_dir / filename


def write_results(
    path: Path,
    meta: RunMeta,
    results: list[PredictionResult],
) -> None:
    """Write run metadata and results to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(meta.to_dict()) + "\n")
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


async def main(args: argparse.Namespace) -> None:
    # Load config
    config_path = Path(args.config) if args.config else Path(__file__).parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    config = substitute_env_vars(config)

    # Load golden set
    golden_path = Path(args.golden_set)
    golden_set = load_golden_set(golden_path, reviewed_only=not args.include_unreviewed)
    if args.sender_type:
        golden_set = [g for g in golden_set if g.expected_sender_type == args.sender_type]

    if not golden_set:
        print("No threads to evaluate.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(golden_set)} threads from {golden_path}", file=sys.stderr)

    if args.dry_run:
        print("Dry run — would evaluate:", file=sys.stderr)
        for gt in golden_set:
            print(f"  {gt.thread_id}: {gt.subject} ({gt.expected_sender_type}/{gt.expected_label})",
                  file=sys.stderr)
        return

    # Build LLM clients (same as daemon.run_daemon())
    cloud_llm_base = LLMClient(
        base_url=os.environ.get("CLOUD_LLM_URL", ""),
        api_key=os.environ.get("CLOUD_LLM_API_KEY", ""),
        model=config["llm"]["cloud"]["model"],
        max_tokens=config["llm"]["cloud"]["max_tokens"],
        temperature=config["llm"]["cloud"]["temperature"],
        timeout=config["llm"]["cloud"]["timeout"],
        extra_body=config["llm"]["cloud"].get("extra_body"),
    )
    local_llm_base = LLMClient(
        base_url=os.environ.get("MLX_URL", ""),
        api_key="",
        model=config["llm"]["local"]["model"],
        max_tokens=config["llm"]["local"]["max_tokens"],
        temperature=config["llm"]["local"]["temperature"],
        timeout=config["llm"]["local"]["timeout"],
        extra_body=config["llm"]["local"].get("extra_body"),
    )

    # Wrap with cache unless --no-cache
    if args.no_cache:
        cloud_llm = cloud_llm_base
        local_llm = local_llm_base
    else:
        cache_path = Path(__file__).parent / "cache" / "llm_cache.jsonl"
        cloud_llm = CachedLLMClient(cloud_llm_base, cache_path)
        local_llm = CachedLLMClient(local_llm_base, cache_path)

    classifier = EmailClassifier(cloud_llm=cloud_llm, local_llm=local_llm, config=config)

    max_thread_chars = config.get("daemon", {}).get("max_thread_chars", 50000)

    # Run evaluation — flush cache even if something fails after evaluation
    try:
        print(f"Running evaluation (stages={args.stages}, parallelism={args.parallelism})...",
              file=sys.stderr)
        results = await run_evaluation(
            golden_set=golden_set,
            classifier=classifier,
            stages=args.stages,
            max_thread_chars=max_thread_chars,
            parallelism=args.parallelism,
        )

        # Build run metadata
        config_bytes = json.dumps(config, sort_keys=True).encode()
        run_id = uuid.uuid4().hex
        meta = RunMeta(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=hashlib.sha256(config_bytes).hexdigest()[:16],
            config_path=str(config_path),
            cloud_model=config["llm"]["cloud"]["model"],
            local_model=config["llm"]["local"]["model"],
            golden_set_path=str(golden_path),
            golden_set_count=len(golden_set),
            stages=args.stages,
            parallelism=args.parallelism,
            tag=args.tag or "",
        )

        # Write results
        output_dir = Path(args.output_dir)
        output_path = build_output_path(output_dir, args.stages, args.tag or "", run_id)
        write_results(output_path, meta, results)

        # Summary
        errors = sum(1 for r in results if r.error)
        st_correct = sum(1 for r in results if r.sender_type_correct is True)
        st_total = sum(1 for r in results if r.sender_type_correct is not None)
        lb_correct = sum(1 for r in results if r.label_correct is True)
        lb_total = sum(1 for r in results if r.label_correct is not None)
        violations = sum(1 for r in results if r.privacy_violation)

        print(f"\nResults written to {output_path}", file=sys.stderr)
        print(f"  Threads: {len(results)} ({errors} errors)", file=sys.stderr)
        if st_total:
            pct = st_correct / st_total
            print(f"  Stage 1 accuracy: {st_correct}/{st_total} ({pct:.1%})", file=sys.stderr)
        if lb_total:
            pct = lb_correct / lb_total
            print(f"  Stage 2 accuracy: {lb_correct}/{lb_total} ({pct:.1%})", file=sys.stderr)
        if violations:
            print(f"  Privacy violations: {violations}", file=sys.stderr)
        if not args.no_cache:
            total_hits = cloud_llm.hits + local_llm.hits
            total_misses = cloud_llm.misses + local_llm.misses
            total_calls = total_hits + total_misses
            if total_calls:
                rate = total_hits / total_calls
                print(f"  Cache: {total_hits}/{total_calls} hits ({rate:.1%})", file=sys.stderr)
    finally:
        if not args.no_cache:
            cloud_llm.flush()
            local_llm.flush()


def cli():
    parser = argparse.ArgumentParser(description="Run classification evaluation")
    parser.add_argument("--golden-set", default="evals/golden_set.jsonl", help="Path to golden set JSONL")
    parser.add_argument("--config", help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--output-dir", default="evals/results/", help="Output directory for results")
    parser.add_argument("--stages", choices=VALID_STAGES, default="full",
                        help="Which stages to evaluate")
    parser.add_argument("--parallelism", type=int, default=3, help="Concurrent evaluations")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Also evaluate threads not yet reviewed (default: reviewed only)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--tag", help="Tag for the results file name")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response cache")
    parser.add_argument("--sender-type", choices=("person", "service"),
                        help="Only evaluate threads with this expected sender type")
    args = parser.parse_args()

    if args.stages not in VALID_STAGES:
        parser.error(f"Invalid stages: {args.stages}")

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
