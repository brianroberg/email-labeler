"""Harvest processed threads from Gmail as golden set data.

Pulls threads labeled agent/processed, infers ground truth from their
existing classification labels, and exports to JSONL.

Usage:
    python -m evals.harvest --output evals/golden_set.jsonl --max-threads 200
    python -m evals.harvest --output evals/golden_set.jsonl --append --sender-type person
"""

import argparse
import asyncio
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from config_utils import substitute_env_vars
from evals.schemas import GoldenThread
from gmail_utils import get_header
from proxy_client import GmailProxyClient

# Label name -> (field, value) for ground truth inference
_SENDER_TYPE_LABELS = {
    "personal": "person",
    "non_personal": "service",
}

_CLASSIFICATION_LABELS = {
    "needs_response": "needs_response",
    "fyi": "fyi",
    "low_priority": "low_priority",
    "unwanted": "unwanted",
}


def load_eval_config(config_path: str | None = None) -> dict:
    """Load and preprocess config.toml from the given path."""
    path = Path(config_path) if config_path else Path(__file__).parent.parent / "config.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return substitute_env_vars(config)


def infer_ground_truth(
    messages: list[dict],
    label_id_to_name: dict[str, str],
    labels_config: dict,
) -> tuple[str, str]:
    """Infer sender_type and classification label from message labelIds.

    Args:
        messages: Gmail message resources (with labelIds).
        label_id_to_name: Mapping from Gmail label ID to label name (e.g. "Label_7" -> "agent/personal").
        labels_config: The [labels] section from config.toml.

    Returns:
        (sender_type, label) tuple. sender_type is "person"/"service"/""
        and label is "needs_response"/"fyi"/"low_priority"/"unwanted"/"".
    """
    # Build reverse map: label_name -> config_key
    name_to_config_key = {}
    for key in list(_SENDER_TYPE_LABELS) + list(_CLASSIFICATION_LABELS):
        label_name = labels_config.get(key, "")
        if label_name:
            name_to_config_key[label_name] = key

    sender_type = ""
    label = ""

    for msg in messages:
        for label_id in msg.get("labelIds", []):
            label_name = label_id_to_name.get(label_id, "")
            config_key = name_to_config_key.get(label_name, "")

            if config_key in _SENDER_TYPE_LABELS:
                sender_type = _SENDER_TYPE_LABELS[config_key]
            elif config_key in _CLASSIFICATION_LABELS:
                label = _CLASSIFICATION_LABELS[config_key]

    return sender_type, label


def deduplicate(new_threads: list[GoldenThread], existing_path: Path) -> list[GoldenThread]:
    """Remove threads already present in the existing golden set file.

    Args:
        new_threads: Newly harvested threads.
        existing_path: Path to existing golden set JSONL file.

    Returns:
        Threads not already in the file.
    """
    existing_ids: set[str] = set()
    if existing_path.exists():
        with open(existing_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing_ids.add(entry["thread_id"])

    return [t for t in new_threads if t.thread_id not in existing_ids]


async def harvest_threads(
    proxy: GmailProxyClient,
    config: dict,
    max_threads: int = 200,
    sender_type_filter: str | None = None,
    label_filter: str | None = None,
) -> list[GoldenThread]:
    """Fetch processed threads and build golden set entries.

    Args:
        proxy: Gmail proxy client.
        config: Full parsed config dict.
        max_threads: Maximum threads to fetch.
        sender_type_filter: Optional filter for "person" or "service".
        label_filter: Optional filter for classification label.

    Returns:
        List of GoldenThread objects.
    """
    labels_config = config["labels"]
    now = datetime.now(timezone.utc).isoformat()

    # Build label ID -> name map
    labels_response = await proxy.list_labels()
    label_id_to_name = {lbl["id"]: lbl["name"] for lbl in labels_response["labels"]}

    # Fetch message stubs with agent/processed label
    processed_label = labels_config["processed"]
    response = await proxy.list_messages(
        q=f"label:{processed_label}",
        max_results=max_threads * 3,  # Over-fetch since we group by thread
    )
    msg_stubs = response.get("messages", [])

    if not msg_stubs:
        print("No processed messages found.", file=sys.stderr)
        return []

    # Group by threadId
    thread_ids: dict[str, list[str]] = {}
    for stub in msg_stubs:
        tid = stub.get("threadId", stub["id"])
        thread_ids.setdefault(tid, []).append(stub["id"])

    print(f"Found {len(thread_ids)} unique threads from {len(msg_stubs)} messages", file=sys.stderr)

    # Fetch each thread and build golden entries
    results: list[GoldenThread] = []
    for i, tid in enumerate(list(thread_ids.keys())[:max_threads]):
        try:
            thread_data = await proxy.get_thread(tid)
            messages = thread_data.get("messages", [])
            if not messages:
                continue

            # Sort chronologically
            messages.sort(key=lambda m: int(m.get("internalDate", "0")))

            # Infer ground truth
            sender_type, label = infer_ground_truth(messages, label_id_to_name, labels_config)
            if not sender_type or not label:
                print(f"  Skipping thread {tid}: incomplete labels (sender={sender_type}, label={label})",
                      file=sys.stderr)
                continue

            # Apply filters
            if sender_type_filter and sender_type != sender_type_filter:
                continue
            if label_filter and label != label_filter:
                continue

            # Extract metadata
            senders = []
            seen: set[str] = set()
            for msg in messages:
                sender = get_header(msg["payload"]["headers"], "From")
                if sender and sender not in seen:
                    senders.append(sender)
                    seen.add(sender)

            first_headers = messages[0]["payload"]["headers"]
            subject = get_header(first_headers, "Subject")
            snippet = messages[-1].get("snippet", "")

            golden = GoldenThread(
                thread_id=tid,
                messages=messages,
                senders=senders,
                subject=subject,
                snippet=snippet,
                expected_sender_type=sender_type,
                expected_label=label,
                source="harvested",
                harvested_at=now,
            )
            results.append(golden)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{min(len(thread_ids), max_threads)} threads...", file=sys.stderr)

        except Exception as exc:
            print(f"  Error fetching thread {tid}: {exc}", file=sys.stderr)

    print(f"Harvested {len(results)} threads", file=sys.stderr)
    return results


def write_golden_set(threads: list[GoldenThread], output_path: Path, append: bool = False) -> None:
    """Write golden set to JSONL file."""
    mode = "a" if append else "w"
    with open(output_path, mode) as f:
        for thread in threads:
            f.write(json.dumps(thread.to_dict()) + "\n")


async def main(args: argparse.Namespace) -> None:
    config = load_eval_config(args.config)
    proxy = GmailProxyClient()

    threads = await harvest_threads(
        proxy=proxy,
        config=config,
        max_threads=args.max_threads,
        sender_type_filter=args.sender_type,
        label_filter=args.label,
    )

    if not threads:
        print("No threads to write.", file=sys.stderr)
        return

    output_path = Path(args.output)
    if args.append:
        threads = deduplicate(threads, output_path)
        if not threads:
            print("All threads already in golden set.", file=sys.stderr)
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_golden_set(threads, output_path, append=args.append)
    print(f"Wrote {len(threads)} threads to {output_path}", file=sys.stderr)


def cli():
    parser = argparse.ArgumentParser(description="Harvest processed emails for golden set")
    parser.add_argument("--output", default="evals/golden_set.jsonl", help="Output JSONL path")
    parser.add_argument("--max-threads", type=int, default=200, help="Max threads to fetch")
    parser.add_argument("--append", action="store_true", help="Append to existing file (deduplicates)")
    parser.add_argument("--sender-type", choices=["person", "service"], help="Filter by sender type")
    parser.add_argument("--label", choices=["needs_response", "fyi", "low_priority", "unwanted"],
                        help="Filter by classification label")
    parser.add_argument("--config", help="Path to config.toml (default: ./config.toml)")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
