"""Harvest candidate newsletters from Gmail as golden-set seed data.

Queries Gmail for threads addressed to the newsletter recipient, guards each
thread with newsletter.is_newsletter(), builds the same transcript body the
production pipeline feeds to story extraction, and seeds a GoldenNewsletter
with NO ground truth (empty story list, reviewed=False). Ground truth is added
later by the human labeling tool.

Always appends to the output file, deduplicating by thread ID — it never
overwrites an existing golden set (that file also holds manual review state).
To start fresh, delete the file manually.

Usage:
    python -m evals.newsletter_harvest --output evals/newsletter_golden_set.jsonl --max-threads 50
    python -m evals.newsletter_harvest --recipient newsletters@dm.org
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from daemon import DEFAULT_MAX_THREAD_CHARS, format_thread_transcript, quiet_http_logging
from evals import format_network_error
from evals.harvest import load_eval_config
from evals.newsletter_schemas import GoldenNewsletter
from gmail_utils import get_header
from newsletter import is_newsletter
from proxy_client import GmailProxyClient, ProxyAuthError, ProxyError, ProxyForbiddenError

_NETWORK_ERRORS = (
    httpx.ConnectError, httpx.TimeoutException, ProxyAuthError, ProxyForbiddenError, ProxyError,
)


def load_existing_thread_ids(existing_path: Path) -> set[str]:
    """Collect thread IDs already present in the golden set JSONL file."""
    existing_ids: set[str] = set()
    if existing_path.exists():
        with open(existing_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # Tolerate a corrupt/partial line (e.g. an interrupted append)
                # so one bad row can't abort every future harvest — the golden
                # set is hand-editable and append-only.
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: skipping malformed line {lineno} in {existing_path}",
                          file=sys.stderr)
                    continue
                thread_id = entry.get("thread_id")
                if thread_id:
                    existing_ids.add(thread_id)
    return existing_ids


def deduplicate(
    new_newsletters: list[GoldenNewsletter], existing_path: Path
) -> list[GoldenNewsletter]:
    """Remove newsletters already present in the existing golden set file.

    Args:
        new_newsletters: Newly harvested newsletters.
        existing_path: Path to existing golden set JSONL file.

    Returns:
        Newsletters not already in the file.
    """
    existing_ids = load_existing_thread_ids(existing_path)
    return [n for n in new_newsletters if n.thread_id not in existing_ids]


async def harvest_newsletters(
    proxy: GmailProxyClient,
    config: dict,
    max_threads: int = 50,
    recipient: str | None = None,
    skip_thread_ids: set[str] | None = None,
) -> list[GoldenNewsletter]:
    """Fetch newsletter threads and seed golden-set entries (no ground truth).

    Args:
        proxy: Gmail proxy client.
        config: Full parsed config dict.
        max_threads: Maximum threads to fetch.
        recipient: Newsletter recipient; defaults to config["newsletter"]["recipient"].
        skip_thread_ids: Thread IDs already in the golden set — skipped before
            get_thread so re-runs don't re-download them.

    Returns:
        List of seeded GoldenNewsletter objects (empty stories, reviewed=False).
    """
    if recipient is None:
        recipient = config["newsletter"]["recipient"]
    max_thread_chars = config.get("daemon", {}).get("max_thread_chars", DEFAULT_MAX_THREAD_CHARS)
    now = datetime.now(timezone.utc).isoformat()

    # Mirror daemon.py's newsletter query narrowing: gmail_query += f" to:{...}".
    query = f"to:{recipient}"
    try:
        response = await proxy.list_messages(
            q=query,
            max_results=max_threads * 3,  # Over-fetch since we group by thread
        )
    except _NETWORK_ERRORS as exc:
        print(f"Error: {format_network_error(exc, 'api-proxy')}", file=sys.stderr)
        sys.exit(1)
    msg_stubs = response.get("messages", [])

    if not msg_stubs:
        print(f"No messages found for query: {query}", file=sys.stderr)
        return []

    # Group by threadId
    thread_ids: dict[str, list[str]] = {}
    for stub in msg_stubs:
        tid = stub.get("threadId", stub["id"])
        thread_ids.setdefault(tid, []).append(stub["id"])

    print(f"Found {len(thread_ids)} unique threads from {len(msg_stubs)} messages", file=sys.stderr)

    skip_thread_ids = skip_thread_ids or set()
    results: list[GoldenNewsletter] = []
    for i, tid in enumerate(list(thread_ids.keys())[:max_threads]):
        if tid in skip_thread_ids:
            print(f"  Skipping thread {tid}: already in golden set", file=sys.stderr)
            continue
        try:
            thread_data = await proxy.get_thread(tid)
            messages = thread_data.get("messages", [])
            if not messages:
                continue

            # Sort chronologically
            messages.sort(key=lambda m: int(m.get("internalDate", "0")))

            # Guard: only threads actually addressed to the newsletter recipient.
            if not is_newsletter(messages, recipient):
                print(f"  Skipping thread {tid}: not addressed to {recipient}", file=sys.stderr)
                continue

            # Build the transcript exactly as production does (daemon.py:345).
            body = format_thread_transcript(messages, max_thread_chars)

            first_headers = messages[0]["payload"]["headers"]
            subject = get_header(first_headers, "Subject")
            sender = get_header(first_headers, "From")
            message_id = messages[-1]["id"]

            golden = GoldenNewsletter(
                thread_id=tid,
                message_id=message_id,
                sender=sender,
                subject=subject,
                body=body,
                stories=[],
                source="harvested",
                harvested_at=now,
                seeded_from="",
                reviewed=False,
            )
            results.append(golden)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{min(len(thread_ids), max_threads)} threads...",
                      file=sys.stderr)

        except _NETWORK_ERRORS as exc:
            print(f"  Error fetching thread {tid}: {format_network_error(exc, 'api-proxy')}",
                  file=sys.stderr)
        except Exception as exc:
            print(f"  Error processing thread {tid}: {exc}", file=sys.stderr)

    print(f"Harvested {len(results)} newsletters", file=sys.stderr)
    return results


def write_golden_set(newsletters: list[GoldenNewsletter], output_path: Path) -> None:
    """Append golden-set entries to the JSONL file.

    Always appends — harvest never truncates an existing golden set, since that
    file also holds manual review state (curated stories, scores, themes, notes).
    To start fresh, delete the file manually.
    """
    with open(output_path, "a") as f:
        for newsletter in newsletters:
            f.write(json.dumps(newsletter.to_dict()) + "\n")


async def main(args: argparse.Namespace) -> None:
    # daemon's module-level logging.basicConfig(INFO) would otherwise let httpx
    # print an 'HTTP Request: ...' line per proxy call, drowning harvest output.
    quiet_http_logging()
    config = load_eval_config(args.config)
    try:
        proxy = GmailProxyClient(proxy_url=args.proxy_url)
    except ProxyAuthError as exc:
        # Raised at construction when PROXY_API_KEY is unset — a traceback here
        # buries the one env var the user needs to set.
        print(f"Error: {exc}", file=sys.stderr)
        print("Set PROXY_API_KEY (e.g. via the .env symlink to agent-stack/.env) "
              "and re-run.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    newsletters = await harvest_newsletters(
        proxy=proxy,
        config=config,
        max_threads=args.max_threads,
        recipient=args.recipient,
        skip_thread_ids=load_existing_thread_ids(output_path),
    )

    if not newsletters:
        print("No newsletters to write.", file=sys.stderr)
        return

    # Backstop: the pre-fetch skip already avoids re-downloading known threads,
    # but re-dedup against the file in case it changed during the harvest.
    newsletters = deduplicate(newsletters, output_path)
    if not newsletters:
        print("All newsletters already in golden set.", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_golden_set(newsletters, output_path)
    print(f"Appended {len(newsletters)} newsletters to {output_path}", file=sys.stderr)
    print(
        "Next: add ground truth with "
        f"`uv run python -m evals.newsletter_label --golden-set {output_path}`",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest candidate newsletters for golden set")
    parser.add_argument("--output", default="evals/newsletter_golden_set.jsonl",
                        help="Output JSONL path, relative to the CWD — run from the repo root "
                             "so the default chains with the other eval tools "
                             "(default: %(default)s)")
    parser.add_argument("--max-threads", type=int, default=50,
                        help="Max threads to fetch (default: %(default)s)")
    parser.add_argument("--recipient",
                        help="Newsletter recipient (default: config[newsletter][recipient])")
    parser.add_argument("--config",
                        help="Path to config.toml (default: the repo-root config.toml, "
                             "regardless of CWD)")
    parser.add_argument("--proxy-url", help="API proxy URL (overrides PROXY_URL env var)")
    return parser


def cli():
    args = build_parser().parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
