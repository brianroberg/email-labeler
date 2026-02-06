"""Email labeler daemon — main entry point.

Continuously polls Gmail for unclassified emails, classifies them
using a two-tier LLM system, and applies labels autonomously.

Privacy invariant: Person email bodies NEVER leave the local network.
"""

import asyncio
import logging
import os
import sys
import tomllib
from pathlib import Path

import httpx
from dotenv import load_dotenv

from classifier import EmailClassifier, EmailLabel, SenderType, ThreadMetadata
from gmail_utils import decode_body, get_header
from labeler import LabelManager, _get_priority
from llm_client import LLMClient
from proxy_client import GmailProxyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("email-labeler")


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def format_thread_transcript(messages: list[dict], max_chars: int) -> str:
    """Format Gmail messages into a chronological thread transcript.

    Messages should be pre-sorted by internalDate (chronological).
    If the transcript exceeds max_chars, the oldest messages are dropped
    and a truncation notice is prepended.

    Args:
        messages: List of Gmail message resources (with payload.headers and payload.body).
        max_chars: Maximum character limit for the transcript.

    Returns:
        Formatted transcript string.
    """
    parts = []
    for msg in messages:
        headers = msg["payload"]["headers"]
        sender = get_header(headers, "From")
        date = get_header(headers, "Date")
        body = decode_body(msg["payload"])

        part = f"--- Message from {sender} on {date} ---\n{body}"
        parts.append(part)

    full = "\n\n".join(parts)
    if len(full) <= max_chars:
        return full

    # Truncate from the oldest messages first
    while len(parts) > 1:
        parts.pop(0)
        candidate = "[Earlier messages truncated]\n\n" + "\n\n".join(parts)
        if len(candidate) <= max_chars:
            return candidate

    # Single message still too long — hard truncate
    return parts[0][:max_chars]


async def process_single_thread(
    thread_id: str,
    msg_ids: list[str],
    proxy_client: GmailProxyClient,
    classifier: EmailClassifier,
    label_manager: LabelManager,
    mlx_available: bool,
    max_thread_chars: int,
) -> bool:
    """Process a single thread through the classification pipeline.

    Fetches the full thread, formats all messages into a transcript,
    classifies once, and applies labels to all messages in the thread.

    Enforces no-downgrade rule: if the thread already has a classification
    with equal or higher priority, it is skipped.

    Args:
        thread_id: Gmail thread ID.
        msg_ids: List of message IDs in the thread (from list_messages stubs).
        proxy_client: Gmail proxy client.
        classifier: Email classifier instance.
        label_manager: Label manager instance.
        mlx_available: Whether the local MLX LLM is available.
        max_thread_chars: Maximum characters for thread transcript.

    Returns:
        True if the thread was successfully classified and labeled,
        False if skipped or errored.
    """
    try:
        # Fetch full thread (all messages in one API call)
        thread_data = await proxy_client.get_thread(thread_id)
        messages = thread_data.get("messages", [])
        if not messages:
            log.warning("Thread %s has no messages, skipping", thread_id)
            return False

        # Sort chronologically
        messages.sort(key=lambda m: int(m.get("internalDate", "0")))

        # Check priority — skip if already classified at max priority
        existing_priority = label_manager.get_existing_priority(messages)
        if existing_priority is not None and existing_priority >= _get_priority(EmailLabel.NEEDS_RESPONSE):
            log.info("Thread %s already at max priority, skipping", thread_id)
            return False

        # Collect all message IDs (thread may have more messages than query returned)
        all_msg_ids = [msg["id"] for msg in messages]

        # Extract unique senders (preserve order)
        senders = []
        seen = set()
        for msg in messages:
            headers = msg["payload"]["headers"]
            sender = get_header(headers, "From")
            if sender and sender not in seen:
                senders.append(sender)
                seen.add(sender)

        if not senders:
            log.warning("Thread %s has no valid senders, skipping", thread_id)
            return False

        first_headers = messages[0]["payload"]["headers"]
        subject = get_header(first_headers, "Subject")
        snippet = messages[-1].get("snippet", "")  # latest message snippet

        metadata = ThreadMetadata(
            thread_id=thread_id,
            senders=senders,
            subject=subject,
            snippet=snippet,
        )

        # Format thread transcript
        transcript = format_thread_transcript(messages, max_thread_chars)

        # Classify
        if not mlx_available:
            sender_type, sender_raw = await classifier.classify_sender(metadata)
            if sender_type == SenderType.PERSON:
                log.info("Skipping person thread %s (MLX unavailable): %s", thread_id, subject)
                return False
            result = await classifier.classify(metadata, transcript, sender_type, sender_raw)
        else:
            result = await classifier.classify(metadata, transcript)

        # Enforce no-downgrade
        new_priority = _get_priority(result.label)
        if existing_priority is not None and existing_priority >= new_priority:
            log.info(
                "Thread %s: existing priority %d >= new %d, skipping downgrade",
                thread_id,
                existing_priority,
                new_priority,
            )
            return False

        # Apply labels to ALL messages in thread
        await label_manager.apply_classification(all_msg_ids, result.label, result.sender_type)
        log.info(
            "Classified thread %s (%d msgs): sender=%s label=%s — %s",
            thread_id,
            len(all_msg_ids),
            result.sender_type.value,
            result.label.value,
            subject,
        )
        return True

    except httpx.ConnectError as exc:
        log.warning("Connection error processing thread %s: %s", thread_id, exc)
        return False
    except Exception:
        log.exception("Error processing thread %s", thread_id)
        return False


async def run_daemon() -> None:
    """Main polling loop."""
    config = load_config()
    daemon_config = config["daemon"]

    proxy_client = GmailProxyClient()
    cloud_llm = LLMClient(
        base_url=os.environ.get("CLOUD_LLM_URL", ""),
        api_key=os.environ.get("CLOUD_LLM_API_KEY", ""),
        model=config["llm"]["cloud"]["model"],
        max_tokens=config["llm"]["cloud"]["max_tokens"],
        temperature=config["llm"]["cloud"]["temperature"],
        timeout=config["llm"]["cloud"]["timeout"],
    )
    local_llm = LLMClient(
        base_url=os.environ.get("MLX_URL", ""),
        api_key="",
        model=config["llm"]["local"]["model"],
        max_tokens=config["llm"]["local"]["max_tokens"],
        temperature=config["llm"]["local"]["temperature"],
        timeout=config["llm"]["local"]["timeout"],
    )

    classifier = EmailClassifier(
        cloud_llm=cloud_llm,
        local_llm=local_llm,
        config=config,
    )
    label_manager = LabelManager(proxy_client=proxy_client, config=config)

    # Wait for api-proxy to become available, then verify labels
    startup_backoff = 5
    max_startup_backoff = 60
    while True:
        try:
            missing = await label_manager.verify_labels()
            if missing:
                log.error("Missing Gmail labels: %s", missing)
                log.error("Create these labels manually in Gmail before running the daemon.")
                sys.exit(1)
            break
        except httpx.ConnectError:
            log.warning(
                "Cannot reach api-proxy at %s — retrying in %ds",
                proxy_client.proxy_url,
                startup_backoff,
            )
            await asyncio.sleep(startup_backoff)
            startup_backoff = min(startup_backoff * 2, max_startup_backoff)

    log.info("All labels verified. Starting poll loop.")

    poll_interval = daemon_config["poll_interval_seconds"]
    max_emails = daemon_config["max_emails_per_cycle"]
    gmail_query = daemon_config["gmail_query"]
    healthcheck_file = Path(daemon_config["healthcheck_file"])
    backoff = poll_interval

    while True:
        try:
            # Check MLX availability each cycle
            mlx_available = await local_llm.is_available()
            if not mlx_available:
                log.warning("Local MLX LLM unavailable — person emails will be skipped")

            response = await proxy_client.list_messages(q=gmail_query, max_results=max_emails)
            messages = response.get("messages", [])

            if messages:
                log.info("Found %d unprocessed message(s)", len(messages))

            # Group messages by threadId
            threads: dict[str, list[str]] = {}
            for msg_stub in messages:
                tid = msg_stub.get("threadId", msg_stub["id"])
                threads.setdefault(tid, []).append(msg_stub["id"])

            if threads:
                log.info("Grouped into %d thread(s)", len(threads))

            max_thread_chars = daemon_config.get("max_thread_chars", 50000)
            processed = 0
            for tid, msg_ids in threads.items():
                success = await process_single_thread(
                    tid,
                    msg_ids,
                    proxy_client,
                    classifier,
                    label_manager,
                    mlx_available,
                    max_thread_chars,
                )
                if success:
                    processed += 1

            if threads:
                log.info("Processed %d/%d threads", processed, len(threads))

            # Update healthcheck
            healthcheck_file.write_text(str(asyncio.get_event_loop().time()))

            # Reset backoff on success
            backoff = poll_interval

        except httpx.ConnectError:
            log.warning(
                "Lost connection to api-proxy at %s — retrying in %ds",
                proxy_client.proxy_url,
                backoff,
            )
            backoff = min(backoff * 2, poll_interval * 10)
        except Exception:
            log.exception("Error in poll cycle")
            backoff = min(backoff * 2, poll_interval * 10)

        await asyncio.sleep(backoff)


def main():
    """Entry point."""
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
