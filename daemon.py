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

from dotenv import load_dotenv

from classifier import EmailClassifier, EmailMetadata, SenderType
from gmail_utils import decode_body, get_header
from labeler import LabelManager
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


async def process_single_email(
    msg_id: str,
    proxy_client: GmailProxyClient,
    classifier: EmailClassifier,
    label_manager: LabelManager,
    mlx_available: bool,
) -> bool:
    """Process a single email through the classification pipeline.

    Args:
        msg_id: Gmail message ID.
        proxy_client: Gmail proxy client.
        classifier: Email classifier instance.
        label_manager: Label manager instance.
        mlx_available: Whether the local MLX LLM is available.

    Returns:
        True if the email was successfully classified and labeled,
        False if skipped or errored.
    """
    try:
        message = await proxy_client.get_message(msg_id)
        headers = message["payload"]["headers"]
        sender = get_header(headers, "From")
        subject = get_header(headers, "Subject")
        snippet = message.get("snippet", "")
        body = decode_body(message["payload"])

        metadata = EmailMetadata(
            message_id=msg_id,
            sender=sender,
            subject=subject,
            snippet=snippet,
        )

        if not mlx_available:
            # Check sender type first — if person, skip (privacy)
            sender_type, _ = await classifier.classify_sender(metadata)
            if sender_type == SenderType.PERSON:
                log.info("Skipping person email %s (MLX unavailable): %s", msg_id, subject)
                return False
            # Service email — can proceed with cloud-only classification
            result = await classifier.classify(metadata, body)
        else:
            result = await classifier.classify(metadata, body)

        await label_manager.apply_classification(msg_id, result.label)
        log.info(
            "Classified %s: sender=%s label=%s — %s",
            msg_id,
            result.sender_type.value,
            result.label.value,
            subject,
        )
        return True

    except Exception:
        log.exception("Error processing email %s", msg_id)
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

    # Verify labels on startup
    missing = await label_manager.verify_labels()
    if missing:
        log.error("Missing Gmail labels: %s", missing)
        log.error("Create these labels manually in Gmail before running the daemon.")
        sys.exit(1)

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
                log.info("Found %d unprocessed email(s)", len(messages))

            processed = 0
            for msg_stub in messages:
                success = await process_single_email(
                    msg_stub["id"],
                    proxy_client,
                    classifier,
                    label_manager,
                    mlx_available,
                )
                if success:
                    processed += 1

            if messages:
                log.info("Processed %d/%d emails", processed, len(messages))

            # Update healthcheck
            healthcheck_file.write_text(str(asyncio.get_event_loop().time()))

            # Reset backoff on success
            backoff = poll_interval

        except Exception:
            log.exception("Error in poll cycle")
            backoff = min(backoff * 2, poll_interval * 10)
            log.info("Backing off for %ds", backoff)

        await asyncio.sleep(backoff)


def main():
    """Entry point."""
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
