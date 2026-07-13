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
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

from classifier import EmailClassifier, EmailLabel, SenderType, ThreadMetadata
from config_utils import substitute_env_vars
from gmail_utils import decode_body, get_header
from labeler import LabelManager, _get_priority
from llm_client import LLMBalanceError, LLMClient, LLMUnavailableError
from newsletter import (
    NewsletterClassifier,
    NewsletterTier,
    aggregate_theme_grades,
    is_newsletter,
    parse_send_date,
    write_assessment,
)
from proxy_client import (
    TRANSIENT_TRANSPORT_ERRORS,
    GmailProxyClient,
    ProxyError,
    ProxyUnavailableError,
)

load_dotenv()

_TIER_RANK = {
    NewsletterTier.POOR: 0,
    NewsletterTier.FAIR: 1,
    NewsletterTier.GOOD: 2,
    NewsletterTier.EXCELLENT: 3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("email-labeler")


def quiet_http_logging() -> None:
    """Silence httpx/httpcore per-request INFO logs (issue #58).

    httpx logs 'HTTP Request: … 200 OK' at INFO for every poll, and the URL
    embeds the gmail_query ('-label:agent/processed -label:agent/attempted') —
    healthy polls read as alarms to anyone scanning the log. Only these library
    loggers are raised; the email-labeler logger stays at INFO. Canonical copy —
    the eval CLIs import it (evals may depend on daemon, never the reverse).
    Each entry point calls it explicitly rather than at import, so merely
    importing daemon's helpers never mutates process-wide logging.
    """
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# Default cap on transcript chars sent to the classifier when config.toml omits
# max_thread_chars. Shared with the eval harness (evals/run_eval.py) so the two
# never drift — an eval must truncate transcripts exactly as production does.
DEFAULT_MAX_THREAD_CHARS = 16000


def load_config() -> dict:
    """Load configuration from config.toml.

    After parsing, any {env.VAR_NAME} placeholders in string values are
    replaced with the corresponding environment variable (empty string if unset).
    """
    config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return substitute_env_vars(config)


def resolve_int_env(env_var: str, default: int, minimum: int = 1) -> int:
    """Return an int from *env_var* if set and valid, otherwise *default*.

    Lets operators override numeric daemon settings (e.g. concurrency) per run
    without editing config.toml. {env.VAR} substitution only works for string
    config values, so numeric overrides are read here instead.

    A value that is unparseable OR below *minimum* falls back to the default with
    a warning rather than crashing. The lower bound matters: these values feed
    asyncio.Semaphore() and Gmail maxResults, where 0 deadlocks the poll loop and
    a negative crashes the daemon at startup.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid %s=%r (expected an integer); using %d", env_var, raw, default)
        return default
    if value < minimum:
        log.warning(
            "%s=%d is below the minimum of %d; using %d", env_var, value, minimum, default
        )
        return default
    return value


def resolve_newsletter_llm_endpoint() -> tuple[str, str]:
    """Return (base_url, api_key) for the newsletter grading LLM.

    Defaults to the cloud classification endpoint (CLOUD_LLM_URL / CLOUD_LLM_API_KEY)
    so a single provider serves both. Set NEWSLETTER_LLM_URL / NEWSLETTER_LLM_API_KEY
    when the newsletter model lives elsewhere — e.g. config.toml grades newsletters with
    a Claude model (`claude-sonnet-4-6`) that the cloud provider doesn't serve, so it must
    target a Claude-serving endpoint (Anthropic's OpenAI-compatible API, or a gateway).

    The override is atomic: once NEWSLETTER_LLM_URL is set, the key comes solely
    from NEWSLETTER_LLM_API_KEY (empty if unset) and never falls back to the cloud
    key — pairing an override endpoint with the cloud provider's credential would
    authenticate against the wrong provider and fail in a confusing way.
    """
    override_url = os.environ.get("NEWSLETTER_LLM_URL")
    if override_url:
        return override_url, os.environ.get("NEWSLETTER_LLM_API_KEY", "")
    return os.environ.get("CLOUD_LLM_URL", ""), os.environ.get("CLOUD_LLM_API_KEY", "")


class FailureTracker:
    """Counts consecutive thread-specific failures to break infinite retry loops.

    Records only failures the caller deems give-up-eligible. An endpoint-wide
    outage never lands here: an LLM outage (ConnectError → LLMUnavailableError) is
    deferred per-thread without counting, and a proxy outage fails the cycle-level
    list_messages first, so the whole cycle defers before any thread is counted.
    A *persistent per-thread* fault — a poison thread, including a deterministic
    per-request proxy 5xx (issue #26) — does accrue here and is given up once it
    hits max_failures. In-memory and session-scoped: counts reset on daemon
    restart, so a thread that failed for a since-resolved reason gets another
    chance after a restart.
    """

    def __init__(self, max_failures: int = 5):
        self.max_failures = max_failures
        self._counts: dict[str, int] = {}
        self._given_up: list[str] = []  # threads abandoned since the last take_given_up()

    def record_failure(self, thread_id: str) -> None:
        self._counts[thread_id] = self._counts.get(thread_id, 0) + 1

    def should_give_up(self, thread_id: str) -> bool:
        return self._counts.get(thread_id, 0) >= self.max_failures

    def clear(self, thread_id: str) -> None:
        self._counts.pop(thread_id, None)

    def prune(self, active_thread_ids) -> None:
        """Drop failure counts for threads no longer pending.

        A thread that fails a few times (below the give-up threshold) and then
        disappears from the query — read, archived, or relabeled externally —
        would otherwise leak its count for the daemon's lifetime. Pruning each
        cycle to the still-pending set keeps the map bounded; a consecutively
        failing thread reappears every cycle, so it is never pruned.
        """
        active = set(active_thread_ids)
        self._counts = {tid: n for tid, n in self._counts.items() if tid in active}

    def record_give_up(self, thread_id: str) -> None:
        """Record that a thread was abandoned (marked agent/attempted, no classification),
        so the per-cycle summary can report give-ups distinctly from classifications."""
        self._given_up.append(thread_id)

    def take_given_up(self) -> list[str]:
        """Return the thread ids given up since the last call, resetting the list."""
        out = self._given_up
        self._given_up = []
        return out


class DaemonHalt:
    """Daemon-wide halt state for account-level faults (provider out of funds).

    Unlike a poison thread (FailureTracker's territory), an out-of-funds provider
    fails EVERY request: retrying per-thread just burns the backlog into
    agent/attempted. Tripping this halts the poll loop entirely until the admin
    adds funds and restarts. In-memory and session-scoped by design — a restart
    is the only way to clear it. First tripper wins: threads in one
    asyncio.gather cycle may race to trip, and the reason must stay stable.
    """

    def __init__(self):
        self.reason: str | None = None

    def trip(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason

    @property
    def tripped(self) -> bool:
        return self.reason is not None


async def _give_up_if_stuck(
    thread_id: str,
    msg_ids: list[str],
    failure_tracker: "FailureTracker | None",
    label_manager: LabelManager,
    write_sem: asyncio.Semaphore | None = None,
) -> bool:
    """Record a thread-specific failure; if it has failed too many times, mark it
    agent/attempted so it stops being retried every cycle.

    Uses agent/attempted (not agent/processed): the thread is excluded from the
    unprocessed query either way, but the distinct label keeps abandoned threads
    findable and separate from successfully-classified mail.

    Returns True if the thread was given up (marked agent/attempted → handled), or
    False if it should simply be retried next cycle.
    """
    if failure_tracker is None:
        return False
    failure_tracker.record_failure(thread_id)
    if not failure_tracker.should_give_up(thread_id):
        return False
    try:
        async with (write_sem or nullcontext()):
            await label_manager.mark_attempted(msg_ids)
    except ProxyUnavailableError as exc:
        # The proxy is transiently down, so the agent/attempted marker can't be written
        # right now — expected during an outage, not a bug. We return without recording
        # the give-up, so the thread stays give-up-eligible: its failure count is left
        # ≥ max_failures (this arm returns False, so the poll loop's success-only
        # count-clear never runs), and the marker write is retried next cycle once the
        # proxy recovers. Log a clean warning rather than spamming a traceback for an
        # expected transient condition. (The thread keeps re-matching the query until it
        # is actually labeled; skipping the re-classification in that window is a
        # separate efficiency concern, tracked in #29.)
        log.warning(
            "Proxy unavailable marking stuck thread %s attempted — will retry next cycle: %s",
            thread_id, exc,
        )
        return False
    except Exception:
        # An unexpected, likely-permanent marker-write failure (a real bug, or a
        # misconfigured label): keep the full traceback so it stays diagnosable rather
        # than looping silently as if benign.
        log.exception("Could not mark stuck thread %s attempted", thread_id)
        return False
    # Logged once here — AFTER the marker write actually lands — not before it: a
    # transient write-outage can retry this for several cycles, and logging on every
    # attempt would re-spam ERROR for what is an expected, recoverable condition.
    log.error(
        "Thread %s failed %d+ times — marked agent/attempted to break the retry loop",
        thread_id, failure_tracker.max_failures,
    )
    failure_tracker.record_give_up(thread_id)
    # Note: the poll loop clears the failure count on every True result (which a
    # give-up returns), so the count is reset there — no need to clear it here too.
    return True


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
    cloud_sem: asyncio.Semaphore,
    local_sem: asyncio.Semaphore,
    max_thread_chars: int,
    newsletter_classifier: NewsletterClassifier | None = None,
    newsletter_recipient: str = "",
    newsletter_output_file: str = "",
    newsletter_only: bool = False,
    failure_tracker: "FailureTracker | None" = None,
    fetch_sem: asyncio.Semaphore | None = None,
    write_sem: asyncio.Semaphore | None = None,
    local_deferrals: list[str] | None = None,
    halt: "DaemonHalt | None" = None,
) -> bool:
    """Process a single thread through the classification pipeline.

    Fetches the full thread, formats all messages into a transcript,
    classifies once, and applies labels to all messages in the thread.

    Uses semaphores to bound concurrent LLM requests:
    - cloud_sem: acquired for Stage 1 (sender classification) and Stage 2 SERVICE emails
    - local_sem: acquired for Stage 2 PERSON emails (local MLX LLM)

    If the local LLM is unreachable, ConnectError is raised immediately.
    If it times out (e.g. model loading), the configured timeout applies.

    Enforces no-downgrade rule: if the thread already has a classification
    with equal or higher priority, it is skipped.

    Args:
        thread_id: Gmail thread ID.
        msg_ids: List of message IDs in the thread (from list_messages stubs).
        proxy_client: Gmail proxy client.
        classifier: Email classifier instance.
        label_manager: Label manager instance.
        cloud_sem: Semaphore bounding concurrent cloud LLM requests.
        local_sem: Semaphore bounding concurrent local LLM requests.
        max_thread_chars: Maximum characters for thread transcript.

    Returns:
        True if the thread was successfully classified and labeled,
        False if skipped or errored.
    """
    # Messages to mark processed if we give up. Starts as the query stubs (all we
    # know before fetching); upgraded to the full message list once the thread is
    # fetched, so a give-up marks every message and the thread stops re-matching
    # the query (otherwise unmarked siblings re-surface it and the retry loop the
    # give-up exists to break never converges).
    ids_to_mark = msg_ids
    try:
        # Fetch full thread (all messages in one API call). Bounded by fetch_sem so
        # a large max_emails_per_cycle can't burst one concurrent proxy read per
        # thread (the cloud/local semaphores gate only the classify calls).
        async with (fetch_sem or nullcontext()):
            thread_data = await proxy_client.get_thread(thread_id)
        messages = thread_data.get("messages", [])
        if not messages:
            log.warning("Thread %s has no messages, skipping", thread_id)
            return False

        # Sort chronologically
        messages.sort(key=lambda m: int(m.get("internalDate", "0")))

        # Full message list — the thread may carry more messages than the query
        # returned. Upgrade the give-up target from the query stubs to every message
        # id now (see the ids_to_mark note above), so any give-up or skip-and-mark
        # path below marks the whole thread and it stops re-matching the query.
        # Every post-fetch branch shares this single computation.
        all_msg_ids = [msg["id"] for msg in messages]
        ids_to_mark = all_msg_ids

        # Newsletter detection — route to newsletter pipeline if applicable
        if newsletter_classifier and newsletter_recipient:
            if is_newsletter(messages, newsletter_recipient):
                first_headers = messages[0]["payload"]["headers"]
                subject = get_header(first_headers, "Subject")
                sender = get_header(first_headers, "From")
                send_date = parse_send_date(
                    get_header(first_headers, "Date"), messages[0].get("internalDate")
                )
                transcript = format_thread_transcript(messages, max_thread_chars)

                async with cloud_sem:
                    story_results = await newsletter_classifier.classify_newsletter(transcript)

                # Determine overall tier (best story's tier)
                best_tier = None
                for sr in story_results:
                    if sr.tier is not None:
                        if best_tier is None or _TIER_RANK.get(sr.tier, 0) > _TIER_RANK.get(best_tier, 0):
                            best_tier = sr.tier
                # Merge graded themes across stories (strongest grade per theme)
                all_themes = aggregate_theme_grades(story_results)

                async with (write_sem or nullcontext()):
                    await label_manager.apply_newsletter_classification(
                        message_ids=all_msg_ids,
                        tier=best_tier,
                        themes=all_themes,
                    )

                # Write structured results
                if newsletter_output_file:
                    try:
                        write_assessment(
                            output_file=newsletter_output_file,
                            message_id=all_msg_ids[0],
                            thread_id=thread_id,
                            sender=sender,
                            subject=subject,
                            overall_tier=best_tier,
                            stories=story_results,
                            send_date=send_date,
                            model=newsletter_classifier.cloud_llm.model,
                        )
                    except Exception:
                        log.exception("Failed to write newsletter assessment for thread %s", thread_id)

                story_count = len(story_results)
                log.info(
                    "Newsletter thread %s: %d stories, tier=%s, themes=%s — %s",
                    thread_id,
                    story_count,
                    best_tier.value if best_tier else "no-stories",
                    all_themes,
                    subject,
                )
                return True

        # Newsletter-only mode: skip non-newsletter threads
        if newsletter_only:
            log.debug("Skipping non-newsletter thread %s (newsletter-only mode)", thread_id)
            return False

        # Check priority — skip if already classified at max priority
        existing_priority = label_manager.get_existing_priority(messages)
        if existing_priority is not None and existing_priority >= _get_priority(EmailLabel.NEEDS_RESPONSE):
            # Already at max priority, so there's nothing to classify — but mark it
            # processed so it drops out of the unprocessed query. Otherwise the thread
            # has no agent/processed label and re-matches every cycle forever, costing
            # a full get_thread round-trip per thread per poll (same retry-loop reasoning
            # as the no-downgrade branch below). ids_to_mark was upgraded to all_msg_ids
            # above the priority check, so a failed write here gives up on the whole
            # thread, not just the query stubs.
            async with (write_sem or nullcontext()):
                await label_manager.mark_processed(all_msg_ids)
            log.info("Thread %s already at max priority, marking processed", thread_id)
            return True

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

        # Stage 1: classify sender (always cloud LLM)
        async with cloud_sem:
            sender_type, sender_raw, sender_cot = await classifier.classify_sender(metadata)

        # Stage 2: classify email (routed by sender type)
        if sender_type == SenderType.PERSON:
            async with local_sem:
                result = await classifier.classify(metadata, transcript, sender_type, sender_raw)
        else:
            async with cloud_sem:
                result = await classifier.classify(metadata, transcript, sender_type, sender_raw)

        # Enforce no-downgrade
        new_priority = _get_priority(result.label)
        if existing_priority is not None and existing_priority >= new_priority:
            log.info(
                "Thread %s: existing priority %d >= new %d, skipping downgrade",
                thread_id,
                existing_priority,
                new_priority,
            )
            # Still mark as processed so the thread isn't retried every cycle
            async with (write_sem or nullcontext()):
                await label_manager.mark_processed(all_msg_ids)
            return True

        # Apply labels to ALL messages in thread (bounded by write_sem so a large
        # max_emails_per_cycle can't burst one concurrent write per thread).
        async with (write_sem or nullcontext()):
            await label_manager.apply_classification(all_msg_ids, result.label, result.sender_type)
        log.info(
            "Classified thread %s (%d msgs): sender=%s label=%s — %s",
            thread_id,
            len(all_msg_ids),
            result.sender_type.value,
            result.label.value,
            subject,
        )
        log.debug("Thread %s CoT — sender: %s", thread_id, result.sender_cot)
        log.debug("Thread %s CoT — label: %s", thread_id, result.label_cot)
        return True

    except LLMUnavailableError as exc:
        # LLM endpoint unreachable or dropped mid-request (server down, connect
        # timeout, reset) — transient. Don't count it toward give-up; just retry
        # next cycle (preserves graceful degradation of the privacy invariant).
        #
        # The LOCAL tier being down is a routine operating condition (the MLX
        # laptop is deliberately offline for hours at a time), not an incident:
        # log per-thread detail at DEBUG and count the deferral so the poll loop
        # can emit a single per-cycle INFO summary instead of N warnings
        # (issue #24). A cloud (or tier-less) outage stays a WARNING.
        if exc.tier == "local":
            log.debug("Local LLM unavailable processing thread %s: %s", thread_id, exc)
            if local_deferrals is not None:
                local_deferrals.append(thread_id)
        else:
            log.warning("LLM unavailable processing thread %s: %s", thread_id, exc)
        return False
    except ProxyUnavailableError as exc:
        # api-proxy unavailable for THIS thread's call — connection refused, a timeout,
        # a dropped connection, a 5xx / exhausted-429 response, or a non-JSON 2xx body.
        #
        # Unlike LLMUnavailableError above, this is give-up-eligible (issue #26). The
        # asymmetry is deliberate: a *fully* endpoint-wide PROXY outage is caught one
        # level up — list_messages (also a proxy call) fails first, so the whole cycle
        # defers (see the poll loop) and no thread is ever processed or counted. A fault
        # that reaches here is therefore either thread-specific (e.g. a deterministic
        # 5xx on one unserializable thread) or a partial/route-selective degradation
        # (e.g. get_thread garbles while list_messages stays healthy); if it PERSISTS
        # across cycles it must be bounded like Timeout/RuntimeError below, not retried
        # forever. A transient blip clears its count via the poll-loop success-clear
        # before reaching the threshold. The accepted residual is that a *sustained*
        # partial outage can abandon the backlog — but to the findable, re-processable
        # agent/attempted (issue #23), not to lost mail. (An LLM outage, by contrast, is
        # invisible at the cycle level — the proxy is up — so LLMUnavailableError must
        # never give up, or one MLX outage would abandon every person thread.)
        log.warning("api-proxy unavailable processing thread %s: %s", thread_id, exc)
        return await _give_up_if_stuck(thread_id, ids_to_mark, failure_tracker, label_manager, write_sem)
    except httpx.ConnectError as exc:
        # Defensive: a raw ConnectError shouldn't escape the wrapped clients, but if
        # one does it's still a transient outage — retry next cycle, don't give up.
        log.warning("Connection error processing thread %s: %s", thread_id, exc)
        return False
    except TimeoutError as exc:
        # Request-specific slowness (e.g. a transcript too large to prefill within
        # the timeout) — eligible for give-up so one huge thread can't be retried
        # forever. (Connect/pool timeouts are LLMUnavailableError, handled above.)
        log.error("Timeout processing thread %s: %s", thread_id, exc)
        return await _give_up_if_stuck(thread_id, ids_to_mark, failure_tracker, label_manager, write_sem)
    except LLMBalanceError as exc:
        # Account-wide, not a thread fault (and must precede the RuntimeError arm,
        # which it subclasses): don't count toward give-up, don't mark anything —
        # the thread is re-processed after the admin adds funds and restarts.
        log.error("Thread %s deferred — %s", thread_id, exc)
        if halt is not None:
            halt.trip(str(exc))
        return False
    except RuntimeError as exc:
        log.error("Thread %s: %s", thread_id, exc)
        return await _give_up_if_stuck(thread_id, ids_to_mark, failure_tracker, label_manager, write_sem)
    except Exception:
        log.exception("Error processing thread %s", thread_id)
        return await _give_up_if_stuck(thread_id, ids_to_mark, failure_tracker, label_manager, write_sem)


def summarize_cycle(
    thread_items: list[tuple[str, list[str]]],
    results: list,
    failure_tracker: FailureTracker,
) -> tuple[int, list[str]]:
    """Tally a poll cycle's outcomes and update the failure tracker.

    A thread is "handled" when process_single_thread returned True (classified,
    skipped at max priority, or given up — all return True); its failure count is
    cleared. Drains the per-cycle give-up list and prunes counts for threads no
    longer pending. Returns (handled_count, given_up_thread_ids); given_up is a
    subset of the handled count (a give-up also returns True).
    """
    processed = 0
    for (tid, _msg_ids), result in zip(thread_items, results):
        if result is True:
            processed += 1
            failure_tracker.clear(tid)  # success/give-up resets the failure count
    given_up = failure_tracker.take_given_up()
    failure_tracker.prune(tid for tid, _msg_ids in thread_items)
    return processed, given_up


def log_local_deferrals(deferred: list[str]) -> None:
    """One INFO line per cycle when person threads deferred on a local-LLM outage.

    The per-thread handler logs each deferral at DEBUG (issue #24: a closed
    laptop with N person emails used to emit N WARNINGs every cycle for hours);
    this summary is the single visible trace of a routine local outage.
    """
    if deferred:
        log.info(
            "Local LLM offline — deferred %d person email thread(s) this cycle",
            len(deferred),
        )


@dataclass
class IdleState:
    """Tracks the caught-up stretch between busy poll cycles (issue #58)."""

    idle_since: float | None = None
    last_heartbeat: float | None = None

    def reset(self) -> None:
        """Forget the current idle stretch (work arrived, or a cycle failed)."""
        self.idle_since = None
        self.last_heartbeat = None


def idle_report(had_work: bool, now: float, state: IdleState, status_interval: float) -> str | None:
    """One call per successful poll cycle. Mutates state; returns a line to log or None.

    Busy → idle logs a one-shot "caught up"; a continuing idle stretch logs a
    heartbeat every status_interval seconds so "healthy and caught up" stays
    distinguishable from "hung". Failed cycles reset the stretch (the except
    arms call state.reset()) so the heartbeat's minute count never includes
    outage time — "Still idle (Nm) — last poll ok" only measures healthy idling.
    """
    if had_work:
        state.reset()
        return None
    if state.idle_since is None:
        state.idle_since = now
        state.last_heartbeat = now
        return "Inbox caught up — nothing to process"
    if now - state.last_heartbeat >= status_interval:
        state.last_heartbeat = now
        return f"Still idle ({int((now - state.idle_since) / 60)}m) — last poll ok"
    return None


async def verify_labels_with_retry(
    label_manager: LabelManager,
    initial_backoff: int = 5,
    max_backoff: int = 60,
) -> list[str]:
    """Verify required Gmail labels, waiting out a transiently-unreachable proxy.

    The api-proxy may be slow, not yet up, or up-but-still-warming when the
    daemon starts. Two failure modes are transient and worth waiting out with
    capped exponential backoff, instead of letting the daemon exit and crash-loop
    under Docker:

      * a transport fault — connection refused, a connect/read timeout, a
        dropped connection; and
      * a proxy that is reachable but whose Gmail backend is still initializing,
        which answers 5xx (or 429).

    ``verify_labels`` reaches the proxy through ``_send``, which already wraps both
    of those into ``proxy_client.ProxyUnavailableError`` (a ``ProxyError`` subclass),
    so catching ``ProxyError`` covers them. The ``TRANSIENT_TRANSPORT_ERRORS`` prefix
    is kept as defence-in-depth in case a future code path reaches a raw transport
    fault here without going through ``_send``.

    Permanent failures propagate immediately so the operator sees an actionable
    error rather than a silent, endless retry: a misconfigured ``PROXY_URL``
    (``httpx.UnsupportedProtocol`` — itself a ``TransportError`` we deliberately
    do NOT catch), a bad key (``ProxyAuthError``), a blocked op
    (``ProxyForbiddenError``), or a programming error.

    Returns the list of missing label names (see LabelManager.verify_labels).
    """
    proxy_url = label_manager.proxy.proxy_url
    backoff = initial_backoff
    while True:
        try:
            return await label_manager.verify_labels()
        except TRANSIENT_TRANSPORT_ERRORS + (ProxyError,) as exc:
            log.warning(
                "Cannot reach api-proxy at %s (%s) — retrying in %ds",
                proxy_url,
                type(exc).__name__,
                backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


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
        extra_body=config["llm"]["cloud"].get("extra_body"),
        tier="cloud",
    )
    local_llm = LLMClient(
        base_url=os.environ.get("MLX_URL", ""),
        api_key=os.environ.get("MLX_API_KEY", ""),
        model=config["llm"]["local"]["model"],
        max_tokens=config["llm"]["local"]["max_tokens"],
        temperature=config["llm"]["local"]["temperature"],
        timeout=config["llm"]["local"]["timeout"],
        extra_body=config["llm"]["local"].get("extra_body"),
        tier="local",
    )

    classifier = EmailClassifier(
        cloud_llm=cloud_llm,
        local_llm=local_llm,
        config=config,
    )
    label_manager = LabelManager(proxy_client=proxy_client, config=config)

    # Newsletter classifier (if configured)
    nl_config = config.get("newsletter")
    newsletter_classifier = None
    newsletter_recipient = ""
    newsletter_output_file = ""
    if nl_config:
        nl_llm_config = nl_config.get("llm")
        if nl_llm_config:
            nl_base_url, nl_api_key = resolve_newsletter_llm_endpoint()
            nl_llm = LLMClient(
                base_url=nl_base_url,
                api_key=nl_api_key,
                model=nl_llm_config["model"],
                max_tokens=nl_llm_config.get("max_tokens", 1024),
                temperature=nl_llm_config.get("temperature", 0),
                timeout=nl_llm_config.get("timeout", 60),
                extra_body=nl_llm_config.get("extra_body"),
                tier="cloud",
            )
        else:
            nl_llm = cloud_llm
        newsletter_classifier = NewsletterClassifier(cloud_llm=nl_llm, config=config)
        newsletter_recipient = nl_config["recipient"]
        newsletter_output_file = nl_config.get("output_file", "")
        log.info("Newsletter classification enabled for: %s", newsletter_recipient)

    newsletter_only = os.environ.get("NEWSLETTER_ONLY", "").strip().lower() in ("1", "true", "yes")
    if newsletter_only:
        log.info("Newsletter-only mode: non-newsletter threads will be skipped")

    cloud_sem = asyncio.Semaphore(daemon_config.get("cloud_parallel", 2))
    local_parallel = resolve_int_env("LOCAL_PARALLEL", daemon_config.get("local_parallel", 1))
    local_sem = asyncio.Semaphore(local_parallel)
    fetch_sem = asyncio.Semaphore(daemon_config.get("fetch_parallel", 4))
    write_parallel = resolve_int_env("WRITE_PARALLEL", daemon_config.get("write_parallel", 4))
    write_sem = asyncio.Semaphore(write_parallel)
    log.info(
        "Concurrency limits: cloud=%d, local=%d, fetch=%d, write=%d",
        cloud_sem._value, local_sem._value, fetch_sem._value, write_sem._value,
    )
    if local_parallel > 8:
        log.warning(
            "local_parallel=%d exceeds 8 — some MLX servers exhibit KV-cache "
            "cross-contamination at high concurrency (mlx-lm at 16+)",
            local_parallel,
        )

    # Breaks infinite retry loops: a thread that keeps failing for a
    # thread-specific reason (not a transient outage) is marked processed after
    # a few attempts. Session-scoped — counts reset on restart.
    failure_tracker = FailureTracker()

    # Account-level fault switch (provider out of funds): once tripped the poll
    # loop stands down until the admin adds funds and restarts. Session-scoped.
    halt = DaemonHalt()

    # Wait for a transiently-unreachable api-proxy to come up, then verify labels.
    missing = await verify_labels_with_retry(label_manager)
    if missing:
        log.error("Missing Gmail labels: %s", missing)
        log.error("Create these labels manually in Gmail before running the daemon.")
        sys.exit(1)

    log.info("All labels verified. Starting poll loop.")

    poll_interval = daemon_config["poll_interval_seconds"]
    max_emails = resolve_int_env("MAX_EMAILS_PER_CYCLE", daemon_config["max_emails_per_cycle"])
    gmail_query = daemon_config["gmail_query"]
    if newsletter_only and newsletter_recipient:
        gmail_query += f" to:{newsletter_recipient}"
        log.info("Gmail query narrowed to: %s", gmail_query)
    healthcheck_file = Path(daemon_config["healthcheck_file"])
    backoff = poll_interval
    status_interval = daemon_config.get("status_interval_seconds", 900)
    idle_state = IdleState()
    proxy_lost = False  # set by the lost-connection arm, cleared on the next good poll

    while True:
        if halt.tripped:
            # An out-of-funds provider fails EVERY request — polling on would
            # only burn the backlog into agent/attempted. Stand down but stay
            # alive: the heartbeat stays fresh (deliberately halted, not hung),
            # and the instruction repeats at ERROR every cycle so it can't
            # scroll out of the logs. Restarting the daemon is the only reset.
            log.error(
                "Daemon halted — %s. Add funds to the provider account, "
                "then restart the daemon to resume processing.",
                halt.reason,
            )
            healthcheck_file.write_text(str(asyncio.get_event_loop().time()))
            await asyncio.sleep(poll_interval)
            continue
        try:
            response = await proxy_client.list_messages(q=gmail_query, max_results=max_emails)
            messages = response.get("messages", [])

            if proxy_lost:
                # Logged before the cycle's work so the narrative reads
                # lost → reconnected → found/processed (issue #58).
                log.info("Reconnected to api-proxy — resuming normal polling")
                proxy_lost = False

            if messages:
                log.info("Found %d unprocessed message(s)", len(messages))

            # Group messages by threadId
            threads: dict[str, list[str]] = {}
            for msg_stub in messages:
                tid = msg_stub.get("threadId", msg_stub["id"])
                threads.setdefault(tid, []).append(msg_stub["id"])

            if threads:
                log.info("Grouped into %d thread(s)", len(threads))

            max_thread_chars = daemon_config.get("max_thread_chars", DEFAULT_MAX_THREAD_CHARS)
            thread_items = list(threads.items())
            local_deferrals: list[str] = []
            results = await asyncio.gather(
                *(
                    process_single_thread(
                        tid,
                        msg_ids,
                        proxy_client,
                        classifier,
                        label_manager,
                        cloud_sem,
                        local_sem,
                        max_thread_chars,
                        newsletter_classifier=newsletter_classifier,
                        newsletter_recipient=newsletter_recipient,
                        newsletter_output_file=newsletter_output_file,
                        newsletter_only=newsletter_only,
                        failure_tracker=failure_tracker,
                        fetch_sem=fetch_sem,
                        write_sem=write_sem,
                        local_deferrals=local_deferrals,
                        halt=halt,
                    )
                    for tid, msg_ids in thread_items
                ),
                return_exceptions=True,
            )
            processed, given_up = summarize_cycle(thread_items, results, failure_tracker)
            log_local_deferrals(local_deferrals)
            if threads:
                if given_up:
                    log.info(
                        "Processed %d/%d threads (%d of them abandoned after repeated failures: %s)",
                        processed, len(threads), len(given_up), given_up,
                    )
                else:
                    log.info("Processed %d/%d threads", processed, len(threads))

            # Update healthcheck
            healthcheck_file.write_text(str(asyncio.get_event_loop().time()))

            # Reset backoff on success
            backoff = poll_interval

            line = idle_report(
                bool(messages), asyncio.get_event_loop().time(), idle_state, status_interval
            )
            if line:
                log.info(line)

        except TRANSIENT_TRANSPORT_ERRORS + (ProxyUnavailableError,) as exc:
            # A transiently-unreachable proxy (raw transport fault, or a wrapped
            # ProxyUnavailableError — connection/timeout/5xx — from list_messages):
            # back off and retry rather than logging a full traceback.
            log.warning(
                "Lost connection to api-proxy at %s (%s) — retrying in %ds",
                proxy_client.proxy_url,
                type(exc).__name__,
                backoff,
            )
            proxy_lost = True
            backoff = min(backoff * 2, poll_interval * 10)
            # A failed cycle breaks the idle stretch: the next quiet poll logs
            # "caught up" afresh, so heartbeat minutes never include downtime.
            idle_state.reset()
        except ProxyError as exc:
            # A request-specific proxy fault from the cycle-level list_messages call:
            # a 4xx, e.g. a malformed gmail_query 400. It won't fix itself, but it's a
            # known, named condition: log it as a warning and back off rather than
            # spewing a full traceback every cycle. (The transient faults — 5xx, an
            # exhausted 429, a non-JSON 2xx body — are ProxyUnavailableError, handled
            # by the arm above.)
            log.warning(
                "api-proxy rejected the poll request (%s: %s) — retrying in %ds",
                type(exc).__name__, exc, backoff,
            )
            backoff = min(backoff * 2, poll_interval * 10)
            idle_state.reset()
        except Exception:
            log.exception("Error in poll cycle")
            backoff = min(backoff * 2, poll_interval * 10)
            idle_state.reset()

        await asyncio.sleep(backoff)


def main():
    """Entry point."""
    quiet_http_logging()
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
