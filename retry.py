"""Retry utilities with exponential backoff for HTTP requests.

Provides a helper that wraps async callables with retry logic for
transient HTTP errors (429 rate-limit, 502/503/504 server errors).
Respects Retry-After headers when present.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds


def parse_retry_after(response: httpx.Response) -> float | None:
    """Extract delay from Retry-After header (integer seconds only).

    Returns None if the header is absent or unparseable.
    """
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        return None


async def retry_with_backoff(
    request_func: Callable[[], Awaitable[httpx.Response]],
    operation: str = "",
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> httpx.Response:
    """Execute an async HTTP call with exponential backoff on retryable errors.

    On 429/5xx responses, waits and retries up to *max_retries* times.
    If the response includes a Retry-After header, that value is used as the
    delay; otherwise the delay doubles each attempt (with ±25% jitter).

    Args:
        request_func: Zero-arg async callable that returns an httpx.Response.
        operation: Human-readable label for log messages.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial backoff delay in seconds.

    Returns:
        The final httpx.Response (successful or the last retryable failure).
    """
    for attempt in range(max_retries + 1):
        response = await request_func()

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        # Last attempt — return whatever we got
        if attempt == max_retries:
            log.warning(
                "%s: retries exhausted after %d attempts (status %d)",
                operation,
                max_retries + 1,
                response.status_code,
            )
            return response

        retry_after = parse_retry_after(response)
        if retry_after is not None:
            delay = retry_after
        else:
            delay = base_delay * (2**attempt)
            jitter = delay * 0.25 * (2 * random.random() - 1)  # ±25%
            delay += jitter

        log.info(
            "%s: %d (attempt %d/%d), retrying in %.1fs",
            operation,
            response.status_code,
            attempt + 1,
            max_retries + 1,
            delay,
        )
        await asyncio.sleep(delay)

    raise RuntimeError("Unreachable: retry loop exited without returning")
