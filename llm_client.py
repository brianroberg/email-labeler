"""LLM client abstraction for cloud and local endpoints.

Supports any OpenAI-compatible chat completion API.
Handles thinking tag stripping for reasoning models.
"""

import re
from dataclasses import dataclass

import httpx

from retry import retry_with_backoff

# Default timeout (seconds) for the lightweight probe() ping. Longer than
# a typical liveness probe because a server that loads models on demand only
# loads the requested model on the first request — a cold load of a large model
# routinely exceeds 10s, and timing out would wrongly report it unreachable.
DEFAULT_AVAILABILITY_TIMEOUT = 60


@dataclass(frozen=True)
class AvailabilityResult:
    """Result of an endpoint availability probe (issue #41 item 7).

    ``ok`` is True only on HTTP 200. ``status_code`` is the HTTP status when a
    response arrived (None if none did); ``error`` is the exception detail when
    the request failed before a response (connect/timeout/unsupported-protocol).
    """

    ok: bool
    status_code: int | None = None
    error: str | None = None


class LLMUnavailableError(Exception):
    """The LLM endpoint is unreachable or dropped the connection (transient).

    Distinct from request-specific failures: a non-200 response or a read/write
    timeout means the *request* is the problem (too-large input, bad payload) and
    is eligible for the daemon's give-up logic, whereas an unavailable endpoint is
    a transient outage that should simply be retried next cycle.
    """


class LLMContentError(RuntimeError):
    """The model returned no usable ``content`` (issue #30).

    A reasoning model that exhausts max_tokens mid-<think>, a GLM reply carrying
    only ``reasoning_content``, or an empty string all yield an unusable response.
    Retrying as-is won't help, so it is request-specific/permanent and
    give-up-eligible. It subclasses ``RuntimeError`` so the email pipeline's
    ``except RuntimeError`` give-up handler catches it unchanged, while giving the
    newsletter pipeline a dedicated type to re-raise (a *bare* per-story
    RuntimeError stays isolated; a content-less one propagates to give-up).
    """


class LLMClient:
    """Client for OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 8096,
        temperature: float = 0.2,
        timeout: int = 60,
        extra_body: dict | None = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def _is_glm_model(self) -> bool:
        """Check if model uses GLM-style reasoning_content instead of inline think tags."""
        return "glm" in self.model.lower()

    def _extra_body_disables_thinking(self) -> bool:
        """True if extra_body asks to turn thinking off.

        Recognizes both the chat_template_kwargs.enable_thinking form (Qwen / LM
        Studio, what --no-think emits) and a top-level enable_thinking flag. GLM
        ignores those fields, so complete() uses this to set GLM's native thinking
        field to disabled rather than contradicting the request with enabled.
        """
        ctk = self.extra_body.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
            return True
        return self.extra_body.get("enable_thinking") is False

    async def complete(
        self, system_prompt: str, user_content: str, include_thinking: bool = False,
    ) -> str | tuple[str, str]:
        """Send a chat completion request and return the stripped response.

        Args:
            system_prompt: System message for the LLM.
            user_content: User message content.
            include_thinking: If True, return (stripped, thinking) tuple.

        Returns:
            If include_thinking is False: stripped response string.
            If include_thinking is True: (stripped_response, thinking_content) tuple.

        Raises:
            TimeoutError: If the request times out (read/write) — request-specific
                (e.g. input too large to prefill within the timeout).
            LLMUnavailableError: If the endpoint is unreachable (connection refused
                or connect/pool timeout) or the connection drops mid-request —
                transient unavailability.
            RuntimeError: If the LLM returns a non-200 response.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            **self.extra_body,
        }

        # GLM models require an explicit thinking field. Honor a disable request
        # from extra_body (e.g. --no-think, which GLM otherwise ignores) instead of
        # contradicting it with enabled, and never override a `thinking` field the
        # caller set explicitly in extra_body.
        if include_thinking and self._is_glm_model() and "thinking" not in body:
            disabled = self._extra_body_disables_thinking()
            body["thinking"] = {"type": "disabled" if disabled else "enabled"}

        async def _do_request() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.post(self.base_url, headers=headers, json=body)

        try:
            response = await retry_with_backoff(_do_request, f"LLM {self.model}")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # Couldn't establish a connection (server down / unreachable / pool
            # exhausted). This is endpoint *unavailability*, not a problem with this
            # request, so callers retry next cycle and never count it toward giving
            # up on a thread. NOTE: ConnectTimeout/PoolTimeout subclass
            # TimeoutException, so they must be caught before the timeout clause.
            raise LLMUnavailableError(
                f"LLM endpoint {self.model} unavailable ({type(exc).__name__}): {exc}"
            ) from exc
        except httpx.TimeoutException:
            # Connection established but the response was too slow (read/write
            # timeout) — request-specific (e.g. a transcript too large to prefill
            # within the timeout). Surfaced as TimeoutError so the daemon may give
            # up on one oversized thread rather than retry it forever.
            raise TimeoutError(
                f"LLM request to {self.model} timed out after {self.timeout}s"
            ) from None
        except httpx.TransportError as exc:
            # Connection established then dropped/reset mid-request (ReadError,
            # WriteError, RemoteProtocolError, ...): the server crashed, restarted,
            # or flapped — an availability event, also transient. (Some of these
            # stringify to ""; the wrapper guarantees an informative, non-empty msg.)
            raise LLMUnavailableError(
                f"LLM connection to {self.model} dropped ({type(exc).__name__}): {exc}"
            ) from exc

        if response.status_code != 200:
            prompt_chars = len(system_prompt) + len(user_content)
            resp_body = response.text[:500]
            raise RuntimeError(
                f"LLM request failed with status {response.status_code} "
                f"(prompt ~{prompt_chars // 4} tokens, {prompt_chars} chars): {resp_body}"
            )

        msg = response.json()["choices"][0]["message"]
        content = msg.get("content")
        if not content:
            # A reasoning model that exhausts max_tokens mid-<think> (or a GLM reply
            # that returns only reasoning_content) yields a message with no usable
            # `content` — whether that arrives as null, a missing key, or an empty
            # string. Retrying as-is won't help, so this is request-specific/
            # permanent: surface a clear RuntimeError (give-up-eligible via the daemon),
            # never LLMUnavailableError (would retry forever) or a raw KeyError.
            # (An empty string must be caught too: it would otherwise parse to a
            # default SERVICE / LOW_PRIORITY label and silently mislabel the email.)
            has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning"))
            raise LLMContentError(
                f"LLM {self.model} returned no content "
                f"(max_tokens={self.max_tokens} likely exhausted before a final answer; "
                f"reasoning_content {'present' if has_reasoning else 'absent'})"
            )

        if include_thinking:
            # GLM models return reasoning in a separate field
            if self._is_glm_model() and msg.get("reasoning_content"):
                thinking = msg["reasoning_content"]
            else:
                # DeepSeek/Qwen models use inline <think> tags
                thinking = self._extract_thinking(content)
            return self._strip_thinking(content), thinking
        return self._strip_thinking(content)

    async def probe(self, timeout: float | None = None) -> "AvailabilityResult":
        """Probe the endpoint, returning status detail (issue #41 item 7).

        Sends a minimal completion request. ``ok`` is True only on HTTP 200.
        ``status_code`` carries the HTTP status when a response arrived — so a
        404 (which usually means the requested model name doesn't match the
        served one) is distinguishable from an endpoint that is simply down.
        ``error`` carries the exception detail when no response arrived at all.

        Args:
            timeout: Seconds to wait for the ping. Defaults to
                DEFAULT_AVAILABILITY_TIMEOUT, generous enough to cover a server
                that loads the requested model on demand (a cold load can take
                far longer than a normal liveness probe).
        """
        if timeout is None:
            timeout = DEFAULT_AVAILABILITY_TIMEOUT
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            body = {
                "model": self.model,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [{"role": "user", "content": "ping"}],
                **self.extra_body,
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.base_url, headers=headers, json=body)

            return AvailabilityResult(ok=response.status_code == 200, status_code=response.status_code)
        except httpx.RequestError as exc:
            # Any failure to obtain a response means "not available": connect /
            # timeout, but also UnsupportedProtocol (unset/schemeless URL) and
            # read/protocol errors (e.g. a dropped connection mid cold-load).
            return AvailabilityResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    # Matches <think>...</think> and <<think>>...</<think>> (DeepSeek variant)
    _THINK_PATTERN = re.compile(r"<<?think>>?.*?</<?think>>?", flags=re.DOTALL)
    _THINK_EXTRACT = re.compile(r"<<?think>>?(.*?)</<?think>>?", flags=re.DOTALL)

    @staticmethod
    def _extract_thinking(content: str) -> str:
        """Extract all think blocks (handles <think> and <<think>> variants)."""
        matches = LLMClient._THINK_EXTRACT.findall(content)
        return "\n\n".join(m.strip() for m in matches)

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Remove think blocks from LLM output (handles <think> and <<think>> variants)."""
        stripped = LLMClient._THINK_PATTERN.sub("", content)
        return stripped.strip()
