"""LLM client abstraction for cloud and local endpoints.

Supports any OpenAI-compatible chat completion API.
Handles thinking tag stripping for reasoning models.
"""

import re

import httpx

from retry import retry_with_backoff


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
            RuntimeError: If the LLM returns a non-200 response.
            httpx.ConnectError: If the server is unreachable.
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

        async def _do_request() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.post(self.base_url, headers=headers, json=body)

        try:
            response = await retry_with_backoff(_do_request, f"LLM {self.model}")
        except httpx.TimeoutException:
            raise TimeoutError(
                f"LLM request to {self.model} timed out after {self.timeout}s"
            ) from None

        if response.status_code != 200:
            raise RuntimeError(f"LLM request failed with status {response.status_code}")

        content = response.json()["choices"][0]["message"]["content"]
        if include_thinking:
            return self._strip_thinking(content), self._extract_thinking(content)
        return self._strip_thinking(content)

    async def is_available(self) -> bool:
        """Check if the LLM endpoint is reachable.

        Sends a minimal completion request to verify connectivity.

        Returns:
            True if the endpoint responds successfully, False otherwise.
        """
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

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self.base_url, headers=headers, json=body)

            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

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
