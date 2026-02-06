"""LLM client abstraction for cloud and local endpoints.

Supports any OpenAI-compatible chat completion API.
Handles thinking tag stripping for reasoning models.
"""

import re

import httpx


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
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    async def complete(self, system_prompt: str, user_content: str) -> str:
        """Send a chat completion request and return the stripped response.

        Args:
            system_prompt: System message for the LLM.
            user_content: User message content.

        Returns:
            The LLM response with thinking tags stripped.

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
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, headers=headers, json=body)

        if response.status_code != 200:
            raise RuntimeError(f"LLM request failed with status {response.status_code}")

        content = response.json()["choices"][0]["message"]["content"]
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
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self.base_url, headers=headers, json=body)

            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Remove <think>...</think> blocks from LLM output."""
        stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return stripped.strip()
