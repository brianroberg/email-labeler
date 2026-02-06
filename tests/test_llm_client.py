"""Tests for LLM client."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm_client import LLMClient


@pytest.fixture
def cloud_client():
    return LLMClient(
        base_url="https://api.cloud.example.com/v1/chat/completions",
        api_key="sk-test-key",
        model="test-cloud-model",
        max_tokens=8096,
        temperature=0.2,
        timeout=60,
    )


@pytest.fixture
def local_client():
    return LLMClient(
        base_url="http://localhost:8080/v1/chat/completions",
        api_key="",
        model="mlx-community/Qwen3-14B-4bit",
        max_tokens=8096,
        temperature=0.2,
        timeout=120,
    )


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx Response."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "https://example.com"),
    )
    return response


class TestComplete:
    async def test_sends_correct_request_format(self, cloud_client):
        """Verify request body matches OpenAI chat completion format."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "PERSON"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("You are a classifier.", "Classify this email")

            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            assert call_kwargs.args[0] == "https://api.cloud.example.com/v1/chat/completions"

            body = call_kwargs.kwargs["json"]
            assert body["model"] == "test-cloud-model"
            assert body["max_tokens"] == 8096
            assert body["temperature"] == 0.2
            assert len(body["messages"]) == 2
            assert body["messages"][0] == {"role": "system", "content": "You are a classifier."}
            assert body["messages"][1] == {"role": "user", "content": "Classify this email"}

    async def test_sends_auth_header_when_api_key_set(self, cloud_client):
        """Verify Authorization header is sent when api_key is provided."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("sys", "user")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs["headers"]
            assert headers["Authorization"] == "Bearer sk-test-key"

    async def test_no_auth_header_when_no_api_key(self, local_client):
        """Verify no Authorization header when api_key is empty."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "NEEDS_RESPONSE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await local_client.complete("sys", "user")

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs["headers"]
            assert "Authorization" not in headers

    async def test_strips_think_tags(self, cloud_client):
        """Verify <think>...</think> tags are stripped from response."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content":
                "<think>The sender is a company sending a shipping notification.</think>\nSERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert result == "SERVICE"

    async def test_strips_multiple_think_tags(self, cloud_client):
        """Verify multiple think blocks are stripped."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content":
                "<think>first thought</think>\n<think>second thought</think>\nPERSON"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert result == "PERSON"

    async def test_returns_raw_content_without_think_tags(self, cloud_client):
        """Content without think tags is returned as-is (stripped)."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "  PERSON  "}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await cloud_client.complete("sys", "user")
            assert result == "PERSON"

    async def test_raises_on_http_error(self, cloud_client):
        """Non-200 responses raise RuntimeError."""
        mock_response = _mock_response(status_code=500, json_data={"error": "Internal Server Error"})

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="LLM request failed"):
                await cloud_client.complete("sys", "user")

    async def test_raises_on_connection_error(self, cloud_client):
        """Connection errors propagate."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(httpx.ConnectError):
                await cloud_client.complete("sys", "user")

    async def test_uses_configured_timeout(self, cloud_client):
        """Verify timeout is passed to httpx client."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "SERVICE"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await cloud_client.complete("sys", "user")

            mock_client_cls.assert_called_once_with(timeout=60)


class TestIsAvailable:
    async def test_available_when_server_responds(self, local_client):
        """is_available returns True when server returns 200."""
        mock_response = _mock_response(json_data={
            "choices": [{"message": {"content": "ok"}}]
        })

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await local_client.is_available() is True

    async def test_unavailable_on_connection_error(self, local_client):
        """is_available returns False when server is unreachable."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await local_client.is_available() is False

    async def test_unavailable_on_timeout(self, local_client):
        """is_available returns False on timeout."""
        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await local_client.is_available() is False

    async def test_unavailable_on_http_error(self, local_client):
        """is_available returns False on non-200 response."""
        mock_response = _mock_response(status_code=503, json_data={"error": "Service Unavailable"})

        with patch("llm_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await local_client.is_available() is False
