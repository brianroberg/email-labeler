"""Tests for retry utilities."""

from unittest.mock import AsyncMock, patch

import httpx

from retry import (
    BASE_DELAY,
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    parse_retry_after,
    retry_with_backoff,
)


def _mock_response(status_code=200, headers=None):
    """Create a mock httpx Response."""
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("GET", "https://example.com"),
    )


class TestParseRetryAfter:
    def test_integer_seconds(self):
        resp = _mock_response(429, headers={"retry-after": "30"})
        assert parse_retry_after(resp) == 30.0

    def test_float_seconds(self):
        resp = _mock_response(429, headers={"retry-after": "1.5"})
        assert parse_retry_after(resp) == 1.5

    def test_zero(self):
        resp = _mock_response(429, headers={"retry-after": "0"})
        assert parse_retry_after(resp) == 0.0

    def test_negative_clamped_to_zero(self):
        resp = _mock_response(429, headers={"retry-after": "-5"})
        assert parse_retry_after(resp) == 0.0

    def test_missing_header_returns_none(self):
        resp = _mock_response(429)
        assert parse_retry_after(resp) is None

    def test_unparseable_returns_none(self):
        resp = _mock_response(429, headers={"retry-after": "not-a-number"})
        assert parse_retry_after(resp) is None


class TestRetryWithBackoff:
    async def test_success_on_first_attempt(self):
        func = AsyncMock(return_value=_mock_response(200))

        result = await retry_with_backoff(func, "test")

        assert result.status_code == 200
        func.assert_awaited_once()

    async def test_non_retryable_error_returned_immediately(self):
        """4xx errors (except 429) are NOT retried."""
        func = AsyncMock(return_value=_mock_response(400))

        result = await retry_with_backoff(func, "test")

        assert result.status_code == 400
        func.assert_awaited_once()

    async def test_retries_on_429(self):
        func = AsyncMock(side_effect=[
            _mock_response(429),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep") as mock_sleep:
            result = await retry_with_backoff(func, "test")

        assert result.status_code == 200
        assert func.await_count == 2
        mock_sleep.assert_awaited_once()

    async def test_retries_on_502(self):
        func = AsyncMock(side_effect=[
            _mock_response(502),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep"):
            result = await retry_with_backoff(func, "test")

        assert result.status_code == 200
        assert func.await_count == 2

    async def test_retries_on_503(self):
        func = AsyncMock(side_effect=[
            _mock_response(503),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep"):
            result = await retry_with_backoff(func, "test")

        assert result.status_code == 200
        assert func.await_count == 2

    async def test_retries_on_504(self):
        func = AsyncMock(side_effect=[
            _mock_response(504),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep"):
            result = await retry_with_backoff(func, "test")

        assert result.status_code == 200
        assert func.await_count == 2

    async def test_exhausts_retries_and_returns_last_response(self):
        func = AsyncMock(return_value=_mock_response(503))

        with patch("retry.asyncio.sleep"):
            result = await retry_with_backoff(func, "test", max_retries=2)

        assert result.status_code == 503
        assert func.await_count == 3  # initial + 2 retries

    async def test_respects_retry_after_header(self):
        func = AsyncMock(side_effect=[
            _mock_response(429, headers={"retry-after": "7"}),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep") as mock_sleep:
            await retry_with_backoff(func, "test")

        mock_sleep.assert_awaited_once_with(7.0)

    async def test_exponential_backoff_delays(self):
        """Verify delay roughly doubles each attempt (within jitter bounds)."""
        func = AsyncMock(side_effect=[
            _mock_response(429),
            _mock_response(429),
            _mock_response(429),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep") as mock_sleep:
            with patch("retry.random.random", return_value=0.5):  # jitter = 0
                await retry_with_backoff(func, "test", base_delay=2.0)

        delays = [call.args[0] for call in mock_sleep.await_args_list]
        assert delays == [2.0, 4.0, 8.0]

    async def test_multiple_retries_then_success(self):
        func = AsyncMock(side_effect=[
            _mock_response(429),
            _mock_response(503),
            _mock_response(502),
            _mock_response(200),
        ])

        with patch("retry.asyncio.sleep"):
            result = await retry_with_backoff(func, "test")

        assert result.status_code == 200
        assert func.await_count == 4

    async def test_default_constants(self):
        assert MAX_RETRIES == 5
        assert BASE_DELAY == 2.0
        assert RETRYABLE_STATUS_CODES == {429, 502, 503, 504}

    async def test_401_not_retried(self):
        func = AsyncMock(return_value=_mock_response(401))

        result = await retry_with_backoff(func, "test")

        assert result.status_code == 401
        func.assert_awaited_once()

    async def test_403_not_retried(self):
        func = AsyncMock(return_value=_mock_response(403))

        result = await retry_with_backoff(func, "test")

        assert result.status_code == 403
        func.assert_awaited_once()
