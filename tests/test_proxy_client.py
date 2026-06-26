"""Tests for the Gmail proxy client's error classification.

Focus: the transient-vs-permanent split (issue #16). The proxy must raise a
transient ProxyUnavailableError for endpoint-unavailability (connection refused,
timeouts, dropped/garbled connections, and 5xx server errors) so the daemon can
defer-and-retry, while request-specific failures (4xx) stay ProxyError so they
remain give-up-eligible. A permanent misconfiguration (an unsupported PROXY_URL
scheme) must propagate, never be retried as if transient.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from proxy_client import (
    GmailProxyClient,
    ProxyAuthError,
    ProxyError,
    ProxyUnavailableError,
)


@pytest.fixture
def client():
    return GmailProxyClient(proxy_url="http://proxy.test", api_key="test-key")


def _mock_response(status_code=200, json_data=None):
    return httpx.Response(
        status_code=status_code,
        json=json_data if json_data is not None else {},
        request=httpx.Request("GET", "http://proxy.test/x"),
    )


def _patch_transport(get_return=None, get_side_effect=None):
    """Patch proxy_client.httpx.AsyncClient so client.get/post is controllable."""
    mock_client_cls = patch("proxy_client.httpx.AsyncClient").start()
    mock_client = AsyncMock()
    if get_side_effect is not None:
        mock_client.get.side_effect = get_side_effect
        mock_client.post.side_effect = get_side_effect
    else:
        mock_client.get.return_value = get_return
        mock_client.post.return_value = get_return
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client_cls


class TestTransientClassification:
    def teardown_method(self):
        patch.stopall()

    async def test_unavailable_is_subclass_of_proxy_error(self):
        """ProxyUnavailableError specializes ProxyError (whose docstring already
        covers '5xx, connection errors'), so existing `except ProxyError` sites
        (e.g. startup label verification) keep catching the transient cases."""
        assert issubclass(ProxyUnavailableError, ProxyError)

    async def test_5xx_raises_unavailable(self, client):
        """A 500 server error is transient → ProxyUnavailableError (not bare ProxyError)."""
        _patch_transport(get_return=_mock_response(500, {"message": "boom"}))
        with pytest.raises(ProxyUnavailableError):
            await client.get_thread("t1")

    async def test_4xx_stays_proxy_error(self, client):
        """A 404 is request-specific → plain ProxyError (give-up-eligible), NOT transient."""
        _patch_transport(get_return=_mock_response(404, {"message": "not found"}))
        with pytest.raises(ProxyError) as exc_info:
            await client.get_thread("missing")
        # Must be the base class, not the transient subclass.
        assert not isinstance(exc_info.value, ProxyUnavailableError)

    async def test_429_is_unavailable(self, client):
        """A 429 (rate-limit) that survives retry.py's retry budget is a sustained
        throttle — transient by nature (it clears on its own), so it must classify as
        ProxyUnavailableError and be retried next cycle, NOT grouped with the permanent
        4xx and routed to the give-up path. Asserted at _handle_response because a 429
        only reaches it once the HTTP-layer retries are exhausted."""
        with pytest.raises(ProxyUnavailableError):
            client._handle_response(_mock_response(429, {"message": "rate limited"}))

    async def test_non_json_2xx_is_unavailable(self, client):
        """A 2xx whose body isn't JSON — a truncated/garbled response, or an upstream
        gateway briefly returning an HTML error page with status 200 — is a transient
        hiccup. It classifies as ProxyUnavailableError (retried next cycle), NOT a
        give-up-eligible plain ProxyError. With the #26 give-up bound a *persistent*
        non-JSON 2xx is still bounded by the FailureTracker, so this can't retry forever.
        Asserted at _handle_response, where the success body is parsed (issue #27)."""
        resp = httpx.Response(
            200, text="<html>502 Bad Gateway</html>",
            request=httpx.Request("GET", "http://proxy.test/x"),
        )
        with pytest.raises(ProxyUnavailableError):
            client._handle_response(resp)

    async def test_read_timeout_raises_unavailable(self, client):
        """A read timeout to the proxy is infrastructure slowness → transient.

        (Unlike the LLM client, where a read timeout means oversized prefill; a
        proxy read timeout is just Gmail/proxy being slow.)"""
        _patch_transport(get_side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProxyUnavailableError):
            await client.get_thread("t1")

    async def test_connect_error_raises_unavailable(self, client):
        """Connection refused (proxy down/restarting) → transient."""
        _patch_transport(get_side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProxyUnavailableError):
            await client.get_thread("t1")

    async def test_remote_protocol_error_raises_unavailable(self, client):
        """A dropped/garbled connection (RemoteProtocolError) → transient."""
        _patch_transport(get_side_effect=httpx.RemoteProtocolError("server disconnected"))
        with pytest.raises(ProxyUnavailableError):
            await client.get_thread("t1")

    async def test_local_protocol_error_propagates_not_transient(self, client):
        """A LocalProtocolError (permanent CLIENT-side request-construction fault — e.g.
        an illegal header value built from our own inputs) is NOT a transient outage.

        It is a sibling of RemoteProtocolError under httpx.ProtocolError, but unlike its
        sibling it must propagate (like UnsupportedProtocol) so it surfaces / is
        give-up-eligible, rather than being wrapped as ProxyUnavailableError and retried
        forever (issue #26)."""
        _patch_transport(get_side_effect=httpx.LocalProtocolError("illegal header value"))
        with pytest.raises(httpx.LocalProtocolError):
            await client.get_thread("t1")

    async def test_unsupported_protocol_propagates_not_transient(self, client):
        """A bad PROXY_URL scheme (UnsupportedProtocol) is a permanent misconfig and
        must propagate, never be wrapped/retried as transient."""
        _patch_transport(get_side_effect=httpx.UnsupportedProtocol("bad scheme"))
        with pytest.raises(httpx.UnsupportedProtocol):
            await client.get_thread("t1")

    async def test_401_still_raises_auth_error(self, client):
        """The 5xx split must not disturb the 401 → ProxyAuthError mapping."""
        _patch_transport(get_return=_mock_response(401, {"message": "bad key"}))
        with pytest.raises(ProxyAuthError):
            await client.get_thread("t1")
