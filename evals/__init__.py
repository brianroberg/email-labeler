"""Classification evaluation suite for the email labeler."""

import httpx

from proxy_client import ProxyAuthError, ProxyError, ProxyForbiddenError


def format_network_error(exc: Exception, service: str = "service") -> str:
    """Format a network exception into a human-readable error message.

    Args:
        exc: The caught exception.
        service: Name of the service being contacted (e.g. "api-proxy", "cloud LLM").

    Returns:
        A one-line error message suitable for printing to stderr.
    """
    if isinstance(exc, httpx.ConnectError):
        msg = str(exc)
        if "Name or service not known" in msg or "getaddrinfo failed" in msg:
            return f"DNS lookup failed for {service} — is the hostname correct? ({msg})"
        if "Connection refused" in msg:
            return f"Connection refused by {service} — is the service running? ({msg})"
        return f"Could not connect to {service}: {msg}"
    if isinstance(exc, httpx.TimeoutException):
        return f"Request to {service} timed out: {exc}"
    if isinstance(exc, ProxyAuthError):
        return f"Authentication failed with {service}: {exc}"
    if isinstance(exc, ProxyForbiddenError):
        return f"Request forbidden by {service}: {exc}"
    if isinstance(exc, ProxyError):
        return f"Error from {service}: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from {service}: {exc}"
    return f"Unexpected error contacting {service}: {exc}"
