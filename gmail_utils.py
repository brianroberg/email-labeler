"""Gmail utility functions for email server v2.

These functions handle parsing of Gmail API message data.
They have no framework or authentication dependencies.
"""

import base64
import html
import re

# Characters that are invisible/zero-width and add no value for classification.
# ZWSP (U+200B), ZWNJ (U+200C), ZWJ (U+200D), BOM (U+FEFF), soft hyphen (U+00AD).
_INVISIBLE_CHARS = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u00ad")
_NO_CONTENT = "(Could not extract text content)"


def get_header(headers: list, name: str) -> str:
    """Extract a header value from Gmail message headers.

    Args:
        headers: List of header dicts with 'name' and 'value' keys
        name: Header name to find (case-insensitive)

    Returns:
        Header value or empty string if not found
    """
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return ""


# ---------------------------------------------------------------------------
# Body cleaning pipeline
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Strip HTML tags from text if the content looks like HTML.

    Uses a simple heuristic: if more than 3 HTML-like opening tags are found,
    treat the whole text as HTML.  Removes ``<style>`` and ``<script>`` blocks
    entirely, replaces remaining tags with a space, and decodes HTML entities.
    """
    if len(re.findall(r"<[a-zA-Z][^>]*>", text)) <= 3:
        return text

    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def shorten_urls(text: str, max_length: int = 80) -> str:
    """Replace URLs longer than *max_length* characters with ``[link]``.

    Common trailing punctuation (``.``, ``,``, ``)``, etc.) is not
    considered part of the URL when measuring length.
    """

    def _replace(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:!?)")
        trailing = match.group(0)[len(url):]
        if len(url) > max_length:
            return "[link]" + trailing
        return match.group(0)

    return re.sub(r"https?://[^\s<>\"{}|\\^`\[\]]+", _replace, text)


def strip_invisible_chars(text: str) -> str:
    """Remove zero-width and other invisible Unicode characters."""
    return text.translate(_INVISIBLE_CHARS)


def collapse_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs of spaces/tabs and excess blank lines."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def clean_body(text: str) -> str:
    """Run the full cleaning pipeline on a decoded email body.

    Steps (in order):
    1. Strip HTML tags if the text looks like raw HTML.
    2. Replace long tracking URLs with ``[link]``.
    3. Remove invisible / zero-width characters.
    4. Collapse excessive whitespace.
    """
    text = strip_html(text)
    text = shorten_urls(text)
    text = strip_invisible_chars(text)
    return collapse_whitespace(text)


def decode_body(payload: dict) -> str:
    """Decode base64url email body from Gmail API payload.

    Handles both simple messages and multipart messages.
    Prefers text/plain content.  The decoded text is run through
    :func:`clean_body` before being returned.

    Args:
        payload: Gmail message payload dict

    Returns:
        Decoded and cleaned email body text
    """
    if "body" in payload and payload["body"].get("data"):
        raw = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        return clean_body(raw)
    if "parts" in payload:
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")
            if mime_type == "text/plain":
                if part["body"].get("data"):
                    raw = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                    return clean_body(raw)
            elif mime_type.startswith("multipart/"):
                result = decode_body(part)
                if result and result != _NO_CONTENT:
                    return result
    return _NO_CONTENT
