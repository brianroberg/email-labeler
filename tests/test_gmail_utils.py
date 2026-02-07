"""Tests for Gmail utility functions (header parsing and body cleaning)."""

import base64

from gmail_utils import (
    clean_body,
    collapse_whitespace,
    decode_body,
    get_header,
    shorten_urls,
    strip_html,
    strip_invisible_chars,
)

# ---------------------------------------------------------------------------
# get_header
# ---------------------------------------------------------------------------

def test_get_header_case_insensitive(sample_headers):
    assert get_header(sample_headers, "from") == "John Doe <john@example.com>"
    assert get_header(sample_headers, "FROM") == "John Doe <john@example.com>"


def test_get_header_missing(sample_headers):
    assert get_header(sample_headers, "X-Custom") == ""


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags():
    html = "<html><body><p>Hello <b>world</b></p></body></html>"
    result = strip_html(html)
    assert "<" not in result
    assert "Hello" in result
    assert "world" in result


def test_strip_html_removes_style_blocks():
    html = (
        '<html><head><style type="text/css">.cls{color:red;}</style></head>'
        "<body><p>Content</p></body></html>"
    )
    result = strip_html(html)
    assert "color:red" not in result
    assert "Content" in result


def test_strip_html_removes_script_blocks():
    html = "<html><script>alert('xss')</script><body><p>Safe</p></body></html>"
    result = strip_html(html)
    assert "alert" not in result
    assert "Safe" in result


def test_strip_html_unescapes_entities():
    html = (
        "<html><body><p>Price: &lt;$50 &amp; free &quot;shipping&quot;</p>"
        "<br><div>More</div></body></html>"
    )
    result = strip_html(html)
    assert '<$50 & free "shipping"' in result


def test_strip_html_leaves_text_with_few_tags():
    """Text with 3 or fewer tags should pass through unchanged."""
    text = "Price is <$50 and <$30"
    assert strip_html(text) == text


def test_strip_html_leaves_plain_text():
    text = "Just a normal email body with no HTML at all."
    assert strip_html(text) == text


# ---------------------------------------------------------------------------
# shorten_urls
# ---------------------------------------------------------------------------

def test_shorten_urls_replaces_long_urls():
    long_url = "https://example.com/track?session=" + "x" * 100
    text = f"Track at {long_url}"
    result = shorten_urls(text)
    assert "[link]" in result
    assert long_url not in result


def test_shorten_urls_keeps_short_urls():
    text = "Visit https://example.com/page"
    result = shorten_urls(text)
    assert "https://example.com/page" in result
    assert "[link]" not in result


def test_shorten_urls_handles_mixed():
    short = "https://a.co"
    long = "https://example.com/" + "x" * 100
    text = f"Short {short} and long {long} end"
    result = shorten_urls(text)
    assert short in result
    assert "[link]" in result
    assert long not in result


def test_shorten_urls_custom_max_length():
    url = "https://example.com/medium-path"
    assert "[link]" not in shorten_urls(url, max_length=80)
    assert "[link]" in shorten_urls(url, max_length=10)


def test_shorten_urls_preserves_surrounding_text():
    text = "Click here ( https://example.com/" + "x" * 100 + " ) for details"
    result = shorten_urls(text)
    assert result.startswith("Click here ( [link]")
    assert result.endswith(") for details")


def test_shorten_urls_strips_trailing_punctuation():
    long_url = "https://example.com/" + "x" * 100
    assert shorten_urls(f"See {long_url}.") == "See [link]."
    assert shorten_urls(f"See {long_url},") == "See [link],"
    assert shorten_urls(f"({long_url})") == "([link])"


# ---------------------------------------------------------------------------
# strip_invisible_chars
# ---------------------------------------------------------------------------

def test_strip_invisible_removes_zwsp():
    assert strip_invisible_chars("Hello\u200Bworld") == "Helloworld"


def test_strip_invisible_removes_zwnj():
    assert strip_invisible_chars("a\u200cb\u200dc") == "abc"


def test_strip_invisible_removes_bom():
    assert strip_invisible_chars("\uFEFFHello") == "Hello"


def test_strip_invisible_removes_soft_hyphen():
    assert strip_invisible_chars("copy\u00ADright") == "copyright"


def test_strip_invisible_leaves_normal_text():
    text = "Normal text with spaces"
    assert strip_invisible_chars(text) == text


# ---------------------------------------------------------------------------
# collapse_whitespace
# ---------------------------------------------------------------------------

def test_collapse_whitespace_multiple_spaces():
    assert collapse_whitespace("Hello    world") == "Hello world"


def test_collapse_whitespace_tabs():
    assert collapse_whitespace("Hello\t\tworld") == "Hello world"


def test_collapse_whitespace_preserves_paragraph_breaks():
    assert collapse_whitespace("Line 1\n\nLine 2") == "Line 1\n\nLine 2"


def test_collapse_whitespace_reduces_excess_newlines():
    assert collapse_whitespace("Line 1\n\n\n\n\nLine 2") == "Line 1\n\nLine 2"


def test_collapse_whitespace_strips_line_edges():
    assert collapse_whitespace("  Line 1  \n  Line 2  ") == "Line 1\nLine 2"


# ---------------------------------------------------------------------------
# clean_body  (integration of the full pipeline)
# ---------------------------------------------------------------------------

def test_clean_body_html_with_tracking_urls():
    """Realistic marketing email with HTML tags and tracking URLs."""
    dirty = (
        '<html><body><p>Hello world</p>'
        '<a href="https://track.example.com/click?id=' + "a" * 200 + '">Learn more</a>'
        "</body></html>"
    )
    result = clean_body(dirty)
    assert "<" not in result
    assert "[link]" not in result or "Hello world" in result
    assert "Hello world" in result
    assert "Learn more" in result


def test_clean_body_invisible_chars_and_whitespace():
    text = "From\u200B party\u200C snacks\u200D   and\u00AD pizza\n\n\n\n\nto game-day fits"
    result = clean_body(text)
    assert "\u200B" not in result
    assert "\u200C" not in result
    assert "   " not in result
    assert result.count("\n\n") <= 1


def test_clean_body_plain_text_passthrough():
    text = "Just a normal email from a friend."
    assert clean_body(text) == text


def test_clean_body_venmo_style():
    """Simulates the Venmo-like email with tracking URLs and invisible chars."""
    tracking_url = "https://ablinks.email.venmo.com/ls/click?upn=u001." + "X" * 300
    text = (
        "From party snacks and pizza to game-day fits, Venmo's got your\n"
        "Sunday covered.\u200b \u200c \ufeff \u200b \u200c \ufeff\n"
        f"Football is hard, Venmo is easy.\n( {tracking_url} )\n"
        "Football is hard, Venmo is easy."
    )
    result = clean_body(text)
    assert "[link]" in result
    assert "\u200b" not in result
    assert "\ufeff" not in result
    assert "Football is hard, Venmo is easy." in result


# ---------------------------------------------------------------------------
# decode_body  (integration with cleaning)
# ---------------------------------------------------------------------------

def _encode(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def test_decode_body_simple(sample_message):
    result = decode_body(sample_message["payload"])
    assert "Hey, can we meet tomorrow at 3pm" in result


def test_decode_body_cleans_html():
    html = "<html><body><p>Hello</p><p>World</p></body></html>"
    payload = {"body": {"data": _encode(html)}}
    result = decode_body(payload)
    assert "<" not in result
    assert "Hello" in result
    assert "World" in result


def test_decode_body_cleans_tracking_urls():
    long_url = "https://example.com/track?session=" + "x" * 200
    text = f"Track here: {long_url}"
    payload = {"body": {"data": _encode(text)}}
    result = decode_body(payload)
    assert "[link]" in result


def test_decode_body_multipart_prefers_plain():
    plain = "Plain text version"
    html = "<html><body><p>HTML version</p></body></html>"
    payload = {
        "parts": [
            {"mimeType": "text/html", "body": {"data": _encode(html)}},
            {"mimeType": "text/plain", "body": {"data": _encode(plain)}},
        ]
    }
    result = decode_body(payload)
    assert result == "Plain text version"


def test_decode_body_nested_multipart_skips_empty():
    """Multipart with an empty sub-multipart should continue to text/plain."""
    plain = "The actual content"
    payload = {
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/html", "body": {"data": _encode("<b>html only</b>")}},
                ],
            },
            {"mimeType": "text/plain", "body": {"data": _encode(plain)}},
        ]
    }
    result = decode_body(payload)
    assert result == "The actual content"


def test_decode_body_empty_payload():
    assert decode_body({}) == "(Could not extract text content)"
