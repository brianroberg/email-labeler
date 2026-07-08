"""Model mocks for the TUI e2e harness.

Only ``evals.newsletter_label`` calls a model: ``LabelApp(..., extract_fn=fn)``
where ``fn(body: str) -> str`` returns the *raw* extractor output that
``newsletter.parse_stories`` parses (the ``STORY: ...`` format). The other three
TUIs (review, edit_tui, newsletter_review) have no model seam.

These fakes are deterministic and offline — no network, no real LLM — and cover
every seed outcome branch: parsed stories, NO_STORIES, unparseable washout, and
a hard extractor failure.
"""


def fake_extract_fn(body: str) -> str:
    """A 'good model': emit one ``STORY:`` block per blank-line-separated paragraph.

    Mirrors the production extraction contract closely enough to drive Phase-A
    seeding: empty/whitespace bodies yield ``NO_STORIES``; otherwise each
    non-empty paragraph (CRLF-tolerant) becomes a ``STORY:`` block.
    """
    normalized = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return "NO_STORIES"
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    if not paragraphs:
        return "NO_STORIES"
    return "\n\n".join(f"STORY: {p}" for p in paragraphs)


def no_stories_extract_fn(body: str) -> str:
    """A model that finds no stories (exercises the NO_STORIES status line)."""
    return "NO_STORIES"


def unparseable_extract_fn(body: str) -> str:
    """A model whose output has no parseable STORY blocks (washout status line)."""
    return "I could not find any stories in the usual format, sorry."


def failing_extract_fn(body: str) -> str:
    """A model call that hard-fails (exercises the 'Seed failed' HintScreen)."""
    raise RuntimeError("simulated extractor outage")


class RecordingExtractor:
    """Wrap an extract_fn and record the bodies it was called with."""

    def __init__(self, fn=fake_extract_fn):
        self._fn = fn
        self.calls: list[str] = []

    def __call__(self, body: str) -> str:
        self.calls.append(body)
        return self._fn(body)
