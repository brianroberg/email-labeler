"""Tests for shared tui_common helpers."""

from tui_common import truncate


class TestTruncate:
    def test_short_text_unchanged(self):
        assert truncate("hi", 10) == "hi"

    def test_exact_width_unchanged(self):
        assert truncate("abcde", 5) == "abcde"

    def test_long_text_truncated_with_ellipsis(self):
        assert truncate("abcdefghij", 6) == "abc..."

    def test_result_never_exceeds_width(self):
        assert len(truncate("x" * 100, 20)) == 20
