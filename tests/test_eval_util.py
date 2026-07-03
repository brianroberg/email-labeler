"""Tests for shared evals helpers."""

from evals import plural


class TestPlural:
    def test_singular(self):
        assert plural(1, "story", "stories") == "1 story"

    def test_plural(self):
        assert plural(2, "story", "stories") == "2 stories"

    def test_zero_is_plural(self):
        assert plural(0, "error", "errors") == "0 errors"

    def test_default_plural_appends_s(self):
        assert plural(3, "row") == "3 rows"
        assert plural(1, "row") == "1 row"
