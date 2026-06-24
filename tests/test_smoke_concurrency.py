"""Tests for the mlx_lm.server concurrency smoke-test helpers."""

from scripts.smoke_concurrency import extract, first_message


class TestFirstMessage:
    def test_extracts_message_object(self):
        body = {"choices": [{"message": {"content": "hi"}}]}
        assert first_message(body) == {"content": "hi"}

    def test_empty_choices_falls_back_to_body_without_raising(self):
        # Regression (review finding #11): choices present-but-empty must not
        # IndexError the warmup the way body["choices"][0] would.
        body = {"choices": []}
        assert first_message(body) == body

    def test_missing_choices_falls_back_to_body(self):
        assert first_message({}) == {}


class TestExtract:
    def test_extracts_content(self):
        body = {"choices": [{"message": {"content": "  SERVICE  "}}]}
        assert extract(body) == "SERVICE"

    def test_empty_choices_does_not_raise(self):
        assert extract({"choices": []}).startswith("<no choices")
