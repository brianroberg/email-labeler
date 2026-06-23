"""Tests for eval runner parallelism defaults and extra_body overrides."""

import pytest

from daemon import load_config
from evals.run_eval import resolve_extra_body, resolve_parallelism


class TestResolveParallelism:
    def test_cli_override_wins(self):
        """Explicit --parallelism on CLI takes precedence over config."""
        config = load_config()
        assert resolve_parallelism(5, config) == 5

    def test_none_falls_back_to_config(self):
        """When --parallelism is not specified, use cloud_parallel from config."""
        config = load_config()
        expected = config["daemon"]["cloud_parallel"]
        assert resolve_parallelism(None, config) == expected

    def test_none_falls_back_to_default_without_config_key(self):
        """When config has no cloud_parallel, fall back to 1."""
        config = {"daemon": {}}
        assert resolve_parallelism(None, config) == 1


class TestResolveExtraBody:
    def test_none_when_nothing_set(self):
        assert resolve_extra_body(None, False, None) is None

    def test_passes_through_base_when_no_overrides(self):
        assert resolve_extra_body({"top_p": 0.9}, False, None) == {"top_p": 0.9}

    def test_no_think_sets_chat_template_kwargs(self):
        assert resolve_extra_body(None, True, None) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_no_think_preserves_existing_chat_template_kwargs(self):
        out = resolve_extra_body({"chat_template_kwargs": {"foo": 1}}, True, None)
        assert out == {"chat_template_kwargs": {"foo": 1, "enable_thinking": False}}

    def test_explicit_json_merges_and_wins(self):
        out = resolve_extra_body({"top_p": 0.9}, False, '{"top_p": 0.5, "min_p": 0.1}')
        assert out == {"top_p": 0.5, "min_p": 0.1}

    def test_explicit_json_overrides_no_think(self):
        out = resolve_extra_body(None, True, '{"chat_template_kwargs": {"enable_thinking": true}}')
        assert out == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            resolve_extra_body(None, False, "{not json}")

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError):
            resolve_extra_body(None, False, "[1, 2, 3]")

    def test_does_not_mutate_base(self):
        base = {"chat_template_kwargs": {"foo": 1}}
        resolve_extra_body(base, True, None)
        assert base == {"chat_template_kwargs": {"foo": 1}}
