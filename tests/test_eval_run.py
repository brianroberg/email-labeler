"""Tests for eval runner parallelism defaults and extra_body overrides."""

import argparse

import pytest

from daemon import DEFAULT_MAX_THREAD_CHARS, load_config
from evals.run_eval import (
    apply_config_overrides,
    resolve_extra_body,
    resolve_max_thread_chars,
    resolve_parallelism,
)


def _override_args(**kw):
    """argparse.Namespace with every override field defaulting to None (not provided)."""
    base = dict(
        cloud_model=None, cloud_temperature=None, cloud_max_tokens=None, cloud_timeout=None,
        local_model=None, local_temperature=None, local_max_tokens=None, local_timeout=None,
        cloud_no_think=False, cloud_extra_body=None,
        local_no_think=False, local_extra_body=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _override_config():
    return {"llm": {
        "cloud": {"model": "cloud-default", "temperature": 0.5, "max_tokens": 100, "timeout": 60},
        "local": {"model": "local-default", "temperature": 0.7, "max_tokens": 200, "timeout": 180},
    }}


class TestApplyConfigOverrides:
    def test_no_overrides_leaves_config_unchanged(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args())
        assert cfg == _override_config()

    def test_local_timeout_override(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_timeout=900))
        assert cfg["llm"]["local"]["timeout"] == 900
        assert cfg["llm"]["cloud"]["timeout"] == 60  # cloud untouched

    def test_cloud_timeout_override(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(cloud_timeout=45))
        assert cfg["llm"]["cloud"]["timeout"] == 45
        assert cfg["llm"]["local"]["timeout"] == 180  # local untouched

    def test_timeout_override_does_not_disturb_other_fields(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_timeout=900))
        assert cfg["llm"]["local"]["model"] == "local-default"
        assert cfg["llm"]["local"]["max_tokens"] == 200

    def test_zero_is_a_valid_override(self):
        # 0.0 is not None, so it must override even though it's falsy.
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_temperature=0.0))
        assert cfg["llm"]["local"]["temperature"] == 0.0

    def test_multiple_overrides_at_once(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(
            local_model="m", local_max_tokens=512, local_timeout=600, cloud_timeout=30))
        assert cfg["llm"]["local"]["model"] == "m"
        assert cfg["llm"]["local"]["max_tokens"] == 512
        assert cfg["llm"]["local"]["timeout"] == 600
        assert cfg["llm"]["cloud"]["timeout"] == 30

    def test_no_think_applies_extra_body_in_one_pass(self):
        # extra_body overrides are folded into apply_config_overrides, not a second
        # parallel loop in main() (review finding #10).
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(local_no_think=True))
        assert cfg["llm"]["local"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert "extra_body" not in cfg["llm"]["cloud"]  # cloud untouched

    def test_extra_body_json_is_merged(self):
        cfg = _override_config()
        apply_config_overrides(cfg, _override_args(cloud_extra_body='{"top_p": 0.9}'))
        assert cfg["llm"]["cloud"]["extra_body"] == {"top_p": 0.9}

    def test_invalid_extra_body_json_raises_valueerror(self):
        cfg = _override_config()
        with pytest.raises(ValueError):
            apply_config_overrides(cfg, _override_args(local_extra_body="{not json}"))


class TestResolveMaxThreadChars:
    def test_uses_config_value_when_present(self):
        assert resolve_max_thread_chars({"daemon": {"max_thread_chars": 12345}}) == 12345

    def test_falls_back_to_shared_daemon_default(self):
        assert resolve_max_thread_chars({"daemon": {}}) == DEFAULT_MAX_THREAD_CHARS
        assert resolve_max_thread_chars({}) == DEFAULT_MAX_THREAD_CHARS

    def test_default_is_not_the_stale_50000(self):
        # Regression (finding #9): the eval fallback drifted to 50000 while the
        # daemon/config canonical default moved to 16000.
        assert resolve_max_thread_chars({}) == 16000


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
