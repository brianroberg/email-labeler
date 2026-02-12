"""Tests for eval runner parallelism defaults."""

from daemon import load_config
from evals.run_eval import resolve_parallelism


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
