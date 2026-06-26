"""Tests for the one-command-per-model eval wrapper (scripts/eval_model.py)."""

import pytest

from scripts.eval_model import build_run_eval_command, resolve_baseline


class TestBuildRunEvalCommand:
    def test_local_only_run_with_default_tag_and_report(self):
        cmd = build_run_eval_command("qwen/qwen3-14b", compare_to=None)
        assert cmd == [
            "uv", "run", "python", "-m", "evals.run_eval",
            "--local-only", "--local-model", "qwen/qwen3-14b", "--report",
        ]

    def test_includes_compare_to_when_given(self):
        cmd = build_run_eval_command("qwen/qwen3-14b", compare_to="evals/results/prior.jsonl")
        assert cmd[-2:] == ["--compare-to", "evals/results/prior.jsonl"]


class TestResolveBaseline:
    def test_none_returns_none(self, tmp_path):
        assert resolve_baseline(None, tmp_path) is None

    def test_existing_path_returned_as_is(self, tmp_path):
        f = tmp_path / "some_run.jsonl"
        f.write_text("{}\n")
        assert resolve_baseline(str(f), tmp_path) == str(f)

    def test_tag_resolves_to_newest_matching_run(self, tmp_path):
        # Timestamp-prefixed names sort chronologically; newest wins.
        old = tmp_path / "20250101_000000_stage2_only_qwen3_aaaa.jsonl"
        new = tmp_path / "20250202_000000_stage2_only_qwen3_bbbb.jsonl"
        old.write_text("{}\n")
        new.write_text("{}\n")
        assert resolve_baseline("qwen3", tmp_path) == str(new)

    def test_tag_match_ignores_cot_sidecars(self, tmp_path):
        run = tmp_path / "20250101_000000_stage2_only_qwen3_aaaa.jsonl"
        sidecar = tmp_path / "20250101_000000_stage2_only_qwen3_aaaa.cot.jsonl"
        run.write_text("{}\n")
        sidecar.write_text("{}\n")
        assert resolve_baseline("qwen3", tmp_path) == str(run)

    def test_no_match_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nomatch"):
            resolve_baseline("nomatch", tmp_path)
