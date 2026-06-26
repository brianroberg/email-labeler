#!/usr/bin/env python3
"""One command to evaluate a downloaded model as the local person-email classifier.

Wraps `python -m evals.run_eval` for the common case: evaluate ONLY the local
classifier (Stage 2b on person bodies) and print a report — optionally compared
against a prior run.

    python scripts/eval_model.py qwen/qwen3-14b            # run + report
    python scripts/eval_model.py qwen/qwen3-14b qwen3-8b   # ...also compare vs newest qwen3-8b run
    python scripts/eval_model.py qwen/qwen3-14b evals/results/20250101_....jsonl  # vs explicit file

PREREQUISITES (inherently manual, not done here):
  1. `mlx_lm.server --model <hf-id> --host 0.0.0.0 --port 8080 --temp 0 --prompt-cache-size 2`
     must already be serving the model, and MLX_URL must point at it. The served
     --model must match the <hf-id> passed here or every request 404s.
  2. Stop the daemon first if your MLX server only serves one model at a time.

This delegates to run_eval, which auto-tags the results by the model name,
runs the endpoint preflight check, and (via --report) prints metrics.
"""
import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_RESULTS_DIR = "evals/results/"


def build_run_eval_command(model: str, compare_to: str | None) -> list[str]:
    """Build the `uv run python -m evals.run_eval` argv for a local-only eval.

    --local-only restricts the run to the local classifier; --local-model sets
    (and, via run_eval's default-tag behaviour, names) the model; --report prints
    metrics inline.
    """
    cmd = [
        "uv", "run", "python", "-m", "evals.run_eval",
        "--local-only", "--local-model", model, "--report",
    ]
    if compare_to:
        cmd += ["--compare-to", compare_to]
    return cmd


def resolve_baseline(token: str | None, results_dir: Path) -> str | None:
    """Resolve a baseline argument to a results file path, or None.

    A token that is an existing path is used as-is. Otherwise it is treated as a
    tag substring and matched against `*<token>*.jsonl` in *results_dir*, ignoring
    `.cot.jsonl` chain-of-thought sidecars; the newest match wins (filenames are
    timestamp-prefixed, so a lexical sort is chronological).

    Raises:
        FileNotFoundError: if a tag token matches no results file.
    """
    if not token:
        return None
    if Path(token).exists():
        return token
    matches = sorted(
        p for p in results_dir.glob(f"*{token}*.jsonl")
        if not p.name.endswith(".cot.jsonl")
    )
    if not matches:
        raise FileNotFoundError(
            f"No results file in {results_dir} matching tag '{token}'"
        )
    return str(matches[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a downloaded model as the local person-email classifier."
    )
    parser.add_argument("model", help="HuggingFace model id served by mlx_lm.server")
    parser.add_argument("baseline", nargs="?",
                        help="Prior results file path, or a tag to match the newest run of")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help=f"Where to look up a baseline tag (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args(argv)

    try:
        compare_to = resolve_baseline(args.baseline, Path(args.results_dir))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cmd = build_run_eval_command(args.model, compare_to)
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
