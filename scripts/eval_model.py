#!/usr/bin/env python3
"""One command to evaluate a downloaded model as the local person-email classifier.

Wraps `python -m evals.run_eval` for the common case: evaluate ONLY the local
classifier (Stage 2b on person bodies) and print a report — optionally compared
against a prior run.

    python scripts/eval_model.py qwen/qwen3-14b            # run + report
    python scripts/eval_model.py qwen/qwen3-14b qwen3-8b   # ...also compare vs newest qwen3-8b run
    python scripts/eval_model.py qwen/qwen3-14b evals/results/20250101_....jsonl  # vs explicit file
    python scripts/eval_model.py qwen/qwen3-14b --preflight-timeout 600   # very slow cold load

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

from evals.run_eval import sanitize_tag

DEFAULT_RESULTS_DIR = "evals/results/"


def build_run_eval_command(
    model: str,
    compare_to: str | None,
    skip_preflight: bool = False,
    preflight_timeout: float | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Build the `uv run python -m evals.run_eval` argv for a local-only eval.

    --local-only restricts the run to the local classifier; --local-model sets
    (and, via run_eval's default-tag behaviour, names) the model; --report prints
    metrics inline. The preflight passthroughs let a caller skip or lengthen the
    endpoint check for a server that cold-loads a large model on demand. When set,
    output_dir is forwarded as --output-dir so the run is written where baselines
    are read from.
    """
    cmd = [
        "uv", "run", "python", "-m", "evals.run_eval",
        "--local-only", "--local-model", model, "--report",
    ]
    if skip_preflight:
        cmd.append("--skip-preflight")
    if preflight_timeout is not None:
        cmd += ["--preflight-timeout", str(preflight_timeout)]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    if compare_to:
        cmd += ["--compare-to", compare_to]
    return cmd


def resolve_baseline(token: str | None, results_dir: Path) -> str | None:
    """Resolve a baseline argument to a results file path, or None.

    A token that is an explicit results file (a `.jsonl` path that exists) is used
    as-is. Otherwise it is treated as a model tag: it is run through the same
    sanitize_tag that named the results file, then matched as a whole underscore-
    delimited segment against `*_<tag>_*.jsonl` in *results_dir* (ignoring
    `.cot.jsonl` sidecars); the newest match wins, since timestamp-prefixed names
    sort chronologically.

    Sanitizing first means a tag with glob metacharacters or a '/' (e.g.
    `qwen/qwen3-14b`) searches the same namespace run_eval wrote, and the segment
    anchoring keeps `qwen3` from matching an unrelated `qwen3-14b` run.

    Raises:
        FileNotFoundError: if a tag token matches no results file.
    """
    if not token:
        return None
    # Only an explicit results file is a path; a bare tag never is (avoids a cwd
    # name collision being mistaken for a path).
    path = Path(token)
    if path.suffix == ".jsonl" and path.exists():
        return token
    tag = sanitize_tag(token)
    if not tag:
        raise FileNotFoundError(f"Baseline tag '{token}' is empty after sanitizing")
    matches = sorted(
        p for p in results_dir.glob(f"*_{tag}_*.jsonl")
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
                        help="Directory the run is written to AND baseline tags are "
                             f"looked up in (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip run_eval's pre-run endpoint reachability check")
    parser.add_argument("--preflight-timeout", type=float, metavar="SECONDS",
                        help="Override the pre-run endpoint check timeout (for a slow cold load)")
    args = parser.parse_args(argv)

    try:
        compare_to = resolve_baseline(args.baseline, Path(args.results_dir))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cmd = build_run_eval_command(
        args.model, compare_to,
        skip_preflight=args.skip_preflight,
        preflight_timeout=args.preflight_timeout,
        output_dir=args.results_dir,
    )
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
