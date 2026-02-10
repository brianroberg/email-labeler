"""Data loading layer for the eval web UI.

Wraps evals.report functions to provide structured data for templates.
"""

import json
from pathlib import Path

from evals.report import compute_metrics, load_golden_context, load_results
from evals.schemas import RunMeta, ThinkingEntry


def list_runs(results_dir: Path) -> list[tuple[Path, RunMeta]]:
    """List all eval runs with metadata, newest first."""
    runs = []
    for path in sorted(results_dir.glob("*.jsonl"), reverse=True):
        if path.name.endswith(".cot.jsonl"):
            continue
        try:
            meta, _ = load_results(path)
            runs.append((path, meta))
        except Exception:
            continue
    return runs


def filter_runs(
    runs: list[tuple[Path, RunMeta]],
    cloud_model: str | None = None,
    local_model: str | None = None,
    stages: str | None = None,
    tag: str | None = None,
) -> list[tuple[Path, RunMeta]]:
    """Filter runs by metadata fields."""
    filtered = runs
    if cloud_model:
        filtered = [(p, m) for p, m in filtered if m.cloud_model == cloud_model]
    if local_model:
        filtered = [(p, m) for p, m in filtered if m.local_model == local_model]
    if stages:
        filtered = [(p, m) for p, m in filtered if m.stages == stages]
    if tag:
        filtered = [(p, m) for p, m in filtered if tag.lower() in (m.tag or "").lower()]
    return filtered


def unique_values(runs: list[tuple[Path, RunMeta]]) -> dict[str, list[str]]:
    """Extract unique filter values from all runs for dropdown population."""
    return {
        "cloud_models": sorted({m.cloud_model for _, m in runs}),
        "local_models": sorted({m.local_model for _, m in runs}),
        "stages": sorted({m.stages for _, m in runs}),
    }


def load_run_detail(path: Path) -> dict:
    """Load run with metrics, context, and duration stats."""
    meta, predictions = load_results(path)
    metrics = compute_metrics(predictions)
    context = load_golden_context(meta)

    valid = [p for p in predictions if p.error is None and p.duration_seconds > 0]
    avg_duration = sum(p.duration_seconds for p in valid) / len(valid) if valid else 0.0

    return {
        "meta": meta,
        "predictions": predictions,
        "metrics": metrics,
        "context": context,
        "avg_duration": avg_duration,
    }


def load_thinking_sidecar(results_path: Path) -> dict[str, ThinkingEntry]:
    """Load chain-of-thought sidecar if it exists."""
    sidecar_path = results_path.with_suffix(".cot.jsonl")
    if not sidecar_path.exists():
        return {}
    cot_map: dict[str, ThinkingEntry] = {}
    with open(sidecar_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ThinkingEntry.from_dict(json.loads(line))
                cot_map[entry.thread_id] = entry
            except Exception:
                continue
    return cot_map


def compare_runs(baseline_path: Path, compare_paths: list[Path]) -> dict:
    """Compare a baseline run against one or more other runs."""
    baseline_meta, baseline_pred = load_results(baseline_path)
    baseline_metrics = compute_metrics(baseline_pred)

    comparisons = []
    for path in compare_paths:
        meta, pred = load_results(path)
        metrics = compute_metrics(pred)

        deltas: dict = {}
        for section in ("stage1", "stage2", "combined"):
            if section in baseline_metrics and section in metrics:
                b_acc = baseline_metrics[section].get("accuracy")
                c_acc = metrics[section].get("accuracy")
                deltas[section] = {
                    "accuracy": c_acc - b_acc if b_acc is not None and c_acc is not None else None,
                }
                if section == "stage1":
                    deltas[section]["privacy_violations"] = (
                        metrics[section]["privacy_violations"]
                        - baseline_metrics[section]["privacy_violations"]
                    )

        comparisons.append({
            "path": path,
            "meta": meta,
            "metrics": metrics,
            "deltas": deltas,
        })

    return {
        "baseline_path": baseline_path,
        "baseline_meta": baseline_meta,
        "baseline_metrics": baseline_metrics,
        "comparisons": comparisons,
    }
