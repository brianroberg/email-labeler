"""Compute and display metrics from newsletter evaluation results.

Usage:
    python -m evals.newsletter_report --results evals/newsletter_results/run.jsonl
    python -m evals.newsletter_report --compare run1.jsonl run2.jsonl
    python -m evals.newsletter_report --results-dir evals/newsletter_results/
    python -m evals.newsletter_report --results run.jsonl --verbose
    python -m evals.newsletter_report --results run.jsonl --format json
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from evals.newsletter_schemas import (
    ExtractionPrediction,
    NewsletterRunMeta,
    StoryPrediction,
)
from evals.report import (
    compute_confusion_matrix,
    compute_precision_recall_f1,
    format_confusion_matrix,
    format_pct,
    format_per_class_table,
    format_table_row,
)

TIERS = ["excellent", "good", "fair", "poor"]
DIMENSIONS = ["simple", "concrete", "personal", "dynamic"]
THEMES = ["scripture", "christlikeness", "church", "vocation_family", "disciple_making"]


def _dimension_pairs(results: list[StoryPrediction], dim: str) -> list[tuple[int, int]]:
    """(expected, predicted) pairs for a dimension over stories where both are present."""
    pairs: list[tuple[int, int]] = []
    for r in results:
        if r.expected_scores is None or r.predicted_scores is None:
            continue
        if dim not in r.expected_scores or dim not in r.predicted_scores:
            continue
        pairs.append((r.expected_scores[dim], r.predicted_scores[dim]))
    return pairs


def compute_dimension_mae(results: list[StoryPrediction]) -> dict[str, float]:
    """Mean absolute error per dimension over stories with both scores present."""
    out: dict[str, float] = {}
    for dim in DIMENSIONS:
        pairs = _dimension_pairs(results, dim)
        if pairs:
            out[dim] = sum(abs(e - p) for e, p in pairs) / len(pairs)
        else:
            out[dim] = None
    return out


def compute_dimension_exact_match(results: list[StoryPrediction]) -> dict[str, float]:
    """Fraction of exact-match predictions per dimension."""
    out: dict[str, float] = {}
    for dim in DIMENSIONS:
        pairs = _dimension_pairs(results, dim)
        if pairs:
            out[dim] = sum(1 for e, p in pairs if e == p) / len(pairs)
        else:
            out[dim] = None
    return out


def compute_dimension_within1(results: list[StoryPrediction]) -> dict[str, float]:
    """Fraction of predictions within 1 point of expected per dimension."""
    out: dict[str, float] = {}
    for dim in DIMENSIONS:
        pairs = _dimension_pairs(results, dim)
        if pairs:
            out[dim] = sum(1 for e, p in pairs if abs(e - p) <= 1) / len(pairs)
        else:
            out[dim] = None
    return out


def format_metric_delta(a: float | None, b: float | None) -> str:
    """Delta for a higher-is-better metric (accuracy, F1), formatted as a percent."""
    if a is None or b is None:
        return "N/A"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1%}"


def format_mae_delta(a: float | None, b: float | None) -> str:
    """Delta for MAE (lower-is-better).

    Reports the raw signed change (b - a) and flags a decrease as an improvement.
    """
    if a is None or b is None:
        return "N/A"
    d = b - a
    sign = "+" if d >= 0 else ""
    label = " (improved)" if d < 0 else (" (worse)" if d > 0 else "")
    return f"{sign}{d:.2f}{label}"


def compute_all_metrics(
    story_results: list[StoryPrediction],
    extraction_results: list[ExtractionPrediction] | None = None,
    match_threshold: float = 0.6,
) -> dict:
    """Bundle every metric section for a single run."""
    return {
        "tier": compute_tier_metrics(story_results),
        "dimension_mae": compute_dimension_mae(story_results),
        "dimension_exact": compute_dimension_exact_match(story_results),
        "dimension_within1": compute_dimension_within1(story_results),
        "themes": compute_multilabel_metrics(story_results, THEMES),
        "extraction": compute_extraction_metrics(
            extraction_results or [], threshold=match_threshold
        ),
    }


def compute_extraction_metrics(
    results: list[ExtractionPrediction],
    threshold: float = 0.6,
) -> dict:
    """Extraction metrics aggregated over newsletters.

    Per newsletter: precision = matched/predicted, recall = matched/golden via
    greedy one-to-one match_stories. Aggregate as micro (pooled counts) and macro
    (mean of per-newsletter precision/recall).
    """
    total_matched = total_pred = total_gold = 0
    per_nl_precision: list[float] = []
    per_nl_recall: list[float] = []
    per_newsletter: list[dict] = []

    for r in results:
        if r.error is not None:
            continue
        matched, n_pred, n_gold = match_stories(
            r.predicted_stories, r.golden_stories, threshold=threshold
        )
        total_matched += matched
        total_pred += n_pred
        total_gold += n_gold
        p = matched / n_pred if n_pred > 0 else 0.0
        rec = matched / n_gold if n_gold > 0 else 0.0
        per_nl_precision.append(p)
        per_nl_recall.append(rec)
        per_newsletter.append({
            "thread_id": r.thread_id,
            "matched": matched,
            "predicted": n_pred,
            "golden": n_gold,
            "precision": p,
            "recall": rec,
        })

    micro_p = total_matched / total_pred if total_pred > 0 else 0.0
    micro_r = total_matched / total_gold if total_gold > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    macro_p = sum(per_nl_precision) / len(per_nl_precision) if per_nl_precision else 0.0
    macro_r = sum(per_nl_recall) / len(per_nl_recall) if per_nl_recall else 0.0
    macro_f1 = (
        2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    )

    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "count": len(per_newsletter),
        "per_newsletter": per_newsletter,
    }


def compute_multilabel_metrics(
    results: list[StoryPrediction],
    themes: list[str],
) -> dict:
    """Multi-label theme metrics.

    Each theme is an independent binary label (present/absent) derived from the
    expected_themes vs predicted_themes sets. Parse-failure rows (predicted_scores
    is None) are excluded. Returns per-theme P/R/F1, micro-F1, macro-F1, and
    exact-set-match (fraction where set(expected) == set(predicted)).
    """
    scored = [r for r in results if r.predicted_scores is not None]

    per_theme: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0
    for theme in themes:
        tp = fp = fn = 0
        for r in scored:
            exp = theme in set(r.expected_themes or [])
            pred = theme in set(r.predicted_themes or [])
            if exp and pred:
                tp += 1
            elif pred and not exp:
                fp += 1
            elif exp and not pred:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_theme[theme] = {"precision": precision, "recall": recall, "f1": f1}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    macro_f1 = sum(per_theme[t]["f1"] for t in themes) / len(themes) if themes else 0.0

    exact = sum(
        1 for r in scored if set(r.expected_themes or []) == set(r.predicted_themes or [])
    )
    exact_set_match = exact / len(scored) if scored else None

    return {
        "per_theme": per_theme,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "exact_set_match": exact_set_match,
        "count": len(scored),
    }


def compute_tier_metrics(results: list[StoryPrediction]) -> dict:
    """Tier metrics over the 4 tier classes.

    Rows whose predicted_scores is None are parse failures: counted as errors
    and excluded from the confusion matrix / accuracy.
    """
    errors = [r for r in results if r.predicted_scores is None]
    valid = [r for r in results if r.predicted_scores is not None]

    matrix = compute_confusion_matrix(valid, "expected_tier", "predicted_tier", TIERS)
    prf = compute_precision_recall_f1(matrix, TIERS)

    scored = [r for r in valid if r.expected_tier in TIERS and r.predicted_tier in TIERS]
    correct = sum(1 for r in scored if r.expected_tier == r.predicted_tier)
    accuracy = correct / len(scored) if scored else None

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "per_class": prf,
        "count": len(scored),
        "errors": len(errors),
    }


def _normalize(story: dict) -> str:
    """Lowercase, whitespace-collapse the story's title-or-text for matching."""
    raw = story.get("title") or story.get("text") or ""
    return re.sub(r"\s+", " ", raw).strip().lower()


def match_stories(
    predicted: list[dict],
    golden: list[dict],
    threshold: float = 0.6,
) -> tuple[int, int, int]:
    """Greedy one-to-one match of predicted stories to golden stories.

    Uses difflib.SequenceMatcher ratio on normalized (lowercased,
    whitespace-collapsed) title-or-text. Highest-ratio candidate pairs are
    assigned first; each predicted and each golden story matches at most once.

    Returns:
        (matched, n_predicted, n_golden)
    """
    pred_norm = [_normalize(p) for p in predicted]
    gold_norm = [_normalize(g) for g in golden]

    # Score every predicted/golden pair, then assign greedily by descending ratio.
    pairs: list[tuple[float, int, int]] = []
    for pi, pn in enumerate(pred_norm):
        for gi, gn in enumerate(gold_norm):
            ratio = difflib.SequenceMatcher(None, pn, gn).ratio()
            if ratio >= threshold:
                pairs.append((ratio, pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched = 0
    for _ratio, pi, gi in pairs:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matched += 1

    return matched, len(predicted), len(golden)


# --- Loading ---

def load_results(
    path: Path,
) -> tuple[NewsletterRunMeta, list[StoryPrediction], list[ExtractionPrediction]]:
    """Load run metadata and predictions from a newsletter results JSONL.

    Dispatches rows on the "type" discriminator.
    """
    meta = None
    story_preds: list[StoryPrediction] = []
    extraction_preds: list[ExtractionPrediction] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("type")
            if t == "run_meta":
                meta = NewsletterRunMeta.from_dict(d)
            elif t == "story_prediction":
                story_preds.append(StoryPrediction.from_dict(d))
            elif t == "extraction_prediction":
                extraction_preds.append(ExtractionPrediction.from_dict(d))
    if meta is None:
        raise ValueError(f"No run_meta found in {path}")
    return meta, story_preds, extraction_preds


# --- Formatters ---

def _format_dim_table(
    mae: dict[str, float | None],
    exact: dict[str, float | None],
    within1: dict[str, float | None],
) -> str:
    widths = [max(len(d) for d in DIMENSIONS), 10, 12, 12]
    lines = []
    lines.append("  " + format_table_row(["Dimension", "MAE", "Exact", "Within-1"], widths))
    lines.append("  " + "-" * sum(w + 2 for w in widths))
    for dim in DIMENSIONS:
        m = mae.get(dim)
        mae_s = "N/A" if m is None else f"{m:.2f}"
        lines.append("  " + format_table_row(
            [dim, mae_s, format_pct(exact.get(dim)), format_pct(within1.get(dim))],
            widths,
        ))
    return "\n".join(lines)


def _format_theme_table(themes: dict) -> str:
    widths = [max(len(t) for t in THEMES), 10, 10, 10]
    lines = []
    lines.append("  " + format_table_row(["Theme", "Precision", "Recall", "F1"], widths))
    lines.append("  " + "-" * sum(w + 2 for w in widths))
    for t in THEMES:
        m = themes["per_theme"][t]
        lines.append("  " + format_table_row(
            [t, format_pct(m["precision"]), format_pct(m["recall"]), format_pct(m["f1"])],
            widths,
        ))
    return "\n".join(lines)


def print_report(
    meta: NewsletterRunMeta,
    metrics: dict,
    verbose: bool = False,
    story_results: list[StoryPrediction] | None = None,
    extraction_results: list[ExtractionPrediction] | None = None,
) -> None:
    """Print a formatted single-run report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Newsletter Evaluation Report: {meta.run_id[:8]}")
    print(f"{'=' * 60}")
    print(f"  Timestamp:    {meta.timestamp}")
    print(f"  Config:       {meta.config_path} ({meta.config_hash})")
    print(f"  Model:        {meta.newsletter_model}")
    print(f"  Mode:         {meta.mode}")
    print(f"  Prompt hash:  {meta.prompt_hash}")
    if meta.tag:
        print(f"  Tag:          {meta.tag}")
    if meta.seeded_from:
        print(f"  Seeded from:  {meta.seeded_from}")
    print(f"  Newsletters:  {meta.golden_set_count}")
    print(f"  Stories:      {meta.story_count}")

    tier = metrics["tier"]
    if tier["count"] or tier["errors"]:
        print("\n--- Tier Classification (4-class) ---")
        print(f"  Accuracy: {format_pct(tier['accuracy'])} "
              f"({tier['count']} stories, {tier['errors']} errors)")
        print("\n  Confusion matrix:")
        print(format_confusion_matrix(tier["confusion_matrix"], TIERS))
        print("\n  Per-class metrics:")
        print(format_per_class_table(tier["per_class"], TIERS))

    print("\n--- Quality Dimensions ---")
    print(_format_dim_table(
        metrics["dimension_mae"], metrics["dimension_exact"], metrics["dimension_within1"]
    ))

    themes = metrics["themes"]
    print("\n--- Themes (multi-label) ---")
    print(f"  Micro-F1: {format_pct(themes['micro_f1'])}   "
          f"Macro-F1: {format_pct(themes['macro_f1'])}   "
          f"Exact-set match: {format_pct(themes['exact_set_match'])}")
    print()
    print(_format_theme_table(themes))

    extraction = metrics["extraction"]
    if extraction["count"]:
        print("\n--- Extraction ---")
        print(f"  Micro:  P={format_pct(extraction['micro_precision'])} "
              f"R={format_pct(extraction['micro_recall'])} "
              f"F1={format_pct(extraction['micro_f1'])}")
        print(f"  Macro:  P={format_pct(extraction['macro_precision'])} "
              f"R={format_pct(extraction['macro_recall'])} "
              f"F1={format_pct(extraction['macro_f1'])}")

    if verbose:
        _print_verbose_single(story_results or [], extraction_results or [])

    print()


def _print_verbose_single(
    story_results: list[StoryPrediction],
    extraction_results: list[ExtractionPrediction],
) -> None:
    print("\n--- Story Disagreements ---")
    disagreements = [
        r for r in story_results
        if r.predicted_scores is not None
        and (r.expected_tier != r.predicted_tier
             or set(r.expected_themes or []) != set(r.predicted_themes or []))
    ]
    if not disagreements:
        print("  None!")
    for r in disagreements:
        parts = [f"  {r.story_id}:"]
        if r.expected_tier != r.predicted_tier:
            parts.append(f"tier={r.expected_tier}->{r.predicted_tier}")
        exp_th = set(r.expected_themes or [])
        pred_th = set(r.predicted_themes or [])
        if exp_th != pred_th:
            parts.append(f"themes={sorted(exp_th)}->{sorted(pred_th)}")
        print(" ".join(parts))

    if extraction_results:
        print("\n--- Extraction Diffs (per newsletter) ---")
        for r in extraction_results:
            if r.error is not None:
                print(f"  {r.thread_id}: ERROR {r.error}")
                continue
            matched, n_pred, n_gold = match_stories(r.predicted_stories, r.golden_stories)
            print(f"  {r.thread_id}: matched={matched} predicted={n_pred} golden={n_gold}")


def print_comparison(
    meta1: NewsletterRunMeta, metrics1: dict,
    meta2: NewsletterRunMeta, metrics2: dict,
    verbose: bool = False,
    story1: list[StoryPrediction] | None = None,
    story2: list[StoryPrediction] | None = None,
) -> None:
    """Print a side-by-side comparison of two newsletter runs."""
    print(f"\n{'=' * 60}")
    print("Newsletter Comparison Report")
    print(f"{'=' * 60}")
    print(f"  Run A: {meta1.run_id[:8]}  prompt_hash={meta1.prompt_hash or 'n/a'}  "
          f"tag={meta1.tag or '-'}")
    print(f"  Run B: {meta2.run_id[:8]}  prompt_hash={meta2.prompt_hash or 'n/a'}  "
          f"tag={meta2.tag or '-'}")

    widths = [22, 12, 12, 16]

    def header(title: str) -> None:
        print(f"\n--- {title} ---")
        print("  " + format_table_row(["Metric", "Run A", "Run B", "Delta"], widths))
        print("  " + "-" * sum(w + 2 for w in widths))

    def pct_row(label: str, a: float | None, b: float | None) -> None:
        print("  " + format_table_row(
            [label, format_pct(a), format_pct(b), format_metric_delta(a, b)], widths
        ))

    # Tier
    ta, tb = metrics1["tier"], metrics2["tier"]
    header("Tier")
    pct_row("Accuracy", ta["accuracy"], tb["accuracy"])
    for cls in TIERS:
        pct_row(f"  {cls} F1", ta["per_class"][cls]["f1"], tb["per_class"][cls]["f1"])

    # Dimensions (MAE, lower is better)
    header("Dimension MAE (lower=better)")
    for dim in DIMENSIONS:
        a = metrics1["dimension_mae"].get(dim)
        b = metrics2["dimension_mae"].get(dim)
        a_s = "N/A" if a is None else f"{a:.2f}"
        b_s = "N/A" if b is None else f"{b:.2f}"
        print("  " + format_table_row([dim, a_s, b_s, format_mae_delta(a, b)], widths))

    # Themes
    tha, thb = metrics1["themes"], metrics2["themes"]
    header("Themes")
    pct_row("Micro-F1", tha["micro_f1"], thb["micro_f1"])
    pct_row("Macro-F1", tha["macro_f1"], thb["macro_f1"])

    # Extraction
    ea, eb = metrics1["extraction"], metrics2["extraction"]
    header("Extraction")
    pct_row("Micro-F1", ea["micro_f1"], eb["micro_f1"])
    pct_row("Macro-F1", ea["macro_f1"], eb["macro_f1"])

    if verbose and story1 is not None and story2 is not None:
        _print_verbose_compare(story1, story2)

    print()


def _print_verbose_compare(
    story1: list[StoryPrediction],
    story2: list[StoryPrediction],
) -> None:
    print("\n--- Per-story Flips (A -> B) ---")
    r2_by_id = {r.story_id: r for r in story2}
    flips = []
    for r1 in story1:
        r2 = r2_by_id.get(r1.story_id)
        if r2 is None:
            continue
        if r1.predicted_scores is None or r2.predicted_scores is None:
            continue
        parts = []
        if r1.predicted_tier != r2.predicted_tier:
            a_right = r1.predicted_tier == r1.expected_tier
            b_right = r2.predicted_tier == r2.expected_tier
            flag = ""
            if a_right and not b_right:
                flag = " [regression]"
            elif b_right and not a_right:
                flag = " [improvement]"
            parts.append(f"tier: {r1.predicted_tier}->{r2.predicted_tier} "
                         f"(expected {r1.expected_tier}){flag}")
        s1, s2 = set(r1.predicted_themes or []), set(r2.predicted_themes or [])
        if s1 != s2:
            parts.append(f"themes: {sorted(s1)}->{sorted(s2)}")
        if parts:
            flips.append((r1.story_id, parts))
    if not flips:
        print("  None!")
    for sid, parts in flips:
        print(f"  {sid}: {', '.join(parts)}")


def report_as_json(meta: NewsletterRunMeta, metrics: dict) -> None:
    print(json.dumps({"meta": meta.to_dict(), "metrics": metrics}, indent=2))


def print_trend(results_dir: Path, match_threshold: float = 0.6) -> None:
    """Print a trend table across all result files in a directory."""
    files = sorted(results_dir.glob("*.jsonl"))
    files = [f for f in files if not f.name.endswith(".cot.jsonl")]
    if not files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        return

    print(f"\n{'=' * 60}")
    print(f"Newsletter Trend Report ({len(files)} runs)")
    print(f"{'=' * 60}")

    widths = [10, 10, 12, 10, 10, 12, 10]
    print("  " + format_table_row(
        ["Run ID", "Mode", "Tag", "TierAcc", "ThemeF1", "MAE(simple)", "ExtractF1"],
        widths,
    ))
    print("  " + "-" * sum(w + 2 for w in widths))

    for f in files:
        try:
            meta, story_preds, extraction_preds = load_results(f)
            metrics = compute_all_metrics(story_preds, extraction_preds, match_threshold)
            mae_simple = metrics["dimension_mae"].get("simple")
            mae_s = "N/A" if mae_simple is None else f"{mae_simple:.2f}"
            print("  " + format_table_row([
                meta.run_id[:8],
                meta.mode,
                meta.tag or meta.prompt_hash[:10],
                format_pct(metrics["tier"]["accuracy"]),
                format_pct(metrics["themes"]["micro_f1"]),
                mae_s,
                format_pct(metrics["extraction"]["micro_f1"]),
            ], widths))
        except Exception as exc:
            print(f"  Error reading {f.name}: {exc}", file=sys.stderr)

    print()


def cli():
    parser = argparse.ArgumentParser(description="Generate newsletter evaluation reports")
    parser.add_argument("--results", help="Path to results JSONL file")
    parser.add_argument("--compare", nargs=2, metavar="PATH", help="Compare two result files")
    parser.add_argument("--results-dir", help="Directory of results for trend view")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-story flips and per-newsletter extraction diffs")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format")
    parser.add_argument("--match-threshold", type=float, default=0.6,
                        help="SequenceMatcher ratio threshold for extraction matching (default 0.6)")
    args = parser.parse_args()

    if not any([args.results, args.compare, args.results_dir]):
        parser.error("Provide --results, --compare, or --results-dir")

    if args.results:
        meta, story_preds, extraction_preds = load_results(Path(args.results))
        metrics = compute_all_metrics(story_preds, extraction_preds, args.match_threshold)
        if args.format == "json":
            report_as_json(meta, metrics)
        else:
            print_report(meta, metrics, verbose=args.verbose,
                         story_results=story_preds, extraction_results=extraction_preds)

    elif args.compare:
        meta1, s1, e1 = load_results(Path(args.compare[0]))
        meta2, s2, e2 = load_results(Path(args.compare[1]))
        m1 = compute_all_metrics(s1, e1, args.match_threshold)
        m2 = compute_all_metrics(s2, e2, args.match_threshold)
        print_comparison(meta1, m1, meta2, m2, verbose=args.verbose, story1=s1, story2=s2)

    elif args.results_dir:
        print_trend(Path(args.results_dir), args.match_threshold)


if __name__ == "__main__":
    cli()
