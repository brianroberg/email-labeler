"""Tests for evals.newsletter_report — newsletter eval metrics on hand-built fixtures."""

from evals.newsletter_report import (
    THEMES,
    TIERS,
    compute_all_metrics,
    compute_dimension_exact_match,
    compute_dimension_mae,
    compute_dimension_within1,
    compute_extraction_metrics,
    compute_multilabel_metrics,
    compute_tier_metrics,
    format_mae_delta,
    format_metric_delta,
    match_stories,
)
from evals.newsletter_schemas import ExtractionPrediction, StoryPrediction


def _pred(
    story_id="s",
    expected_scores=None,
    expected_tier=None,
    expected_themes=None,
    predicted_scores=None,
    predicted_tier=None,
    predicted_themes=None,
    error=None,
) -> StoryPrediction:
    return StoryPrediction(
        story_id=story_id,
        thread_id="t",
        expected_scores=expected_scores,
        expected_tier=expected_tier,
        expected_themes=expected_themes or [],
        predicted_scores=predicted_scores,
        predicted_tier=predicted_tier,
        predicted_themes=predicted_themes or [],
        error=error,
    )


class TestMatchStories:
    def test_exact_match(self):
        predicted = [{"title": "A", "text": "the quick brown fox"}]
        golden = [{"story_id": "t:0", "title": "A", "text": "the quick brown fox"}]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1
        assert n_pred == 1
        assert n_gold == 1

    def test_fuzzy_match_above_threshold(self):
        # Same story with a couple words changed — ratio stays well above 0.6.
        predicted = [{"title": "", "text": "the quick brown fox jumped over the lazy dog"}]
        golden = [{"story_id": "t:0", "title": "",
                   "text": "the quick brown fox jumped over the lazy cat"}]
        matched, _, _ = match_stories(predicted, golden)
        assert matched == 1

    def test_below_threshold_no_match(self):
        predicted = [{"title": "", "text": "completely unrelated content here"}]
        golden = [{"story_id": "t:0", "title": "",
                   "text": "an entirely different subject about gardening"}]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 0
        assert n_pred == 1
        assert n_gold == 1

    def test_one_to_one_greedy(self):
        # One predicted story is similar to two golden stories; it may match only one.
        predicted = [{"title": "", "text": "alpha beta gamma delta"}]
        golden = [
            {"story_id": "t:0", "title": "", "text": "alpha beta gamma delta"},
            {"story_id": "t:1", "title": "", "text": "alpha beta gamma delta epsilon"},
        ]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1  # cannot match both golden stories
        assert n_pred == 1
        assert n_gold == 2

    def test_extra_predicted_drops_precision(self):
        # 2 predicted, 1 golden -> 1 match -> precision 1/2, recall 1/1.
        predicted = [
            {"title": "", "text": "the quick brown fox"},
            {"title": "", "text": "spurious hallucinated story"},
        ]
        golden = [{"story_id": "t:0", "title": "", "text": "the quick brown fox"}]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1
        assert n_pred == 2
        assert n_gold == 1

    def test_missing_predicted_drops_recall(self):
        # 1 predicted, 2 golden -> 1 match -> precision 1/1, recall 1/2.
        predicted = [{"title": "", "text": "the quick brown fox"}]
        golden = [
            {"story_id": "t:0", "title": "", "text": "the quick brown fox"},
            {"story_id": "t:1", "title": "", "text": "a wholly separate tale of woe"},
        ]
        matched, n_pred, n_gold = match_stories(predicted, golden)
        assert matched == 1
        assert n_pred == 1
        assert n_gold == 2


class TestTierMetrics:
    def test_confusion_and_prf(self):
        results = [
            _pred(predicted_scores={"simple": 5}, expected_tier="excellent",
                  predicted_tier="excellent"),
            _pred(predicted_scores={"simple": 4}, expected_tier="excellent",
                  predicted_tier="good"),  # off-diagonal
            _pred(predicted_scores={"simple": 3}, expected_tier="good",
                  predicted_tier="good"),
            _pred(predicted_scores={"simple": 1}, expected_tier="poor",
                  predicted_tier="poor"),
        ]
        m = compute_tier_metrics(results)
        assert m["errors"] == 0
        assert m["count"] == 4
        assert abs(m["accuracy"] - 0.75) < 1e-9
        cm = m["confusion_matrix"]
        assert cm["excellent"]["excellent"] == 1
        assert cm["excellent"]["good"] == 1
        assert cm["good"]["good"] == 1
        assert cm["poor"]["poor"] == 1
        # excellent recall = 1 TP / (1 TP + 1 FN) = 0.5
        assert abs(m["per_class"]["excellent"]["recall"] - 0.5) < 1e-9
        # good precision = 1 TP / (1 TP + 1 FP) = 0.5
        assert abs(m["per_class"]["good"]["precision"] - 0.5) < 1e-9

    def test_parse_failure_counted_as_error_excluded_from_matrix(self):
        results = [
            _pred(predicted_scores={"simple": 5}, expected_tier="excellent",
                  predicted_tier="excellent"),
            # parse failure: predicted_scores is None -> error, excluded from matrix.
            # A stray predicted_tier is present but must be ignored because scores
            # failed to parse (guards exclusion-by-None-scores, not by None-tier).
            _pred(predicted_scores=None, expected_tier="good", predicted_tier="good"),
        ]
        m = compute_tier_metrics(results)
        assert m["errors"] == 1
        assert m["count"] == 1
        assert m["accuracy"] == 1.0
        # The good-row must NOT appear in the matrix despite predicted_tier="good"
        assert m["confusion_matrix"]["good"]["good"] == 0
        assert sum(
            m["confusion_matrix"][e][p] for e in TIERS for p in TIERS
        ) == 1




class TestDimensionMetrics:
    def test_mae_and_exact_and_within1(self):
        results = [
            _pred(
                expected_scores={"simple": 5, "concrete": 3, "personal": 4, "dynamic": 2},
                predicted_scores={"simple": 5, "concrete": 1, "personal": 5, "dynamic": 2},
            ),
            _pred(
                expected_scores={"simple": 2, "concrete": 2, "personal": 2, "dynamic": 2},
                predicted_scores={"simple": 4, "concrete": 2, "personal": 2, "dynamic": 2},
            ),
        ]
        mae = compute_dimension_mae(results)
        # simple: |5-5|=0, |2-4|=2 -> mean 1.0
        assert abs(mae["simple"] - 1.0) < 1e-9
        # concrete: |3-1|=2, |2-2|=0 -> mean 1.0
        assert abs(mae["concrete"] - 1.0) < 1e-9
        # personal: |4-5|=1, |2-2|=0 -> mean 0.5
        assert abs(mae["personal"] - 0.5) < 1e-9
        # dynamic: 0, 0 -> 0.0
        assert abs(mae["dynamic"] - 0.0) < 1e-9

        exact = compute_dimension_exact_match(results)
        # simple: exact on row1 only -> 1/2
        assert abs(exact["simple"] - 0.5) < 1e-9
        # dynamic: both exact -> 1.0
        assert abs(exact["dynamic"] - 1.0) < 1e-9
        # concrete: row1 wrong, row2 exact -> 1/2
        assert abs(exact["concrete"] - 0.5) < 1e-9

        within1 = compute_dimension_within1(results)
        # personal: diff 1 and 0 both within 1 -> 1.0
        assert abs(within1["personal"] - 1.0) < 1e-9
        # simple: diff 0 (within) and 2 (not) -> 1/2
        assert abs(within1["simple"] - 0.5) < 1e-9

    def test_only_over_stories_with_both_scores(self):
        results = [
            _pred(
                expected_scores={"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
                predicted_scores={"simple": 4, "concrete": 4, "personal": 4, "dynamic": 4},
            ),
            # parse failure -> excluded entirely
            _pred(expected_scores={"simple": 1}, predicted_scores=None),
            # no expected -> excluded
            _pred(expected_scores=None, predicted_scores={"simple": 1}),
        ]
        mae = compute_dimension_mae(results)
        assert mae["simple"] == 0.0
        exact = compute_dimension_exact_match(results)
        assert exact["simple"] == 1.0




class TestMultilabelThemeMetrics:
    def test_per_theme_micro_macro_and_exact_set(self):
        # Story 1: expected {scripture, church}, predicted {scripture}
        #   scripture: TP; church: FN
        # Story 2: expected {christlikeness}, predicted {christlikeness, church}
        #   christlikeness: TP; church: FP
        results = [
            _pred(expected_themes=["scripture", "church"], predicted_themes=["scripture"],
                  predicted_scores={"simple": 3}),
            _pred(expected_themes=["christlikeness"],
                  predicted_themes=["christlikeness", "church"],
                  predicted_scores={"simple": 3}),
        ]
        m = compute_multilabel_metrics(results, THEMES)

        # scripture: TP=1, FP=0, FN=0 -> P=R=F1=1
        assert m["per_theme"]["scripture"]["f1"] == 1.0
        # church: TP=0, FP=1, FN=1 -> P=0, R=0, F1=0
        assert m["per_theme"]["church"]["precision"] == 0.0
        assert m["per_theme"]["church"]["recall"] == 0.0
        # christlikeness: TP=1 -> F1=1
        assert m["per_theme"]["christlikeness"]["f1"] == 1.0

        # Micro: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
        assert abs(m["micro_f1"] - (2 / 3)) < 1e-9

        # Macro F1: mean over 5 themes.
        # scripture=1, christlikeness=1, church=0, vocation_family=0(no support->0),
        # disciple_making=0 -> (1+1+0+0+0)/5 = 0.4
        assert abs(m["macro_f1"] - 0.4) < 1e-9

        # Exact set match: neither story matches exactly -> 0/2
        assert m["exact_set_match"] == 0.0

    def test_exact_set_match_positive(self):
        results = [
            _pred(expected_themes=["scripture"], predicted_themes=["scripture"],
                  predicted_scores={"simple": 3}),
            _pred(expected_themes=["church", "scripture"],
                  predicted_themes=["scripture", "church"],  # order-insensitive
                  predicted_scores={"simple": 3}),
        ]
        m = compute_multilabel_metrics(results, THEMES)
        assert m["exact_set_match"] == 1.0




class TestExtractionMetrics:
    def test_micro_and_macro(self):
        # Newsletter A: 2 predicted, 2 golden, both match -> P=1, R=1
        # Newsletter B: 2 predicted, 1 golden, 1 match -> P=1/2, R=1
        nl_a = ExtractionPrediction(
            thread_id="A",
            golden_stories=[
                {"story_id": "A:0", "title": "", "text": "alpha story about faith"},
                {"story_id": "A:1", "title": "", "text": "beta story about hope"},
            ],
            predicted_stories=[
                {"title": "", "text": "alpha story about faith"},
                {"title": "", "text": "beta story about hope"},
            ],
        )
        nl_b = ExtractionPrediction(
            thread_id="B",
            golden_stories=[
                {"story_id": "B:0", "title": "", "text": "gamma story of grace"},
            ],
            predicted_stories=[
                {"title": "", "text": "gamma story of grace"},
                {"title": "", "text": "spurious hallucination unrelated"},
            ],
        )
        m = compute_extraction_metrics([nl_a, nl_b])

        # Micro: matched=3, predicted=4, golden=3 -> P=3/4, R=3/3=1
        assert abs(m["micro_precision"] - 0.75) < 1e-9
        assert abs(m["micro_recall"] - 1.0) < 1e-9
        micro_f1 = 2 * 0.75 * 1.0 / (0.75 + 1.0)
        assert abs(m["micro_f1"] - micro_f1) < 1e-9

        # Macro precision: mean(1.0, 0.5) = 0.75; macro recall mean(1,1)=1
        assert abs(m["macro_precision"] - 0.75) < 1e-9
        assert abs(m["macro_recall"] - 1.0) < 1e-9




class TestComparisonDeltaSign:
    def test_f1_improvement_positive(self):
        # Run B has higher tier accuracy than Run A -> positive delta.
        assert format_metric_delta(0.5, 0.75).startswith("+")
        assert format_metric_delta(0.75, 0.5).startswith("-")

    def test_mae_decrease_is_improvement(self):
        # MAE dropped from 1.0 to 0.5 -> improvement -> flagged as improvement (+).
        improved = format_mae_delta(1.0, 0.5)
        assert "improve" in improved.lower() or improved.strip().startswith("-")
        # The raw delta must reflect the decrease (b - a = -0.5).
        assert "-0.5" in improved
        # MAE increase is a regression.
        worsened = format_mae_delta(0.5, 1.0)
        assert "+0.5" in worsened
        assert "improve" not in worsened.lower()

    def test_compute_all_metrics_bundles_sections(self):
        results = [
            _pred(
                expected_scores={"simple": 5, "concrete": 5, "personal": 5, "dynamic": 5},
                predicted_scores={"simple": 5, "concrete": 5, "personal": 5, "dynamic": 5},
                expected_tier="excellent", predicted_tier="excellent",
                expected_themes=["scripture"], predicted_themes=["scripture"],
            ),
        ]
        extraction = [
            ExtractionPrediction(
                thread_id="A",
                golden_stories=[{"story_id": "A:0", "title": "", "text": "alpha faith"}],
                predicted_stories=[{"title": "", "text": "alpha faith"}],
            ),
        ]
        m = compute_all_metrics(results, extraction)
        assert m["tier"]["accuracy"] == 1.0
        assert m["themes"]["micro_f1"] == 1.0
        assert m["dimension_mae"]["simple"] == 0.0
        assert m["extraction"]["micro_f1"] == 1.0
