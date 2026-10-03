"""
Tests for pipeline/evaluation.py — metrics, per-group metrics, fairness gap.
"""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pipeline.config import TARGET
from pipeline.evaluation import (
    compute_group_metrics,
    compute_metrics,
    evaluate_model,
    fairness_gap,
)
from pipeline.preprocessing import add_derived_features
from pipeline.training import build_pipeline


class TestEvaluation:
    """Metrics, slices and the fairness gap."""

    @pytest.fixture(scope="class")
    def scored(self, small):
        """Fit a small model and return (y_true, y_proba, sensitive_attribute)."""
        X = add_derived_features(small.drop(columns=[TARGET]))
        y = small[TARGET]
        pipe = build_pipeline("hgb", list(X.columns), max_iter=40)
        pipe.fit(X, y)
        return y, pipe.predict_proba(X)[:, 1], small["SEX"]

    def test_metrics_are_in_range(self, scored):
        """roc_auc, pr_auc, precision, recall and f1 are all within [0, 1]."""
        y, proba, _ = scored
        m = compute_metrics(y, proba)
        for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier"):
            assert 0.0 <= m[key] <= 1.0, key
        assert m["roc_auc"] > 0.5, "a fitted model must beat a coin flip on its own data"

    def test_confusion_counts_sum_to_n(self, scored):
        """tp + fp + fn + tn equals the number of rows scored."""
        y, proba, _ = scored
        m = compute_metrics(y, proba)
        total = (m["true_positives"] + m["false_positives"]
                 + m["false_negatives"] + m["true_negatives"])
        assert total == len(y)
        # Internal consistency: recall is computed from the same counts.
        assert m["recall"] == pytest.approx(
            m["true_positives"] / (m["true_positives"] + m["false_negatives"])
        )

    def test_lower_threshold_never_lowers_recall(self, scored):
        """Recall at 0.2 is >= recall at 0.5.

        A property test, not a value test: it holds for any model, before and
        after retraining, which is exactly what makes it worth writing.
        """
        y, proba, _ = scored
        assert compute_metrics(y, proba, threshold=0.2)["recall"] >= \
            compute_metrics(y, proba, threshold=0.5)["recall"]

    def test_single_class_slice_does_not_crash(self):
        """A one-class slice still gives four confusion counts, and is not a group."""
        y = pd.Series([0] * 60)
        proba = np.linspace(0.0, 1.0, 60)
        m = compute_metrics(y, proba, threshold=0.5)
        assert m["true_positives"] + m["false_negatives"] == 0
        assert m["false_positives"] + m["true_negatives"] == 60
        assert np.isnan(m["roc_auc"]), "AUC is undefined with one class"
        # Large enough, but only one class: skipped rather than reported as evidence.
        assert compute_group_metrics(y, proba, pd.Series([1] * 60, name="SEX")) == {}

    def test_group_metrics_cover_every_group(self, scored):
        """Both SEX values appear, and the group sizes sum to the total."""
        y, proba, sex = scored
        gm = compute_group_metrics(y, proba, sex)
        assert set(gm) == {"1", "2"}
        assert sum(g["n"] for g in gm.values()) == len(y)
        for g in gm.values():
            assert 0.0 <= g["selection_rate"] <= 1.0

    def test_small_groups_are_skipped(self, scored):
        """A group under 50 rows is noise, so it is left out of the slices."""
        y, proba, sex = scored
        groups = sex.copy()
        groups.iloc[:10] = 3
        gm = compute_group_metrics(y, proba, groups)
        assert "3" not in gm
        assert set(gm) == {"1", "2"}

    def test_fairness_gap_is_non_negative(self, scored):
        """The gap is a magnitude, so it is never negative."""
        y, proba, sex = scored
        gm = compute_group_metrics(y, proba, sex)
        gap = fairness_gap(gm)
        assert gap >= 0.0
        rates = [g["selection_rate"] for g in gm.values()]
        assert gap == pytest.approx(max(rates) - min(rates))
        # Order of the groups must not change the answer.
        assert fairness_gap(dict(reversed(list(gm.items())))) == pytest.approx(gap)

    def test_fairness_gap_is_zero_for_one_group(self):
        """With a single group there is nothing to compare — return 0.0."""
        assert fairness_gap({"1": {"selection_rate": 0.4}}) == 0.0
        assert fairness_gap({}) == 0.0

    def test_evaluate_model_logs_metrics_and_slices(self, small, tracking):
        """evaluate_model logs aggregate and per-group metrics to the training run."""
        X, y = small.drop(columns=[TARGET]), small[TARGET]
        pipe = build_pipeline("logreg", list(add_derived_features(X).columns))
        pipe.fit(add_derived_features(X), y)
        with mlflow.start_run() as run:
            run_id = run.info.run_id

        result = evaluate_model(pipe, X, y, run_id=run_id)

        logged = MlflowClient().get_run(run_id).data
        assert logged.metrics["roc_auc"] == pytest.approx(result["roc_auc"])
        assert logged.metrics["fairness_gap"] == pytest.approx(result["fairness_gap"])
        assert "group_SEX_1_selection_rate" in logged.metrics
        assert logged.tags["stage"] == "evaluated"
        # The full result, slices included, is an artifact and must be valid JSON.
        evaluation = mlflow.artifacts.load_dict(f"runs:/{run_id}/evaluation.json")
        assert set(evaluation["group_metrics"]) == {"1", "2"}
        json.dumps(result)
