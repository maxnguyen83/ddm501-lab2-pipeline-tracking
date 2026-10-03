"""
Tests for the credit default pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline --cov-report=term-missing

These run against a temporary MLflow file store, so they need no server and
leave nothing behind.

TestDataIngestion is written for you — use it as the template for the rest.

TODO: Complete the five remaining test classes.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.config import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    RAW_FEATURES,
    TARGET,
)
from pipeline.data_ingestion import dataset_stats, load_raw, split_data
from pipeline.evaluation import compute_group_metrics, compute_metrics, fairness_gap
from pipeline.preprocessing import add_derived_features, build_preprocessor
from pipeline.registry import passes_quality_gate
from pipeline.training import build_model, build_pipeline
from pipeline.validation import DataValidationError, validate_dataset


# =============================================================================
@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """The real dataset, loaded once for the whole module."""
    return load_raw()


@pytest.fixture(scope="module")
def small(raw) -> pd.DataFrame:
    """A stratified 3,000-row sample — enough to fit, fast enough for CI."""
    return raw.groupby(TARGET, group_keys=False).apply(
        lambda g: g.sample(min(len(g), 1500), random_state=0)
    ).reset_index(drop=True)


# =============================================================================
class TestDataIngestion:
    """Loading and splitting."""

    def test_dataset_has_expected_shape(self, raw):
        assert len(raw) == 30_000
        assert TARGET in raw.columns
        assert all(c in raw.columns for c in RAW_FEATURES)

    def test_split_is_stratified(self, raw):
        X_train, X_test, y_train, y_test = split_data(raw)
        assert abs(y_train.mean() - y_test.mean()) < 0.01

    def test_split_is_reproducible(self, raw):
        a = split_data(raw)[0].index.tolist()
        b = split_data(raw)[0].index.tolist()
        assert a == b, "same seed must give the same split, or metrics are not comparable"

    def test_no_leakage_between_train_and_test(self, raw):
        X_train, X_test, _, _ = split_data(raw)
        assert not set(X_train.index) & set(X_test.index)

    def test_stats_report_positive_rate(self, raw):
        s = dataset_stats(raw)
        assert 0.05 < s["positive_rate"] < 0.60
        assert s["n_rows"] == len(raw)


# =============================================================================
# =============================================================================
# TODO 1: Tests for pipeline/validation.py
# =============================================================================
class TestValidation:
    """The quality gate in front of training."""

    def test_clean_data_passes(self, raw):
        """validate_dataset(raw) returns passed=True and n_errors == 0."""
        # TODO: implement
        pass

    def test_missing_column_is_caught(self, small):
        """Drop LIMIT_BAL and assert DataValidationError is raised.

        Hint: with pytest.raises(DataValidationError): ...
        """
        # TODO: implement
        pass

    def test_out_of_domain_category_is_caught(self, small):
        """Set SEX to 7 on a few rows and assert it raises."""
        # TODO: implement
        pass

    def test_impossible_age_is_caught(self, small):
        """Set AGE to 400 on a few rows and assert it raises."""
        # TODO: implement
        pass

    def test_negative_payment_is_caught(self, small):
        """Set PAY_AMT1 to -100 on a few rows and assert it raises."""
        # TODO: implement
        pass

    def test_degenerate_target_is_caught(self, small):
        """Set the whole target column to 0 and assert it raises.

        This is the check that matters most. Without it, a broken upstream
        extract produces a model with 100% accuracy that predicts "no default"
        for every applicant, and nothing in the pipeline objects.
        """
        # TODO: implement
        pass

    def test_report_is_returned_when_not_raising(self, small):
        """With raise_on_error=False, a bad frame returns passed=False."""
        # TODO: implement
        pass


# =============================================================================
# TODO 2: Tests for pipeline/preprocessing.py
# =============================================================================
class TestPreprocessing:
    """Feature engineering and the transformer."""

    def test_derived_features_are_added(self, small):
        """Every name in DERIVED_FEATURES appears in the output."""
        # TODO: implement
        pass

    def test_original_frame_is_not_mutated(self, small):
        """Calling add_derived_features must not widen the caller's frame."""
        # TODO: implement
        pass

    def test_utilisation_ratio_is_sensible(self, small):
        """Non-null values are within [0, 5] — no inf from a zero limit."""
        # TODO: implement
        pass

    def test_max_delay_matches_pay_columns(self, small):
        """max_delay equals the row-wise max of the six PAY_* columns."""
        # TODO: implement
        pass

    def test_preprocessor_one_hots_categoricals(self, small):
        """Fitted output has more columns than the input, and the same rows."""
        # TODO: implement
        pass

    def test_preprocessor_handles_unseen_category(self, small):
        """Fit, then transform a frame with EDUCATION=99. It must not raise.

        This is handle_unknown="ignore" doing its job. A category the model has
        never seen will turn up in production; the request should still get an
        answer rather than a 500.
        """
        # TODO: implement
        pass


# =============================================================================
# TODO 3: Tests for pipeline/training.py
# =============================================================================
class TestTraining:
    """Model construction and fitting."""

    @pytest.mark.parametrize("model_type", ["logreg", "rf", "hgb"])
    def test_every_model_type_builds(self, model_type):
        """build_model returns something for each supported type."""
        # TODO: implement
        pass

    def test_unknown_model_type_raises(self):
        """build_model("does-not-exist") raises ValueError."""
        # TODO: implement
        pass

    def test_params_override_defaults(self):
        """build_model("hgb", max_iter=7) gives an estimator with max_iter == 7."""
        # TODO: implement
        pass

    def test_pipeline_fits_and_predicts_probabilities(self, small):
        """Fit build_pipeline on the sample; probabilities are in [0, 1]."""
        # TODO: implement
        pass

    def test_preprocessing_travels_with_the_model(self, small):
        """The Pipeline has both a "preprocess" and a "classifier" step.

        If preprocessing happens outside the Pipeline, the serving code has to
        reproduce it by hand — and one day it will not. This test is how you
        stop that from being possible.
        """
        # TODO: implement
        pass


# =============================================================================
# TODO 4: Tests for pipeline/evaluation.py
# =============================================================================
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
        # TODO: implement
        pass

    def test_confusion_counts_sum_to_n(self, scored):
        """tp + fp + fn + tn equals the number of rows scored."""
        # TODO: implement
        pass

    def test_lower_threshold_never_lowers_recall(self, scored):
        """Recall at 0.2 is >= recall at 0.5.

        A property test, not a value test: it holds for any model, before and
        after retraining, which is exactly what makes it worth writing.
        """
        # TODO: implement
        pass

    def test_group_metrics_cover_every_group(self, scored):
        """Both SEX values appear, and the group sizes sum to the total."""
        # TODO: implement
        pass

    def test_fairness_gap_is_non_negative(self, scored):
        """The gap is a magnitude, so it is never negative."""
        # TODO: implement
        pass

    def test_fairness_gap_is_zero_for_one_group(self):
        """With a single group there is nothing to compare — return 0.0."""
        # TODO: implement
        pass


# =============================================================================
# TODO 5: Tests for the quality gate
# =============================================================================
class TestQualityGate:
    """The promotion rule, tested without touching a registry."""

    def test_good_model_passes(self):
        """roc_auc 0.75, pr_auc 0.55, gap 0.03 -> passed, no failed checks."""
        # TODO: implement
        pass

    def test_weak_model_is_rejected(self):
        """roc_auc 0.60, pr_auc 0.30 -> both appear in failed_checks."""
        # TODO: implement
        pass

    def test_accurate_but_unfair_model_is_rejected(self):
        """roc_auc 0.90, pr_auc 0.80, gap 0.40 -> rejected, on fairness alone.

        The test that makes the gate worth having. If this one passes when it
        should not, your gate is a report and not a gate.
        """
        # TODO: implement
        pass

    def test_missing_metric_fails_closed(self):
        """passes_quality_gate({}) must be False.

        An absent metric is not evidence of quality. If a bug stopped computing
        the fairness gap, a gate that read a missing value as a pass would keep
        reporting green while checking nothing.
        """
        # TODO: implement
        pass
