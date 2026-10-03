"""
Tests for the credit default pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline --cov-report=term-missing

These run against a temporary MLflow file store, so they need no server and
leave nothing behind. Shared fixtures (`raw`, `small`, `tracking`) live in
tests/conftest.py.

TestDataIngestion is the template the other classes follow. They are split one
module per pipeline stage, so each stage's tests sit with one owner:

    TestValidation       tests/test_validation.py
    TestPreprocessing    tests/test_preprocessing.py
    TestTraining         tests/test_training.py
    TestEvaluation       tests/test_evaluation.py
    TestQualityGate      tests/test_registry.py   (+ TestRegistry, end to end on a file store)
    sweep + analysis     tests/test_experiments.py
    DAG callables        tests/test_dag.py
    CLI end to end       tests/test_run_pipeline.py
"""

from pipeline.config import RAW_FEATURES, TARGET
from pipeline.data_ingestion import dataset_stats, split_data


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
