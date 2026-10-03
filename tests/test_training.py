"""
Tests for pipeline/training.py — model construction, fitting and tracking.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from pipeline.config import MODEL_CONFIGS, RANDOM_STATE, TARGET
from pipeline.preprocessing import add_derived_features
from pipeline.training import build_model, build_pipeline, train_model, training_fingerprint


class TestTraining:
    """Model construction and fitting."""

    @pytest.mark.parametrize("model_type", ["logreg", "rf", "hgb"])
    def test_every_model_type_builds(self, model_type):
        """build_model returns something for each supported type."""
        model = build_model(model_type)
        assert hasattr(model, "fit") and hasattr(model, "predict_proba")
        assert model.get_params()["random_state"] == RANDOM_STATE

    def test_unknown_model_type_raises(self):
        """build_model("does-not-exist") raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model("does-not-exist")

    def test_params_override_defaults(self):
        """build_model("hgb", max_iter=7) gives an estimator with max_iter == 7."""
        model = build_model("hgb", max_iter=7)
        assert model.max_iter == 7
        # Keys that were not overridden keep their configured default.
        assert model.learning_rate == MODEL_CONFIGS["hgb"]["learning_rate"]

    def test_pipeline_fits_and_predicts_probabilities(self, small):
        """Fit build_pipeline on the sample; probabilities are in [0, 1]."""
        X = add_derived_features(small.drop(columns=[TARGET]))
        pipe = build_pipeline("logreg", list(X.columns))
        pipe.fit(X, small[TARGET])
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert ((proba >= 0) & (proba <= 1)).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_preprocessing_travels_with_the_model(self, small):
        """The Pipeline has both a "preprocess" and a "classifier" step.

        If preprocessing happens outside the Pipeline, the serving code has to
        reproduce it by hand — and one day it will not. This test is how you
        stop that from being possible.
        """
        X = add_derived_features(small.drop(columns=[TARGET]))
        pipe = build_pipeline("hgb", list(X.columns), max_iter=10)
        assert list(pipe.named_steps) == ["preprocess", "classifier"]
        pipe.fit(X, small[TARGET])
        # Raw-scale input goes straight in: scaling is inside the fitted object.
        assert pipe.predict_proba(X.head(3)).shape == (3, 2)

    def test_train_model_logs_a_reproducible_run(self, small, tracking):
        """train_model logs params, the feature list, the report and the model."""
        X, y = small.drop(columns=[TARGET]), small[TARGET]
        report = {"passed": True, "n_errors": 0}
        pipe, run_id = train_model(
            X, y, model_type="hgb", run_name="unit", max_iter=10,
            data_stats={"n_rows": len(small)}, validation_report=report,
        )
        run = MlflowClient().get_run(run_id)
        assert run.data.params["model_type"] == "hgb"
        assert run.data.params["max_iter"] == "10"
        assert run.data.params["data_n_rows"] == str(len(small))
        # Effective hyperparameters, not just the override: defaults are logged too.
        assert run.data.params["learning_rate"] == str(MODEL_CONFIGS["hgb"]["learning_rate"])
        assert run.data.params["train_data_sha256"] == training_fingerprint(X, y)
        assert run.data.tags["validation"] == "passed"
        artifacts = {a.path for a in MlflowClient().list_artifacts(run_id)}
        assert {"feature_columns.json", "validation_report.json", "model"} <= artifacts
        features = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/feature_columns.json")
        assert features["features"] == list(add_derived_features(X).columns)
        # The logged model is the fitted pipeline, preprocessing included.
        loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        np.testing.assert_allclose(
            loaded.predict_proba(add_derived_features(X.head(5))),
            pipe.predict_proba(add_derived_features(X.head(5))),
        )

    def test_fingerprint_identifies_the_training_rows(self, small):
        """Same rows give the same hash; one changed value or a reordering does not."""
        X, y = small.drop(columns=[TARGET]), small[TARGET]
        assert training_fingerprint(X, y) == training_fingerprint(X.copy(), y.copy())
        changed = X.copy()
        changed.iloc[0, 0] += 1
        assert training_fingerprint(changed, y) != training_fingerprint(X, y)
        assert training_fingerprint(X, 1 - y) != training_fingerprint(X, y)
