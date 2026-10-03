"""
Tests for pipeline/registry.py — the quality gate and alias-based promotion.

TestQualityGate needs no registry at all. TestRegistry runs against a throwaway
MLflow file store (the `tracking` fixture), so it needs no server either.
"""

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from pipeline.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MAX_FAIRNESS_GAP,
    MIN_PR_AUC,
    MIN_ROC_AUC,
)
from pipeline.registry import (
    beats_champion,
    compare_runs,
    find_best_run,
    get_model_version_by_alias,
    list_registered_models,
    passes_quality_gate,
    promote_model,
    register_model,
    set_alias,
)

MODEL = "test-credit-model"
GOOD = {"roc_auc": 0.75, "pr_auc": 0.55, "fairness_gap": 0.03}


def log_run(metrics: dict) -> str:
    """A finished run with a tiny model artifact and the given metrics."""
    with mlflow.start_run() as run:
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "dummy")
    return run.info.run_id


class TestQualityGate:
    """The promotion rule, tested without touching a registry."""

    def test_good_model_passes(self):
        """roc_auc 0.75, pr_auc 0.55, gap 0.03 -> passed, no failed checks."""
        gate = passes_quality_gate({"roc_auc": 0.75, "pr_auc": 0.55, "fairness_gap": 0.03})
        assert gate["passed"] is True
        assert gate["failed_checks"] == []
        assert set(gate["detail"]) == {"roc_auc", "pr_auc", "fairness_gap"}

    def test_weak_model_is_rejected(self):
        """roc_auc 0.60, pr_auc 0.30 -> both appear in failed_checks."""
        gate = passes_quality_gate({"roc_auc": 0.60, "pr_auc": 0.30, "fairness_gap": 0.03})
        assert gate["passed"] is False
        assert set(gate["failed_checks"]) == {"roc_auc", "pr_auc"}

    def test_accurate_but_unfair_model_is_rejected(self):
        """roc_auc 0.90, pr_auc 0.80, gap 0.40 -> rejected, on fairness alone.

        The test that makes the gate worth having. If this one passes when it
        should not, your gate is a report and not a gate.
        """
        gate = passes_quality_gate({"roc_auc": 0.90, "pr_auc": 0.80, "fairness_gap": 0.40})
        assert gate["passed"] is False
        assert gate["failed_checks"] == ["fairness_gap"]

    def test_missing_metric_fails_closed(self):
        """passes_quality_gate({}) must be False.

        An absent metric is not evidence of quality. If a bug stopped computing
        the fairness gap, a gate that read a missing value as a pass would keep
        reporting green while checking nothing.
        """
        assert passes_quality_gate({})["passed"] is False
        failed = set(passes_quality_gate({})["failed_checks"])
        assert failed == {"roc_auc", "pr_auc", "fairness_gap"}
        # Only the fairness metric missing: accurate is not enough.
        gate = passes_quality_gate({"roc_auc": 0.9, "pr_auc": 0.8})
        assert gate["failed_checks"] == ["fairness_gap"]

    def test_thresholds_are_inclusive(self):
        """A model exactly on every threshold passes; the rule is >= and <=."""
        gate = passes_quality_gate(
            {"roc_auc": MIN_ROC_AUC, "pr_auc": MIN_PR_AUC, "fairness_gap": MAX_FAIRNESS_GAP}
        )
        assert gate["passed"] is True


# =============================================================================
class TestRegistry:
    """Register, alias and promote against a throwaway MLflow file store."""

    def test_find_best_run_orders_by_metric(self, tracking):
        log_run({"roc_auc": 0.71, "brier": 0.20})
        best_id = log_run({"roc_auc": 0.78, "brier": 0.15})
        log_run({"roc_auc": 0.74, "brier": 0.10})

        best = find_best_run(experiment_name="test")
        assert best["run_id"] == best_id
        assert best["metrics"]["roc_auc"] == pytest.approx(0.78)
        assert best["params"]["model_type"] == "dummy"
        # ascending=True is for metrics where lower is better.
        assert find_best_run(experiment_name="test", metric="brier",
                             ascending=True)["metrics"]["brier"] == pytest.approx(0.10)

    def test_find_best_run_raises_when_there_is_nothing_to_rank(self, tracking):
        with pytest.raises(ValueError, match="does not exist"):
            find_best_run(experiment_name="no-such-experiment")
        with pytest.raises(ValueError, match="No runs"):
            find_best_run(experiment_name="test")
        log_run({"pr_auc": 0.5})
        with pytest.raises(ValueError, match="No runs"):
            find_best_run(experiment_name="test", metric="roc_auc")

    def test_register_and_alias(self, tracking):
        run_id = log_run(GOOD)
        version = register_model(run_id, MODEL)
        assert version == "1"
        assert get_model_version_by_alias(MODEL, CHAMPION_ALIAS) is None

        set_alias(MODEL, version, CHAMPION_ALIAS)
        current = get_model_version_by_alias(MODEL, CHAMPION_ALIAS)
        # The file store returns int versions, the REST store strings: compare as str.
        assert str(current["version"]) == "1"
        assert current["run_id"] == run_id
        assert current["model_uri"] == f"models:/{MODEL}@{CHAMPION_ALIAS}"
        # The alias is loadable, which is all a serving layer needs.
        assert mlflow.sklearn.load_model(current["model_uri"]).predict([[0]]).shape == (1,)

    def test_three_outcomes(self, tracking):
        """champion, then challenger (not better by the margin), then rejected."""
        first = promote_model(log_run(GOOD), GOOD, MODEL)
        assert first["outcome"] == "champion"

        almost = {**GOOD, "roc_auc": GOOD["roc_auc"] + 0.001}   # inside the 0.002 margin
        second = promote_model(log_run(almost), almost, MODEL)
        assert second["outcome"] == "challenger"

        unfair = {"roc_auc": 0.90, "pr_auc": 0.80, "fairness_gap": 0.40}
        third = promote_model(log_run(unfair), unfair, MODEL)
        assert third["outcome"] == "rejected"
        assert third["quality_gate"]["failed_checks"] == ["fairness_gap"]

        champion = get_model_version_by_alias(MODEL, CHAMPION_ALIAS)
        challenger = get_model_version_by_alias(MODEL, CHALLENGER_ALIAS)
        assert str(champion["version"]) == first["version"]
        assert str(challenger["version"]) == second["version"]

        # The rejected model is registered and tagged, but carries no alias.
        client = MlflowClient()
        rejected = client.get_model_version(MODEL, third["version"])
        assert rejected.tags["quality_gate"] == "failed"
        assert rejected.tags["failed_checks"] == "fairness_gap"
        assert list(rejected.aliases) == []
        assert client.get_model_version(MODEL, first["version"]).tags["quality_gate"] == "passed"

    def test_better_model_takes_over_the_champion_alias(self, tracking):
        first = promote_model(log_run(GOOD), GOOD, MODEL)
        better = {**GOOD, "roc_auc": GOOD["roc_auc"] + 0.01}
        assert beats_champion(better, MODEL)
        second = promote_model(log_run(better), better, MODEL)
        assert second["outcome"] == "champion"
        champion = get_model_version_by_alias(MODEL, CHAMPION_ALIAS)
        assert str(champion["version"]) == second["version"] != first["version"]
        # Versions are immutable: the old one still exists, it just lost the pointer.
        models = {m["name"]: m for m in list_registered_models()}
        assert len(models[MODEL]["versions"]) == 2
        assert {k: str(v) for k, v in models[MODEL]["aliases"].items()} == \
            {CHAMPION_ALIAS: second["version"]}

    def test_compare_runs_returns_top_n(self, tracking):
        for auc in (0.70, 0.74, 0.72):
            log_run({"roc_auc": auc})
        rows = compare_runs(experiment_name="test", top_n=2)
        assert [round(r["metrics"]["roc_auc"], 2) for r in rows] == [0.74, 0.72]
        assert compare_runs(experiment_name="no-such-experiment") == []
