"""
Tests for experiments/ — the sweep, the leaderboard and the sweep analysis.

Everything runs against a throwaway file store named like the real experiment,
with a two-configuration grid so CI stays fast.
"""

import json

import numpy as np
import pandas as pd
import pytest

import experiments.analyse_sweep as analyse_sweep
import experiments.run_experiments as run_experiments
from pipeline.config import MLFLOW_EXPERIMENT_NAME, REGISTERED_MODEL_NAME
from pipeline.training import setup_mlflow

TINY_GRID = [
    {"model_type": "hgb", "max_iter": 20, "learning_rate": 0.1, "max_depth": 3},
    {"model_type": "logreg", "C": 1.0, "max_iter": 500},
]


@pytest.fixture
def sweep_store(tmp_path, monkeypatch):
    """A file store whose experiment has the real name, so the CLI helpers find it."""
    uri = setup_mlflow(tracking_uri=f"file://{tmp_path / 'mlruns'}",
                       experiment_name=MLFLOW_EXPERIMENT_NAME)
    # The CLI calls setup_mlflow() with the configured defaults; keep it on the temp store.
    for module in (run_experiments, analyse_sweep):
        monkeypatch.setattr(module, "setup_mlflow", lambda *a, **k: uri)
    return uri


@pytest.fixture
def swept(sweep_store):
    """Run the tiny grid once and return its results."""
    return run_experiments.run_grid(TINY_GRID)


class TestSweep:
    """The grid runs, every configuration is its own comparable run."""

    def test_every_config_is_a_run_with_its_own_params(self, swept):
        assert [r["run_name"] for r in swept] == ["hgb-01", "logreg-02"]
        assert len({r["run_id"] for r in swept}) == 2
        # Different configurations must give different models, not just different names.
        assert swept[0]["metrics"]["roc_auc"] != swept[1]["metrics"]["roc_auc"]
        assert swept[0]["params"] == {k: v for k, v in TINY_GRID[0].items() if k != "model_type"}

    def test_leaderboard_lists_runs_best_first(self, swept, capsys):
        run_experiments.leaderboard(top=5)
        out = capsys.readouterr().out
        best = max(swept, key=lambda r: r["metrics"]["roc_auc"])
        rows = [line for line in out.splitlines() if line.startswith(("hgb-", "logreg-"))]
        assert len(rows) == 2
        assert rows[0].startswith(best["run_name"])

    def test_leaderboard_with_no_runs(self, sweep_store, capsys):
        run_experiments.leaderboard()
        assert "No runs found" in capsys.readouterr().out

    def test_promote_best_goes_through_the_gate(self, swept, capsys):
        result = run_experiments.promote_best()
        best = max(swept, key=lambda r: r["metrics"]["roc_auc"])
        assert result["run_id"] == best["run_id"]
        assert result["model_name"] == REGISTERED_MODEL_NAME
        assert result["outcome"] in {"champion", "challenger", "rejected"}
        assert (result["outcome"] == "rejected") == (not result["quality_gate"]["passed"])
        assert "registry:" in capsys.readouterr().out


class TestSweepAnalysis:
    """The bootstrap that says whether leaderboard differences are real."""

    def test_selection_gap(self):
        proba = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
        groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        # Group 1 selects 2/4, group 2 selects 1/4.
        assert analyse_sweep.selection_gap(proba, groups, threshold=0.5) == pytest.approx(0.25)
        assert analyse_sweep.selection_gap(proba, np.ones(8), threshold=0.5) == 0.0

    def test_paired_bootstrap_on_known_models(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 2000)
        groups = rng.integers(1, 3, 2000)
        scores = {
            "ref": np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1),
            "same": None,
            "noise": rng.random(2000),
        }
        scores["same"] = scores["ref"].copy()
        out = analyse_sweep.paired_bootstrap(y, scores, groups, reference="ref", n_boot=50)
        # A model compared with itself differs by exactly zero on every resample.
        assert out["same"]["d_auc_ci"] == [0.0, 0.0]
        assert out["ref"]["d_auc_ci"] == [0.0, 0.0]
        # A random scorer is reliably worse than the informative one.
        assert out["noise"]["d_auc_ci"][1] < 0
        lo, hi = out["ref"]["roc_auc_ci"]
        assert lo <= out["ref"]["roc_auc"] <= hi

    def test_baselines(self, raw):
        b = analyse_sweep.baselines(raw.drop(columns=["default_payment_next_month"]),
                                    raw["default_payment_next_month"])
        assert 0.5 < b["pay0_rule_roc_auc"] < 0.9
        assert set(b["default_rate_by_group"]) == {"1", "2"}
        rates = b["default_rate_by_group"].values()
        assert b["default_rate_gap"] == pytest.approx(max(rates) - min(rates), abs=2e-4)

    def test_analyse_end_to_end(self, swept, capsys):
        result = analyse_sweep.analyse(reference="hgb-01", n_boot=20,
                                       run_names=["hgb-01", "logreg-02"])
        assert set(result["runs"]) == {"hgb-01", "logreg-02"}
        assert result["runs"]["hgb-01"]["d_auc_ci"] == [0.0, 0.0]
        # Bootstrap re-scoring reproduces the metric the sweep logged.
        logged = {r["run_name"]: r["metrics"]["roc_auc"] for r in swept}
        for name, r in result["runs"].items():
            assert r["roc_auc"] == pytest.approx(logged[name], abs=1e-4)
        json.dumps(result)
        analyse_sweep.print_report(result)
        assert "d_auc vs ref" in capsys.readouterr().out

    def test_missing_reference_is_an_error(self, swept):
        with pytest.raises(ValueError, match="reference"):
            analyse_sweep.analyse(reference="rf-99", n_boot=5, run_names=["hgb-01"])

    def test_missing_experiment_is_an_error(self, tmp_path):
        setup_mlflow(tracking_uri=f"file://{tmp_path / 'empty'}", experiment_name="other")
        with pytest.raises(ValueError, match="does not exist"):
            analyse_sweep.latest_runs_by_name(["hgb-01"], experiment_name="missing")


def test_experiment_grid_covers_every_model_family():
    """Every real grid entry names a model type the trainer knows."""
    from pipeline.config import EXPERIMENT_GRID
    from pipeline.training import MODEL_CLASSES

    assert len(EXPERIMENT_GRID) == 7
    assert {c["model_type"] for c in EXPERIMENT_GRID} == set(MODEL_CLASSES)
    assert pd.Series([c["model_type"] for c in EXPERIMENT_GRID]).value_counts().to_dict() == \
        {"logreg": 2, "rf": 2, "hgb": 3}
