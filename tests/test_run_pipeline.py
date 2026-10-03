"""
End-to-end test of the CLI entry point: ingest -> validate -> train -> evaluate -> promote.

The same function the smoke workflow runs, pointed at a throwaway file store.
"""

import json
import sys

import pytest

import pipeline.run_pipeline as run_pipeline
from pipeline.config import MLFLOW_EXPERIMENT_NAME, REGISTERED_MODEL_NAME
from pipeline.registry import get_model_version_by_alias
from pipeline.training import setup_mlflow


@pytest.fixture
def cli_store(tmp_path, monkeypatch):
    """Temp tracking store under the real experiment name, and a temp artifacts dir."""
    uri = setup_mlflow(tracking_uri=f"file://{tmp_path / 'mlruns'}",
                       experiment_name=MLFLOW_EXPERIMENT_NAME)
    monkeypatch.setattr(run_pipeline, "setup_mlflow", lambda *a, **k: uri)
    monkeypatch.setattr(run_pipeline, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


class TestRunPipeline:

    def test_no_register_trains_and_evaluates_only(self, cli_store):
        summary = run_pipeline.run(model_type="hgb", run_name="t", register=False, max_iter=30)
        assert summary["promotion"] is None
        assert 0.5 < summary["metrics"]["roc_auc"] <= 1.0
        assert "fairness_gap" in summary["metrics"]
        assert set(summary["group_metrics"]) == {"1", "2"}
        written = json.loads((cli_store / "last_run.json").read_text())
        assert written["run_id"] == summary["run_id"]
        assert get_model_version_by_alias(REGISTERED_MODEL_NAME) is None

    def test_cli_registers_and_promotes(self, cli_store, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run_pipeline", "--model-type", "hgb",
                                          "--max-iter", "60", "--run-name", "cli"])
        run_pipeline.main()
        out = capsys.readouterr().out
        summary = json.loads((cli_store / "last_run.json").read_text())
        promotion = summary["promotion"]
        assert f"registry {REGISTERED_MODEL_NAME} v1 -> {promotion['outcome']}" in out
        # First passing model has no champion to beat, so it becomes the champion.
        if promotion["quality_gate"]["passed"]:
            assert promotion["outcome"] == "champion"
            assert get_model_version_by_alias(REGISTERED_MODEL_NAME) is not None
        else:
            assert promotion["outcome"] == "rejected"
            assert "gate failed" in out
