"""
Tests for dags/credit_training_dag.py — the graph and the task callables.

Airflow is deliberately not in requirements.txt, so when it is missing these
tests install a minimal stand-in for the three classes the DAG file imports
(DAG, PythonOperator/BranchPythonOperator, EmptyOperator). The stand-in uses the
same attribute names as Airflow (task_dict, downstream_task_ids, trigger_rule),
so with real Airflow installed the same assertions run against the real DAG.

The callables are then executed in order with a fake TaskInstance whose XCom
store insists on what the real one needs: JSON-serialisable, small values.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import pipeline.training
from pipeline.config import REGISTERED_MODEL_NAME

DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "credit_training_dag.py"
MAX_XCOM_BYTES = 48 * 1024   # Airflow stores XCom in the metadata DB; keep it small


# =============================================================================
# A stand-in for the bits of Airflow the DAG file uses
# =============================================================================
class _StubDAG:
    current = None

    def __init__(self, dag_id, **kwargs):
        self.dag_id = dag_id
        self.task_dict = {}
        self.schedule_interval = kwargs.get("schedule")

    def __enter__(self):
        _StubDAG.current = self
        return self

    def __exit__(self, *exc):
        _StubDAG.current = None


class _StubOperator:
    def __init__(self, task_id, python_callable=None, trigger_rule="all_success", **kwargs):
        self.task_id = task_id
        self.python_callable = python_callable
        self.trigger_rule = trigger_rule
        self.upstream_task_ids, self.downstream_task_ids = set(), set()
        _StubDAG.current.task_dict[task_id] = self

    def __rshift__(self, other):
        for op in other if isinstance(other, list) else [other]:
            self.downstream_task_ids.add(op.task_id)
            op.upstream_task_ids.add(self.task_id)
        return other

    def __rrshift__(self, other):          # [a, b] >> self
        for op in other:
            op >> self
        return self


def _airflow_stub_modules():
    airflow = types.ModuleType("airflow")
    airflow.DAG = _StubDAG
    operators = types.ModuleType("airflow.operators")
    empty = types.ModuleType("airflow.operators.empty")
    empty.EmptyOperator = _StubOperator
    python = types.ModuleType("airflow.operators.python")
    python.PythonOperator = python.BranchPythonOperator = _StubOperator
    return {"airflow": airflow, "airflow.operators": operators,
            "airflow.operators.empty": empty, "airflow.operators.python": python}


@pytest.fixture
def dag_module(tmp_path, monkeypatch, tracking):
    """Import the DAG file with its run directory and MLflow pointed at temp paths."""
    if importlib.util.find_spec("airflow") is None:
        for name, module in _airflow_stub_modules().items():
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("PIPELINE_RUN_DIR", str(tmp_path / "runs"))
    # Tasks call setup_mlflow() with the configured defaults; keep them on the temp store.
    monkeypatch.setattr(pipeline.training, "setup_mlflow", lambda *a, **k: tracking)

    spec = importlib.util.spec_from_file_location("credit_training_dag", DAG_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTI:
    """XCom keyed by (task_id, key), with the checks the metadata DB would impose."""

    def __init__(self, store, task_id):
        self.store, self.task_id = store, task_id

    def xcom_push(self, key, value):
        size = len(json.dumps(value).encode())       # raises if not JSON-serialisable
        assert size < MAX_XCOM_BYTES, f"XCom {key} is {size} bytes — put data on the volume"
        self.store[(self.task_id, key)] = value

    def xcom_pull(self, task_ids=None, key="return_value"):
        for (task, k), value in self.store.items():
            if k == key and (task_ids is None or task == task_ids):
                return value
        return None


def run_task(dag_module, store, task_id, conf=None):
    """Call a task's python_callable the way Airflow would."""
    task = dag_module.dag.task_dict[task_id]
    context = {
        "ti": FakeTI(store, task_id),
        "run_id": "manual__2026-10-03T00:00:00+00:00",
        "ds": "2026-10-03",
        "dag_run": types.SimpleNamespace(conf=conf or {}),
    }
    return task.python_callable(**context)


# =============================================================================
class TestDagStructure:
    """The graph the brief asks for, and the trigger rule that makes cleanup run."""

    def test_all_tasks_exist(self, dag_module):
        assert set(dag_module.dag.task_dict) == {
            "ingest", "validate", "train", "evaluate", "decide",
            "promote_model", "skip_promotion", "cleanup",
        }

    def test_linear_chain_then_branch(self, dag_module):
        t = dag_module.dag.task_dict
        chain = ["ingest", "validate", "train", "evaluate", "decide"]
        for up, down in zip(chain, chain[1:], strict=False):
            assert t[up].downstream_task_ids == {down}
        assert t["decide"].downstream_task_ids == {"promote_model", "skip_promotion"}
        assert t["promote_model"].downstream_task_ids == {"cleanup"}
        assert t["skip_promotion"].downstream_task_ids == {"cleanup"}

    def test_cleanup_runs_down_either_branch(self, dag_module):
        assert dag_module.dag.task_dict["cleanup"].trigger_rule == "none_failed_min_one_success"


class TestDecide:
    """A BranchPythonOperator must return a real downstream task_id."""

    @pytest.mark.parametrize("metrics,expected", [
        ({"roc_auc": 0.75, "pr_auc": 0.55, "fairness_gap": 0.03}, "promote_model"),
        ({"roc_auc": 0.90, "pr_auc": 0.80, "fairness_gap": 0.40}, "skip_promotion"),
        ({}, "skip_promotion"),
    ])
    def test_branch_target(self, dag_module, metrics, expected):
        import mlflow
        from mlflow.tracking import MlflowClient

        with mlflow.start_run() as run:
            pass
        store = {("evaluate", "metrics"): metrics, ("train", "mlflow_run_id"): run.info.run_id}
        branch = run_task(dag_module, store, "decide")
        assert branch == expected
        assert branch in dag_module.dag.task_dict["decide"].downstream_task_ids
        assert store[("decide", "quality_gate")]["passed"] is (expected == "promote_model")
        # The verdict is on the MLflow run even when nothing is registered.
        tag = MlflowClient().get_run(run.info.run_id).data.tags["quality_gate"]
        assert tag == ("passed" if expected == "promote_model" else "failed")


class TestDagRun:
    """Every task end to end, in order, on the real data."""

    @pytest.mark.parametrize("model_type,expected_branch", [
        # On the committed data: hgb clears the gate, logreg's selection-rate gap
        # (~0.054) is over MAX_FAIRNESS_GAP — the same split the sweep shows.
        ("hgb", "promote_model"),
        ("logreg", "skip_promotion"),
    ])
    def test_full_run(self, dag_module, model_type, expected_branch):
        store = {}
        run_task(dag_module, store, "ingest")
        run_dir = Path(store[("ingest", "run_dir")])
        assert (run_dir / "split.joblib").exists() and (run_dir / "raw.parquet").exists()

        run_task(dag_module, store, "validate")
        assert store[("validate", "validation_report")]["passed"] is True
        assert (run_dir / "validation_report.json").exists()

        run_task(dag_module, store, "train", conf={"model_type": model_type})
        assert (run_dir / "model.joblib").exists()
        assert isinstance(store[("train", "mlflow_run_id")], str)

        run_task(dag_module, store, "evaluate")
        metrics = store[("evaluate", "metrics")]
        # Scalars only in XCom; the per-group detail lives on the volume.
        assert all(isinstance(v, (int, float)) for v in metrics.values())
        assert "group_metrics" not in metrics
        evaluation = json.loads((run_dir / "evaluation.json").read_text())
        assert set(evaluation["group_metrics"]) == {"1", "2"}

        branch = run_task(dag_module, store, "decide")
        assert branch == expected_branch
        if branch == "promote_model":
            summary = run_task(dag_module, store, "promote_model")
            assert summary == f"{REGISTERED_MODEL_NAME} v1 -> champion"
            assert store[("promote_model", "promotion")]["outcome"] == "champion"

        assert run_task(dag_module, store, "cleanup") == f"removed {run_dir}"
        assert not run_dir.exists()
        assert run_task(dag_module, store, "cleanup") == "nothing to clean"

    def test_bad_data_stops_the_run_at_validate(self, dag_module):
        from pipeline.data_ingestion import load_raw
        from pipeline.validation import DataValidationError

        store = {}
        run_task(dag_module, store, "ingest")
        run_dir = Path(store[("ingest", "run_dir")])
        bad = load_raw()
        bad["default_payment_next_month"] = 0
        bad.to_parquet(run_dir / "raw.parquet", index=False)
        with pytest.raises(DataValidationError):
            run_task(dag_module, store, "validate")
        assert ("validate", "validation_report") not in store
