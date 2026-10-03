"""
Read the sweep, not just the leaderboard.

The leaderboard ranks runs by a point estimate on one 6,000-row test set. This
script asks the question the ranking cannot answer: are the differences between
runs larger than the noise in that test set?

For each sweep run it reloads the logged model from MLflow, scores the same
seeded test split, and computes paired bootstrap intervals (every model is
scored on the SAME resampled rows each round, so split noise cancels out of the
differences):

  roc_auc             with a 95% interval
  fairness_gap        selection-rate gap between SEX groups, with a 95% interval
  d_auc, d_gap        difference from a reference run, with a 95% interval

It also prints two baselines the quality gate is calibrated against: the
one-feature PAY_0 rule an analyst would use without a model, and the gap in the
ACTUAL default rate between groups.

Usage:
    python -m experiments.analyse_sweep
    python -m experiments.analyse_sweep --reference hgb-06 --n-boot 1000
"""

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline.config import (
    ARTIFACTS_DIR,
    EXPERIMENT_GRID,
    MLFLOW_EXPERIMENT_NAME,
    REVIEW_THRESHOLD,
    SENSITIVE_ATTRIBUTE,
)
from pipeline.data_ingestion import load_raw, split_data
from pipeline.preprocessing import prepare_features
from pipeline.training import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analyse_sweep")

SWEEP_RUN_NAMES = [f"{cfg['model_type']}-{i:02d}" for i, cfg in enumerate(EXPERIMENT_GRID, 1)]


def latest_runs_by_name(run_names: List[str],
                        experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> Dict[str, str]:
    """Most recent finished run id for each run name (the sweep may have run twice)."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist — run the sweep first")
    out: Dict[str, str] = {}
    for name in run_names:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{name}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            out[name] = runs[0].info.run_id
    return out


def selection_gap(y_proba: np.ndarray, groups: np.ndarray,
                  threshold: float = REVIEW_THRESHOLD) -> float:
    """Largest difference in selection rate between any two groups."""
    selected = y_proba >= threshold
    rates = [selected[groups == g].mean() for g in np.unique(groups)]
    return float(max(rates) - min(rates)) if len(rates) > 1 else 0.0


def paired_bootstrap(
    y_true: np.ndarray,
    scores: Dict[str, np.ndarray],
    groups: np.ndarray,
    reference: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Point estimates and 95% intervals, every model on the same resamples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    names = list(scores)
    boot_auc = {m: np.empty(n_boot) for m in names}
    boot_gap = {m: np.empty(n_boot) for m in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in names:
            boot_auc[m][b] = roc_auc_score(y_true[idx], scores[m][idx])
            boot_gap[m][b] = selection_gap(scores[m][idx], groups[idx])

    def ci(values: np.ndarray) -> List[float]:
        return [round(float(np.percentile(values, 2.5)), 4),
                round(float(np.percentile(values, 97.5)), 4)]

    out: Dict[str, Dict[str, Any]] = {}
    for m in names:
        out[m] = {
            "roc_auc": round(float(roc_auc_score(y_true, scores[m])), 4),
            "roc_auc_ci": ci(boot_auc[m]),
            "pr_auc": round(float(average_precision_score(y_true, scores[m])), 4),
            "fairness_gap": round(selection_gap(scores[m], groups), 4),
            "fairness_gap_ci": ci(boot_gap[m]),
            "d_auc_ci": ci(boot_auc[m] - boot_auc[reference]),
            "d_gap_ci": ci(boot_gap[m] - boot_gap[reference]),
            "d_auc_se": round(float(np.std(boot_auc[m] - boot_auc[reference])), 4),
        }
    return out


def baselines(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """What the gate should be compared with: a no-model rule and the outcome gap."""
    rates = y_test.groupby(X_test[SENSITIVE_ATTRIBUTE]).mean()
    return {
        "positive_rate": round(float(y_test.mean()), 4),
        "pay0_rule_roc_auc": round(float(roc_auc_score(y_test, X_test["PAY_0"])), 4),
        "pay0_rule_pr_auc": round(float(average_precision_score(y_test, X_test["PAY_0"])), 4),
        "default_rate_by_group": {str(k): round(float(v), 4) for k, v in rates.items()},
        "default_rate_gap": round(float(rates.max() - rates.min()), 4),
    }


def analyse(reference: str = "hgb-06", n_boot: int = 1000,
            run_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Score every sweep run on the test split and bootstrap the comparisons."""
    setup_mlflow()
    run_ids = latest_runs_by_name(run_names or SWEEP_RUN_NAMES)
    if reference not in run_ids:
        raise ValueError(f"reference run '{reference}' not found; have {sorted(run_ids)}")

    _, X_test, _, y_test = split_data(load_raw())
    X_prepared = prepare_features(X_test)
    scores = {}
    for name, run_id in run_ids.items():
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        scores[name] = model.predict_proba(X_prepared)[:, 1]

    result = {
        "reference": reference,
        "n_boot": n_boot,
        "n_test": int(len(y_test)),
        "run_ids": run_ids,
        "baselines": baselines(X_test, y_test),
        "runs": paired_bootstrap(
            y_test.to_numpy(), scores, X_test[SENSITIVE_ATTRIBUTE].to_numpy(),
            reference=reference, n_boot=n_boot,
        ),
    }
    return result


def print_report(result: Dict[str, Any]) -> None:
    """The table that goes into the report."""
    b = result["baselines"]
    print("\n" + "=" * 100)
    print(f"test rows {result['n_test']}, bootstrap rounds {result['n_boot']}, "
          f"reference {result['reference']}")
    print(f"PAY_0 rule: roc_auc {b['pay0_rule_roc_auc']:.4f}  pr_auc {b['pay0_rule_pr_auc']:.4f}"
          f"   (no-skill pr_auc = positive rate {b['positive_rate']:.4f})")
    print(f"actual default rate by {SENSITIVE_ATTRIBUTE}: {b['default_rate_by_group']} "
          f"-> outcome gap {b['default_rate_gap']:.4f}")
    print("-" * 100)
    print(f"{'run':<11}{'roc_auc':>8}{'95% CI':>18}{'d_auc vs ref':>22}"
          f"{'gap':>8}{'95% CI':>18}{'d_gap vs ref':>22}")
    for name, r in sorted(result["runs"].items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"{name:<11}{r['roc_auc']:>8.4f}{str(r['roc_auc_ci']):>18}{str(r['d_auc_ci']):>22}"
              f"{r['fairness_gap']:>8.4f}{str(r['fairness_gap_ci']):>18}{str(r['d_gap_ci']):>22}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="hgb-06",
                        help="run name the differences are measured against")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    result = analyse(reference=args.reference, n_boot=args.n_boot)
    print_report(result)
    out = ARTIFACTS_DIR / "sweep_analysis.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
