"""
Hyperparameter sweep.

Runs every configuration in config.EXPERIMENT_GRID as its own MLflow run, then
prints a leaderboard.

Usage:
    python -m experiments.run_experiments
    python -m experiments.run_experiments --model-type hgb
    python -m experiments.run_experiments --top 5
    python -m experiments.run_experiments --leaderboard-only --promote-best
"""

import argparse
import json
import logging
from typing import Any, Dict, List

from pipeline.config import EXPERIMENT_GRID, PRIMARY_METRIC, REGISTERED_MODEL_NAME
from pipeline.data_ingestion import load_and_split, load_raw
from pipeline.evaluation import evaluate_model
from pipeline.registry import compare_runs, find_best_run, promote_model
from pipeline.training import setup_mlflow, train_model
from pipeline.validation import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("experiments")


def run_grid(grid: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Train and evaluate one run per configuration."""
    grid = grid or EXPERIMENT_GRID
    setup_mlflow()

    raw = load_raw()
    report = validate_dataset(raw)
    X_train, X_test, y_train, y_test, stats = load_and_split()

    results = []
    for i, cfg in enumerate(grid, 1):
        cfg = dict(cfg)
        model_type = cfg.pop("model_type")
        name = f"{model_type}-{i:02d}"
        logger.info("[%s/%s] %s %s", i, len(grid), name, cfg)

        model, run_id = train_model(
            X_train, y_train,
            model_type=model_type,
            run_name=name,
            data_stats=stats,
            validation_report=report,
            **cfg,
        )
        metrics = evaluate_model(model, X_test, y_test, run_id=run_id)
        results.append({"run_name": name, "run_id": run_id, "model_type": model_type,
                        "params": cfg, "metrics": metrics})

    return results


def leaderboard(top: int = 10) -> None:
    """Print the best runs recorded so far."""
    rows = compare_runs(metric=PRIMARY_METRIC, top_n=top, ascending=False)
    if not rows:
        print("No runs found.")
        return
    print("\n" + "=" * 86)
    print(f"{'run':<14}{'model':<9}{'roc_auc':>9}{'pr_auc':>9}{'recall':>9}{'fair gap':>10}{'run_id':>26}")
    print("-" * 86)
    for r in rows:
        m = r["metrics"]
        print(f"{r['run_name'][:13]:<14}{r['params'].get('model_type', '?'):<9}"
              f"{m.get('roc_auc', 0):>9.4f}{m.get('pr_auc', 0):>9.4f}"
              f"{m.get('recall', 0):>9.4f}{m.get('fairness_gap', 0):>10.4f}"
              f"{r['run_id'][:24]:>26}")
    print("=" * 86)


def promote_best(metric: str = PRIMARY_METRIC) -> Dict[str, Any]:
    """Send the top run by `metric` through the same gate the pipeline uses.

    Ranking picks the candidate; the quality gate and the champion comparison
    still decide what alias it gets. The best-scoring run is not automatically
    the one that ships.
    """
    setup_mlflow()
    best = find_best_run(metric=metric)
    result = promote_model(best["run_id"], best["metrics"], REGISTERED_MODEL_NAME)
    gate = result["quality_gate"]
    print(f"\nbest by {metric}: run {best['run_id'][:12]} "
          f"({best['params'].get('model_type', '?')}, {metric}={best['metrics'][metric]:.4f})")
    print(f"registry: {result['model_name']} v{result['version']} -> {result['outcome']}")
    if not gate["passed"]:
        print("gate failed: " + json.dumps(
            {k: gate["detail"][k] for k in gate["failed_checks"]}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", default=None, choices=["logreg", "rf", "hgb"],
                        help="only run configurations for this model type")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--leaderboard-only", action="store_true")
    parser.add_argument("--promote-best", action="store_true",
                        help=f"register the best run by {PRIMARY_METRIC} and apply the gate")
    args = parser.parse_args()

    if not args.leaderboard_only:
        grid = EXPERIMENT_GRID
        if args.model_type:
            grid = [c for c in grid if c["model_type"] == args.model_type]
        run_grid(grid)

    leaderboard(args.top)

    if args.promote_best:
        promote_best()


if __name__ == "__main__":
    main()
