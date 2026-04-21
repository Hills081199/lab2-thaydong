"""
Run multiple hyperparameter configurations and compare them in MLflow.

Usage:
    python -m experiments.run_experiments
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import mlflow

from pipeline.config import EXPERIMENT_CONFIGS, MLFLOW_EXPERIMENT_NAME
from pipeline.data_ingestion import load_and_split
from pipeline.evaluation import evaluate_model
from pipeline.registry import compare_runs
from pipeline.training import setup_mlflow, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_single_experiment(
    trainset: Any,
    testset: Any,
    config: Dict[str, Any],
    experiment_name: str = "hyperparameter-tuning",
) -> Dict[str, Any]:
    """Train and evaluate one configuration; return results dict."""
    mlflow.set_experiment(experiment_name)

    cfg = config.copy()
    mtype = cfg.pop("model_type")

    label_parts = [f"{k}={v}" for k, v in cfg.items() if not isinstance(v, dict)]
    run_label = f"{mtype}_" + "_".join(label_parts)

    algo, run_id = train_model(trainset, model_type=mtype, run_name=run_label, **cfg)
    scores = evaluate_model(algo, testset, run_id)

    return {"config": config, "run_id": run_id, "metrics": scores}


def run_all_experiments(
    configs: List[Dict[str, Any]] = EXPERIMENT_CONFIGS,
    experiment_name: str = "hyperparameter-tuning",
) -> List[Dict[str, Any]]:
    """Load data once and iterate through all configs."""
    log.info("Starting %d experiments", len(configs))
    trainset, testset, _ = load_and_split()

    results: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        log.info("Experiment %d/%d  config=%s", idx, len(configs), cfg)
        try:
            outcome = run_single_experiment(trainset, testset, cfg, experiment_name)
            results.append(outcome)
            log.info("  → RMSE=%.4f  MAE=%.4f", outcome["metrics"]["rmse"], outcome["metrics"]["mae"])
        except Exception as exc:
            log.error("  → Failed: %s", exc)
            results.append({"config": cfg, "error": str(exc)})

    return results


def generate_experiment_report(
    results: List[Dict[str, Any]],
    output_path: str = "experiment_report.md",
) -> str:
    """Build a Markdown summary report from experiment results."""
    successful = [r for r in results if "metrics" in r]

    lines = [
        "# Hyperparameter Experiment Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "## Summary\n",
        f"- Total experiments: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(results) - len(successful)}\n",
        "## Results\n",
        "| Model | Parameters | RMSE | MAE |",
        "|-------|------------|------|-----|",
    ]

    for r in successful:
        mtype  = r["config"].get("model_type", "?")
        params = {k: v for k, v in r["config"].items() if k != "model_type"}
        rmse   = r["metrics"].get("rmse", float("nan"))
        mae    = r["metrics"].get("mae",  float("nan"))
        lines.append(f"| {mtype} | {params} | {rmse:.4f} | {mae:.4f} |")

    if successful:
        best = min(successful, key=lambda x: x["metrics"].get("rmse", float("inf")))
        lines += [
            "\n## Best Model\n",
            f"- Config: `{best['config']}`",
            f"- RMSE: {best['metrics']['rmse']:.4f}",
            f"- Run ID: `{best['run_id']}`",
        ]

    content = "\n".join(lines)
    with open(output_path, "w") as fh:
        fh.write(content)

    return content


def main() -> None:
    log.info("=" * 60)
    log.info("Experiment Runner")
    log.info("=" * 60)

    setup_mlflow()
    results = run_all_experiments(EXPERIMENT_CONFIGS, experiment_name="hyperparameter-tuning")
    report  = generate_experiment_report(results, "experiment_report.md")

    successful = [r for r in results if "metrics" in r]
    if successful:
        best = min(successful, key=lambda x: x["metrics"].get("rmse", float("inf")))
        log.info("Total: %d  Successful: %d  Best RMSE: %.4f",
                 len(results), len(successful), best["metrics"]["rmse"])
        log.info("Best config: %s", best["config"])

    log.info("Top 5 runs:")
    for i, run in enumerate(compare_runs(metric="rmse", top_n=5), 1):
        log.info("  %d. RMSE=%.4f  params=%s", i, run["metrics"].get("rmse", 0), run["params"])

    log.info("Report saved to experiment_report.md")
    log.info("MLflow UI: http://localhost:5000")


if __name__ == "__main__":
    main()
