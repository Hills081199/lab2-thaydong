"""Model evaluation stage — computes metrics and logs them to MLflow."""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from surprise import accuracy

from pipeline.config import ARTIFACTS_DIR

log = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    testset: List,
    run_id: str,
    log_to_mlflow: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the trained model against the test set.

    Computes RMSE and MAE, optionally logs them (and a distribution plot)
    to the existing MLflow run identified by run_id.
    """
    log.info("Evaluating model on %d test samples", len(testset))
    preds = model.test(testset)

    rmse_val = accuracy.rmse(preds, verbose=False)
    mae_val  = accuracy.mae(preds,  verbose=False)
    scores = {"rmse": rmse_val, "mae": mae_val}

    if log_to_mlflow:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(scores)
            fig = create_prediction_distribution_plot(preds)
            mlflow.log_figure(fig, "pred_distribution.png")
            plt.close(fig)

    log.info("Evaluation done  |  RMSE=%.4f  MAE=%.4f", rmse_val, mae_val)
    return scores


def calculate_additional_metrics(predictions: List) -> Dict[str, Any]:
    """Compute MSE, MAPE and coverage beyond the standard RMSE/MAE."""
    actuals   = np.array([p.r_ui for p in predictions])
    estimated = np.array([p.est  for p in predictions])

    mse = float(np.mean((actuals - estimated) ** 2))

    nonzero = actuals != 0
    mape: Optional[float] = None
    if np.any(nonzero):
        mape = float(
            np.mean(np.abs((actuals[nonzero] - estimated[nonzero]) / actuals[nonzero])) * 100
        )

    return {
        "mse":           mse,
        "mape":          mape,
        "n_predictions": len(predictions),
    }


def create_prediction_distribution_plot(predictions: List) -> plt.Figure:
    """Return a 3-panel figure: scatter, actual distribution, error distribution."""
    actuals   = [p.r_ui for p in predictions]
    estimated = [p.est  for p in predictions]
    errors    = np.array(estimated) - np.array(actuals)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(actuals, estimated, alpha=0.1, s=1)
    axes[0].plot([1, 5], [1, 5], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual Rating")
    axes[0].set_ylabel("Predicted Rating")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    axes[1].hist(actuals, bins=20, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Actual Rating Distribution")

    axes[2].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[2].axvline(x=0, color="r", linestyle="--")
    axes[2].set_xlabel("Prediction Error")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Error Distribution")

    plt.tight_layout()
    return fig


def create_error_by_rating_plot(predictions: List) -> plt.Figure:
    """Box-plot of prediction errors grouped by actual rating."""
    groups: Dict[int, List[float]] = {}
    for p in predictions:
        bucket = round(p.r_ui)
        groups.setdefault(bucket, []).append(p.est - p.r_ui)

    fig, ax = plt.subplots(figsize=(10, 6))
    ratings = sorted(groups)
    ax.boxplot([groups[r] for r in ratings], positions=range(len(ratings)), widths=0.6)
    ax.set_xticklabels([str(r) for r in ratings])
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Error by Actual Rating")
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    return fig


def save_evaluation_report(metrics: Dict, filepath: str) -> None:
    """Persist evaluation metrics to a plain-text file."""
    with open(filepath, "w") as fh:
        fh.write("Model Evaluation Report\n")
        fh.write("=" * 40 + "\n\n")
        for name, val in metrics.items():
            fh.write(f"{name}: {val:.4f}\n" if isinstance(val, float) else f"{name}: {val}\n")
    log.info("Evaluation report saved to %s", filepath)
