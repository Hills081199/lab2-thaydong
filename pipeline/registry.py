"""Model registry stage — find, register, and promote the best model."""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import MLFLOW_EXPERIMENT_NAME

log = logging.getLogger(__name__)


def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True,
) -> Dict[str, Any]:
    """
    Query MLflow for the best-performing run in an experiment.

    Returns a dict with run_id, metrics, params, and artifact_uri.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        raise ValueError(f"No experiment named '{experiment_name}' was found")

    direction = "ASC" if ascending else "DESC"
    hits = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} {direction}"],
        max_results=1,
    )

    if not hits:
        raise ValueError(f"Experiment '{experiment_name}' has no completed runs")

    top = hits[0]
    log.info("Best run  |  run_id=%s  %s=%.4f", top.info.run_id, metric, top.data.metrics.get(metric, float("nan")))
    return {
        "run_id":       top.info.run_id,
        "metrics":      top.data.metrics,
        "params":       top.data.params,
        "artifact_uri": top.info.artifact_uri,
    }


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> str:
    """
    Register a model artifact from an MLflow run into the Model Registry.

    Returns the new model version string.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    log.info("Registering model  |  uri=%s  name=%s", model_uri, model_name)

    result = mlflow.register_model(model_uri, model_name)
    log.info("Registered  |  name=%s  version=%s", model_name, result.version)
    return result.version


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production",
) -> None:
    """Transition a registered model version to the given stage."""
    client = MlflowClient()
    log.info("Transitioning %s v%s → %s", model_name, version, stage)
    client.transition_model_version_stage(name=model_name, version=version, stage=stage)
    log.info("Transition complete")


def register_best_model(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = "movie-rating-model",
    metric: str = "rmse",
    stage: str = "Production",
) -> Dict[str, Any]:
    """
    End-to-end helper: find the best run, register it, and promote it to stage.
    """
    best = find_best_run(experiment_name, metric, ascending=True)
    log.info("Best run selected  |  run_id=%s  %s=%.4f", best["run_id"], metric, best["metrics"].get(metric))

    version = register_model(best["run_id"], model_name)
    transition_model_stage(model_name, version, stage)

    return {
        "run_id":     best["run_id"],
        "model_name": model_name,
        "version":    version,
        "stage":      stage,
        "metrics":    best["metrics"],
    }


def list_registered_models() -> List[Dict[str, Any]]:
    """Return a summary list of all models in the registry."""
    client = MlflowClient()
    return [
        {
            "name": m.name,
            "latest_versions": [
                {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                for v in m.latest_versions
            ],
        }
        for m in client.search_registered_models()
    ]


def get_production_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Return info about the current Production version of a model, or None."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            return {"name": model_name, "version": v.version, "stage": v.current_stage, "run_id": v.run_id}
    except Exception as exc:
        log.error("Error fetching production model: %s", exc)
    return None


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve the top-N runs from an experiment sorted by metric."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n,
    )
    return [{"run_id": r.info.run_id, "metrics": r.data.metrics, "params": r.data.params} for r in runs]
