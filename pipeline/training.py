"""Model training stage with MLflow experiment tracking."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from surprise import KNNBasic, NMF, SVD

from pipeline.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_CONFIGS,
    MODELS_DIR,
)

log = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {
    "svd": SVD,
    "nmf": NMF,
    "knn": KNNBasic,
}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Configure MLflow tracking URI and active experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    log.info("MLflow ready  |  uri=%s  experiment=%s", tracking_uri, experiment_name)


def get_model_class(model_type: str):
    """Return the Surprise algorithm class for the given model type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. Supported: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type]


def train_model(
    trainset: Any,
    model_type: str = "svd",
    run_name: Optional[str] = None,
    **model_params,
) -> Tuple[Any, str]:
    """
    Train a collaborative-filtering model and log everything to MLflow.

    Returns the trained algorithm and the MLflow run_id.
    """
    algo_cls = get_model_class(model_type)
    algo = algo_cls(**model_params)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        for key, val in model_params.items():
            mlflow.log_param(key, val)

        log.info("Training %s  params=%s", model_type, model_params)
        algo.fit(trainset)

        out_path = MODELS_DIR / f"{model_type}_model.pkl"
        with out_path.open("wb") as fh:
            pickle.dump(algo, fh)
        mlflow.log_artifact(str(out_path))

        run_id = run.info.run_id
        log.info("Run complete  |  run_id=%s", run_id)

    return algo, run_id


def train_with_config(trainset: Any, config: Dict[str, Any]) -> Tuple[Any, str]:
    """Train using a flat configuration dict (model_type + hyperparameters)."""
    cfg = config.copy()
    mtype = cfg.pop("model_type")
    return train_model(trainset, model_type=mtype, **cfg)


def get_default_params(model_type: str) -> Dict[str, Any]:
    """Return the default hyperparameters for a given model type."""
    return MODEL_CONFIGS.get(model_type, {})


def list_available_models() -> list:
    """List all supported model type names."""
    return list(MODEL_REGISTRY)


if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split

    setup_mlflow()
    trainset, _, _ = load_and_split()
    algo, rid = train_model(trainset, model_type="svd", run_name="smoke_test", n_factors=10, n_epochs=5)
    print("Run ID:", rid)
    print("Available models:", list_available_models())
