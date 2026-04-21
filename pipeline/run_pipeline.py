"""
Orchestrate the full ML pipeline end-to-end.

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --model-type svd --n-factors 100 --register
"""

import argparse
import logging
from typing import Any, Dict

from pipeline.config import DEFAULT_MODEL_TYPE, MODEL_CONFIGS
from pipeline.data_ingestion import load_and_split
from pipeline.evaluation import evaluate_model
from pipeline.preprocessing import preprocess_data
from pipeline.registry import register_best_model
from pipeline.training import setup_mlflow, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def run_pipeline(
    model_type: str = DEFAULT_MODEL_TYPE,
    register: bool = False,
    **model_params,
) -> Dict[str, Any]:
    """Execute all pipeline stages and return a results summary."""
    log.info("=" * 60)
    log.info("ML Pipeline starting")
    log.info("=" * 60)

    outcome: Dict[str, Any] = {"status": "started", "stages": {}}

    try:
        log.info("[1/5] Configuring MLflow")
        setup_mlflow()
        outcome["stages"]["mlflow_setup"] = "success"

        log.info("[2/5] Ingesting data")
        trainset, testset, data_stats = load_and_split()
        outcome["stages"]["data_ingestion"] = {"status": "success", "stats": data_stats}

        log.info("[3/5] Preprocessing")
        pp_report = preprocess_data(trainset, testset)
        outcome["stages"]["preprocessing"] = {
            "status": "success" if pp_report["preprocessing_successful"] else "warning",
            "report": pp_report,
        }

        log.info("[4/5] Training %s", model_type)
        if not model_params:
            model_params = MODEL_CONFIGS.get(model_type, {})
        algo, run_id = train_model(
            trainset,
            model_type=model_type,
            run_name=f"pipeline_{model_type}",
            **model_params,
        )
        outcome["stages"]["training"] = {
            "status": "success",
            "run_id": run_id,
            "model_type": model_type,
            "params": model_params,
        }

        log.info("[5/5] Evaluating")
        scores = evaluate_model(algo, testset, run_id)
        outcome["stages"]["evaluation"] = {"status": "success", "metrics": scores}

        if register:
            log.info("[Optional] Registering best model")
            reg = register_best_model()
            outcome["stages"]["registration"] = {"status": "success", "info": reg}

        outcome["status"] = "completed"
        log.info("=" * 60)
        log.info("Pipeline complete  |  model=%s  run_id=%s  RMSE=%.4f  MAE=%.4f",
                 model_type, run_id, scores.get("rmse", 0), scores.get("mae", 0))
        log.info("=" * 60)

    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        outcome["status"] = "failed"
        outcome["error"] = str(exc)
        raise

    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ML training pipeline")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE, choices=["svd", "nmf", "knn"])
    parser.add_argument("--n-factors", type=int, default=None)
    parser.add_argument("--n-epochs",  type=int, default=None)
    parser.add_argument("--register",  action="store_true")
    args = parser.parse_args()

    extra: Dict[str, Any] = {}
    if args.n_factors:
        extra["n_factors"] = args.n_factors
    if args.n_epochs:
        extra["n_epochs"] = args.n_epochs

    run_pipeline(model_type=args.model_type, register=args.register, **extra)


if __name__ == "__main__":
    main()
