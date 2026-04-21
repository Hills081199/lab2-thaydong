"""Airflow DAG — weekly ML training pipeline for movie rating prediction."""

import os
import pickle
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# ---------------------------------------------------------------------------
# DAG defaults
# ---------------------------------------------------------------------------
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "movie_rating_training",
    default_args=default_args,
    description="ML Training Pipeline for Movie Rating Prediction",
    schedule_interval="0 0 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "movie-rating"],
)

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
TMP_DIR = "/tmp/airflow_ml_pipeline"


def load_data_task(**ctx):
    from pipeline.data_ingestion import load_and_split

    os.makedirs(TMP_DIR, exist_ok=True)
    trainset, testset, stats = load_and_split()

    with open(f"{TMP_DIR}/trainset.pkl", "wb") as fh:
        pickle.dump(trainset, fh)
    with open(f"{TMP_DIR}/testset.pkl", "wb") as fh:
        pickle.dump(testset, fh)

    ctx["ti"].xcom_push(key="data_stats", value=stats)
    ctx["ti"].xcom_push(key="data_path",  value=TMP_DIR)
    return f"Data loaded: {stats['n_ratings']} ratings"


def preprocess_data_task(**ctx):
    from pipeline.preprocessing import preprocess_data

    tmp = ctx["ti"].xcom_pull(key="data_path")

    with open(f"{tmp}/trainset.pkl", "rb") as fh:
        trainset = pickle.load(fh)
    with open(f"{tmp}/testset.pkl", "rb") as fh:
        testset = pickle.load(fh)

    rpt = preprocess_data(trainset, testset)
    ctx["ti"].xcom_push(key="preprocess_report", value=rpt)
    return "Preprocessing complete"


def train_model_task(**ctx):
    from pipeline.training import setup_mlflow, train_model

    tmp = ctx["ti"].xcom_pull(key="data_path")

    with open(f"{tmp}/trainset.pkl", "rb") as fh:
        trainset = pickle.load(fh)

    setup_mlflow()
    algo, run_id = train_model(
        trainset,
        model_type="svd",
        run_name=f"airflow_{ctx['ds']}",
        n_factors=100,
        n_epochs=20,
    )

    with open(f"{tmp}/model.pkl", "wb") as fh:
        pickle.dump(algo, fh)

    ctx["ti"].xcom_push(key="run_id", value=run_id)
    return f"Training done — run_id={run_id}"


def evaluate_model_task(**ctx):
    from pipeline.evaluation import evaluate_model

    tmp    = ctx["ti"].xcom_pull(key="data_path")
    run_id = ctx["ti"].xcom_pull(key="run_id")

    with open(f"{tmp}/model.pkl", "rb") as fh:
        model = pickle.load(fh)
    with open(f"{tmp}/testset.pkl", "rb") as fh:
        testset = pickle.load(fh)

    scores = evaluate_model(model, testset, run_id)
    ctx["ti"].xcom_push(key="metrics", value=scores)
    return f"Evaluation done — RMSE={scores['rmse']:.4f}"


def decide_registration(**ctx):
    metrics = ctx["ti"].xcom_pull(key="metrics")
    if metrics and metrics.get("rmse", float("inf")) < 1.0:
        return "register_model"
    return "skip_registration"


def register_model_task(**ctx):
    from pipeline.registry import register_best_model

    result = register_best_model()
    return f"Registered {result['model_name']} v{result['version']}"


def cleanup_task(**ctx):
    import shutil

    tmp = ctx["ti"].xcom_pull(key="data_path")
    if tmp and os.path.exists(tmp):
        shutil.rmtree(tmp)
    return "Cleanup complete"


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
t_load      = PythonOperator(task_id="load_data",          python_callable=load_data_task,       dag=dag)
t_preproc   = PythonOperator(task_id="preprocess_data",     python_callable=preprocess_data_task, dag=dag)
t_train     = PythonOperator(task_id="train_model",         python_callable=train_model_task,     dag=dag)
t_eval      = PythonOperator(task_id="evaluate_model",      python_callable=evaluate_model_task,  dag=dag)
t_decide    = BranchPythonOperator(task_id="decide_registration", python_callable=decide_registration, dag=dag)
t_register  = PythonOperator(task_id="register_model",      python_callable=register_model_task,  dag=dag)
t_skip      = DummyOperator(task_id="skip_registration",    dag=dag)
t_cleanup   = PythonOperator(
    task_id="cleanup",
    python_callable=cleanup_task,
    trigger_rule="none_failed",
    dag=dag,
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
t_load >> t_preproc >> t_train >> t_eval >> t_decide
t_decide >> [t_register, t_skip]
[t_register, t_skip] >> t_cleanup
