"""Data preprocessing and validation stage."""

import logging
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)


def validate_trainset(trainset: Any) -> Dict[str, Any]:
    """Validate a Surprise trainset and return a report dict."""
    report: Dict[str, Any] = {
        "is_valid": True,
        "n_users": trainset.n_users,
        "n_items": trainset.n_items,
        "n_ratings": trainset.n_ratings,
        "issues": [],
    }

    if trainset.n_users < 10:
        report["is_valid"] = False
        report["issues"].append("Too few users (< 10)")
    if trainset.n_items < 10:
        report["is_valid"] = False
        report["issues"].append("Too few items (< 10)")
    if trainset.n_ratings < 100:
        report["is_valid"] = False
        report["issues"].append("Too few ratings (< 100)")

    lo, hi = trainset.rating_scale
    if lo < 0:
        report["issues"].append(f"Negative rating minimum detected: {lo}")

    log.info("Trainset validation: valid=%s issues=%s", report["is_valid"], report["issues"])
    return report


def validate_testset(testset: List) -> Dict[str, Any]:
    """Validate a Surprise testset."""
    report: Dict[str, Any] = {"is_valid": True, "n_ratings": len(testset), "issues": []}

    if len(testset) < 10:
        report["is_valid"] = False
        report["issues"].append("Too few test ratings (< 10)")

    for _, _, r in testset[:100]:
        if r is None:
            report["is_valid"] = False
            report["issues"].append("Missing ratings detected")
            break

    log.info("Testset validation: valid=%s  n=%d", report["is_valid"], len(testset))
    return report


def get_rating_distribution(trainset: Any) -> Dict[str, Any]:
    """Compute descriptive statistics for the rating distribution."""
    vals = np.array([r for uid in trainset.all_users() for _, r in trainset.ur[uid]])
    dist = {
        "mean":   float(np.mean(vals)),
        "std":    float(np.std(vals)),
        "min":    float(np.min(vals)),
        "max":    float(np.max(vals)),
        "median": float(np.median(vals)),
        "q25":    float(np.percentile(vals, 25)),
        "q75":    float(np.percentile(vals, 75)),
    }
    log.info("Rating distribution  mean=%.2f  std=%.2f", dist["mean"], dist["std"])
    return dist


def get_user_activity_stats(trainset: Any) -> Dict[str, Any]:
    """Compute per-user rating count statistics."""
    counts = np.array([len(trainset.ur[uid]) for uid in trainset.all_users()])
    return {
        "mean_ratings_per_user":   float(np.mean(counts)),
        "median_ratings_per_user": float(np.median(counts)),
        "min_ratings_per_user":    int(np.min(counts)),
        "max_ratings_per_user":    int(np.max(counts)),
        "users_with_few_ratings":  int(np.sum(counts < 5)),
    }


def get_item_popularity_stats(trainset: Any) -> Dict[str, Any]:
    """Compute per-item rating count statistics."""
    counts = np.array([len(trainset.ir[iid]) for iid in trainset.all_items()])
    return {
        "mean_ratings_per_item":   float(np.mean(counts)),
        "median_ratings_per_item": float(np.median(counts)),
        "min_ratings_per_item":    int(np.min(counts)),
        "max_ratings_per_item":    int(np.max(counts)),
        "items_with_few_ratings":  int(np.sum(counts < 5)),
    }


def preprocess_data(trainset: Any, testset: List) -> Dict[str, Any]:
    """Run all validation and analysis steps; return a comprehensive report."""
    log.info("Starting preprocessing pipeline")
    report = {
        "trainset_validation": validate_trainset(trainset),
        "testset_validation":  validate_testset(testset),
        "rating_distribution": get_rating_distribution(trainset),
        "user_activity":       get_user_activity_stats(trainset),
        "item_popularity":     get_item_popularity_stats(trainset),
    }
    report["preprocessing_successful"] = (
        report["trainset_validation"]["is_valid"]
        and report["testset_validation"]["is_valid"]
    )
    log.info("Preprocessing finished  success=%s", report["preprocessing_successful"])
    return report


if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split

    trainset, testset, _ = load_and_split()
    rpt = preprocess_data(trainset, testset)
    print("Success:", rpt["preprocessing_successful"])
    print("Mean rating:", round(rpt["rating_distribution"]["mean"], 2))
