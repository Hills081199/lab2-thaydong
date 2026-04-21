"""Data ingestion stage — loads and splits the MovieLens dataset."""

import logging
from typing import Any, Dict, Tuple

from surprise import Dataset
from surprise.model_selection import train_test_split

from pipeline.config import DATASET_NAME, RANDOM_STATE, TEST_SIZE

log = logging.getLogger(__name__)


def load_data(dataset_name: str = DATASET_NAME) -> Any:
    """Download and return a Surprise built-in dataset."""
    log.info("Loading dataset '%s'", dataset_name)
    try:
        ds = Dataset.load_builtin(dataset_name)
        log.info("Dataset '%s' ready", dataset_name)
        return ds
    except Exception as exc:
        log.error("Could not load dataset '%s': %s", dataset_name, exc)
        raise


def split_data(
    data: Any,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Any]:
    """Split dataset into train and test sets."""
    log.info("Splitting data  |  test_size=%.2f  random_state=%d", test_size, random_state)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    log.info("Split done  |  train=%d  test=%d", trainset.n_ratings, len(testset))
    return trainset, testset


def get_data_stats(data: Any) -> Dict[str, Any]:
    """Return summary statistics for the full dataset."""
    full_ts = data.build_full_trainset()
    stats = {
        "n_users": full_ts.n_users,
        "n_items": full_ts.n_items,
        "n_ratings": full_ts.n_ratings,
        "rating_scale": (full_ts.rating_scale[0], full_ts.rating_scale[1]),
        "global_mean": full_ts.global_mean,
    }
    log.info("Dataset stats: %s", stats)
    return stats


def load_and_split(
    dataset_name: str = DATASET_NAME,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Convenience wrapper: load, compute stats, and split in one call."""
    ds = load_data(dataset_name)
    stats = get_data_stats(ds)
    trainset, testset = split_data(ds, test_size, random_state)
    return trainset, testset, stats


if __name__ == "__main__":
    trainset, testset, stats = load_and_split()
    print(f"Users: {stats['n_users']}  Items: {stats['n_items']}  Ratings: {stats['n_ratings']}")
    print(f"Train: {trainset.n_ratings}  Test: {len(testset)}")
