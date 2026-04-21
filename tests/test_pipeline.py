"""
Test suite for the ML pipeline modules.

Run:
    pytest tests/ -v
    pytest tests/ -v -m "not slow" --cov=pipeline
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
class TestDataIngestion:
    def test_load_data_returns_dataset(self):
        from pipeline.data_ingestion import load_data
        ds = load_data("ml-100k")
        assert ds is not None

    def test_split_data_returns_train_and_test(self):
        from pipeline.data_ingestion import load_data, split_data
        ds = load_data("ml-100k")
        trainset, testset = split_data(ds, test_size=0.2)
        assert trainset is not None
        assert len(testset) > 0

    def test_get_data_stats_has_required_keys(self):
        from pipeline.data_ingestion import get_data_stats, load_data
        ds = load_data("ml-100k")
        stats = get_data_stats(ds)
        for key in ("n_users", "n_items", "n_ratings"):
            assert key in stats
        assert stats["n_users"] > 0
        assert stats["n_items"] > 0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_validate_trainset_is_valid(self):
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import validate_trainset
        trainset, _, _ = load_and_split()
        rpt = validate_trainset(trainset)
        assert rpt["is_valid"] is True

    def test_rating_distribution_mean_in_range(self):
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import get_rating_distribution
        trainset, _, _ = load_and_split()
        dist = get_rating_distribution(trainset)
        assert "mean" in dist and "std" in dist
        assert 1.0 <= dist["mean"] <= 5.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TestTraining:
    def test_list_available_models(self):
        from pipeline.training import list_available_models
        models = list_available_models()
        assert "svd" in models
        assert "nmf" in models
        assert "knn" in models

    def test_get_default_params_svd(self):
        from pipeline.training import get_default_params
        p = get_default_params("svd")
        assert "n_factors" in p
        assert "n_epochs" in p

    def test_get_model_class_valid(self):
        from pipeline.training import get_model_class
        from surprise import SVD
        cls = get_model_class("svd")
        assert cls is SVD

    def test_get_model_class_invalid(self):
        from pipeline.training import get_model_class
        with pytest.raises(ValueError):
            get_model_class("unknown_algo")

    @pytest.mark.slow
    def test_train_model_returns_algo_and_run_id(self):
        from pipeline.data_ingestion import load_and_split
        from pipeline.training import setup_mlflow, train_model
        setup_mlflow()
        trainset, _, _ = load_and_split()
        algo, run_id = train_model(trainset, model_type="svd", n_factors=10, n_epochs=3)
        assert algo is not None
        assert isinstance(run_id, str) and run_id


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class TestEvaluation:
    def test_create_plot_returns_figure(self):
        from pipeline.evaluation import create_prediction_distribution_plot
        from surprise import SVD, Dataset
        from surprise.model_selection import train_test_split
        ds = Dataset.load_builtin("ml-100k")
        tr, te = train_test_split(ds, test_size=0.1)
        m = SVD(n_factors=10, n_epochs=5)
        m.fit(tr)
        preds = m.test(te[:100])
        fig = create_prediction_distribution_plot(preds)
        assert fig is not None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_list_registered_models_returns_list(self):
        from pipeline.registry import list_registered_models
        result = list_registered_models()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_required_config_values_present(self):
        from pipeline.config import (
            DATASET_NAME,
            DEFAULT_MODEL_TYPE,
            MODEL_CONFIGS,
            TEST_SIZE,
        )
        assert DATASET_NAME == "ml-100k"
        assert 0 < TEST_SIZE < 1
        assert DEFAULT_MODEL_TYPE in ("svd", "nmf", "knn")
        assert "svd" in MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
class TestIntegration:
    @pytest.mark.slow
    def test_full_pipeline_smoke(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
