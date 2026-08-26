"""Unit tests for ModelSelector.

Tests:
1. test_classification_all_models_succeed — with Iris-like data
2. test_regression_all_models_succeed
3. test_custom_model_list
4. test_sort_order_accuracy_descending
5. test_failed_model_recorded
6. test_invalid_cv_folds_raises

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from kaggle_ml_toolkit.model_selector import ModelSelector


@pytest.fixture
def classification_data():
    """Iris-like classification dataset (small, 2-class)."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"]), pd.Series(y)


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=4,
        n_informative=3,
        random_state=42,
    )
    return pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"]), pd.Series(y)


@pytest.fixture
def selector():
    """ModelSelector instance."""
    return ModelSelector()


class TestClassificationAllModelsSucceed:
    """Test that all classification models train successfully on clean data."""

    def test_classification_all_models_succeed(self, selector, classification_data):
        """All default classification models succeed with Iris-like data."""
        X, y = classification_data
        result = selector.compare(X, y, problem_type="classification", cv_folds=3)

        # Should have a row for each classification model
        assert len(result) == len(ModelSelector.CLASSIFICATION_MODELS)

        # All should succeed
        assert all(result["status"] == "success")

        # All should have valid scores
        assert all(result["mean_score"].notna())
        assert all(result["std_score"].notna())

        # All error_message should be None
        assert all(result["error_message"].isna())

        # Required columns present
        expected_cols = {"model_name", "mean_score", "std_score", "status", "error_message"}
        assert set(result.columns) == expected_cols


class TestRegressionAllModelsSucceed:
    """Test that all regression models train successfully on clean data."""

    def test_regression_all_models_succeed(self, selector, regression_data):
        """All default regression models succeed with simple regression data."""
        X, y = regression_data
        result = selector.compare(X, y, problem_type="regression", cv_folds=3)

        # Should have a row for each regression model
        assert len(result) == len(ModelSelector.REGRESSION_MODELS)

        # All should succeed
        assert all(result["status"] == "success")

        # All should have valid scores
        assert all(result["mean_score"].notna())
        assert all(result["std_score"].notna())


class TestCustomModelList:
    """Test that a custom model list parameter is respected."""

    def test_custom_model_list(self, selector, classification_data):
        """Only requested models are evaluated when models param is given."""
        X, y = classification_data
        custom_models = ["logistic_regression", "random_forest"]

        result = selector.compare(
            X, y,
            problem_type="classification",
            models=custom_models,
            cv_folds=3,
        )

        assert len(result) == 2
        assert set(result["model_name"].tolist()) == set(custom_models)
        assert all(result["status"] == "success")


class TestSortOrderAccuracyDescending:
    """Test that results are sorted by mean_score descending for accuracy."""

    def test_sort_order_accuracy_descending(self, selector, classification_data):
        """Results are sorted by mean_score in descending order for accuracy."""
        X, y = classification_data
        result = selector.compare(
            X, y,
            problem_type="classification",
            metric="accuracy",
            cv_folds=3,
        )

        successful = result[result["status"] == "success"]
        scores = successful["mean_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_sort_order_rmse_ascending(self, selector, regression_data):
        """Results are sorted by mean_score ascending for RMSE (lower is better)."""
        X, y = regression_data
        result = selector.compare(
            X, y,
            problem_type="regression",
            metric="rmse",
            cv_folds=3,
        )

        successful = result[result["status"] == "success"]
        scores = successful["mean_score"].tolist()
        assert scores == sorted(scores)


class TestFailedModelRecorded:
    """Test that a model that fails is recorded with an error message."""

    def test_failed_model_recorded(self, selector, classification_data):
        """An unknown model name is recorded as failed with error_message."""
        X, y = classification_data
        models = ["logistic_regression", "totally_fake_model"]

        result = selector.compare(
            X, y,
            problem_type="classification",
            models=models,
            cv_folds=3,
        )

        assert len(result) == 2

        # Valid model should succeed
        lr_row = result[result["model_name"] == "logistic_regression"].iloc[0]
        assert lr_row["status"] == "success"
        assert pd.notna(lr_row["mean_score"])

        # Fake model should be recorded as failed
        fake_row = result[result["model_name"] == "totally_fake_model"].iloc[0]
        assert fake_row["status"] == "failed"
        assert pd.notna(fake_row["error_message"])
        assert np.isnan(fake_row["mean_score"])

    def test_failed_model_does_not_disrupt_others(self, selector, classification_data):
        """A failed model does not prevent other models from being evaluated."""
        X, y = classification_data
        models = ["bad_model_1", "random_forest", "bad_model_2", "knn"]

        result = selector.compare(
            X, y,
            problem_type="classification",
            models=models,
            cv_folds=3,
        )

        assert len(result) == 4

        # Good models should succeed
        rf_row = result[result["model_name"] == "random_forest"].iloc[0]
        assert rf_row["status"] == "success"

        knn_row = result[result["model_name"] == "knn"].iloc[0]
        assert knn_row["status"] == "success"

        # Bad models should fail
        bad1 = result[result["model_name"] == "bad_model_1"].iloc[0]
        assert bad1["status"] == "failed"

        bad2 = result[result["model_name"] == "bad_model_2"].iloc[0]
        assert bad2["status"] == "failed"


class TestInvalidCvFoldsRaises:
    """Test that invalid cv_folds raises ValueError."""

    def test_cv_folds_below_minimum(self, selector, classification_data):
        """cv_folds < 2 raises ValueError."""
        X, y = classification_data
        with pytest.raises(ValueError, match="cv_folds must be between 2 and 20"):
            selector.compare(X, y, cv_folds=1)

    def test_cv_folds_above_maximum(self, selector, classification_data):
        """cv_folds > 20 raises ValueError."""
        X, y = classification_data
        with pytest.raises(ValueError, match="cv_folds must be between 2 and 20"):
            selector.compare(X, y, cv_folds=21)

    def test_cv_folds_zero(self, selector, classification_data):
        """cv_folds = 0 raises ValueError."""
        X, y = classification_data
        with pytest.raises(ValueError, match="cv_folds must be between 2 and 20"):
            selector.compare(X, y, cv_folds=0)

    def test_cv_folds_negative(self, selector, classification_data):
        """cv_folds = -1 raises ValueError."""
        X, y = classification_data
        with pytest.raises(ValueError, match="cv_folds must be between 2 and 20"):
            selector.compare(X, y, cv_folds=-1)
