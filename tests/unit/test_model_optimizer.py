"""Unit tests for ModelOptimizer.

Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from kaggle_ml_toolkit.model_optimizer import ModelOptimizer


@pytest.fixture
def classifier_data():
    """Small classification dataset for fast tests."""
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def optimizer():
    return ModelOptimizer()


@pytest.fixture
def simple_param_grid():
    return {
        "n_estimators": [5, 10, 20],
        "max_depth": [2, 3],
    }


class TestGridSearch:
    """Tests for grid search optimization."""

    def test_grid_search_returns_best(self, optimizer, classifier_data, simple_param_grid):
        """Verify best_params, best_score, best_model keys are present."""
        X, y = classifier_data
        result = optimizer.optimize(
            model=RandomForestClassifier(random_state=42),
            param_grid=simple_param_grid,
            X=X,
            y=y,
            method="grid",
            scoring="accuracy",
            cv_folds=2,
            random_state=42,
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "best_model" in result
        assert isinstance(result["best_params"], dict)
        assert isinstance(result["best_score"], float)
        assert result["best_model"] is not None

    def test_best_params_from_grid_values(self, optimizer, classifier_data, simple_param_grid):
        """Verify all param values exist in original grid."""
        X, y = classifier_data
        result = optimizer.optimize(
            model=RandomForestClassifier(random_state=42),
            param_grid=simple_param_grid,
            X=X,
            y=y,
            method="grid",
            scoring="accuracy",
            cv_folds=2,
            random_state=42,
        )

        for param_name, param_value in result["best_params"].items():
            assert param_name in simple_param_grid
            assert param_value in simple_param_grid[param_name]


class TestRandomizedSearch:
    """Tests for randomized search optimization."""

    def test_randomized_search_works(self, optimizer, classifier_data, simple_param_grid):
        """Verify method='randomized' succeeds and returns expected keys."""
        X, y = classifier_data
        result = optimizer.optimize(
            model=RandomForestClassifier(random_state=42),
            param_grid=simple_param_grid,
            X=X,
            y=y,
            method="randomized",
            scoring="accuracy",
            cv_folds=2,
            n_iter=3,
            random_state=42,
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "best_model" in result
        assert result["method"] == "randomized"


class TestErrorHandling:
    """Tests for error conditions."""

    def test_empty_param_grid_raises(self, optimizer, classifier_data):
        """ValueError for empty grid."""
        X, y = classifier_data
        with pytest.raises(ValueError, match="param_grid must not be empty"):
            optimizer.optimize(
                model=RandomForestClassifier(random_state=42),
                param_grid={},
                X=X,
                y=y,
                method="grid",
                scoring="accuracy",
                cv_folds=2,
                random_state=42,
            )

    def test_runtime_error_all_fail(self, optimizer, classifier_data):
        """RuntimeError when all combos error."""
        X, y = classifier_data
        # Use an invalid parameter that will cause all fits to fail
        bad_grid = {"n_estimators": [-1]}
        with pytest.raises(RuntimeError, match="No valid parameter combination"):
            optimizer.optimize(
                model=RandomForestClassifier(random_state=42),
                param_grid=bad_grid,
                X=X,
                y=y,
                method="grid",
                scoring="accuracy",
                cv_folds=2,
                random_state=42,
            )
