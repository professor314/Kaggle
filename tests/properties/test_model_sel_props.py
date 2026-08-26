"""Property-based tests for ModelSelector.

Properties tested:
- Property 11: Model Comparison Result Completeness
- Property 12: Model Comparison Sort Order
- Property 13: Failed Model Graceful Handling

Uses small synthetic datasets with sklearn.datasets.make_classification for speed.

Validates: Requirements 8.3, 8.4, 8.6
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis import Phase
from sklearn.datasets import make_classification, make_regression

from kaggle_ml_toolkit.model_selector import ModelSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classification_data(n_samples=80, n_features=5, random_state=42):
    """Create a small synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=random_state,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), pd.Series(y)


def _make_regression_data(n_samples=80, n_features=5, random_state=42):
    """Create a small synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        random_state=random_state,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), pd.Series(y)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: pick a subset of classification models (1 to all 5)
classification_model_subsets = st.lists(
    st.sampled_from(list(ModelSelector.CLASSIFICATION_MODELS.keys())),
    min_size=1,
    max_size=5,
    unique=True,
)

# Strategy: pick a subset of regression models (1 to all 4)
regression_model_subsets = st.lists(
    st.sampled_from(list(ModelSelector.REGRESSION_MODELS.keys())),
    min_size=1,
    max_size=4,
    unique=True,
)

# Strategy: valid cv_folds
cv_folds_strategy = st.integers(min_value=2, max_value=5)


# ---------------------------------------------------------------------------
# Property 11: Model Comparison Result Completeness
# ---------------------------------------------------------------------------


class TestProperty11ResultCompleteness:
    """For N models requested, result DataFrame has N rows.

    **Validates: Requirements 8.3**
    """

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(models=classification_model_subsets, cv_folds=cv_folds_strategy)
    def test_classification_n_models_n_rows(self, models, cv_folds):
        """N classification models requested produces N rows in result."""
        X, y = _make_classification_data()
        selector = ModelSelector()
        result = selector.compare(
            X, y,
            problem_type="classification",
            models=models,
            cv_folds=cv_folds,
        )
        assert len(result) == len(models)
        assert all(result["model_name"].notna())

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(models=regression_model_subsets, cv_folds=cv_folds_strategy)
    def test_regression_n_models_n_rows(self, models, cv_folds):
        """N regression models requested produces N rows in result."""
        X, y = _make_regression_data()
        selector = ModelSelector()
        result = selector.compare(
            X, y,
            problem_type="regression",
            models=models,
            cv_folds=cv_folds,
        )
        assert len(result) == len(models)
        assert all(result["model_name"].notna())


# ---------------------------------------------------------------------------
# Property 12: Model Comparison Sort Order
# ---------------------------------------------------------------------------


class TestProperty12SortOrder:
    """Result sorted by mean_score in optimal direction.

    - Descending for metrics where higher is better (accuracy, f1, etc.)
    - Ascending for RMSE/MAE (after negation, lower is better)

    **Validates: Requirements 8.4**
    """

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(models=classification_model_subsets)
    def test_classification_sorted_descending(self, models):
        """Classification results sorted by mean_score descending."""
        X, y = _make_classification_data()
        selector = ModelSelector()
        result = selector.compare(
            X, y,
            problem_type="classification",
            models=models,
            metric="accuracy",
            cv_folds=3,
        )
        # Get only successful rows (non-NaN scores)
        successful = result[result["status"] == "success"]
        if len(successful) > 1:
            scores = successful["mean_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(models=regression_model_subsets)
    def test_regression_rmse_sorted_ascending(self, models):
        """RMSE results sorted ascending (lower RMSE is better)."""
        X, y = _make_regression_data()
        selector = ModelSelector()
        result = selector.compare(
            X, y,
            problem_type="regression",
            models=models,
            metric="rmse",
            cv_folds=3,
        )
        successful = result[result["status"] == "success"]
        if len(successful) > 1:
            scores = successful["mean_score"].tolist()
            assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Property 13: Failed Model Graceful Handling
# ---------------------------------------------------------------------------


class TestProperty13FailedModelHandling:
    """A model that raises has non-null error_message without disrupting others.

    **Validates: Requirements 8.6**
    """

    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        valid_models=st.lists(
            st.sampled_from(list(ModelSelector.CLASSIFICATION_MODELS.keys())),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    def test_unknown_model_has_error_message(self, valid_models):
        """An unknown model name produces a row with non-null error_message."""
        X, y = _make_classification_data()
        selector = ModelSelector()

        # Add a bogus model name that doesn't exist
        models_with_bad = valid_models + ["nonexistent_model_xyz"]

        result = selector.compare(
            X, y,
            problem_type="classification",
            models=models_with_bad,
            cv_folds=3,
        )

        # Should have N+1 rows total
        assert len(result) == len(models_with_bad)

        # The bad model should be marked failed with error message
        bad_row = result[result["model_name"] == "nonexistent_model_xyz"]
        assert len(bad_row) == 1
        assert bad_row.iloc[0]["status"] == "failed"
        assert bad_row.iloc[0]["error_message"] is not None
        assert pd.notna(bad_row.iloc[0]["error_message"])

        # Valid models should still have results
        valid_rows = result[result["model_name"].isin(valid_models)]
        assert len(valid_rows) == len(valid_models)
        for _, row in valid_rows.iterrows():
            assert row["status"] == "success"
            assert pd.notna(row["mean_score"])
