"""Unit tests for EnsembleBuilder.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

import warnings

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from kaggle_ml_toolkit.ensemble_builder import EnsembleBuilder


@pytest.fixture
def classification_data():
    """Small classification dataset for fast tests."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    X_train, X_val = X[:70], X[70:]
    y_train, y_val = y[:70], y[70:]
    return X_train, y_train, X_val, y_val


@pytest.fixture
def diverse_base_models():
    """Two diverse base models for ensemble testing."""
    return [
        RandomForestClassifier(n_estimators=10, random_state=42),
        LogisticRegression(max_iter=200, random_state=42),
    ]


@pytest.fixture
def builder():
    return EnsembleBuilder()


class TestVotingSoft:
    """Tests for soft voting ensemble strategy."""

    def test_voting_soft_returns_correct_structure(
        self, builder, classification_data, diverse_base_models
    ):
        """Verify all expected keys present in result dict."""
        X_train, y_train, X_val, y_val = classification_data
        result = builder.build(
            base_models=diverse_base_models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy="voting_soft",
        )

        assert "ensemble_model" in result
        assert "ensemble_score" in result
        assert "base_scores" in result
        assert "improvement" in result
        assert "documentation" in result

        assert result["ensemble_model"] is not None
        assert isinstance(result["ensemble_score"], float)
        assert isinstance(result["base_scores"], dict)
        assert isinstance(result["improvement"], float)
        assert isinstance(result["documentation"], str)

    def test_base_scores_count_matches_models(
        self, builder, classification_data
    ):
        """N base models should produce N entries in base_scores."""
        X_train, y_train, X_val, y_val = classification_data

        models_3 = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(max_iter=200, random_state=42),
            DecisionTreeClassifier(max_depth=3, random_state=42),
        ]

        result = builder.build(
            base_models=models_3,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy="voting_soft",
        )

        assert len(result["base_scores"]) == 3
        assert "model_0" in result["base_scores"]
        assert "model_1" in result["base_scores"]
        assert "model_2" in result["base_scores"]


class TestStacking:
    """Tests for stacking ensemble strategy."""

    def test_stacking_uses_meta_learner(
        self, builder, classification_data, diverse_base_models
    ):
        """Stacking strategy works with a custom meta_learner."""
        X_train, y_train, X_val, y_val = classification_data
        meta_learner = LogisticRegression(max_iter=200, random_state=7)

        result = builder.build(
            base_models=diverse_base_models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy="stacking",
            meta_learner=meta_learner,
        )

        assert "ensemble_model" in result
        assert "ensemble_score" in result
        assert isinstance(result["ensemble_score"], float)
        # Stacking should return a StackingClassifier
        assert hasattr(result["ensemble_model"], "final_estimator_")


class TestIdenticalPredictionsWarning:
    """Tests for the identical predictions warning."""

    def test_identical_predictions_warning(self, builder, classification_data):
        """All identical base models should emit UserWarning."""
        X_train, y_train, X_val, y_val = classification_data

        # Use identical model configurations to get identical predictions
        identical_models = [
            DecisionTreeClassifier(max_depth=2, random_state=42),
            DecisionTreeClassifier(max_depth=2, random_state=42),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.build(
                base_models=identical_models,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                strategy="voting_hard",
            )

            user_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "identical predictions" in str(warning.message)
            ]
            assert len(user_warnings) >= 1


class TestDocumentation:
    """Tests for documentation generation."""

    def test_documentation_is_markdown(
        self, builder, classification_data, diverse_base_models
    ):
        """Documentation string should contain markdown headers."""
        X_train, y_train, X_val, y_val = classification_data
        result = builder.build(
            base_models=diverse_base_models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy="voting_soft",
        )

        doc = result["documentation"]
        assert "# Ensemble Documentation" in doc
        assert "## Strategy" in doc
        assert "## Base Models" in doc
        assert "## Ensemble Performance" in doc


class TestEvaluateContributions:
    """Tests for evaluate_contributions method."""

    def test_evaluate_contributions_returns_dataframe(
        self, builder, classification_data, diverse_base_models
    ):
        """evaluate_contributions should return DataFrame with correct columns and shape."""
        X_train, y_train, X_val, y_val = classification_data

        result = builder.build(
            base_models=diverse_base_models,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy="voting_soft",
        )

        # Fit the base models for contribution evaluation
        from sklearn.base import clone
        fitted_models = [
            clone(m).fit(X_train, y_train) for m in diverse_base_models
        ]

        contributions_df = builder.evaluate_contributions(
            base_models=fitted_models,
            ensemble_model=result["ensemble_model"],
            X_val=X_val,
            y_val=y_val,
            metric="accuracy",
        )

        assert isinstance(contributions_df, pd.DataFrame)
        assert "model" in contributions_df.columns
        assert "score_with" in contributions_df.columns
        assert "score_without" in contributions_df.columns
        assert "contribution" in contributions_df.columns
        assert len(contributions_df) == len(diverse_base_models)
