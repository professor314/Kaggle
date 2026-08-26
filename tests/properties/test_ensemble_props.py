"""Property-based tests for EnsembleBuilder.

**Validates: Requirements 10.3, 10.6**
"""

import warnings

from hypothesis import given, settings, strategies as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from kaggle_ml_toolkit.ensemble_builder import EnsembleBuilder


@st.composite
def base_model_lists(draw):
    """Generate a list of 2-4 base models from a pool of classifiers."""
    n_models = draw(st.integers(min_value=2, max_value=4))
    model_pool = [
        RandomForestClassifier(n_estimators=10, random_state=42),
        LogisticRegression(max_iter=200, random_state=42),
        DecisionTreeClassifier(max_depth=3, random_state=42),
        RandomForestClassifier(n_estimators=5, max_depth=2, random_state=7),
    ]
    # Select n_models from the pool (first n_models)
    return model_pool[:n_models]


@settings(max_examples=10, deadline=None)
@given(models=base_model_lists())
def test_ensemble_score_reporting_completeness(models):
    """Property 18: Ensemble Score Reporting Completeness.

    For any ensemble built from N base models, the result should contain
    ensemble_score, and base_scores should contain exactly N entries.

    **Validates: Requirements 10.3**
    """
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    # Split into train/val
    X_train, X_val = X[:70], X[70:]
    y_train, y_val = y[:70], y[70:]

    builder = EnsembleBuilder()
    result = builder.build(
        base_models=models,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        strategy="voting_soft",
    )

    # ensemble_score must be present and numeric
    assert "ensemble_score" in result
    assert isinstance(result["ensemble_score"], float)

    # base_scores must contain exactly N entries
    n_models = len(models)
    assert "base_scores" in result
    assert len(result["base_scores"]) == n_models, (
        f"Expected {n_models} base_scores entries, got {len(result['base_scores'])}"
    )

    # Each base_score key should follow the naming pattern
    for i in range(n_models):
        assert f"model_{i}" in result["base_scores"]


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1000))
def test_identical_prediction_warning(seed):
    """Property 19: Identical Prediction Warning.

    If all base models produce identical predictions, a warning should
    be emitted.

    **Validates: Requirements 10.6**
    """
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=seed,
    )
    X_train, X_val = X[:70], X[70:]
    y_train, y_val = y[:70], y[70:]

    # Use identical models (same type, same params, same seed) to produce
    # identical predictions
    identical_models = [
        DecisionTreeClassifier(max_depth=2, random_state=42),
        DecisionTreeClassifier(max_depth=2, random_state=42),
    ]

    builder = EnsembleBuilder()

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

        # At least one UserWarning about identical predictions should be emitted
        user_warnings = [
            warning for warning in w
            if issubclass(warning.category, UserWarning)
            and "identical predictions" in str(warning.message)
        ]
        assert len(user_warnings) >= 1, (
            "Expected UserWarning about identical predictions but none was emitted"
        )
