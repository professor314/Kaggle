"""Property-based tests for Augmenter size invariants and degradation flagging.

Validates: Requirements 33.1, 33.2
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from kaggle_ml_toolkit.augmenter import Augmenter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def classification_dataframes(draw, min_rows=10, max_rows=40):
    """Generate simple binary classification DataFrames with numeric features."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_features = draw(st.integers(min_value=2, max_value=4))

    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10000)))
    data = {}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(0, 1, size=n_rows)

    # Binary target with at least 3 samples in each class
    target = np.zeros(n_rows, dtype=int)
    # Ensure minority has at least 3 samples
    minority_count = max(3, n_rows // 4)
    target[:minority_count] = 1
    rng.shuffle(target)
    data["target"] = target

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Property 25: Augmentation Size Invariants
# ---------------------------------------------------------------------------


@given(df=classification_dataframes())
@settings(max_examples=10, deadline=None)
def test_augmentation_size_invariants(df):
    """Augmented size >= original size, and reported sizes match actual dimensions.

    For any valid input DataFrame and noise_injection strategy:
    1. augmented_size >= original_size
    2. original_size matches len(df)
    3. augmented_size matches len(augmented_df)

    **Validates: Requirements 33.1**
    """
    augmenter = Augmenter()
    result = augmenter.augment(
        df,
        target_column="target",
        strategy="noise_injection",
        domain_params={"noise_scale": 0.01, "random_state": 42},
    )

    # Augmented size >= original size
    assert result["augmented_size"] >= result["original_size"]

    # Reported original_size matches input
    assert result["original_size"] == len(df)

    # Reported augmented_size matches actual augmented DataFrame
    assert result["augmented_size"] == len(result["augmented_df"])


# ---------------------------------------------------------------------------
# Property 26: Augmentation Degradation Flagging
# ---------------------------------------------------------------------------


@given(df=classification_dataframes(min_rows=20, max_rows=40))
@settings(max_examples=10, deadline=None)
def test_augmentation_degradation_flagging(df):
    """When augmented_score < original_score, degradation_warning is True.

    We construct a scenario and verify the invariant:
    if augmented_score < original_score then degradation_warning must be True,
    and if augmented_score >= original_score then degradation_warning must be False.

    **Validates: Requirements 33.2**
    """
    augmenter = Augmenter()

    # Split data into train and validation
    n_val = max(4, len(df) // 4)
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    if len(train_df) < 4:
        return  # Not enough data for meaningful test

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    # Check we have both classes in train and val
    if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
        return  # Skip if single-class split

    # Augment training data
    aug_result = augmenter.augment(
        train_df,
        target_column="target",
        strategy="noise_injection",
        domain_params={"noise_scale": 0.5, "random_state": 42},
    )
    augmented_df = aug_result["augmented_df"]
    X_augmented = augmented_df.drop(columns=["target"])
    y_augmented = augmented_df["target"]

    model = LogisticRegression(max_iter=200, random_state=42)

    result = augmenter.compare_impact(
        model=model,
        X_original=X_train,
        y_original=y_train,
        X_augmented=X_augmented,
        y_augmented=y_augmented,
        X_val=X_val,
        y_val=y_val,
        metric="accuracy",
    )

    # The key invariant: degradation_warning matches the score comparison
    if result["augmented_score"] < result["original_score"]:
        assert result["degradation_warning"] is True
    else:
        assert result["degradation_warning"] is False
