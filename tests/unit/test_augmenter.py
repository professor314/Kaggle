"""Unit tests for Augmenter data augmentation and impact comparison.

Validates: Requirements 33.1, 33.2
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from kaggle_ml_toolkit.augmenter import Augmenter


def _make_classification_df(n=20, imbalance=False):
    """Create a simple binary classification DataFrame."""
    rng = np.random.default_rng(42)
    data = {
        "feat_a": rng.normal(0, 1, size=n),
        "feat_b": rng.normal(5, 2, size=n),
    }
    if imbalance:
        # 80% class 0, 20% class 1
        target = np.zeros(n, dtype=int)
        target[: n // 5] = 1
    else:
        target = np.array([0, 1] * (n // 2), dtype=int)
    data["target"] = target
    return pd.DataFrame(data)


class TestNoiseInjection:
    """Tests for noise_injection augmentation strategy."""

    def test_noise_injection_preserves_shape(self):
        """Augmented DataFrame has 2x original rows (noise creates a duplicate set)."""
        df = _make_classification_df(n=20)
        augmenter = Augmenter()
        result = augmenter.augment(
            df,
            target_column="target",
            strategy="noise_injection",
            domain_params={"noise_scale": 0.01, "random_state": 42},
        )
        augmented_df = result["augmented_df"]
        # Noise injection concatenates original + noisy copy
        assert len(augmented_df) == 2 * len(df)
        # Same number of columns
        assert set(augmented_df.columns) == set(df.columns)


class TestSmoteBalancing:
    """Tests for SMOTE augmentation strategy."""

    def test_smote_balances_classes(self):
        """Minority class count increases after SMOTE (skip if imblearn missing)."""
        try:
            import imblearn  # noqa: F401
        except ImportError:
            pytest.skip("imbalanced-learn not installed")

        # Need enough minority samples for SMOTE's k_neighbors (default=5)
        df = _make_classification_df(n=50, imbalance=True)
        minority_before = (df["target"] == 1).sum()

        augmenter = Augmenter()
        result = augmenter.augment(
            df,
            target_column="target",
            strategy="smote",
            domain_params={"random_state": 42},
        )
        augmented_df = result["augmented_df"]
        minority_after = (augmented_df["target"] == 1).sum()

        assert minority_after > minority_before


class TestDomainClipping:
    """Tests for domain range clipping."""

    def test_domain_clipping_works(self):
        """Values outside valid_ranges are clipped to the range boundaries."""
        df = pd.DataFrame({
            "feat_a": [-100.0, 0.5, 50.0, 200.0],
            "feat_b": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1],
        })
        augmenter = Augmenter()
        result = augmenter.augment(
            df,
            target_column="target",
            strategy="noise_injection",
            domain_params={
                "noise_scale": 0.0,  # No noise, just test clipping
                "random_state": 42,
                "valid_ranges": {"feat_a": [0.0, 100.0]},
            },
        )
        augmented_df = result["augmented_df"]

        # All values in feat_a should now be within [0, 100]
        assert augmented_df["feat_a"].min() >= 0.0
        assert augmented_df["feat_a"].max() <= 100.0

        # Samples were clipped
        assert result["samples_clipped"] > 0


class TestCompareImpact:
    """Tests for Augmenter.compare_impact()."""

    def test_compare_impact_returns_expected_keys(self):
        """compare_impact returns dict with required keys."""
        df = _make_classification_df(n=40)
        augmenter = Augmenter()

        # Split
        train_df = df.iloc[:30]
        val_df = df.iloc[30:]

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]
        X_val = val_df.drop(columns=["target"])
        y_val = val_df["target"]

        # Augment
        aug_result = augmenter.augment(
            train_df,
            target_column="target",
            strategy="noise_injection",
            domain_params={"noise_scale": 0.01, "random_state": 42},
        )
        X_augmented = aug_result["augmented_df"].drop(columns=["target"])
        y_augmented = aug_result["augmented_df"]["target"]

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

        expected_keys = {"original_score", "augmented_score", "improvement",
                         "degradation_warning", "documentation"}
        assert expected_keys == set(result.keys())

        # Types
        assert isinstance(result["original_score"], float)
        assert isinstance(result["augmented_score"], float)
        assert isinstance(result["improvement"], float)
        assert isinstance(result["degradation_warning"], bool)
        assert isinstance(result["documentation"], str)
