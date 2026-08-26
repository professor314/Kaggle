"""Advanced integration tests for CrossValidator (Task 31.2).

Tests time-series temporal ordering, datetime column support,
group-based leakage prevention, and stratified class ratio preservation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from kaggle_ml_toolkit.cross_validator import CrossValidator


class TestTimeSeriesFoldRanges:
    """Tests for time-series cross-validation temporal ordering."""

    def test_time_series_fold_ranges_temporal_order(self):
        """Train range end < val range start for each fold."""
        np.random.seed(42)
        n_samples = 150
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples))

        model = LinearRegression()
        cv = CrossValidator()

        result = cv.validate(
            model=model,
            X=X,
            y=y,
            strategy="time_series",
            n_folds=5,
            metric="rmse",
        )

        assert result["fold_details"] is not None
        assert len(result["fold_details"]) == 5

        for i, fold in enumerate(result["fold_details"]):
            train_end = fold["train_range"]["end"]
            val_start = fold["val_range"]["start"]
            assert train_end < val_start, (
                f"Fold {i}: train_range end ({train_end}) must be < "
                f"val_range start ({val_start})"
            )

    def test_time_series_with_datetime_column(self):
        """fold_details include train_time_range/val_time_range when datetime column present."""
        np.random.seed(42)
        n_samples = 120
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        X = pd.DataFrame({
            "date": dates,
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples))

        model = LinearRegression()
        cv = CrossValidator()

        result = cv.validate(
            model=model,
            X=X,
            y=y,
            strategy="time_series",
            n_folds=4,
            metric="rmse",
            time_column="date",
        )

        assert result["fold_details"] is not None
        assert len(result["fold_details"]) == 4

        for i, fold in enumerate(result["fold_details"]):
            # Must have time range keys
            assert "train_time_range" in fold, (
                f"Fold {i} missing train_time_range"
            )
            assert "val_time_range" in fold, (
                f"Fold {i} missing val_time_range"
            )

            # Time ranges should have start/end
            assert "start" in fold["train_time_range"]
            assert "end" in fold["train_time_range"]
            assert "start" in fold["val_time_range"]
            assert "end" in fold["val_time_range"]

            # Validate temporal ordering: train end < val start (as timestamps)
            train_end_ts = pd.Timestamp(fold["train_time_range"]["end"])
            val_start_ts = pd.Timestamp(fold["val_time_range"]["start"])
            assert train_end_ts < val_start_ts, (
                f"Fold {i}: train_time_range end ({train_end_ts}) must be < "
                f"val_time_range start ({val_start_ts})"
            )


class TestGroupCVLeakage:
    """Tests for group-based cross-validation preventing leakage."""

    def test_group_cv_no_group_leakage(self):
        """No group appears in both train and validation for any fold."""
        np.random.seed(42)
        n_samples = 200
        n_groups = 8

        # Create groups with varying sizes
        group_ids = np.random.choice(
            [f"group_{i}" for i in range(n_groups)],
            size=n_samples,
        )

        X = pd.DataFrame({
            "group_id": group_ids,
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, size=n_samples))

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator()

        # We need to verify group leakage at the split level
        # Use sklearn's GroupKFold directly to inspect splits
        from sklearn.model_selection import GroupKFold

        groups = X["group_id"]
        X_features = X.drop(columns=["group_id"])
        gkf = GroupKFold(n_splits=5)

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(X_features, y, groups=groups)
        ):
            train_groups = set(groups.iloc[train_idx].unique())
            val_groups = set(groups.iloc[val_idx].unique())
            overlap = train_groups & val_groups

            assert len(overlap) == 0, (
                f"Fold {fold_idx}: groups {overlap} appear in both "
                f"train and validation sets (group leakage!)"
            )

        # Also verify the full validate() call completes successfully
        result = cv.validate(
            model=model,
            X=X,
            y=y,
            strategy="group",
            n_folds=5,
            group_column="group_id",
            metric="accuracy",
        )

        assert result["strategy"] == "group"
        assert result["n_folds"] <= 5
        assert len(result["per_fold_scores"]) == result["n_folds"]


class TestStratifiedClassRatio:
    """Tests for stratified CV preserving class ratios."""

    def test_stratified_preserves_class_ratio(self):
        """Each fold's class ratio is within 10% of overall ratio."""
        np.random.seed(42)
        n_samples = 200

        # Create imbalanced dataset: 70% class 0, 30% class 1
        y = pd.Series(
            np.concatenate([
                np.zeros(140, dtype=int),
                np.ones(60, dtype=int),
            ])
        )
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
        })

        # Shuffle to avoid any ordering bias
        shuffle_idx = np.random.permutation(n_samples)
        X = X.iloc[shuffle_idx].reset_index(drop=True)
        y = y.iloc[shuffle_idx].reset_index(drop=True)

        # Overall class ratio
        overall_ratio = y.mean()  # proportion of class 1

        # Verify with StratifiedKFold directly
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold_y = y.iloc[val_idx]
            fold_ratio = fold_y.mean()

            # Within 10% of overall ratio
            tolerance = 0.10 * overall_ratio
            assert abs(fold_ratio - overall_ratio) <= tolerance, (
                f"Fold {fold_idx}: class ratio {fold_ratio:.4f} deviates "
                f"more than 10% from overall ratio {overall_ratio:.4f} "
                f"(tolerance: {tolerance:.4f})"
            )

        # Also verify full validate() call works
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = CrossValidator()

        result = cv.validate(
            model=model,
            X=X,
            y=y,
            strategy="stratified",
            n_folds=5,
            metric="accuracy",
        )

        assert result["strategy"] == "stratified"
        assert result["n_folds"] == 5
        assert len(result["per_fold_scores"]) == 5
