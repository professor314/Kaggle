"""Unit tests for CrossValidator time-series fold reporting enhancement."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from kaggle_ml_toolkit.cross_validator import CrossValidator


class TestTimeSeriesFoldReporting:
    """Tests for time-series fold reporting with train_range/val_range."""

    def setup_method(self):
        """Create a simple temporal regression dataset."""
        np.random.seed(42)
        n = 100
        self.X = pd.DataFrame({
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n),
        })
        self.y = pd.Series(np.random.randn(n), name="target")
        self.model = LinearRegression()
        self.cv = CrossValidator()

    def test_fold_details_has_train_and_val_range(self):
        """fold_details should include train_range and val_range dicts."""
        result = self.cv.validate(
            self.model, self.X, self.y,
            strategy="time_series", n_folds=3, metric="rmse"
        )

        assert result["fold_details"] is not None
        assert len(result["fold_details"]) == 3

        for fold in result["fold_details"]:
            assert "train_size" in fold
            assert "val_size" in fold
            assert "train_range" in fold
            assert "val_range" in fold
            assert "start" in fold["train_range"]
            assert "end" in fold["train_range"]
            assert "start" in fold["val_range"]
            assert "end" in fold["val_range"]

    def test_fold_ranges_are_integers(self):
        """train_range and val_range start/end should be integer indices."""
        result = self.cv.validate(
            self.model, self.X, self.y,
            strategy="time_series", n_folds=3, metric="rmse"
        )

        for fold in result["fold_details"]:
            assert isinstance(fold["train_range"]["start"], int)
            assert isinstance(fold["train_range"]["end"], int)
            assert isinstance(fold["val_range"]["start"], int)
            assert isinstance(fold["val_range"]["end"], int)

    def test_fold_ranges_temporal_ordering(self):
        """Train range end should be < val range start (temporal ordering)."""
        result = self.cv.validate(
            self.model, self.X, self.y,
            strategy="time_series", n_folds=3, metric="rmse"
        )

        for fold in result["fold_details"]:
            assert fold["train_range"]["end"] < fold["val_range"]["start"]

    def test_no_time_range_without_datetime_column(self):
        """Without datetime column, fold_details should NOT have time_time_range."""
        result = self.cv.validate(
            self.model, self.X, self.y,
            strategy="time_series", n_folds=3, metric="rmse"
        )

        for fold in result["fold_details"]:
            assert "train_time_range" not in fold
            assert "val_time_range" not in fold

    def test_time_column_parameter_adds_time_range(self):
        """Providing time_column should add train_time_range/val_time_range."""
        X_with_dates = self.X.copy()
        X_with_dates["date"] = pd.date_range("2020-01-01", periods=100, freq="D")

        result = self.cv.validate(
            self.model, X_with_dates, self.y,
            strategy="time_series", n_folds=3, metric="rmse",
            time_column="date"
        )

        for fold in result["fold_details"]:
            assert "train_time_range" in fold
            assert "val_time_range" in fold
            assert "start" in fold["train_time_range"]
            assert "end" in fold["train_time_range"]
            assert "start" in fold["val_time_range"]
            assert "end" in fold["val_time_range"]

    def test_auto_detect_datetime_column(self):
        """If X has a datetime column, it should be auto-detected."""
        X_with_dates = self.X.copy()
        X_with_dates["timestamp"] = pd.date_range("2020-01-01", periods=100, freq="D")

        result = self.cv.validate(
            self.model, X_with_dates, self.y,
            strategy="time_series", n_folds=3, metric="rmse"
        )

        for fold in result["fold_details"]:
            assert "train_time_range" in fold
            assert "val_time_range" in fold

    def test_time_range_values_are_iso_strings(self):
        """Time range values should be ISO-format date strings."""
        X_with_dates = self.X.copy()
        X_with_dates["date"] = pd.date_range("2020-01-01", periods=100, freq="D")

        result = self.cv.validate(
            self.model, X_with_dates, self.y,
            strategy="time_series", n_folds=3, metric="rmse",
            time_column="date"
        )

        for fold in result["fold_details"]:
            # ISO format strings should be parseable
            pd.Timestamp(fold["train_time_range"]["start"])
            pd.Timestamp(fold["train_time_range"]["end"])
            pd.Timestamp(fold["val_time_range"]["start"])
            pd.Timestamp(fold["val_time_range"]["end"])

    def test_stratified_has_no_fold_details(self):
        """Stratified strategy should still return fold_details=None."""
        # Need classification target for stratified
        y_cls = pd.Series(np.random.choice([0, 1], size=100))
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)

        result = self.cv.validate(
            model, self.X, y_cls,
            strategy="stratified", n_folds=3, metric="accuracy"
        )

        assert result["fold_details"] is None

    def test_time_column_ignored_for_non_time_series(self):
        """time_column param should be ignored for non-time_series strategies."""
        y_cls = pd.Series(np.random.choice([0, 1], size=100))
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200)

        X_with_dates = self.X.copy()
        X_with_dates["date"] = pd.date_range("2020-01-01", periods=100, freq="D")

        result = self.cv.validate(
            model, X_with_dates, y_cls,
            strategy="stratified", n_folds=3, metric="accuracy",
            time_column="date"
        )

        assert result["fold_details"] is None
