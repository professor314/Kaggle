"""Unit tests for DataCleaner.detect_outliers method."""

import numpy as np
import pandas as pd
import pytest

from kaggle_ml_toolkit.cleaner import DataCleaner


@pytest.fixture
def cleaner():
    return DataCleaner()


@pytest.fixture
def df_with_outliers():
    """DataFrame with known outliers for IQR-based detection."""
    # Values: 1-10 are normal, 100 is a clear outlier
    return pd.DataFrame({
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
        "normal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    })


class TestDetectOutliersIQR:
    def test_detects_known_outlier(self, cleaner, df_with_outliers):
        result = cleaner.detect_outliers(df_with_outliers, ["values"], method="iqr")
        assert result["columns"]["values"]["count"] >= 1
        assert 10 in result["columns"]["values"]["indices"]  # index of 100

    def test_report_structure(self, cleaner, df_with_outliers):
        result = cleaner.detect_outliers(df_with_outliers, ["values"], method="iqr")
        assert "columns" in result
        assert "total_outliers" in result
        assert "method" in result
        assert "threshold" in result
        assert result["method"] == "iqr"
        assert result["threshold"] == 1.5

    def test_column_report_structure(self, cleaner, df_with_outliers):
        result = cleaner.detect_outliers(df_with_outliers, ["values"], method="iqr")
        col_report = result["columns"]["values"]
        assert "count" in col_report
        assert "indices" in col_report
        assert col_report["count"] == len(col_report["indices"])

    def test_total_outliers_is_sum(self, cleaner, df_with_outliers):
        result = cleaner.detect_outliers(
            df_with_outliers, ["values", "normal"], method="iqr"
        )
        total = sum(
            col["count"] for col in result["columns"].values()
        )
        assert result["total_outliers"] == total

    def test_custom_threshold(self, cleaner):
        # With a very large threshold, nothing should be an outlier
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        result = cleaner.detect_outliers(df, ["x"], method="iqr", threshold=100.0)
        assert result["columns"]["x"]["count"] == 0

    def test_no_outliers_in_uniform_data(self, cleaner):
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        result = cleaner.detect_outliers(df, ["x"], method="iqr")
        assert result["columns"]["x"]["count"] == 0


class TestDetectOutliersZscore:
    def test_detects_known_outlier(self, cleaner):
        # Use many normal values so one extreme value has a clear z-score > 2
        data = list(range(1, 51)) + [500]  # 1-50 plus extreme 500
        df = pd.DataFrame({"x": data})
        result = cleaner.detect_outliers(df, ["x"], method="zscore", threshold=2.5)
        assert result["columns"]["x"]["count"] >= 1
        assert 50 in result["columns"]["x"]["indices"]  # index of 500

    def test_report_method_field(self, cleaner):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = cleaner.detect_outliers(df, ["x"], method="zscore", threshold=3.0)
        assert result["method"] == "zscore"
        assert result["threshold"] == 3.0

    def test_zero_std_no_outliers(self, cleaner):
        # All same values: std=0, no outliers possible
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        result = cleaner.detect_outliers(df, ["x"], method="zscore")
        assert result["columns"]["x"]["count"] == 0


class TestDetectOutliersGeneral:
    def test_does_not_modify_dataframe(self, cleaner, df_with_outliers):
        original = df_with_outliers.copy()
        cleaner.detect_outliers(df_with_outliers, ["values"], method="iqr")
        pd.testing.assert_frame_equal(df_with_outliers, original)

    def test_handles_nan_values(self, cleaner):
        df = pd.DataFrame({"x": [1, 2, 3, np.nan, 5, 100]})
        result = cleaner.detect_outliers(df, ["x"], method="iqr")
        # NaN should not be counted as an outlier
        assert 3 not in result["columns"]["x"]["indices"]

    def test_invalid_method_raises(self, cleaner):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown method"):
            cleaner.detect_outliers(df, ["x"], method="invalid")

    def test_multiple_columns(self, cleaner):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 100],
            "b": [10, 20, 30, 40, 50, 500],
        })
        result = cleaner.detect_outliers(df, ["a", "b"], method="iqr")
        assert "a" in result["columns"]
        assert "b" in result["columns"]
