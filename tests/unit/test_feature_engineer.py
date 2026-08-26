"""Unit tests for FeatureEngineer module.

Tests:
1. test_create_interaction_correct_values
2. test_create_interaction_raises_for_non_numeric
3. test_bin_continuous_equal_width
4. test_bin_continuous_quantile
5. test_extract_date_components_correct
6. test_extract_date_raises_for_non_datetime
7. test_transform_new_replays_all
8. test_immutability_original_unchanged

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import numpy as np
import pandas as pd
import pytest

from kaggle_ml_toolkit.feature_engineer import FeatureEngineer


@pytest.fixture
def fe():
    return FeatureEngineer()


@pytest.fixture
def numeric_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def date_df():
    return pd.DataFrame({
        "dt": pd.to_datetime([
            "2023-01-15",
            "2023-06-20",
            "2022-12-31",
            "2020-02-29",
            "2021-07-04",
        ])
    })


class TestCreateInteractionCorrectValues:
    """Test interaction feature produces correct element-wise product."""

    def test_create_interaction_correct_values(self, fe, numeric_df):
        result = fe.create_interaction(numeric_df, "a", "b")

        expected = [10.0, 40.0, 90.0, 160.0, 250.0]
        assert list(result["a_x_b"]) == expected

    def test_create_interaction_custom_name(self, fe, numeric_df):
        result = fe.create_interaction(numeric_df, "a", "b", name="product")

        assert "product" in result.columns
        assert list(result["product"]) == [10.0, 40.0, 90.0, 160.0, 250.0]

    def test_create_interaction_preserves_original_columns(self, fe, numeric_df):
        result = fe.create_interaction(numeric_df, "a", "b")

        assert "a" in result.columns
        assert "b" in result.columns
        assert list(result["a"]) == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestCreateInteractionRaisesForNonNumeric:
    """Test TypeError raised for non-numeric columns."""

    def test_raises_for_non_numeric_col_a(self, fe):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
        with pytest.raises(TypeError, match="'a' is not numeric"):
            fe.create_interaction(df, "a", "b")

    def test_raises_for_non_numeric_col_b(self, fe):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with pytest.raises(TypeError, match="'b' is not numeric"):
            fe.create_interaction(df, "a", "b")


class TestBinContinuousEqualWidth:
    """Test equal-width binning."""

    def test_bin_continuous_equal_width(self, fe):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        result = fe.bin_continuous(df, "x", n_bins=5, method="equal_width")

        assert "x_binned" in result.columns
        binned = result["x_binned"]
        # Should have at most 5 unique labels
        assert binned.nunique() <= 5
        # No NaN
        assert binned.notna().all()

    def test_bin_continuous_equal_width_labels_are_integers(self, fe):
        df = pd.DataFrame({"x": list(range(20))})
        result = fe.bin_continuous(df, "x", n_bins=4, method="equal_width")

        # Labels should be integer-like (0 to n_bins-1)
        unique_labels = sorted(result["x_binned"].dropna().unique())
        for label in unique_labels:
            assert 0 <= label < 4


class TestBinContinuousQuantile:
    """Test quantile-based binning."""

    def test_bin_continuous_quantile(self, fe):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        result = fe.bin_continuous(df, "x", n_bins=4, method="quantile")

        assert "x_binned" in result.columns
        binned = result["x_binned"]
        # Should have at most 4 unique labels
        assert binned.nunique() <= 4
        # No NaN
        assert binned.notna().all()

    def test_bin_continuous_quantile_roughly_equal_sizes(self, fe):
        # With 100 evenly distributed values, quantile bins should be ~equal size
        df = pd.DataFrame({"x": list(range(100))})
        result = fe.bin_continuous(df, "x", n_bins=4, method="quantile")

        counts = result["x_binned"].value_counts()
        # Each bin should have roughly 25 items (allow some tolerance)
        for count in counts:
            assert 20 <= count <= 30


class TestExtractDateComponentsCorrect:
    """Test date component extraction with known dates."""

    def test_extract_date_components_correct(self, fe, date_df):
        result = fe.extract_date_components(date_df, "dt")

        # Check 2023-01-15 (Sunday = 6)
        assert result["dt_year"].iloc[0] == 2023
        assert result["dt_month"].iloc[0] == 1
        assert result["dt_day"].iloc[0] == 15
        assert result["dt_dayofweek"].iloc[0] == 6  # Sunday

        # Check 2020-02-29 (leap year, Saturday = 5)
        assert result["dt_year"].iloc[3] == 2020
        assert result["dt_month"].iloc[3] == 2
        assert result["dt_day"].iloc[3] == 29
        assert result["dt_dayofweek"].iloc[3] == 5  # Saturday

    def test_extract_date_components_columns_present(self, fe, date_df):
        result = fe.extract_date_components(date_df, "dt")

        assert "dt_year" in result.columns
        assert "dt_month" in result.columns
        assert "dt_day" in result.columns
        assert "dt_dayofweek" in result.columns

    def test_extract_date_components_are_integers(self, fe, date_df):
        result = fe.extract_date_components(date_df, "dt")

        assert result["dt_year"].dtype == int
        assert result["dt_month"].dtype == int
        assert result["dt_day"].dtype == int
        assert result["dt_dayofweek"].dtype == int


class TestExtractDateRaisesForNonDatetime:
    """Test TypeError raised for non-datetime columns."""

    def test_extract_date_raises_for_non_datetime(self, fe):
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        with pytest.raises(TypeError, match="'col' is not datetime"):
            fe.extract_date_components(df, "col")

    def test_extract_date_raises_for_string_column(self, fe):
        df = pd.DataFrame({"col": ["2023-01-01", "2023-01-02", "2023-01-03"]})
        with pytest.raises(TypeError, match="'col' is not datetime"):
            fe.extract_date_components(df, "col")


class TestTransformNewReplaysAll:
    """Test that transform_new replays all recorded transformations."""

    def test_transform_new_replays_all(self, fe):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "dt": pd.to_datetime([
                "2023-01-01", "2023-02-01", "2023-03-01",
                "2023-04-01", "2023-05-01"
            ]),
        })

        # Apply multiple transformations
        df_after_interaction = fe.create_interaction(df, "a", "b")
        df_after_bin = fe.bin_continuous(df_after_interaction, "a", n_bins=3, method="equal_width")
        df_after_date = fe.extract_date_components(df_after_bin, "dt")

        # Replay on fresh identical data
        replay = fe.transform_new(df.copy())

        # Interaction column should match
        pd.testing.assert_series_equal(
            df_after_date["a_x_b"], replay["a_x_b"], check_names=False
        )
        # Date columns should match
        for col in ["dt_year", "dt_month", "dt_day", "dt_dayofweek"]:
            pd.testing.assert_series_equal(
                df_after_date[col], replay[col], check_names=False
            )
        # Binned column should match
        pd.testing.assert_series_equal(
            df_after_date["a_binned"], replay["a_binned"], check_names=False
        )


class TestImmutabilityOriginalUnchanged:
    """Test that original DataFrame is never modified."""

    def test_immutability_create_interaction(self, fe, numeric_df):
        original = numeric_df.copy()
        fe.create_interaction(numeric_df, "a", "b")
        pd.testing.assert_frame_equal(numeric_df, original)

    def test_immutability_bin_continuous(self, fe):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        original = df.copy()
        fe.bin_continuous(df, "x", n_bins=3)
        pd.testing.assert_frame_equal(df, original)

    def test_immutability_extract_date_components(self, fe, date_df):
        original = date_df.copy()
        fe.extract_date_components(date_df, "dt")
        pd.testing.assert_frame_equal(date_df, original)

    def test_immutability_transform_new(self, fe, numeric_df):
        fe.create_interaction(numeric_df, "a", "b")
        fresh_df = numeric_df.copy()
        original = fresh_df.copy()
        fe.transform_new(fresh_df)
        pd.testing.assert_frame_equal(fresh_df, original)
