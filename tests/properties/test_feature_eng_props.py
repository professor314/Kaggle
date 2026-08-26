"""Property-based tests for FeatureEngineer module.

Tests properties 6-9 as defined in the spec:
- Property 6: Interaction Feature Correctness
- Property 7: Binning Coverage
- Property 8: Date Component Extraction Consistency
- Property 9: Feature Engineer Transform Replay

Validates: Requirements 6.1, 6.2, 6.3, 6.4
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st, assume

from kaggle_ml_toolkit.feature_engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# Custom strategies
# ---------------------------------------------------------------------------

@st.composite
def numeric_dataframe_with_two_cols(draw):
    """Generate a DataFrame with at least two numeric columns named 'a' and 'b'."""
    n_rows = draw(st.integers(min_value=2, max_value=50))
    col_a = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    col_b = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    return pd.DataFrame({"a": col_a, "b": col_b})


@st.composite
def numeric_column_for_binning(draw):
    """Generate a DataFrame with a numeric column suitable for binning."""
    n_rows = draw(st.integers(min_value=5, max_value=50))
    # Ensure at least some variation in values for meaningful binning
    values = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    # Ensure not all values are the same (binning needs variation)
    assume(len(set(values)) >= 2)
    n_bins = draw(st.integers(min_value=2, max_value=min(10, n_rows)))
    return pd.DataFrame({"x": values}), n_bins


@st.composite
def datetime_dataframe(draw):
    """Generate a DataFrame with a datetime column."""
    n_rows = draw(st.integers(min_value=2, max_value=50))
    # Generate timestamps between 2000-01-01 and 2023-12-31
    timestamps = draw(
        st.lists(
            st.datetimes(
                min_value=pd.Timestamp("2000-01-01").to_pydatetime(),
                max_value=pd.Timestamp("2023-12-31").to_pydatetime(),
            ),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    return pd.DataFrame({"dt": pd.to_datetime(timestamps)})


# ---------------------------------------------------------------------------
# Property 6: Interaction Feature Correctness
# ---------------------------------------------------------------------------


class TestInteractionFeatureCorrectness:
    """Property 6: For any DataFrame with two numeric columns a and b,
    the interaction column equals element-wise a * b.

    Validates: Requirements 6.1
    """

    @settings(max_examples=100)
    @given(df=numeric_dataframe_with_two_cols())
    def test_interaction_equals_elementwise_product(self, df):
        """Interaction column must equal a * b for every row."""
        fe = FeatureEngineer()
        result = fe.create_interaction(df, "a", "b")

        expected = df["a"] * df["b"]
        pd.testing.assert_series_equal(
            result["a_x_b"], expected, check_names=False
        )


# ---------------------------------------------------------------------------
# Property 7: Binning Coverage
# ---------------------------------------------------------------------------


class TestBinningCoverage:
    """Property 7: For any numeric column and n_bins, the binned result has
    at most n_bins unique labels and every row has a bin (no NaN in binned column).

    Validates: Requirements 6.2
    """

    @settings(max_examples=100)
    @given(data=numeric_column_for_binning())
    def test_binning_coverage_equal_width(self, data):
        """Equal-width binning: at most n_bins labels, no NaN."""
        df, n_bins = data
        fe = FeatureEngineer()
        result = fe.bin_continuous(df, "x", n_bins=n_bins, method="equal_width")

        binned = result["x_binned"]
        # At most n_bins unique labels
        unique_labels = binned.dropna().unique()
        assert len(unique_labels) <= n_bins
        # Every row has a bin (no NaN)
        assert binned.notna().all()

    @settings(max_examples=100)
    @given(data=numeric_column_for_binning())
    def test_binning_coverage_quantile(self, data):
        """Quantile binning: at most n_bins labels, no NaN."""
        df, n_bins = data
        fe = FeatureEngineer()
        result = fe.bin_continuous(df, "x", n_bins=n_bins, method="quantile")

        binned = result["x_binned"]
        # At most n_bins unique labels (quantile may produce fewer due to duplicates)
        unique_labels = binned.dropna().unique()
        assert len(unique_labels) <= n_bins
        # Every row has a bin (no NaN)
        assert binned.notna().all()


# ---------------------------------------------------------------------------
# Property 8: Date Component Extraction Consistency
# ---------------------------------------------------------------------------


class TestDateComponentExtractionConsistency:
    """Property 8: For any datetime column, extracted year/month/day/dayofweek
    match the original datetime attributes.

    Validates: Requirements 6.3
    """

    @settings(max_examples=100)
    @given(df=datetime_dataframe())
    def test_date_components_match_original(self, df):
        """Extracted components must match original datetime attributes."""
        fe = FeatureEngineer()
        result = fe.extract_date_components(df, "dt")

        for idx in range(len(df)):
            original_dt = df["dt"].iloc[idx]
            assert result["dt_year"].iloc[idx] == original_dt.year
            assert result["dt_month"].iloc[idx] == original_dt.month
            assert result["dt_day"].iloc[idx] == original_dt.day
            assert result["dt_dayofweek"].iloc[idx] == original_dt.dayofweek


# ---------------------------------------------------------------------------
# Property 9: Feature Engineer Transform Replay
# ---------------------------------------------------------------------------


class TestFeatureEngineerTransformReplay:
    """Property 9: For any sequence of transformations, calling transform_new
    on identical data produces the same results.

    Validates: Requirements 6.4
    """

    @settings(max_examples=100)
    @given(df=numeric_dataframe_with_two_cols())
    def test_transform_replay_interaction(self, df):
        """transform_new on identical data reproduces interaction results."""
        fe = FeatureEngineer()
        result_original = fe.create_interaction(df, "a", "b")

        # Replay on identical data
        replay = fe.transform_new(df.copy())

        pd.testing.assert_series_equal(
            result_original["a_x_b"], replay["a_x_b"], check_names=False
        )

    @settings(max_examples=100)
    @given(data=numeric_column_for_binning())
    def test_transform_replay_binning(self, data):
        """transform_new on identical data reproduces binning results."""
        df, n_bins = data
        fe = FeatureEngineer()
        result_original = fe.bin_continuous(df, "x", n_bins=n_bins, method="equal_width")

        # Replay on identical data
        replay = fe.transform_new(df.copy())

        pd.testing.assert_series_equal(
            result_original["x_binned"], replay["x_binned"], check_names=False
        )

    @settings(max_examples=100)
    @given(df=datetime_dataframe())
    def test_transform_replay_date_components(self, df):
        """transform_new on identical data reproduces date extraction results."""
        fe = FeatureEngineer()
        result_original = fe.extract_date_components(df, "dt")

        # Replay on identical data
        replay = fe.transform_new(df.copy())

        for col in ["dt_year", "dt_month", "dt_day", "dt_dayofweek"]:
            pd.testing.assert_series_equal(
                result_original[col], replay[col], check_names=False
            )
