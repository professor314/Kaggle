"""Property-based tests for DataCleaner imputation, encoding, and outlier detection.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume

from kaggle_ml_toolkit.cleaner import DataCleaner
from tests.conftest import dataframes_with_missing, dataframes_numeric


# ---------------------------------------------------------------------------
# Custom strategies for cleaner property tests
# ---------------------------------------------------------------------------

column_names = st.text(
    min_size=1,
    max_size=10,
    alphabet=st.characters(whitelist_categories=("Ll",)),
).filter(lambda x: x.isidentifier())

CATEGORY_VALUES = ["alpha", "beta", "gamma", "delta", "epsilon"]


@st.composite
def dataframes_with_categorical(draw, min_cols=2, max_cols=4, min_rows=5, max_rows=30):
    """Generate DataFrames with categorical string columns.

    Ensures at least 2 unique values per column for meaningful encoding.
    """
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cols = draw(st.lists(column_names, min_size=n_cols, max_size=n_cols, unique=True))
    data = {}
    for col in cols:
        # Ensure at least 2 unique categories by drawing from at least 2
        n_categories = draw(st.integers(min_value=2, max_value=min(5, n_rows)))
        categories = CATEGORY_VALUES[:n_categories]
        data[col] = draw(
            st.lists(
                st.sampled_from(categories),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Property 2: Imputation Completeness
# Feature: kaggle-ml-toolkit, Property 2: Imputation Completeness
# ---------------------------------------------------------------------------


@given(df=dataframes_with_missing(), strategy=st.sampled_from(["mean", "median", "constant"]))
@settings(max_examples=100)
def test_imputation_completeness(df, strategy):
    """For any DataFrame with missing values, after applying any supported
    imputation strategy (mean, median, constant), the result should have
    zero missing values in the imputed columns.

    **Validates: Requirements 5.1, 5.2**
    """
    # Skip if any column is entirely null (cleaner raises ValueError)
    for col in df.columns:
        assume(not df[col].isna().all())

    # Need at least one missing value to make the test meaningful
    assume(df.isna().any().any())

    cleaner = DataCleaner()
    columns = df.columns.tolist()

    if strategy == "constant":
        result = cleaner.impute_numeric(df, columns, strategy=strategy, fill_value=0.0)
    else:
        result = cleaner.impute_numeric(df, columns, strategy=strategy)

    # After imputation, specified columns must have zero missing values
    for col in columns:
        assert result[col].isna().sum() == 0, (
            f"Column '{col}' still has missing values after {strategy} imputation"
        )


# ---------------------------------------------------------------------------
# Property 3: Transformation Immutability
# Feature: kaggle-ml-toolkit, Property 3: Transformation Immutability
# ---------------------------------------------------------------------------


@given(
    df=dataframes_with_missing(),
    strategy=st.sampled_from(["mean", "median", "constant"]),
)
@settings(max_examples=100)
def test_imputation_immutability(df, strategy):
    """For any DataFrame and any imputation operation, the original
    DataFrame should be unchanged after the operation.

    **Validates: Requirements 5.3**
    """
    for col in df.columns:
        assume(not df[col].isna().all())

    cleaner = DataCleaner()
    columns = df.columns.tolist()

    # Capture original state (deep copy for comparison)
    original_values = df.copy()

    if strategy == "constant":
        _ = cleaner.impute_numeric(df, columns, strategy=strategy, fill_value=0.0)
    else:
        _ = cleaner.impute_numeric(df, columns, strategy=strategy)

    # Original DataFrame must be unchanged
    pd.testing.assert_frame_equal(df, original_values)


@given(df=dataframes_with_categorical())
@settings(max_examples=100)
def test_encoding_immutability(df):
    """For any DataFrame and a label encoding operation, the original
    DataFrame should be unchanged after the operation.

    **Validates: Requirements 5.3**
    """
    cleaner = DataCleaner()
    columns = df.columns.tolist()

    # Capture original state
    original_values = df.copy()

    _ = cleaner.encode(df, columns, method="label")

    # Original DataFrame must be unchanged
    pd.testing.assert_frame_equal(df, original_values)


# ---------------------------------------------------------------------------
# Property 4: Encoding Round-Trip
# Feature: kaggle-ml-toolkit, Property 4: Encoding Round-Trip
# ---------------------------------------------------------------------------


@given(df=dataframes_with_categorical())
@settings(max_examples=100)
def test_encoding_round_trip_transform_new(df):
    """For any DataFrame with categorical columns, applying label encoding
    and then checking that transform_new on the same data produces identical
    results to the original encoding.

    **Validates: Requirements 5.4, 5.5**
    """
    cleaner = DataCleaner()
    columns = df.columns.tolist()

    # First encode
    encoded = cleaner.encode(df, columns, method="label")

    # Apply transform_new on the same data
    transformed = cleaner.transform_new(df)

    # Both should produce identical results for the encoded columns
    for col in columns:
        pd.testing.assert_series_equal(
            encoded[col].reset_index(drop=True),
            transformed[col].reset_index(drop=True),
            check_names=False,
            obj=f"Column '{col}' mismatch between encode() and transform_new()",
        )


# ---------------------------------------------------------------------------
# Property 5: Outlier Detection Correctness
# Feature: kaggle-ml-toolkit, Property 5: Outlier Detection Correctness
# ---------------------------------------------------------------------------


@given(
    df=dataframes_numeric(min_cols=1, max_cols=4, min_rows=5, max_rows=50),
    k=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_outlier_detection_count_matches_indices(df, k):
    """For any numeric column with an IQR-based outlier detection (multiplier k),
    the reported count should equal the number of indices reported.

    **Validates: Requirements 5.6, 5.7**
    """
    cleaner = DataCleaner()
    columns = df.columns.tolist()

    report = cleaner.detect_outliers(df, columns, method="iqr", threshold=k)

    # For each column, count must equal length of indices
    for col in columns:
        col_report = report["columns"][col]
        assert col_report["count"] == len(col_report["indices"]), (
            f"Column '{col}': count ({col_report['count']}) != "
            f"len(indices) ({len(col_report['indices'])})"
        )

    # Total outliers should be sum of per-column counts
    expected_total = sum(
        report["columns"][col]["count"] for col in columns
    )
    assert report["total_outliers"] == expected_total
