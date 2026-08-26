# Feature: kaggle-ml-toolkit, Property 1: Data Loading Round-Trip
"""Property-based tests for the data loader module.

Validates: Requirements 4.1, 4.3, 4.6
"""

import os
import tempfile

import pandas as pd
from hypothesis import given, settings

from kaggle_ml_toolkit.loader import load_csv

# Import the shared strategy from conftest
from tests.conftest import dataframes_numeric


@settings(max_examples=100)
@given(df=dataframes_numeric())
def test_data_loading_round_trip(df: pd.DataFrame):
    """Property 1: Data Loading Round-Trip.

    For any valid pandas DataFrame (generated via Hypothesis), writing it to CSV
    and then loading via `load_csv` should produce a DataFrame with identical
    column names, identical row count, and compatible data types.

    Validates: Requirements 4.1, 4.3, 4.6
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_data.csv")

        # Write generated DataFrame to CSV
        df.to_csv(csv_path, index=False)

        # Load it back with load_csv
        loaded_df = load_csv(csv_path)

        # Assert: same column names
        assert list(loaded_df.columns) == list(df.columns), (
            f"Column names differ: {list(loaded_df.columns)} != {list(df.columns)}"
        )

        # Assert: same row count
        assert len(loaded_df) == len(df), (
            f"Row count differs: {len(loaded_df)} != {len(df)}"
        )

        # Assert: values are close (float precision may differ after CSV round-trip)
        pd.testing.assert_frame_equal(loaded_df, df, check_exact=False)
