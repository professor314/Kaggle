"""Shared test fixtures and Hypothesis strategies for Kaggle ML Toolkit.

Provides custom Hypothesis strategies for generating:
- Numeric DataFrames with configurable shape
- DataFrames with missing values
- Valid CompetitionConfig parameters
- Matched predictions and IDs

Also provides common pytest fixtures for test setup.

Validates: Requirements 19.1
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings, strategies as st

# ---------------------------------------------------------------------------
# Hypothesis profile configuration
# ---------------------------------------------------------------------------

settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")

# ---------------------------------------------------------------------------
# Custom Hypothesis Strategies
# ---------------------------------------------------------------------------

# Strategy: Generate column names (valid Python identifiers, no duplicates)
column_names = st.text(
    min_size=1,
    max_size=10,
    alphabet=st.characters(whitelist_categories=("Ll",)),
).filter(lambda x: x.isidentifier())


@st.composite
def dataframes_numeric(draw, min_cols=2, max_cols=8, min_rows=2, max_rows=50):
    """Generate random DataFrames with numeric columns.

    Parameters
    ----------
    min_cols, max_cols : int
        Range for number of columns.
    min_rows, max_rows : int
        Range for number of rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with float64 columns and unique column names.
    """
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cols = draw(st.lists(column_names, min_size=n_cols, max_size=n_cols, unique=True))
    data = {}
    for col in cols:
        data[col] = draw(
            st.lists(
                st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
    return pd.DataFrame(data)


@st.composite
def dataframes_with_missing(draw, min_cols=2, max_cols=5, min_rows=5, max_rows=30):
    """Generate DataFrames with some missing values.

    Randomly introduces NaN values but never makes an entire column null,
    ensuring at least one non-null value per column.

    Parameters
    ----------
    min_cols, max_cols : int
        Range for number of columns.
    min_rows, max_rows : int
        Range for number of rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with float64 columns containing some NaN values.
    """
    df = draw(
        dataframes_numeric(
            min_cols=min_cols, max_cols=max_cols, min_rows=min_rows, max_rows=max_rows
        )
    )
    # Randomly introduce NaN values (but never make entire column null)
    for col in df.columns:
        mask_size = draw(st.integers(min_value=0, max_value=len(df) - 2))
        if mask_size > 0:
            indices = draw(
                st.lists(
                    st.integers(min_value=0, max_value=len(df) - 1),
                    min_size=mask_size,
                    max_size=mask_size,
                    unique=True,
                )
            )
            df.loc[indices, col] = np.nan
    return df


# Strategy: Generate valid CompetitionConfig parameters
VALID_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1", "auc_roc"],
    "regression": ["rmse", "mae", "r_squared"],
    "clustering": ["silhouette_score"],
}


@st.composite
def valid_competition_configs(draw):
    """Generate valid CompetitionConfig parameter dictionaries.

    Ensures problem_type and primary_metric are compatible.

    Returns
    -------
    dict
        Dictionary suitable for constructing a CompetitionConfig.
    """
    problem_type = draw(st.sampled_from(["classification", "regression", "clustering"]))
    metric = draw(st.sampled_from(VALID_METRICS[problem_type]))
    return {
        "competition_name": draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            )
        ),
        "target_column": draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Ll",)),
            )
        ),
        "id_column": draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Ll",)),
            )
        ),
        "problem_type": problem_type,
        "primary_metric": metric,
        "random_seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
    }


@st.composite
def matched_predictions(draw, min_size=5, max_size=100):
    """Generate matched predictions and IDs for submission testing.

    Produces a numpy array of predictions in [0, 1] and a pandas Series
    of sequential integer IDs with the same length.

    Parameters
    ----------
    min_size, max_size : int
        Range for the number of predictions/IDs.

    Returns
    -------
    tuple[np.ndarray, pd.Series]
        (predictions array, id Series)
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    predictions = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    ids = list(range(1, n + 1))
    return np.array(predictions), pd.Series(ids, name="id")


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_classification_df():
    """Simple classification DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2": [
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
            ],
            "category": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
