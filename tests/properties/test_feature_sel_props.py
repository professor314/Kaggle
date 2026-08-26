"""Property-based tests for FeatureSelector module.

Property 10: Feature Ranking Sort Order — rankings DataFrame sorted by
combined score descending.

Validates: Requirements 7.3
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st, assume

from kaggle_ml_toolkit.feature_selector import FeatureSelector


# ---------------------------------------------------------------------------
# Custom strategies
# ---------------------------------------------------------------------------

@st.composite
def classification_dataset(draw):
    """Generate a numeric DataFrame (X) and binary target (y) suitable for feature selection.

    Ensures at least 3 features and 20 rows so all selection methods can operate.
    """
    n_rows = draw(st.integers(min_value=20, max_value=50))
    n_cols = draw(st.integers(min_value=3, max_value=6))

    data = {}
    col_names = [f"f{i}" for i in range(n_cols)]
    for col in col_names:
        values = draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col] = values

    X = pd.DataFrame(data)

    # Ensure columns have some variance (needed for L1/RFE)
    for col in X.columns:
        assume(X[col].std() > 0.01)

    # Binary classification target
    y_values = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    y = pd.Series(y_values, name="target")
    # Ensure both classes present
    assume(y.nunique() == 2)

    return X, y


# ---------------------------------------------------------------------------
# Property 10: Feature Ranking Sort Order
# ---------------------------------------------------------------------------


class TestFeatureRankingSortOrder:
    """Property 10: For any feature selection result, the rankings DataFrame
    should be sorted by combined score in descending order.

    **Validates: Requirements 7.3**
    """

    @settings(max_examples=20, deadline=None)
    @given(data=classification_dataset())
    def test_rankings_sorted_by_combined_descending(self, data):
        """Rankings DataFrame combined column is monotonically non-increasing."""
        X, y = data
        selector = FeatureSelector()

        result = selector.select(X, y, methods=["mutual_info", "correlation"])

        rankings = result["rankings"]
        combined_scores = rankings["combined"].values

        # Verify descending sort: each score >= next score
        for i in range(len(combined_scores) - 1):
            assert combined_scores[i] >= combined_scores[i + 1], (
                f"Rankings not sorted descending at index {i}: "
                f"{combined_scores[i]} < {combined_scores[i + 1]}"
            )
