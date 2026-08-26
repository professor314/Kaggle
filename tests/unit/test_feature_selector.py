"""Unit tests for FeatureSelector module.

Tests:
1. test_select_returns_expected_keys — rankings, recommended, explanation_report, flagged
2. test_rankings_sorted_descending — combined score is monotonically non-increasing
3. test_domain_priority_boosts_score — domain feature gets higher combined score
4. test_compare_subsets_returns_dataframe — correct columns: subset_name, mean_score, std_score, n_features

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from kaggle_ml_toolkit.feature_selector import FeatureSelector


@pytest.fixture
def selector():
    return FeatureSelector()


@pytest.fixture
def classification_data():
    """Simple classification dataset with 30 rows and 4 numeric features."""
    np.random.seed(42)
    n = 30
    X = pd.DataFrame({
        "important": np.random.randn(n) * 10,
        "moderate": np.random.randn(n) * 5,
        "weak": np.random.randn(n) * 0.1,
        "noise": np.random.randn(n) * 0.01,
    })
    # Target correlated with 'important' feature
    y = pd.Series((X["important"] > 0).astype(int), name="target")
    return X, y


class TestSelectReturnsExpectedKeys:
    """FeatureSelector.select() returns all required dictionary keys."""

    def test_select_returns_expected_keys(self, selector, classification_data):
        X, y = classification_data
        result = selector.select(X, y, methods=["mutual_info", "correlation"])

        expected_keys = {"rankings", "recommended", "explanation_report", "flagged"}
        assert expected_keys == set(result.keys())

    def test_rankings_is_dataframe(self, selector, classification_data):
        X, y = classification_data
        result = selector.select(X, y, methods=["mutual_info", "correlation"])

        assert isinstance(result["rankings"], pd.DataFrame)

    def test_recommended_is_list(self, selector, classification_data):
        X, y = classification_data
        result = selector.select(X, y, methods=["mutual_info", "correlation"])

        assert isinstance(result["recommended"], list)

    def test_explanation_report_is_string(self, selector, classification_data):
        X, y = classification_data
        result = selector.select(X, y, methods=["mutual_info", "correlation"])

        assert isinstance(result["explanation_report"], str)
        assert len(result["explanation_report"]) > 0


class TestRankingsSortedDescending:
    """Combined score in rankings is monotonically non-increasing."""

    def test_rankings_sorted_descending(self, selector, classification_data):
        X, y = classification_data
        result = selector.select(X, y, methods=["mutual_info", "correlation"])

        combined = result["rankings"]["combined"].values
        for i in range(len(combined) - 1):
            assert combined[i] >= combined[i + 1]


class TestDomainPriorityBoostsScore:
    """Domain priority features get a boosted combined score."""

    def test_domain_priority_boosts_score(self, selector, classification_data):
        X, y = classification_data

        # Run without domain priority
        result_no_priority = selector.select(
            X, y, methods=["mutual_info", "correlation"]
        )
        score_without = result_no_priority["rankings"].loc["noise", "combined"]

        # Run with 'noise' as domain priority — it should get a 1.5x boost
        result_with_priority = selector.select(
            X, y, methods=["mutual_info", "correlation"],
            domain_priorities=["noise"],
        )
        score_with = result_with_priority["rankings"].loc["noise", "combined"]

        assert score_with > score_without


class TestCompareSubsetsReturnsDataframe:
    """compare_subsets returns DataFrame with correct columns."""

    def test_compare_subsets_returns_dataframe(self, selector, classification_data):
        X, y = classification_data

        subsets = {
            "all_features": list(X.columns),
            "top_two": ["important", "moderate"],
            "single": ["important"],
        }

        model = LogisticRegression(max_iter=5000, random_state=42)
        result = selector.compare_subsets(X, y, model, subsets, metric="accuracy")

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"subset_name", "mean_score", "std_score", "n_features"}
        assert expected_cols == set(result.columns)
        assert len(result) == 3

    def test_compare_subsets_n_features_correct(self, selector, classification_data):
        X, y = classification_data

        subsets = {
            "all": list(X.columns),
            "pair": ["important", "moderate"],
        }

        model = LogisticRegression(max_iter=5000, random_state=42)
        result = selector.compare_subsets(X, y, model, subsets, metric="accuracy")

        all_row = result[result["subset_name"] == "all"]
        pair_row = result[result["subset_name"] == "pair"]
        assert all_row["n_features"].iloc[0] == 4
        assert pair_row["n_features"].iloc[0] == 2
