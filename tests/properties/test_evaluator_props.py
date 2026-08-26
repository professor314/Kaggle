"""Property-based tests for the Evaluator module.

Tests metric range invariants, persistence round-trip, append-only semantics,
and history filtering correctness using Hypothesis.

Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from kaggle_ml_toolkit.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Strategies for generating classification predictions
# ---------------------------------------------------------------------------


@st.composite
def binary_classification_data(draw, min_size=10, max_size=100):
    """Generate matched y_true and y_pred for binary classification.

    Both arrays contain only 0s and 1s with at least one of each class
    in y_true to avoid degenerate metrics.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    # Ensure at least one of each class in y_true
    y_true = [0, 1] + draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n - 2, max_size=n - 2)
    )
    y_pred = draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n)
    )
    return np.array(y_true), np.array(y_pred)


@st.composite
def regression_data(draw, min_size=10, max_size=100):
    """Generate matched y_true and y_pred for regression."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    y_true = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    y_pred = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Property 20: Metric Range Invariants
# ---------------------------------------------------------------------------


class TestMetricRangeInvariants:
    """Property 20: Classification metrics in [0,1], regression RMSE/MAE >= 0.

    **Validates: Requirements 11.1, 11.2**
    """

    @settings(max_examples=50)
    @given(data=binary_classification_data())
    def test_classification_metrics_bounded(self, data):
        """Classification metrics (accuracy, precision, recall, f1) must be in [0, 1]."""
        y_true, y_pred = data

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                persist=False,
            )

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in metrics, f"Missing metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0, (
                f"{key}={metrics[key]} not in [0, 1]"
            )

        # auc_roc may be None for edge cases, but if present should be in [0, 1]
        if metrics.get("auc_roc") is not None:
            assert 0.0 <= metrics["auc_roc"] <= 1.0

    @settings(max_examples=50)
    @given(data=regression_data())
    def test_regression_rmse_mae_non_negative(self, data):
        """Regression RMSE and MAE must be >= 0."""
        y_true, y_pred = data

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                y_true, y_pred,
                problem_type="regression",
                persist=False,
            )

        assert metrics["rmse"] >= 0.0, f"RMSE={metrics['rmse']} is negative"
        assert metrics["mae"] >= 0.0, f"MAE={metrics['mae']} is negative"


# ---------------------------------------------------------------------------
# Property 21: Evaluation Persistence Round-Trip
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    """Property 21: Persist then load recovers same experiment_id, model_name, and metrics.

    **Validates: Requirements 11.4, 11.5**
    """

    @settings(max_examples=50, deadline=None)
    @given(
        data=binary_classification_data(),
        experiment_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Nd"))),
        model_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Nd"))),
    )
    def test_persist_then_load_recovers_values(self, data, experiment_id, model_name):
        """Persisted evaluation can be loaded back with same identifiers and metrics."""
        y_true, y_pred = data

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            metrics = evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id=experiment_id,
                model_name=model_name,
                persist=True,
            )

            history = evaluator.load_history()

        assert len(history) >= 1
        row = history.iloc[-1]
        assert row["experiment_id"] == experiment_id
        assert row["model_name"] == model_name

        # Verify metrics match
        for key in ["accuracy", "precision", "recall", "f1"]:
            if metrics[key] is not None:
                assert abs(row[key] - metrics[key]) < 1e-10, (
                    f"Round-trip mismatch for {key}: {row[key]} vs {metrics[key]}"
                )


# ---------------------------------------------------------------------------
# Property 22: Evaluation Append-Only
# ---------------------------------------------------------------------------


class TestAppendOnly:
    """Property 22: N evaluations results in at least N entries in loaded history.

    **Validates: Requirements 11.4, 11.5**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(n=st.integers(min_value=1, max_value=5))
    def test_n_evaluations_produce_n_entries(self, n):
        """Evaluating N times should produce at least N entries in history."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            for i in range(n):
                evaluator.evaluate(
                    y_true, y_pred,
                    problem_type="classification",
                    experiment_id=f"exp_{i}",
                    model_name=f"model_{i}",
                    persist=True,
                )

            history = evaluator.load_history()

        assert len(history) >= n, (
            f"Expected at least {n} entries, got {len(history)}"
        )


# ---------------------------------------------------------------------------
# Property 23: History Filtering
# ---------------------------------------------------------------------------


class TestHistoryFiltering:
    """Property 23: All returned records satisfy the filter criteria.

    **Validates: Requirements 11.6**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        model_names=st.lists(
            st.sampled_from(["rf", "lr", "xgb", "svm"]),
            min_size=3,
            max_size=8,
        ),
        filter_name=st.sampled_from(["rf", "lr", "xgb", "svm"]),
    )
    def test_filter_by_model_name(self, model_names, filter_name):
        """All records returned when filtering by model_name must match."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            for i, name in enumerate(model_names):
                evaluator.evaluate(
                    y_true, y_pred,
                    problem_type="classification",
                    experiment_id=f"exp_{i}",
                    model_name=name,
                    persist=True,
                )

            filtered = evaluator.load_history(model_name=filter_name)

        # All returned records must have the correct model_name
        if len(filtered) > 0:
            assert all(filtered["model_name"] == filter_name), (
                f"Filter returned records with wrong model_name: "
                f"{filtered['model_name'].unique()}"
            )

        # Count should match how many times filter_name appears in model_names
        expected_count = model_names.count(filter_name)
        assert len(filtered) == expected_count, (
            f"Expected {expected_count} entries for '{filter_name}', got {len(filtered)}"
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        experiment_ids=st.lists(
            st.sampled_from(["exp_a", "exp_b", "exp_c"]),
            min_size=3,
            max_size=8,
        ),
        filter_id=st.sampled_from(["exp_a", "exp_b", "exp_c"]),
    )
    def test_filter_by_experiment_id(self, experiment_ids, filter_id):
        """All records returned when filtering by experiment_id must match."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            for i, eid in enumerate(experiment_ids):
                evaluator.evaluate(
                    y_true, y_pred,
                    problem_type="classification",
                    experiment_id=eid,
                    model_name=f"model_{i}",
                    persist=True,
                )

            filtered = evaluator.load_history(experiment_id=filter_id)

        # All returned records must have the correct experiment_id
        if len(filtered) > 0:
            assert all(filtered["experiment_id"] == filter_id), (
                f"Filter returned records with wrong experiment_id: "
                f"{filtered['experiment_id'].unique()}"
            )

        # Count should match
        expected_count = experiment_ids.count(filter_id)
        assert len(filtered) == expected_count, (
            f"Expected {expected_count} entries for '{filter_id}', got {len(filtered)}"
        )
