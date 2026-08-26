"""Unit tests for the Evaluator module.

Tests metric computation, persistence, history loading, and filtering
for the Evaluator class.

Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
"""

import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from kaggle_ml_toolkit.evaluator import Evaluator


class TestClassificationMetrics:
    """Test classification metric computation."""

    def setup_method(self):
        """Set up known classification data."""
        self.y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])

    def test_classification_metrics_computed(self):
        """Verify accuracy, precision, recall, f1, auc_roc keys are present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                self.y_true, self.y_pred,
                problem_type="classification",
                persist=False,
            )

        expected_keys = {"accuracy", "precision", "recall", "f1", "auc_roc"}
        assert expected_keys == set(metrics.keys()), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    def test_classification_metrics_values_reasonable(self):
        """Verify metric values are reasonable for known data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                self.y_true, self.y_pred,
                problem_type="classification",
                persist=False,
            )

        # With the given data, accuracy = 8/10 = 0.8
        assert metrics["accuracy"] == pytest.approx(0.8)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


class TestRegressionMetrics:
    """Test regression metric computation."""

    def setup_method(self):
        """Set up known regression data."""
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    def test_regression_metrics_computed(self):
        """Verify rmse, mae, r_squared keys are present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                self.y_true, self.y_pred,
                problem_type="regression",
                persist=False,
            )

        expected_keys = {"rmse", "mae", "r_squared"}
        assert expected_keys == set(metrics.keys()), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    def test_regression_metrics_non_negative(self):
        """RMSE and MAE must be non-negative."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                self.y_true, self.y_pred,
                problem_type="regression",
                persist=False,
            )

        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

    def test_regression_perfect_predictions(self):
        """Perfect predictions should yield RMSE=0, MAE=0, R²=1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)
            metrics = evaluator.evaluate(
                self.y_true, self.y_true,  # perfect predictions
                problem_type="regression",
                persist=False,
            )

        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r_squared"] == pytest.approx(1.0)


class TestPersistence:
    """Test JSON persistence behavior."""

    def test_persist_creates_file(self):
        """Verify JSON file is created after evaluation with persist=True."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            assert not os.path.exists(path)

            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="test_exp",
                model_name="test_model",
                persist=True,
            )

            assert os.path.exists(path)

            with open(path, "r") as f:
                data = json.load(f)

            assert "experiments" in data
            assert len(data["experiments"]) == 1

    def test_persist_appends(self):
        """Two evaluations should produce 2 entries in the JSON file."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_1",
                model_name="model_a",
                persist=True,
            )
            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_2",
                model_name="model_b",
                persist=True,
            )

            with open(path, "r") as f:
                data = json.load(f)

            assert len(data["experiments"]) == 2
            assert data["experiments"][0]["experiment_id"] == "exp_1"
            assert data["experiments"][1]["experiment_id"] == "exp_2"

    def test_timestamp_is_iso_format(self):
        """Persisted timestamp should be valid ISO 8601 format."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_ts",
                model_name="model_ts",
                persist=True,
            )

            with open(path, "r") as f:
                data = json.load(f)

            timestamp_str = data["experiments"][0]["timestamp"]
            # Should parse without error
            parsed = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed, datetime)


class TestLoadHistory:
    """Test history loading and filtering."""

    def test_load_history_empty_file(self):
        """Loading from a non-existent file returns an empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "nonexistent.json")
            evaluator = Evaluator(results_path=path)

            history = evaluator.load_history()

        assert isinstance(history, pd.DataFrame)
        assert len(history) == 0
        assert "experiment_id" in history.columns
        assert "timestamp" in history.columns
        assert "model_name" in history.columns

    def test_load_history_filter_by_model_name(self):
        """Filtering by model_name returns only matching records."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_1",
                model_name="random_forest",
                persist=True,
            )
            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_2",
                model_name="logistic_regression",
                persist=True,
            )
            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_3",
                model_name="random_forest",
                persist=True,
            )

            history = evaluator.load_history(model_name="random_forest")

        assert len(history) == 2
        assert all(history["model_name"] == "random_forest")

    def test_load_history_filter_by_experiment_id(self):
        """Filtering by experiment_id returns only matching records."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="titanic_v1",
                model_name="rf",
                persist=True,
            )
            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="titanic_v2",
                model_name="rf",
                persist=True,
            )
            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="titanic_v1",
                model_name="lr",
                persist=True,
            )

            history = evaluator.load_history(experiment_id="titanic_v1")

        assert len(history) == 2
        assert all(history["experiment_id"] == "titanic_v1")

    def test_load_history_no_match_returns_empty(self):
        """Filtering with a non-existent value returns empty DataFrame."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "experiments.json")
            evaluator = Evaluator(results_path=path)

            evaluator.evaluate(
                y_true, y_pred,
                problem_type="classification",
                experiment_id="exp_1",
                model_name="model_a",
                persist=True,
            )

            history = evaluator.load_history(model_name="nonexistent_model")

        assert len(history) == 0
