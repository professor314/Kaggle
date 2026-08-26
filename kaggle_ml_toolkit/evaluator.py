"""Evaluator module for metric computation and experiment persistence.

Provides the Evaluator class for computing classification, regression, and
clustering metrics, with optional persistence to a local JSON file.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


class Evaluator:
    """Metric computation and JSON-based experiment persistence.

    Computes standard metrics for classification, regression, and clustering
    tasks, and optionally persists results to a local JSON file for
    historical comparison.

    Attributes:
        _results_path: Path to the JSON file where experiment results are stored.
    """

    def __init__(self, results_path: str = "./experiments.json") -> None:
        """Initialize the Evaluator.

        Args:
            results_path: Path to the JSON file for persisting experiment
                results. Defaults to "./experiments.json".
        """
        self._results_path = results_path

    def evaluate(
        self,
        y_true,
        y_pred,
        problem_type: str = "classification",
        experiment_id: Optional[str] = None,
        model_name: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, float]:
        """Compute metrics based on problem type and optionally persist.

        Args:
            y_true: True labels/values. For clustering, this should be the
                feature matrix X (used to compute silhouette score).
            y_pred: Predicted labels/values. For clustering, these are
                cluster labels.
            problem_type: One of "classification", "regression", or
                "clustering". Defaults to "classification".
            experiment_id: Optional identifier for the experiment.
            model_name: Optional name of the model being evaluated.
            persist: Whether to persist results to the JSON file.
                Defaults to True.

        Returns:
            Dictionary mapping metric names to their computed values.
            Values may be None if a metric cannot be computed (e.g.,
            AUC-ROC for multiclass without probabilities).
        """
        if problem_type == "classification":
            metrics = self._compute_classification_metrics(y_true, y_pred)
        elif problem_type == "regression":
            metrics = self._compute_regression_metrics(y_true, y_pred)
        elif problem_type == "clustering":
            metrics = self._compute_clustering_metrics(y_true, y_pred)
        else:
            raise ValueError(
                f"Unsupported problem_type: '{problem_type}'. "
                "Must be one of: 'classification', 'regression', 'clustering'."
            )

        if persist:
            self._persist(metrics, experiment_id, model_name)

        return metrics

    def _compute_classification_metrics(
        self, y_true, y_pred
    ) -> Dict[str, Optional[float]]:
        """Compute classification metrics.

        Args:
            y_true: True class labels.
            y_pred: Predicted class labels.

        Returns:
            Dictionary with accuracy, precision, recall, f1, and auc_roc.
        """
        metrics: Dict[str, Optional[float]] = {}

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # AUC-ROC may fail for multiclass without probability estimates
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
        except (ValueError, TypeError):
            metrics["auc_roc"] = None

        return metrics

    def _compute_regression_metrics(
        self, y_true, y_pred
    ) -> Dict[str, float]:
        """Compute regression metrics.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            Dictionary with rmse, mae, and r_squared.
        """
        metrics: Dict[str, float] = {}

        metrics["rmse"] = math.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r_squared"] = r2_score(y_true, y_pred)

        return metrics

    def _compute_clustering_metrics(
        self, X, labels
    ) -> Dict[str, float]:
        """Compute clustering metrics.

        Args:
            X: Feature matrix (samples × features).
            labels: Cluster labels assigned to each sample.

        Returns:
            Dictionary with silhouette score.
        """
        metrics: Dict[str, float] = {}

        metrics["silhouette"] = silhouette_score(X, labels)

        return metrics

    def _persist(
        self,
        metrics: Dict[str, Optional[float]],
        experiment_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Persist evaluation results to the JSON file.

        Creates the file if it doesn't exist, then appends a new experiment
        record with timestamp, identifiers, and metric values.

        Args:
            metrics: Dictionary of computed metric values. None values are
                stored as JSON null.
            experiment_id: Optional experiment identifier. Defaults to
                "unnamed" if not provided.
            model_name: Optional model name. Defaults to "unnamed" if not
                provided.
        """
        path = Path(self._results_path)

        # Load existing data or create initial structure
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"experiments": []}

        # Build the experiment record
        record: Dict[str, Any] = {
            "experiment_id": experiment_id if experiment_id is not None else "unnamed",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name if model_name is not None else "unnamed",
            "metrics": metrics,
        }

        # Append and write back
        data["experiments"].append(record)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False, default=str)

    def load_history(
        self,
        experiment_id: Optional[str] = None,
        model_name: Optional[str] = None,
        date_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Load historical experiment results with optional filtering.

        Reads the JSON persistence file and returns a DataFrame of experiment
        records, optionally filtered by experiment ID, model name, or date
        range.

        Args:
            experiment_id: If provided, return only records with this exact
                experiment_id.
            model_name: If provided, return only records with this exact
                model_name.
            date_range: If provided, a tuple of (start_date_str, end_date_str)
                in ISO 8601 format. Only records whose timestamp falls within
                this range (inclusive) are returned.

        Returns:
            DataFrame with columns: experiment_id, timestamp, model_name,
            and one column per metric found in the records. Returns an empty
            DataFrame with base columns if the file doesn't exist or contains
            no matching records.
        """
        base_columns = ["experiment_id", "timestamp", "model_name"]
        path = Path(self._results_path)

        if not path.exists():
            return pd.DataFrame(columns=base_columns)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        experiments: List[Dict[str, Any]] = data.get("experiments", [])

        if not experiments:
            return pd.DataFrame(columns=base_columns)

        # Apply filters
        filtered = experiments

        if experiment_id is not None:
            filtered = [r for r in filtered if r.get("experiment_id") == experiment_id]

        if model_name is not None:
            filtered = [r for r in filtered if r.get("model_name") == model_name]

        if date_range is not None:
            start_str, end_str = date_range
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
            filtered = [
                r
                for r in filtered
                if start_dt <= datetime.fromisoformat(r["timestamp"]) <= end_dt
            ]

        if not filtered:
            return pd.DataFrame(columns=base_columns)

        # Flatten records: pull metrics into top-level columns
        rows: List[Dict[str, Any]] = []
        for record in filtered:
            row: Dict[str, Any] = {
                "experiment_id": record.get("experiment_id"),
                "timestamp": record.get("timestamp"),
                "model_name": record.get("model_name"),
            }
            metrics_dict = record.get("metrics", {})
            for metric_key, metric_val in metrics_dict.items():
                row[metric_key] = metric_val
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure base columns come first
        all_cols = base_columns + [c for c in df.columns if c not in base_columns]
        df = df.reindex(columns=all_cols)

        return df
