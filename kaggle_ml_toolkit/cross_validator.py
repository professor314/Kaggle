"""Cross-validation module with stratified, time-series, and group-based strategies.

Provides advanced cross-validation beyond standard k-fold, supporting
imbalanced classes, temporal data, and grouped observations.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)


class CrossValidator:
    """Stratified, time-series, and group-based cross-validation."""

    # Map user-friendly metric names to sklearn scorers (same as ModelSelector)
    METRIC_MAPPING: Dict[str, str] = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "auc_roc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r_squared": "r2",
    }

    # Metrics that use neg_* scoring (lower raw value = better)
    _NEG_METRICS = {"neg_root_mean_squared_error", "neg_mean_absolute_error"}

    def validate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "stratified",
        n_folds: int = 5,
        group_column: Optional[str] = None,
        metric: str = "accuracy",
        time_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run cross-validation with the specified strategy.

        Args:
            model: A scikit-learn compatible estimator.
            X: Feature matrix as a DataFrame.
            y: Target variable as a Series.
            strategy: Cross-validation strategy. One of "stratified",
                "time_series", or "group".
            n_folds: Number of cross-validation folds.
            group_column: Column name in X for group-based splitting.
                Required when strategy is "group".
            metric: Evaluation metric name (e.g., "accuracy", "f1", "rmse").
            time_column: Optional column name containing datetime values.
                When provided and strategy is "time_series", fold_details
                will include actual timestamp ranges per fold.

        Returns:
            Dictionary containing:
                - per_fold_scores: List of scores for each fold.
                - mean: Mean of fold scores.
                - std: Standard deviation of fold scores.
                - min: Minimum fold score.
                - max: Maximum fold score.
                - strategy: Strategy used.
                - n_folds: Number of folds used.
                - metric: Metric name used.
                - fold_details: For time_series, list of dicts with
                    train_size, val_size, train_range, val_range,
                    and optionally train_time_range/val_time_range
                    per fold. None otherwise.

        Raises:
            ValueError: If strategy is "group" and group_column is None,
                or if group_column has fewer than 2 unique groups.
        """
        # Resolve sklearn scorer name
        sklearn_scorer = self.METRIC_MAPPING.get(metric, metric)

        # Build the cross-validation splitter
        groups = None
        fold_details: Optional[List[Dict[str, Any]]] = None

        if strategy == "stratified":
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42
            )
        elif strategy == "time_series":
            cv = TimeSeriesSplit(n_splits=n_folds)
        elif strategy == "group":
            if group_column is None:
                raise ValueError(
                    "group_column must be specified for group-based "
                    "cross-validation"
                )
            groups = X[group_column]
            unique_groups = groups.nunique()
            if unique_groups < 2:
                raise ValueError(
                    "group-based splitting requires at least two distinct groups"
                )
            cv = GroupKFold(n_splits=min(n_folds, unique_groups))
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of "
                f"'stratified', 'time_series', or 'group'."
            )

        # For group strategy, exclude the group column from training features
        if strategy == "group" and group_column is not None:
            X_features = X.drop(columns=[group_column])
        else:
            X_features = X

        # Exclude datetime columns from training features
        # (sklearn models can't handle datetime dtypes)
        datetime_cols_to_drop = []
        for col in X_features.columns:
            if pd.api.types.is_datetime64_any_dtype(X_features[col]):
                datetime_cols_to_drop.append(col)
        if datetime_cols_to_drop:
            X_features = X_features.drop(columns=datetime_cols_to_drop)

        # Perform cross-validation fold by fold
        per_fold_scores: List[float] = []

        if strategy == "time_series":
            fold_details = []

        for train_idx, val_idx in cv.split(X, y, groups=groups):
            X_train, X_val = X_features.iloc[train_idx], X_features.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone the model to avoid state leakage between folds
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Score the model using sklearn scoring
            scorer = get_scorer(sklearn_scorer)
            score = scorer(model_clone, X_val, y_val)

            # For neg_* metrics, negate so higher = better
            if sklearn_scorer in self._NEG_METRICS:
                score = -score

            per_fold_scores.append(float(score))

            # Record fold details for time_series strategy
            if strategy == "time_series":
                fold_info: Dict[str, Any] = {
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "train_range": {"start": int(train_idx[0]), "end": int(train_idx[-1])},
                    "val_range": {"start": int(val_idx[0]), "end": int(val_idx[-1])},
                }

                # Determine the datetime column to use for timestamp reporting
                datetime_col = None
                if time_column is not None and time_column in X.columns:
                    datetime_col = time_column
                else:
                    # Auto-detect first datetime column if no explicit time_column
                    for col in X.columns:
                        if pd.api.types.is_datetime64_any_dtype(X[col]):
                            datetime_col = col
                            break

                if datetime_col is not None:
                    train_dates = X[datetime_col].iloc[train_idx]
                    val_dates = X[datetime_col].iloc[val_idx]
                    fold_info["train_time_range"] = {
                        "start": train_dates.min().isoformat(),
                        "end": train_dates.max().isoformat(),
                    }
                    fold_info["val_time_range"] = {
                        "start": val_dates.min().isoformat(),
                        "end": val_dates.max().isoformat(),
                    }

                fold_details.append(fold_info)

        scores_array = np.array(per_fold_scores)

        return {
            "per_fold_scores": per_fold_scores,
            "mean": float(scores_array.mean()),
            "std": float(scores_array.std()),
            "min": float(scores_array.min()),
            "max": float(scores_array.max()),
            "strategy": strategy,
            "n_folds": len(per_fold_scores),
            "metric": metric,
            "fold_details": fold_details,
        }
