"""Model selector module for comparing multiple ML model candidates.

Trains and compares multiple model candidates using cross-validation,
returning a ranked summary DataFrame.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class ModelSelector:
    """Multi-model comparison with cross-validated evaluation."""

    CLASSIFICATION_MODELS: Dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "knn": KNeighborsClassifier(),
        "svm": SVC(random_state=42),
    }

    REGRESSION_MODELS: Dict[str, object] = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "ridge": Ridge(random_state=42),
    }

    # Map user-friendly metric names to sklearn scorers
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

    def compare(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
        models: Optional[List[str]] = None,
        metric: Optional[str] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Train and evaluate all candidate models using cross-validation.

        Args:
            X: Feature matrix.
            y: Target variable.
            problem_type: Either "classification" or "regression".
            models: Optional list of model names to evaluate. Defaults to all
                models for the specified problem_type.
            metric: Evaluation metric name. Defaults to "accuracy" for
                classification, "rmse" for regression.
            cv_folds: Number of cross-validation folds (2-20).
            random_state: Random seed for reproducibility.

        Returns:
            DataFrame with columns: model_name, mean_score, std_score,
            status, error_message. Sorted by mean_score (descending for
            metrics where higher is better).

        Raises:
            ValueError: If cv_folds is not between 2 and 20.
        """
        if cv_folds < 2 or cv_folds > 20:
            raise ValueError(
                f"cv_folds must be between 2 and 20, got {cv_folds}"
            )

        # Determine default metric
        if metric is None:
            metric = "accuracy" if problem_type == "classification" else "rmse"

        # Resolve sklearn scorer name
        sklearn_scorer = self.METRIC_MAPPING.get(metric, metric)

        # Determine which models to use
        if problem_type == "classification":
            available_models = self.CLASSIFICATION_MODELS
        else:
            available_models = self.REGRESSION_MODELS

        if models is None:
            model_names = list(available_models.keys())
        else:
            model_names = models

        results = []

        for model_name in model_names:
            if model_name not in available_models:
                results.append(
                    {
                        "model_name": model_name,
                        "mean_score": np.nan,
                        "std_score": np.nan,
                        "status": "failed",
                        "error_message": f"Unknown model: {model_name}",
                    }
                )
                continue

            model = available_models[model_name]

            try:
                # Clone the model to avoid state leakage between runs
                from sklearn.base import clone

                model_clone = clone(model)

                from kaggle_ml_toolkit.compute import n_jobs

                scores = cross_val_score(
                    model_clone,
                    X,
                    y,
                    cv=cv_folds,
                    scoring=sklearn_scorer,
                    n_jobs=n_jobs(),  # run CV folds across all cores
                )

                # For neg_* metrics, negate scores so higher = better
                if sklearn_scorer in self._NEG_METRICS:
                    scores = -scores

                results.append(
                    {
                        "model_name": model_name,
                        "mean_score": scores.mean(),
                        "std_score": scores.std(),
                        "status": "success",
                        "error_message": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "model_name": model_name,
                        "mean_score": np.nan,
                        "std_score": np.nan,
                        "status": "failed",
                        "error_message": str(e),
                    }
                )

        df = pd.DataFrame(results)

        # Sort by mean_score in optimal direction:
        # - Descending for metrics where higher is better (accuracy, f1, etc.)
        # - Ascending for neg_* metrics (after negation: lower RMSE/MAE = better)
        sort_ascending = sklearn_scorer in self._NEG_METRICS
        df = df.sort_values(
            by="mean_score", ascending=sort_ascending, na_position="last"
        ).reset_index(drop=True)

        return df
