"""Model optimizer module for hyperparameter tuning.

Provides grid search and randomized search over parameter grids,
returning the best parameters, score, and fitted model.
"""

from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV


class ModelOptimizer:
    """Hyperparameter optimization with experiment logging."""

    # Map user-friendly metric names to sklearn scorers
    METRIC_MAPPING: Dict[str, str] = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r_squared": "r2",
        "auc_roc": "roc_auc",
    }

    # Metrics that use neg_* scoring (scores need to be negated for display)
    _NEG_METRICS = {"neg_root_mean_squared_error", "neg_mean_absolute_error"}

    def optimize(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, Any],
        X,
        y,
        method: str = "randomized",
        scoring: str = "accuracy",
        cv_folds: int = 5,
        n_iter: int = 10,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Run hyperparameter search.

        Args:
            model: A scikit-learn compatible estimator.
            param_grid: Dictionary mapping parameter names to lists of values.
            X: Feature matrix.
            y: Target variable.
            method: Search method - "grid" or "randomized" (default).
            scoring: Metric name for evaluation.
            cv_folds: Number of cross-validation folds.
            n_iter: Number of iterations for randomized search.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with keys: best_params, best_score, best_model,
            method, n_iter.

        Raises:
            ValueError: If param_grid is empty or metric is unsupported.
            RuntimeError: If all parameter combinations fail.
        """
        # Validate param_grid is not empty
        if not param_grid or all(
            len(v) == 0 for v in param_grid.values() if isinstance(v, list)
        ):
            raise ValueError("param_grid must not be empty")

        # Resolve sklearn scorer name
        sklearn_scorer = self.METRIC_MAPPING.get(scoring, scoring)

        # Build the search object
        if method == "grid":
            search = GridSearchCV(
                estimator=clone(model),
                param_grid=param_grid,
                scoring=sklearn_scorer,
                cv=cv_folds,
                error_score="raise",
            )
        else:
            search = RandomizedSearchCV(
                estimator=clone(model),
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=sklearn_scorer,
                cv=cv_folds,
                random_state=random_state,
                error_score="raise",
            )

        # Fit the search - if all combos fail, this raises an error
        try:
            search.fit(X, y)
        except Exception as e:
            raise RuntimeError(
                "No valid parameter combination was found"
            ) from e

        # Compute best_score, negating for neg_* metrics
        best_score = search.best_score_
        if sklearn_scorer in self._NEG_METRICS:
            best_score = -best_score

        # Determine n_iter for result
        if method == "grid":
            result_n_iter = len(ParameterGrid(param_grid))
        else:
            result_n_iter = n_iter

        return {
            "best_params": search.best_params_,
            "best_score": best_score,
            "best_model": search.best_estimator_,
            "method": method,
            "n_iter": result_n_iter,
        }
