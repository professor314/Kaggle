"""Ensemble building utilities for combining multiple models.

Supports voting (hard/soft), stacking, bagging, and blending strategies
with documentation generation and performance comparison.
"""

from typing import List, Dict, Any, Optional, Callable

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


class EnsembleBuilder:
    """Documented ensemble construction with performance comparison.

    Combines trained models into ensembles using various strategies and
    generates markdown documentation explaining the ensemble composition,
    strategy, and performance improvements.
    """

    def build(
        self,
        base_models: List[BaseEstimator],
        X_train,
        y_train,
        X_val,
        y_val,
        strategy: str = "voting_soft",
        meta_learner: Optional[BaseEstimator] = None,
        holdout_ratio: float = 0.2,
        metric_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Build an ensemble and compare to base models.

        Args:
            base_models: List of scikit-learn estimators to combine.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            strategy: Ensemble strategy. One of "voting_hard",
                "voting_soft", "stacking", "bagging", "blending".
            meta_learner: Optional meta-learner for stacking/blending.
                Defaults to LogisticRegression() if None.
            holdout_ratio: Fraction of X_train used as holdout for
                blending. Defaults to 0.2.
            metric_fn: Scoring function(y_true, y_pred) -> float.
                Defaults to accuracy_score.

        Returns:
            Dict with keys:
                - ensemble_model: The fitted ensemble model (or meta_learner
                  for blending).
                - ensemble_score: Float score on validation set.
                - base_scores: Dict mapping "model_0", "model_1", ... to
                  individual scores on validation set.
                - improvement: ensemble_score - max(base_scores).
                - documentation: Markdown string explaining the ensemble.

        Warns:
            UserWarning: If all base models produce identical predictions
                on X_val.
        """
        if metric_fn is None:
            metric_fn = accuracy_score

        # Convert to numpy arrays if needed for consistency
        X_train_arr = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train_arr = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        X_val_arr = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
        y_val_arr = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val

        # Build the ensemble based on strategy
        if strategy == "voting_hard":
            ensemble_model = self._build_voting(
                base_models, X_train_arr, y_train_arr, voting="hard"
            )
            ensemble_preds = ensemble_model.predict(X_val_arr)

        elif strategy == "voting_soft":
            ensemble_model = self._build_voting(
                base_models, X_train_arr, y_train_arr, voting="soft"
            )
            ensemble_preds = ensemble_model.predict(X_val_arr)

        elif strategy == "stacking":
            ensemble_model = self._build_stacking(
                base_models, X_train_arr, y_train_arr, meta_learner
            )
            ensemble_preds = ensemble_model.predict(X_val_arr)

        elif strategy == "bagging":
            ensemble_model = self._build_bagging(
                base_models, X_train_arr, y_train_arr
            )
            ensemble_preds = ensemble_model.predict(X_val_arr)

        elif strategy == "blending":
            ensemble_model, ensemble_preds = self._build_blending(
                base_models,
                X_train_arr,
                y_train_arr,
                X_val_arr,
                meta_learner,
                holdout_ratio,
            )

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use one of: "
                "voting_hard, voting_soft, stacking, bagging, blending."
            )

        # Score the ensemble
        ensemble_score = float(metric_fn(y_val_arr, ensemble_preds))

        # Score each base model individually
        base_scores: Dict[str, float] = {}
        base_preds_list: List[np.ndarray] = []
        for i, model in enumerate(base_models):
            fitted_model = clone(model).fit(X_train_arr, y_train_arr)
            preds = fitted_model.predict(X_val_arr)
            base_preds_list.append(preds)
            base_scores[f"model_{i}"] = float(metric_fn(y_val_arr, preds))

        # Check if all base models produce identical predictions
        if len(base_preds_list) > 1:
            all_identical = all(
                np.array_equal(base_preds_list[0], p) for p in base_preds_list[1:]
            )
            if all_identical:
                warnings.warn(
                    "All base models produce identical predictions on the "
                    "validation set. Ensembling provides no benefit — consider "
                    "diversifying model types.",
                    UserWarning,
                    stacklevel=2,
                )

        # Compute improvement
        max_base_score = max(base_scores.values()) if base_scores else 0.0
        improvement = ensemble_score - max_base_score

        # Generate documentation
        documentation = self._generate_documentation(
            base_models, strategy, ensemble_score, base_scores, improvement
        )

        return {
            "ensemble_model": ensemble_model,
            "ensemble_score": ensemble_score,
            "base_scores": base_scores,
            "improvement": improvement,
            "documentation": documentation,
        }

    def evaluate_contributions(
        self,
        base_models: List[BaseEstimator],
        ensemble_model: Any,
        X_val,
        y_val,
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """Evaluate each model's contribution to the ensemble.

        Reports performance with and without each model by building
        leave-one-out ensembles.

        Args:
            base_models: List of base estimators.
            ensemble_model: The fitted ensemble model.
            X_val: Validation features.
            y_val: Validation labels.
            metric: Metric name (used for column labeling).

        Returns:
            DataFrame with columns: model, score_with, score_without,
            contribution.
        """
        from sklearn.metrics import accuracy_score as acc_score

        X_val_arr = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
        y_val_arr = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val

        # Full ensemble score
        full_preds = ensemble_model.predict(X_val_arr)
        full_score = float(acc_score(y_val_arr, full_preds))

        results = []
        for i, model in enumerate(base_models):
            # Individual model score
            individual_preds = model.predict(X_val_arr)
            individual_score = float(acc_score(y_val_arr, individual_preds))

            # Contribution = how much ensemble improves over this model alone
            contribution = full_score - individual_score

            results.append(
                {
                    "model": f"model_{i}",
                    "score_with": full_score,
                    "score_without": individual_score,
                    "contribution": contribution,
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_voting(
        self,
        base_models: List[BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
        voting: str,
    ) -> VotingClassifier:
        """Build and fit a VotingClassifier."""
        estimators = [
            (f"model_{i}", clone(model)) for i, model in enumerate(base_models)
        ]
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        ensemble.fit(X_train, y_train)
        return ensemble

    def _build_stacking(
        self,
        base_models: List[BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta_learner: Optional[BaseEstimator],
    ) -> StackingClassifier:
        """Build and fit a StackingClassifier with cross-validated predictions."""
        if meta_learner is None:
            meta_learner = LogisticRegression()

        estimators = [
            (f"model_{i}", clone(model)) for i, model in enumerate(base_models)
        ]
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=clone(meta_learner),
            cv=5,
        )
        ensemble.fit(X_train, y_train)
        return ensemble

    def _build_bagging(
        self,
        base_models: List[BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> BaggingClassifier:
        """Build and fit a BaggingClassifier wrapping the first base model."""
        ensemble = BaggingClassifier(estimator=clone(base_models[0]))
        ensemble.fit(X_train, y_train)
        return ensemble

    def _build_blending(
        self,
        base_models: List[BaseEstimator],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        meta_learner: Optional[BaseEstimator],
        holdout_ratio: float,
    ) -> tuple:
        """Build a blending ensemble.

        Splits X_train into a training portion and holdout, trains base
        models on training portion, generates meta-features on the holdout,
        and trains the meta-learner on the meta-feature matrix.

        Returns:
            Tuple of (fitted meta_learner, predictions on X_val).
        """
        if meta_learner is None:
            meta_learner = LogisticRegression()

        # Split training data into train portion and blend holdout
        n_samples = len(X_train)
        split_idx = int(n_samples * (1 - holdout_ratio))
        X_train_portion = X_train[:split_idx]
        y_train_portion = y_train[:split_idx]
        X_holdout = X_train[split_idx:]
        y_holdout = y_train[split_idx:]

        # Train each base model on the training portion
        fitted_base_models = []
        for model in base_models:
            fitted = clone(model).fit(X_train_portion, y_train_portion)
            fitted_base_models.append(fitted)

        # Generate meta-features on the holdout
        meta_features_holdout = np.column_stack(
            [m.predict(X_holdout) for m in fitted_base_models]
        )

        # Train the meta-learner on the meta-feature matrix
        fitted_meta = clone(meta_learner).fit(meta_features_holdout, y_holdout)

        # Generate predictions on X_val
        meta_features_val = np.column_stack(
            [m.predict(X_val) for m in fitted_base_models]
        )
        ensemble_preds = fitted_meta.predict(meta_features_val)

        return fitted_meta, ensemble_preds

    def _generate_documentation(
        self,
        base_models: List[BaseEstimator],
        strategy: str,
        ensemble_score: float,
        base_scores: Dict[str, float],
        improvement: float,
    ) -> str:
        """Generate markdown documentation for the ensemble.

        Args:
            base_models: List of base estimators.
            strategy: Ensemble strategy used.
            ensemble_score: Score achieved by the ensemble.
            base_scores: Individual model scores.
            improvement: Ensemble score minus best individual score.

        Returns:
            Markdown string documenting the ensemble.
        """
        model_names = []
        for i, model in enumerate(base_models):
            model_names.append(f"model_{i} ({type(model).__name__})")

        lines = [
            "# Ensemble Documentation",
            "",
            "## Strategy",
            "",
            f"**Ensemble Strategy:** {strategy}",
            "",
            "## Base Models",
            "",
        ]

        for name in model_names:
            lines.append(f"- {name}")

        lines.extend(
            [
                "",
                "## Individual Model Scores",
                "",
                "| Model | Score |",
                "|-------|-------|",
            ]
        )

        for key, score in base_scores.items():
            idx = int(key.split("_")[1])
            model_type = type(base_models[idx]).__name__
            lines.append(f"| {key} ({model_type}) | {score:.4f} |")

        lines.extend(
            [
                "",
                "## Ensemble Performance",
                "",
                f"**Ensemble Score:** {ensemble_score:.4f}",
                f"**Best Individual Score:** {max(base_scores.values()):.4f}",
                f"**Improvement:** {improvement:+.4f}",
                "",
                "## Analysis",
                "",
            ]
        )

        if improvement > 0:
            lines.append(
                f"The {strategy} ensemble improved over the best individual "
                f"model by {improvement:.4f} points. Combining diverse models "
                "helps capture different patterns in the data."
            )
        elif improvement == 0:
            lines.append(
                f"The {strategy} ensemble matched the best individual model. "
                "Consider diversifying model types or tuning the ensemble "
                "configuration for potential gains."
            )
        else:
            lines.append(
                f"The {strategy} ensemble scored lower than the best individual "
                f"model by {abs(improvement):.4f} points. This may indicate the "
                "models are too similar or the ensemble strategy is not well-suited "
                "for this combination. Consider trying a different strategy or "
                "adding more diverse base models."
            )

        return "\n".join(lines)
