"""Data augmentation with impact evaluation.

Provides augmentation strategies (SMOTE, noise injection, synthetic)
and before/after performance comparison to ensure augmentation helps
rather than hurts model performance.
"""

from typing import Dict, Any, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

try:
    from imblearn.over_sampling import SMOTE

    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False


class Augmenter:
    """Data augmentation with before/after performance comparison.

    Supports multiple augmentation strategies and evaluates whether
    augmentation actually improves model performance on a held-out set.
    """

    def augment(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: str = "smote",
        domain_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply augmentation strategy to the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing features and target.
        target_column : str
            Name of the target column in the DataFrame.
        strategy : str, default="smote"
            Augmentation strategy: "smote", "noise_injection", or "synthetic".
        domain_params : dict, optional
            Domain-specific parameters including:
            - "noise_scale" (float): Scale for Gaussian noise (default 0.01).
            - "valid_ranges" (dict): Column name → [min, max] for clipping.
            - "random_state" (int): Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary with keys: augmented_df, original_size, augmented_size,
            class_distribution_before, class_distribution_after,
            samples_clipped, strategy.

        Raises
        ------
        ImportError
            If strategy is "smote" and imblearn is not installed.
        ValueError
            If strategy is not recognized.
        """
        if domain_params is None:
            domain_params = {}

        random_state = domain_params.get("random_state", 42)
        rng = np.random.default_rng(random_state)

        original_size = len(df)
        class_distribution_before = df[target_column].value_counts().to_dict()

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if strategy == "smote":
            augmented_df = self._augment_smote(X, y, target_column, random_state)
        elif strategy == "noise_injection":
            noise_scale = domain_params.get("noise_scale", 0.01)
            augmented_df = self._augment_noise(
                df, target_column, noise_scale, rng
            )
        elif strategy == "synthetic":
            augmented_df = self._augment_synthetic(
                df, target_column, rng
            )
        else:
            raise ValueError(
                f"Unknown augmentation strategy: '{strategy}'. "
                f"Supported strategies: 'smote', 'noise_injection', 'synthetic'."
            )

        # Apply domain range clipping if valid_ranges provided
        samples_clipped = 0
        valid_ranges = domain_params.get("valid_ranges")
        if valid_ranges:
            samples_clipped = self._clip_to_domain(augmented_df, valid_ranges)

        augmented_size = len(augmented_df)
        class_distribution_after = (
            augmented_df[target_column].value_counts().to_dict()
        )

        return {
            "augmented_df": augmented_df,
            "original_size": original_size,
            "augmented_size": augmented_size,
            "class_distribution_before": class_distribution_before,
            "class_distribution_after": class_distribution_after,
            "samples_clipped": samples_clipped,
            "strategy": strategy,
        }

    def compare_impact(
        self,
        model: BaseEstimator,
        X_original,
        y_original,
        X_augmented,
        y_augmented,
        X_val,
        y_val,
        metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """Compare model performance with and without augmentation.

        Trains the model on both original and augmented data, then
        evaluates on a held-out validation set to determine whether
        augmentation helps or hurts performance.

        Parameters
        ----------
        model : BaseEstimator
            Scikit-learn compatible model to train and evaluate.
        X_original : array-like
            Original training features.
        y_original : array-like
            Original training target.
        X_augmented : array-like
            Augmented training features.
        y_augmented : array-like
            Augmented training target.
        X_val : array-like
            Validation features.
        y_val : array-like
            Validation target.
        metric : str, default="accuracy"
            Scoring metric (e.g., "accuracy", "f1", "roc_auc").

        Returns
        -------
        dict
            Dictionary with keys: original_score, augmented_score,
            improvement, degradation_warning, documentation.
        """
        # Train on original data
        model_original = clone(model)
        model_original.fit(X_original, y_original)
        original_score = self._score_model(model_original, X_val, y_val, metric)

        # Train on augmented data
        model_augmented = clone(model)
        model_augmented.fit(X_augmented, y_augmented)
        augmented_score = self._score_model(model_augmented, X_val, y_val, metric)

        improvement = augmented_score - original_score
        degradation_warning = augmented_score < original_score

        if degradation_warning:
            warnings.warn(
                f"Augmentation degraded performance: "
                f"{augmented_score:.4f} < {original_score:.4f} ({metric}). "
                f"Consider not using augmentation for this dataset.",
                UserWarning,
                stacklevel=2,
            )

        documentation = self._generate_documentation(
            metric=metric,
            original_score=original_score,
            augmented_score=augmented_score,
            improvement=improvement,
            degradation_warning=degradation_warning,
        )

        return {
            "original_score": original_score,
            "augmented_score": augmented_score,
            "improvement": improvement,
            "degradation_warning": degradation_warning,
            "documentation": documentation,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _augment_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_column: str,
        random_state: int,
    ) -> pd.DataFrame:
        """Apply SMOTE oversampling to minority class."""
        if not _IMBLEARN_AVAILABLE:
            raise ImportError(
                "The 'imbalanced-learn' package is required for SMOTE augmentation. "
                "Install it with: pip install imbalanced-learn"
            )

        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        augmented_df = pd.DataFrame(X_resampled, columns=X.columns)
        augmented_df[target_column] = y_resampled
        return augmented_df

    def _augment_noise(
        self,
        df: pd.DataFrame,
        target_column: str,
        noise_scale: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Add Gaussian noise to numeric features."""
        augmented_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Don't add noise to the target column
        numeric_cols = [c for c in numeric_cols if c != target_column]

        # Create noisy copy of the original data
        noisy_rows = df.copy()
        for col in numeric_cols:
            std = df[col].std()
            if std == 0 or np.isnan(std):
                continue
            noise = rng.normal(0, noise_scale * std, size=len(df))
            noisy_rows[col] = noisy_rows[col] + noise

        # Concatenate original with noisy copy
        augmented_df = pd.concat([df, noisy_rows], ignore_index=True)
        return augmented_df

    def _augment_synthetic(
        self,
        df: pd.DataFrame,
        target_column: str,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Duplicate minority class samples with small random perturbation."""
        # Identify minority class
        class_counts = df[target_column].value_counts()
        minority_class = class_counts.idxmin()
        majority_count = class_counts.max()

        minority_df = df[df[target_column] == minority_class].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_column]

        # Determine how many synthetic samples needed to balance
        n_synthetic = majority_count - len(minority_df)
        if n_synthetic <= 0:
            # Already balanced, just return original
            return df.copy()

        # Sample with replacement from minority and add perturbation
        synthetic_indices = rng.choice(len(minority_df), size=n_synthetic, replace=True)
        synthetic_df = minority_df.iloc[synthetic_indices].copy().reset_index(drop=True)

        for col in numeric_cols:
            std = minority_df[col].std()
            if std == 0 or np.isnan(std):
                continue
            perturbation = rng.normal(0, 0.01 * std, size=n_synthetic)
            synthetic_df[col] = synthetic_df[col].values + perturbation

        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
        return augmented_df

    def _clip_to_domain(
        self, df: pd.DataFrame, valid_ranges: Dict[str, list]
    ) -> int:
        """Clip values to valid domain ranges and count affected samples.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to clip in-place.
        valid_ranges : dict
            Column name → [min_val, max_val].

        Returns
        -------
        int
            Number of samples that had at least one value clipped.
        """
        clipped_mask = pd.Series(False, index=df.index)

        for col, (min_val, max_val) in valid_ranges.items():
            if col not in df.columns:
                continue
            # Identify rows that will be clipped
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            clipped_mask = clipped_mask | out_of_range
            # Clip values
            df[col] = df[col].clip(lower=min_val, upper=max_val)

        return int(clipped_mask.sum())

    def _score_model(
        self,
        model: BaseEstimator,
        X_val,
        y_val,
        metric: str,
    ) -> float:
        """Score a fitted model on validation data.

        Uses sklearn's scoring conventions where applicable.
        """
        from sklearn.metrics import get_scorer

        scorer = get_scorer(metric)
        return scorer(model, X_val, y_val)

    def _generate_documentation(
        self,
        metric: str,
        original_score: float,
        augmented_score: float,
        improvement: float,
        degradation_warning: bool,
    ) -> str:
        """Generate markdown documentation of augmentation impact."""
        direction = "improved" if improvement > 0 else "degraded"
        abs_change = abs(improvement)

        doc = f"""## Data Augmentation Impact Report

### Summary

| Metric | Original | Augmented | Change |
|--------|----------|-----------|--------|
| {metric} | {original_score:.4f} | {augmented_score:.4f} | {improvement:+.4f} |

### Analysis

Augmentation **{direction}** model performance by **{abs_change:.4f}** on the `{metric}` metric.

"""
        if degradation_warning:
            doc += """### ⚠️ Degradation Warning

Augmentation **degraded** model performance on the validation set. This suggests that the augmentation strategy may be introducing noise or distributing synthetic samples in ways that do not reflect the true data distribution.

**Recommendation:** Consider not using augmentation for this dataset, or try a different augmentation strategy with more conservative parameters.

"""
        else:
            doc += """### ✅ Positive Impact

Augmentation improved model performance. The synthetic data appears to provide useful additional signal for the model to learn from.

"""
        return doc
