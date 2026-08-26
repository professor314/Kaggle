"""Domain-informed feature selection combining statistical methods with research knowledge."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score


def _is_classification(y: pd.Series) -> bool:
    """Determine whether the target represents a classification problem."""
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_object_dtype(y):
        return y.nunique() <= 20
    return False


class FeatureSelector:
    """Multi-method feature selection with domain priority support.

    Combines statistical selection methods (mutual information, recursive
    feature elimination, L1-regularization, correlation) with domain
    knowledge to produce ranked feature recommendations and explanations.
    """

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        domain_priorities: Optional[List[str]] = None,
        n_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run selection methods and produce a ranked feature list.

        Args:
            X: Feature DataFrame (numeric columns only recommended).
            y: Target series.
            methods: List of selection methods to use. Supported values:
                "mutual_info", "rfe", "l1", "correlation".
            domain_priorities: Optional list of feature names identified as
                domain-relevant from the Research_Document.
            n_features: Optional maximum number of features to recommend.

        Returns:
            Dict with keys:
                - rankings: DataFrame of features × method scores, sorted by
                    combined score descending.
                - recommended: List of recommended feature names.
                - explanation_report: Markdown string explaining selections.
                - flagged: Domain priority features scoring below median on
                    statistical methods.
        """
        if methods is None:
            methods = ["mutual_info", "rfe", "l1", "correlation"]

        features = list(X.columns)
        is_clf = _is_classification(y)
        scores: Dict[str, np.ndarray] = {}

        if "mutual_info" in methods:
            scores["mutual_info"] = self._score_mutual_info(X, y, is_clf)

        if "rfe" in methods:
            scores["rfe"] = self._score_rfe(X, y, is_clf)

        if "l1" in methods:
            scores["l1"] = self._score_l1(X, y, is_clf)

        if "correlation" in methods:
            scores["correlation"] = self._score_correlation(X, y)

        # Build rankings DataFrame with normalized 0-1 scores per method
        rankings = pd.DataFrame(index=features)
        for method_name, raw_scores in scores.items():
            arr = np.array(raw_scores, dtype=float)
            score_range = arr.max() - arr.min()
            if score_range > 0:
                normalized = (arr - arr.min()) / score_range
            else:
                normalized = np.ones_like(arr) * 0.5
            rankings[method_name] = normalized

        # Compute combined score: average of method scores
        rankings["combined"] = rankings[list(scores.keys())].mean(axis=1)

        # Apply domain priority weighting (multiply combined by 1.5)
        if domain_priorities:
            priority_mask = rankings.index.isin(domain_priorities)
            rankings.loc[priority_mask, "combined"] *= 1.5

        # Sort by combined score descending
        rankings = rankings.sort_values("combined", ascending=False)

        # Determine recommended features
        if n_features is not None:
            recommended = list(rankings.index[:n_features])
        else:
            recommended = list(rankings.index)

        # Flag domain priority features scoring below median on statistical methods
        flagged: List[str] = []
        if domain_priorities:
            stat_methods = [m for m in scores.keys()]
            if stat_methods:
                stat_scores = rankings[stat_methods]
                median_scores = stat_scores.median()
                for feat in domain_priorities:
                    if feat in rankings.index:
                        feat_scores = stat_scores.loc[feat]
                        # Feature is flagged if below median on majority of methods
                        below_median_count = sum(
                            feat_scores[m] < median_scores[m] for m in stat_methods
                        )
                        if below_median_count > len(stat_methods) / 2:
                            flagged.append(feat)

        # Generate explanation report
        explanation_report = self._generate_explanation(
            rankings, recommended, flagged, domain_priorities, n_features
        )

        return {
            "rankings": rankings,
            "recommended": recommended,
            "explanation_report": explanation_report,
            "flagged": flagged,
        }

    def compare_subsets(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        subsets: Dict[str, List[str]],
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """Compare model performance across feature subsets.

        Args:
            X: Full feature DataFrame.
            y: Target series.
            model: Scikit-learn estimator to evaluate.
            subsets: Dict mapping subset names to lists of feature names.
            metric: Scoring metric for cross_val_score.

        Returns:
            DataFrame with columns: subset_name, mean_score, std_score,
            n_features — sorted by mean_score descending.
        """
        results: List[Dict[str, Any]] = []

        for subset_name, feature_list in subsets.items():
            estimator = clone(model)
            cv_scores = cross_val_score(
                estimator, X[feature_list], y, cv=5, scoring=metric
            )
            results.append(
                {
                    "subset_name": subset_name,
                    "mean_score": cv_scores.mean(),
                    "std_score": cv_scores.std(),
                    "n_features": len(feature_list),
                }
            )

        df = pd.DataFrame(results)
        df = df.sort_values("mean_score", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _score_mutual_info(
        self, X: pd.DataFrame, y: pd.Series, is_clf: bool
    ) -> np.ndarray:
        """Compute mutual information scores for each feature."""
        if is_clf:
            mi = mutual_info_classif(X, y, random_state=42)
        else:
            mi = mutual_info_regression(X, y, random_state=42)
        return mi

    def _score_rfe(
        self, X: pd.DataFrame, y: pd.Series, is_clf: bool
    ) -> np.ndarray:
        """Use RFE to rank features; return inverse ranking as scores."""
        estimator = self._default_estimator(is_clf)
        n_features = X.shape[1]
        # Select half the features for ranking purposes
        n_select = max(1, n_features // 2)
        rfe = RFE(estimator, n_features_to_select=n_select, step=1)
        rfe.fit(X, y)
        # Convert ranking (1=best) to score (higher=better)
        # ranking_ gives 1 for selected, >1 for eliminated
        max_rank = rfe.ranking_.max()
        scores = (max_rank - rfe.ranking_) / max(1, max_rank - 1)
        return scores.astype(float)

    def _score_l1(
        self, X: pd.DataFrame, y: pd.Series, is_clf: bool
    ) -> np.ndarray:
        """Fit L1-regularized model; use absolute coefficients as scores."""
        if is_clf:
            model = LogisticRegression(
                solver="saga", max_iter=5000, random_state=42,
                l1_ratio=1.0,
            )
        else:
            model = Lasso(alpha=0.01, max_iter=5000, random_state=42)

        model.fit(X, y)

        if hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                # Multi-class: average across classes
                coefs = coefs.mean(axis=0)
        else:
            coefs = np.zeros(X.shape[1])

        return coefs

    def _score_correlation(
        self, X: pd.DataFrame, y: pd.Series
    ) -> np.ndarray:
        """Compute absolute correlation of each feature with the target."""
        correlations = X.corrwith(y).abs().fillna(0.0)
        return correlations.values

    def _default_estimator(self, is_clf: bool) -> BaseEstimator:
        """Return a default estimator for RFE based on problem type."""
        if is_clf:
            return LogisticRegression(
                max_iter=5000, random_state=42, solver="saga", l1_ratio=1.0
            )
        else:
            return Lasso(alpha=0.01, max_iter=5000, random_state=42)

    # ------------------------------------------------------------------
    # Explanation report generation
    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        rankings: pd.DataFrame,
        recommended: List[str],
        flagged: List[str],
        domain_priorities: Optional[List[str]],
        n_features: Optional[int],
    ) -> str:
        """Generate a markdown explanation report for feature selection."""
        lines: List[str] = []
        lines.append("# Feature Selection Report\n")
        lines.append("## Summary\n")

        total = len(rankings)
        kept = len(recommended)
        removed = total - kept

        lines.append(f"- **Total features evaluated:** {total}")
        lines.append(f"- **Features recommended:** {kept}")
        lines.append(f"- **Features removed:** {removed}")

        if domain_priorities:
            lines.append(
                f"- **Domain priority features:** {', '.join(domain_priorities)}"
            )
        if flagged:
            lines.append(
                f"- **Flagged domain features (scored poorly):** {', '.join(flagged)}"
            )

        lines.append("")
        lines.append("## Feature Rankings\n")
        lines.append("| Feature | Combined Score | Status |")
        lines.append("|---------|---------------|--------|")

        for feat in rankings.index:
            score = rankings.loc[feat, "combined"]
            status = "✅ Kept" if feat in recommended else "❌ Removed"
            if feat in flagged:
                status += " ⚠️ Flagged"
            lines.append(f"| {feat} | {score:.4f} | {status} |")

        lines.append("")
        lines.append("## Detailed Explanations\n")

        for feat in rankings.index:
            lines.append(f"### {feat}\n")
            score = rankings.loc[feat, "combined"]

            if feat in recommended:
                reason_parts = [f"Combined score: {score:.4f}"]
                if domain_priorities and feat in domain_priorities:
                    reason_parts.append("domain-priority feature (1.5x weight)")
                lines.append(
                    f"**Kept** — {'; '.join(reason_parts)}.\n"
                )
            else:
                lines.append(
                    f"**Removed** — Combined score {score:.4f} fell below the "
                    f"top-{n_features} threshold.\n"
                )

            if feat in flagged:
                lines.append(
                    "⚠️ **Warning:** This domain-recommended feature scored below "
                    "median on the majority of statistical methods. Consider reviewing "
                    "whether domain knowledge justifies its inclusion.\n"
                )

        return "\n".join(lines)
