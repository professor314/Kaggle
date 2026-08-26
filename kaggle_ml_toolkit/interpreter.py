"""Model interpretability with SHAP, permutation importance, and PDP.

Provides global feature importance, partial dependence plots, SHAP
analysis, and cross-model comparison with educational narrative output.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


class Interpreter:
    """Model interpretability with educational narrative output.

    Computes SHAP values (when available), permutation importance,
    and partial dependence plots to explain model behavior. Generates
    visualizations and markdown narratives suitable for blog content.

    Falls back gracefully to permutation importance when SHAP is not
    installed or doesn't support the model type.
    """

    def analyze(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        features: Optional[List[str]] = None,
        output_dir: str = "./interpretability",
    ) -> Dict[str, Any]:
        """Run full interpretability suite on a fitted model.

        Computes SHAP values (if possible), permutation importance,
        generates visualizations, and produces a narrative summary.

        Args:
            model: A fitted scikit-learn compatible estimator.
            X: Feature DataFrame used for explanations.
            y: Target Series for permutation importance scoring.
            features: Optional subset of feature names to analyze.
                If None, uses all columns in X.
            output_dir: Directory path to save visualizations and
                narrative. Created if it doesn't exist.

        Returns:
            Dict with keys:
                - shap_values: SHAP values array or None if unavailable.
                - feature_importance: DataFrame with columns 'feature'
                  and 'importance', sorted descending.
                - visualizations: List of file paths to saved plots.
                - narrative: Markdown string explaining findings.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if features is not None:
            X = X[features]

        feature_names = list(X.columns)

        # Attempt SHAP analysis
        shap_values = None
        shap_importance = None
        used_shap = False
        fallback_note = ""

        if _SHAP_AVAILABLE:
            shap_values, shap_importance = self._compute_shap(
                model, X, feature_names
            )
            if shap_values is not None:
                used_shap = True

        if not used_shap:
            fallback_note = (
                "SHAP analysis unavailable"
                + (" (shap not installed)" if not _SHAP_AVAILABLE else " (unsupported model type)")
                + ". Using permutation importance as fallback."
            )

        # Compute permutation importance (always, for comparison or fallback)
        perm_result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": perm_result.importances_mean,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        # Determine the primary feature importance source
        if used_shap and shap_importance is not None:
            feature_importance = shap_importance
        else:
            feature_importance = perm_importance

        # Generate visualizations
        visualizations = self._generate_visualizations(
            model=model,
            X=X,
            feature_names=feature_names,
            feature_importance=feature_importance,
            shap_values=shap_values,
            used_shap=used_shap,
            output_path=output_path,
        )

        # Generate narrative
        narrative = self._generate_narrative(
            feature_importance=feature_importance,
            perm_importance=perm_importance,
            used_shap=used_shap,
            fallback_note=fallback_note,
            feature_names=feature_names,
        )

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "visualizations": visualizations,
            "narrative": narrative,
        }

    def compare_importance(
        self,
        models: List[BaseEstimator],
        model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Compare feature importance rankings across multiple models.

        Computes permutation importance for each model and creates a
        comparison DataFrame highlighting consistently important features
        versus model-specific ones.

        Args:
            models: List of fitted scikit-learn estimators.
            model_names: Human-readable names for each model.
            X: Feature DataFrame for importance computation.
            y: Target Series for scoring.

        Returns:
            DataFrame with feature as index, one column per model
            containing importance scores, and a 'consistency' column
            (standard deviation across models — low values indicate
            consistently important features).
        """
        feature_names = list(X.columns)
        importance_dict: Dict[str, List[float]] = {
            name: [] for name in feature_names
        }

        for model in models:
            perm_result = permutation_importance(
                model, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            for i, feat in enumerate(feature_names):
                importance_dict[feat].append(perm_result.importances_mean[i])

        # Build the comparison DataFrame
        data: Dict[str, Any] = {}
        for i, model_name in enumerate(model_names):
            data[model_name] = [
                importance_dict[feat][i] for feat in feature_names
            ]

        comparison_df = pd.DataFrame(data, index=feature_names)
        comparison_df.index.name = "feature"

        # Add consistency column: std across model columns (low = consistent)
        comparison_df["consistency"] = comparison_df[model_names].std(axis=1)

        # Sort by mean importance descending
        comparison_df["_mean"] = comparison_df[model_names].mean(axis=1)
        comparison_df = comparison_df.sort_values("_mean", ascending=False)
        comparison_df = comparison_df.drop(columns=["_mean"])

        return comparison_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_shap(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        feature_names: List[str],
    ) -> tuple:
        """Attempt to compute SHAP values for the model.

        Tries TreeExplainer first (for tree-based models), then
        LinearExplainer (for linear models). Returns (None, None) if
        the model type is not supported.

        Returns:
            Tuple of (shap_values array, importance DataFrame) or
            (None, None) on failure.
        """
        try:
            # Try tree-based explainer first
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            shap_vals_array = self._normalize_shap_values(shap_vals)

            # Compute mean absolute SHAP values per feature
            mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": mean_abs_shap,
                }
            ).sort_values("importance", ascending=False).reset_index(drop=True)

            return shap_vals_array, importance_df

        except Exception:
            pass

        try:
            # Try linear explainer
            explainer = shap.LinearExplainer(model, X)
            shap_vals = explainer.shap_values(X)
            shap_vals_array = self._normalize_shap_values(shap_vals)

            mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": mean_abs_shap,
                }
            ).sort_values("importance", ascending=False).reset_index(drop=True)

            return shap_vals_array, importance_df

        except Exception:
            pass

        return None, None

    def _normalize_shap_values(self, shap_vals) -> np.ndarray:
        """Normalize SHAP values to a 2D array (samples × features).

        Handles different SHAP output formats:
            - List of arrays (older SHAP for multi-class)
            - 3D array (samples × features × classes) from newer SHAP
            - 2D array (samples × features) for binary/regression

        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        if isinstance(shap_vals, list):
            # Older SHAP: list of arrays, one per class
            # Use mean absolute across classes
            return np.abs(np.array(shap_vals)).mean(axis=0)
        
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            # Newer SHAP: (samples, features, classes)
            # Use mean absolute across classes
            return np.abs(arr).mean(axis=2)
        
        # Already 2D (samples, features)
        return arr

    def _generate_visualizations(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        feature_names: List[str],
        feature_importance: pd.DataFrame,
        shap_values: Optional[np.ndarray],
        used_shap: bool,
        output_path: Path,
    ) -> List[str]:
        """Generate and save interpretability plots.

        Creates:
            - feature_importance.png: Bar chart of feature importances.
            - partial_dependence_*.png: PDP for top 3-4 features.
            - shap_summary.png: SHAP summary plot (if available).

        Args:
            model: Fitted estimator.
            X: Feature DataFrame.
            feature_names: List of feature names.
            feature_importance: DataFrame with 'feature' and 'importance'.
            shap_values: SHAP values array or None.
            used_shap: Whether SHAP was successfully computed.
            output_path: Directory to save plots.

        Returns:
            List of saved file paths as strings.
        """
        visualizations: List[str] = []

        # 1. Feature importance bar chart
        fi_path = self._plot_feature_importance(feature_importance, output_path)
        visualizations.append(fi_path)

        # 2. Partial dependence plots for top features
        top_n = min(4, len(feature_names))
        top_features = feature_importance["feature"].head(top_n).tolist()
        pdp_paths = self._plot_partial_dependence(
            model, X, top_features, output_path
        )
        visualizations.extend(pdp_paths)

        # 3. SHAP summary plot (if available)
        if used_shap and shap_values is not None and _SHAP_AVAILABLE:
            shap_path = self._plot_shap_summary(shap_values, X, output_path)
            if shap_path:
                visualizations.append(shap_path)

        return visualizations

    def _plot_feature_importance(
        self, feature_importance: pd.DataFrame, output_path: Path
    ) -> str:
        """Generate and save a feature importance bar chart."""
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_importance) * 0.4)))

        # Show top 20 features max for readability
        plot_data = feature_importance.head(20)

        ax.barh(
            range(len(plot_data)),
            plot_data["importance"].values,
            color="steelblue",
            edgecolor="none",
        )
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        save_path = str(output_path / "feature_importance.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return save_path

    def _plot_partial_dependence(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        features: List[str],
        output_path: Path,
    ) -> List[str]:
        """Generate and save partial dependence plots for specified features."""
        paths: List[str] = []

        for feature in features:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                PartialDependenceDisplay.from_estimator(
                    model,
                    X,
                    features=[feature],
                    ax=ax,
                    kind="average",
                )
                ax.set_title(f"Partial Dependence: {feature}")
                plt.tight_layout()

                # Sanitize feature name for filename
                safe_name = feature.replace(" ", "_").replace("/", "_")
                save_path = str(output_path / f"partial_dependence_{safe_name}.png")
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                paths.append(save_path)
            except Exception:
                # Skip features that can't generate PDP (e.g., categorical)
                plt.close("all")
                continue

        return paths

    def _plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        output_path: Path,
    ) -> Optional[str]:
        """Generate and save a SHAP summary plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.sca(ax)
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()

            save_path = str(output_path / "shap_summary.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        except Exception:
            plt.close("all")
            return None

    def _generate_narrative(
        self,
        feature_importance: pd.DataFrame,
        perm_importance: pd.DataFrame,
        used_shap: bool,
        fallback_note: str,
        feature_names: List[str],
    ) -> str:
        """Generate a markdown narrative explaining interpretability findings.

        Describes which features matter most, their relative importance,
        and any surprising findings.

        Args:
            feature_importance: Primary importance DataFrame.
            perm_importance: Permutation importance DataFrame.
            used_shap: Whether SHAP was the primary method.
            fallback_note: Note about SHAP fallback if applicable.
            feature_names: All feature names.

        Returns:
            Markdown string suitable for blog inclusion.
        """
        lines: List[str] = [
            "# Model Interpretability Analysis",
            "",
        ]

        # Method description
        if used_shap:
            lines.append(
                "## Method: SHAP (SHapley Additive exPlanations)"
            )
            lines.append("")
            lines.append(
                "SHAP values provide a unified measure of feature importance "
                "based on cooperative game theory. Each feature's contribution "
                "to every prediction is quantified, giving both global and local "
                "explanations."
            )
        else:
            lines.append("## Method: Permutation Importance")
            lines.append("")
            lines.append(
                "Permutation importance measures how much model performance "
                "degrades when a feature's values are randomly shuffled. Features "
                "that cause large performance drops when shuffled are more important."
            )
            if fallback_note:
                lines.append("")
                lines.append(f"> **Note:** {fallback_note}")

        lines.append("")

        # Top features summary
        lines.append("## Key Features")
        lines.append("")

        top_5 = feature_importance.head(5)
        total_importance = feature_importance["importance"].sum()

        lines.append("The most influential features for this model are:")
        lines.append("")
        lines.append("| Rank | Feature | Importance | % of Total |")
        lines.append("|------|---------|------------|------------|")

        for rank, (_, row) in enumerate(top_5.iterrows(), 1):
            pct = (row["importance"] / total_importance * 100) if total_importance > 0 else 0
            lines.append(
                f"| {rank} | {row['feature']} | {row['importance']:.4f} | {pct:.1f}% |"
            )

        lines.append("")

        # Concentration analysis
        if total_importance > 0:
            top_3_importance = feature_importance.head(3)["importance"].sum()
            concentration = top_3_importance / total_importance * 100
            lines.append("## Importance Concentration")
            lines.append("")
            if concentration > 70:
                lines.append(
                    f"The top 3 features account for **{concentration:.1f}%** of "
                    "total importance — the model relies heavily on a small feature "
                    "subset. This suggests strong, clear signals in the data but "
                    "also potential fragility if these features are noisy."
                )
            elif concentration > 40:
                lines.append(
                    f"The top 3 features account for **{concentration:.1f}%** of "
                    "total importance — a balanced distribution indicating the "
                    "model uses multiple informative signals."
                )
            else:
                lines.append(
                    f"The top 3 features account for only **{concentration:.1f}%** "
                    "of total importance — importance is spread across many features, "
                    "suggesting no single dominant predictor."
                )
            lines.append("")

        # Surprising findings
        lines.append("## Observations")
        lines.append("")

        # Check for features with near-zero importance
        low_importance_features = feature_importance[
            feature_importance["importance"] < 0.001
        ]
        if len(low_importance_features) > 0:
            n_low = len(low_importance_features)
            lines.append(
                f"- **{n_low} feature(s)** have near-zero importance and could "
                "potentially be removed without affecting model performance."
            )

        # Check for negative permutation importance (features that hurt)
        negative_perm = perm_importance[perm_importance["importance"] < 0]
        if len(negative_perm) > 0:
            neg_features = negative_perm["feature"].tolist()[:3]
            lines.append(
                f"- Features with negative permutation importance (shuffling "
                f"*improves* the model): {', '.join(neg_features)}. These may "
                "be adding noise rather than signal."
            )

        if len(low_importance_features) == 0 and len(negative_perm) == 0:
            lines.append(
                "- All features contribute positively to model performance. "
                "The feature set appears well-curated."
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "*This analysis was generated automatically by the Kaggle ML Toolkit "
            "Interpreter module.*"
        )

        return "\n".join(lines)
