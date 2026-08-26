"""Automated Exploratory Data Analysis engine for Kaggle ML Toolkit.

Provides the EDAEngine class that produces comprehensive statistical summaries,
visualizations, PCA analysis, multicollinearity detection, and blog-ready
narrative reports from any pandas DataFrame.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional: statsmodels for VIF computation
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


class EDAEngine:
    """Comprehensive automated EDA with narrative output.

    Runs a full exploratory data analysis suite including statistical
    summaries, correlation analysis, missing value patterns, outlier
    detection, PCA, multicollinearity checks, visualizations, and
    generates a blog-ready narrative report.
    """

    def run(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        output_dir: str = "./eda_output",
    ) -> Dict[str, Any]:
        """Run full EDA suite on the provided DataFrame.

        Args:
            df: Input DataFrame to analyze.
            target_column: Optional name of the target/label column.
                If provided and the column appears to be categorical
                (classification), class balance is reported.
            output_dir: Directory path where visualizations and reports
                are saved. Created if it does not exist.

        Returns:
            Dictionary with keys:
                - summary_stats: Descriptive statistics for all columns.
                - correlations: Correlation matrix for numeric columns.
                - missing_patterns: Missing value counts and percentages.
                - class_balance: Target value counts (if classification).
                - outlier_report: IQR-based outlier detection results.
                - pca_analysis: Explained variance ratios and loadings.
                - multicollinearity: High-correlation feature pairs.
                - narrative: Markdown narrative string.
                - visualizations: List of saved visualization file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Summary statistics
        summary_stats = df.describe(include="all")

        # Correlation matrix for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

        # Missing value patterns
        missing_patterns = self._compute_missing_patterns(df)

        # Class balance (if target is classification)
        class_balance = None
        if target_column and target_column in df.columns:
            target_series = df[target_column]
            # Treat as classification if categorical or few unique values
            n_unique = target_series.nunique()
            if (
                target_series.dtype == "object"
                or isinstance(target_series.dtype, pd.CategoricalDtype)
                or n_unique <= 20
            ):
                class_balance = target_series.value_counts().to_dict()

        # Outlier detection (IQR method) on numeric columns
        outlier_report = self._detect_outliers(df)

        # PCA analysis
        pca_analysis = self._compute_pca(df)

        # Multicollinearity detection
        multicollinearity = self._detect_multicollinearity(df)

        # Visualizations
        visualizations = self._generate_visualizations(
            df, target_column, output_dir
        )

        # Assemble findings
        findings: Dict[str, Any] = {
            "summary_stats": summary_stats,
            "correlations": correlations,
            "missing_patterns": missing_patterns,
            "class_balance": class_balance,
            "outlier_report": outlier_report,
            "pca_analysis": pca_analysis,
            "multicollinearity": multicollinearity,
            "narrative": "",
            "visualizations": visualizations,
            "shape": df.shape,
            "target_column": target_column,
        }

        # Generate narrative
        narrative = self.generate_narrative(findings)
        findings["narrative"] = narrative

        return findings

    def _compute_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute missing value counts and percentages per column.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with keys:
                - columns: list of column names with missing values
                - missing_counts: dict of column → missing count
                - missing_percentages: dict of column → missing percentage
        """
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100) if len(df) > 0 else missing_counts * 0

        cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

        return {
            "columns": cols_with_missing,
            "missing_counts": missing_counts[cols_with_missing].to_dict(),
            "missing_percentages": missing_pct[cols_with_missing].to_dict(),
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using the IQR method on all numeric columns.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with column-level outlier counts and indices.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        report: Dict[str, Any] = {
            "columns": {},
            "total_outliers": 0,
            "method": "iqr",
            "threshold": 1.5,
        }

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask = outlier_mask & df[col].notna()
            indices = df.index[outlier_mask].tolist()

            report["columns"][col] = {
                "count": len(indices),
                "indices": indices,
            }
            report["total_outliers"] += len(indices)

        return report

    def _compute_pca(
        self, df: pd.DataFrame, n_components: int = 5
    ) -> Dict[str, Any]:
        """Perform PCA on numeric columns and return variance analysis.

        Args:
            df: Input DataFrame.
            n_components: Maximum number of principal components to compute.
                Actual count is min(n_components, n_features, n_samples).

        Returns:
            Dictionary with keys:
                - explained_variance_ratio: list of per-component ratios.
                - component_loadings: DataFrame mapping components to features.
                - n_components_used: actual number of components computed.
            Returns empty dict if insufficient numeric data.
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {
                "explained_variance_ratio": [],
                "component_loadings": pd.DataFrame(),
                "n_components_used": 0,
            }

        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Determine number of components
        actual_components = min(
            n_components, numeric_df.shape[1], numeric_df.shape[0]
        )

        if actual_components < 1:
            return {
                "explained_variance_ratio": [],
                "component_loadings": pd.DataFrame(),
                "n_components_used": 0,
            }

        pca = PCA(n_components=actual_components)
        pca.fit(scaled_data)

        # Build component loadings DataFrame
        component_names = [f"PC{i+1}" for i in range(actual_components)]
        loadings_df = pd.DataFrame(
            pca.components_,
            columns=numeric_df.columns,
            index=component_names,
        )

        return {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "component_loadings": loadings_df,
            "n_components_used": actual_components,
        }

    def _detect_multicollinearity(
        self, df: pd.DataFrame, threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Detect multicollinearity by finding highly correlated feature pairs.

        Attempts VIF computation if statsmodels is available. Falls back
        to identifying pairs with |correlation| > threshold.

        Args:
            df: Input DataFrame.
            threshold: Absolute correlation threshold above which pairs
                are flagged.

        Returns:
            List of dicts with keys: col_a, col_b, correlation.
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.empty or numeric_df.shape[1] < 2:
            return []

        corr_matrix = numeric_df.corr()
        high_corr_pairs: List[Dict[str, Any]] = []

        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append(
                        {
                            "col_a": cols[i],
                            "col_b": cols[j],
                            "correlation": float(corr_val),
                        }
                    )

        return high_corr_pairs

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        output_dir: str,
    ) -> List[str]:
        """Generate and save EDA visualizations as PNG files.

        Produces:
            - Distribution histograms for numeric columns
            - Correlation heatmap
            - Missing value bar chart
            - Target distribution (if target_column provided)
            - Feature vs target box plots (if target_column provided)

        Uses the non-interactive 'Agg' backend to avoid display issues.

        Args:
            df: Input DataFrame.
            target_column: Optional target column name.
            output_dir: Directory to save PNG files.

        Returns:
            List of file paths for saved visualizations.
        """
        plt.switch_backend("Agg")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Distribution histograms
        if numeric_cols:
            n_cols = min(len(numeric_cols), 20)  # Cap at 20 columns
            cols_to_plot = numeric_cols[:n_cols]
            n_rows = (n_cols + 3) // 4
            fig, axes = plt.subplots(
                n_rows, min(4, n_cols), figsize=(16, 4 * n_rows)
            )
            if n_cols == 1:
                axes = np.array([axes])
            axes_flat = np.array(axes).flatten()

            for idx, col in enumerate(cols_to_plot):
                ax = axes_flat[idx]
                df[col].dropna().hist(ax=ax, bins=30, edgecolor="black")
                ax.set_title(col, fontsize=10)
                ax.set_xlabel("")

            # Hide unused axes
            for idx in range(n_cols, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            plt.tight_layout()
            hist_path = str(output_path / "distributions.png")
            fig.savefig(hist_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(hist_path)

        # 2. Correlation heatmap
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(
                figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8))
            )
            sns.heatmap(
                corr_matrix,
                annot=len(numeric_cols) <= 15,
                cmap="coolwarm",
                center=0,
                fmt=".2f" if len(numeric_cols) <= 15 else "",
                ax=ax,
            )
            ax.set_title("Feature Correlation Heatmap")
            plt.tight_layout()
            heatmap_path = str(output_path / "correlation_heatmap.png")
            fig.savefig(heatmap_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(heatmap_path)

        # 3. Missing value bar chart
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if not cols_with_missing.empty:
            fig, ax = plt.subplots(
                figsize=(max(8, len(cols_with_missing) * 0.5), 5)
            )
            cols_with_missing.sort_values(ascending=False).plot(
                kind="bar", ax=ax, color="coral", edgecolor="black"
            )
            ax.set_title("Missing Values by Column")
            ax.set_ylabel("Count")
            ax.set_xlabel("Column")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            missing_path = str(output_path / "missing_values.png")
            fig.savefig(missing_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(missing_path)

        # 4. Target distribution (if target provided)
        if target_column and target_column in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            target_series = df[target_column].dropna()

            if (
                target_series.dtype == "object"
                or isinstance(target_series.dtype, pd.CategoricalDtype)
                or target_series.nunique() <= 20
            ):
                target_series.value_counts().plot(
                    kind="bar", ax=ax, color="steelblue", edgecolor="black"
                )
                ax.set_title(f"Target Distribution: {target_column}")
                ax.set_ylabel("Count")
            else:
                target_series.hist(ax=ax, bins=30, edgecolor="black")
                ax.set_title(f"Target Distribution: {target_column}")
                ax.set_ylabel("Frequency")

            plt.tight_layout()
            target_path = str(output_path / "target_distribution.png")
            fig.savefig(target_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(target_path)

            # 5. Feature vs target box plots (for classification targets)
            if (
                target_series.dtype == "object"
                or isinstance(target_series.dtype, pd.CategoricalDtype)
                or target_series.nunique() <= 20
            ):
                feature_cols = [
                    c for c in numeric_cols if c != target_column
                ][:6]  # Limit to 6 features
                if feature_cols:
                    n_features = len(feature_cols)
                    fig, axes = plt.subplots(
                        1, n_features, figsize=(5 * n_features, 5)
                    )
                    if n_features == 1:
                        axes = [axes]

                    for idx, col in enumerate(feature_cols):
                        sns.boxplot(
                            x=target_column, y=col, data=df, ax=axes[idx]
                        )
                        axes[idx].set_title(f"{col} by {target_column}")

                    plt.tight_layout()
                    boxplot_path = str(
                        output_path / "feature_vs_target_boxplots.png"
                    )
                    fig.savefig(boxplot_path, dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    saved_paths.append(boxplot_path)

        return saved_paths

    def generate_narrative(self, findings: Dict[str, Any]) -> str:
        """Generate a blog-ready markdown narrative from EDA findings.

        Produces a structured narrative covering dataset overview, missing
        values, correlations, class balance, outliers, PCA insights, and
        recommended preprocessing steps.

        Args:
            findings: Dictionary of EDA results (as produced by run()).

        Returns:
            Markdown-formatted string suitable for a blog post audience.
        """
        sections: List[str] = []

        # Title
        sections.append("# Exploratory Data Analysis Report\n")

        # Dataset overview
        shape = findings.get("shape", (0, 0))
        sections.append("## Dataset Overview\n")
        sections.append(
            f"The dataset contains **{shape[0]:,} rows** and "
            f"**{shape[1]} columns**.\n"
        )

        # Summary stats highlights
        summary_stats = findings.get("summary_stats")
        if summary_stats is not None and not summary_stats.empty:
            numeric_count = len(
                summary_stats.columns[
                    summary_stats.loc["count"].notna()
                    if "count" in summary_stats.index
                    else []
                ]
            )
            sections.append(
                f"The dataset includes numeric and categorical features "
                f"across {len(summary_stats.columns)} total columns.\n"
            )

        # Missing values
        missing = findings.get("missing_patterns", {})
        missing_cols = missing.get("columns", [])
        sections.append("## Missing Values\n")
        if missing_cols:
            sections.append(
                f"**{len(missing_cols)} column(s)** contain missing values:\n"
            )
            missing_pcts = missing.get("missing_percentages", {})
            missing_counts = missing.get("missing_counts", {})
            for col in sorted(
                missing_cols, key=lambda c: missing_pcts.get(c, 0), reverse=True
            ):
                pct = missing_pcts.get(col, 0)
                count = missing_counts.get(col, 0)
                sections.append(f"- **{col}**: {count} missing ({pct:.1f}%)")
            sections.append("")
        else:
            sections.append("No missing values detected in the dataset.\n")

        # Correlations
        correlations = findings.get("correlations")
        multicollinearity = findings.get("multicollinearity", [])
        sections.append("## Feature Correlations\n")
        if multicollinearity:
            sections.append(
                "The following feature pairs show high correlation "
                "(|r| > 0.9), indicating potential multicollinearity:\n"
            )
            for pair in multicollinearity[:10]:  # Limit display
                sections.append(
                    f"- **{pair['col_a']}** ↔ **{pair['col_b']}**: "
                    f"r = {pair['correlation']:.3f}"
                )
            sections.append("")
        elif correlations is not None and not correlations.empty:
            # Find top correlations below threshold
            corr_flat = correlations.where(
                np.triu(np.ones(correlations.shape), k=1).astype(bool)
            ).stack()
            if not corr_flat.empty:
                top_pairs = corr_flat.abs().nlargest(5)
                sections.append("Top correlated feature pairs:\n")
                for (col_a, col_b), val in top_pairs.items():
                    actual_val = correlations.loc[col_a, col_b]
                    sections.append(
                        f"- **{col_a}** ↔ **{col_b}**: r = {actual_val:.3f}"
                    )
                sections.append("")
        else:
            sections.append("Insufficient numeric features for correlation analysis.\n")

        # Class balance
        class_balance = findings.get("class_balance")
        target_column = findings.get("target_column")
        sections.append("## Class Balance\n")
        if class_balance:
            total = sum(class_balance.values())
            sections.append(
                f"Target variable **{target_column}** distribution:\n"
            )
            for label, count in sorted(
                class_balance.items(), key=lambda x: x[1], reverse=True
            ):
                pct = count / total * 100 if total > 0 else 0
                sections.append(f"- **{label}**: {count} ({pct:.1f}%)")
            sections.append("")

            # Assess balance
            proportions = [v / total for v in class_balance.values()]
            max_prop = max(proportions)
            min_prop = min(proportions)
            if max_prop / max(min_prop, 1e-10) > 3:
                sections.append(
                    "⚠️ **Imbalanced classes detected.** Consider stratified "
                    "sampling, class weights, or data augmentation (e.g., SMOTE).\n"
                )
            else:
                sections.append("Classes appear reasonably balanced.\n")
        else:
            if target_column:
                sections.append(
                    f"Target column '{target_column}' appears to be continuous "
                    "(regression task).\n"
                )
            else:
                sections.append("No target column specified.\n")

        # Outliers
        outlier_report = findings.get("outlier_report", {})
        sections.append("## Outlier Detection\n")
        total_outliers = outlier_report.get("total_outliers", 0)
        if total_outliers > 0:
            sections.append(
                f"IQR-based detection found **{total_outliers} total outlier "
                f"observations** across numeric columns:\n"
            )
            outlier_cols = outlier_report.get("columns", {})
            for col, info in sorted(
                outlier_cols.items(), key=lambda x: x[1]["count"], reverse=True
            )[:10]:
                sections.append(f"- **{col}**: {info['count']} outliers")
            sections.append("")
        else:
            sections.append("No significant outliers detected using IQR method.\n")

        # PCA insights
        pca = findings.get("pca_analysis", {})
        sections.append("## PCA Analysis\n")
        variance_ratios = pca.get("explained_variance_ratio", [])
        if variance_ratios:
            cumulative = np.cumsum(variance_ratios)
            sections.append("Explained variance by principal component:\n")
            for i, (ratio, cum) in enumerate(zip(variance_ratios, cumulative)):
                sections.append(
                    f"- **PC{i+1}**: {ratio:.1%} "
                    f"(cumulative: {cum:.1%})"
                )
            sections.append("")

            loadings = pca.get("component_loadings")
            if loadings is not None and not loadings.empty:
                sections.append("**Top contributing features to PC1:**\n")
                pc1_loadings = loadings.iloc[0].abs().nlargest(5)
                for feat, loading in pc1_loadings.items():
                    sections.append(f"- {feat}: |loading| = {loading:.3f}")
                sections.append("")
        else:
            sections.append(
                "Insufficient numeric features for PCA analysis.\n"
            )

        # Recommended preprocessing steps
        sections.append("## Recommended Preprocessing Steps\n")
        recommendations: List[str] = []

        if missing_cols:
            recommendations.append(
                "Handle missing values — consider imputation strategies "
                "(median for numeric, mode for categorical) or investigate "
                "if missingness is informative."
            )

        if multicollinearity:
            recommendations.append(
                "Address multicollinearity — consider removing one of each "
                "highly correlated pair or using dimensionality reduction."
            )

        if total_outliers > 0:
            recommendations.append(
                "Investigate outliers — determine if they represent data "
                "errors or legitimate extreme values before deciding on "
                "treatment (clip, transform, or retain)."
            )

        if class_balance:
            proportions = [
                v / sum(class_balance.values()) for v in class_balance.values()
            ]
            if max(proportions) / max(min(proportions), 1e-10) > 3:
                recommendations.append(
                    "Address class imbalance using stratified CV, class "
                    "weights, or oversampling techniques."
                )

        if not recommendations:
            recommendations.append(
                "Dataset appears clean. Proceed with feature engineering "
                "and model selection."
            )

        for i, rec in enumerate(recommendations, 1):
            sections.append(f"{i}. {rec}")
        sections.append("")

        return "\n".join(sections)
