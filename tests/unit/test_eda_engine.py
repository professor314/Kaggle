"""Unit tests for EDAEngine module.

Tests:
1. test_run_returns_expected_keys
2. test_missing_patterns_detection
3. test_pca_analysis_has_variance_ratios
4. test_narrative_is_markdown
5. test_class_balance_for_classification_target

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

import numpy as np
import pandas as pd
import pytest

from kaggle_ml_toolkit.eda_engine import EDAEngine


@pytest.fixture
def engine():
    return EDAEngine()


@pytest.fixture
def sample_df():
    """Small 10-row, 5-column DataFrame with some missing values and integer target."""
    return pd.DataFrame({
        "feat_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, 10.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "feat_c": [5.0, np.nan, 3.0, 2.0, 1.0, 4.0, np.nan, 6.0, 7.0, 8.0],
        "feat_d": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


class TestRunReturnsExpectedKeys:
    """EDAEngine.run() returns all required dictionary keys."""

    def test_run_returns_expected_keys(self, engine, sample_df, tmp_path):
        result = engine.run(sample_df, target_column="target", output_dir=str(tmp_path))

        expected_keys = {
            "summary_stats",
            "correlations",
            "missing_patterns",
            "class_balance",
            "outlier_report",
            "pca_analysis",
            "multicollinearity",
            "narrative",
            "visualizations",
        }
        assert expected_keys.issubset(set(result.keys()))


class TestMissingPatternsDetection:
    """DataFrame with known missing values reports correct columns and counts."""

    def test_missing_patterns_detection(self, engine, sample_df, tmp_path):
        result = engine.run(sample_df, target_column="target", output_dir=str(tmp_path))

        missing = result["missing_patterns"]
        # feat_a has 2 missing, feat_c has 2 missing
        assert "feat_a" in missing["columns"]
        assert "feat_c" in missing["columns"]
        assert missing["missing_counts"]["feat_a"] == 2
        assert missing["missing_counts"]["feat_c"] == 2
        # feat_b, feat_d, and target have no missing
        assert "feat_b" not in missing["columns"]
        assert "feat_d" not in missing["columns"]
        assert "target" not in missing["columns"]


class TestPcaAnalysisHasVarianceRatios:
    """PCA analysis produces explained_variance_ratio as a non-empty list summing to <= 1.0."""

    def test_pca_analysis_has_variance_ratios(self, engine, sample_df, tmp_path):
        result = engine.run(sample_df, target_column="target", output_dir=str(tmp_path))

        pca = result["pca_analysis"]
        ratios = pca["explained_variance_ratio"]
        assert isinstance(ratios, list)
        assert len(ratios) > 0
        assert sum(ratios) <= 1.0 + 1e-9  # Allow tiny float rounding


class TestNarrativeIsMarkdown:
    """Narrative string starts with a markdown header."""

    def test_narrative_is_markdown(self, engine, sample_df, tmp_path):
        result = engine.run(sample_df, target_column="target", output_dir=str(tmp_path))

        narrative = result["narrative"]
        assert isinstance(narrative, str)
        assert narrative.startswith("# ")


class TestClassBalanceForClassificationTarget:
    """Classification target produces value_counts dict."""

    def test_class_balance_for_classification_target(self, engine, sample_df, tmp_path):
        result = engine.run(sample_df, target_column="target", output_dir=str(tmp_path))

        class_balance = result["class_balance"]
        assert isinstance(class_balance, dict)
        # target has 5 zeros and 5 ones
        assert class_balance[0] == 5
        assert class_balance[1] == 5
