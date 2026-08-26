"""Comprehensive smoke test for interpreter module (Tasks 32.1, 32.2, 32.3)."""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from kaggle_ml_toolkit.interpreter import Interpreter, _SHAP_AVAILABLE


def test_analyze_with_shap():
    """Task 32.1: analyze() computes SHAP and returns correct structure."""
    X_arr, y_arr = make_classification(
        n_samples=150, n_features=6, n_informative=4, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["a", "b", "c", "d", "e", "f"])
    y = pd.Series(y_arr)

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        assert "shap_values" in result
        assert "feature_importance" in result
        assert "visualizations" in result
        assert "narrative" in result

        # SHAP should work for RandomForest
        if _SHAP_AVAILABLE:
            assert result["shap_values"] is not None
            assert result["shap_values"].shape == (150, 6)

        # Feature importance structure
        fi = result["feature_importance"]
        assert fi.shape[0] == 6
        assert list(fi.columns) == ["feature", "importance"]
        # Sorted descending
        vals = fi["importance"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_analyze_fallback_permutation():
    """Task 32.1: analyze() falls back to permutation importance gracefully."""
    X_arr, y_arr = make_classification(
        n_samples=80, n_features=4, n_informative=2, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["x1", "x2", "x3", "x4"])
    y = pd.Series(y_arr)

    # KNN doesn't support SHAP Tree or Linear explainer
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        # Should still produce valid results even without SHAP
        assert result["feature_importance"] is not None
        assert result["feature_importance"].shape[0] == 4
        assert len(result["visualizations"]) >= 1  # At least importance plot


def test_visualizations_and_narrative():
    """Task 32.2: Generates correct visualizations and markdown narrative."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(y_arr)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        viz_names = [os.path.basename(v) for v in result["visualizations"]]

        # Feature importance plot
        assert "feature_importance.png" in viz_names

        # PDP plots for top features
        pdp_plots = [v for v in viz_names if v.startswith("partial_dependence_")]
        assert len(pdp_plots) >= 3

        # SHAP summary (if available)
        if _SHAP_AVAILABLE:
            assert "shap_summary.png" in viz_names

        # All files exist on disk
        for path in result["visualizations"]:
            assert os.path.exists(path)

        # Narrative is valid markdown
        narrative = result["narrative"]
        assert narrative.startswith("# Model Interpretability Analysis")
        assert "## Key Features" in narrative
        assert "| Rank |" in narrative
        assert "## Observations" in narrative


def test_compare_importance():
    """Task 32.3: compare_importance() returns correct DataFrame structure."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["a", "b", "c", "d", "e"])
    y = pd.Series(y_arr)

    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=10, random_state=42).fit(X, y)
    lr = LogisticRegression(random_state=42, max_iter=200).fit(X, y)

    interp = Interpreter()
    comparison = interp.compare_importance(
        models=[rf, gb, lr],
        model_names=["RF", "GB", "LR"],
        X=X,
        y=y,
    )

    # Structure checks
    assert comparison.index.name == "feature"
    assert "RF" in comparison.columns
    assert "GB" in comparison.columns
    assert "LR" in comparison.columns
    assert "consistency" in comparison.columns
    assert len(comparison) == 5

    # Consistency is non-negative (std)
    assert (comparison["consistency"] >= 0).all()

    # Sorted by mean importance descending
    means = comparison[["RF", "GB", "LR"]].mean(axis=1)
    assert all(means.iloc[i] >= means.iloc[i + 1] for i in range(len(means) - 1))


# ---------------------------------------------------------------
# Task 32.4: Explicit named tests for required coverage
# ---------------------------------------------------------------


def test_analyze_returns_expected_keys():
    """analyze() returns dict with shap_values, feature_importance, visualizations, narrative."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["a", "b", "c", "d", "e"])
    y = pd.Series(y_arr)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        expected_keys = {"shap_values", "feature_importance", "visualizations", "narrative"}
        assert expected_keys == set(result.keys())


def test_feature_importance_sorted_descending():
    """feature_importance DataFrame is sorted by importance descending."""
    X_arr, y_arr = make_classification(
        n_samples=120, n_features=8, n_informative=5, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(8)])
    y = pd.Series(y_arr)

    model = RandomForestClassifier(n_estimators=15, random_state=42)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        fi = result["feature_importance"]
        vals = fi["importance"].values
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1], (
                f"Feature importance not sorted descending at index {i}: "
                f"{vals[i]} < {vals[i + 1]}"
            )


def test_permutation_importance_fallback():
    """KNN model (no SHAP support) still produces valid results via permutation importance."""
    from sklearn.neighbors import KNeighborsClassifier

    X_arr, y_arr = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["x1", "x2", "x3", "x4", "x5"])
    y = pd.Series(y_arr)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        # Should have valid feature_importance even without SHAP
        fi = result["feature_importance"]
        assert fi is not None
        assert fi.shape[0] == 5
        assert list(fi.columns) == ["feature", "importance"]

        # Narrative and visualizations still generated
        assert result["narrative"] is not None
        assert len(result["visualizations"]) >= 1


def test_partial_dependence_plots_generated():
    """At least 1 PDP file exists in visualizations output."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(y_arr)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        pdp_files = [
            v for v in result["visualizations"]
            if "partial_dependence" in os.path.basename(v)
        ]
        assert len(pdp_files) >= 1, "Expected at least 1 partial dependence plot"

        # Verify files actually exist on disk
        for path in pdp_files:
            assert os.path.exists(path), f"PDP file not found: {path}"


def test_compare_importance_returns_dataframe():
    """compare_importance() returns DataFrame with consistency column and feature index."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=4, n_informative=2, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["a", "b", "c", "d"])
    y = pd.Series(y_arr)

    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    lr = LogisticRegression(random_state=42, max_iter=200).fit(X, y)

    interp = Interpreter()
    comparison = interp.compare_importance(
        models=[rf, lr],
        model_names=["RF", "LR"],
        X=X,
        y=y,
    )

    # Returns a DataFrame
    assert isinstance(comparison, pd.DataFrame)

    # Has consistency column
    assert "consistency" in comparison.columns

    # Index is named "feature"
    assert comparison.index.name == "feature"

    # Contains all features
    assert set(comparison.index) == {"a", "b", "c", "d"}


def test_narrative_is_markdown():
    """Narrative output starts with '#' (markdown heading)."""
    X_arr, y_arr = make_classification(
        n_samples=100, n_features=4, n_informative=2, random_state=42
    )
    X = pd.DataFrame(X_arr, columns=["x1", "x2", "x3", "x4"])
    y = pd.Series(y_arr)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    interp = Interpreter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = interp.analyze(model, X, y, output_dir=tmpdir)

        narrative = result["narrative"]
        assert narrative.startswith("#"), (
            f"Narrative should start with '#' but starts with: "
            f"'{narrative[:20]}...'"
        )


if __name__ == "__main__":
    test_analyze_with_shap()
    print("test_analyze_with_shap PASSED")
    test_analyze_fallback_permutation()
    print("test_analyze_fallback_permutation PASSED")
    test_visualizations_and_narrative()
    print("test_visualizations_and_narrative PASSED")
    test_compare_importance()
    print("test_compare_importance PASSED")
    test_analyze_returns_expected_keys()
    print("test_analyze_returns_expected_keys PASSED")
    test_feature_importance_sorted_descending()
    print("test_feature_importance_sorted_descending PASSED")
    test_permutation_importance_fallback()
    print("test_permutation_importance_fallback PASSED")
    test_partial_dependence_plots_generated()
    print("test_partial_dependence_plots_generated PASSED")
    test_compare_importance_returns_dataframe()
    print("test_compare_importance_returns_dataframe PASSED")
    test_narrative_is_markdown()
    print("test_narrative_is_markdown PASSED")
    print("\nALL TESTS PASSED")
