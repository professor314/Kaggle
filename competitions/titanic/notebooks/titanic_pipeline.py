"""Titanic Competition Pipeline - Full Workflow Template

This script demonstrates using the kaggle_ml_toolkit package for the
Titanic: Machine Learning from Disaster competition on Kaggle.

It covers the complete pipeline from data loading through submission,
serving as both a working pipeline AND a template for how to use the toolkit
on any Kaggle competition.

Usage:
    1. Download Titanic data: kaggle competitions download -c titanic
    2. Extract to competitions/titanic/data/
    3. Run this script: python competitions/titanic/notebooks/titanic_pipeline.py

Requirements validated: 17.4, 3.1, 8.3, 12.1
"""

import sys
import os

# Ensure the project root is importable when running from the notebooks/ dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# =============================================================================
# Phase 1: Setup and Data Loading
# =============================================================================
# The toolkit exposes a clean import surface — all key classes from one package.

from kaggle_ml_toolkit import (
    CompetitionConfig,
    load_competition_data,
    DataCleaner,
    FeatureEngineer,
    ModelSelector,
    ModelOptimizer,
    Evaluator,
    SubmissionGenerator,
    EDAEngine,
)
import pandas as pd
import numpy as np

# --- Load Configuration ---
# CompetitionConfig centralizes metadata about the competition so all toolkit
# components (submission generator, evaluator, etc.) stay consistent.
config = CompetitionConfig.from_yaml(
    os.path.join(os.path.dirname(__file__), "..", "competition_config.yaml")
)
print(f"Competition: {config.competition_name}")
print(f"Target: {config.target_column}, ID: {config.id_column}")
print(f"Problem type: {config.problem_type}, Metric: {config.primary_metric}")
print(f"Random seed: {config.random_seed}")

# --- Load Data ---
# load_competition_data expects train.csv and test.csv in the given directory.
# It returns a DataBundle with .train and .test attributes.
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
try:
    data = load_competition_data(data_dir)
    train_df = data.train
    test_df = data.test
    print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
except FileNotFoundError as e:
    print(f"\n❌ Data not found: {e}")
    print("Download with: kaggle competitions download -c titanic")
    print("Then extract to competitions/titanic/data/")
    sys.exit(1)


# =============================================================================
# Phase 2: Exploratory Data Analysis (EDA)
# =============================================================================
# EDAEngine automates the heavy lifting of initial data exploration:
# summary stats, missing patterns, correlations, class balance, and visualizations.

print("\n" + "=" * 60)
print("Phase 2: Exploratory Data Analysis")
print("=" * 60)

eda = EDAEngine()
eda_output_dir = os.path.join(os.path.dirname(__file__), "eda_output")
eda_results = eda.run(
    train_df,
    target_column=config.target_column,
    output_dir=eda_output_dir,
)

# Inspect key findings
print(f"\nMissing value columns: {eda_results['missing_patterns']['columns']}")
print(f"Class balance (Survived): {eda_results['class_balance']}")
print(f"Correlation with target (top 5):")
if "correlations" in eda_results:
    target_corr = eda_results["correlations"].get(config.target_column, {})
    sorted_corr = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr_val in sorted_corr[:5]:
        if feat != config.target_column:
            print(f"  {feat}: {corr_val:.3f}")


# =============================================================================
# Phase 3: Feature Engineering
# =============================================================================
# Feature engineering is where domain knowledge from the research document
# gets encoded into the data. For Titanic, the research identifies:
# - Title extraction (social status + age/gender proxy)
# - Family size (non-linear survival effect)
# - Cabin presence (wealth/deck proximity indicator)

print("\n" + "=" * 60)
print("Phase 3: Feature Engineering")
print("=" * 60)


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from passenger Name.

    Names follow the format: "Last, Title. First Middle"
    Titles reveal gender, age (Master = boy), and social status.
    Rare titles are grouped to reduce cardinality.
    """
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Group rare titles — informed by research document analysis
    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    df["Title"] = df["Title"].replace(title_mapping)
    return df


def engineer_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create family-related features.

    Research hypothesis H4: Family size has a non-linear effect.
    - Solo travelers: no one to help during evacuation
    - Small families (2-4): mutual support
    - Large families (5+): difficulty evacuating together
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def engineer_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cabin-related features.

    ~77% of cabin values are missing. Missingness itself is informative:
    passengers with recorded cabins tend to be higher-class.
    When present, the first letter indicates the deck (proximity to lifeboats).
    """
    df = df.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    # Extract deck letter where available
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    return df


# Apply feature engineering to both train and test
train_df = extract_title(train_df)
test_df = extract_title(test_df)

train_df = engineer_family_features(train_df)
test_df = engineer_family_features(test_df)

train_df = engineer_cabin_features(train_df)
test_df = engineer_cabin_features(test_df)

print(f"New features added: Title, FamilySize, IsAlone, HasCabin, Deck")
print(f"Title distribution:\n{train_df['Title'].value_counts().to_string()}")
print(f"\nFamilySize distribution:\n{train_df['FamilySize'].value_counts().sort_index().to_string()}")


# =============================================================================
# Phase 4: Data Cleaning
# =============================================================================
# DataCleaner handles imputation and encoding. All operations return new
# DataFrames (immutable pattern) — the originals are never modified.

print("\n" + "=" * 60)
print("Phase 4: Data Cleaning")
print("=" * 60)

cleaner = DataCleaner()

# --- Impute Age using Title-group medians ---
# Research recommends title-based imputation over global median because
# "Master" = child, "Miss" = young woman, "Mr" = adult male, etc.
print("\nImputing Age using Title-group medians...")
for title in train_df["Title"].unique():
    median_age = train_df.loc[train_df["Title"] == title, "Age"].median()
    train_df.loc[
        (train_df["Title"] == title) & (train_df["Age"].isna()), "Age"
    ] = median_age
    test_df.loc[
        (test_df["Title"] == title) & (test_df["Age"].isna()), "Age"
    ] = median_age

# --- Fill remaining missing values ---
# Embarked: only 2 missing in training, fill with mode ('S' = Southampton)
train_df["Embarked"] = train_df["Embarked"].fillna("S")
# Fare: 1 missing in test, fill with class-specific median
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

print(f"Missing values after cleaning:")
print(f"  Train Age: {train_df['Age'].isna().sum()}")
print(f"  Train Embarked: {train_df['Embarked'].isna().sum()}")
print(f"  Test Fare: {test_df['Fare'].isna().sum()}")

# --- Encode categorical features ---
# DataCleaner.encode() stores the encoding mapping so transform_new()
# can apply the same encoding to test data consistently.
features_to_encode = ["Sex", "Embarked", "Title"]
train_encoded = cleaner.encode(train_df, features_to_encode, method="onehot")
test_encoded = cleaner.transform_new(test_df)

# --- Select features for modeling ---
# Drop raw text/ID columns and the original categoricals (now encoded)
drop_cols = [
    "Survived", "PassengerId", "Name", "Ticket", "Cabin",
    "Sex", "Embarked", "Title", "Deck",
]
feature_cols = [col for col in train_encoded.columns if col not in drop_cols]

# Keep only columns present in both train and test
common_cols = [col for col in feature_cols if col in test_encoded.columns]

X = train_encoded[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = train_df["Survived"]
X_test = test_encoded[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"\nFinal feature set: {len(common_cols)} features")
print(f"Features: {common_cols}")
print(f"X shape: {X.shape}, y shape: {y.shape}, X_test shape: {X_test.shape}")


# =============================================================================
# Phase 5: Model Selection
# =============================================================================
# ModelSelector.compare() trains multiple model candidates with cross-validation
# and returns a ranked comparison table. This identifies which 2-3 models
# deserve hyperparameter tuning (we don't waste time tuning poor candidates).

print("\n" + "=" * 60)
print("Phase 5: Model Selection")
print("=" * 60)

selector = ModelSelector()
results = selector.compare(
    X, y,
    problem_type="classification",
    metric="accuracy",
    cv_folds=5,
)

print("\nModel Comparison Results:")
print(results.to_string(index=False))
print(f"\nBest model: {results.iloc[0]['model_name']} "
      f"(accuracy: {results.iloc[0]['mean_score']:.4f} ± {results.iloc[0]['std_score']:.4f})")


# =============================================================================
# Phase 6: Hyperparameter Optimization
# =============================================================================
# ModelOptimizer tunes the top model(s) using grid or randomized search.
# Randomized search is preferred — it covers more of the parameter space
# per evaluation than grid search in high-dimensional spaces.

print("\n" + "=" * 60)
print("Phase 6: Hyperparameter Optimization")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier

optimizer = ModelOptimizer()

# Define parameter grid informed by research (standard competitive ranges)
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

print(f"Optimizing Random Forest with randomized search (20 iterations)...")
opt_result = optimizer.optimize(
    RandomForestClassifier(random_state=config.random_seed),
    rf_params,
    X, y,
    method="randomized",
    scoring="accuracy",
    n_iter=20,
    random_state=config.random_seed,
)

print(f"\nBest CV accuracy: {opt_result['best_score']:.4f}")
print(f"Best parameters: {opt_result['best_params']}")


# =============================================================================
# Phase 7: Evaluation
# =============================================================================
# Evaluator computes metrics and persists experiment results to JSON.
# This creates a history of all experiments for comparison and reproducibility.

print("\n" + "=" * 60)
print("Phase 7: Evaluation")
print("=" * 60)

from sklearn.model_selection import cross_val_predict

evaluator = Evaluator(
    results_path=os.path.join(os.path.dirname(__file__), "..", "experiments.json")
)

# Generate cross-validated predictions for unbiased evaluation
best_model = opt_result["best_model"]
cv_predictions = cross_val_predict(best_model, X, y, cv=5)

# Evaluate and persist
metrics = evaluator.evaluate(
    y, cv_predictions,
    problem_type="classification",
    experiment_id="titanic_v1",
    model_name="random_forest_optimized",
)
print(f"\nClassification Metrics:")
for metric_name, value in metrics.items():
    print(f"  {metric_name}: {value:.4f}")

# Compare against gender baseline (research: ~77% accuracy)
gender_baseline = (train_df["Sex"] == "female").astype(int)
baseline_acc = (gender_baseline == y).mean()
print(f"\nGender baseline accuracy: {baseline_acc:.4f}")
print(f"Improvement over baseline: {metrics.get('accuracy', 0) - baseline_acc:+.4f}")


# =============================================================================
# Phase 8: Submission Generation
# =============================================================================
# SubmissionGenerator creates a properly formatted CSV for Kaggle upload.
# It uses CompetitionConfig to ensure correct column names and format.

print("\n" + "=" * 60)
print("Phase 8: Submission Generation")
print("=" * 60)

# Train final model on ALL training data (no holdout for final submission)
best_model.fit(X, y)
test_predictions = best_model.predict(X_test)

submission_gen = SubmissionGenerator()
submission_dir = os.path.join(os.path.dirname(__file__), "..", "submissions")
submission_path = submission_gen.generate(
    predictions=test_predictions,
    test_ids=test_df["PassengerId"],
    config=config,
    model_name="rf_optimized",
    output_dir=submission_dir,
)

print(f"Submission saved to: {submission_path}")
print(f"Predictions distribution: {pd.Series(test_predictions).value_counts().to_dict()}")

# Preview the submission file
submission_preview = pd.read_csv(submission_path)
print(f"\nSubmission preview (first 5 rows):")
print(submission_preview.head().to_string(index=False))
print(f"\nSubmission shape: {submission_preview.shape}")

print(f"\nTo submit via CLI:")
print(f"  kaggle competitions submit -c titanic -f {submission_path} -m 'RF optimized v1'")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("✅ Pipeline Complete!")
print("=" * 60)
print(f"""
Results Summary:
  - Features engineered: {len(common_cols)}
  - Best model: Random Forest (optimized)
  - CV Accuracy: {opt_result['best_score']:.4f}
  - Gender baseline: {baseline_acc:.4f}
  - Improvement: {opt_result['best_score'] - baseline_acc:+.4f}
  - Submission: {os.path.basename(submission_path)}

Next Steps:
  1. Submit to Kaggle and check leaderboard score
  2. Compare CV score vs LB score (gap indicates overfitting)
  3. Try Gradient Boosting optimization for potential improvement
  4. Consider ensemble of RF + GBM if both perform well
  5. Generate blog post: ContentGenerator.generate_blog_post()
""")
