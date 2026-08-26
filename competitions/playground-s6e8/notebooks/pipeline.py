"""
Playground Series S6E8 — Phone Addiction Binary Classification Pipeline
=====================================================================
Target: addicted_label (probability prediction for AUC evaluation)
Dataset: 691K train rows, 296K test rows
Strategy: HistGradientBoosting (handles NaN, fast on large data)
"""

import sys
import os
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "competitions" / "playground-s6e8" / "data"
SUBMISSION_DIR = ROOT / "competitions" / "playground-s6e8" / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

COMPETITION_SLUG = "playground-series-s6e8"

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading data...")
print("=" * 70)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"  Train: {train.shape}")
print(f"  Test:  {test.shape}")
print(f"  Sample submission columns: {sample_sub.columns.tolist()}")
print(f"  Sample submission values (head): {sample_sub['addicted_label'].head().tolist()}")
print(f"  -> Expects PROBABILITY predictions (float, likely AUC metric)")
print()

y = train["addicted_label"].copy()
test_ids = test["id"].copy()

print(f"  Target distribution: {y.value_counts().to_dict()}")
print(f"  Positive rate: {y.mean():.4f}")
print()

# ---------------------------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 2: Handling missing values...")
print("=" * 70)

numeric_cols = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
categorical_cols = ["gender", "stress_level", "academic_work_impact"]

# Report missing values
for col in numeric_cols + categorical_cols:
    n_missing = train[col].isna().sum()
    pct = n_missing / len(train) * 100
    if n_missing > 0:
        print(f"  {col}: {n_missing} missing ({pct:.1f}%)")

# For HistGBM we can leave NaN as-is (it handles them natively)
# But we need to encode categoricals

# ---------------------------------------------------------------------------
# 3. Encode Categorical Features
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3: Encoding categorical features...")
print("=" * 70)

# Gender: Male=0, Female=1, Other=2
gender_map = {"Male": 0, "Female": 1, "Other": 2}
train["gender_encoded"] = train["gender"].map(gender_map)
test["gender_encoded"] = test["gender"].map(gender_map)
print(f"  gender: {train['gender'].unique()} -> numeric")

# Stress level: ordinal Low=0, Medium=1, High=2
stress_map = {"Low": 0, "Medium": 1, "High": 2}
train["stress_encoded"] = train["stress_level"].map(stress_map)
test["stress_encoded"] = test["stress_level"].map(stress_map)
print(f"  stress_level: {train['stress_level'].unique()} -> ordinal")

# Academic work impact: binary No=0, Yes=1
impact_map = {"No": 0, "Yes": 1}
train["impact_encoded"] = train["academic_work_impact"].map(impact_map)
test["impact_encoded"] = test["academic_work_impact"].map(impact_map)
print(f"  academic_work_impact: {train['academic_work_impact'].unique()} -> binary")
print()

# ---------------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 4: Feature engineering...")
print("=" * 70)


def engineer_features(df):
    """Create engineered features."""
    # Total hours (sanity check — should be ~24h if all accounted for)
    df["total_hours"] = (
        df["daily_screen_time_hours"] + df["social_media_hours"] +
        df["gaming_hours"] + df["work_study_hours"] + df["sleep_hours"]
    )

    # Screen to sleep ratio
    df["screen_to_sleep_ratio"] = (
        df["daily_screen_time_hours"] / df["sleep_hours"].replace(0, np.nan)
    )

    # Notifications per screen hour
    df["notification_per_screen_hour"] = (
        df["notifications_per_day"] / df["daily_screen_time_hours"].replace(0, np.nan)
    )

    # Social media fraction of total screen time
    df["social_media_fraction"] = (
        df["social_media_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)
    )

    # Gaming fraction of total screen time
    df["gaming_fraction"] = (
        df["gaming_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)
    )

    # Is heavy user (screen time > 10 hours/day)
    df["is_heavy_user"] = (df["daily_screen_time_hours"] > 10).astype(float)
    # Keep as float so NaN rows don't break it
    df.loc[df["daily_screen_time_hours"].isna(), "is_heavy_user"] = np.nan

    # Weekend vs weekday ratio
    df["weekend_weekday_ratio"] = (
        df["weekend_screen_time"] / df["daily_screen_time_hours"].replace(0, np.nan)
    )

    # App opens per notification (engagement density)
    df["app_opens_per_notification"] = (
        df["app_opens_per_day"] / df["notifications_per_day"].replace(0, np.nan)
    )

    # Age group bins
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 18, 25, 35, 50, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Productive vs leisure ratio
    df["productive_ratio"] = (
        df["work_study_hours"] / (
            df["social_media_hours"] + df["gaming_hours"]
        ).replace(0, np.nan)
    )

    return df


train = engineer_features(train)
test = engineer_features(test)

# Define feature sets
core_features = numeric_cols + ["gender_encoded", "stress_encoded", "impact_encoded"]

engineered_features = core_features + [
    "total_hours", "screen_to_sleep_ratio", "notification_per_screen_hour",
    "social_media_fraction", "gaming_fraction", "is_heavy_user",
    "weekend_weekday_ratio", "app_opens_per_notification", "age_group",
    "productive_ratio",
]

print(f"  Core features: {len(core_features)} columns")
print(f"  Engineered features: {len(engineered_features)} columns")
print(f"  New features: {[f for f in engineered_features if f not in core_features]}")
print()

# ---------------------------------------------------------------------------
# 5. Model Arena — Compare models
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 5: Running Model Arena (time_budget=10 min)...")
print("=" * 70)
print()

# Use the toolkit's ArenaGenerator approach manually since we want
# HistGBM which isn't in the default config

X_train = train[engineered_features].copy()

# Build a custom arena
from kaggle_ml_toolkit import ModelArena

arena = ModelArena(
    X=X_train,
    y=y,
    cv_folds=5,
    metric="auc_roc",
    random_state=42,
    prior_cv_lb_gap=0.005,
)

# HistGradientBoosting configs (our primary — handles NaN natively, fastest)
hgb_configs = [
    {"max_iter": 200, "max_depth": 6, "learning_rate": 0.1, "min_samples_leaf": 20, "max_leaf_nodes": 31},
    {"max_iter": 300, "max_depth": 8, "learning_rate": 0.05, "min_samples_leaf": 15, "max_leaf_nodes": 63},
    {"max_iter": 500, "max_depth": 10, "learning_rate": 0.03, "min_samples_leaf": 10, "max_leaf_nodes": 127},
    {"max_iter": 500, "max_depth": 12, "learning_rate": 0.05, "min_samples_leaf": 5, "max_leaf_nodes": 255},
]

for i, params in enumerate(hgb_configs):
    arena.add(
        name=f"HGB_eng_v{i+1}",
        model=HistGradientBoostingClassifier(random_state=42, **params),
        feature_cols=engineered_features,
        description=f"HistGBM depth={params['max_depth']} iter={params['max_iter']}",
    )

# Also test with core features
arena.add(
    name="HGB_core_v1",
    model=HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05,
        min_samples_leaf=15, random_state=42
    ),
    feature_cols=core_features,
    description="HistGBM with core features only",
)

# GBM (for diversity — smaller subsample for speed on large data)
arena.add(
    name="GBM_eng_v1",
    model=GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        min_samples_leaf=20, subsample=0.3, random_state=42
    ),
    feature_cols=core_features,  # Use core to keep GBM manageable
    description="GBM conservative with subsampling",
)

# Random Forest
arena.add(
    name="RF_eng_v1",
    model=RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        n_jobs=-1, random_state=42
    ),
    feature_cols=engineered_features,
    description="Random Forest depth=10",
)

# Logistic Regression (baseline, needs imputed data)
# We'll use core features with median fill for LR
arena.add(
    name="LR_core_v1",
    model=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    feature_cols=core_features,
    description="LogisticRegression baseline",
)

print(f"Arena has {len(arena._entries)} pipelines")
print()

# Run the arena
start_time = time.time()
results = arena.run(verbose=True, timeout_minutes=10)
arena_time = time.time() - start_time

print(f"\nArena completed in {arena_time:.1f}s")
print()

# Print leaderboard
print("=" * 70)
print("LEADERBOARD")
print("=" * 70)
lb = arena.leaderboard()
print(lb.to_string(index=False))
print()

# Get recommendation
rec = arena.recommend()
print(f"Recommendation: {rec['recommended_name']}")
print(f"  CV AUC: {rec['cv_score']:.5f} ± {rec['cv_std']:.5f}")
print(f"  Estimated LB: {rec['estimated_lb_score']:.5f}")
print()

# ---------------------------------------------------------------------------
# 6. Quick Optimization of the Winner
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 6: Optimizing winner with 15-iteration randomized search...")
print("=" * 70)
print()

# Optimize the best HistGBM
param_distributions = {
    "max_iter": randint(200, 800),
    "max_depth": randint(6, 14),
    "learning_rate": uniform(0.01, 0.15),
    "min_samples_leaf": randint(5, 30),
    "max_leaf_nodes": [31, 63, 127, 255, 511],
    "l2_regularization": uniform(0.0, 1.0),
}

opt_model = HistGradientBoostingClassifier(random_state=42)

print("Running RandomizedSearchCV (15 iterations, 5-fold)...")
opt_start = time.time()

search = RandomizedSearchCV(
    estimator=opt_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y)

opt_time = time.time() - opt_start
print(f"\nOptimization completed in {opt_time:.1f}s")
print(f"Best AUC: {search.best_score_:.5f}")
print(f"Best params: {search.best_params_}")
print()

# Compare optimized vs arena winner
best_arena_score = rec["cv_score"]
best_opt_score = search.best_score_
improvement = best_opt_score - best_arena_score

print(f"Arena winner AUC:    {best_arena_score:.5f}")
print(f"Optimized AUC:       {best_opt_score:.5f}")
print(f"Improvement:         {improvement:+.5f}")
print()

# ---------------------------------------------------------------------------
# 7. Generate Submission
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 7: Generating submission...")
print("=" * 70)

# Use the optimized model if it's better, otherwise arena winner
if best_opt_score > best_arena_score:
    print("Using optimized model (better than arena winner)")
    final_model = search.best_estimator_
    final_features = engineered_features
    model_name = "HGB_optimized"
else:
    print("Using arena winner (optimization didn't improve)")
    final_model, final_features = arena.get_best_model()
    model_name = rec["recommended_name"]

# Fit on full training data
print(f"Fitting {model_name} on full training data ({len(X_train)} rows)...")
final_model.fit(X_train[final_features], y)

# Predict probabilities on test
X_test = test[final_features].copy()
predictions = final_model.predict_proba(X_test)[:, 1]

print(f"  Predictions shape: {predictions.shape}")
print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"  Predictions mean: {predictions.mean():.4f} (target mean was {y.mean():.4f})")
print()

# Create submission file
from kaggle_ml_toolkit import CompetitionConfig, SubmissionGenerator

config = CompetitionConfig(
    competition_name=COMPETITION_SLUG,
    target_column="addicted_label",
    id_column="id",
    problem_type="classification",
    primary_metric="auc_roc",
)

gen = SubmissionGenerator()
submission_path = gen.generate(
    predictions=predictions,
    test_ids=test_ids,
    config=config,
    model_name=model_name,
    output_dir=str(SUBMISSION_DIR),
)

print(f"Submission saved: {submission_path}")

# Validate submission
sub_df = pd.read_csv(submission_path)
print(f"  Shape: {sub_df.shape} (expected: {sample_sub.shape})")
print(f"  Columns: {sub_df.columns.tolist()}")
print(f"  Head:\n{sub_df.head()}")
print()

# ---------------------------------------------------------------------------
# 8. Submit via Kaggle CLI
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 8: Submitting to Kaggle...")
print("=" * 70)

submit_message = (
    f"{model_name} | CV AUC: {max(best_arena_score, best_opt_score):.5f} | "
    f"Features: {len(final_features)} | "
    f"HistGBM optimized"
)

try:
    result = gen.submit(
        file_path=submission_path,
        competition_name=COMPETITION_SLUG,
        message=submit_message,
    )
    print(f"  Status: {result['status']}")
    print(f"  Competition: {result['competition']}")
    print(f"  Message: {result['message']}")
except RuntimeError as e:
    print(f"  Submission error: {e}")
    print(f"  Manual submission: kaggle competitions submit -c {COMPETITION_SLUG} -f {submission_path} -m \"{submit_message}\"")

print()
print("=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"  Final model: {model_name}")
print(f"  CV AUC: {max(best_arena_score, best_opt_score):.5f}")
print(f"  Features used: {len(final_features)}")
print(f"  Submission file: {submission_path}")
print("=" * 70)
