"""
Playground Series S6E8 — Screen Addiction Prediction V2
=======================================================
Goal: Beat 0.96437 AUC → Target 0.97+
Strategy:
  1. Targeted feature engineering (26 features, no noise)
  2. Fast tuning on 20% subsample, then validate on full data
  3. Multi-seed OOF with best params for stable predictions
  4. Blend diverse configs for ensemble gain
  
Key insight: 691K rows makes full CV slow. Use subsample for param search,
full data only for final OOF validation.
"""

import sys
import os
import warnings
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "competitions" / "playground-s6e8" / "data"
SUBMISSION_DIR = ROOT / "competitions" / "playground-s6e8" / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

COMPETITION_SLUG = "playground-series-s6e8"
OLD_SCORE = 0.96437

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading data...")
print("=" * 70)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"  Train: {train.shape}, Test: {test.shape}")
y = train["addicted_label"].copy()
test_ids = test["id"].copy()
print(f"  Positive rate: {y.mean():.4f}")
print()

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 2: Feature engineering...")
print("=" * 70)

numeric_cols = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]


def encode_and_engineer(df):
    """Encode categoricals + create targeted engineered features."""
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    df["gender_encoded"] = df["gender"].map(gender_map)
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    df["stress_encoded"] = df["stress_level"].map(stress_map)
    impact_map = {"No": 0, "Yes": 1}
    df["impact_encoded"] = df["academic_work_impact"].map(impact_map)

    # Ratios
    df["total_hours"] = (
        df["daily_screen_time_hours"] + df["social_media_hours"] +
        df["gaming_hours"] + df["work_study_hours"] + df["sleep_hours"]
    )
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"] / df["sleep_hours"].replace(0, np.nan)
    df["notification_per_screen_hour"] = df["notifications_per_day"] / df["daily_screen_time_hours"].replace(0, np.nan)
    df["social_media_fraction"] = df["social_media_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)
    df["gaming_fraction"] = df["gaming_hours"] / df["daily_screen_time_hours"].replace(0, np.nan)
    df["weekend_weekday_ratio"] = df["weekend_screen_time"] / df["daily_screen_time_hours"].replace(0, np.nan)
    df["app_opens_per_notification"] = df["app_opens_per_day"] / df["notifications_per_day"].replace(0, np.nan)
    df["productive_ratio"] = df["work_study_hours"] / (df["social_media_hours"] + df["gaming_hours"]).replace(0, np.nan)

    # Targeted interactions
    df["screen_x_notifications"] = df["daily_screen_time_hours"] * df["notifications_per_day"]
    df["screen_x_app_opens"] = df["daily_screen_time_hours"] * df["app_opens_per_day"]
    df["notifications_x_app_opens"] = df["notifications_per_day"] * df["app_opens_per_day"]
    df["engagement_score"] = (
        df["daily_screen_time_hours"] / 15.0 +
        df["notifications_per_day"] / 200.0 +
        df["app_opens_per_day"] / 100.0
    )
    df["screen_minus_work"] = df["daily_screen_time_hours"] - df["work_study_hours"]
    df["leisure_total"] = df["social_media_hours"] + df["gaming_hours"]
    return df


train = encode_and_engineer(train)
test = encode_and_engineer(test)

features = numeric_cols + [
    "gender_encoded", "stress_encoded", "impact_encoded",
    "total_hours", "screen_to_sleep_ratio", "notification_per_screen_hour",
    "social_media_fraction", "gaming_fraction", "weekend_weekday_ratio",
    "app_opens_per_notification", "productive_ratio",
    "screen_x_notifications", "screen_x_app_opens", "notifications_x_app_opens",
    "engagement_score", "screen_minus_work", "leisure_total",
]

print(f"  Feature count: {len(features)}")

X_train = train[features].copy()
X_test = test[features].copy()
print()

# ---------------------------------------------------------------------------
# 3. Fast Tuning on Subsample
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 3: Fast hyperparameter search on 15% subsample...")
print("=" * 70)

# Subsample for speed
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), size=int(len(X_train) * 0.15), replace=False)
X_sample = X_train.iloc[sample_idx]
y_sample = y.iloc[sample_idx]
print(f"  Subsample size: {len(X_sample)} rows")

param_distributions = {
    "max_iter": randint(300, 1200),
    "max_depth": randint(5, 14),
    "learning_rate": uniform(0.01, 0.14),
    "min_samples_leaf": randint(5, 40),
    "max_leaf_nodes": [31, 63, 127, 200, 255],
    "l2_regularization": uniform(0.0, 1.5),
    "max_bins": [128, 200, 255],
}

cv_sub = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

t0 = time.time()
search = RandomizedSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42, early_stopping=True,
                                             n_iter_no_change=20, validation_fraction=0.1),
    param_distributions=param_distributions,
    n_iter=40,
    scoring="roc_auc",
    cv=cv_sub,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_sample, y_sample)
search_time = time.time() - t0

print(f"\n  Search completed in {search_time:.1f}s")
print(f"  Best subsample CV AUC: {search.best_score_:.5f}")
print(f"  Best params: {search.best_params_}")
print()

# ---------------------------------------------------------------------------
# 4. Define multiple strong configs (search result + hand-tuned variants)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 4: Defining model configs for full OOF...")
print("=" * 70)

best_params = search.best_params_.copy()

# Config 1: Exact search result
config1 = best_params.copy()

# Config 2: Deeper + more iterations + lower LR (exploration)
config2 = best_params.copy()
config2["max_depth"] = min(14, best_params.get("max_depth", 8) + 2)
config2["max_iter"] = min(1500, best_params.get("max_iter", 500) + 300)
config2["learning_rate"] = max(0.005, best_params.get("learning_rate", 0.05) * 0.7)

# Config 3: Shallower + more regularization (diversity)
config3 = best_params.copy()
config3["max_depth"] = max(4, best_params.get("max_depth", 8) - 2)
config3["l2_regularization"] = best_params.get("l2_regularization", 0.1) + 0.5
config3["min_samples_leaf"] = best_params.get("min_samples_leaf", 10) + 10
config3["max_iter"] = min(1500, best_params.get("max_iter", 500) + 200)

# Config 4: Known strong default (from competition experience)
config4 = {
    "max_iter": 800,
    "max_depth": 10,
    "learning_rate": 0.03,
    "min_samples_leaf": 10,
    "max_leaf_nodes": 255,
    "l2_regularization": 0.1,
    "max_bins": 255,
}

# Config 5: High iterations, low LR
config5 = {
    "max_iter": 1200,
    "max_depth": 8,
    "learning_rate": 0.02,
    "min_samples_leaf": 15,
    "max_leaf_nodes": 127,
    "l2_regularization": 0.3,
    "max_bins": 255,
}

configs = {
    "search_best": config1,
    "deeper": config2,
    "regularized": config3,
    "strong_default": config4,
    "low_lr": config5,
}

for name, cfg in configs.items():
    print(f"  {name}: depth={cfg.get('max_depth')}, iter={cfg.get('max_iter')}, "
          f"lr={cfg.get('learning_rate', 0.05):.3f}")
print()

# ---------------------------------------------------------------------------
# 5. Full OOF for each config (5-fold on full 691K rows)
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 5: Full OOF predictions (5-fold, 691K rows)...")
print("=" * 70)
print()

cv_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {}
test_preds_dict = {}
cv_scores = {}

for name, cfg in configs.items():
    oof = np.zeros(len(X_train))
    tpreds = np.zeros(len(X_test))
    fold_scores = []
    t0 = time.time()

    for fold_idx, (tr_idx, val_idx) in enumerate(cv_full.split(X_train, y)):
        model = HistGradientBoostingClassifier(
            **cfg, random_state=42,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
        )
        model.fit(X_train.iloc[tr_idx], y.iloc[tr_idx])
        
        val_pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        oof[val_idx] = val_pred
        tpreds += model.predict_proba(X_test)[:, 1] / 5
        fold_scores.append(roc_auc_score(y.iloc[val_idx], val_pred))

    elapsed = time.time() - t0
    oof_auc = roc_auc_score(y, oof)
    cv_scores[name] = oof_auc
    oof_preds[name] = oof
    test_preds_dict[name] = tpreds
    
    print(f"  {name}: OOF AUC = {oof_auc:.5f} "
          f"(folds: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}) "
          f"[{elapsed:.1f}s]")

print()

# ---------------------------------------------------------------------------
# 6. Multi-seed averaging for the best config
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 6: Multi-seed OOF for best config...")
print("=" * 70)

best_config_name = max(cv_scores, key=cv_scores.get)
best_config = configs[best_config_name]
print(f"  Best config: {best_config_name} (AUC = {cv_scores[best_config_name]:.5f})")

seeds = [42, 123, 456, 789, 2024]
multiseed_oofs = []
multiseed_tests = []

for seed in seeds:
    oof = np.zeros(len(X_train))
    tpreds = np.zeros(len(X_test))
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for tr_idx, val_idx in cv_s.split(X_train, y):
        model = HistGradientBoostingClassifier(
            **best_config, random_state=seed,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
        )
        model.fit(X_train.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        tpreds += model.predict_proba(X_test)[:, 1] / 5
    
    seed_auc = roc_auc_score(y, oof)
    multiseed_oofs.append(oof)
    multiseed_tests.append(tpreds)
    print(f"  Seed {seed}: OOF AUC = {seed_auc:.5f}")

ms_avg_oof = np.mean(multiseed_oofs, axis=0)
ms_avg_test = np.mean(multiseed_tests, axis=0)
ms_avg_score = roc_auc_score(y, ms_avg_oof)
print(f"\n  Multi-seed average: OOF AUC = {ms_avg_score:.5f}")
print()

# ---------------------------------------------------------------------------
# 7. Ensemble: blend all configs + multi-seed
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 7: Ensemble blending...")
print("=" * 70)

# Option A: Average all 5 configs
all_configs_oof = np.mean(list(oof_preds.values()), axis=0)
all_configs_test = np.mean(list(test_preds_dict.values()), axis=0)
all_configs_score = roc_auc_score(y, all_configs_oof)
print(f"  All 5 configs average: {all_configs_score:.5f}")

# Option B: Top-3 configs average
sorted_configs = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
top3_names = [n for n, _ in sorted_configs[:3]]
top3_oof = np.mean([oof_preds[n] for n in top3_names], axis=0)
top3_test = np.mean([test_preds_dict[n] for n in top3_names], axis=0)
top3_score = roc_auc_score(y, top3_oof)
print(f"  Top-3 configs average: {top3_score:.5f}")

# Option C: Top-2 configs average
top2_names = [n for n, _ in sorted_configs[:2]]
top2_oof = np.mean([oof_preds[n] for n in top2_names], axis=0)
top2_test = np.mean([test_preds_dict[n] for n in top2_names], axis=0)
top2_score = roc_auc_score(y, top2_oof)
print(f"  Top-2 configs average: {top2_score:.5f}")

# Option D: Multi-seed of best config (from step 6)
print(f"  Multi-seed best config: {ms_avg_score:.5f}")

# Option E: Blend multi-seed + all-configs
blend_oof = 0.5 * ms_avg_oof + 0.5 * all_configs_oof
blend_test = 0.5 * ms_avg_test + 0.5 * all_configs_test
blend_score = roc_auc_score(y, blend_oof)
print(f"  50/50 multi-seed + all-configs: {blend_score:.5f}")

# Option F: Blend multi-seed + top3
blend2_oof = 0.5 * ms_avg_oof + 0.5 * top3_oof
blend2_test = 0.5 * ms_avg_test + 0.5 * top3_test
blend2_score = roc_auc_score(y, blend2_oof)
print(f"  50/50 multi-seed + top-3: {blend2_score:.5f}")

# Option G: Weighted blend optimized
best_ensemble_score = 0
best_ensemble_oof = None
best_ensemble_test = None
best_ensemble_name = ""

candidates = {
    f"Best single ({best_config_name})": (cv_scores[best_config_name], test_preds_dict[best_config_name]),
    "Multi-seed average": (ms_avg_score, ms_avg_test),
    "All configs average": (all_configs_score, all_configs_test),
    "Top-3 average": (top3_score, top3_test),
    "Top-2 average": (top2_score, top2_test),
    "Blend: ms + all-configs": (blend_score, blend_test),
    "Blend: ms + top-3": (blend2_score, blend2_test),
}

print(f"\n  FINAL COMPARISON:")
for name, (score, _) in sorted(candidates.items(), key=lambda x: x[1][0], reverse=True):
    delta = score - OLD_SCORE
    marker = " <<<" if score == max(s for s, _ in candidates.values()) else ""
    print(f"    {name}: {score:.5f} ({delta:+.5f} vs V1){marker}")

# Select best
best_name = max(candidates, key=lambda k: candidates[k][0])
final_score, final_test = candidates[best_name]
improvement = final_score - OLD_SCORE
print(f"\n  SELECTED: {best_name} → OOF AUC = {final_score:.5f}")
print(f"  Improvement: {improvement:+.5f}")
print()

# ---------------------------------------------------------------------------
# 8. Generate and Submit
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 8: Submission...")
print("=" * 70)

submission = pd.DataFrame({"id": test_ids, "addicted_label": final_test})
print(f"  Shape: {submission.shape}")
print(f"  Range: [{final_test.min():.6f}, {final_test.max():.6f}]")
print(f"  Mean: {final_test.mean():.4f}, NaN: {submission['addicted_label'].isna().any()}")

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_path = SUBMISSION_DIR / f"improve_v2_{timestamp}.csv"
submission.to_csv(submission_path, index=False)
print(f"  Saved: {submission_path}")

submit_message = f"V2 {best_name} | OOF AUC: {final_score:.5f} | 26 feat | multi-config ensemble"

result = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMPETITION_SLUG,
     "-f", str(submission_path), "-m", submit_message],
    capture_output=True, text=True
)
print(f"  Kaggle: {result.stdout.strip()}")
if result.stderr:
    print(f"  stderr: {result.stderr.strip()}")
print()

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Old score (V1):     {OLD_SCORE:.5f}")
print(f"  New CV score (V2):  {final_score:.5f}")
print(f"  Improvement:        {improvement:+.5f}")
print(f"  Approach:           {best_name}")
print(f"  Best config params: {best_config}")
print(f"  Submission:         {submission_path}")
print()
if final_score >= 0.97:
    print("  >>> TARGET MET: 0.97+ AUC achieved in CV! <<<")
elif final_score > OLD_SCORE:
    print(f"  >>> IMPROVEMENT: +{improvement:.5f} over previous best <<<")
else:
    print("  ✗ No improvement over previous best")
print("=" * 70)
