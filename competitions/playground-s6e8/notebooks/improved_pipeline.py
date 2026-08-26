"""Playground Series S6E8 — Screen Time Addiction Prediction.

Binary classification: predict addicted_label from screen time features.
Large dataset (691K rows), mix of numeric + categorical features, has NaNs.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\playground-s6e8\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\playground-s6e8\submissions")

# Load
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"  Train: {train.shape}, Test: {test.shape}")

# Target
y = train["addicted_label"].values
print(f"  Target mean: {y.mean():.4f} (imbalanced: {y.mean()*100:.1f}% positive)")

# Features
feature_cols = [c for c in train.columns if c not in ["id", "addicted_label"]]
print(f"  Features: {feature_cols}")

# Encode categoricals
for col in ["gender", "stress_level", "academic_work_impact"]:
    combined = pd.concat([train[col], test[col]])
    codes = combined.astype("category").cat.codes
    train[col] = codes.iloc[:len(train)].values
    test[col] = codes.iloc[len(train):].values

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Feature engineering
for df in [X_train, X_test]:
    df["total_screen_time"] = df["daily_screen_time_hours"].fillna(0) + df["social_media_hours"].fillna(0) + df["gaming_hours"].fillna(0)
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"].fillna(0) / (df["sleep_hours"].fillna(7) + 0.1)
    df["notifications_per_open"] = df["notifications_per_day"].fillna(0) / (df["app_opens_per_day"].fillna(1) + 0.1)
    df["weekend_vs_weekday"] = df["weekend_screen_time"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)

feature_cols_extended = feature_cols + ["total_screen_time", "screen_to_sleep_ratio", "notifications_per_open", "weekend_vs_weekday"]

print(f"  Extended features: {len(feature_cols_extended)}")

# LightGBM with CV
print("\nTraining LightGBM with 5-fold CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y)):
    X_tr = X_train.iloc[train_idx][feature_cols_extended]
    y_tr = y[train_idx]
    X_val = X_train.iloc[val_idx][feature_cols_extended]
    y_val = y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test[feature_cols_extended]) / 5

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f} (iter {model.best_iteration})")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\n  Overall OOF AUC: {overall_auc:.5f}")

# Feature importance
importance = pd.DataFrame({
    "feature": feature_cols_extended,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print(f"\n  Top 10 features:")
print(importance.head(10).to_string(index=False))

# Submission
submission = pd.DataFrame({
    "id": test["id"],
    "addicted_label": test_preds,
})
sub_file = SUB_DIR / "lgbm_improved.csv"
submission.to_csv(sub_file, index=False)
print(f"\nSubmission saved: {sub_file}")
print(f"Prediction range: {test_preds.min():.4f} - {test_preds.max():.4f}")
print(f"\n=== DONE === OOF AUC: {overall_auc:.5f}")
