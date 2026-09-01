"""Playground S6E8 — Fast LightGBM Ensemble (Arena-lite)
=====================================================
Focused approach: 4 LightGBM configs + multi-seed + rank ensemble.
Skips slow sklearn models — they can't beat LGBM on this data anyway.
Target: beat 0.965 AUC.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\playground-s6e8\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\playground-s6e8\submissions")
SUB_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("S6E8 — Fast LightGBM Arena Ensemble")
print("=" * 60)

# Load
print("\n[1] Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"    Train: {train.shape}, Test: {test.shape}")

y = train["addicted_label"].values
test_ids = test["id"].values

# Encode categoricals
for col in ["gender", "stress_level", "academic_work_impact"]:
    combined = pd.concat([train[col], test[col]])
    codes = combined.astype("category").cat.codes
    train[col] = codes.iloc[:len(train)].values
    test[col] = codes.iloc[len(train):].values

raw_cols = [c for c in train.columns if c not in ["id", "addicted_label"]]
X_train = train[raw_cols].copy()
X_test = test[raw_cols].copy()

# Feature engineering
print("[2] Engineering features...")
for df in [X_train, X_test]:
    df["total_screen_time"] = df["daily_screen_time_hours"].fillna(0) + df["social_media_hours"].fillna(0) + df["gaming_hours"].fillna(0)
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"].fillna(0) / (df["sleep_hours"].fillna(7) + 0.1)
    df["notifications_per_open"] = df["notifications_per_day"].fillna(0) / (df["app_opens_per_day"].fillna(1) + 0.1)
    df["weekend_vs_weekday"] = df["weekend_screen_time"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)
    df["social_media_ratio"] = df["social_media_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
    df["gaming_ratio"] = df["gaming_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
    df["productive_ratio"] = df["work_study_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
    df["age_screen_interaction"] = df["age"] * df["daily_screen_time_hours"].fillna(0)
    df["notif_per_hour"] = df["notifications_per_day"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)
    df["sleep_deficit"] = 8.0 - df["sleep_hours"].fillna(7)
    df["screen_minus_productive"] = df["total_screen_time"] - (df["work_study_hours"].fillna(0) + df["sleep_hours"].fillna(7))
    df["notif_x_opens"] = df["notifications_per_day"].fillna(0) * df["app_opens_per_day"].fillna(0)
    df["weekend_minus_weekday"] = df["weekend_screen_time"].fillna(0) - df["daily_screen_time_hours"].fillna(0)

features = list(X_train.columns)
print(f"    Features: {len(features)}")

# 4 diverse LGBM configs
configs = {
    "lgbm_v1": {  # Current best (baseline)
        "n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 95, "max_depth": 8,
        "min_child_samples": 40, "colsample_bytree": 0.75, "subsample": 0.75,
        "subsample_freq": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
    },
    "lgbm_v2": {  # Deeper, slower learning
        "n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 127, "max_depth": 10,
        "min_child_samples": 30, "colsample_bytree": 0.7, "subsample": 0.7,
        "subsample_freq": 5, "reg_alpha": 0.05, "reg_lambda": 0.5,
    },
    "lgbm_v3": {  # Shallow, more regularized
        "n_estimators": 4000, "learning_rate": 0.01, "num_leaves": 63, "max_depth": 6,
        "min_child_samples": 60, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 5, "reg_alpha": 0.3, "reg_lambda": 2.0,
    },
    "lgbm_v4": {  # Wide leaves, strong reg
        "n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 200, "max_depth": -1,
        "min_child_samples": 50, "colsample_bytree": 0.6, "subsample": 0.7,
        "subsample_freq": 5, "reg_alpha": 0.5, "reg_lambda": 5.0,
    },
}

# Run each config through 5-fold CV
print("\n[3] Running 4 LightGBM configs (5-fold CV each)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_all = {}
test_all = {}

for name, params in configs.items():
    print(f"\n  --- {name} ---")
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y)):
        dtrain = lgb.Dataset(X_train.iloc[train_idx][features], label=y[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx][features], label=y[val_idx], reference=dtrain)

        lgb_params = {**params, "objective": "binary", "metric": "auc",
                      "verbosity": -1, "seed": 42}

        model = lgb.train(
            lgb_params, dtrain, num_boost_round=params["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        oof_preds[val_idx] = model.predict(X_train.iloc[val_idx][features])
        test_preds += model.predict(X_test[features]) / 5

        fold_auc = roc_auc_score(y[val_idx], oof_preds[val_idx])
        print(f"    Fold {fold+1}: AUC={fold_auc:.5f} (iter {model.best_iteration})")

    auc = roc_auc_score(y, oof_preds)
    print(f"    Overall: AUC={auc:.5f}")
    oof_all[name] = oof_preds
    test_all[name] = test_preds

# Ensemble strategies
print("\n\n[4] Ensemble strategies...")

# Simple average
avg_oof = np.mean(list(oof_all.values()), axis=0)
avg_auc = roc_auc_score(y, avg_oof)
print(f"  Simple average:   {avg_auc:.5f}")

# Rank average
rank_oof = np.mean([rankdata(p) for p in oof_all.values()], axis=0)
rank_auc = roc_auc_score(y, rank_oof)
print(f"  Rank average:     {rank_auc:.5f}")

# Weighted by individual AUC
aucs = {name: roc_auc_score(y, oof) for name, oof in oof_all.items()}
weights = np.array(list(aucs.values()))
weights = weights / weights.sum()
weighted_oof = np.average(list(oof_all.values()), axis=0, weights=weights)
weighted_auc = roc_auc_score(y, weighted_oof)
print(f"  Weighted average: {weighted_auc:.5f}")

# Best single
best_name = max(aucs, key=aucs.get)
best_auc = aucs[best_name]
print(f"  Best single ({best_name}): {best_auc:.5f}")

# Multi-seed on best single config
print(f"\n[5] Multi-seed ensemble on {best_name}...")
best_params = configs[best_name]
ms_oof = np.zeros(len(X_train))
ms_test = np.zeros(len(X_test))

for seed in [42, 123, 7]:
    oof_s = np.zeros(len(X_train))
    test_s = np.zeros(len(X_test))
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(cv_s.split(X_train, y)):
        dtrain = lgb.Dataset(X_train.iloc[train_idx][features], label=y[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx][features], label=y[val_idx], reference=dtrain)
        
        lgb_params = {**best_params, "objective": "binary", "metric": "auc",
                      "verbosity": -1, "seed": seed}
        
        model = lgb.train(
            lgb_params, dtrain, num_boost_round=best_params["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        oof_s[val_idx] = model.predict(X_train.iloc[val_idx][features])
        test_s += model.predict(X_test[features]) / 5
    
    seed_auc = roc_auc_score(y, oof_s)
    print(f"    Seed {seed}: {seed_auc:.5f}")
    ms_oof += oof_s / 3
    ms_test += test_s / 3

ms_auc = roc_auc_score(y, ms_oof)
print(f"    Multi-seed AUC: {ms_auc:.5f}")

# Pick best overall
all_results = {
    "simple_avg": (avg_auc, np.mean(list(test_all.values()), axis=0)),
    "rank_avg": (rank_auc, rankdata(np.mean([rankdata(p) for p in test_all.values()], axis=0)) / len(X_test)),
    "weighted_avg": (weighted_auc, np.average(list(test_all.values()), axis=0, weights=weights)),
    "best_single": (best_auc, test_all[best_name]),
    "multi_seed": (ms_auc, ms_test),
}

winner = max(all_results, key=lambda k: all_results[k][0])
winner_auc, winner_preds = all_results[winner]

# Save submissions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_file = SUB_DIR / f"arena_fast_{winner}_{timestamp}.csv"
pd.DataFrame({"id": test_ids, "addicted_label": winner_preds}).to_csv(sub_file, index=False)

print(f"\n{'=' * 60}")
print(f"RESULTS")
print(f"{'=' * 60}")
print(f"  Current LB best:  0.965")
print(f"  Winner strategy:  {winner} (OOF AUC = {winner_auc:.5f})")
print(f"  Submission:       {sub_file.name}")
print(f"\n  Submit:")
print(f'  kaggle competitions submit -c playground-series-s6e8 -f "{sub_file}" -m "Arena ensemble {winner}: {winner_auc:.5f}"')
print(f"{'=' * 60}")
