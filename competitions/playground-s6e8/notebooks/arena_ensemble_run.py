"""Playground S6E8 — Model Arena Ensemble Run
=============================================
Goal: Beat 0.965 AUC using the ModelArena to compare diverse model configs,
then ensemble the top performers for a final submission.

Strategy:
1. Multiple LightGBM configs (varying depth, leaves, regularization)
2. XGBoost configs for diversity
3. CatBoost if available
4. HistGradientBoosting from sklearn (fast, handles NaN natively)
5. Ensemble top-N models via rank averaging

Uses ModelArena for fair 5-fold CV comparison on identical splits.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [info] xgboost not available, skipping XGB configs")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("  [info] catboost not available, skipping CatBoost configs")

from kaggle_ml_toolkit import ModelArena

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\playground-s6e8\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\playground-s6e8\submissions")
SUB_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================
print("=" * 60)
print("PLAYGROUND S6E8 — Arena Ensemble Run")
print("=" * 60)

print("\n[1] Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"    Train: {train.shape}, Test: {test.shape}")

y = train["addicted_label"].values
test_ids = test["id"].values
print(f"    Target mean: {y.mean():.4f} ({y.mean()*100:.1f}% positive)")

# Raw feature columns (before engineering)
raw_feature_cols = [c for c in train.columns if c not in ["id", "addicted_label"]]

# Encode categoricals
cat_cols = ["gender", "stress_level", "academic_work_impact"]
for col in cat_cols:
    combined = pd.concat([train[col], test[col]])
    codes = combined.astype("category").cat.codes
    train[col] = codes.iloc[:len(train)].values
    test[col] = codes.iloc[len(train):].values

X_train = train[raw_feature_cols].copy()
X_test = test[raw_feature_cols].copy()

# Feature engineering
print("\n[2] Engineering features...")
for df in [X_train, X_test]:
    # Totals and ratios
    df["total_screen_time"] = df["daily_screen_time_hours"].fillna(0) + df["social_media_hours"].fillna(0) + df["gaming_hours"].fillna(0)
    df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"].fillna(0) / (df["sleep_hours"].fillna(7) + 0.1)
    df["notifications_per_open"] = df["notifications_per_day"].fillna(0) / (df["app_opens_per_day"].fillna(1) + 0.1)
    df["weekend_vs_weekday"] = df["weekend_screen_time"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)
    df["social_media_ratio"] = df["social_media_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
    df["gaming_ratio"] = df["gaming_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
    df["productive_ratio"] = df["work_study_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
    df["age_screen_interaction"] = df["age"] * df["daily_screen_time_hours"].fillna(0)
    df["notif_per_hour"] = df["notifications_per_day"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)
    # New features for this run
    df["sleep_deficit"] = 8.0 - df["sleep_hours"].fillna(7)
    df["total_productive"] = df["work_study_hours"].fillna(0) + df["sleep_hours"].fillna(7)
    df["screen_minus_productive"] = df["total_screen_time"] - df["total_productive"]
    df["notif_x_opens"] = df["notifications_per_day"].fillna(0) * df["app_opens_per_day"].fillna(0)
    df["gaming_x_age"] = df["gaming_hours"].fillna(0) * df["age"]
    df["weekend_minus_weekday"] = df["weekend_screen_time"].fillna(0) - df["daily_screen_time_hours"].fillna(0)

all_features = list(X_train.columns)
print(f"    Total features: {len(all_features)}")

# Define feature subsets for arena diversity
core_features = raw_feature_cols  # 12 raw features
extended_features = all_features  # All 27 features

# ============================================================================
# ARENA SETUP — Multiple model configurations
# ============================================================================
print("\n[3] Setting up Model Arena...")

# We'll use custom LightGBM wrappers since LGBMClassifier is sklearn-compatible
from lightgbm import LGBMClassifier

# Define model configurations to compete
configs = []

# --- LightGBM variants (the current champion) ---
configs.append(("LGBM_baseline", LGBMClassifier(
    n_estimators=2000, learning_rate=0.03, num_leaves=95, max_depth=8,
    min_child_samples=40, colsample_bytree=0.75, subsample=0.75, subsample_freq=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
), extended_features))

configs.append(("LGBM_deeper", LGBMClassifier(
    n_estimators=2000, learning_rate=0.02, num_leaves=127, max_depth=10,
    min_child_samples=30, colsample_bytree=0.7, subsample=0.7, subsample_freq=5,
    reg_alpha=0.05, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1,
), extended_features))

configs.append(("LGBM_shallow", LGBMClassifier(
    n_estimators=3000, learning_rate=0.01, num_leaves=63, max_depth=6,
    min_child_samples=60, colsample_bytree=0.8, subsample=0.8, subsample_freq=5,
    reg_alpha=0.2, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1,
), extended_features))

configs.append(("LGBM_wide", LGBMClassifier(
    n_estimators=2000, learning_rate=0.03, num_leaves=200, max_depth=12,
    min_child_samples=20, colsample_bytree=0.6, subsample=0.7, subsample_freq=5,
    reg_alpha=0.3, reg_lambda=3.0, random_state=42, verbose=-1, n_jobs=-1,
), extended_features))

configs.append(("LGBM_core_only", LGBMClassifier(
    n_estimators=2000, learning_rate=0.03, num_leaves=95, max_depth=8,
    min_child_samples=40, colsample_bytree=0.75, subsample=0.75, subsample_freq=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
), core_features))

configs.append(("LGBM_dart", LGBMClassifier(
    n_estimators=1500, learning_rate=0.05, num_leaves=95, max_depth=8,
    min_child_samples=40, colsample_bytree=0.75, subsample=0.75, subsample_freq=5,
    reg_alpha=0.1, reg_lambda=1.0, boosting_type="dart", drop_rate=0.1,
    random_state=42, verbose=-1, n_jobs=-1,
), extended_features))

# --- XGBoost variants (if available) ---
if HAS_XGB:
    configs.append(("XGB_standard", xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=40, colsample_bytree=0.75, subsample=0.75,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="auc",
        random_state=42, verbosity=0, n_jobs=-1,
        early_stopping_rounds=None,  # no early stop in arena CV
    ), extended_features))

    configs.append(("XGB_deeper", xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.02, max_depth=10,
        min_child_weight=30, colsample_bytree=0.7, subsample=0.7,
        reg_alpha=0.05, reg_lambda=0.5, eval_metric="auc",
        random_state=42, verbosity=0, n_jobs=-1,
    ), extended_features))

# --- CatBoost (if available) ---
if HAS_CB:
    configs.append(("CatBoost_standard", cb.CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, random_seed=42, verbose=0,
    ), extended_features))

# --- HistGradientBoosting (sklearn, fast, handles NaN natively) ---
configs.append(("HGB_standard", HistGradientBoostingClassifier(
    max_iter=1000, learning_rate=0.05, max_depth=8,
    min_samples_leaf=40, max_leaf_nodes=95, l2_regularization=1.0,
    random_state=42,
), extended_features))

configs.append(("HGB_deep", HistGradientBoostingClassifier(
    max_iter=1500, learning_rate=0.03, max_depth=10,
    min_samples_leaf=30, max_leaf_nodes=127, l2_regularization=0.5,
    random_state=42,
), extended_features))

# --- Random Forest (for ensemble diversity) ---
configs.append(("RF_large", RandomForestClassifier(
    n_estimators=500, max_depth=15, min_samples_leaf=10,
    max_features="sqrt", random_state=42, n_jobs=-1,
), extended_features))

print(f"    Total configs: {len(configs)}")
for name, model, feats in configs:
    print(f"      - {name} ({type(model).__name__}, {len(feats)} features)")

# ============================================================================
# RUN ARENA — 5-fold CV with AUC scoring
# ============================================================================
print("\n[4] Running Model Arena (5-fold CV, metric=roc_auc)...")
print("    This dataset is large (691K rows), so this will take a while.\n")

# Use ModelArena with roc_auc metric
arena = ModelArena(
    X=X_train,
    y=pd.Series(y),
    cv_folds=5,
    metric="auc_roc",
    random_state=42,
)

for name, model, feats in configs:
    arena.add(name=name, model=model, feature_cols=feats)

results = arena.run(verbose=True, timeout_minutes=60)

# Print leaderboard
print("\n" + "=" * 60)
print("ARENA LEADERBOARD")
print("=" * 60)
lb = arena.leaderboard()
print(lb.to_string(index=False))

# ============================================================================
# ENSEMBLE — Combine top-N models via rank averaging
# ============================================================================
print("\n\n[5] Building ensemble from top models...")

# Get the top models (those above threshold or top 5)
top_n = min(5, len(results))
top_results = results[:top_n]
print(f"    Ensembling top {top_n} models:")
for r in top_results:
    print(f"      - {r.name}: AUC={r.mean_score:.5f}")

# Re-run top models with OOF predictions for proper ensemble evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_all = {}  # name -> oof predictions
test_preds_all = {}  # name -> test predictions

for name, model_template, feats in configs:
    # Only ensemble the top performers
    if name not in [r.name for r in top_results]:
        continue
    
    print(f"\n    Re-training {name} with OOF predictions...")
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    X_feat_train = X_train[feats]
    X_feat_test = X_test[feats]
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_feat_train, y)):
        X_tr = X_feat_train.iloc[train_idx]
        y_tr = y[train_idx]
        X_val = X_feat_train.iloc[val_idx]
        y_val = y[val_idx]
        
        from sklearn.base import clone
        model = clone(model_template)
        model.fit(X_tr, y_tr)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_feat_test)[:, 1] / 5
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"      Fold {fold+1}: AUC={fold_auc:.5f}")
    
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"      Overall OOF AUC: {overall_auc:.5f}")
    
    oof_preds_all[name] = oof_preds
    test_preds_all[name] = test_preds

# Ensemble methods
print("\n\n[6] Evaluating ensemble strategies...")

# Method 1: Simple average
avg_oof = np.mean(list(oof_preds_all.values()), axis=0)
avg_auc = roc_auc_score(y, avg_oof)
print(f"    Simple average (top {top_n}): AUC = {avg_auc:.5f}")

# Method 2: Rank averaging (more robust)
from scipy.stats import rankdata
rank_oof = np.mean([rankdata(p) for p in oof_preds_all.values()], axis=0)
rank_auc = roc_auc_score(y, rank_oof)
print(f"    Rank average (top {top_n}): AUC = {rank_auc:.5f}")

# Method 3: Weighted average (weight by CV score)
weights = np.array([r.mean_score for r in top_results])
weights = weights / weights.sum()
weighted_oof = np.average(list(oof_preds_all.values()), axis=0, weights=weights)
weighted_auc = roc_auc_score(y, weighted_oof)
print(f"    Weighted average (top {top_n}): AUC = {weighted_auc:.5f}")

# Method 4: Best single model
best_single_name = results[0].name
best_single_auc = roc_auc_score(y, oof_preds_all[best_single_name])
print(f"    Best single ({best_single_name}): AUC = {best_single_auc:.5f}")

# Pick the best strategy
strategies = {
    "simple_avg": (avg_auc, np.mean(list(test_preds_all.values()), axis=0)),
    "rank_avg": (rank_auc, np.mean([rankdata(p) for p in test_preds_all.values()], axis=0)),
    "weighted_avg": (weighted_auc, np.average(list(test_preds_all.values()), axis=0, weights=weights)),
    "best_single": (best_single_auc, test_preds_all[best_single_name]),
}

best_strategy_name = max(strategies, key=lambda k: strategies[k][0])
best_strategy_auc, best_test_preds = strategies[best_strategy_name]

# Normalize rank-averaged predictions to [0, 1] for submission
if best_strategy_name == "rank_avg":
    best_test_preds = (best_test_preds - best_test_preds.min()) / (best_test_preds.max() - best_test_preds.min())

print(f"\n    ★ Best strategy: {best_strategy_name} (OOF AUC = {best_strategy_auc:.5f})")

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n[7] Generating submission...")

# Also save individual best model submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub_file = SUB_DIR / f"arena_ensemble_{best_strategy_name}_{timestamp}.csv"

submission = pd.DataFrame({
    "id": test_ids,
    "addicted_label": best_test_preds,
})
submission.to_csv(sub_file, index=False)
print(f"    Saved: {sub_file}")
print(f"    Prediction range: [{best_test_preds.min():.4f}, {best_test_preds.max():.4f}]")

# Also save best single model if different
if best_strategy_name != "best_single":
    sub_file_single = SUB_DIR / f"arena_best_single_{best_single_name}_{timestamp}.csv"
    submission_single = pd.DataFrame({
        "id": test_ids,
        "addicted_label": test_preds_all[best_single_name],
    })
    submission_single.to_csv(sub_file_single, index=False)
    print(f"    Also saved best single: {sub_file_single}")

# ============================================================================
# MULTI-SEED ENSEMBLE (bonus: reduces variance)
# ============================================================================
print("\n[8] Multi-seed ensemble (3 seeds) with best config...")

# Take the arena winner and train with multiple seeds
best_config = next((name, model, feats) for name, model, feats in configs if name == best_single_name)
bname, bmodel_template, bfeats = best_config

multi_seed_test = np.zeros(len(X_test))
multi_seed_oof = np.zeros(len(X_train))
seeds = [42, 123, 7]

for seed_i, seed in enumerate(seeds):
    cv_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_s = np.zeros(len(X_train))
    test_s = np.zeros(len(X_test))
    
    X_feat_train = X_train[bfeats]
    X_feat_test = X_test[bfeats]
    
    for fold, (train_idx, val_idx) in enumerate(cv_seed.split(X_feat_train, y)):
        model = clone(bmodel_template)
        # Update seed in model if possible
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        
        model.fit(X_feat_train.iloc[train_idx], y[train_idx])
        oof_s[val_idx] = model.predict_proba(X_feat_train.iloc[val_idx])[:, 1]
        test_s += model.predict_proba(X_feat_test)[:, 1] / 5
    
    seed_auc = roc_auc_score(y, oof_s)
    print(f"    Seed {seed}: OOF AUC = {seed_auc:.5f}")
    multi_seed_oof += oof_s / len(seeds)
    multi_seed_test += test_s / len(seeds)

multi_seed_auc = roc_auc_score(y, multi_seed_oof)
print(f"    Multi-seed ensemble AUC: {multi_seed_auc:.5f}")

# Save if better
if multi_seed_auc > best_strategy_auc:
    print(f"    ★ Multi-seed beats arena ensemble! ({multi_seed_auc:.5f} > {best_strategy_auc:.5f})")
    sub_file_ms = SUB_DIR / f"arena_multiseed_{bname}_{timestamp}.csv"
    submission_ms = pd.DataFrame({
        "id": test_ids,
        "addicted_label": multi_seed_test,
    })
    submission_ms.to_csv(sub_file_ms, index=False)
    print(f"    Saved: {sub_file_ms}")
    final_auc = multi_seed_auc
    final_file = sub_file_ms
else:
    final_auc = best_strategy_auc
    final_file = sub_file

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Current best LB:        0.965")
print(f"  Best arena single:      {best_single_auc:.5f} ({best_single_name})")
print(f"  Best ensemble:          {best_strategy_auc:.5f} ({best_strategy_name})")
print(f"  Multi-seed ensemble:    {multi_seed_auc:.5f}")
print(f"  Final submission:       {final_file.name}")
print(f"  Final OOF AUC:          {final_auc:.5f}")
print(f"\n  Submit with:")
print(f'  kaggle competitions submit -c playground-series-s6e8 -f "{final_file}" -m "Arena ensemble: {final_auc:.5f} OOF AUC"')
print("=" * 60)
