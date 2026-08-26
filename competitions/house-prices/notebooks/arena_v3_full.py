"""
House Prices Arena V3 — Full ArenaGenerator → Hyperparameter Tune → Stack → Submit Pipeline
"""

import sys
import os
import warnings
import subprocess
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import skew

from kaggle_ml_toolkit import ArenaGenerator, ModelArena, ModelOptimizer

# ============================================================
# 1. LOAD AND ENGINEER FEATURES
# ============================================================
BASE = r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\house-prices"
train = pd.read_csv(os.path.join(BASE, "data", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "test.csv"))

print(f"Raw train: {train.shape}, test: {test.shape}")

# Target: log-transform SalePrice
y = np.log1p(train['SalePrice'])
test_ids = test['Id']

# Remove outliers (GrLivArea > 4000 with low price)
outlier_mask = (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
train = train[~outlier_mask].reset_index(drop=True)
y = np.log1p(train['SalePrice'])
print(f"After outlier removal: {train.shape}")

def engineer_features(df):
    """Feature engineering for House Prices."""
    df = df.copy()
    
    # Total square footage
    df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalPorchSF'] = (df['OpenPorchSF'].fillna(0) + df['EnclosedPorch'].fillna(0) + 
                           df['3SsnPorch'].fillna(0) + df['ScreenPorch'].fillna(0))
    df['TotalBathrooms'] = (df['FullBath'].fillna(0) + 0.5 * df['HalfBath'].fillna(0) +
                            df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0))
    
    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    # Quality interactions
    df['OverallQual_SF'] = df['OverallQual'] * df['TotalSF']
    df['OverallQual_sq'] = df['OverallQual'] ** 2
    
    # Quality ordinal encoding
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_map).fillna(0)
    
    # Other ordinal encodings
    bsmt_exposure = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
    df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exposure).fillna(0)
    
    bsmt_fin = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].map(bsmt_fin).fillna(0)
    
    garage_fin = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
    df['GarageFinish'] = df['GarageFinish'].map(garage_fin).fillna(0)
    
    fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
    df['Fence'] = df['Fence'].map(fence_map).fillna(0)
    
    # Label encode remaining categoricals
    from sklearn.preprocessing import LabelEncoder
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')
        df[col] = le.fit_transform(df[col])
    
    # Fill remaining NaN with 0
    df = df.fillna(0)
    
    return df

train_eng = engineer_features(train)
test_eng = engineer_features(test)

# Drop non-feature columns
drop_cols = ['Id', 'SalePrice']
for col in drop_cols:
    if col in train_eng.columns:
        train_eng.drop(col, axis=1, inplace=True)
    if col in test_eng.columns:
        test_eng.drop(col, axis=1, inplace=True)

# Align columns
common_cols = list(set(train_eng.columns) & set(test_eng.columns))
common_cols.sort()
train_eng = train_eng[common_cols]
test_eng = test_eng[common_cols]

# Log-transform skewed numeric features
numeric_cols = train_eng.select_dtypes(include=[np.number]).columns
skewed = train_eng[numeric_cols].apply(lambda x: skew(x.dropna()))
skewed_cols = skewed[skewed.abs() > 0.75].index.tolist()
for col in skewed_cols:
    train_eng[col] = np.log1p(train_eng[col].clip(lower=0))
    test_eng[col] = np.log1p(test_eng[col].clip(lower=0))

print(f"Engineered features: {train_eng.shape[1]}")

# Feature sets
# Numeric top 30 by correlation with target
correlations = train_eng.corrwith(y).abs().sort_values(ascending=False)
numeric_30 = correlations.head(30).index.tolist()
full_features = list(train_eng.columns)

X = train_eng.copy()
X_test = test_eng.copy()

print(f"Feature sets: numeric_30={len(numeric_30)}, full={len(full_features)}")

# ============================================================
# 2. ARENA GENERATOR
# ============================================================
print("\n" + "="*60)
print("PHASE 1: ARENA GENERATOR")
print("="*60)

gen = ArenaGenerator(
    X=X,
    y=y,
    problem_type="regression",
    feature_sets={"numeric_30": numeric_30, "full": full_features},
    time_budget_minutes=10,
    prior_cv_lb_gap=0.01,
)

arena = gen.build_arena(verbose=True)

# Add HGB models
arena.add("HGB_full_conservative", HistGradientBoostingRegressor(
    max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=10, random_state=42
), feature_cols=full_features)
arena.add("HGB_full_strong", HistGradientBoostingRegressor(
    max_iter=500, max_depth=5, learning_rate=0.03, min_samples_leaf=5, random_state=42
), feature_cols=full_features)
arena.add("HGB_30_strong", HistGradientBoostingRegressor(
    max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=8, random_state=42
), feature_cols=numeric_30)

results = arena.run(verbose=True)

print("\n--- ARENA LEADERBOARD ---")
lb = arena.leaderboard()
print(lb.to_string(index=False))

# ============================================================
# 3. HYPERPARAMETER TUNING (Top 3)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: HYPERPARAMETER TUNING (Top 3)")
print("="*60)

optimizer = ModelOptimizer()
top3 = results[:3]
tuned_results = []

for i, result in enumerate(top3):
    print(f"\n--- Tuning #{i+1}: {result.name} (CV RMSE: {result.mean_score:.4f}) ---")
    
    model_type = result.model_type
    feature_cols = result.feature_cols
    X_feat = X[feature_cols]
    
    if model_type == "GradientBoostingRegressor":
        base_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "min_samples_leaf": [3, 5, 8, 10, 15],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        }
    elif model_type == "HistGradientBoostingRegressor":
        base_model = HistGradientBoostingRegressor(random_state=42)
        param_grid = {
            "max_iter": [200, 300, 500, 800],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "min_samples_leaf": [3, 5, 8, 10, 15, 20],
            "l2_regularization": [0.0, 0.01, 0.1, 0.5, 1.0],
        }
    elif model_type == "Ridge":
        base_model = Ridge()
        param_grid = {
            "alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0],
        }
    elif model_type == "Lasso":
        base_model = Lasso(max_iter=10000)
        param_grid = {
            "alpha": [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        }
    elif model_type == "ElasticNet":
        base_model = ElasticNet(max_iter=10000)
        param_grid = {
            "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    elif model_type == "RandomForestRegressor":
        base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 12, 15],
            "min_samples_leaf": [2, 3, 5, 8, 10],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        }
    else:
        print(f"  Skipping tuning for {model_type}")
        tuned_results.append({
            "name": result.name, "cv_score": result.mean_score,
            "model": None, "features": feature_cols
        })
        continue
    
    opt_result = optimizer.optimize(
        model=base_model,
        param_grid=param_grid,
        X=X_feat,
        y=y,
        method="randomized",
        scoring="rmse",
        cv_folds=5,
        n_iter=50,
        random_state=42,
    )
    
    print(f"  Best params: {opt_result['best_params']}")
    print(f"  Best CV RMSE: {opt_result['best_score']:.4f}")
    
    tuned_results.append({
        "name": result.name,
        "cv_score": opt_result["best_score"],
        "model": opt_result["best_model"],
        "features": feature_cols,
        "params": opt_result["best_params"],
    })

# ============================================================
# 4. STACKING (Average of top 3 tuned models)
# ============================================================
print("\n" + "="*60)
print("PHASE 3: STACKING")
print("="*60)

# Get valid tuned models
valid_tuned = [t for t in tuned_results if t["model"] is not None]

if len(valid_tuned) >= 2:
    # Cross-validate the stack
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stack_scores = []
    
    for train_idx, val_idx in kf.split(X):
        preds_val = []
        for t in valid_tuned:
            model = t["model"].__class__(**t["model"].get_params())
            X_feat_train = X[t["features"]].iloc[train_idx]
            X_feat_val = X[t["features"]].iloc[val_idx]
            model.fit(X_feat_train, y.iloc[train_idx])
            pred = model.predict(X_feat_val)
            preds_val.append(pred)
        
        avg_pred = np.mean(preds_val, axis=0)
        rmse = np.sqrt(np.mean((avg_pred - y.iloc[val_idx]) ** 2))
        stack_scores.append(rmse)
    
    stack_cv = np.mean(stack_scores)
    print(f"\nStack (avg of {len(valid_tuned)} models) CV RMSE: {stack_cv:.4f}")
    
    # Compare with best individual
    best_individual = min(valid_tuned, key=lambda x: x["cv_score"])
    print(f"Best individual CV RMSE: {best_individual['cv_score']:.4f}")
    
    use_stack = stack_cv < best_individual["cv_score"]
    print(f"Using {'stack' if use_stack else 'individual'}")
else:
    use_stack = False
    best_individual = valid_tuned[0] if valid_tuned else tuned_results[0]

# ============================================================
# 5. GENERATE SUBMISSION
# ============================================================
print("\n" + "="*60)
print("PHASE 4: SUBMISSION")
print("="*60)

if use_stack:
    # Stack predictions on test
    test_preds = []
    for t in valid_tuned:
        model = t["model"].__class__(**t["model"].get_params())
        model.fit(X[t["features"]], y)
        pred = model.predict(X_test[t["features"]])
        test_preds.append(pred)
    
    final_pred_log = np.mean(test_preds, axis=0)
    final_cv = stack_cv
    model_name = f"Stack_{len(valid_tuned)}models"
else:
    best = best_individual
    model = best["model"].__class__(**best["model"].get_params())
    model.fit(X[best["features"]], y)
    final_pred_log = model.predict(X_test[best["features"]])
    final_cv = best["cv_score"]
    model_name = best["name"]

# Convert from log scale
final_pred = np.expm1(final_pred_log)
final_pred = np.clip(final_pred, 0, None)  # No negative prices

# Save submission
sub_dir = os.path.join(BASE, "submissions")
os.makedirs(sub_dir, exist_ok=True)
sub_path = os.path.join(sub_dir, "arena_v3_tuned.csv")

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": final_pred
})
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Price range: ${final_pred.min():.0f} - ${final_pred.max():.0f}")
print(f"Mean price: ${final_pred.mean():.0f}")

# Submit via CLI
print("\n--- Submitting to Kaggle ---")
COMP_NAME = "house-prices-advanced-regression-techniques"
message = f"Arena_v3_{model_name}_RMSE{final_cv:.4f}"

result = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", COMP_NAME, "-f", sub_path, "-m", message],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(result.stderr)

# Wait and check score
print("Waiting 15s for score...")
time.sleep(15)
result = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", COMP_NAME],
    capture_output=True, text=True
)
print(result.stdout)

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "="*60)
print("HOUSE PRICES ARENA V3 SUMMARY")
print("="*60)
print(f"Previous best LB: 0.12728 RMSLE")
print(f"Arena winner: {results[0].name} (CV RMSE: {results[0].mean_score:.4f})")
if use_stack:
    print(f"Stack CV RMSE: {stack_cv:.4f}")
else:
    print(f"Best tuned: {best_individual['name']} (CV RMSE: {best_individual['cv_score']:.4f})")
print(f"Submitted: {model_name} (CV: {final_cv:.4f})")
