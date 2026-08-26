"""House Prices Competition — Full pipeline with Model Arena.

Applies lessons from Titanic:
- Start conservative (from strategies.md)
- Fewer features, more regularization on < 2000 rows (1460 training rows)
- Log-transform the target (SalePrice is right-skewed)
- Establish CV-LB gap early with a simple submission
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import numpy as np
from kaggle_ml_toolkit import (
    ModelArena, CompetitionConfig, SubmissionGenerator, EDAEngine
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# === LOAD DATA ===
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Target (SalePrice) range: ${train_df['SalePrice'].min():,.0f} - ${train_df['SalePrice'].max():,.0f}")
print(f"Target mean: ${train_df['SalePrice'].mean():,.0f}, median: ${train_df['SalePrice'].median():,.0f}")

# === STRATEGY: Log-transform target ===
# SalePrice is right-skewed; log transform makes it more normal
# Metric is RMSLE on Kaggle, which is equivalent to RMSE on log(price)
y = np.log1p(train_df['SalePrice'])
print(f"\nLog-transformed target range: {y.min():.2f} - {y.max():.2f}")

# === FEATURE ENGINEERING ===
# From strategies.md: on ~1460 rows, keep features < 15-20 and moderate complexity

def prepare_features(df):
    """Engineer features based on domain knowledge of housing prices."""
    df = df.copy()
    
    # Total square footage (key driver of home price)
    df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Total bathrooms
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                       df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0))
    
    # Age of house at sale
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # Was the house remodeled?
    df['WasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    # Years since remodel
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    
    # Overall quality × condition interaction
    df['QualCond'] = df['OverallQual'] * df['OverallCond']
    
    # Has garage
    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int)
    
    # Has pool
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    # Has fireplace
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    # Neighborhood quality (encode as ordinal based on median price if available)
    # For now, leave as-is and use numeric features only
    
    return df

train_df = prepare_features(train_df)
test_df = prepare_features(test_df)

# === SELECT NUMERIC FEATURES ===
# Start with the most important numeric predictors
core_features = [
    'OverallQual',      # #1 predictor in most analyses
    'TotalSF',          # total living area
    'GrLivArea',        # above-ground living area
    'TotalBath',        # total bathrooms
    'GarageCars',       # garage size
    'HouseAge',         # age at sale
    'YearsSinceRemodel',
    'TotalBsmtSF',      # basement area
    'OverallCond',      # overall condition
    'LotArea',          # lot size
]

extended_features = core_features + [
    'QualCond',         # quality × condition
    'WasRemodeled',     # binary: was it remodeled?
    'HasGarage',
    'HasFireplace',
    'Fireplaces',
]

# Fill NaN in numeric features with median
for col in extended_features:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med)
    test_df[col] = test_df[col].fillna(med)

# Quick verification
X_all = train_df[extended_features]
assert X_all.isna().sum().sum() == 0, f"NaN remaining: {X_all.isna().sum()}"
print(f"\nFeatures prepared: {len(extended_features)} features, 0 NaN")

# === MODEL ARENA ===
# From strategies.md: on 1460 rows, use moderate depth, conservative params
# Metric: RMSE on log-transformed target (equivalent to RMSLE on raw prices)

arena = ModelArena(
    X=train_df,
    y=y,
    cv_folds=5,
    metric='rmse',
    random_state=42,
    prior_cv_lb_gap=0.01,  # regression gaps are typically smaller than classification
)

# --- Linear models (good for regression with numeric features) ---
arena.add('Ridge_core', Ridge(alpha=10.0, random_state=42), core_features,
           'Ridge regression with L2 penalty on core features')
arena.add('Ridge_extended', Ridge(alpha=10.0, random_state=42), extended_features,
           'Ridge with extended features')
arena.add('Ridge_strong_reg', Ridge(alpha=50.0, random_state=42), extended_features,
           'Ridge with stronger regularization')
arena.add('Lasso_core', Lasso(alpha=0.001, random_state=42, max_iter=5000), core_features,
           'Lasso (feature selection via L1)')
arena.add('ElasticNet', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000), extended_features,
           'ElasticNet blend')

# --- Tree-based models (from strategies: depth 3-5 for ~1500 rows) ---
arena.add('RF_conservative', RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42), core_features,
           'Conservative RF on core features')
arena.add('RF_moderate', RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42), extended_features,
           'Moderate RF on extended features')
arena.add('RF_shallow', RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=15, random_state=42), extended_features,
           'Very shallow RF')

# --- Gradient Boosting (our Titanic lesson: conservative wins) ---
arena.add('GBM_conservative', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=10, subsample=0.8, random_state=42), core_features,
           'Conservative GBM')
arena.add('GBM_moderate', GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=8, subsample=0.8, random_state=42), extended_features,
           'Moderate GBM with more trees, lower LR')
arena.add('GBM_strong', GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_samples_leaf=5, subsample=0.8, random_state=42), extended_features,
           'Stronger GBM — watch for overfitting')
arena.add('GBM_very_conservative', GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.15, min_samples_leaf=20, subsample=0.7, random_state=42), core_features,
           'Ultra-conservative GBM')

# --- Run ---
print("\n")
results = arena.run()

print("\n")
print(arena.leaderboard().to_string(index=False))

rec = arena.recommend()
print(f"\nRECOMMENDATION: {rec['reasoning']}")
if rec['warnings']:
    for w in rec['warnings']:
        print(f"  ⚠️  {w}")

# === GENERATE SUBMISSION ===
# Use the arena's best model but predict on test data
# Remember: we trained on log(price), so predictions are log(price) — need to expm1
config = CompetitionConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "competition_config.yaml"))

best_model, best_features = arena.get_best_model()
test_preds_log = best_model.predict(test_df[best_features])
test_preds = np.expm1(test_preds_log)  # convert back from log scale

# Clip negative predictions (can't have negative house prices)
test_preds = np.clip(test_preds, 0, None)

gen = SubmissionGenerator()
sub_dir = os.path.join(os.path.dirname(__file__), "..", "submissions")
path = gen.generate(test_preds, test_df['Id'], config, model_name=f'arena_{rec["recommended_name"]}', output_dir=sub_dir)
print(f"\nSubmission: {os.path.basename(path)}")
print(f"Price range: ${test_preds.min():,.0f} - ${test_preds.max():,.0f}")
print(f"Mean predicted price: ${test_preds.mean():,.0f}")

# Save report
report = arena.summary_report()
report_path = os.path.join(os.path.dirname(__file__), "..", "content", "arena_results.md")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# House Prices — Arena Results\n\n{report}")
print(f"Report: {report_path}")
