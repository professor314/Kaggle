"""House Prices Research V2 — Advanced Feature Engineering with Arena Evaluation.

Research-driven features:
1. Remove outliers (GrLivArea > 4000 with low SalePrice)
2. Log-transform all skewed numeric features (skewness > 0.75)
3. Encode ALL categorical features (ordinal for quality, label for others)
4. Polynomial features for top predictors
5. Interaction features (OverallQual × GrLivArea, OverallQual × TotalBsmtSF)
6. More features (TotalPorchSF, TotalBath, flags)
7. Stack predictions (Ridge + Lasso + GBM average)

Previous best: 0.14545 LB (GBM very conservative, 10 features)
Target: 0.12 LB
"""

import sys
import os
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from kaggle_ml_toolkit import ModelArena, CompetitionConfig, SubmissionGenerator
from kaggle_ml_toolkit.arena_generator import ArenaGenerator

# ==============================================================================
# LOAD DATA
# ==============================================================================
data_dir = project_root / "competitions" / "house-prices" / "data"
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

test_ids = test_df["Id"]

# ==============================================================================
# 1. REMOVE OUTLIERS
# ==============================================================================
# Famous outliers: GrLivArea > 4000 with SalePrice < 300000
outlier_mask = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)
print(f"Removing {outlier_mask.sum()} outliers")
train_df = train_df[~outlier_mask].reset_index(drop=True)

# Target: log1p transform
y = np.log1p(train_df["SalePrice"])
train_df.drop("SalePrice", axis=1, inplace=True)

# Combine for consistent processing
train_df["_is_train"] = True
test_df["_is_train"] = False
ntrain = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)
combined.drop("Id", axis=1, inplace=True)

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

# --- 6. Aggregate features ---
combined["TotalSF"] = combined["TotalBsmtSF"].fillna(0) + combined["1stFlrSF"] + combined["2ndFlrSF"]
combined["TotalPorchSF"] = (
    combined["OpenPorchSF"].fillna(0) + combined["EnclosedPorch"].fillna(0) +
    combined["3SsnPorch"].fillna(0) + combined["ScreenPorch"].fillna(0) +
    combined["WoodDeckSF"].fillna(0)
)
combined["TotalBath"] = (
    combined["FullBath"].fillna(0) + 0.5 * combined["HalfBath"].fillna(0) +
    combined["BsmtFullBath"].fillna(0) + 0.5 * combined["BsmtHalfBath"].fillna(0)
)
combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
combined["YearsSinceRemodel"] = combined["YrSold"] - combined["YearRemodAdd"]

# Flag features
combined["HasBasement"] = (combined["TotalBsmtSF"].fillna(0) > 0).astype(int)
combined["HasGarage"] = (combined["GarageArea"].fillna(0) > 0).astype(int)
combined["HasPool"] = (combined["PoolArea"].fillna(0) > 0).astype(int)
combined["Has2ndFloor"] = (combined["2ndFlrSF"] > 0).astype(int)
combined["HasFireplace"] = (combined["Fireplaces"].fillna(0) > 0).astype(int)

# --- 3. Encode ALL categorical features ---
# Quality ordinal mapping
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
quality_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

for col in quality_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna("NA").map(quality_map).fillna(0).astype(int)

# Other ordinal features
bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0}
combined["BsmtExposure"] = combined["BsmtExposure"].fillna("NA").map(bsmt_exposure_map).fillna(0).astype(int)

bsmt_finish_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NA": 0}
for col in ["BsmtFinType1", "BsmtFinType2"]:
    combined[col] = combined[col].fillna("NA").map(bsmt_finish_map).fillna(0).astype(int)

functional_map = {"Typ": 8, "Min1": 7, "Min2": 6, "Mod": 5, "Maj1": 4, "Maj2": 3, "Sev": 2, "Sal": 1}
combined["Functional"] = combined["Functional"].fillna("Typ").map(functional_map).fillna(8).astype(int)

garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0}
combined["GarageFinish"] = combined["GarageFinish"].fillna("NA").map(garage_finish_map).fillna(0).astype(int)

fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "NA": 0}
combined["Fence"] = combined["Fence"].fillna("NA").map(fence_map).fillna(0).astype(int)

# Label encode remaining categorical features
categorical_cols = combined.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != "_is_train"]

label_encoders = {}
for col in categorical_cols:
    combined[col] = combined[col].fillna("Missing")
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# --- Fill remaining NaN in numeric columns ---
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "_is_train"]
combined[numeric_cols] = combined[numeric_cols].fillna(0)

# --- 2. Log-transform skewed numeric features ---
skewed_feats = []
for col in numeric_cols:
    sk = skew(combined[col].dropna())
    if abs(sk) > 0.75:
        skewed_feats.append(col)
        combined[col] = np.log1p(np.maximum(combined[col], 0))

print(f"Log-transformed {len(skewed_feats)} skewed features")

# --- 4. Polynomial features for top predictors ---
combined["OverallQual_sq"] = combined["OverallQual"] ** 2
combined["TotalSF_sq"] = combined["TotalSF"] ** 2
combined["GrLivArea_sq"] = combined["GrLivArea"] ** 2

# --- 5. Interaction features ---
combined["Qual_x_GrLivArea"] = combined["OverallQual"] * combined["GrLivArea"]
combined["Qual_x_TotalBsmtSF"] = combined["OverallQual"] * combined["TotalBsmtSF"]
combined["Qual_x_TotalSF"] = combined["OverallQual"] * combined["TotalSF"]

# ==============================================================================
# DEFINE FEATURE SETS
# ==============================================================================

# Numeric-only features (~30 top predictors)
numeric_only_features = [
    "OverallQual", "TotalSF", "GrLivArea", "TotalBath", "GarageCars",
    "HouseAge", "YearsSinceRemodel", "TotalBsmtSF", "OverallCond", "LotArea",
    "TotalPorchSF", "HasBasement", "HasGarage", "HasPool", "Has2ndFloor",
    "HasFireplace", "OverallQual_sq", "TotalSF_sq", "GrLivArea_sq",
    "Qual_x_GrLivArea", "Qual_x_TotalBsmtSF", "Qual_x_TotalSF",
    "ExterQual", "KitchenQual", "BsmtQual", "GarageFinish",
    "1stFlrSF", "2ndFlrSF", "GarageArea", "Fireplaces",
]

# Full feature set (numerics + all encoded categoricals)
# Get all columns except _is_train
all_features = [c for c in combined.columns if c != "_is_train"]
# Filter to ensure they exist
full_features = [c for c in all_features if c in combined.columns]

print(f"\nNumeric-only feature count: {len(numeric_only_features)}")
print(f"Full feature count: {len(full_features)}")

# ==============================================================================
# SPLIT BACK INTO TRAIN/TEST
# ==============================================================================
train_processed = combined[combined["_is_train"] == True].copy()
test_processed = combined[combined["_is_train"] == False].copy()

X_train = train_processed.drop("_is_train", axis=1)
X_test = test_processed.drop("_is_train", axis=1)

# Verify no NaNs
print(f"\nTrain NaN check: {X_train[numeric_only_features].isna().sum().sum()}")
print(f"Test NaN check: {X_test[numeric_only_features].isna().sum().sum()}")

# ==============================================================================
# RUN ARENA
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING ARENA — House Prices Research V2")
print("=" * 60)

feature_sets = {
    "numeric_only": numeric_only_features,
    "full": full_features,
}

gen = ArenaGenerator(
    X=X_train,
    y=y,
    problem_type="regression",
    feature_sets=feature_sets,
    time_budget_minutes=10,
    metric="rmse",
    random_state=42,
    prior_cv_lb_gap=0.005,
    model_types=["Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting"],
)

arena = gen.build_arena(verbose=True)
results = arena.run(verbose=True)

# Print leaderboard
print("\n" + "=" * 60)
print("LEADERBOARD")
print("=" * 60)
lb = arena.leaderboard()
print(lb.to_string(index=False))

# Print recommendation
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
rec = arena.recommend()
for k, v in rec.items():
    print(f"  {k}: {v}")

# ==============================================================================
# STACKING (Simple average of Ridge + Lasso + GBM)
# ==============================================================================
print("\n" + "=" * 60)
print("STACKING — Ridge + Lasso + GBM Average")
print("=" * 60)

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use full features for stacking
X_stack = X_train[full_features]
X_test_stack = X_test[full_features]

# Ridge
ridge = Ridge(alpha=10.0)
ridge_oof = cross_val_predict(ridge, X_stack, y, cv=kf)
ridge.fit(X_stack, y)
ridge_pred = ridge.predict(X_test_stack)

# Lasso
lasso = Lasso(alpha=0.0005, max_iter=10000)
lasso_oof = cross_val_predict(lasso, X_stack, y, cv=kf)
lasso.fit(X_stack, y)
lasso_pred = lasso.predict(X_test_stack)

# GBM
gbm = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42
)
gbm_oof = cross_val_predict(gbm, X_stack, y, cv=kf)
gbm.fit(X_stack, y)
gbm_pred = gbm.predict(X_test_stack)

# Stacking CV score
from sklearn.metrics import mean_squared_error
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_oof))
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_oof))
gbm_rmse = np.sqrt(mean_squared_error(y, gbm_oof))

# Simple average
stack_oof = (ridge_oof + lasso_oof + gbm_oof) / 3
stack_rmse = np.sqrt(mean_squared_error(y, stack_oof))

print(f"  Ridge CV RMSE:   {ridge_rmse:.5f}")
print(f"  Lasso CV RMSE:   {lasso_rmse:.5f}")
print(f"  GBM CV RMSE:     {gbm_rmse:.5f}")
print(f"  Stack CV RMSE:   {stack_rmse:.5f}")

# Use stacking if it beats the arena winner
arena_best_score = results[0].mean_score if results else 999
stack_predictions = (ridge_pred + lasso_pred + gbm_pred) / 3

# Determine which to submit
if stack_rmse < arena_best_score:
    print(f"\n  ✅ Stack wins! ({stack_rmse:.5f} < {arena_best_score:.5f})")
    final_predictions = np.expm1(stack_predictions)
    model_name = "research_v2_stack"
else:
    print(f"\n  Arena winner is better ({arena_best_score:.5f} < {stack_rmse:.5f})")
    # Use arena best model
    model, feature_cols = arena.get_best_model()
    final_predictions = np.expm1(model.predict(X_test[feature_cols]))
    model_name = f"research_v2_arena_{results[0].name}"

# Ensure no negative predictions
final_predictions = np.maximum(final_predictions, 0)

# ==============================================================================
# GENERATE SUBMISSION
# ==============================================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

config = CompetitionConfig(
    competition_name="house-prices-advanced-regression-techniques",
    problem_type="regression",
    primary_metric="rmse",
    id_column="Id",
    target_column="SalePrice",
)

submission_dir = project_root / "competitions" / "house-prices" / "submissions"
submission_dir.mkdir(parents=True, exist_ok=True)

sub_gen = SubmissionGenerator()
submission_path = sub_gen.generate(
    predictions=final_predictions,
    test_ids=test_ids,
    config=config,
    model_name=model_name,
    output_dir=str(submission_dir),
)
print(f"Submission saved: {submission_path}")

# Submit to Kaggle
print("\nSubmitting to Kaggle...")
result = subprocess.run(
    ["kaggle", "competitions", "submit", "-c", "house-prices-advanced-regression-techniques",
     "-f", submission_path,
     "-m", f"Research V2: {model_name}, outlier removal, log-transform skewed, all categoricals encoded, polynomial+interaction features, stacking"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print(f"stderr: {result.stderr}")

# Check score
print("\nWaiting for score...")
time.sleep(15)
result = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", "house-prices-advanced-regression-techniques"],
    capture_output=True, text=True
)
print(result.stdout)

print("\n✅ House Prices Research V2 complete!")
