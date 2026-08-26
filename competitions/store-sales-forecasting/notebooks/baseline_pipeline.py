"""Store Sales Forecasting — Baseline Pipeline (Phase 1).

Global LightGBM model with safe lag features (lag >= 16).
Trains on log1p(sales), validates on last 16 days of training data.
"""

import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error

from kaggle_ml_toolkit.time_series import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    temporal_train_test_split,
)

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\store-sales-forecasting\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\store-sales-forecasting\submissions")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
stores = pd.read_csv(DATA_DIR / "stores.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

print(f"  Train: {train.shape[0]:,} rows, {train['date'].min()} to {train['date'].max()}")
print(f"  Test: {test.shape[0]:,} rows, {test['date'].min()} to {test['date'].max()}")

# ============================================================
# 2. PREP TARGET (log transform)
# ============================================================
train["log_sales"] = np.log1p(train["sales"].clip(lower=0))

# ============================================================
# 3. MERGE EXTERNAL DATA
# ============================================================
# Oil prices (forward fill missing)
oil = oil.rename(columns={"dcoilwtico": "oil_price"})
oil["oil_price"] = oil["oil_price"].ffill().bfill()

# Merge oil
train = train.merge(oil, on="date", how="left")
test = test.merge(oil, on="date", how="left")

# Forward fill any remaining oil NAs
train["oil_price"] = train["oil_price"].ffill().bfill()
test["oil_price"] = test["oil_price"].ffill().bfill()

# Merge store metadata
train = train.merge(stores, on="store_nbr", how="left")
test = test.merge(stores, on="store_nbr", how="left")

# Holidays — simplified: national holidays only, binary flag
national_holidays = holidays[
    (holidays["locale"] == "National") & 
    (holidays["type"] != "Work Day") &
    (holidays["transferred"] == False)
]["date"].unique()
train["is_holiday"] = train["date"].isin(national_holidays).astype(int)
test["is_holiday"] = test["date"].isin(national_holidays).astype(int)

print("  External data merged.")

# ============================================================
# 4. COMBINE TRAIN + TEST FOR FEATURE ENGINEERING
# ============================================================
# Need combined to compute lags that span the train/test boundary
test["log_sales"] = np.nan
test["sales"] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
print("Engineering features...")

# Time features
combined = add_time_features(combined, "date")

# Encode family as integer
combined["family_encoded"] = combined["family"].astype("category").cat.codes

# Encode store type
combined["store_type_encoded"] = combined["type"].astype("category").cat.codes

# Is payday (15th or last day of month)
combined["is_payday"] = (
    (combined["day_of_month"] == 15) | combined["is_month_end"]
).astype(int)

# Lag features (safe: >= 16 days)
combined = add_lag_features(
    combined, "log_sales", 
    group_cols=["store_nbr", "family"],
    lags=[16, 17, 21, 28, 35, 42, 49, 56]
)

# Rolling features (shifted by 16)
combined = add_rolling_features(
    combined, "log_sales",
    group_cols=["store_nbr", "family"],
    windows=[7, 14, 28, 56],
    min_shift=16
)

# Oil lag
combined["oil_lag_14"] = combined["oil_price"].shift(14)

print(f"  Features engineered. Shape: {combined.shape}")

# ============================================================
# 6. DEFINE FEATURES
# ============================================================
FEATURES = [
    "store_nbr", "family_encoded", "store_type_encoded", "cluster",
    "onpromotion", "oil_price", "oil_lag_14", "is_holiday",
    "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end", "is_payday",
    "quarter", "year",
    "log_sales_lag_16", "log_sales_lag_17", "log_sales_lag_21",
    "log_sales_lag_28", "log_sales_lag_35", "log_sales_lag_42",
    "log_sales_lag_49", "log_sales_lag_56",
    "log_sales_roll_mean_7", "log_sales_roll_mean_14",
    "log_sales_roll_mean_28", "log_sales_roll_mean_56",
    "log_sales_roll_std_7", "log_sales_roll_std_14",
]

TARGET = "log_sales"

# ============================================================
# 7. SPLIT: TRAIN / VALIDATION / TEST
# ============================================================
# Validation: last 16 days of training period (Aug 1-15, 2017)
# This mirrors the test period (Aug 16-31, 2017)
train_data = combined[combined["date"] <= "2017-07-31"].copy()
val_data = combined[
    (combined["date"] >= "2017-08-01") & (combined["date"] <= "2017-08-15")
].copy()
test_data = combined[combined["date"] >= "2017-08-16"].copy()

# Drop rows with NaN target (from early lags not being available)
train_data = train_data.dropna(subset=[TARGET])
val_data = val_data.dropna(subset=[TARGET])

# Drop rows with NaN features (early period before lags are available)
available_features = [f for f in FEATURES if f in combined.columns]
train_data = train_data.dropna(subset=available_features)
val_data = val_data.dropna(subset=available_features)

print(f"  Train: {len(train_data):,} rows")
print(f"  Val: {len(val_data):,} rows")
print(f"  Test: {len(test_data):,} rows")

# ============================================================
# 8. TRAIN LIGHTGBM
# ============================================================
print("\nTraining LightGBM...")

X_train = train_data[available_features]
y_train = train_data[TARGET]
X_val = val_data[available_features]
y_val = val_data[TARGET]

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

callbacks = [
    lgb.early_stopping(100),
    lgb.log_evaluation(100),
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=callbacks,
)

# ============================================================
# 9. EVALUATE
# ============================================================
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log).clip(0)
val_actual = np.expm1(y_val.values).clip(0)

# RMSLE
rmsle = np.sqrt(mean_squared_error(np.log1p(val_actual), np.log1p(val_preds)))
rmse = np.sqrt(mean_squared_error(val_actual, val_preds))

print(f"\n=== VALIDATION RESULTS ===")
print(f"  RMSLE: {rmsle:.5f}")
print(f"  RMSE:  {rmse:.2f}")
print(f"  Best iteration: {model.best_iteration}")

# Feature importance
importance = pd.DataFrame({
    "feature": available_features,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print(f"\n=== TOP 15 FEATURES ===")
print(importance.head(15).to_string(index=False))

# ============================================================
# 10. GENERATE SUBMISSION
# ============================================================
print("\nGenerating submission...")

# Fill NaN features in test with 0 (lags that don't exist)
X_test = test_data[available_features].fillna(0)
test_preds_log = model.predict(X_test)
test_preds = np.expm1(test_preds_log).clip(0)

submission = pd.DataFrame({
    "id": test_data["id"].values,
    "sales": test_preds,
})

sub_file = SUB_DIR / "baseline_lgbm_global.csv"
submission.to_csv(sub_file, index=False)
print(f"  Submission saved: {sub_file}")
print(f"  Shape: {submission.shape}")
print(f"  Sales range: {submission['sales'].min():.2f} - {submission['sales'].max():.2f}")
print(f"  Mean: {submission['sales'].mean():.2f}")

print("\n=== DONE ===")
