"""Store Sales — Per-Family LightGBM models.

Train 33 separate models (one per product family) for better family-specific
patterns. Uses same safe lag features as baseline.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle")

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error
from kaggle_ml_toolkit.time_series import add_time_features, add_lag_features, add_rolling_features

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\store-sales-forecasting\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\store-sales-forecasting\submissions")

# Load
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
stores = pd.read_csv(DATA_DIR / "stores.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

# Prep
train["log_sales"] = np.log1p(train["sales"].clip(lower=0))

# Oil
oil = oil.rename(columns={"dcoilwtico": "oil_price"})
oil["oil_price"] = oil["oil_price"].ffill().bfill()
train = train.merge(oil, on="date", how="left")
test = test.merge(oil, on="date", how="left")
train["oil_price"] = train["oil_price"].ffill().bfill()
test["oil_price"] = test["oil_price"].ffill().bfill()

# Stores
train = train.merge(stores, on="store_nbr", how="left")
test = test.merge(stores, on="store_nbr", how="left")

# Holidays
national_holidays = holidays[
    (holidays["locale"] == "National") &
    (holidays["type"] != "Work Day") &
    (holidays["transferred"] == False)
]["date"].unique()
train["is_holiday"] = train["date"].isin(national_holidays).astype(int)
test["is_holiday"] = test["date"].isin(national_holidays).astype(int)

# Combine for feature engineering
test["log_sales"] = np.nan
test["sales"] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Features
combined = add_time_features(combined, "date")
combined["store_type_encoded"] = combined["type"].astype("category").cat.codes
combined["is_payday"] = ((combined["day_of_month"] == 15) | combined["is_month_end"]).astype(int)
combined["oil_lag_14"] = combined["oil_price"].shift(14)

# Lag and rolling features per group
combined = add_lag_features(combined, "log_sales", ["store_nbr", "family"], lags=[16, 21, 28, 35, 42, 56])
combined = add_rolling_features(combined, "log_sales", ["store_nbr", "family"], windows=[7, 14, 28], min_shift=16)

FEATURES = [
    "store_nbr", "store_type_encoded", "cluster", "onpromotion",
    "oil_price", "oil_lag_14", "is_holiday",
    "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_payday", "quarter", "year",
    "log_sales_lag_16", "log_sales_lag_21", "log_sales_lag_28",
    "log_sales_lag_35", "log_sales_lag_42", "log_sales_lag_56",
    "log_sales_roll_mean_7", "log_sales_roll_mean_14", "log_sales_roll_mean_28",
    "log_sales_roll_std_7",
]

TARGET = "log_sales"

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

# Train per family
families = combined["family"].unique()
print(f"\nTraining {len(families)} per-family models...")

all_val_preds = []
all_val_actuals = []
test_predictions = pd.DataFrame({"id": combined[combined["date"] >= "2017-08-16"]["id"].values})
test_predictions["log_pred"] = 0.0

for i, family in enumerate(sorted(families)):
    fam_data = combined[combined["family"] == family].copy()

    fam_train = fam_data[fam_data["date"] <= "2017-07-31"].dropna(subset=[TARGET]).dropna(subset=FEATURES)
    fam_val = fam_data[(fam_data["date"] >= "2017-08-01") & (fam_data["date"] <= "2017-08-15")].dropna(subset=[TARGET]).dropna(subset=FEATURES)
    fam_test = fam_data[fam_data["date"] >= "2017-08-16"].copy()

    if len(fam_train) < 100:
        print(f"  [{i+1}/{len(families)}] {family}: skipped (too few rows)")
        continue

    X_tr = fam_train[FEATURES]
    y_tr = fam_train[TARGET]
    X_val = fam_val[FEATURES]
    y_val = fam_val[TARGET]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain, num_boost_round=1500,
        valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # Validate
    val_pred = model.predict(X_val)
    all_val_preds.extend(val_pred)
    all_val_actuals.extend(y_val.values)

    # Predict test
    X_test_fam = fam_test[FEATURES].fillna(0)
    fam_test_pred = model.predict(X_test_fam)
    test_predictions.loc[test_predictions["id"].isin(fam_test["id"]), "log_pred"] = fam_test_pred

    if (i + 1) % 10 == 0 or (i + 1) == len(families):
        print(f"  [{i+1}/{len(families)}] {family}: iter={model.best_iteration}")

# Overall validation RMSLE
val_preds_actual = np.expm1(np.array(all_val_preds)).clip(0)
val_actuals_actual = np.expm1(np.array(all_val_actuals)).clip(0)
rmsle = np.sqrt(mean_squared_error(np.log1p(val_actuals_actual), np.log1p(val_preds_actual)))
print(f"\n=== Per-Family Validation RMSLE: {rmsle:.5f} ===")

# Submission
submission = test_predictions[["id"]].copy()
submission["sales"] = np.expm1(test_predictions["log_pred"]).clip(0)
sub_file = SUB_DIR / "per_family_lgbm.csv"
submission.to_csv(sub_file, index=False)
print(f"Saved: {sub_file}")
print(f"=== DONE ===")
