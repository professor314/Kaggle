"""Store Sales - Per-Family LightGBM (holiday proximity + transaction lags).

A separate LightGBM model is trained per product family (33 families). Features
combine calendar signals, oil prices (Ecuador is petroleum-dependent), national
holiday proximity, per-store transaction lags, and safe (16-day-shifted) sales
lag / rolling-mean features so nothing leaks across the 16-day forecast horizon.

Best public LB with this approach: 0.421 RMSLE.

Self-contained Kaggle version: reads from
/kaggle/input/store-sales-time-series-forecasting/. The small time-series helper
functions (calendar, lag, rolling) are inlined below.
"""
def _find_input(slug):
    """Locate a competition's input dir across Kaggle mount layouts."""
    import os, glob
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{slug} not found. Inputs: {glob.glob('/kaggle/input/*')}")


from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

DATA_DIR = Path(_find_input("store-sales-time-series-forecasting"))


# --- inlined time-series helpers (no external toolkit dependency) ---
def add_time_features(df, date_col="date"):
    dt = pd.to_datetime(df[date_col])
    df = df.copy()
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    return df


def add_lag_features(df, target_col, group_cols, lags):
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, group_cols, windows, min_shift=16):
    df = df.copy()
    grouped = df.groupby(group_cols)[target_col]
    for window in windows:
        shifted = grouped.shift(min_shift)
        rolling = shifted.rolling(window, min_periods=1)
        df[f"{target_col}_roll_mean_{window}"] = rolling.mean().values
        df[f"{target_col}_roll_std_{window}"] = rolling.std().values
    return df


print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
stores = pd.read_csv(DATA_DIR / "stores.csv")
oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])

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

# National holidays flag
national_holidays = holidays[
    (holidays["locale"] == "National") &
    (holidays["type"] != "Work Day") &
    (holidays["transferred"] == False)
]["date"].unique()
train["is_holiday"] = train["date"].isin(national_holidays).astype(int)
test["is_holiday"] = test["date"].isin(national_holidays).astype(int)

# Holiday proximity features
holiday_dates = np.sort(pd.to_datetime(national_holidays))


def holiday_proximity(dates):
    """days_until (>=0) and days_since (>=0) nearest national holiday."""
    d = np.asarray(pd.to_datetime(dates)).astype("datetime64[D]")
    hd = np.asarray(holiday_dates).astype("datetime64[D]")
    until, since = [], []
    for x in d:
        diff = (hd - x).astype("timedelta64[D]").astype(int)
        future = diff[diff >= 0]
        past = diff[diff <= 0]
        until.append(int(future.min()) if len(future) else 999)
        since.append(int(-past.max()) if len(past) else 999)
    return np.array(until), np.array(since)


# Compute proximity once on the unique dates, then map back.
all_dates = pd.concat([train["date"], test["date"]]).drop_duplicates().sort_values()
u, s = holiday_proximity(all_dates)
prox = pd.DataFrame({"date": all_dates.values, "days_until_holiday": u, "days_since_holiday": s})
prox["is_holiday_window"] = ((prox["days_until_holiday"] <= 3) | (prox["days_since_holiday"] <= 3)).astype(int)
train = train.merge(prox, on="date", how="left")
test = test.merge(prox, on="date", how="left")

# Combine
test["log_sales"] = np.nan
test["sales"] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# Transaction lag features (per store, shifted safely)
tx = transactions.sort_values(["store_nbr", "date"]).copy()
tx["log_tx"] = np.log1p(tx["transactions"].clip(lower=0))
tx["tx_lag_16"] = tx.groupby("store_nbr")["log_tx"].shift(16)
tx["tx_roll_mean_7"] = tx.groupby("store_nbr")["log_tx"].shift(16).rolling(7).mean().reset_index(0, drop=True)
combined = combined.merge(tx[["date", "store_nbr", "tx_lag_16", "tx_roll_mean_7"]],
                          on=["date", "store_nbr"], how="left")

# Time + other features
combined = add_time_features(combined, "date")
combined["store_type_encoded"] = combined["type"].astype("category").cat.codes
combined["is_payday"] = ((combined["day_of_month"] == 15) | combined["is_month_end"]).astype(int)
combined["oil_lag_14"] = combined["oil_price"].shift(14)

combined = add_lag_features(combined, "log_sales", ["store_nbr", "family"], lags=[16, 21, 28, 35, 42, 56])
combined = add_rolling_features(combined, "log_sales", ["store_nbr", "family"], windows=[7, 14, 28], min_shift=16)

FEATURES = [
    "store_nbr", "store_type_encoded", "cluster", "onpromotion",
    "oil_price", "oil_lag_14", "is_holiday",
    "days_until_holiday", "days_since_holiday", "is_holiday_window",
    "tx_lag_16", "tx_roll_mean_7",
    "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_payday", "quarter", "year",
    "log_sales_lag_16", "log_sales_lag_21", "log_sales_lag_28",
    "log_sales_lag_35", "log_sales_lag_42", "log_sales_lag_56",
    "log_sales_roll_mean_7", "log_sales_roll_mean_14", "log_sales_roll_mean_28",
    "log_sales_roll_std_7",
]
TARGET = "log_sales"

params = {
    "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
    "num_leaves": 63, "max_depth": 7, "min_child_samples": 20,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "verbosity": -1, "seed": 42,
}

families = combined["family"].unique()
print(f"Training {len(families)} per-family models (+holiday proximity, +transactions)...")

all_val_preds, all_val_actuals = [], []
test_predictions = pd.DataFrame({"id": combined[combined["date"] >= "2017-08-16"]["id"].values})
test_predictions["log_pred"] = 0.0

for i, family in enumerate(sorted(families)):
    fam = combined[combined["family"] == family].copy()
    fam_train = fam[fam["date"] <= "2017-07-31"].dropna(subset=[TARGET]).dropna(subset=FEATURES)
    fam_val = fam[(fam["date"] >= "2017-08-01") & (fam["date"] <= "2017-08-15")].dropna(subset=[TARGET]).dropna(subset=FEATURES)
    fam_test = fam[fam["date"] >= "2017-08-16"].copy()
    if len(fam_train) < 100:
        continue

    dtrain = lgb.Dataset(fam_train[FEATURES], label=fam_train[TARGET])
    dval = lgb.Dataset(fam_val[FEATURES], label=fam_val[TARGET], reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    all_val_preds.extend(model.predict(fam_val[FEATURES]))
    all_val_actuals.extend(fam_val[TARGET].values)
    fam_test_pred = model.predict(fam_test[FEATURES].fillna(0))
    test_predictions.loc[test_predictions["id"].isin(fam_test["id"]), "log_pred"] = fam_test_pred

    if (i + 1) % 11 == 0 or (i + 1) == len(families):
        print(f"  [{i+1}/{len(families)}] {family}: iter={model.best_iteration}")

val_p = np.expm1(np.array(all_val_preds)).clip(0)
val_a = np.expm1(np.array(all_val_actuals)).clip(0)
rmsle = np.sqrt(mean_squared_error(np.log1p(val_a), np.log1p(val_p)))
print(f"\n=== Per-Family Validation RMSLE: {rmsle:.5f} (best public LB 0.421) ===")

submission = test_predictions[["id"]].copy()
submission["sales"] = np.expm1(test_predictions["log_pred"]).clip(0)
submission.to_csv("submission.csv", index=False)
print(f"Saved: submission.csv  rows={len(submission)}")
