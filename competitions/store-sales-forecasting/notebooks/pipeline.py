"""
Store Sales - Time Series Forecasting
Competition: store-sales-time-series-forecasting
Task: Predict store sales (regression, time series)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

COMPETITION_SLUG = "store-sales-time-series-forecasting"

print("=" * 60)
print("COMPETITION 2: Store Sales Time Series Forecasting")
print("=" * 60)

# Check if data exists
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(train_path):
    print(f"\nSKIPPED: Need to join competition at kaggle.com/competitions/{COMPETITION_SLUG}/rules")
    print("Data files not found - download returned 403 (rules not accepted).")
    sys.exit(0)

# Load data
print("\n[1/6] Loading data...")
train = pd.read_csv(train_path, parse_dates=['date'])
test = pd.read_csv(test_path, parse_dates=['date'])
print(f"  Train shape: {train.shape}")
print(f"  Test shape: {test.shape}")
print(f"  Train date range: {train['date'].min()} to {train['date'].max()}")
print(f"  Test date range: {test['date'].min()} to {test['date'].max()}")
print(f"  Unique stores: {train['store_nbr'].nunique()}, Unique families: {train['family'].nunique()}")

# Encode family (categorical)
print("\n[2/6] Encoding categoricals...")
le_family = LabelEncoder()
train['family_encoded'] = le_family.fit_transform(train['family'])
test['family_encoded'] = le_family.transform(test['family'])

# Feature engineering
print("\n[3/6] Feature engineering (time-series features)...")

def create_time_features(df):
    """Create date-based features."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    return df

train = create_time_features(train)
test = create_time_features(test)

# Lag features - for time series, we need to be careful about leakage.
# We create lag features based on historical data only.
print("\n[4/6] Creating lag features (7, 14, 28 days)...")

# Sort by store, family, date for lag computation
train = train.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

# Create lag features per store-family group
# Since test starts at 2017-08-16 and train ends 2017-08-15, lags of 7+ are safe
for lag in [7, 14, 28]:
    col_name = f'sales_lag_{lag}'
    train[col_name] = train.groupby(['store_nbr', 'family'])['sales'].shift(lag)

# Rolling mean features
for window in [7, 14, 28]:
    col_name = f'sales_rolling_mean_{window}'
    train[col_name] = train.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# For test data, compute lags from the tail of train
print("  Computing lag features for test set...")
# Get the last 28 days of train for each store-family to compute test lags
train_tail = train[train['date'] >= train['date'].max() - pd.Timedelta(days=28)].copy()

# Build a lookup: for each (store, family, date), what was the sales?
sales_lookup = train.set_index(['store_nbr', 'family', 'date'])['sales'].to_dict()

# For test rows, compute lags by looking back into train
for lag in [7, 14, 28]:
    col_name = f'sales_lag_{lag}'
    test[col_name] = test.apply(
        lambda row: sales_lookup.get(
            (row['store_nbr'], row['family'], row['date'] - pd.Timedelta(days=lag)),
            np.nan
        ), axis=1
    )

# Rolling means for test - use the last N days from train
for window in [7, 14, 28]:
    col_name = f'sales_rolling_mean_{window}'
    values = []
    for _, row in test.iterrows():
        lookback_sales = []
        for d in range(1, window + 1):
            lookup_date = row['date'] - pd.Timedelta(days=d)
            val = sales_lookup.get((row['store_nbr'], row['family'], lookup_date), np.nan)
            if not np.isnan(val):
                lookback_sales.append(val)
        values.append(np.mean(lookback_sales) if lookback_sales else np.nan)
    test[col_name] = values

# Define feature columns
feature_cols = [
    'store_nbr', 'family_encoded', 'onpromotion',
    'day_of_week', 'month', 'year', 'day_of_month',
    'is_weekend', 'week_of_year', 'is_month_start', 'is_month_end',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_28',
]

# Drop rows with NaN in features (early rows without lag history)
print(f"\n  Features: {feature_cols}")
train_clean = train.dropna(subset=feature_cols + ['sales']).copy()
print(f"  Training rows after dropping NaN lags: {len(train_clean):,} (from {len(train):,})")

# Use only last year of data for efficiency (recent patterns most relevant)
cutoff_date = train_clean['date'].max() - pd.Timedelta(days=365)
train_recent = train_clean[train_clean['date'] >= cutoff_date].copy()
print(f"  Using data from {cutoff_date.date()} onwards: {len(train_recent):,} rows")

X_train = train_recent[feature_cols].values
y_train = train_recent['sales'].values

# Clip negative sales to 0
y_train = np.clip(y_train, 0, None)

# Train model
print("\n[5/6] Training HistGradientBoostingRegressor...")
model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=8,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
)

# Quick validation on time split
print("  Running time-series validation...")
tscv = TimeSeriesSplit(n_splits=3)
val_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model.fit(X_tr, y_tr)
    y_pred = np.clip(model.predict(X_val), 0, None)
    # RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_val + 1, y_pred + 1))
    val_scores.append(rmsle)
    print(f"    Fold {fold+1}: RMSLE = {rmsle:.4f}")

print(f"  Mean RMSLE: {np.mean(val_scores):.4f} (+/- {np.std(val_scores):.4f})")

# Retrain on all data
print("\n  Retraining on full training data...")
model.fit(X_train, y_train)

# Generate predictions
print("\n[6/6] Generating submission...")
X_test = test[feature_cols].values
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, None)  # Sales can't be negative

submission = pd.DataFrame({
    'id': test['id'],
    'sales': predictions
})
submission_path = os.path.join(SUB_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"  Submission saved to: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"  Predictions stats: mean={predictions.mean():.2f}, "
      f"std={predictions.std():.2f}, min={predictions.min():.2f}, max={predictions.max():.2f}")

# Submit
print("\n  Submitting to Kaggle...")
submit_msg = f"Auto-pipeline: HGB, RMSLE={np.mean(val_scores):.4f}, lag+rolling features"
exit_code = os.system(
    f'kaggle competitions submit -c {COMPETITION_SLUG} '
    f'-f "{submission_path}" -m "{submit_msg}"'
)
if exit_code == 0:
    print("  ✓ Submission successful!")
else:
    print("  ✗ Submission failed (may need to accept competition rules)")

print("\n" + "=" * 60)
print("COMPETITION 2 COMPLETE")
print("=" * 60)
