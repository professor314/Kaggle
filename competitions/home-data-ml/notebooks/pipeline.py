"""Home Data for ML Course — Quick pipeline using our House Prices approach.

Same dataset as House Prices (Ames, Iowa), same metric (RMSLE).
Apply our proven strategy: log transform + numeric features + GBM stack.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\home-data-ml\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Kaggle\competitions\home-data-ml\submissions")
SUB_DIR.mkdir(exist_ok=True)

# Load
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"  Train: {train.shape}, Test: {test.shape}")

# Target: log transform
y = np.log1p(train["SalePrice"])

# Combine for feature engineering
train_ids = train["Id"]
test_ids = test["Id"]
combined = pd.concat([train.drop("SalePrice", axis=1), test], ignore_index=True)

# Feature engineering (proven approach from House Prices)
combined["TotalSF"] = combined["TotalBsmtSF"].fillna(0) + combined["1stFlrSF"] + combined["2ndFlrSF"]
combined["TotalBath"] = (
    combined["FullBath"] + 0.5 * combined["HalfBath"] +
    combined["BsmtFullBath"].fillna(0) + 0.5 * combined["BsmtHalfBath"].fillna(0)
)
combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
combined["YearsSinceRemodel"] = combined["YrSold"] - combined["YearRemodAdd"]

# Select numeric features
numeric_features = [
    "OverallQual", "OverallCond", "TotalSF", "GrLivArea", "TotalBath",
    "GarageCars", "HouseAge", "YearsSinceRemodel", "TotalBsmtSF",
    "LotArea", "1stFlrSF", "2ndFlrSF", "GarageArea", "WoodDeckSF",
    "OpenPorchSF", "LotFrontage", "MasVnrArea", "BsmtFinSF1",
    "Fireplaces", "ScreenPorch",
]

X_all = combined[numeric_features].copy()

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_all = pd.DataFrame(imputer.fit_transform(X_all), columns=numeric_features)

# Log-transform skewed features
from scipy.stats import skew
skewed = X_all.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_feats = skewed[skewed > 0.75].index
X_all[skewed_feats] = np.log1p(X_all[skewed_feats])

# Split back
X_train = X_all.iloc[:len(train)]
X_test = X_all.iloc[len(train):]

print(f"  Features: {X_train.shape[1]}")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# GBM
gbm = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8, random_state=42
)
gbm_scores = -cross_val_score(gbm, X_train, y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"  GBM RMSE (log): {gbm_scores.mean():.4f} +/- {gbm_scores.std():.4f}")

# Ridge
ridge = Ridge(alpha=10.0)
ridge_scores = -cross_val_score(ridge, X_train, y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"  Ridge RMSE (log): {ridge_scores.mean():.4f} +/- {ridge_scores.std():.4f}")

# Lasso
lasso = Lasso(alpha=0.001, max_iter=10000)
lasso_scores = -cross_val_score(lasso, X_train, y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"  Lasso RMSE (log): {lasso_scores.mean():.4f} +/- {lasso_scores.std():.4f}")

# Stack: average of all three
print("\nTraining final stack...")
gbm.fit(X_train, y)
ridge.fit(X_train, y)
lasso.fit(X_train, y)

pred_gbm = gbm.predict(X_test)
pred_ridge = ridge.predict(X_test)
pred_lasso = lasso.predict(X_test)
pred_stack = (pred_gbm + pred_ridge + pred_lasso) / 3

# Convert back from log
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": np.expm1(pred_stack).clip(0),
})

sub_file = SUB_DIR / "stack_submission.csv"
submission.to_csv(sub_file, index=False)
print(f"\nSubmission saved: {sub_file}")
print(f"Price range: ${submission['SalePrice'].min():.0f} - ${submission['SalePrice'].max():.0f}")
print(f"Mean: ${submission['SalePrice'].mean():.0f}")
print(f"\n=== DONE ===")
