"""Home Data — Full feature pipeline (ported from House Prices best submission).

Uses 95 features: log-transform skewed, ordinal encode quality features,
polynomial interactions, stacking Ridge+Lasso+GBM.
"""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\home-data-ml\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\home-data-ml\submissions")

print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

y = np.log1p(train["SalePrice"])
train_ids = train["Id"]
test_ids = test["Id"]

# Combine for consistent encoding
combined = pd.concat([train.drop("SalePrice", axis=1), test], ignore_index=True)
combined.drop("Id", axis=1, inplace=True)

# --- Feature Engineering ---
# Engineered features
combined["TotalSF"] = combined["TotalBsmtSF"].fillna(0) + combined["1stFlrSF"] + combined["2ndFlrSF"]
combined["TotalBath"] = (
    combined["FullBath"] + 0.5 * combined["HalfBath"] +
    combined["BsmtFullBath"].fillna(0) + 0.5 * combined["BsmtHalfBath"].fillna(0)
)
combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
combined["YearsSinceRemodel"] = combined["YrSold"] - combined["YearRemodAdd"]
combined["TotalPorchSF"] = (
    combined["OpenPorchSF"] + combined["EnclosedPorch"] +
    combined["3SsnPorch"] + combined["ScreenPorch"]
)
combined["HasBasement"] = (combined["TotalBsmtSF"] > 0).astype(int)
combined["HasGarage"] = (combined["GarageArea"].fillna(0) > 0).astype(int)
combined["HasPool"] = (combined["PoolArea"] > 0).astype(int)

# Polynomial features
combined["OverallQual_sq"] = combined["OverallQual"] ** 2
combined["TotalSF_sq"] = combined["TotalSF"] ** 2
combined["GrLivArea_sq"] = combined["GrLivArea"] ** 2
combined["OverallQual_x_TotalSF"] = combined["OverallQual"] * combined["TotalSF"]
combined["OverallQual_x_GrLivArea"] = combined["OverallQual"] * combined["GrLivArea"]

# Ordinal encoding for quality features
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
quality_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
for col in quality_cols:
    combined[col] = combined[col].map(quality_map).fillna(0)

# Label encode remaining categoricals
label_cols = combined.select_dtypes(include=["object"]).columns
for col in label_cols:
    combined[col] = combined[col].fillna("None")
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Fill remaining NaN with median
combined = combined.fillna(combined.median())

# Log-transform skewed numeric features
numeric_cols = combined.select_dtypes(include=[np.number]).columns
skewed_feats = numeric_cols[combined[numeric_cols].apply(lambda x: skew(x.dropna())).abs() > 0.75]
combined[skewed_feats] = np.log1p(combined[skewed_feats].clip(lower=0))

# Remove outliers indicator (GrLivArea > 4000 in original scale)
# Already log-transformed, just proceed

# Split back
X_train = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]

print(f"  Features: {X_train.shape[1]}")

# --- Cross-Validation ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

gbm = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8, random_state=42
)
ridge = Ridge(alpha=10.0)
lasso = Lasso(alpha=0.0005, max_iter=10000)

gbm_scores = -cross_val_score(gbm, X_train, y, cv=cv, scoring="neg_root_mean_squared_error")
ridge_scores = -cross_val_score(ridge, X_train, y, cv=cv, scoring="neg_root_mean_squared_error")
lasso_scores = -cross_val_score(lasso, X_train, y, cv=cv, scoring="neg_root_mean_squared_error")

print(f"  GBM:   {gbm_scores.mean():.5f} +/- {gbm_scores.std():.4f}")
print(f"  Ridge: {ridge_scores.mean():.5f} +/- {ridge_scores.std():.4f}")
print(f"  Lasso: {lasso_scores.mean():.5f} +/- {lasso_scores.std():.4f}")
print(f"  Stack: ~{(gbm_scores.mean() + ridge_scores.mean() + lasso_scores.mean()) / 3:.5f}")

# --- Train Final and Stack ---
gbm.fit(X_train, y)
ridge.fit(X_train, y)
lasso.fit(X_train, y)

pred_gbm = gbm.predict(X_test)
pred_ridge = ridge.predict(X_test)
pred_lasso = lasso.predict(X_test)

# Weighted stack (lasso tends to be best on this dataset)
pred_stack = 0.3 * pred_gbm + 0.35 * pred_ridge + 0.35 * pred_lasso

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": np.expm1(pred_stack).clip(0),
})

sub_file = SUB_DIR / "full_stack_v2.csv"
submission.to_csv(sub_file, index=False)
print(f"\nSaved: {sub_file}")
print(f"Price range: ${submission['SalePrice'].min():.0f} - ${submission['SalePrice'].max():.0f}")
print(f"=== DONE ===")
