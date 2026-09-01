"""Home Data — LightGBM + Lasso stack for better MAE."""
import sys
sys.path.insert(0, r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle")

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew

DATA_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\home-data-ml\data")
SUB_DIR = Path(r"c:\Users\profe\OneDrive\Desktop\Sean Obsidian\Projects\Kaggle\competitions\home-data-ml\submissions")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

y = np.log1p(train["SalePrice"])
test_ids = test["Id"]

combined = pd.concat([train.drop("SalePrice", axis=1), test], ignore_index=True)
combined.drop("Id", axis=1, inplace=True)

# Engineered features
combined["TotalSF"] = combined["TotalBsmtSF"].fillna(0) + combined["1stFlrSF"] + combined["2ndFlrSF"]
combined["TotalBath"] = combined["FullBath"] + 0.5*combined["HalfBath"] + combined["BsmtFullBath"].fillna(0) + 0.5*combined["BsmtHalfBath"].fillna(0)
combined["HouseAge"] = combined["YrSold"] - combined["YearBuilt"]
combined["YearsSinceRemodel"] = combined["YrSold"] - combined["YearRemodAdd"]
combined["TotalPorchSF"] = combined["OpenPorchSF"] + combined["EnclosedPorch"] + combined["3SsnPorch"] + combined["ScreenPorch"]
combined["OverallQual_sq"] = combined["OverallQual"] ** 2
combined["TotalSF_sq"] = combined["TotalSF"] ** 2
combined["OverallQual_x_TotalSF"] = combined["OverallQual"] * combined["TotalSF"]
combined["OverallQual_x_GrLivArea"] = combined["OverallQual"] * combined["GrLivArea"]

# Ordinal encode quality
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]:
    combined[col] = combined[col].map(quality_map).fillna(0)

# Label encode categoricals
for col in combined.select_dtypes(include=["object", "string"]).columns:
    combined[col] = combined[col].fillna("None")
    combined[col] = LabelEncoder().fit_transform(combined[col])

combined = combined.fillna(combined.median())

# Log skewed
numeric = combined.select_dtypes(include=[np.number]).columns
skewed_feats = numeric[combined[numeric].apply(lambda x: skew(x.dropna())).abs() > 0.75]
combined[skewed_feats] = np.log1p(combined[skewed_feats].clip(lower=0))

X_train = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]

# 5-fold LightGBM + Lasso blend
cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X_train))
oof_lasso = np.zeros(len(X_train))
test_lgb = np.zeros(len(X_test))
test_lasso = np.zeros(len(X_test))

params = {"objective": "regression", "metric": "mae", "learning_rate": 0.03,
          "num_leaves": 63, "max_depth": 6, "min_child_samples": 10,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5,
          "reg_alpha": 0.1, "reg_lambda": 1.0, "verbosity": -1, "seed": 42}

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(params, dtrain, 3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = model.predict(X_val)
    test_lgb += model.predict(X_test) / 5

    lasso = Lasso(alpha=0.0003, max_iter=10000)
    lasso.fit(X_tr, y_tr)
    oof_lasso[val_idx] = lasso.predict(X_val)
    test_lasso += lasso.predict(X_test) / 5

# Blend
oof_blend = 0.6 * oof_lgb + 0.4 * oof_lasso
test_blend = 0.6 * test_lgb + 0.4 * test_lasso

mae_log = mean_absolute_error(y, oof_blend)
print(f"OOF MAE (log space): {mae_log:.5f}")
# Approximate real MAE
real_preds = np.expm1(oof_blend)
real_actual = np.expm1(y)
mae_real = mean_absolute_error(real_actual, real_preds)
print(f"OOF MAE (real): ${mae_real:.0f}")

submission = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(test_blend).clip(0)})
sub_file = SUB_DIR / "lgbm_lasso_blend.csv"
submission.to_csv(sub_file, index=False)
print(f"Saved: {sub_file}")
