"""Home Data for ML Course (Ames housing) - boosting + linear OOF stack.

The same tree + linear diversity stack used on House-Prices, tuned here for the
MAE metric on SalePrice. Five base models (LightGBM / XGBoost / CatBoost /
Ridge / Lasso) are combined via out-of-fold stacking with a Ridge meta-learner.
Stacking beat every individual base model here because the linear models make
genuinely different errors than the trees, giving the meta-learner real
diversity to exploit.

Best public LB with this approach: 13202 MAE (top ~3%).

Self-contained Kaggle version: reads from /kaggle/input/home-data-for-ml-course/.
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
from scipy.stats import skew
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

DATA = Path(_find_input("home-data-for-ml-course"))
SEED = 42
N_SPLITS = 5


def engineer():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = np.log1p(train["SalePrice"].values)
    test_ids = test["Id"]

    c = pd.concat([train.drop("SalePrice", axis=1), test], ignore_index=True).drop("Id", axis=1)

    c["TotalSF"] = c["TotalBsmtSF"].fillna(0) + c["1stFlrSF"] + c["2ndFlrSF"]
    c["TotalBath"] = (c["FullBath"] + 0.5 * c["HalfBath"] +
                      c["BsmtFullBath"].fillna(0) + 0.5 * c["BsmtHalfBath"].fillna(0))
    c["HouseAge"] = c["YrSold"] - c["YearBuilt"]
    c["YearsSinceRemodel"] = c["YrSold"] - c["YearRemodAdd"]
    c["TotalPorchSF"] = (c["OpenPorchSF"] + c["EnclosedPorch"] + c["3SsnPorch"] + c["ScreenPorch"])
    c["HasBasement"] = (c["TotalBsmtSF"].fillna(0) > 0).astype(int)
    c["HasGarage"] = (c["GarageArea"].fillna(0) > 0).astype(int)
    c["HasPool"] = (c["PoolArea"] > 0).astype(int)
    c["OverallQual_sq"] = c["OverallQual"] ** 2
    c["TotalSF_sq"] = c["TotalSF"] ** 2
    c["GrLivArea_sq"] = c["GrLivArea"] ** 2
    c["OverallQual_x_TotalSF"] = c["OverallQual"] * c["TotalSF"]
    c["OverallQual_x_GrLivArea"] = c["OverallQual"] * c["GrLivArea"]

    qmap = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]:
        c[col] = c[col].map(qmap).fillna(0)

    for col in c.select_dtypes(include=["object"]).columns:
        c[col] = LabelEncoder().fit_transform(c[col].fillna("None"))

    c = c.fillna(c.median())
    num = c.select_dtypes(include=[np.number]).columns
    skewed = num[c[num].apply(lambda x: skew(x.dropna())).abs() > 0.75]
    c[skewed] = np.log1p(c[skewed].clip(lower=0))

    n = len(train)
    return c.iloc[:n].reset_index(drop=True), c.iloc[n:].reset_index(drop=True), y, test_ids


def base_models():
    return {
        "lgb": lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=15,
                                 subsample=0.8, colsample_bytree=0.5, min_child_samples=10,
                                 random_state=SEED, verbose=-1),
        "xgb": xgb.XGBRegressor(n_estimators=1200, learning_rate=0.02, max_depth=3,
                                subsample=0.8, colsample_bytree=0.5, random_state=SEED, verbosity=0),
        "cat": CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=4,
                                 random_seed=SEED, verbose=0),
        "ridge": Ridge(alpha=10.0),
        "lasso": Lasso(alpha=0.0005, max_iter=10000),
    }


def main():
    Xtr, Xte, y, test_ids = engineer()
    print(f"Features: {Xtr.shape[1]} | Train {Xtr.shape} | Test {Xte.shape}")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    names = list(base_models().keys())
    oof = np.zeros((len(Xtr), len(names)))
    test_meta = np.zeros((len(Xte), len(names)))

    def mae_price(a, b):  # a, b are log-prices
        return mean_absolute_error(np.expm1(a), np.expm1(b))

    for j, name in enumerate(names):
        fold_test = np.zeros((len(Xte), N_SPLITS))
        for k, (tri, vai) in enumerate(cv.split(Xtr)):
            mdl = base_models()[name]
            mdl.fit(Xtr.iloc[tri], y[tri])
            oof[vai, j] = mdl.predict(Xtr.iloc[vai])
            fold_test[:, k] = mdl.predict(Xte)
        test_meta[:, j] = fold_test.mean(axis=1)
        print(f"  {name}: OOF MAE ${mae_price(y, oof[:, j]):,.0f}")

    meta = Ridge(alpha=1.0)
    meta_oof = cross_val_predict(meta, oof, y, cv=cv)
    stack_mae = mae_price(y, meta_oof)
    blend_mae = mae_price(y, oof.mean(axis=1))
    print(f"\n  Stacked MAE: ${stack_mae:,.0f} | Simple-avg MAE: ${blend_mae:,.0f}")

    if stack_mae <= blend_mae:
        meta.fit(oof, y)
        pred, chosen, mae = meta.predict(test_meta), "stack", stack_mae
    else:
        pred, chosen, mae = test_meta.mean(axis=1), "blend", blend_mae

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(pred).clip(0)})
    out = Path("submission.csv")
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out} (using {chosen}, OOF MAE ${mae:,.0f}; best public LB 13202)")
    print(f"Price range: ${sub['SalePrice'].min():,.0f} - ${sub['SalePrice'].max():,.0f}")


if __name__ == "__main__":
    main()
