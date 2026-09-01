"""House Prices (Ames) - boosting stack + feature selection.

The v2 research flagged that the 95-feature stack overfit (CV-LB gap widened
from 3.5% to 10.2%). This variant adds importance-based feature selection
(keep the top-K by aggregated LightGBM gain) to cut noise, then runs the same
tree+linear OOF stack. It compares full vs selected feature sets on OOF RMSLE
and writes the better one.

Run with the DL env (has lgb/xgb/catboost):
    .venv-dl\\Scripts\\python.exe competitions\\house-prices\\notebooks\\boosting_stack_fs.py

Prior best LB: 0.1235 RMSLE.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "house-prices" / "data"
SUB = REPO / "competitions" / "house-prices" / "submissions"
SEED = 42
N_SPLITS = 5
TOP_K = 45  # keep the top-K features by LightGBM gain


def engineer():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].reset_index(drop=True)
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


def select_features(Xtr, y, k):
    """Rank features by LightGBM gain over CV folds, return the top-k names."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gain = np.zeros(Xtr.shape[1])
    for tri, _ in cv.split(Xtr):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.5, min_child_samples=10,
                              random_state=SEED, verbose=-1, importance_type="gain")
        m.fit(Xtr.iloc[tri], y[tri])
        gain += m.feature_importances_
    order = np.argsort(gain)[::-1]
    return list(Xtr.columns[order[:k]])


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


def rmsle(a, b):
    return np.sqrt(mean_squared_error(a, b))


def run_stack(Xtr, Xte, y):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    names = list(base_models().keys())
    oof = np.zeros((len(Xtr), len(names)))
    test_meta = np.zeros((len(Xte), len(names)))
    for j, name in enumerate(names):
        fold_test = np.zeros((len(Xte), N_SPLITS))
        for k, (tri, vai) in enumerate(cv.split(Xtr)):
            mdl = base_models()[name]
            mdl.fit(Xtr.iloc[tri], y[tri])
            oof[vai, j] = mdl.predict(Xtr.iloc[vai])
            fold_test[:, k] = mdl.predict(Xte)
        test_meta[:, j] = fold_test.mean(axis=1)
        print(f"    {name}: OOF RMSLE {rmsle(y, oof[:, j]):.5f}")
    meta = Ridge(alpha=1.0)
    meta_oof = cross_val_predict(meta, oof, y, cv=cv)
    stack_r = rmsle(y, meta_oof)
    blend_r = rmsle(y, oof.mean(axis=1))
    if stack_r <= blend_r:
        meta.fit(oof, y)
        return meta.predict(test_meta), "stack", stack_r
    return test_meta.mean(axis=1), "blend", blend_r


def main():
    Xtr, Xte, y, test_ids = engineer()
    print(f"Full features: {Xtr.shape[1]}")

    print("\n[FULL feature set]")
    pred_full, mode_full, r_full = run_stack(Xtr, Xte, y)
    print(f"  -> {mode_full} OOF RMSLE {r_full:.5f}")

    print(f"\n[SELECTED top-{TOP_K} features]")
    feats = select_features(Xtr, y, TOP_K)
    pred_sel, mode_sel, r_sel = run_stack(Xtr[feats], Xte[feats], y)
    print(f"  -> {mode_sel} OOF RMSLE {r_sel:.5f}")

    if r_sel <= r_full:
        pred, tag, r = pred_sel, f"selected-top{TOP_K}-{mode_sel}", r_sel
    else:
        pred, tag, r = pred_full, f"full-{mode_full}", r_full

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(pred).clip(0)})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "boosting_stack_fs.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Chosen: {tag} (OOF RMSLE {r:.5f}; prior best LB 0.1235)")


if __name__ == "__main__":
    main()
