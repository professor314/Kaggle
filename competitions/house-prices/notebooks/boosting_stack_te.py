"""House Prices (Ames) - boosting ensemble + OOF stack + target encoding.

Extends boosting_stack.py with the documented next lever: leakage-safe
out-of-fold target encoding of high-cardinality categoricals (Neighborhood,
MSSubClass, Exterior, etc.), plus a weighted stack. Everything else matches
the proven 92-feature pipeline. Fully offline.

Prior best: 0.12345 RMSLE (equal-weight tree+linear blend), PLATEAUED.
Goal: break below ~0.123 via Neighborhood target encoding.

Data read from local kaggle-data (not the repo). Run with anaconda python:
    & "C:\\Users\\profe\\anaconda3\\python.exe" competitions\\house-prices\\notebooks\\boosting_stack_te.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\house-prices\data")
SUB = Path(__file__).resolve().parents[1] / "submissions"
SEED = 42
N_SPLITS = 5

# High-cardinality / high-signal categoricals to target-encode (raw string cols).
TE_COLS = ["Neighborhood", "MSSubClass", "Exterior1st", "Exterior2nd",
           "SaleType", "Condition1"]


def smooth_target_encode(train_vals, y, test_vals, cv, smoothing=10.0):
    """Leakage-safe OOF target encoding for one column.

    Returns (oof_encoded_train, encoded_test). Train encoding uses only
    out-of-fold means; test uses full-train means. Smoothed toward global mean.
    """
    train_vals = pd.Series(train_vals).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    global_mean = y.mean()
    oof = np.zeros(len(train_vals))

    for tri, vai in cv.split(train_vals):
        tr_v, tr_y = train_vals.iloc[tri], y.iloc[tri]
        agg = tr_y.groupby(tr_v).agg(["mean", "count"])
        smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        oof[vai] = train_vals.iloc[vai].map(smooth).fillna(global_mean).values

    # Test: full-train smoothed means.
    agg = y.groupby(train_vals).agg(["mean", "count"])
    smooth = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    test_enc = pd.Series(test_vals).map(smooth).fillna(global_mean).values
    return oof, test_enc


def engineer():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].reset_index(drop=True)
    y = np.log1p(train["SalePrice"].values)
    test_ids = test["Id"]

    # Keep raw copies of the TE columns before encoding.
    n = len(train)
    raw_cat = pd.concat([train[TE_COLS].astype(str), test[TE_COLS].astype(str)],
                        ignore_index=True)

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

    Xtr = c.iloc[:n].reset_index(drop=True)
    Xte = c.iloc[n:].reset_index(drop=True)
    raw_tr = raw_cat.iloc[:n].reset_index(drop=True)
    raw_te = raw_cat.iloc[n:].reset_index(drop=True)
    return Xtr, Xte, y, test_ids, raw_tr, raw_te


def base_models():
    return {
        "lgb": lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=15,
                                 subsample=0.8, colsample_bytree=0.5, min_child_samples=10,
                                 random_state=SEED, verbose=-1, n_jobs=-1),
        "xgb": xgb.XGBRegressor(n_estimators=1200, learning_rate=0.02, max_depth=3,
                                subsample=0.8, colsample_bytree=0.5, random_state=SEED,
                                verbosity=0, n_jobs=-1),
        "cat": CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=4,
                                 random_seed=SEED, verbose=0, thread_count=-1),
        "ridge": Ridge(alpha=10.0),
        "lasso": Lasso(alpha=0.0005, max_iter=10000),
    }


def rmsle(a, b):
    return np.sqrt(mean_squared_error(a, b))


def add_target_encoding(Xtr, Xte, y, raw_tr, raw_te, cv):
    """Append OOF-target-encoded columns for each TE col."""
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    for col in TE_COLS:
        oof_enc, test_enc = smooth_target_encode(raw_tr[col].values, y, raw_te[col].values, cv)
        Xtr[f"{col}_te"] = oof_enc
        Xte[f"{col}_te"] = test_enc
    return Xtr, Xte


def run_stack(Xtr, Xte, y, cv, label):
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
        print(f"  [{label}] {name}: OOF RMSLE {rmsle(y, oof[:, j]):.5f}")

    meta = Ridge(alpha=1.0)
    meta_oof = cross_val_predict(meta, oof, y, cv=cv)
    stack_r = rmsle(y, meta_oof)
    blend_r = rmsle(y, oof.mean(axis=1))
    print(f"  [{label}] Stack RMSLE {stack_r:.5f} | Blend RMSLE {blend_r:.5f}")

    if stack_r <= blend_r:
        meta.fit(oof, y)
        return meta.predict(test_meta), min(stack_r, blend_r), "stack"
    return test_meta.mean(axis=1), blend_r, "blend"


def main():
    Xtr, Xte, y, test_ids, raw_tr, raw_te = engineer()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    print(f"Base features: {Xtr.shape[1]} | Train {Xtr.shape} | Test {Xte.shape}")

    # Baseline (no TE) for direct comparison.
    _, base_r, _ = run_stack(Xtr, Xte, y, cv, "no-TE")

    # With OOF target encoding.
    Xtr2, Xte2 = add_target_encoding(Xtr, Xte, y, raw_tr, raw_te, cv)
    print(f"With TE features: {Xtr2.shape[1]}")
    pred, te_r, chosen = run_stack(Xtr2, Xte2, y, cv, "with-TE")

    print(f"\n=== RESULT ===")
    print(f"  Prior best LB:   0.12345 (plateaued)")
    print(f"  No-TE OOF:       {base_r:.5f}")
    print(f"  With-TE OOF:     {te_r:.5f}  ({'IMPROVED' if te_r < base_r else 'no gain'})")

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(pred).clip(0)})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "boosting_stack_te.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out} (using {chosen}, OOF RMSLE {te_r:.5f})")
    # Sanity on prediction distribution.
    print(f"Pred SalePrice: min {sub.SalePrice.min():.0f} "
          f"median {sub.SalePrice.median():.0f} max {sub.SalePrice.max():.0f}")


if __name__ == "__main__":
    main()
