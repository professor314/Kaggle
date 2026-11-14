"""Home-Data-ML (Ames) - boosting + linear stack + OOF target encoding.

Same Ames data as House-Prices but scored by MAE on raw SalePrice. Reuses the
92-feature engineering + outlier drop, adds leakage-safe OOF target encoding of
high-cardinality categoricals (the documented next lever), and evaluates in MAE
on dollars (predictions trained on log target, then expm1'd).

Fully offline. Run with anaconda python:
    & "C:\\Users\\profe\\anaconda3\\python.exe" competitions\\home-data-ml\\notebooks\\boosting_stack_te.py

Prior best LB MAE: 13202 (5-model Ridge-meta stack, no TE).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\home-data-ml\data")
SUB = Path(__file__).resolve().parents[1] / "submissions"
SEED = 42
N_SPLITS = 5
TE_COLS = ["Neighborhood", "MSSubClass", "Exterior1st", "Exterior2nd",
           "SaleType", "Condition1"]


def smooth_target_encode(train_vals, y, test_vals, cv, smoothing=10.0):
    train_vals = pd.Series(train_vals).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    gmean = y.mean()
    oof = np.zeros(len(train_vals))
    for tri, vai in cv.split(train_vals):
        tr_v, tr_y = train_vals.iloc[tri], y.iloc[tri]
        agg = tr_y.groupby(tr_v).agg(["mean", "count"])
        sm = (agg["mean"] * agg["count"] + gmean * smoothing) / (agg["count"] + smoothing)
        oof[vai] = train_vals.iloc[vai].map(sm).fillna(gmean).values
    agg = y.groupby(train_vals).agg(["mean", "count"])
    sm = (agg["mean"] * agg["count"] + gmean * smoothing) / (agg["count"] + smoothing)
    test_enc = pd.Series(test_vals).map(sm).fillna(gmean).values
    return oof, test_enc


def engineer():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].reset_index(drop=True)
    y_raw = train["SalePrice"].values.astype(float)
    y_log = np.log1p(y_raw)
    test_ids = test["Id"]
    n = len(train)
    raw_cat = pd.concat([train[TE_COLS].astype(str), test[TE_COLS].astype(str)], ignore_index=True)
    c = pd.concat([train.drop("SalePrice", axis=1), test], ignore_index=True).drop("Id", axis=1)

    c["TotalSF"] = c["TotalBsmtSF"].fillna(0) + c["1stFlrSF"] + c["2ndFlrSF"]
    c["TotalBath"] = (c["FullBath"] + 0.5 * c["HalfBath"] +
                      c["BsmtFullBath"].fillna(0) + 0.5 * c["BsmtHalfBath"].fillna(0))
    c["HouseAge"] = c["YrSold"] - c["YearBuilt"]
    c["YearsSinceRemodel"] = c["YrSold"] - c["YearRemodAdd"]
    c["TotalPorchSF"] = c["OpenPorchSF"] + c["EnclosedPorch"] + c["3SsnPorch"] + c["ScreenPorch"]
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
    return Xtr, Xte, y_log, y_raw, test_ids, raw_tr, raw_te


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


def add_te(Xtr, Xte, y_log, raw_tr, raw_te, cv):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for col in TE_COLS:
        oof_enc, test_enc = smooth_target_encode(raw_tr[col].values, y_log, raw_te[col].values, cv)
        Xtr[f"{col}_te"] = oof_enc
        Xte[f"{col}_te"] = test_enc
    return Xtr, Xte


def run_stack(Xtr, Xte, y_log, y_raw, cv, label):
    names = list(base_models().keys())
    oof = np.zeros((len(Xtr), len(names)))
    test_meta = np.zeros((len(Xte), len(names)))
    for j, name in enumerate(names):
        fold_test = np.zeros((len(Xte), N_SPLITS))
        for k, (tri, vai) in enumerate(cv.split(Xtr)):
            mdl = base_models()[name]
            mdl.fit(Xtr.iloc[tri], y_log[tri])
            oof[vai, j] = mdl.predict(Xtr.iloc[vai])
            fold_test[:, k] = mdl.predict(Xte)
        test_meta[:, j] = fold_test.mean(axis=1)
        mae_j = mean_absolute_error(y_raw, np.expm1(oof[:, j]))
        print(f"  [{label}] {name}: OOF MAE {mae_j:,.0f}")

    # Ridge meta on log preds, plus simple blend. Pick lower MAE (on dollars).
    from sklearn.model_selection import cross_val_predict
    meta = Ridge(alpha=1.0)
    meta_oof = cross_val_predict(meta, oof, y_log, cv=cv)
    stack_mae = mean_absolute_error(y_raw, np.expm1(meta_oof))
    blend_mae = mean_absolute_error(y_raw, np.expm1(oof.mean(axis=1)))
    print(f"  [{label}] Stack MAE {stack_mae:,.0f} | Blend MAE {blend_mae:,.0f}")
    if stack_mae <= blend_mae:
        meta.fit(oof, y_log)
        return np.expm1(meta.predict(test_meta)), stack_mae, "stack"
    return np.expm1(test_meta.mean(axis=1)), blend_mae, "blend"


def main():
    Xtr, Xte, y_log, y_raw, test_ids, raw_tr, raw_te = engineer()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    print(f"Base features: {Xtr.shape[1]} | Train {Xtr.shape} | Test {Xte.shape}")

    _, base_mae, _ = run_stack(Xtr, Xte, y_log, y_raw, cv, "no-TE")
    Xtr2, Xte2 = add_te(Xtr, Xte, y_log, raw_tr, raw_te, cv)
    print(f"With TE features: {Xtr2.shape[1]}")
    pred, te_mae, chosen = run_stack(Xtr2, Xte2, y_log, y_raw, cv, "with-TE")

    print(f"\n=== RESULT ===")
    print(f"  Prior best LB MAE: 13202")
    print(f"  No-TE OOF MAE:     {base_mae:,.0f}")
    print(f"  With-TE OOF MAE:   {te_mae:,.0f}  ({'IMPROVED' if te_mae < base_mae else 'no gain'})")

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": pred.clip(0)})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "boosting_stack_te.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out} (using {chosen})")
    print(f"Pred SalePrice: min {sub.SalePrice.min():,.0f} "
          f"median {sub.SalePrice.median():,.0f} max {sub.SalePrice.max():,.0f}")


if __name__ == "__main__":
    main()
