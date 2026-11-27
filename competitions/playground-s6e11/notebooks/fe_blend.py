"""Playground S6E11 (credit card fraud) - feature engineering + LGBM/XGB blend.

Problem: predict is_fraud (0.0167 positive rate). Metric looks AUC-like (LB tracked OOF in Exp 1).
Baseline (Exp 1): plain LightGBM, OOF AUC 0.94594, LB 0.94673. Gap to top ~0.008.

Approach (Exp 2):
  1. Ratio / interaction features. Fraud is about *unusual* behavior, so we compare a
     transaction to the account's own scale (amount vs balance) and combine risk signals.
     Trees can learn interactions, but explicit ratios save depth and help with limited trees.
  2. Cyclic encoding of hour. 23h and 0h are neighbors; the raw integer hides that.
  3. Count of risk flags. Several weak binary signals add up; a sum is easy for trees to split on.
  4. Two different boosters (LightGBM, XGBoost) averaged by rank. S6E9 lesson: correlated
     boosters blend poorly, so we also vary the feature set (XGB sees categorical codes as-is).
Run: C:\\Users\\profe\\anaconda3\\python.exe fe_blend.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "playground-s6e11" / "data"
SUB = REPO / "competitions" / "playground-s6e11" / "submissions"
SEED, N_SPLITS = 42, 5


def load():
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].astype(int).values
    ids = te["id"]
    tr, te = tr.drop(columns=["id", "is_fraud"]), te.drop(columns=["id"])
    # bools -> ints, text -> shared category codes (pandas 3: 'str' dtype, not object)
    for d in (tr, te):
        for c in d.columns:
            if d[c].dtype == bool:
                d[c] = d[c].astype(int)
    cats = [c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
    for c in cats:
        codes = pd.concat([tr[c], te[c]]).astype("category").cat.codes
        tr[c], te[c] = codes.iloc[:len(tr)].values, codes.iloc[len(tr):].values
    return tr, te, y, ids, cats


def add_features(d):
    d = d.copy()
    eps = 1e-6
    d["amt_to_balance"] = d["amount_usd"] / (d["account_balance_usd"].abs() + 1)
    d["amt_per_txn24"] = d["amount_usd"] * d["txn_count_last_24h"]
    d["amt_x_dist"] = d["amount_usd"] * d["distance_from_home_km"]
    d["vel_x_dist"] = d["velocity_score"] * d["distance_from_home_km"]
    d["amt_per_hour"] = d["amount_usd"] / (d["hours_since_last_txn"] + 0.1)
    d["risk_x_amt"] = d["merchant_risk_score"] * np.log1p(d["amount_usd"])
    d["card_age_years"] = d["card_age_months"] / 12.0
    d["hour_sin"] = np.sin(2 * np.pi * d["time_of_day_hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["time_of_day_hour"] / 24)
    d["night"] = ((d["time_of_day_hour"] <= 5) | (d["time_of_day_hour"] >= 23)).astype(int)
    flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    d["flag_sum"] = d[flags].sum(axis=1)
    d["flag_sum_cvv"] = d["flag_sum"] + (d["cvv_retry_count"] > 0).astype(int)
    return d


def run_cv(name, make, fit, X, y, Xt):
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof, pred = np.zeros(len(X)), np.zeros(len(Xt))
    for k, (a, b) in enumerate(cv.split(X, y)):
        m = make()
        fit(m, X.iloc[a], y[a], X.iloc[b], y[b])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
        print(f"  {name} fold {k}: {roc_auc_score(y[b], oof[b]):.5f}", flush=True)
    print(f"{name} OOF AUC {roc_auc_score(y, oof):.5f} AUPRC {average_precision_score(y, oof):.5f}", flush=True)
    return oof, pred


def main():
    tr, te, y, ids, cats = load()
    X, Xt = add_features(tr), add_features(te)
    print(f"features: {X.shape[1]}", flush=True)

    lgb_make = lambda: lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, min_child_samples=60,
        reg_lambda=5.0, random_state=SEED, verbose=-1, n_jobs=-1)
    lgb_fit = lambda m, a, ya, b, yb: m.fit(a, ya, eval_set=[(b, yb)],
        callbacks=[lgb.early_stopping(150, verbose=False)])
    xgb_make = lambda: xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5, reg_lambda=5.0,
        tree_method="hist", eval_metric="auc", early_stopping_rounds=150,
        random_state=SEED, n_jobs=-1)
    xgb_fit = lambda m, a, ya, b, yb: m.fit(a, ya, eval_set=[(b, yb)], verbose=False)

    o1, p1 = run_cv("lgb_fe", lgb_make, lgb_fit, X, y, Xt)
    o2, p2 = run_cv("xgb_fe", xgb_make, xgb_fit, X, y, Xt)

    r = lambda v: rankdata(v) / len(v)
    oof, pred = (r(o1) + r(o2)) / 2, (r(p1) + r(p2)) / 2
    print(f"BLEND OOF AUC {roc_auc_score(y, oof):.5f}", flush=True)

    SUB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(SUB / "fe_blend.csv", index=False)
    np.save(SUB / "fe_blend_oof.npy", oof)
    print("wrote fe_blend.csv", flush=True)


if __name__ == "__main__":
    main()
