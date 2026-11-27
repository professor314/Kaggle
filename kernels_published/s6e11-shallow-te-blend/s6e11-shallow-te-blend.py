"""Playground S6E11 (credit card fraud) - shallow trees on in-fold target encodings.

Problem: predict is_fraud (1.7% positives). Metric: ROC AUC (only ranking matters).
Results (our run): shallow LightGBM OOF 0.94768, depth-2 XGBoost OOF 0.94772,
rank blend OOF ~0.9480, public LB 0.94849.

Ideas, each measured on our side (see the repo README for the evidence):
 1. The target is close to a SUM of per-column effects, so tiny trees (4 leaves / depth 2)
    beat deeper ones (31 leaves: 0.94425 vs 4 leaves: 0.94565 on fold 0).
 2. Target-encode every column by its EXACT value (value -> fraud rate). A tiny tree can then
    read a whole per-value curve from one column. The encoder must be fitted INSIDE each
    training fold (sklearn TargetEncoder cross-fits internally), or the CV score leaks labels.
 3. The competition data is synthetic, generated from the public dataset
    uditjain13/credit-card-fraud-detection-2026. Its rows are appended to TRAINING folds only.
 4. Average by rank, not probability: AUC only needs order.
Repo: https://github.com/professor314/Kaggle (competitions/playground-s6e11).
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

SEED, N_SPLITS = 42, 5
IN = Path("/kaggle/input")
train_csv = next(IN.rglob("train.csv"))                      # mount path differs between images
DATA = train_csv.parent
ORIG = next(IN.rglob("credit_card_fraud_2026.csv"))          # attached dataset (original data)


def engineered(d):
    """A few ratio/interaction features. They add little (+0.0003 AUC) but are cheap."""
    d = d.copy()
    d["amt_to_balance"] = d["amount_usd"] / (d["account_balance_usd"].abs() + 1)
    d["amt_x_dist"] = d["amount_usd"] * d["distance_from_home_km"]
    d["vel_x_dist"] = d["velocity_score"] * d["distance_from_home_km"]
    d["risk_x_amt"] = d["merchant_risk_score"] * np.log1p(d["amount_usd"])
    d["hour_sin"] = np.sin(2 * np.pi * d["time_of_day_hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["time_of_day_hour"] / 24)
    flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
             "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
    d["flag_sum"] = d[flags].sum(axis=1)
    return d


tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
og = pd.read_csv(ORIG).drop(columns=["transaction_id"])
y, yo = tr["is_fraud"].astype(int).values, og["is_fraud"].astype(int).values
feats = [c for c in te.columns if c != "id"]
ids = te["id"]
tr, te, og = tr[feats].copy(), te[feats].copy(), og[feats].copy()
for d in (tr, te, og):
    for c in feats:
        if d[c].dtype == bool:
            d[c] = d[c].astype(int)
for c in [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]:   # shared codes
    codes = pd.concat([tr[c], te[c], og[c]]).astype("category").cat.codes.values
    tr[c], te[c], og[c] = codes[:len(tr)], codes[len(tr):len(tr) + len(te)], codes[len(tr) + len(te):]
base = list(tr.columns)
X, Xt, Xo = engineered(tr), engineered(te), engineered(og)

# Exact-value keys; continuous columns rounded to 1 decimal so rare values pool together.
allx = pd.concat([tr, og, te], ignore_index=True)
K = pd.DataFrame({f"k_{c}": (np.round(allx[c], 1) if allx[c].nunique() > 200 else allx[c]) for c in base})
K = K.apply(lambda s: pd.factorize(s)[0]).astype("int32")
n, no = len(X), len(Xo)
Ktr, Kor, Kte = K.iloc[:n].reset_index(drop=True), K.iloc[n:n + no].reset_index(drop=True), K.iloc[n + no:].reset_index(drop=True)

oof = {"lgb": np.zeros(n), "xgb": np.zeros(n)}
pred = {"lgb": np.zeros(len(Xt)), "xgb": np.zeros(len(Xt))}
for k, (a, b) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y)):
    Ka, ya = pd.concat([Ktr.iloc[a], Kor], ignore_index=True), np.r_[y[a], yo]
    enc = TargetEncoder(target_type="binary", smooth="auto", random_state=SEED)   # fitted on training rows only
    Ea = pd.DataFrame(enc.fit_transform(Ka, ya), columns=Ka.columns).add_prefix("te_")
    Eb = pd.DataFrame(enc.transform(Ktr.iloc[b]), columns=Ka.columns).add_prefix("te_")
    Et = pd.DataFrame(enc.transform(Kte), columns=Ka.columns).add_prefix("te_")
    Xa = pd.concat([pd.concat([X.iloc[a], Xo], ignore_index=True), Ea], axis=1)
    Xb = pd.concat([X.iloc[b].reset_index(drop=True), Eb], axis=1)
    Xte = pd.concat([Xt.reset_index(drop=True), Et], axis=1)
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=4, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                           random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(Xa, ya, eval_set=[(Xb, y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    oof["lgb"][b] = m.predict_proba(Xb)[:, 1]; pred["lgb"] += m.predict_proba(Xte)[:, 1] / N_SPLITS
    m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.5,
                          min_child_weight=20, reg_lambda=5.0, tree_method="hist", eval_metric="auc",
                          early_stopping_rounds=200, random_state=SEED, n_jobs=-1)
    m.fit(Xa, ya, eval_set=[(Xb, y[b])], verbose=False)
    oof["xgb"][b] = m.predict_proba(Xb)[:, 1]; pred["xgb"] += m.predict_proba(Xte)[:, 1] / N_SPLITS
    print(f"fold {k}: lgb {roc_auc_score(y[b], oof['lgb'][b]):.5f} xgb {roc_auc_score(y[b], oof['xgb'][b]):.5f}", flush=True)

r = lambda v: rankdata(v) / len(v)
for name in oof:
    print(f"{name} OOF AUC {roc_auc_score(y, oof[name]):.5f}")
print(f"blend OOF AUC {roc_auc_score(y, 0.5 * r(oof['lgb']) + 0.5 * r(oof['xgb'])):.5f}")
final = 0.5 * r(pred["lgb"]) + 0.5 * r(pred["xgb"])
pd.DataFrame({"id": ids, "is_fraud": final}).to_csv("/kaggle/working/submission.csv", index=False)
print("wrote submission.csv", final.shape)