"""Playground S6E11 - 5-fold shallow LGBM (4 leaves) + original rows + in-fold target encoding.

Fold-0 ablation (shallow_orig.py, VERIFIED): 4 leaves 0.94565 vs 31 leaves 0.94425, so the
target is close to additive and small trees win. Original rows added +0.0001.

Here we add the lever the top public notebooks use: target-encode every column by its EXACT
value (each distinct value = one category), fitted INSIDE each training fold with sklearn's
cross-fitted TargetEncoder so a row never sees its own label. A 4-leaf tree can then read a
whole per-value fraud curve from one encoded column (a "shape function").
Writes shallow_te.csv / shallow_te_oof.npy and a rank blend with the previous best.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from fe_blend import SUB, SEED, N_SPLITS  # noqa: E402
from shallow_orig import load_all  # noqa: E402


def keys(df, cols):
    """Integer key per exact value (continuous columns rounded to 1 decimal to pool rare values)."""
    out = {}
    for c in cols:
        v = df[c]
        out[f"k_{c}"] = np.round(v, 1) if v.nunique() > 200 else v
    return pd.DataFrame(out, index=df.index)


def main():
    X, Xt, Xo, y, yo = load_all()
    base = [c for c in X.columns if not c.startswith(("hour_", "amt_", "vel_", "risk_", "flag_", "card_age_y", "night"))]
    allx = pd.concat([X, Xo, Xt], ignore_index=True)
    K = keys(allx, base)
    # factorize jointly so train/orig/test share key ids
    K = K.apply(lambda s: pd.factorize(s)[0]).astype("int32")
    n, no = len(X), len(Xo)
    Ktr, Kor, Kte = K.iloc[:n], K.iloc[n:n + no].reset_index(drop=True), K.iloc[n + no:].reset_index(drop=True)
    Ktr = Ktr.reset_index(drop=True)

    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof, pred = np.zeros(n), np.zeros(len(Xt))
    for k, (a, b) in enumerate(cv.split(X, y)):
        # training set = train fold + original rows; encoder fitted on these rows only
        Ka = pd.concat([Ktr.iloc[a], Kor], ignore_index=True)
        ya = np.r_[y[a], yo]
        enc = TargetEncoder(target_type="binary", smooth="auto", random_state=SEED)
        Ea = pd.DataFrame(enc.fit_transform(Ka, ya), columns=Ka.columns)  # cross-fitted inside
        Eb = pd.DataFrame(enc.transform(Ktr.iloc[b]), columns=Ka.columns)
        Et = pd.DataFrame(enc.transform(Kte), columns=Ka.columns)
        Xa = pd.concat([pd.concat([X.iloc[a], Xo], ignore_index=True), Ea.add_prefix("te_")], axis=1)
        Xb = pd.concat([X.iloc[b].reset_index(drop=True), Eb.add_prefix("te_")], axis=1)
        Xte = pd.concat([Xt.reset_index(drop=True), Et.add_prefix("te_")], axis=1)
        m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=4, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                               random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(Xa, ya, eval_set=[(Xb, y[b])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict_proba(Xb)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / N_SPLITS
        print(f"  fold {k}: {roc_auc_score(y[b], oof[b]):.5f} (iters {m.best_iteration_})", flush=True)
    print(f"shallow_te OOF AUC {roc_auc_score(y, oof):.5f}", flush=True)
    np.save(SUB / "shallow_te_oof.npy", oof)
    ids = pd.read_csv(SUB / "fe_blend.csv")["id"]
    pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(SUB / "shallow_te.csv", index=False)

    r = lambda v: rankdata(v) / len(v)
    prev_oof = 0.3 * r(np.load(SUB / "cat_oof.npy")) + 0.7 * np.load(SUB / "fe_blend_oof.npy")
    prev_te = 0.3 * r(pd.read_csv(SUB / "cat.csv")["is_fraud"].values) + 0.7 * r(pd.read_csv(SUB / "fe_blend.csv")["is_fraud"].values)
    for w in (0.3, 0.5, 0.7, 0.9):
        print(f"blend w_shallow={w}: OOF {roc_auc_score(y, w * r(oof) + (1 - w) * prev_oof):.5f}", flush=True)
    for w in (0.5, 0.7):
        pd.DataFrame({"id": ids, "is_fraud": w * r(pred) + (1 - w) * prev_te}).to_csv(SUB / f"shallow_blend_w{int(w*10)}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
