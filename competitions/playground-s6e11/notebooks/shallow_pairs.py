"""Playground S6E11 - shallow LGBM + in-fold TE on singles AND pairs, 3 seeds averaged.

Builds on shallow_te_full.py (OOF 0.94768). Two levers:
  1. Pair-crossed target encodings among the 10 columns with the strongest per-value signal.
     (Public notebooks measured that crossing ALL 276 pairs hurt; the top-10 subset is the
     compromise.) Pairs let 4-leaf trees see two-column interactions without growing deep.
  2. Seed averaging: different fold-internal subsampling seeds reduce variance for free.
Folds stay fixed (seed 42) so OOF is comparable across members.
Writes shallow_pairs.csv / _oof.npy and rank blends with the previous best blend.
"""
import sys
from itertools import combinations
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
from shallow_te_full import keys  # noqa: E402

SEEDS = (1, 2, 3)
N_PAIR_COLS = 10


def signal(k, y):
    """Spread of per-value fraud rate (keys with >=100 rows): bigger = stronger per-value signal."""
    g = pd.DataFrame({"k": k, "y": y}).groupby("k")["y"].agg(["mean", "size"])
    g = g[g["size"] >= 100]
    return float(np.average((g["mean"] - y.mean()) ** 2, weights=g["size"])) if len(g) else 0.0


def main():
    X, Xt, Xo, y, yo = load_all()
    base = [c for c in X.columns if not c.startswith(("hour_", "amt_", "vel_", "risk_", "flag_", "card_age_y", "night"))]
    allx = pd.concat([X, Xo, Xt], ignore_index=True)
    K = keys(allx, base).apply(lambda s: pd.factorize(s)[0]).astype("int32")
    n, no = len(X), len(Xo)
    # rank columns by signal on train rows only (labels of train, not validation-specific; mild, fixed choice)
    sig = {c: signal(K[c].iloc[:n].values, y) for c in K.columns}
    top = sorted(sig, key=sig.get, reverse=True)[:N_PAIR_COLS]
    print("pair columns:", top, flush=True)
    for a, b in combinations(top, 2):
        K[f"{a}__{b}"] = pd.factorize(K[a].astype("int64") * 1_000_003 + K[b].astype("int64"))[0].astype("int32")
    Ktr = K.iloc[:n].reset_index(drop=True)
    Kor = K.iloc[n:n + no].reset_index(drop=True)
    Kte = K.iloc[n + no:].reset_index(drop=True)
    print("TE columns:", K.shape[1], flush=True)

    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof, pred = np.zeros(n), np.zeros(len(Xt))
    for k, (a, b) in enumerate(cv.split(X, y)):
        Ka = pd.concat([Ktr.iloc[a], Kor], ignore_index=True)
        ya = np.r_[y[a], yo]
        enc = TargetEncoder(target_type="binary", smooth="auto", random_state=SEED)
        Ea = pd.DataFrame(enc.fit_transform(Ka, ya), columns=Ka.columns).add_prefix("te_")
        Eb = pd.DataFrame(enc.transform(Ktr.iloc[b]), columns=Ka.columns).add_prefix("te_")
        Et = pd.DataFrame(enc.transform(Kte), columns=Ka.columns).add_prefix("te_")
        Xa = pd.concat([pd.concat([X.iloc[a], Xo], ignore_index=True), Ea], axis=1)
        Xb = pd.concat([X.iloc[b].reset_index(drop=True), Eb], axis=1)
        Xte = pd.concat([Xt.reset_index(drop=True), Et], axis=1)
        for s in SEEDS:
            m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=4, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0,
                                   random_state=s, verbose=-1, n_jobs=-1)
            m.fit(Xa, ya, eval_set=[(Xb, y[b])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[b] += m.predict_proba(Xb)[:, 1] / len(SEEDS)
            pred += m.predict_proba(Xte)[:, 1] / (N_SPLITS * len(SEEDS))
        print(f"  fold {k}: {roc_auc_score(y[b], oof[b]):.5f}", flush=True)
    print(f"shallow_pairs OOF AUC {roc_auc_score(y, oof):.5f}", flush=True)
    np.save(SUB / "shallow_pairs_oof.npy", oof)
    ids = pd.read_csv(SUB / "fe_blend.csv")["id"]
    pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(SUB / "shallow_pairs.csv", index=False)

    r = lambda v: rankdata(v) / len(v)
    prev_oof = 0.7 * r(np.load(SUB / "shallow_te_oof.npy")) + 0.3 * (0.3 * r(np.load(SUB / "cat_oof.npy")) + 0.7 * np.load(SUB / "fe_blend_oof.npy"))
    sh_te = pd.read_csv(SUB / "shallow_te.csv")["is_fraud"].values
    prev_te = pd.read_csv(SUB / "shallow_blend_w7.csv")["is_fraud"].values
    for w in (0.3, 0.5, 0.7, 0.9):
        print(f"blend w_pairs={w}: OOF {roc_auc_score(y, w * r(oof) + (1 - w) * prev_oof):.5f}", flush=True)
    for w in (0.5, 0.7):
        pd.DataFrame({"id": ids, "is_fraud": w * r(pred) + (1 - w) * r(prev_te)}).to_csv(SUB / f"pairs_blend_w{int(w*10)}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
