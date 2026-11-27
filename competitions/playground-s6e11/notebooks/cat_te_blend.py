"""Playground S6E11 - CatBoost with native categoricals, blended with the FE LGB/XGB blend.

Why CatBoost: it encodes categoricals with ordered target statistics computed inside training
(no leakage), which is the "OOF target encoding" lever that broke the House Prices plateau,
done natively. It also uses symmetric trees, so it is less correlated with LGB/XGB. S6E9 lesson:
blend gains come from diversity, not from more of the same booster.

Reads fe_blend_oof.npy / fe_blend.csv written by fe_blend.py, writes cat_oof.npy, cat.csv and
the final rank blend cat_blend.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from fe_blend import SUB, SEED, N_SPLITS, add_features, load  # noqa: E402


def main():
    tr, te, y, ids, cats = load()
    X, Xt = add_features(tr), add_features(te)
    cat_idx = [X.columns.get_loc(c) for c in cats]  # codes are ints; CatBoost needs the indices
    oof, pred = np.zeros(len(X)), np.zeros(len(Xt))
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(cv.split(X, y)):
        m = CatBoostClassifier(iterations=4000, learning_rate=0.06, depth=6, l2_leaf_reg=5,
                               eval_metric="AUC", od_type="Iter", od_wait=150,
                               random_seed=SEED, thread_count=-1, verbose=0,
                               cat_features=cat_idx)
        m.fit(X.iloc[a], y[a], eval_set=(X.iloc[b], y[b]))
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
        print(f"  cat fold {k}: {roc_auc_score(y[b], oof[b]):.5f} (iters {m.get_best_iteration()})", flush=True)
    print(f"cat OOF AUC {roc_auc_score(y, oof):.5f}", flush=True)
    np.save(SUB / "cat_oof.npy", oof)

    r = lambda v: rankdata(v) / len(v)
    fe_oof = np.load(SUB / "fe_blend_oof.npy")
    fe_pred = pd.read_csv(SUB / "fe_blend.csv")["is_fraud"].values
    for w in (0.3, 0.5, 0.7):  # weight on CatBoost; pick by OOF
        print(f"blend w_cat={w}: OOF {roc_auc_score(y, w * r(oof) + (1 - w) * fe_oof):.5f}", flush=True)
    w = 0.5
    pd.DataFrame({"id": ids, "is_fraud": w * r(pred) + (1 - w) * r(fe_pred)}).to_csv(SUB / "cat_blend.csv", index=False)
    pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(SUB / "cat.csv", index=False)
    print("wrote cat_blend.csv", flush=True)


if __name__ == "__main__":
    main()
