"""House-Prices - tune TE smoothing + optimize non-negative blend weights.

Builds on boosting_stack_te.py (OOF RMSLE 0.11218, LB 0.12162). Two levers:
  1. Sweep target-encoding smoothing over a small grid.
  2. Replace equal-weight blend with non-negative least-squares weights fit on
     OOF predictions (leakage-safe), which lets Lasso/Cat carry more weight.
Only writes a new submission if OOF beats 0.11218. Offline, anaconda python.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bte", HERE / "boosting_stack_te.py")
bte = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bte)

SEED, N_SPLITS = 42, 5
SUB = HERE.parent / "submissions"


def rmsle(a, b):
    return np.sqrt(mean_squared_error(a, b))


def te_with_smoothing(Xtr, Xte, y, raw_tr, raw_te, cv, smoothing):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for col in bte.TE_COLS:
        oof_enc, test_enc = bte.smooth_target_encode(
            raw_tr[col].values, y, raw_te[col].values, cv, smoothing=smoothing)
        Xtr[f"{col}_te"] = oof_enc
        Xte[f"{col}_te"] = test_enc
    return Xtr, Xte


def oof_and_test(Xtr, Xte, y, cv):
    names = list(bte.base_models().keys())
    oof = np.zeros((len(Xtr), len(names)))
    test_meta = np.zeros((len(Xte), len(names)))
    for j, name in enumerate(names):
        fold_test = np.zeros((len(Xte), N_SPLITS))
        for k, (tri, vai) in enumerate(cv.split(Xtr)):
            mdl = bte.base_models()[name]
            mdl.fit(Xtr.iloc[tri], y[tri])
            oof[vai, j] = mdl.predict(Xtr.iloc[vai])
            fold_test[:, k] = mdl.predict(Xte)
        test_meta[:, j] = fold_test.mean(axis=1)
    return oof, test_meta, names


def main():
    Xtr, Xte, y, test_ids, raw_tr, raw_te = bte.engineer()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    best = {"rmsle": 1.0}
    for sm in [2.0, 5.0, 10.0, 20.0, 50.0]:
        Xtr2, Xte2 = te_with_smoothing(Xtr, Xte, y, raw_tr, raw_te, cv, sm)
        oof, test_meta, names = oof_and_test(Xtr2, Xte2, y, cv)

        equal = rmsle(y, oof.mean(axis=1))
        # Non-negative weights fit on OOF, normalized.
        w, _ = nnls(oof, y)
        if w.sum() > 0:
            w = w / w.sum()
        wblend = rmsle(y, oof @ w)
        pick_r = min(equal, wblend)
        kind = "weighted" if wblend < equal else "equal"
        print(f"smoothing={sm:>4}: equal {equal:.5f} | weighted {wblend:.5f} "
              f"(w={np.round(w,2)}) -> {pick_r:.5f}")
        if pick_r < best["rmsle"]:
            pred = (test_meta @ w) if kind == "weighted" else test_meta.mean(axis=1)
            best = {"rmsle": pick_r, "sm": sm, "kind": kind, "pred": pred, "w": w}

    print(f"\nBest: smoothing={best['sm']} {best['kind']} OOF RMSLE {best['rmsle']:.5f}")
    print(f"Prior: equal-weight TE OOF 0.11218 / LB 0.12162")

    if best["rmsle"] < 0.11218:
        sub = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(best["pred"]).clip(0)})
        out = SUB / "boosting_stack_te_tuned.csv"
        sub.to_csv(out, index=False)
        print(f"IMPROVED -> saved {out} (median {sub.SalePrice.median():,.0f})")
    else:
        print("No OOF improvement over 0.11218; not writing a submission.")


if __name__ == "__main__":
    main()
