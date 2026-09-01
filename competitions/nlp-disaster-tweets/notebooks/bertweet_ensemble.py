"""NLP Disaster Tweets - add BERTweet as a diverse 3rd ensemble base (Phase 3).

The DistilBERT + TF-IDF ensemble (LB 0.838) was limited by high correlation
(0.83) between its two bases. BERTweet is pretrained on tweets, so it should make
different errors and add genuine diversity.

This script:
1. Loads saved DistilBERT + TF-IDF OOF/test probs (artifacts/oof_probs.npz)
2. Trains BERTweet with the SAME 5 folds, saving its OOF/test probs
3. Blends all three, tuning weights + threshold on OOF F1

Run with the DL environment.
Baselines: DistilBERT 0.836 | DistilBERT+TF-IDF ensemble 0.838
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit.deep_learning import TransformerClassifier, gpu_info

COMP = REPO / "competitions" / "nlp-disaster-tweets"
DATA = COMP / "data"
SUB = COMP / "submissions"
ART = COMP / "artifacts"
N_SPLITS = 5
SEED = 42


def clean(text):
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("GPU:", gpu_info() or "none")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train["clean"] = train["text"].apply(clean)
    test["clean"] = test["text"].apply(clean)
    y = train["target"].values

    # Load saved OOF/test probs from the earlier ensemble run.
    prev = np.load(ART / "oof_probs.npz")
    oof_bert, oof_tfidf = prev["oof_bert"], prev["oof_tfidf"]
    test_bert, test_tfidf = prev["test_bert"], prev["test_tfidf"]
    assert len(oof_bert) == len(train), "OOF length mismatch — regenerate artifacts"

    # Train BERTweet on the SAME folds (same seed/splitter).
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_bt = np.zeros(len(train))
    test_bt = np.zeros(len(test))

    for fold, (tri, vai) in enumerate(cv.split(train["clean"].values, y), start=1):
        print(f"\n===== BERTweet fold {fold}/{N_SPLITS} =====")
        clf = TransformerClassifier(
            model_name="vinai/bertweet-base", num_labels=2, max_length=128,
            epochs=3, batch_size=16, seed=SEED,
            output_dir=str(COMP / "_hf_out" / f"bt_fold{fold}"),
        )
        clf.fit(train["clean"].values[tri], y[tri],
                train["clean"].values[vai], y[vai])
        oof_bt[vai] = clf.predict_proba(train["clean"].values[vai])[:, 1]
        test_bt += clf.predict_proba(test["clean"].values)[:, 1] / N_SPLITS
        print(f"  fold BERTweet F1: {f1_score(y[vai], (oof_bt[vai]>=0.5).astype(int)):.4f}")

    bt_f1 = f1_score(y, (oof_bt >= 0.5).astype(int))
    print(f"\nBERTweet OOF F1: {bt_f1:.4f}")
    print("Correlations (OOF):")
    print(f"  BERTweet vs DistilBERT: {np.corrcoef(oof_bt, oof_bert)[0,1]:.3f}")
    print(f"  BERTweet vs TF-IDF:     {np.corrcoef(oof_bt, oof_tfidf)[0,1]:.3f}")

    # Save updated artifacts.
    np.savez(ART / "oof_probs_v2.npz", oof_bert=oof_bert, oof_tfidf=oof_tfidf,
             oof_bt=oof_bt, test_bert=test_bert, test_tfidf=test_tfidf,
             test_bt=test_bt, y=y, ids=test["id"].values)

    # Grid over 3-way weights (simplex) + threshold, tuned on OOF F1.
    best = {"f1": 0.0}
    for wb in np.arange(0, 1.01, 0.1):
        for wt in np.arange(0, 1.01 - wb + 1e-9, 0.1):
            wbt = 1 - wb - wt
            if wbt < -1e-9:
                continue
            blend = wb * oof_bert + wt * oof_tfidf + wbt * oof_bt
            for thr in np.arange(0.35, 0.66, 0.01):
                f1 = f1_score(y, (blend >= thr).astype(int))
                if f1 > best["f1"]:
                    best = {"f1": f1, "wb": round(wb, 2), "wt": round(wt, 2),
                            "wbt": round(wbt, 2), "thr": round(thr, 2)}
    print(f"\nBest 3-way blend: DistilBERT={best['wb']} TFIDF={best['wt']} "
          f"BERTweet={best['wbt']} thr={best['thr']} -> OOF F1 {best['f1']:.4f}")

    test_blend = (best["wb"] * test_bert + best["wt"] * test_tfidf + best["wbt"] * test_bt)
    preds = (test_blend >= best["thr"]).astype(int)
    sub = pd.DataFrame({"id": test["id"], "target": preds})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "ensemble_3way.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}  balance {pd.Series(preds).value_counts().to_dict()}")
    print(f"=== DONE === 3-way OOF F1 {best['f1']:.4f} (2-way was 0.808, LB 0.838)")


if __name__ == "__main__":
    main()
