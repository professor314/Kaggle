"""NLP Disaster Tweets - DistilBERT + TF-IDF ensemble (Phase 3).

Generates out-of-fold (OOF) probabilities for BOTH models on identical folds,
then blends them. The blend weight and decision threshold are tuned on OOF
predictions only (no leakage), and applied to averaged test probabilities.

Run with the DL environment:
    .venv-dl\\Scripts\\python.exe competitions\\nlp-disaster-tweets\\notebooks\\ensemble_pipeline.py

Baselines: TF-IDF LB 0.801 | DistilBERT LB 0.836
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit.deep_learning import TransformerClassifier, gpu_info

COMP = REPO / "competitions" / "nlp-disaster-tweets"
DATA_DIR = COMP / "data"
SUB_DIR = COMP / "submissions"
ART_DIR = COMP / "artifacts"

N_SPLITS = 5
EPOCHS = 3
SEED = 42


def clean_light(text: str) -> str:
    """Light cleaning for the transformer (keep casing/punctuation)."""
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_tfidf(text: str) -> str:
    """Lowercasing clean for the TF-IDF model (matches Experiment 1)."""
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return text.lower().strip()


def build_tfidf_features(train, test):
    """Word + char + keyword TF-IDF, fit on combined train+test text."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                           min_df=2, max_df=0.95, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                           min_df=2, max_df=0.95, sublinear_tf=True)
    kw = TfidfVectorizer(analyzer="word", max_features=300)

    all_word = pd.concat([train["tfidf_text"], test["tfidf_text"]])
    word.fit(all_word); char.fit(all_word)
    train["kw"] = train["keyword"].fillna("unknown").str.replace("%20", " ")
    test["kw"] = test["keyword"].fillna("unknown").str.replace("%20", " ")
    kw.fit(pd.concat([train["kw"], test["kw"]]))

    Xtr = hstack([word.transform(train["tfidf_text"]),
                  char.transform(train["tfidf_text"]),
                  kw.transform(train["kw"])]).tocsr()
    Xte = hstack([word.transform(test["tfidf_text"]),
                  char.transform(test["tfidf_text"]),
                  kw.transform(test["kw"])]).tocsr()
    return Xtr, Xte


def main():
    print("GPU:", gpu_info() or "none")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train["bert_text"] = train["text"].apply(clean_light)
    test["bert_text"] = test["text"].apply(clean_light)
    train["tfidf_text"] = train["text"].apply(clean_tfidf)
    test["tfidf_text"] = test["text"].apply(clean_tfidf)

    y = train["target"].values
    Xtr_tfidf, Xte_tfidf = build_tfidf_features(train, test)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_bert = np.zeros(len(train))
    oof_tfidf = np.zeros(len(train))
    test_bert = np.zeros(len(test))
    test_tfidf = np.zeros(len(test))

    for fold, (tr, va) in enumerate(cv.split(train, y), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")

        # --- TF-IDF + Logistic Regression ---
        lr = LogisticRegression(C=0.5, max_iter=1000, solver="liblinear", random_state=SEED)
        lr.fit(Xtr_tfidf[tr], y[tr])
        oof_tfidf[va] = lr.predict_proba(Xtr_tfidf[va])[:, 1]
        test_tfidf += lr.predict_proba(Xte_tfidf)[:, 1] / N_SPLITS

        # --- DistilBERT ---
        clf = TransformerClassifier(
            model_name="distilbert-base-uncased", num_labels=2, max_length=128,
            epochs=EPOCHS, batch_size=16, seed=SEED,
            output_dir=str(COMP / "_hf_out" / f"ens_fold{fold}"),
        )
        clf.fit(train["bert_text"].values[tr], y[tr],
                train["bert_text"].values[va], y[va])
        oof_bert[va] = clf.predict_proba(train["bert_text"].values[va])[:, 1]
        test_bert += clf.predict_proba(test["bert_text"].values)[:, 1] / N_SPLITS

        print(f"  fold TF-IDF F1: {f1_score(y[va], (oof_tfidf[va]>=0.5).astype(int)):.4f}"
              f" | BERT F1: {f1_score(y[va], (oof_bert[va]>=0.5).astype(int)):.4f}")

    # Persist OOF/test probabilities for reuse.
    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(ART_DIR / "oof_probs.npz", oof_bert=oof_bert, oof_tfidf=oof_tfidf,
             test_bert=test_bert, test_tfidf=test_tfidf, y=y, ids=test["id"].values)

    bert_f1 = f1_score(y, (oof_bert >= 0.5).astype(int))
    tfidf_f1 = f1_score(y, (oof_tfidf >= 0.5).astype(int))
    corr = np.corrcoef(oof_bert, oof_tfidf)[0, 1]
    print("\n===== OOF SUMMARY =====")
    print(f"  DistilBERT OOF F1: {bert_f1:.4f}")
    print(f"  TF-IDF     OOF F1: {tfidf_f1:.4f}")
    print(f"  Prob correlation:  {corr:.4f} (lower = more diverse = better blend)")

    # Tune blend weight w (on BERT) and threshold on OOF F1.
    best = {"f1": 0.0, "w": 1.0, "thr": 0.5}
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * oof_bert + (1 - w) * oof_tfidf
        for thr in np.arange(0.30, 0.71, 0.01):
            f1 = f1_score(y, (blend >= thr).astype(int))
            if f1 > best["f1"]:
                best = {"f1": f1, "w": round(float(w), 2), "thr": round(float(thr), 2)}
    print(f"\n  Best blend: w_bert={best['w']} thr={best['thr']} -> OOF F1 {best['f1']:.4f}")

    test_blend = best["w"] * test_bert + (1 - best["w"]) * test_tfidf
    preds = (test_blend >= best["thr"]).astype(int)
    sub = pd.DataFrame({"id": test["id"], "target": preds})
    out = SUB_DIR / "ensemble_bert_tfidf.csv"
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"  balance: {pd.Series(preds).value_counts().to_dict()}")
    print(f"\n=== DONE === Ensemble OOF F1 {best['f1']:.4f} "
          f"(BERT alone {bert_f1:.4f}, TF-IDF alone {tfidf_f1:.4f})")


if __name__ == "__main__":
    main()
