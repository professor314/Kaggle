"""NLP Disaster Tweets - DistilBERT fine-tuning pipeline (Phase 3).

Fine-tunes DistilBERT with stratified 5-fold CV, reports out-of-fold F1, and
builds a submission by averaging test probabilities across folds.

Run with the deep-learning environment (Python 3.12 + torch):
    .venv-dl\\Scripts\\python.exe competitions\\nlp-disaster-tweets\\notebooks\\distilbert_pipeline.py

Baseline to beat: 0.801 F1 (TF-IDF + LogisticRegression).
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
DATA_DIR = COMP / "data"
SUB_DIR = COMP / "submissions"

MODEL_NAME = "distilbert-base-uncased"
N_SPLITS = 5
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
SEED = 42


def clean_text(text: str) -> str:
    """Light cleaning: drop URLs and @mentions, collapse whitespace.

    Transformers handle casing/punctuation well, so we keep cleaning minimal.
    """
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("GPU:", gpu_info() or "none (running on CPU)")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train.shape[0]} rows | Test: {test.shape[0]} rows")

    train["clean"] = train["text"].apply(clean_text)
    test["clean"] = test["text"].apply(clean_text)

    X = train["clean"].values
    y = train["target"].values
    X_test = test["clean"].values

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    test_proba = np.zeros(len(test))
    fold_f1s = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        clf = TransformerClassifier(
            model_name=MODEL_NAME,
            num_labels=2,
            max_length=MAX_LENGTH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            output_dir=str(COMP / "_hf_out" / f"fold{fold}"),
        )
        clf.fit(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])

        val_proba = clf.predict_proba(X[va_idx])[:, 1]
        oof[va_idx] = val_proba
        fold_f1 = f1_score(y[va_idx], (val_proba >= 0.5).astype(int))
        fold_f1s.append(fold_f1)
        print(f"  Fold {fold} F1: {fold_f1:.4f}")

        test_proba += clf.predict_proba(X_test)[:, 1] / N_SPLITS

    oof_f1 = f1_score(y, (oof >= 0.5).astype(int))
    print("\n===== CV SUMMARY =====")
    print(f"  Fold F1s: {[round(f, 4) for f in fold_f1s]}")
    print(f"  Mean fold F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"  OOF F1 (threshold 0.5): {oof_f1:.4f}")

    # Threshold search on OOF predictions.
    best_thr, best_thr_f1 = 0.5, oof_f1
    for thr in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y, (oof >= thr).astype(int))
        if f1 > best_thr_f1:
            best_thr_f1, best_thr = f1, thr
    print(f"  Best OOF threshold: {best_thr:.2f} -> F1 {best_thr_f1:.4f}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    preds = (test_proba >= best_thr).astype(int)
    submission = pd.DataFrame({"id": test["id"], "target": preds})
    out = SUB_DIR / "distilbert_5fold.csv"
    submission.to_csv(out, index=False)
    print(f"\nSaved submission: {out}")
    print(f"  Prediction balance: {pd.Series(preds).value_counts().to_dict()}")
    print(f"\n=== DONE === OOF F1: {best_thr_f1:.4f} (baseline was 0.801)")


if __name__ == "__main__":
    main()
