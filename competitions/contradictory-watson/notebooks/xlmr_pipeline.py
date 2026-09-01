"""Contradictory, My Dear Watson - XLM-RoBERTa NLI pipeline (Phase 3).

Multilingual natural language inference (3 classes: 0=entailment, 1=neutral,
2=contradiction) across 15 languages. Fine-tunes xlm-roberta-base on the
premise/hypothesis sentence pairs.

Run with the DL environment:
    .venv-dl\\Scripts\\python.exe competitions\\contradictory-watson\\notebooks\\xlmr_pipeline.py

Baseline to beat: TF-IDF ~0.33-0.40 (barely above the 3-class chance of 0.33).
Target: 0.85+ accuracy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit.deep_learning import TransformerClassifier, gpu_info

COMP = REPO / "competitions" / "contradictory-watson"
DATA_DIR = COMP / "data"
SUB_DIR = COMP / "submissions"

MODEL_NAME = "xlm-roberta-base"
SEED = 42


def main():
    print("GPU:", gpu_info() or "none")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Train {train.shape} | Test {test.shape}")
    print(f"Label dist: {train['label'].value_counts().sort_index().to_dict()}")

    # Hold out 10% (stratified) to estimate accuracy before the full-data refit.
    tr, va = train_test_split(
        train, test_size=0.1, random_state=SEED, stratify=train["label"]
    )

    clf = TransformerClassifier(
        model_name=MODEL_NAME,
        num_labels=3,
        max_length=96,
        epochs=3,
        batch_size=16,
        learning_rate=1e-5,
        seed=SEED,
        output_dir=str(COMP / "_hf_out" / "val"),
    )
    clf.fit(
        tr["premise"].tolist(), tr["label"].tolist(),
        va["premise"].tolist(), va["label"].tolist(),
        train_text_pairs=tr["hypothesis"].tolist(),
        val_text_pairs=va["hypothesis"].tolist(),
    )

    val_pred = clf.predict(va["premise"].tolist(), text_pairs=va["hypothesis"].tolist())
    val_acc = accuracy_score(va["label"].values, val_pred)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    # Per-language breakdown (English vs. non-English generalization).
    va = va.copy()
    va["pred"] = val_pred
    en_mask = va["language"] == "English"
    print(f"  English acc:     {accuracy_score(va[en_mask]['label'], va[en_mask]['pred']):.4f}")
    print(f"  Non-English acc: {accuracy_score(va[~en_mask]['label'], va[~en_mask]['pred']):.4f}")

    # Refit on ALL training data for the final submission.
    print("\nRefitting on full training set...")
    final = TransformerClassifier(
        model_name=MODEL_NAME, num_labels=3, max_length=96,
        epochs=3, batch_size=16, learning_rate=1e-5, seed=SEED,
        output_dir=str(COMP / "_hf_out" / "full"),
    )
    final.fit(
        train["premise"].tolist(), train["label"].tolist(),
        train_text_pairs=train["hypothesis"].tolist(),
    )

    preds = final.predict(test["premise"].tolist(), text_pairs=test["hypothesis"].tolist())
    sub = pd.DataFrame({"id": test["id"], "prediction": preds.astype(int)})
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = SUB_DIR / "xlmr_nli.csv"
    sub.to_csv(out, index=False)

    (COMP / "artifacts").mkdir(parents=True, exist_ok=True)
    final.save(str(COMP / "artifacts" / "xlmr_nli"))

    print(f"\nSaved: {out}  shape {sub.shape}")
    print(f"Pred dist: {sub['prediction'].value_counts().sort_index().to_dict()}")
    print(f"=== DONE === val acc {val_acc:.4f}")


if __name__ == "__main__":
    main()
