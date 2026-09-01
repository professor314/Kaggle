"""Digit Recognizer (MNIST) - CNN pipeline (Phase 3).

Trains a compact CNN with light augmentation using the toolkit's CNNClassifier,
then generates a Kaggle submission. Target: 99%+ accuracy (top tier), up from
the sklearn baseline (rank ~435).

Run with the DL environment:
    .venv-dl\\Scripts\\python.exe competitions\\digit-recognizer\\notebooks\\cnn_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit.deep_learning import CNNClassifier
from kaggle_ml_toolkit.gpu_utils import gpu_info

COMP = REPO / "competitions" / "digit-recognizer"
DATA_DIR = COMP / "data"
SUB_DIR = COMP / "submissions"


def main():
    print("GPU:", gpu_info())
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Train {train.shape} | Test {test.shape}")

    y = train["label"].values
    X = train.drop("label", axis=1).values.astype(np.float32)
    X_test = test.values.astype(np.float32)

    clf = CNNClassifier(
        num_classes=10,
        image_size=(28, 28),
        epochs=15,
        batch_size=128,
        learning_rate=1e-3,
        augment=True,
        val_split=0.1,
        seed=42,
    )
    clf.fit(X, y)

    preds = clf.predict(X_test)
    sub = pd.DataFrame({
        "ImageId": np.arange(1, len(preds) + 1),
        "Label": preds.astype(int),
    })
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = SUB_DIR / "cnn_submission.csv"
    sub.to_csv(out, index=False)

    # Save the model for reuse.
    (COMP / "artifacts").mkdir(parents=True, exist_ok=True)
    clf.save(str(COMP / "artifacts" / "cnn_mnist.pt"))

    print(f"\nSaved: {out}  shape {sub.shape}")
    print(f"Label dist: {sub['Label'].value_counts().sort_index().to_dict()}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
