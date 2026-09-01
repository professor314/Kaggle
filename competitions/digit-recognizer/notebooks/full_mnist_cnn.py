"""Digit Recognizer - train on full MNIST, leak-free (Phase 3).

Kaggle's Digit Recognizer test set (28K) is a subset of the original MNIST.
Naively training on all of torchvision's MNIST (70K) could include Kaggle's TEST
images with their true labels = leakage/cheating. To stay clean:

  1. Load all 70K MNIST images (train 60K + test 10K) with labels.
  2. Hash Kaggle's 28K test images by exact pixel content.
  3. EXCLUDE any MNIST image whose pixels match a Kaggle test image.
  4. Train the CNN on (Kaggle train 42K) + (remaining leak-free MNIST) and
     predict Kaggle's test.

This gives more training data without touching the test labels. We verify the
number of excluded images (should be ~28K if MNIST covers Kaggle's test).

Prior best: 0.99446 (5-fold CNN + TTA on Kaggle's 42K only).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit.deep_learning import CNNClassifier
from kaggle_ml_toolkit.gpu_utils import gpu_info

DATA = REPO / "competitions" / "digit-recognizer" / "data"
SUB = REPO / "competitions" / "digit-recognizer" / "submissions"


def mnist_arrays():
    root = os.path.expanduser("~/.cache/mnist")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    X = np.concatenate([tr.data.numpy(), te.data.numpy()]).reshape(-1, 784).astype(np.uint8)
    y = np.concatenate([tr.targets.numpy(), te.targets.numpy()]).astype(np.int64)
    return X, y


def pixel_hashes(arr_uint8):
    """Hash each row's exact bytes for fast exact-match set membership."""
    return {row.tobytes() for row in arr_uint8}


def main():
    print("GPU:", gpu_info())
    kaggle_train = pd.read_csv(DATA / "train.csv")
    kaggle_test = pd.read_csv(DATA / "test.csv")
    yk = kaggle_train["label"].values
    Xk = kaggle_train.drop("label", axis=1).values.astype(np.uint8)
    Xk_test = kaggle_test.values.astype(np.uint8)

    Xm, ym = mnist_arrays()
    print(f"MNIST total {len(Xm)} | Kaggle train {len(Xk)} test {len(Xk_test)}")

    # Build the set of Kaggle TEST pixel signatures and exclude them from MNIST.
    test_sigs = pixel_hashes(Xk_test)
    keep = np.array([row.tobytes() not in test_sigs for row in Xm])
    n_excluded = (~keep).sum()
    print(f"MNIST images matching Kaggle TEST (excluded to avoid leakage): {n_excluded}")
    Xm_safe, ym_safe = Xm[keep], ym[keep]

    # Also drop MNIST rows that duplicate Kaggle TRAIN (harmless, avoids double-count).
    train_sigs = pixel_hashes(Xk)
    not_dup = np.array([row.tobytes() not in train_sigs for row in Xm_safe])
    Xm_extra, ym_extra = Xm_safe[not_dup], ym_safe[not_dup]
    print(f"Leak-free MNIST added on top of Kaggle train: {len(Xm_extra)}")

    X_full = np.concatenate([Xk, Xm_extra]).astype(np.float32)
    y_full = np.concatenate([yk, ym_extra])
    print(f"Total training rows: {len(X_full)} (was {len(Xk)})")

    clf = CNNClassifier(num_classes=10, image_size=(28, 28), epochs=15,
                        batch_size=128, augment=True, val_split=0.05, seed=42)
    clf.fit(X_full, y_full)

    preds = clf.predict(Xk_test.astype(np.float32))
    sub = pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds.astype(int)})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "full_mnist_cnn.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}  dist {sub['Label'].value_counts().sort_index().to_dict()}")
    print("=== DONE === (prior best LB 0.99446)")


if __name__ == "__main__":
    main()
