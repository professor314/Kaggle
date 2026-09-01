"""Digit Recognizer - 5-fold CNN ensemble + test-time augmentation (Phase 3).

Trains 5 CNN folds (each on 80% of data) and averages their predicted
probabilities over several lightly-augmented views of each test image (TTA).
Ensembling + TTA typically pushes MNIST from ~99.4% to ~99.5%+.

Run with the DL environment.
Prior best: 0.99385 LB (single CNN, 15 epochs).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit.deep_learning import CNNClassifier
from kaggle_ml_toolkit.gpu_utils import get_device, gpu_info

COMP = REPO / "competitions" / "digit-recognizer"
DATA = COMP / "data"
SUB = COMP / "submissions"
N_SPLITS = 5
EPOCHS = 15
SEED = 42
N_TTA = 4  # number of augmented views (plus the clean view)


def tta_proba(clf, X, device):
    """Average predicted probs over the clean image + N_TTA augmented views."""
    model = clf._model.to(device).eval()
    h, w = clf.image_size
    Xn = np.asarray(X, dtype=np.float32)
    if Xn.max() > 1.5:
        Xn = Xn / clf._scale
    Xn = (Xn.reshape(-1, h, w) - clf._mean) / (clf._std + 1e-6)
    base = torch.from_numpy(Xn).unsqueeze(1)

    aug = transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.92, 1.08))
    views = [base] + [aug(base) for _ in range(N_TTA)]

    total = None
    with torch.no_grad():
        for v in views:
            probs = []
            for start in range(0, len(v), 512):
                xb = v[start:start + 512].to(device)
                probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
            p = np.vstack(probs)
            total = p if total is None else total + p
    return total / len(views)


def main():
    print("GPU:", gpu_info())
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].values
    X = train.drop("label", axis=1).values.astype(np.float32)
    X_test = test.values.astype(np.float32)

    device = get_device()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    test_proba = np.zeros((len(test), 10))
    oof_acc = []

    for fold, (tri, vai) in enumerate(cv.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        clf = CNNClassifier(num_classes=10, image_size=(28, 28), epochs=EPOCHS,
                            batch_size=128, augment=True, val_split=0.0, seed=SEED + fold)
        clf.fit(X[tri], y[tri])
        # Validation accuracy with TTA on the held-out fold.
        val_proba = tta_proba(clf, X[vai], device)
        acc = (val_proba.argmax(1) == y[vai]).mean()
        oof_acc.append(acc)
        print(f"  fold {fold} val acc (TTA): {acc:.4f}")
        test_proba += tta_proba(clf, X_test, device) / N_SPLITS

    print(f"\nMean fold val acc (TTA): {np.mean(oof_acc):.4f} +/- {np.std(oof_acc):.4f}")
    preds = test_proba.argmax(1)
    sub = pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds.astype(int)})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "cnn_ensemble_tta.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}  dist {sub['Label'].value_counts().sort_index().to_dict()}")
    print(f"=== DONE === mean fold acc {np.mean(oof_acc):.4f} (prior single-CNN LB 0.99385)")


if __name__ == "__main__":
    main()
