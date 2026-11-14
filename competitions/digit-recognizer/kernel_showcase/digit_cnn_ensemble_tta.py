"""Digit Recognizer - 5-fold CNN ensemble + test-time augmentation.

Trains 5 CNN folds (each on 80% of the data) and averages their predicted
probabilities over several lightly-augmented views of each test image (TTA).
Ensembling + TTA pushes MNIST from ~99.4% (single CNN) to ~99.45%+.

Architecture (SmallCNN): two conv blocks (32 then 64 channels, each two 3x3
convs + BatchNorm + ReLU, MaxPool, Dropout), then adaptive-avg-pool -> a 256
unit head with dropout. Trained with AdamW + OneCycleLR, light random-affine
augmentation, and mixed precision on GPU.

Best public LB with this approach: 0.9945 accuracy.

Self-contained Kaggle version: reads from /kaggle/input/digit-recognizer/.
Enable the GPU accelerator for this kernel.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

DATA = Path("/kaggle/input/digit-recognizer")
N_SPLITS = 5
EPOCHS = 15
SEED = 42
N_TTA = 4  # augmented views in addition to the clean view
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Linear(64 * 3 * 3, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(X, scale, mean, std):
    Xn = np.asarray(X, dtype=np.float32).reshape(-1, 28, 28) / scale
    Xn = (Xn - mean) / (std + 1e-6)
    return torch.from_numpy(Xn).unsqueeze(1)


def train_fold(X, y, seed):
    """Train one SmallCNN on all provided (X, y). Returns model + norm stats."""
    seed_everything(seed)
    Xf = np.asarray(X, dtype=np.float32)
    scale = 255.0 if Xf.max() > 1.5 else 1.0
    scaled = Xf / scale
    mean, std = float(scaled.mean()), float(scaled.std())

    Xt = normalize(Xf, scale, mean, std)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=128, shuffle=True,
                        pin_memory=torch.cuda.is_available())

    aug = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=EPOCHS,
                                                steps_per_epoch=max(1, len(loader)))
    loss_fn = nn.CrossEntropyLoss()
    use_fp16 = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = aug(xb)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * xb.size(0)
        print(f"  epoch {epoch}/{EPOCHS} train_loss {running/len(Xt):.4f}")
    model.eval()
    return model, scale, mean, std


def tta_proba(model, X, scale, mean, std):
    """Average predicted probs over the clean image + N_TTA augmented views."""
    base = normalize(X, scale, mean, std)
    aug = transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.92, 1.08))
    views = [base] + [aug(base) for _ in range(N_TTA)]
    total = None
    with torch.no_grad():
        for v in views:
            probs = []
            for start in range(0, len(v), 512):
                xb = v[start:start + 512].to(DEVICE)
                probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
            p = np.vstack(probs)
            total = p if total is None else total + p
    return total / len(views)


def main():
    print("Device:", DEVICE)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["label"].values
    X = train.drop("label", axis=1).values.astype(np.float32)
    X_test = test.values.astype(np.float32)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    test_proba = np.zeros((len(test), 10))
    fold_accs = []

    for fold, (tri, vai) in enumerate(cv.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        model, scale, mean, std = train_fold(X[tri], y[tri], seed=SEED + fold)
        val_proba = tta_proba(model, X[vai], scale, mean, std)
        acc = (val_proba.argmax(1) == y[vai]).mean()
        fold_accs.append(acc)
        print(f"  fold {fold} val acc (TTA): {acc:.4f}")
        test_proba += tta_proba(model, X_test, scale, mean, std) / N_SPLITS

    print(f"\nMean fold val acc (TTA): {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    preds = test_proba.argmax(1)
    sub = pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds.astype(int)})
    sub.to_csv("submission.csv", index=False)
    print(f"Saved: submission.csv  dist {sub['Label'].value_counts().sort_index().to_dict()}")
    print(f"=== DONE === mean fold acc {np.mean(fold_accs):.4f} (best public LB 0.9945)")


if __name__ == "__main__":
    main()
