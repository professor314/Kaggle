"""RSNA Knee - conservative 2.5D CNN baseline (self-contained Kaggle kernel).

Trains a small timm EfficientNet-B0 on the ~58 labeled studies (12-sigmoid
multi-label, BCE), sampling K slices per study and mean-pooling. Falls back to
the per-finding train prevalence if training can't run or produces nothing
usable, so a VALID submission is always written.

Runs on a Kaggle T4 (machine_shape NvidiaTeslaT4). Internet ON so timm can fetch
pretrained weights. Everything is bounded to finish well within the GPU cap:
K slices/study, few epochs, EfficientNet-B0.

Design notes are in competitions/rsna-knee/.kiro/specs/rsna-knee-baseline.
"""
import os
import glob
import time

import numpy as np
import pandas as pd

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
ID_COL = "StudyInstanceUID"
BASE = "/kaggle/input"

K = 8            # slices per study
IMG = 224
EPOCHS = 8
BATCH = 16
LR = 3e-4
SEED = 42
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def find_data_dir():
    for cand in glob.glob(os.path.join(BASE, "*")) + glob.glob(os.path.join(BASE, "*", "*")):
        if os.path.isdir(cand) and "knee" in os.path.basename(cand).lower():
            return cand
    dirs = [d for d in glob.glob(os.path.join(BASE, "*")) if os.path.isdir(d)]
    return dirs[0] if dirs else BASE


def slice_paths(data_dir, split, study_uid, k):
    root = os.path.join(data_dir, split, study_uid)
    if not os.path.isdir(root):
        return []
    paths = []
    for ser in sorted(os.listdir(root)):
        sdir = os.path.join(root, ser)
        if os.path.isdir(sdir):
            paths.extend(os.path.join(sdir, f) for f in sorted(os.listdir(sdir)) if f.endswith(".dcm"))
    if not paths:
        return []
    if len(paths) <= k:
        return paths
    idx = np.linspace(0, len(paths) - 1, k).round().astype(int)
    return [paths[i] for i in idx]


def read_slice(path):
    """DICOM -> (IMG,IMG,3) float32 in [0,1] via percentile windowing."""
    import pydicom
    import cv2
    arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr = cv2.resize(arr, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.repeat(arr[:, :, None], 3, axis=2)


def study_tensor(data_dir, split, sid, k):
    """(n_slices, 3, IMG, IMG) float32 for a study; empty if unreadable."""
    imgs = []
    for p in slice_paths(data_dir, split, sid, k):
        try:
            imgs.append(read_slice(p).transpose(2, 0, 1))
        except Exception:
            continue
    if not imgs:
        return None
    return np.stack(imgs).astype(np.float32)


def main():
    root = find_data_dir()
    log(f"data dir: {root}")
    sample = pd.read_csv(os.path.join(root, "sample_submission.csv"))
    train = pd.read_csv(os.path.join(root, "train.csv"))
    study_ids = list(sample[ID_COL])

    # Prevalence fallback vector (always valid).
    prev = []
    for f in FINDINGS:
        c = pd.to_numeric(train[f], errors="coerce").dropna()
        prev.append(float(c.mean()) if len(c) else 0.5)
    prev = np.clip(np.array(prev, dtype=np.float32), 0, 1)
    log(f"prevalence fallback: {prev.round(3).tolist()}")

    # Labeled studies (have any finding value).
    lab_mask = train[FINDINGS].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)
    labeled = train[lab_mask].reset_index(drop=True)
    log(f"labeled studies: {len(labeled)}")

    use_model = False
    model = None
    device = "cpu"
    try:
        import torch
        import torch.nn as nn
        import timm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"torch {torch.__version__} device={device}")
        torch.manual_seed(SEED)

        # Build training tensors (mean slice per study -> one image per study to
        # keep it small and fast; label = the study's 12 findings).
        Xtr, Ytr = [], []
        for _, row in labeled.iterrows():
            t = study_tensor(root, "train_series", row[ID_COL], K)
            if t is None:
                continue
            Xtr.append(t.mean(axis=0))  # (3,IMG,IMG) mean over slices
            Ytr.append([float(pd.to_numeric(row[f], errors="coerce")) for f in FINDINGS])
        if len(Xtr) >= 20:
            X = torch.tensor(np.stack(Xtr))
            Y = torch.tensor(np.array(Ytr, dtype=np.float32))
            log(f"train tensor {tuple(X.shape)} labels {tuple(Y.shape)}")

            # Internet is disabled in this competition's kernels, so we cannot
            # download pretrained weights at runtime. Train from scratch. (Future
            # upgrade: attach the timm weights as a Kaggle dataset and set
            # pretrained=True with a local checkpoint.)
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=12)
            model = model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            lossf = nn.BCEWithLogitsLoss()
            model.train()
            n = len(X)
            for ep in range(EPOCHS):
                perm = torch.randperm(n)
                tot = 0.0
                for i in range(0, n, BATCH):
                    idx = perm[i:i + BATCH]
                    xb = X[idx].to(device)
                    yb = Y[idx].to(device)
                    opt.zero_grad()
                    out = model(xb)
                    loss = lossf(out, yb)
                    loss.backward()
                    opt.step()
                    tot += loss.item() * len(idx)
                log(f"  epoch {ep+1}/{EPOCHS} loss {tot/n:.4f}")
            use_model = True
        else:
            log(f"only {len(Xtr)} readable labeled studies (<20); using fallback")
    except Exception as e:
        log(f"training path failed ({e!r}); using prevalence fallback")

    # Inference
    probs = np.tile(prev, (len(study_ids), 1)).astype(np.float32)
    if use_model:
        import torch
        model.eval()
        with torch.no_grad():
            for r, sid in enumerate(study_ids):
                t = study_tensor(root, "test_series", sid, K)
                if t is None:
                    continue  # keep fallback row
                xb = torch.tensor(t.mean(axis=0)[None]).to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()[0]
                probs[r] = np.clip(p, 0, 1)
        log("inference with trained model done")
    else:
        log("submission uses prevalence fallback for all studies")

    sub = pd.DataFrame(probs, columns=FINDINGS)
    sub.insert(0, ID_COL, study_ids)
    sub = sub[list(sample.columns)]
    assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample)
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    log(f"wrote submission {sub.shape}; model_used={use_model}")


if __name__ == "__main__":
    main()
