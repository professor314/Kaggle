"""Compute gold macro-AUC of the NLI labels (the log didn't persist last run).
Reads train.csv (gold 58) + the NLI labels dataset, prints per-finding + macro AUC."""
import os, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ID = "StudyInstanceUID"

def root():
    for c in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(c) and "knee" in os.path.basename(c).lower():
            return c
    return "/kaggle/input"

R = root()
train = pd.read_csv(os.path.join(R, "train.csv"))
nli_path = next(iter(glob.glob("/kaggle/input/**/nli_train_labels.csv", recursive=True)), None)
print("nli file:", nli_path, flush=True)
nli = pd.read_csv(nli_path)

m = train.merge(nli, on=ID, suffixes=("", "_nli"))
gold = m[m[FIND].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)]
print(f"gold rows matched: {len(gold)}", flush=True)
aucs = []
for f in FIND:
    y = pd.to_numeric(gold[f], errors="coerce").values
    p = gold[f + "_nli"].values
    if len(np.unique(y)) == 2:
        a = roc_auc_score(y, p); aucs.append(a)
        print(f"  {f}: AUC {a:.3f} (pos {int(y.sum())}/{len(y)})", flush=True)
print(f"GOLD macro-AUC (NLI): {np.mean(aucs):.4f}  (rules 0.607, image ~0.52)", flush=True)
