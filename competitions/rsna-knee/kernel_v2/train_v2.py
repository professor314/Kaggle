"""RSNA Knee v2 - report weak-labels -> image model (the real approach).

Pipeline (self-contained Kaggle kernel, T4, internet OFF):
  1. Weak-label ALL 4,407 train studies from their reports (bilingual rules).
  2. Train EfficientNet-B0 (12 sigmoids) on weakly-labeled studies, K slices
     mean-pooled per study, offline-loaded ImageNet weights.
  3. VALIDATE against the 58 gold-labeled studies (macro-AUC) — the honest metric.
  4. Predict the test images -> submission.csv (prevalence fallback if unreadable).

Decisions locked from profiling:
  - uint16, mixed sizes -> resize 224; per-image percentile windowing.
  - blanks are UNLABELED, not negatives (only matters for the gold val set).
  - weak labels come from the report; gold 58 are held out purely for validation.

Time budget: subsample train studies (WEAK_TRAIN_N) and slices (K) to stay well
under the GPU cap; bump later once it runs clean.
"""
import os, glob, re, time
import numpy as np
import pandas as pd

FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
ID = "StudyInstanceUID"
K = 6                 # slices per study (kept low for the big train set)
IMG = 192
EPOCHS = 6
BATCH = 32
LR = 3e-4
WEAK_TRAIN_N = 1500   # subsample of weakly-labeled studies to fit the time budget
SEED = 42
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ---------------- report weak-labeling (from Exp 4b, macro-AUC ~0.61) ----------
POS = {
    "ACL": ["acl", "anterior cruciate", "lca", "cruzado anterior"],
    "MCL": ["mcl", "medial collateral", "lcm", "colateral medial", "lateral interno"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisco interno"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisco externo"],
}
PRESENCE = {
    "Medial OA": ["medial compartment osteoarth", "medial osteoarth", "artrosis femorotibial medial",
                  "medial femorotibial", "medial compartment chondral", "medial compartment cartilage",
                  "chondrosis medial", "medial chondral", "condral medial"],
    "Lateral OA": ["lateral compartment osteoarth", "lateral osteoarth", "artrosis femorotibial lateral",
                   "lateral femorotibial", "lateral compartment chondral", "lateral compartment cartilage",
                   "chondrosis lateral", "lateral chondral", "condral lateral"],
    "PF OA": ["patellofemoral osteoarth", "patellofemoral chondr", "trochlea", "patellar facet",
              "patellofemoral", "femoropatelar", "rotulian", "troclea", "patelofemoral", "chondromalacia"],
    "Effusion": ["effusion", "joint fluid", "fluid accumulation", "derrame", "fluid in the",
                 "fluid distension", "hydrarthrosis"],
    "Synovitis": ["synovitis", "synovial thickening", "synovial membrane thick", "sinovitis",
                  "sinovial", "synovial proliferation"],
    "Baker's": ["baker", "popliteal cyst", "quiste de baker", "quiste popliteo", "popliteal fossa cyst"],
    "Contusion": ["contusion", "bone bruise", "bone marrow edema", "marrow edema", "edema oseo",
                  "edema medular", "medullary edema", "subchondral edema", "bone edema"],
    "Fracture": ["fracture", "fractura"],
}
INJURY = ["tear", "torn", "rupture", "ruptur", "rotura", "roto", "rota", "lesion", "sprain",
          "esguince", "injury", "disrupt", "chondral defect", "cartilage defect", "chondrosis",
          "chondromalacia"]
NEG = ["no ", "not ", "intact", "preserved", "normal", "without", "unremarkable", "sin ",
       "conservad", "not torn", "negative", "descarta", "no evidence", "is not", "are not",
       "no significant", "no acute"]

def score(text, f):
    t = " " + str(text).lower().replace("\n", " ") + " "
    if f in PRESENCE:
        best = 0.1
        for cue in PRESENCE[f]:
            i = t.find(cue)
            while i != -1:
                w = t[max(0, i-45):i+55]
                best = max(best, 0.2 if any(n in w for n in NEG) else 0.9)
                i = t.find(cue, i+1)
        return best
    hit = None
    for cue in POS[f]:
        j = t.find(cue)
        if j != -1: hit = j; break
    if hit is None: return 0.08
    w = t[max(0, hit-70):hit+90]
    neg = any(n in w for n in NEG); inj = any(x in w for x in INJURY)
    if inj and not neg: return 0.9
    if neg and not inj: return 0.1
    if inj and neg: return 0.5
    return 0.3

# ---------------- data helpers ----------
def find_root():
    for c in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(c) and "knee" in os.path.basename(c).lower():
            return c
    return "/kaggle/input"

def slice_paths(root, split, sid, k):
    sd = os.path.join(root, split, sid)
    if not os.path.isdir(sd): return []
    paths = []
    for ser in sorted(os.listdir(sd)):
        p = os.path.join(sd, ser)
        if os.path.isdir(p):
            paths += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".dcm")]
    if not paths: return []
    if len(paths) <= k: return paths
    idx = np.linspace(0, len(paths)-1, k).round().astype(int)
    return [paths[i] for i in idx]

def read_slice(path):
    import pydicom, cv2
    a = pydicom.dcmread(path).pixel_array.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo: hi = lo + 1
    a = np.clip((a-lo)/(hi-lo), 0, 1)
    a = cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.repeat(a[:, :, None], 3, axis=2)

def study_img(root, split, sid, k):
    imgs = []
    for p in slice_paths(root, split, sid, k):
        try: imgs.append(read_slice(p).transpose(2, 0, 1))
        except Exception: continue
    return np.stack(imgs).mean(axis=0).astype(np.float32) if imgs else None


def main():
    root = find_root(); log(f"root {root}")
    sample = pd.read_csv(os.path.join(root, "sample_submission.csv"))
    train = pd.read_csv(os.path.join(root, "train.csv"))

    # gold-labeled 58 (held out ONLY for validation)
    gold_mask = train[FIND].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)
    gold = train[gold_mask].reset_index(drop=True)
    pool = train[~gold_mask].reset_index(drop=True)
    log(f"gold {len(gold)} | unlabeled pool {len(pool)}")

    # weak labels for the pool (subsample for time)
    rng = np.random.RandomState(SEED)
    if len(pool) > WEAK_TRAIN_N:
        pool = pool.iloc[rng.permutation(len(pool))[:WEAK_TRAIN_N]].reset_index(drop=True)
    weak = np.stack([[score(r, f) for f in FIND] for r in pool["Report"]])
    weak_bin = (weak >= 0.5).astype(np.float32)   # threshold soft scores to 0/1 targets
    log(f"weak-labeled train studies: {len(pool)} (pos rate {weak_bin.mean():.3f})")

    prev = np.clip([pd.to_numeric(train[f], errors="coerce").dropna().mean() for f in FIND], 0, 1)

    use_model = False; model = None; device = "cpu"
    try:
        import torch, torch.nn as nn, timm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"torch {torch.__version__} device {device}")
        torch.manual_seed(SEED)

        # Build weak-train tensors (read images for the pool studies)
        Xtr, Ytr = [], []
        for i in range(len(pool)):
            im = study_img(root, "train_series", pool[ID].iloc[i], K)
            if im is None: continue
            Xtr.append(im); Ytr.append(weak_bin[i])
            if (i+1) % 300 == 0: log(f"  read {i+1}/{len(pool)} train studies")
        log(f"read {len(Xtr)} train images")

        # Gold val tensors
        Xg, Yg = [], []
        for i in range(len(gold)):
            im = study_img(root, "train_series", gold[ID].iloc[i], K)
            if im is None: continue
            Xg.append(im); Yg.append([float(pd.to_numeric(gold[f].iloc[i], errors="coerce")) for f in FIND])
        log(f"read {len(Xg)} gold-val images")

        if len(Xtr) >= 100:
            X = torch.tensor(np.stack(Xtr)); Y = torch.tensor(np.stack(Ytr))
            Xv = torch.tensor(np.stack(Xg)); Yv = np.stack(Yg)

            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=12)
            w = next(iter(glob.glob("/kaggle/input/**/efficientnet_b0.pth", recursive=True)), None)
            if w:
                model.load_state_dict(torch.load(w, map_location="cpu"), strict=False)
                log(f"loaded pretrained backbone {w}")
            model = model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            lossf = nn.BCEWithLogitsLoss()
            from sklearn.metrics import roc_auc_score
            n = len(X)
            for ep in range(EPOCHS):
                model.train(); perm = torch.randperm(n)
                for i in range(0, n, BATCH):
                    idx = perm[i:i+BATCH]
                    xb = X[idx].to(device); yb = Y[idx].to(device)
                    opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
                # gold val macro-AUC
                model.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(model(Xv.to(device))).cpu().numpy()
                aucs = [roc_auc_score(Yv[:, j], pv[:, j]) for j in range(12) if len(np.unique(Yv[:, j])) == 2]
                log(f"  epoch {ep+1}/{EPOCHS} loss {loss.item():.4f} gold_val_macroAUC {np.mean(aucs):.4f}")
            use_model = True
        else:
            log(f"only {len(Xtr)} train images (<100); fallback")
    except Exception as e:
        log(f"training failed ({e!r}); fallback")

    # inference on test
    ids = list(sample[ID]); probs = np.tile(prev, (len(ids), 1)).astype(np.float32)
    if use_model:
        import torch
        model.eval()
        with torch.no_grad():
            for r, sid in enumerate(ids):
                im = study_img(root, "test_series", sid, K)
                if im is None: continue
                probs[r] = np.clip(torch.sigmoid(model(torch.tensor(im[None]).to(device))).cpu().numpy()[0], 0, 1)
        log("test inference done")

    sub = pd.DataFrame(probs, columns=FIND); sub.insert(0, ID, ids)
    sub = sub[list(sample.columns)]
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    log(f"wrote submission {sub.shape} model_used={use_model}")


if __name__ == "__main__":
    main()
