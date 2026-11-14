"""RSNA Knee - zero-shot NLI report labeling (free, no API).

Loads mDeBERTa-v3-base-mnli-xnli (multilingual NLI) offline from an attached
dataset. For each finding, forms an English hypothesis; scores entailment of the
report -> hypothesis. Reports are EN + ES; the model is multilingual so we feed
the raw report as premise. Validates against the 58 gold studies (macro-AUC).

Writes per-study finding probabilities for ALL train studies to
/kaggle/working/nli_train_labels.csv (the weak-label set for the image model),
and prints the gold macro-AUC so we know if this beats the 0.607 rules.
"""
import os, glob, time
import numpy as np
import pandas as pd

FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
ID = "StudyInstanceUID"

# One entailment hypothesis per finding.
HYP = {
    "ACL": "There is a tear or injury of the anterior cruciate ligament.",
    "MCL": "There is a tear or injury of the medial collateral ligament.",
    "Medial Meniscus": "There is a tear of the medial meniscus.",
    "Lateral Meniscus": "There is a tear of the lateral meniscus.",
    "Medial OA": "There is osteoarthritis or cartilage damage in the medial compartment.",
    "Lateral OA": "There is osteoarthritis or cartilage damage in the lateral compartment.",
    "PF OA": "There is patellofemoral osteoarthritis or cartilage damage.",
    "Effusion": "There is a joint effusion or fluid in the knee.",
    "Synovitis": "There is synovitis or synovial thickening.",
    "Baker's": "There is a Baker's cyst or popliteal cyst.",
    "Contusion": "There is a bone contusion or bone marrow edema.",
    "Fracture": "There is a fracture.",
}
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def find_root():
    for c in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(c) and "knee" in os.path.basename(c).lower():
            return c
    return "/kaggle/input"


def find_model():
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if "mdeberta" in d.lower() or "xnli" in d.lower() or "mnli" in d.lower():
            return d
    # fallback: any dir with a model file
    for c in glob.glob("/kaggle/input/**/model.safetensors", recursive=True):
        return os.path.dirname(c)
    return None


def main():
    root = find_root()
    mdir = find_model()
    log(f"root {root} | model {mdir}")
    train = pd.read_csv(os.path.join(root, "train.csv"))

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mdir)
    mdl = AutoModelForSequenceClassification.from_pretrained(mdir).to(device).eval()
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    ent_idx = [i for i, v in id2label.items() if v.lower().startswith("entail")][0]
    con_idx = [i for i, v in id2label.items() if v.lower().startswith("contra")][0]
    log(f"model loaded on {device}; entail idx {ent_idx}")

    hyps = [HYP[f] for f in FIND]

    def score_report(report):
        """Return 12 probs: softmax over (entail vs contra) for each finding."""
        prem = str(report)[:2000]
        enc = tok([prem] * 12, hyps, return_tensors="pt", truncation=True,
                  padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits  # (12, 3)
        # prob positive = entail / (entail + contra), ignoring neutral
        e = logits[:, ent_idx]
        c = logits[:, con_idx]
        p = torch.sigmoid(e - c).cpu().numpy()
        return p

    # Validate on the 58 gold
    gold_mask = train[FIND].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)
    gold = train[gold_mask].reset_index(drop=True)
    from sklearn.metrics import roc_auc_score
    gp = np.stack([score_report(r) for r in gold["Report"]])
    aucs = []
    for j, f in enumerate(FIND):
        y = pd.to_numeric(gold[f], errors="coerce").values
        if len(np.unique(y)) == 2:
            a = roc_auc_score(y, gp[:, j]); aucs.append(a)
            log(f"  {f}: AUC {a:.3f} (pos {int(y.sum())}/{len(y)})")
    log(f"GOLD macro-AUC (NLI): {np.mean(aucs):.4f}  (rules were 0.607)")

    # Label ALL train studies -> weak-label file
    log("labeling all train studies...")
    out = np.zeros((len(train), 12), dtype=np.float32)
    for i in range(len(train)):
        out[i] = score_report(train["Report"].iloc[i])
        if (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(train)}")
    df = pd.DataFrame(out, columns=FIND)
    df.insert(0, ID, train[ID].values)
    df.to_csv("/kaggle/working/nli_train_labels.csv", index=False)
    log(f"wrote nli_train_labels.csv {df.shape}")


if __name__ == "__main__":
    main()
