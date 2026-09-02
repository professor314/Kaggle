"""RSNA Knee - report-mining baseline (text -> 12 findings).

The reports ARE the radiologist's findings (EN + ES). We map each finding to
positive/negative phrase patterns (bilingual) with simple negation handling, and
score each test study's report. This uses no training and no images; it reads the
words a radiologist wrote. Validated against the 58 labeled studies (macro-AUC).

Read-only-ish (writes submission.csv). No GPU, no internet.
"""
import os, glob, re
import numpy as np
import pandas as pd

FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
ID = "StudyInstanceUID"

# Positive cue phrases per finding (English + Spanish), lowercase.
# Structure-anchor cues (need injury language nearby to be positive).
POS = {
    "ACL": ["acl", "anterior cruciate", "lca", "cruzado anterior"],
    "MCL": ["mcl", "medial collateral", "lcm", "colateral medial",
            "lateral interno"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisco interno"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisco externo"],
}
# Presence cues: positive if mentioned and not negated (compartment-scoped for OA).
PRESENCE = {
    "Medial OA": ["medial compartment osteoarth", "medial osteoarth",
                  "artrosis femorotibial medial", "medial femorotibial",
                  "medial compartment chondral", "medial compartment cartilage",
                  "chondrosis medial", "medial chondral", "condral medial"],
    "Lateral OA": ["lateral compartment osteoarth", "lateral osteoarth",
                   "artrosis femorotibial lateral", "lateral femorotibial",
                   "lateral compartment chondral", "lateral compartment cartilage",
                   "chondrosis lateral", "lateral chondral", "condral lateral"],
    "PF OA": ["patellofemoral osteoarth", "patellofemoral chondr", "trochlea",
              "patellar facet", "patellofemoral", "femoropatelar", "rotulian",
              "troclea", "patelofemoral", "chondromalacia"],
    "Effusion": ["effusion", "joint fluid", "fluid accumulation", "joint fluid",
                 "derrame", "fluid in the", "fluid distension", "hydrarthrosis"],
    "Synovitis": ["synovitis", "synovial thickening", "synovial membrane thick",
                  "sinovitis", "sinovial", "synovial proliferation"],
    "Baker's": ["baker", "popliteal cyst", "quiste de baker", "quiste popliteo",
                "popliteal fossa cyst"],
    "Contusion": ["contusion", "bone bruise", "bone marrow edema", "marrow edema",
                  "edema oseo", "edema medular", "medullary edema", "subchondral edema",
                  "bone edema"],
    "Fracture": ["fracture", "fractura"],
}
INJURY = ["tear", "torn", "rupture", "ruptur", "rotura", "roto", "rota",
          "lesion", "sprain", "esguince", "injury", "disrupt", "chondral defect",
          "cartilage defect", "chondrosis", "chondromalacia"]
NEG = ["no ", "not ", "intact", "preserved", "normal", "without", "unremarkable",
       "sin ", "conservad", "not torn", "negative", "descarta", "no evidence",
       "is not", "are not", "no significant", "no acute"]


def find_root():
    for c in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(c) and "knee" in os.path.basename(c).lower():
            return c
    return "/kaggle/input"


def _first_hit(t, cues):
    for cue in cues:
        idx = t.find(cue)
        if idx != -1:
            return idx
    return None


def score_report(text, finding):
    """Heuristic probability in [0,1] that `finding` is positive in `text`."""
    t = " " + str(text).lower().replace("\n", " ") + " "

    if finding in PRESENCE:
        # Positive if any presence cue appears and isn't negated in its window.
        best = 0.1
        for cue in PRESENCE[finding]:
            idx = t.find(cue)
            while idx != -1:
                window = t[max(0, idx - 45): idx + 55]
                neg = any(n in window for n in NEG)
                best = max(best, 0.2 if neg else 0.9)
                idx = t.find(cue, idx + 1)
        return best

    # Structure findings (ACL/MCL/menisci): positive only with injury language,
    # not negated, near the mention.
    hit = _first_hit(t, POS[finding])
    if hit is None:
        return 0.08
    window = t[max(0, hit - 70): hit + 90]
    neg = any(n in window for n in NEG)
    inj = any(w in window for w in INJURY)
    if inj and not neg:
        return 0.9
    if neg and not inj:
        return 0.1
    if inj and neg:
        return 0.5
    return 0.3  # mentioned, no clear injury/negation


def main():
    root = find_root()
    sample = pd.read_csv(os.path.join(root, "sample_submission.csv"))
    train = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))

    # Validate on the 58 labeled studies
    lab = train[train[FIND].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)].reset_index(drop=True)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for f in FIND:
        y = pd.to_numeric(lab[f], errors="coerce").values
        p = lab["Report"].map(lambda r: score_report(r, f)).values
        if len(np.unique(y)) == 2:
            a = roc_auc_score(y, p)
            aucs.append(a)
            print(f"  {f}: AUC {a:.3f}  (pos {int(y.sum())}/{len(y)})", flush=True)
    print(f"VALIDATION macro-AUC over labeled studies: {np.mean(aucs):.4f}", flush=True)

    # Test reports: test.csv has only StudyInstanceUID; reports for test studies
    # live where? train.csv has Report for train only. Check if test has reports.
    has_report = "Report" in test.columns
    print(f"test has Report column: {has_report}", flush=True)

    probs = np.zeros((len(sample), 12), dtype=float)
    if has_report:
        rep_by_id = dict(zip(test[ID], test["Report"]))
    else:
        rep_by_id = {}
    prev = [pd.to_numeric(train[f], errors="coerce").dropna().mean() for f in FIND]
    for r, sid in enumerate(sample[ID]):
        rep = rep_by_id.get(sid)
        if rep is None or (isinstance(rep, float) and np.isnan(rep)):
            probs[r] = prev  # fallback if no report for this test study
        else:
            probs[r] = [score_report(rep, f) for f in FIND]

    sub = pd.DataFrame(probs, columns=FIND)
    sub.insert(0, ID, list(sample[ID]))
    sub = sub[list(sample.columns)]
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"wrote submission {sub.shape}", flush=True)


if __name__ == "__main__":
    main()
