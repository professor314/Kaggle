"""Peek at the radiology reports + the 58 labeled rows to design report-mining
rules. Read-only, no GPU. Prints sample reports for labeled studies alongside
their finding labels, so we can see which words map to which finding."""
import os, glob, csv
BASE = "/kaggle/input"
FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]

root = None
for c in glob.glob(os.path.join(BASE, "*")) + glob.glob(os.path.join(BASE, "*", "*")):
    if os.path.isdir(c) and "knee" in os.path.basename(c).lower():
        root = c; break

import pandas as pd
tr = pd.read_csv(os.path.join(root, "train.csv"))
lab = tr[tr[FIND].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)].reset_index(drop=True)
print(f"labeled rows: {len(lab)}", flush=True)

# language hint: count reports that look Spanish vs English
import re
def lang(s):
    s = str(s).lower()
    es = sum(w in s for w in [" de ", " menisco", " rodilla", " sin ", " con ", "roturas", "artrosis"])
    en = sum(w in s for w in [" the ", " tear", " knee", " with ", " joint", " no "])
    return "es" if es > en else "en"
lab["lang"] = lab["Report"].map(lang)
print("labeled report languages:", lab["lang"].value_counts().to_dict(), flush=True)

# print first 8 labeled reports with their positive findings
for i in range(min(8, len(lab))):
    row = lab.iloc[i]
    pos = [f for f in FIND if str(row[f]) == "1" or row[f] == 1 or str(row[f]) == "1.0"]
    print(f"\n--- [{i}] lang={row['lang']} positives={pos}", flush=True)
    print("    " + str(row["Report"])[:500], flush=True)

# per-finding positive rate among labeled
print("\npositive rates (labeled):", flush=True)
for f in FIND:
    c = pd.to_numeric(lab[f], errors="coerce")
    print(f"  {f}: {int(c.sum())}/{len(c)}", flush=True)
