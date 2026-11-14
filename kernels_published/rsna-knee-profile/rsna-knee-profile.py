"""RSNA Knee - data profiling before modeling.

Answers, in one read-only pass:
  1. Does the TEST set include report text? (train_cursor / test.csv / test_series)
  2. Cleaning checks: missing values, empty reports, duplicate studies, label
     value set (are blanks truly empty vs 0), study/series/slice count spread.
  3. Series metadata distributions (Fluid_Sensitive, Fat_Suppression,
     Anatomical_Plane) for train vs test -> informs which series to feed a model.
  4. DICOM sanity across a few studies (shape, dtype, value range, per-series
     slice counts) -> informs preprocessing (windowing, resize, which plane).
No GPU, no internet. Shallow listdir only (never a full-tree glob).
"""
import os, glob, csv
import numpy as np
import pandas as pd

FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
ID = "StudyInstanceUID"

def root():
    for c in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
        if os.path.isdir(c) and "knee" in os.path.basename(c).lower():
            return c
    return "/kaggle/input"

R = root()
print("=== FILES ===", flush=True)
for f in sorted(os.listdir(R)):
    p = os.path.join(R, f)
    print(f"  {f}  ({'dir' if os.path.isdir(p) else str(os.path.getsize(p))+' B'})", flush=True)

def peek_csv(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        print(f"\n{name}: MISSING", flush=True)
        return None
    df = pd.read_csv(p)
    print(f"\n=== {name}: {df.shape} ===", flush=True)
    print(f"  columns: {list(df.columns)}", flush=True)
    return df

train = peek_csv("train.csv")
test = peek_csv("test.csv")
peek_csv("test_series.csv")
train_series = peek_csv("train_series.csv")
peek_csv("sample_submission.csv")

# 1. TEST reports?
print("\n=== Q1: does TEST have report text? ===", flush=True)
if test is not None:
    print(f"  test.csv columns: {list(test.columns)}", flush=True)
    print(f"  'Report' in test.csv: {'Report' in test.columns}", flush=True)
# any file that looks like a test report/label file?
extra = [f for f in os.listdir(R) if f.endswith(".csv") and f not in
         ("train.csv","test.csv","train_series.csv","test_series.csv","sample_submission.csv")]
print(f"  other csv files: {extra}", flush=True)

# 2. Cleaning checks on train
if train is not None:
    print("\n=== Q2: cleaning checks (train.csv) ===", flush=True)
    print(f"  duplicate StudyInstanceUID: {train[ID].duplicated().sum()}", flush=True)
    empty_report = train['Report'].isna().sum() + (train['Report'].astype(str).str.strip()=="").sum()
    print(f"  empty/NaN reports: {empty_report}", flush=True)
    # label value set
    vals = set()
    for f in FIND:
        vals |= set(pd.unique(train[f].dropna()))
    print(f"  distinct non-null label values across findings: {sorted(map(str, vals))}", flush=True)
    labeled = train[FIND].apply(lambda c: pd.to_numeric(c, errors='coerce')).notna().any(axis=1).sum()
    print(f"  studies with >=1 label: {labeled} / {len(train)}", flush=True)
    # report length distribution
    rl = train['Report'].astype(str).str.len()
    print(f"  report length: min {rl.min()} median {int(rl.median())} max {rl.max()}", flush=True)

# 3. Series metadata distributions
def dist(df, col):
    if df is None or col not in df.columns:
        return "n/a"
    return df[col].value_counts(dropna=False).to_dict()

print("\n=== Q3: series metadata (train_series) ===", flush=True)
for c in ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]:
    print(f"  {c}: {dist(train_series, c)}", flush=True)
if train_series is not None:
    per_study = train_series.groupby(ID).size()
    print(f"  series per study: min {per_study.min()} median {int(per_study.median())} max {per_study.max()}", flush=True)

# 4. DICOM sanity across a few train studies
print("\n=== Q4: DICOM sanity (first 3 train studies) ===", flush=True)
try:
    import pydicom
    tdir = os.path.join(R, "train_series")
    studies = sorted(os.listdir(tdir))[:3] if os.path.isdir(tdir) else []
    for st in studies:
        sp = os.path.join(tdir, st)
        sers = sorted(os.listdir(sp))
        print(f"  study {st[:24]}...: {len(sers)} series", flush=True)
        for se in sers[:2]:
            sep = os.path.join(sp, se)
            dcms = [f for f in os.listdir(sep) if f.endswith(".dcm")]
            if dcms:
                ds = pydicom.dcmread(os.path.join(sep, dcms[0]))
                a = ds.pixel_array
                print(f"    series {se[:20]}...: {len(dcms)} slices, "
                      f"{a.shape} {a.dtype} range {a.min()}..{a.max()} "
                      f"desc='{getattr(ds,'SeriesDescription','?')}' "
                      f"plane_rows={getattr(ds,'Rows','?')}", flush=True)
except Exception as e:
    print(f"  dicom error: {e!r}", flush=True)

print("\nDONE", flush=True)
