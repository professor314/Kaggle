"""RSNA Knee - FAST exploration (no full-tree walk).

Lesson from the previous version: recursively globbing ~820k DICOMs stalls the
kernel. This version only:
  - lists the competition dir's TOP level + first CSV rows
  - reads every CSV header + row count + per-column coverage (the 12 findings live
    in sample_submission.csv's columns)
  - derives ONE dicom path from *_series.csv (or a shallow listdir of one study),
    reads just that single DICOM
No recursive globs. Should finish in well under a minute.
"""
import os
import csv
import glob

BASE = "/kaggle/input"

out = []
def log(m):
    print(m, flush=True)
    out.append(str(m))


# Locate the competition dir directly (one or two levels), no deep walk.
ROOT = None
for cand in glob.glob(os.path.join(BASE, "*")) + glob.glob(os.path.join(BASE, "*", "*")):
    if os.path.isdir(cand) and "knee" in os.path.basename(cand).lower():
        ROOT = cand
        break
if ROOT is None:
    dirs = [d for d in glob.glob(os.path.join(BASE, "*")) if os.path.isdir(d)]
    ROOT = dirs[0] if dirs else BASE
log(f"ROOT = {ROOT}")

log("\n=== TOP LEVEL ===")
for e in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, e)
    log(f"  {e}  ({'dir' if os.path.isdir(p) else str(os.path.getsize(p)) + ' bytes'})")

log("\n=== CSVs (header + coverage) ===")
for name in sorted(os.listdir(ROOT)):
    if not name.endswith(".csv"):
        continue
    p = os.path.join(ROOT, name)
    with open(p, newline="") as f:
        r = list(csv.reader(f))
    header, data = (r[0], r[1:]) if r else ([], [])
    log(f"\n  {name}: {len(data)} rows, {len(header)} cols")
    log(f"    columns: {header}")
    if data and len(header) <= 30:
        for j, col in enumerate(header):
            ne = sum(1 for row in data if j < len(row) and row[j] not in ("", "NaN"))
            log(f"      '{col}': {ne}/{len(data)} non-empty")
        log(f"    row0: {data[0]}")

# Derive ONE dicom path from test_series.csv if present, else shallow listdir.
log("\n=== ONE DICOM ===")
dcm = None
ts = os.path.join(ROOT, "test_series.csv")
if os.path.exists(ts):
    with open(ts, newline="") as f:
        rr = list(csv.reader(f))
    log(f"  test_series.csv columns: {rr[0] if rr else '(empty)'}")
    if len(rr) > 1:
        log(f"  test_series.csv row0: {rr[1]}")

for series_dir in ("test_series", "train_series"):
    sd = os.path.join(ROOT, series_dir)
    if os.path.isdir(sd):
        studies = os.listdir(sd)[:1]
        for st in studies:
            stp = os.path.join(sd, st)
            sers = os.listdir(stp)[:1] if os.path.isdir(stp) else []
            for se in sers:
                sep = os.path.join(stp, se)
                dcms = [x for x in os.listdir(sep) if x.endswith(".dcm")][:1]
                log(f"  {series_dir}: study has {len(os.listdir(stp))} series; "
                    f"series has {len(os.listdir(sep))} slices")
                if dcms:
                    dcm = os.path.join(sep, dcms[0])
        if dcm:
            break

if dcm:
    log(f"  sample dicom: {dcm}")
    try:
        import pydicom
        ds = pydicom.dcmread(dcm)
        arr = ds.pixel_array
        log(f"  shape {arr.shape} dtype {arr.dtype} min {arr.min()} max {arr.max()}")
        for t in ("Modality", "SeriesDescription", "Rows", "Columns",
                  "PhotometricInterpretation", "BitsStored", "PixelSpacing",
                  "SliceThickness", "InstanceNumber"):
            log(f"  {t}: {getattr(ds, t, 'N/A')}")
    except Exception as e:
        log(f"  pydicom error: {e!r}")
else:
    log("  no dicom located via shallow listdir")

with open("/kaggle/working/explore_summary.txt", "w") as f:
    f.write("\n".join(out))
log("\nDONE -> /kaggle/working/explore_summary.txt")
