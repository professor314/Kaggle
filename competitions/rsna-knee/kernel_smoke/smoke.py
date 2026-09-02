"""RSNA Knee - PIPELINE SMOKE TEST (proves the path, no model).

Goal: confirm on Kaggle that we can (1) reach the data, (2) read K DICOM slices
per test study within time, (3) write a valid submission.csv, and (4) have the
code-competition submit flow accept it. The "model" is just the per-finding train
prevalence, so a low-but-valid macro-AUC (~0.5) is SUCCESS here.

Self-contained (mirrors competitions/rsna-knee/rsna_lib.py) so it runs as a
single Kaggle script kernel. No GPU needed.
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
K = 8            # slices to read per study (proves timed multi-DICOM reads)
BASE = "/kaggle/input"


def find_data_dir():
    for cand in glob.glob(os.path.join(BASE, "*")) + glob.glob(os.path.join(BASE, "*", "*")):
        if os.path.isdir(cand) and "knee" in os.path.basename(cand).lower():
            return cand
    dirs = [d for d in glob.glob(os.path.join(BASE, "*")) if os.path.isdir(d)]
    return dirs[0] if dirs else BASE


def sample_slice_paths(data_dir, study_uid, k):
    study_root = os.path.join(data_dir, "test_series", study_uid)
    if not os.path.isdir(study_root):
        return []
    paths = []
    for ser in sorted(os.listdir(study_root)):
        sdir = os.path.join(study_root, ser)
        if os.path.isdir(sdir):
            paths.extend(os.path.join(sdir, f) for f in sorted(os.listdir(sdir)) if f.endswith(".dcm"))
    if len(paths) <= k:
        return paths
    idx = np.linspace(0, len(paths) - 1, k).round().astype(int)
    return [paths[i] for i in idx]


def main():
    t0 = time.time()
    root = find_data_dir()
    print(f"data dir: {root}", flush=True)

    sample = pd.read_csv(os.path.join(root, "sample_submission.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))
    train = pd.read_csv(os.path.join(root, "train.csv"))
    print(f"sample {sample.shape} | test {test.shape} | train {train.shape}", flush=True)

    # Prevalence over labeled train rows
    prev = []
    for f in FINDINGS:
        col = pd.to_numeric(train[f], errors="coerce").dropna()
        prev.append(float(col.mean()) if len(col) else 0.5)
    prev = np.clip(np.array(prev), 0, 1)
    print(f"prevalence: {dict(zip(FINDINGS, prev.round(3)))}", flush=True)

    # Prove we can read K slices per test study, and time it.
    study_ids = list(sample[ID_COL])
    read_ok, read_fail, total_slices = 0, 0, 0
    try:
        import pydicom
        for sid in study_ids:
            paths = sample_slice_paths(root, sid, K)
            for p in paths:
                try:
                    _ = pydicom.dcmread(p).pixel_array
                    total_slices += 1
                except Exception:
                    read_fail += 1
            read_ok += 1
        print(f"read {total_slices} slices across {read_ok} studies "
              f"({read_fail} slice errors) in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"pydicom unavailable or read loop failed: {e!r}", flush=True)

    # Trivial predictor: prevalence for every study.
    probs = np.tile(prev, (len(study_ids), 1))
    sub = pd.DataFrame(probs, columns=FINDINGS)
    sub.insert(0, ID_COL, study_ids)
    # match sample column order exactly
    sub = sub[list(sample.columns)]
    assert list(sub.columns) == list(sample.columns), "column mismatch"
    assert len(sub) == len(sample), "row count mismatch"
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"wrote submission {sub.shape} in {time.time()-t0:.1f}s total", flush=True)
    print(sub.head().to_string(), flush=True)


if __name__ == "__main__":
    main()
