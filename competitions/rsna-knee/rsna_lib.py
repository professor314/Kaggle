"""Reusable helpers for the RSNA Knee baseline.

Pure functions with no Kaggle dependency so they can be unit-tested locally and
imported by the Kaggle kernels. Everything indexes images from the CSVs (never a
recursive DICOM-tree walk, which stalls the kernel).
"""

from __future__ import annotations

import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd

# The 12 findings, in the exact order they appear in sample_submission.csv
# (VERIFIED 2026-09-02 from the exploration kernel).
FINDINGS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"


def find_data_dir(base: str = "/kaggle/input") -> str:
    """Locate the competition data dir without a deep walk.

    Checks base/<*> and base/<*>/<*> for a dir whose name mentions the comp.
    Falls back to the first directory under base.
    """
    for cand in glob.glob(os.path.join(base, "*")) + glob.glob(os.path.join(base, "*", "*")):
        if os.path.isdir(cand) and "knee" in os.path.basename(cand).lower():
            return cand
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    return dirs[0] if dirs else base


def load_finding_columns(sample_submission_path: str) -> List[str]:
    """Read the finding column names from sample_submission (source of truth)."""
    cols = list(pd.read_csv(sample_submission_path, nrows=0).columns)
    return [c for c in cols if c != ID_COL]


def prevalence_vector(train_df: pd.DataFrame, findings: List[str] = FINDINGS) -> np.ndarray:
    """Per-finding positive rate over the labeled rows.

    Labeled rows are those with non-null finding values. Returns a vector in
    [0,1] of length len(findings). If a finding has no labeled rows, uses 0.5.
    """
    vals = []
    for f in findings:
        col = pd.to_numeric(train_df[f], errors="coerce")
        labeled = col.dropna()
        vals.append(float(labeled.mean()) if len(labeled) else 0.5)
    v = np.array(vals, dtype=float)
    return np.clip(v, 0.0, 1.0)


def labeled_studies(train_df: pd.DataFrame, findings: List[str] = FINDINGS) -> pd.DataFrame:
    """Rows that have at least one non-null finding label."""
    mask = train_df[findings].apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().any(axis=1)
    return train_df[mask].reset_index(drop=True)


def sample_slice_paths(data_dir: str, series_split: str, study_uid: str,
                       series_df: pd.DataFrame, k: int) -> List[str]:
    """Return up to k evenly-spaced .dcm paths for a study.

    series_split is "train_series" or "test_series". Uses series_df (the
    *_series.csv) to find this study's series, then a shallow listdir of each
    series dir (no recursive glob). Slices are pooled across the study's series
    and evenly subsampled to k.
    """
    study_root = os.path.join(data_dir, series_split, study_uid)
    if not os.path.isdir(study_root):
        return []
    slice_paths: List[str] = []
    for series_uid in sorted(os.listdir(study_root)):
        sdir = os.path.join(study_root, series_uid)
        if not os.path.isdir(sdir):
            continue
        dcms = sorted(f for f in os.listdir(sdir) if f.endswith(".dcm"))
        slice_paths.extend(os.path.join(sdir, f) for f in dcms)
    if not slice_paths:
        return []
    if len(slice_paths) <= k:
        return slice_paths
    idx = np.linspace(0, len(slice_paths) - 1, k).round().astype(int)
    return [slice_paths[i] for i in idx]


def preprocess_dicom(pixel_array: np.ndarray, size: int = 224,
                     low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Window an int16 MR slice to [0,1], resize to size, repeat to 3 channels.

    Percentile windowing (MR has no fixed scale). Returns float32 (size,size,3).
    Resize uses cv2 if available, else a simple numpy nearest-neighbor.
    """
    arr = pixel_array.astype(np.float32)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr = _resize(arr, size).astype(np.float32)
    return np.repeat(arr[:, :, None], 3, axis=2).astype(np.float32)


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        # nearest-neighbor fallback, no external deps
        h, w = arr.shape
        ys = np.linspace(0, h - 1, size).round().astype(int)
        xs = np.linspace(0, w - 1, size).round().astype(int)
        return arr[np.ix_(ys, xs)]


def build_submission(study_ids: List[str], probs: np.ndarray,
                     findings: List[str] = FINDINGS) -> pd.DataFrame:
    """Assemble a submission DataFrame matching the sample format exactly.

    probs: (n_studies, n_findings) in [0,1]. Column order = [ID_COL] + findings.
    """
    probs = np.asarray(probs, dtype=float)
    assert probs.shape == (len(study_ids), len(findings)), \
        f"probs shape {probs.shape} != {(len(study_ids), len(findings))}"
    df = pd.DataFrame(probs, columns=findings)
    df.insert(0, ID_COL, list(study_ids))
    # safety: clip and fill any stray NaN with 0.5
    for f in findings:
        df[f] = np.clip(pd.to_numeric(df[f], errors="coerce").fillna(0.5), 0.0, 1.0)
    return df
