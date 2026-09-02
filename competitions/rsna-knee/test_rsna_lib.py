"""Local unit tests for rsna_lib (no Kaggle, no real data)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsna_lib as R


def test_preprocess_shape_and_range():
    arr = (np.random.RandomState(0).randn(512, 512) * 500).astype(np.int16)
    out = R.preprocess_dicom(arr, size=224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocess_constant_image_no_crash():
    arr = np.full((64, 64), 7, dtype=np.int16)  # hi==lo edge case
    out = R.preprocess_dicom(arr, size=32)
    assert out.shape == (32, 32, 3)
    assert np.isfinite(out).all()


def test_prevalence_vector():
    # 4 labeled rows, rest unlabeled (NaN). ACL positive rate = 0.75.
    n = 10
    df = pd.DataFrame({R.ID_COL: [f"s{i}" for i in range(n)]})
    for f in R.FINDINGS:
        df[f] = [np.nan] * n
    df.loc[0:3, "ACL"] = [1, 1, 1, 0]
    v = R.prevalence_vector(df)
    assert v.shape == (12,)
    assert abs(v[R.FINDINGS.index("ACL")] - 0.75) < 1e-9
    # a finding with no labels defaults to 0.5
    assert abs(v[R.FINDINGS.index("Fracture")] - 0.5) < 1e-9
    assert (v >= 0).all() and (v <= 1).all()


def test_labeled_studies():
    df = pd.DataFrame({R.ID_COL: ["a", "b", "c"]})
    for f in R.FINDINGS:
        df[f] = [np.nan, np.nan, np.nan]
    df.loc[1, "ACL"] = 1
    lab = R.labeled_studies(df)
    assert list(lab[R.ID_COL]) == ["b"]


def test_build_submission_format():
    ids = ["x", "y", "z"]
    probs = np.full((3, 12), 0.3)
    df = R.build_submission(ids, probs)
    assert list(df.columns) == [R.ID_COL] + R.FINDINGS
    assert len(df) == 3
    assert df[R.FINDINGS].to_numpy().min() >= 0
    assert df[R.FINDINGS].to_numpy().max() <= 1
    assert not df.isna().any().any()


def test_build_submission_clips_and_fills():
    ids = ["x"]
    probs = np.array([[2.0, -1.0, np.nan] + [0.5] * 9])
    df = R.build_submission(ids, probs)
    row = df[R.FINDINGS].iloc[0].to_numpy()
    assert row[0] == 1.0 and row[1] == 0.0 and row[2] == 0.5


def test_sample_slice_paths(tmp_path):
    # Build a fake study/series/slice tree and check even subsampling to k.
    data = tmp_path
    study = "study1"
    root = data / "test_series" / study / "seriesA"
    root.mkdir(parents=True)
    for i in range(20):
        (root / f"{i:03d}.dcm").write_bytes(b"x")
    series_df = pd.DataFrame({R.ID_COL: [study], "SeriesInstanceUID": ["seriesA"]})
    paths = R.sample_slice_paths(str(data), "test_series", study, series_df, k=5)
    assert len(paths) == 5
    assert all(p.endswith(".dcm") for p in paths)
    # deterministic
    paths2 = R.sample_slice_paths(str(data), "test_series", study, series_df, k=5)
    assert paths == paths2


def test_sample_slice_paths_missing_study(tmp_path):
    series_df = pd.DataFrame({R.ID_COL: ["nope"], "SeriesInstanceUID": ["s"]})
    assert R.sample_slice_paths(str(tmp_path), "test_series", "nope", series_df, k=5) == []
