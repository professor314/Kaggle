"""Tier 0.5 — local tests for the correctness modules.

Run: python -m pytest test_tier0.py -q
Covers: clinical rules (negation / unmentioned / each rule family), DICOM ETL
shape+range+invert, series selection (fluid/plane), fold split excludes gold,
submission assembler format.
"""
import numpy as np
import pandas as pd

from labels.rules import score_finding, score_report, FINDINGS
from image.etl import (apply_rescale, maybe_invert_monochrome1, percentile_window,
                       window_to_unit, resize, to_uint8, preprocess_series)
from image.series import pick_series
from image.folds import assign_folds, is_gold
from rsna_lib import build_submission


# ---- rules -----------------------------------------------------------------
def test_unmentioned_is_zero():
    # a report about the ACL says nothing about effusion/baker/fracture
    txt = "Anterior cruciate ligament tear."
    assert score_finding(txt, "Effusion") == 0.0
    assert score_finding(txt, "Baker's") == 0.0
    assert score_finding(txt, "Fracture") == 0.0


def test_negation_is_low():
    assert score_finding("No joint effusion.", "Effusion") < 0.2
    assert score_finding("ACL intact, no tear.", "ACL") < 0.2
    assert score_finding("Sin rotura del menisco interno.", "Medial Meniscus") < 0.6


def test_positive_injury_is_high():
    assert score_finding("Complete tear of the ACL.", "ACL") > 0.8
    assert score_finding("Rotura de menisco interno.", "Medial Meniscus") > 0.8
    assert score_finding("Joint effusion present.", "Effusion") > 0.7


def test_impression_boosts():
    body = "Findings: acl tear. Impression: acl tear."
    assert score_finding(body, "ACL") >= 0.9


def test_contusion_needs_trauma_not_degenerative():
    degen = "Subchondral marrow edema with degenerative osteoarthritis."
    trauma = "Bone contusion of the lateral femoral condyle."
    assert score_finding(trauma, "Contusion") > score_finding(degen, "Contusion")


def test_fracture_includes_avulsion():
    assert score_finding("Avulsion fracture of the tibial spine.", "Fracture") > 0.7


def test_synovitis_breadth():
    assert score_finding("Hoffa impingement noted.", "Synovitis") > 0.5


def test_score_report_returns_all_findings():
    out = score_report("ACL tear with joint effusion.")
    assert set(out.keys()) == set(FINDINGS)
    assert all(0.0 <= v <= 1.0 for v in out.values())


# ---- ETL -------------------------------------------------------------------
def test_rescale_and_invert():
    a = np.array([[0.0, 10.0]], np.float32)
    assert np.allclose(apply_rescale(a, 2.0, 1.0), [[1.0, 21.0]])
    inv = maybe_invert_monochrome1(np.array([[0.0, 10.0]]), "MONOCHROME1")
    assert np.allclose(inv, [[10.0, 0.0]])
    same = maybe_invert_monochrome1(np.array([[0.0, 10.0]]), "MONOCHROME2")
    assert np.allclose(same, [[0.0, 10.0]])


def test_window_range():
    s = np.arange(100, dtype=np.float32).reshape(10, 10)
    lo, hi = percentile_window(s, 1, 99)
    u = window_to_unit(s, lo, hi)
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_resize_and_uint8_shape():
    s = np.random.rand(50, 60).astype(np.float32)
    r = resize(s, 288)
    assert r.shape == (288, 288)
    u = to_uint8(np.clip(r, 0, 1))
    assert u.dtype == np.uint8 and u.max() <= 255


def test_preprocess_series_output():
    raw = [np.random.randint(0, 4000, (40, 48)).astype(np.float32) for _ in range(5)]
    vol = preprocess_series(raw, photometric="MONOCHROME1", size=288)
    assert vol.shape == (5, 288, 288)
    assert vol.dtype == np.uint8


# ---- series selection ------------------------------------------------------
def _row(uid, plane, fluid):
    return {"SeriesInstanceUID": uid, "Anatomical_Plane": plane, "Fluid_Sensitive": fluid}


def test_pick_series_prefers_fluid_and_planes():
    rows = [
        _row("s1", "Sagittal", 0), _row("s2", "Sagittal", 1),
        _row("c1", "Coronal", 1), _row("a1", "Axial", 0),
    ]
    picked = pick_series(rows, k=3)
    assert len(picked) == 3
    planes = {r["Anatomical_Plane"] for r in picked}
    assert planes == {"Sagittal", "Coronal", "Axial"}  # one per plane
    sag = [r for r in picked if r["Anatomical_Plane"] == "Sagittal"][0]
    assert sag["Fluid_Sensitive"] == 1  # fluid-sensitive preferred


def test_pick_series_caps_at_k():
    rows = [_row(f"s{i}", "Sagittal", 0) for i in range(6)]
    assert len(pick_series(rows, k=3)) == 3


def test_pick_series_empty():
    assert pick_series([], k=3) == []


# ---- folds -----------------------------------------------------------------
def _make_df(n_train=50, n_gold=8):
    rows = []
    for i in range(n_train):
        r = {"StudyInstanceUID": f"train{i}"}
        for f in FINDINGS:
            r[f] = ""  # unlabeled
        rows.append(r)
    for i in range(n_gold):
        r = {"StudyInstanceUID": f"gold{i}"}
        for j, f in enumerate(FINDINGS):
            r[f] = 1 if j == 0 else 0  # has expert labels
        rows.append(r)
    return pd.DataFrame(rows)


def test_gold_excluded_from_folds():
    df = _make_df()
    folds = assign_folds(df, n_splits=5, seed=42)
    gold = is_gold(df)
    assert (folds[gold] == -1).all()          # every gold study held out
    assert (folds[~gold] >= 0).all()          # every train study in a real fold
    assert folds[~gold].nunique() == 5        # 5 folds used


def test_fold_grouping_by_study():
    # duplicate study ids should land in the same fold
    df = pd.DataFrame([{"StudyInstanceUID": "A", **{f: "" for f in FINDINGS}}] * 4 +
                      [{"StudyInstanceUID": f"B{i}", **{f: "" for f in FINDINGS}} for i in range(20)])
    folds = assign_folds(df, n_splits=5, seed=0)
    a_folds = folds[df["StudyInstanceUID"] == "A"].unique()
    assert len(a_folds) == 1


# ---- submission assembler --------------------------------------------------
def test_build_submission_format():
    ids = ["x", "y", "z"]
    probs = np.random.rand(3, 12)
    sub = build_submission(ids, probs)
    assert list(sub.columns) == ["StudyInstanceUID"] + FINDINGS
    assert sub["StudyInstanceUID"].tolist() == ids
    vals = sub[FINDINGS].values
    assert np.isfinite(vals).all() and vals.min() >= 0 and vals.max() <= 1
