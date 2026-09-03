"""DICOM ETL (Tier 0.2) — one MR slice -> windowed uint8, MIL-ready.

Encodes the preprocessing the research notes call for:
- RescaleSlope/Intercept via modality LUT.
- Invert MONOCHROME1.
- Sort a series by InstanceNumber (caller supplies ordered arrays).
- Per-series 1-99 percentile window -> [0,1].
- Resize (default 288) with INTER_AREA when cv2 is present.
- Grayscale uint8 output.

The pure array functions here have no pydicom/torch dependency so they unit-test
locally; `read_dicom_slice` is the thin pydicom wrapper used on Kaggle.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def apply_rescale(arr: np.ndarray, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
    """Apply RescaleSlope/Intercept (modality LUT)."""
    return arr.astype(np.float32) * float(slope) + float(intercept)


def maybe_invert_monochrome1(arr: np.ndarray, photometric: str) -> np.ndarray:
    """MONOCHROME1 stores white-is-low; invert so high value = bright."""
    if str(photometric).strip().upper() == "MONOCHROME1":
        return arr.max() - arr
    return arr


def percentile_window(series: np.ndarray, low_pct: float = 1.0,
                      high_pct: float = 99.0) -> Tuple[float, float]:
    """Per-series window bounds from pooled pixels."""
    lo, hi = np.percentile(series.astype(np.float32), [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def window_to_unit(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo,hi] and scale to [0,1]."""
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def resize(arr: np.ndarray, size: int = 288) -> np.ndarray:
    """Resize to (size,size). cv2 INTER_AREA if available, else NN fallback."""
    try:
        import cv2
        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        h, w = arr.shape
        ys = np.linspace(0, h - 1, size).round().astype(int)
        xs = np.linspace(0, w - 1, size).round().astype(int)
        return arr[np.ix_(ys, xs)]


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint8 grayscale."""
    return np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)


def preprocess_series(
    raw_slices: List[np.ndarray],
    photometric: str = "MONOCHROME2",
    slope: float = 1.0,
    intercept: float = 0.0,
    size: int = 288,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """Full pipeline for an ORDERED list of raw slice arrays.

    Returns a (D, size, size) uint8 volume. Windowing is computed once over the
    whole series (pooled pixels) then applied per slice.
    """
    lut = [maybe_invert_monochrome1(apply_rescale(s, slope, intercept), photometric)
           for s in raw_slices]
    pooled = np.concatenate([s.ravel() for s in lut]) if lut else np.zeros(1, np.float32)
    lo, hi = percentile_window(pooled, low_pct, high_pct)
    out = np.empty((len(lut), size, size), dtype=np.uint8)
    for i, s in enumerate(lut):
        out[i] = to_uint8(resize(window_to_unit(s, lo, hi), size))
    return out


def read_dicom_slice(path: str):
    """Read one .dcm -> (float array with rescale+invert applied, photometric).

    Kaggle-only (needs pydicom). Kept separate from the pure functions above.
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    photo = str(getattr(d, "PhotometricInterpretation", "MONOCHROME2"))
    a = maybe_invert_monochrome1(a, photo)
    return a, photo
