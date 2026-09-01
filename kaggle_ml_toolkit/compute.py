"""Compute configuration: multicore + GPU defaults for the toolkit.

Problem this solves: the gradient boosters (LightGBM / XGBoost / CatBoost) and
sklearn CV loops in this toolkit were being created with default settings, which
means single-/few-threaded CPU and never the GPU. On big Playground datasets
(hundreds of thousands of rows) that made runs crawl.

This module centralizes the right knobs so every notebook/module can opt into
all CPU cores, and the GPU when a CUDA build is present.

Usage:
    from kaggle_ml_toolkit.compute import (
        n_jobs, lgbm_params, xgb_params, catboost_params, gpu_available,
    )
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**lgbm_params(), n_estimators=1500, learning_rate=0.03)

The *_params helpers return only the compute-related kwargs (device/threads);
merge them with your model-specific hyperparameters.
"""

from __future__ import annotations

import os
import functools


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def n_jobs() -> int:
    """Number of parallel workers to use = all logical cores.

    Use as sklearn `n_jobs=n_jobs()` (cross_val_score, permutation_importance,
    RandomForest, etc.). sklearn treats -1 as "all cores" but an explicit count
    parallelizes more predictably inside nested contexts.
    """
    return os.cpu_count() or 1


def sklearn_parallel_kwargs() -> dict:
    """Kwargs to parallelize sklearn utilities that accept n_jobs."""
    return {"n_jobs": n_jobs()}


# ---------------------------------------------------------------------------
# GPU detection (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True if a usable CUDA GPU is present (via torch, if importable)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Gradient boosters
# ---------------------------------------------------------------------------

def lgbm_params(force_cpu: bool = False) -> dict:
    """Compute kwargs for LightGBM.

    Always uses all cores. Uses the GPU device when available (LightGBM's
    'gpu' device works on most CUDA builds; if a wheel lacks GPU support the
    caller should fall back to CPU).
    """
    params = {"n_jobs": n_jobs()}
    if gpu_available() and not force_cpu:
        params["device_type"] = "gpu"
    return params


def xgb_params(force_cpu: bool = False) -> dict:
    """Compute kwargs for XGBoost (>=2.0 API).

    XGBoost 2.x uses `device="cuda"` + `tree_method="hist"` for GPU, and
    `n_jobs` for CPU threading.
    """
    params = {"n_jobs": n_jobs(), "tree_method": "hist"}
    if gpu_available() and not force_cpu:
        params["device"] = "cuda"
    return params


def catboost_params(force_cpu: bool = False) -> dict:
    """Compute kwargs for CatBoost.

    CatBoost uses task_type='GPU' for the GPU, else CPU with thread_count.
    """
    if gpu_available() and not force_cpu:
        return {"task_type": "GPU"}
    return {"thread_count": n_jobs()}


def summary() -> str:
    """One-line description of the compute environment for logging."""
    gpu = "GPU" if gpu_available() else "CPU-only"
    return f"cores={n_jobs()} | {gpu}"
