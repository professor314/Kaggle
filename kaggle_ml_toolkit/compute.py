"""Compute configuration: multicore + GPU defaults, overridable per machine.

Problem this solves: the gradient boosters (LightGBM / XGBoost / CatBoost) and
sklearn CV loops were created with default settings = single-/few-threaded CPU
and never the GPU. On big datasets that made runs crawl. This module centralizes
the right knobs.

By default everything AUTO-DETECTS (all logical cores; GPU if a CUDA torch build
is present), so it "just works" with no config. When someone runs this on their
own machine and wants different behavior, they can override without touching code
in two ways (env vars win over the file):

1. Environment variables:
     KAGGLE_TOOLKIT_CORES=8        # cap worker count (default: all cores)
     KAGGLE_TOOLKIT_USE_GPU=0      # 1/true = force GPU, 0/false = force CPU,
                                   # unset = auto-detect

2. A YAML file (searched: $KAGGLE_TOOLKIT_COMPUTE, ./compute.yaml,
   ~/.kaggle_toolkit/compute.yaml):
     cores: 8
     use_gpu: false

Usage in code/notebooks:
    from kaggle_ml_toolkit import compute
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**compute.lgbm_params(), n_estimators=1500)
    print(compute.summary())   # e.g. "cores=8 (capped) | GPU off (config)"

The *_params helpers return ONLY compute kwargs (threads/device); merge them
with your model hyperparameters.
"""

from __future__ import annotations

import os
import functools
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config loading (env vars > yaml file > auto-detect)
# ---------------------------------------------------------------------------

def _load_file_config() -> dict:
    """Load the first compute.yaml found, or {} if none/unparseable."""
    candidates = []
    env_path = os.environ.get("KAGGLE_TOOLKIT_COMPUTE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "compute.yaml")
    candidates.append(Path.home() / ".kaggle_toolkit" / "compute.yaml")
    for p in candidates:
        try:
            if p.is_file():
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return {}


@functools.lru_cache(maxsize=1)
def _config() -> dict:
    """Merged config: file values, then env-var overrides on top."""
    cfg = _load_file_config()

    cores_env = os.environ.get("KAGGLE_TOOLKIT_CORES")
    if cores_env:
        try:
            cfg["cores"] = int(cores_env)
        except ValueError:
            pass

    gpu_env = os.environ.get("KAGGLE_TOOLKIT_USE_GPU")
    if gpu_env is not None:
        cfg["use_gpu"] = gpu_env.strip().lower() in ("1", "true", "yes", "on")

    return cfg


def reload_config() -> None:
    """Clear cached config + GPU detection (call after changing env/file)."""
    _config.cache_clear()
    _detect_gpu.cache_clear()


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def _auto_cores() -> int:
    return os.cpu_count() or 1


def n_jobs() -> int:
    """Worker count. Config `cores` caps it; otherwise all logical cores."""
    configured = _config().get("cores")
    if configured is not None and int(configured) > 0:
        return min(int(configured), _auto_cores())
    return _auto_cores()


def cores_are_capped() -> bool:
    return _config().get("cores") is not None and n_jobs() < _auto_cores()


def sklearn_parallel_kwargs() -> dict:
    """Kwargs to parallelize sklearn utilities that accept n_jobs."""
    return {"n_jobs": n_jobs()}


# ---------------------------------------------------------------------------
# GPU (config override, else auto-detect via torch)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _detect_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_available() -> bool:
    """Whether to use the GPU. Config `use_gpu` overrides; else auto-detect.

    If use_gpu is forced True but no CUDA is detected, we still honor the
    request only when torch actually reports a device (never fabricate a GPU).
    """
    forced = _config().get("use_gpu")
    if forced is False:
        return False
    if forced is True:
        return _detect_gpu()  # respect the flag but don't invent a device
    return _detect_gpu()


def _gpu_source() -> str:
    forced = _config().get("use_gpu")
    if forced is False:
        return "off (config)"
    if forced is True:
        return "on (config)" if _detect_gpu() else "requested but no CUDA"
    return "GPU" if _detect_gpu() else "CPU-only"


# ---------------------------------------------------------------------------
# Gradient boosters
# ---------------------------------------------------------------------------

def lgbm_params(force_cpu: bool = False) -> dict:
    """Compute kwargs for LightGBM (all cores; GPU device when enabled)."""
    params = {"n_jobs": n_jobs()}
    if gpu_available() and not force_cpu:
        params["device_type"] = "gpu"
    return params


def xgb_params(force_cpu: bool = False) -> dict:
    """Compute kwargs for XGBoost >=2.0 (n_jobs + hist; device=cuda on GPU)."""
    params = {"n_jobs": n_jobs(), "tree_method": "hist"}
    if gpu_available() and not force_cpu:
        params["device"] = "cuda"
    return params


def catboost_params(force_cpu: bool = False) -> dict:
    """Compute kwargs for CatBoost (task_type=GPU, else thread_count)."""
    if gpu_available() and not force_cpu:
        return {"task_type": "GPU"}
    return {"thread_count": n_jobs()}


def summary() -> str:
    """One-line description of the resolved compute environment for logging."""
    cap = " (capped)" if cores_are_capped() else ""
    return f"cores={n_jobs()}{cap} | {_gpu_source()}"
