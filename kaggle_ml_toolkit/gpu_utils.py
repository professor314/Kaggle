"""GPU utilities for the Kaggle ML Toolkit (Phase 3).

Small helpers for device selection, GPU inspection, seeding, and mixed
precision. Requires torch (available in the `.venv-dl` environment).
"""

from __future__ import annotations

import os
import random


def get_device():
    """Return a torch.device for CUDA if available, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_info() -> dict:
    """Return GPU name, total VRAM (GB), and current utilization if available."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    info = {
        "cuda": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 1),
        "capability": f"{props.major}.{props.minor}",
        "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1024**3, 2),
    }
    return info


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducibility."""
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(enabled: bool = True):
    """Return a torch.amp.autocast context manager for mixed precision.

    Falls back to a no-op context on CPU.
    """
    import contextlib

    import torch

    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()
