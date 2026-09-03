"""Series selection (Tier 0.3) — pick <=3 informative series per study.

Prefer fluid-sensitive sequences (show effusion/swelling/acute injury), one per
anatomical plane (sagittal/coronal/axial), fill remaining slots from the rest.
Pure function over series-metadata dicts so it unit-tests without Kaggle.

Expected row keys (from *_series.csv): SeriesInstanceUID, Anatomical_Plane,
Fluid_Sensitive (0/1). Missing keys degrade gracefully.
"""
from __future__ import annotations

from typing import Dict, List

PLANE_ORDER = ["Sagittal", "Coronal", "Axial"]


def _plane(row: Dict) -> str:
    return str(row.get("Anatomical_Plane", "") or "")


def _fluid(row: Dict) -> int:
    try:
        return int(row.get("Fluid_Sensitive", 0) or 0)
    except (ValueError, TypeError):
        return 0


def pick_series(rows: List[Dict], k: int = 3) -> List[Dict]:
    """Return up to k series rows, preferring fluid-sensitive + one per plane.

    Strategy:
      1. For each plane in PLANE_ORDER, take its best series (fluid-sensitive
         first), one per plane, until we have k.
      2. If still short, fill from remaining series, fluid-sensitive first.
    """
    if not rows:
        return []
    used = set()
    picked: List[Dict] = []

    def take(row):
        picked.append(row)
        used.add(row.get("SeriesInstanceUID", id(row)))

    # one per plane, fluid-sensitive preferred
    for plane in PLANE_ORDER:
        if len(picked) >= k:
            break
        cands = [r for r in rows
                 if _plane(r) == plane and r.get("SeriesInstanceUID", id(r)) not in used]
        if not cands:
            continue
        cands.sort(key=lambda r: (-_fluid(r),))
        take(cands[0])

    # fill remaining slots, fluid-sensitive preferred
    if len(picked) < k:
        rest = [r for r in rows if r.get("SeriesInstanceUID", id(r)) not in used]
        rest.sort(key=lambda r: (-_fluid(r),))
        for r in rest:
            if len(picked) >= k:
                break
            take(r)

    return picked[:k]
