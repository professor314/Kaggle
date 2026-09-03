"""Fold assignment (Tier 0.4) — GroupKFold(5) by study, EXCLUDING the 58 gold.

The 58 gold-labeled studies must never enter a training fold, or the honest
gold-58 metric leaks. Gold studies get fold = -1 (held out for evaluation only);
every other study is split into n_splits GroupKFold folds keyed by StudyInstanceUID.

Pure function over a DataFrame so it unit-tests locally.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

FINDINGS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"


def is_gold(df: pd.DataFrame, findings: List[str] = FINDINGS) -> pd.Series:
    """Rows with at least one non-null expert label (the 58 gold studies)."""
    num = df[findings].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    return num.notna().any(axis=1)


def assign_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                 findings: List[str] = FINDINGS) -> pd.Series:
    """Return an int fold per row: -1 for gold (held out), 0..n_splits-1 otherwise.

    Grouping is by StudyInstanceUID so all slices/series of a study share a fold.
    """
    gold = is_gold(df, findings)
    fold = pd.Series(np.full(len(df), -1, dtype=int), index=df.index)

    train_df = df[~gold]
    if len(train_df) == 0:
        return fold

    from sklearn.model_selection import GroupKFold
    groups = train_df[ID_COL].astype(str).values
    # GroupKFold is deterministic; shuffle groups by a seeded permutation of unique ids
    uniq = pd.unique(groups)
    rng = np.random.RandomState(seed)
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    order = np.array([perm[g] for g in groups])
    sort_idx = np.argsort(order, kind="stable")
    inv = np.empty_like(sort_idx)
    inv[sort_idx] = np.arange(len(sort_idx))

    gkf = GroupKFold(n_splits=min(n_splits, len(uniq)))
    fold_of_row = np.empty(len(train_df), dtype=int)
    sorted_groups = groups[sort_idx]
    dummy_X = np.zeros(len(train_df))
    for f, (_, val_idx) in enumerate(gkf.split(dummy_X, groups=sorted_groups)):
        fold_of_row[sort_idx[val_idx]] = f

    fold.loc[train_df.index] = fold_of_row
    return fold
