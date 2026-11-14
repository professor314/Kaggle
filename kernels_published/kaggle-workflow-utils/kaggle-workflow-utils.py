"""Reusable Kaggle helper utilities.

Set this notebook as a "Utility Script" (File menu) so it can be imported into
other notebooks. Small, dependency-light helpers for competition workflows.
"""

import glob
import os

import numpy as np
import pandas as pd


def find_input(name):
    """Locate a file by name anywhere under /kaggle/input (mount-name agnostic)."""
    matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return matches[0]


def load_train_test(train_name="train.csv", test_name="test.csv"):
    """Load train/test CSVs from the attached competition input."""
    return pd.read_csv(find_input(train_name)), pd.read_csv(find_input(test_name))


def make_submission(ids, preds, id_col="id", target_col="target",
                    path="/kaggle/working/submission.csv"):
    """Write a two-column Kaggle submission file and return the DataFrame."""
    sub = pd.DataFrame({id_col: np.asarray(ids), target_col: np.asarray(preds)})
    sub.to_csv(path, index=False)
    return sub


def sanity_check_distribution(preds, n_classes=None):
    """Return the class distribution of predictions for a pre-submission check."""
    s = pd.Series(np.asarray(preds))
    return s.value_counts().sort_index().to_dict()
