"""Data loading utilities for Kaggle ML Toolkit.

Provides functions to load competition datasets (train.csv, test.csv) into
pandas DataFrames with validation and clear error reporting.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataBundle:
    """Named container for train/test DataFrames.

    Attributes:
        train: The training DataFrame loaded from train.csv.
        test: The test DataFrame loaded from test.csv.
    """

    train: pd.DataFrame
    test: pd.DataFrame


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a single CSV file into a DataFrame.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        A pandas DataFrame with original column names and inferred types.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file contains no data rows.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError(f"No data rows found in file: {file_path}")

    return df


def load_competition_data(directory: str) -> DataBundle:
    """Load train.csv and test.csv from a competition directory.

    Args:
        directory: Path to the directory containing train.csv and test.csv.

    Returns:
        A DataBundle with train and test DataFrames.

    Raises:
        FileNotFoundError: If the directory does not exist, or if either
            train.csv or test.csv is missing from the directory.
        ValueError: If either file contains no data rows.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    train_path = dir_path / "train.csv"
    test_path = dir_path / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing expected file: train.csv in {directory}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing expected file: test.csv in {directory}"
        )

    train_df = load_csv(str(train_path))
    test_df = load_csv(str(test_path))

    return DataBundle(train=train_df, test=test_df)
