"""Time series utilities for lag features, rolling stats, and temporal CV.

Provides reusable components for any time series forecasting competition.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Sequence


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based time features from a date column.

    Adds: day_of_week, day_of_month, month, week_of_year, is_weekend,
          is_month_start, is_month_end, quarter.
    """
    dt = pd.to_datetime(df[date_col])
    df = df.copy()
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    lags: Sequence[int] = (16, 21, 28, 35, 42),
) -> pd.DataFrame:
    """Add lag features for a target column, grouped by specified columns.

    Args:
        df: DataFrame sorted by date within groups.
        target_col: Column to compute lags on.
        group_cols: Columns defining the time series (e.g., ['store_nbr', 'family']).
        lags: Lag periods to compute.

    Returns:
        DataFrame with new lag columns.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    windows: Sequence[int] = (7, 14, 28),
    min_shift: int = 16,
) -> pd.DataFrame:
    """Add rolling mean and std features, shifted to avoid data leakage.

    Args:
        df: DataFrame sorted by date within groups.
        target_col: Column to compute rolling stats on.
        group_cols: Columns defining the time series.
        windows: Rolling window sizes.
        min_shift: Minimum shift before the window starts (prevents leakage for forecast horizon).

    Returns:
        DataFrame with rolling mean and std columns.
    """
    df = df.copy()
    grouped = df.groupby(group_cols)[target_col]

    for window in windows:
        shifted = grouped.shift(min_shift)
        rolling = shifted.rolling(window, min_periods=1)

        df[f"{target_col}_roll_mean_{window}"] = rolling.mean().values
        df[f"{target_col}_roll_std_{window}"] = rolling.std().values

    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_start: str | None = None,
    val_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by date for time-based validation.

    Args:
        df: Full DataFrame with a date column.
        date_col: Name of the date column.
        train_end: Last date (inclusive) for training.
        val_start: First date (inclusive) for validation. Defaults to day after train_end.
        val_end: Last date (inclusive) for validation. If None, uses all data after val_start.

    Returns:
        (train_df, val_df) tuple.
    """
    dates = pd.to_datetime(df[date_col])

    train_mask = dates <= pd.Timestamp(train_end)
    train_df = df[train_mask].copy()

    if val_start is None:
        val_start_ts = pd.Timestamp(train_end) + pd.Timedelta(days=1)
    else:
        val_start_ts = pd.Timestamp(val_start)

    if val_end is None:
        val_mask = dates >= val_start_ts
    else:
        val_mask = (dates >= val_start_ts) & (dates <= pd.Timestamp(val_end))

    val_df = df[val_mask].copy()

    return train_df, val_df
