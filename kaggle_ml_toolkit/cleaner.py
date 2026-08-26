"""Data cleaning utilities for Kaggle ML Toolkit.

Provides the DataCleaner class with immutable transformations for handling
missing values, encoding categorical columns, and detecting outliers.
Every method returns a new DataFrame without modifying the original.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class DataCleaner:
    """Stateful cleaner that stores encoding mappings for reuse.

    All transformation methods return new DataFrames — the original input
    is never modified. Encoding mappings are stored internally so that
    the same transformations can be applied to new data via transform_new().

    Attributes:
        _encodings: Internal dictionary storing encoding mappings keyed
            by column name.
    """

    def __init__(self) -> None:
        self._encodings: Dict[str, Any] = {}

    def impute_numeric(
        self,
        df: pd.DataFrame,
        columns: List[str],
        strategy: str = "median",
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """Impute missing numeric values.

        Args:
            df: Input DataFrame containing numeric columns with missing values.
            columns: List of column names to impute.
            strategy: Imputation strategy. One of 'mean', 'median', or
                'constant'.
            fill_value: Value to use when strategy is 'constant'. Required
                if strategy is 'constant'.

        Returns:
            A new DataFrame with missing values filled in the specified columns.

        Raises:
            ValueError: If a column is entirely missing values and cannot
                be imputed, or if strategy is 'constant' and fill_value
                is None.
        """
        if strategy == "constant" and fill_value is None:
            raise ValueError(
                "fill_value must be provided when strategy is 'constant'"
            )

        result = df.copy()

        for col in columns:
            if result[col].isna().all():
                raise ValueError(
                    f"Column '{col}' is entirely null and cannot be imputed"
                )

            if strategy == "mean":
                result[col] = result[col].fillna(result[col].mean())
            elif strategy == "median":
                result[col] = result[col].fillna(result[col].median())
            elif strategy == "constant":
                result[col] = result[col].fillna(fill_value)
            else:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    "Must be one of: 'mean', 'median', 'constant'"
                )

        return result

    def impute_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        strategy: str = "mode",
        fill_value: Optional[str] = None,
    ) -> pd.DataFrame:
        """Impute missing categorical values.

        Args:
            df: Input DataFrame containing categorical columns with missing
                values.
            columns: List of column names to impute.
            strategy: Imputation strategy. One of 'mode' or 'constant'.
            fill_value: Value to use when strategy is 'constant'. Required
                if strategy is 'constant'.

        Returns:
            A new DataFrame with missing values filled in the specified columns.

        Raises:
            ValueError: If a column is entirely missing values and cannot
                be imputed, or if strategy is 'constant' and fill_value
                is None.
        """
        if strategy == "constant" and fill_value is None:
            raise ValueError(
                "fill_value must be provided when strategy is 'constant'"
            )

        result = df.copy()

        for col in columns:
            if result[col].isna().all():
                raise ValueError(
                    f"Column '{col}' is entirely null and cannot be imputed"
                )

            if strategy == "mode":
                mode_value = result[col].mode().iloc[0]
                result[col] = result[col].fillna(mode_value)
            elif strategy == "constant":
                result[col] = result[col].fillna(fill_value)
            else:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    "Must be one of: 'mode', 'constant'"
                )

        return result

    def encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "onehot",
    ) -> pd.DataFrame:
        """Encode categorical columns.

        Stores encoding mappings internally so that the same encoding
        can be applied to new data via transform_new().

        Args:
            df: Input DataFrame containing categorical columns to encode.
            columns: List of column names to encode.
            method: Encoding method. One of 'onehot' or 'label'.

        Returns:
            A new DataFrame with encoded columns. For 'onehot', the
            original columns are replaced by dummy columns. For 'label',
            the original columns are replaced with integer codes.

        Raises:
            ValueError: If method is not 'onehot' or 'label'.
        """
        if method not in ("onehot", "label"):
            raise ValueError(
                f"Unknown encoding method '{method}'. "
                "Must be one of: 'onehot', 'label'"
            )

        result = df.copy()

        for col in columns:
            if method == "onehot":
                categories = result[col].dropna().unique().tolist()
                dummies = pd.get_dummies(result[col], prefix=col).astype(int)
                # Store the encoding mapping
                self._encodings[col] = {
                    "method": "onehot",
                    "categories": categories,
                    "columns": dummies.columns.tolist(),
                }
                # Drop original column and add dummy columns
                result = result.drop(columns=[col])
                result = pd.concat([result, dummies], axis=1)

            elif method == "label":
                unique_values = result[col].dropna().unique().tolist()
                mapping = {val: code for code, val in enumerate(unique_values)}
                self._encodings[col] = {
                    "method": "label",
                    "mapping": mapping,
                }
                result[col] = result[col].map(mapping)

        return result

    def transform_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all stored encodings to new data.

        For 'onehot': uses pd.get_dummies and reindexes to match stored
        columns, filling missing categories with 0.

        For 'label': maps values using the stored mapping. Unknown values
        that were not seen during the original encode() call receive -1.

        Args:
            df: Input DataFrame to transform using stored encodings.

        Returns:
            A new DataFrame with stored encodings applied.
        """
        result = df.copy()

        for col, encoding_info in self._encodings.items():
            if col not in result.columns:
                continue

            if encoding_info["method"] == "onehot":
                stored_columns = encoding_info["columns"]
                dummies = pd.get_dummies(result[col], prefix=col).astype(int)
                # Reindex to match stored columns, fill missing with 0
                dummies = dummies.reindex(columns=stored_columns, fill_value=0)
                result = result.drop(columns=[col])
                result = pd.concat([result, dummies], axis=1)

            elif encoding_info["method"] == "label":
                mapping = encoding_info["mapping"]
                # Map known values; unknown values get -1
                result[col] = result[col].map(mapping).fillna(-1).astype(int)

        return result

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, Any]:
        """Detect outliers in numeric columns without modifying the DataFrame.

        Args:
            df: Input DataFrame containing numeric columns to check.
            columns: List of column names to check for outliers.
            method: Detection method. One of 'iqr' or 'zscore'.
            threshold: For 'iqr', the IQR multiplier (default 1.5).
                For 'zscore', the z-score cutoff (default 3.0).

        Returns:
            A report dictionary with structure:
                {
                    "columns": {
                        "col_name": {
                            "count": int,
                            "indices": List[int],
                        }
                    },
                    "total_outliers": int,
                    "method": str,
                    "threshold": float,
                }

        Raises:
            ValueError: If method is not 'iqr' or 'zscore'.
        """
        if method not in ("iqr", "zscore"):
            raise ValueError(
                f"Unknown method '{method}'. Must be one of: 'iqr', 'zscore'"
            )

        report: Dict[str, Any] = {
            "columns": {},
            "total_outliers": 0,
            "method": method,
            "threshold": threshold,
        }

        for col in columns:
            series = df[col].dropna()

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:  # zscore
                mean = series.mean()
                std = series.std()
                if std == 0:
                    outlier_mask = pd.Series(False, index=df.index)
                else:
                    z_scores = (df[col] - mean) / std
                    outlier_mask = z_scores.abs() > threshold

            # Exclude NaN positions from outlier detection
            outlier_mask = outlier_mask & df[col].notna()
            indices = df.index[outlier_mask].tolist()

            report["columns"][col] = {
                "count": len(indices),
                "indices": indices,
            }
            report["total_outliers"] += len(indices)

        return report
