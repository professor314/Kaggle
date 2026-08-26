"""Feature engineering utilities for creating informative features."""

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Stateful feature engineer that records transformations for reproducibility.

    All methods return new DataFrames without modifying the original input.
    Transformations are recorded so they can be replayed on new data via
    transform_new().
    """

    def __init__(self) -> None:
        self._transformations: List[Dict[str, Any]] = []

    def create_interaction(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Multiply two numeric columns to create an interaction feature.

        Args:
            df: Input DataFrame.
            col_a: First numeric column name.
            col_b: Second numeric column name.
            name: Optional name for the new column. Defaults to
                "{col_a}_x_{col_b}".

        Returns:
            A new DataFrame with the interaction column appended.

        Raises:
            TypeError: If either column is not numeric.
        """
        if not pd.api.types.is_numeric_dtype(df[col_a]):
            raise TypeError(f"Column '{col_a}' is not numeric")
        if not pd.api.types.is_numeric_dtype(df[col_b]):
            raise TypeError(f"Column '{col_b}' is not numeric")

        if name is None:
            name = f"{col_a}_x_{col_b}"

        result = df.copy()
        result[name] = df[col_a] * df[col_b]

        self._transformations.append(
            {"type": "interaction", "col_a": col_a, "col_b": col_b, "name": name}
        )

        return result

    def bin_continuous(
        self,
        df: pd.DataFrame,
        column: str,
        n_bins: int = 5,
        method: str = "equal_width",
    ) -> pd.DataFrame:
        """Bin a continuous column into discrete categories.

        Args:
            df: Input DataFrame.
            column: Name of the continuous column to bin.
            n_bins: Number of bins to create.
            method: Binning strategy - "equal_width" (pd.cut) or
                "quantile" (pd.qcut).

        Returns:
            A new DataFrame with a "{column}_binned" column appended
            containing integer bin labels (0 to n_bins-1).
        """
        result = df.copy()
        binned_col_name = f"{column}_binned"

        if method == "equal_width":
            binned, bin_edges = pd.cut(
                df[column], bins=n_bins, labels=False, retbins=True
            )
            result[binned_col_name] = binned.astype("Int64")
            self._transformations.append(
                {
                    "type": "bin_continuous",
                    "column": column,
                    "n_bins": n_bins,
                    "method": method,
                    "name": binned_col_name,
                    "bin_edges": bin_edges.tolist(),
                }
            )
        elif method == "quantile":
            binned, bin_edges = pd.qcut(
                df[column], q=n_bins, labels=False, retbins=True, duplicates="drop"
            )
            result[binned_col_name] = binned.astype("Int64")
            self._transformations.append(
                {
                    "type": "bin_continuous",
                    "column": column,
                    "n_bins": n_bins,
                    "method": method,
                    "name": binned_col_name,
                    "bin_edges": bin_edges.tolist(),
                }
            )
        else:
            raise ValueError(
                f"Unknown binning method '{method}'. Use 'equal_width' or 'quantile'."
            )

        return result

    def extract_date_components(
        self, df: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Extract year, month, day, and day_of_week from a datetime column.

        Args:
            df: Input DataFrame.
            column: Name of the datetime column.

        Returns:
            A new DataFrame with four new integer columns appended:
            "{column}_year", "{column}_month", "{column}_day",
            "{column}_dayofweek".

        Raises:
            TypeError: If the column is not a datetime dtype.
        """
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not datetime")

        result = df.copy()
        result[f"{column}_year"] = df[column].dt.year.astype(int)
        result[f"{column}_month"] = df[column].dt.month.astype(int)
        result[f"{column}_day"] = df[column].dt.day.astype(int)
        result[f"{column}_dayofweek"] = df[column].dt.dayofweek.astype(int)

        self._transformations.append({"type": "date_components", "column": column})

        return result

    def transform_new(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replay all recorded transformations on new data.

        Args:
            df: New DataFrame to transform.

        Returns:
            A new DataFrame with all recorded transformations applied.
        """
        result = df.copy()

        for t in self._transformations:
            if t["type"] == "interaction":
                result[t["name"]] = result[t["col_a"]] * result[t["col_b"]]

            elif t["type"] == "bin_continuous":
                bin_edges = np.array(t["bin_edges"])
                # Use the stored bin edges to apply consistent binning
                result[t["name"]] = pd.cut(
                    result[t["column"]],
                    bins=bin_edges,
                    labels=False,
                    include_lowest=True,
                ).astype("Int64")

            elif t["type"] == "date_components":
                col = t["column"]
                result[f"{col}_year"] = result[col].dt.year.astype(int)
                result[f"{col}_month"] = result[col].dt.month.astype(int)
                result[f"{col}_day"] = result[col].dt.day.astype(int)
                result[f"{col}_dayofweek"] = result[col].dt.dayofweek.astype(int)

        return result
