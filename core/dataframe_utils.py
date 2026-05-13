"""Utilities for working with cell-data DataFrames."""

import pandas as pd

# The metadata columns that are always present at the start of a cell data DataFrame.
# These columns are NOT protein/marker measurements.
METADATA_COLUMNS: frozenset[str] = frozenset({"CellID", "Global X", "Global Y"})


def get_marker_columns(df: pd.DataFrame) -> pd.Index:
    """Return column names that are protein/marker measurements (excludes metadata).

    Filters out CellID, Global X, and Global Y regardless of their position
    in the DataFrame.
    """
    return df.columns[~df.columns.isin(METADATA_COLUMNS)]
