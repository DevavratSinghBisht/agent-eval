from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame for a missing or empty file."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
