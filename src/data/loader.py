"""
src/data/loader.py — Efficient loader for the MetroPT-3 CSV.

Loads in chunks using pyarrow/pandas and saves to Parquet for fast
downstream access. Raw CSV is never modified.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(
    path: Path,
    nrows: int | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Load the MetroPT-3 CSV efficiently.

    Parameters
    ----------
    path:      Path to the raw CSV file.
    nrows:     Optional row limit (for development/testing).
    chunksize: Rows per chunk during full load.

    Returns
    -------
    DataFrame with datetime index named 'timestamp'.
    """
    logger.info("Loading raw CSV: %s", path)

    dtype_map = {
        "TP2": "float32", "TP3": "float32", "H1": "float32",
        "DV_pressure": "float32", "Reservoirs": "float32",
        "Oil_temperature": "float32", "Motor_current": "float32",
        "COMP": "int8", "DV_eletric": "int8", "Towers": "int8",
        "MPG": "int8", "LPS": "int8", "Pressure_switch": "int8",
        "Oil_level": "int8", "Caudal_impulses": "int8",
    }

    if nrows is not None:
        df = pd.read_csv(
            path, index_col=0, parse_dates=["timestamp"], dtype=dtype_map, nrows=nrows
        )
        df = df.set_index("timestamp").sort_index()
        return df

    # Chunked load for full dataset
    chunks = []
    reader = pd.read_csv(
        path, index_col=0, parse_dates=["timestamp"], dtype=dtype_map,
        chunksize=chunksize,
    )
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        logger.debug("Loaded chunk %d (%d rows)", i, len(chunk))

    df = pd.concat(chunks, ignore_index=False)
    df = df.set_index("timestamp").sort_index()
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as Parquet (fast, compressed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")
    logger.info("Saved %d rows to %s", len(df), path)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a previously saved Parquet file."""
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df
