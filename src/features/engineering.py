"""
src/features/engineering.py — Gap-aware rolling feature engineering.

Design decisions
----------------
1. Rolling windows are computed WITHIN each gap_segment_id only — never
   across gap boundaries. A crossing would mix readings from different
   operational sessions separated by maintenance or shutdown, fabricating
   trends that do not exist.

2. Three window sizes (from Stage 2 recommendations):
     - 30 min  → captures rapid within-cycle changes
     - 2 h     → captures slow drift across cycles
     - 6 h     → captures longer thermal accumulation / progressive degradation

3. Feature set (from Stage 2 EDA findings):
     Primary discriminators (large failure/normal separation):
       - H1           (collapses ~97% during failure)
       - load_fraction (rises from ~15% to ~99% during failure)
       - Motor_current (rises from ~1.2A to ~5.5A)
       - Oil_temperature (rises +8 to +19°C during failure)
     Supporting:
       - TP2           (rises during failure)
       - DV_pressure   (rises during failure)
       - LPS           (rolling activation rate — progressive degradation indicator)

4. Rolling statistics computed per feature per window:
     - mean (signal level trend)
     - std  (variability — compressor cycling pattern changes before failure)
     - min, max (extreme values within window)

5. A short-horizon trend (slope) feature is added for H1 and load_fraction
   to capture the direction of change, not just the level.

6. The scaler (StandardScaler) fitted on the anomaly-training split is saved
   here as an artifact so the dashboard can preprocess live data consistently.

Usage:
    from src.features.engineering import build_feature_matrix
    df_features = build_feature_matrix(df_processed)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Rolling window sizes in minutes (as strings for pandas .rolling())
WINDOWS: dict[str, int] = {
    "30m": 30,
    "2h":  120,
    "6h":  360,
}

# Analogue features to compute rolling statistics for
ROLLING_FEATURES = [
    "H1",
    "load_fraction",
    "Motor_current",
    "Oil_temperature",
    "TP2",
    "DV_pressure",
]

# Digital features — rolling mean = activation rate within window
ROLLING_DIGITAL = ["LPS"]

# Trend window in minutes (for slope calculation)
TREND_WINDOW_M = 30

# Minimum number of non-NaN values required in a rolling window to produce output
MIN_PERIODS_FRACTION = 0.5   # at least 50% of the window must be observed


# ---------------------------------------------------------------------------
# Gap-aware rolling
# ---------------------------------------------------------------------------

def _rolling_within_segments(
    series: pd.Series,
    gap_segment_ids: pd.Series,
    window: int,
    stat: str,
) -> pd.Series:
    """
    Compute a rolling statistic for `series`, resetting at each gap boundary.

    Parameters
    ----------
    series:          Numeric sensor series (NaN where is_gap=True).
    gap_segment_ids: gap_segment_id column (NaN at gap rows).
    window:          Rolling window length in number of 1-minute rows.
    stat:            'mean', 'std', 'min', or 'max'.

    Returns
    -------
    Series of the same length with NaN at gap rows and at the start of each
    segment where there are fewer than min_periods valid observations.
    """
    min_periods = max(1, int(window * MIN_PERIODS_FRACTION))
    result = pd.Series(np.nan, index=series.index, dtype="float64")

    for seg_id, grp_idx in gap_segment_ids.dropna().groupby(gap_segment_ids.dropna()):
        seg = series.loc[grp_idx.index]
        roller = seg.rolling(window=window, min_periods=min_periods)
        if stat == "mean":
            rolled = roller.mean()
        elif stat == "std":
            rolled = roller.std()
        elif stat == "min":
            rolled = roller.min()
        elif stat == "max":
            rolled = roller.max()
        else:
            raise ValueError(f"Unknown stat: {stat}")
        result.iloc[result.index.get_indexer(grp_idx.index)] = rolled.values

    return result.astype("float32")


def _trend_within_segments(
    series: pd.Series,
    gap_segment_ids: pd.Series,
    window: int,
) -> pd.Series:
    """
    Compute the linear slope (per-minute change) of `series` over `window`
    minutes, within each gap segment.

    Uses a vectorised rolling-window linear regression:
      slope = (rolling_mean(t*x) - rolling_mean(t)*rolling_mean(x)) / var(t)
    where t is the integer time offset within the window (0, 1, ..., w-1).

    A positive slope means the signal is rising; negative means falling.
    Returns NaN where insufficient data is available.
    """
    min_periods = max(2, int(window * MIN_PERIODS_FRACTION))
    result = pd.Series(np.nan, index=series.index, dtype="float64")

    # Pre-compute t = rolling index offset series (same for all segments)
    # Trick: create a t-series where each row = its position within the window
    # We compute rolling cov(t, x) / var(t) using rolling statistics.
    # t for window of size w: mean(t) = (w-1)/2, var(t) = (w^2-1)/12
    t_mean = (window - 1) / 2.0
    t_var  = (window ** 2 - 1) / 12.0  # variance of 0..w-1

    for seg_id, grp_idx in gap_segment_ids.dropna().groupby(gap_segment_ids.dropna()):
        seg = series.loc[grp_idx.index].astype("float64")
        n = len(seg)

        # Build a t-index series (0, 1, 2, ...) of same length
        t_idx = np.arange(n, dtype="float64")
        t_series = pd.Series(t_idx, index=seg.index)

        # Rolling statistics
        roller_x  = seg.rolling(window=window, min_periods=min_periods)
        roller_tx = (t_series * seg).rolling(window=window, min_periods=min_periods)
        roller_t  = t_series.rolling(window=window, min_periods=min_periods)

        # Within each window the t values are NOT 0..w-1 but i-w+1..i
        # Use rolling mean of t and rolling mean of t*x
        roll_x_mean  = roller_x.mean()
        roll_t_mean  = roller_t.mean()
        roll_tx_mean = roller_tx.mean()

        # cov(t, x) ≈ E[t*x] - E[t]*E[x]
        cov_tx = roll_tx_mean - roll_t_mean * roll_x_mean
        # var(t) is constant within each full window = (w^2-1)/12
        # For partial windows (at start), var(t) differs; skip those by min_periods
        slopes = cov_tx / t_var

        result.iloc[result.index.get_indexer(grp_idx.index)] = slopes.values

    return result.astype("float32")


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    include_raw: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix from the processed 1-minute DataFrame.

    Includes:
    - Raw analogue readings and digital signals (if include_raw=True)
    - Rolling mean/std/min/max for ROLLING_FEATURES at each window size
    - Rolling activation rate for LPS
    - 30-minute linear trend for H1 and load_fraction
    - Derived operational features already in the processed data

    Gap rows (is_gap=True) are retained with NaN feature values so that
    downstream code can filter them consistently.

    Parameters
    ----------
    df:          Processed 1-minute DataFrame (output of preprocessor).
    include_raw: If True, include raw sensor readings in the output matrix.

    Returns
    -------
    DataFrame with DatetimeIndex, same length as input, with feature columns.
    """
    logger.info("Building feature matrix for %d rows …", len(df))

    gap_ids = df["gap_segment_ids"] if "gap_segment_ids" in df.columns else df.get(
        "gap_segment_id", pd.Series(np.nan, index=df.index)
    )

    features: dict[str, pd.Series] = {}

    # ── Raw readings ──────────────────────────────────────────────────────────
    if include_raw:
        for col in ROLLING_FEATURES:
            if col in df.columns:
                features[col] = df[col].astype("float32")
        for col in ROLLING_DIGITAL:
            if col in df.columns:
                features[col] = df[col].astype("float32")
        # Additional raw features that discriminate failure state
        for col in ["pressure_diff", "reservoir_panel_diff"]:
            if col in df.columns:
                features[col] = df[col].astype("float32")

    # ── Rolling features ──────────────────────────────────────────────────────
    for window_name, window_m in WINDOWS.items():
        for feat in ROLLING_FEATURES:
            if feat not in df.columns:
                continue
            series = df[feat].where(~df["is_gap"]).astype("float64")
            seg_ids = df["gap_segment_id"]
            for stat in ("mean", "std", "min", "max"):
                col_name = f"{feat}_{window_name}_{stat}"
                features[col_name] = _rolling_within_segments(
                    series, seg_ids, window_m, stat
                )
                logger.debug("Computed %s", col_name)

        # LPS activation rate within window
        if "LPS" in df.columns:
            lps = df["LPS"].where(~df["is_gap"]).astype("float64")
            col_name = f"LPS_{window_name}_rate"
            features[col_name] = _rolling_within_segments(
                lps, df["gap_segment_id"], window_m, "mean"
            )

    # ── Trend features ────────────────────────────────────────────────────────
    for feat in ("H1", "load_fraction"):
        if feat in df.columns:
            series = df[feat].where(~df["is_gap"]).astype("float64")
            features[f"{feat}_trend30m"] = _trend_within_segments(
                series, df["gap_segment_id"], TREND_WINDOW_M
            )

    # ── Assemble ──────────────────────────────────────────────────────────────
    feature_df = pd.DataFrame(features, index=df.index)

    # Preserve metadata columns for downstream use
    for meta_col in ["is_gap", "gap_segment_id", "failure_id", "motor_state", "is_loaded"]:
        if meta_col in df.columns:
            feature_df[meta_col] = df[meta_col]

    n_feat = len([c for c in feature_df.columns if c not in
                  ["is_gap", "gap_segment_id", "failure_id", "motor_state", "is_loaded"]])
    logger.info(
        "Feature matrix built: %d rows × %d feature columns",
        len(feature_df), n_feat,
    )
    return feature_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the names of numeric feature columns (excludes metadata)."""
    meta = {"is_gap", "gap_segment_id", "failure_id", "motor_state", "is_loaded"}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def build_model_ready(
    df_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    From the feature matrix, return (X, meta_series) for model use.

    - Drops gap rows
    - Drops rows where >50% of features are NaN (window startup artifacts)
    - Returns X (float32 feature matrix) and the failure_id metadata column

    Parameters
    ----------
    df_features: Output of build_feature_matrix().

    Returns
    -------
    X:        DataFrame of feature columns, gap rows removed.
    meta:     Series of failure_id for the same rows.
    """
    # Drop gap rows
    df_clean = df_features[~df_features["is_gap"]].copy()

    feat_cols = get_feature_columns(df_clean)
    X = df_clean[feat_cols]

    # Drop rows where more than half the features are NaN
    # (these occur at the start of each segment before windows fill)
    too_many_nan = X.isna().sum(axis=1) > (len(feat_cols) * 0.5)
    n_dropped = int(too_many_nan.sum())
    if n_dropped > 0:
        logger.info("Dropping %d rows with >50%% NaN features (window startup)", n_dropped)
    X = X[~too_many_nan]

    # Fill remaining NaN (isolated gaps) with column median
    X = X.fillna(X.median())
    X = X.astype("float32")

    meta = df_clean.loc[X.index, "failure_id"]
    return X, meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.loader import load_parquet
    df = load_parquet(PROCESSED_DIR / "processed_1min.parquet")
    feat_df = build_feature_matrix(df)
    X, meta = build_model_ready(feat_df)
    print(f"Feature matrix: {X.shape}")
    print(f"Columns: {list(X.columns)[:10]} …")
    print(f"Failure rows: {(meta != '').sum()}")
