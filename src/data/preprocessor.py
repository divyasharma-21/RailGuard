"""
src/data/preprocessor.py — Preprocessing pipeline for MetroPT-3.

Aggregation decision: 1-minute resolution
──────────────────────────────────────────
The raw data has an effective sampling interval of ~10 seconds (confirmed in
Stage 1). We resample to 1-minute buckets for the following reasons:

1. Noise reduction: 10s jitter in timestamps (9–12s gaps are normal) produces
   ragged time series. 1-minute buckets absorb this jitter cleanly.
2. Scale: 1.5M rows at 10s is expensive for rolling-window feature work and
   downstream modeling. 1min gives ~215K rows — tractable for all ML methods.
3. Operational granularity: air-compressor cycles span minutes. The documented
   failure signatures (continuous load operation, temperature drift) develop
   over minutes to hours, not seconds. 1-minute resolution preserves all
   operationally relevant dynamics.
4. Alignment: the config.yaml already specifies `resample_freq: "1min"`.

Aggregation rules:
  • Analogue sensors → mean over the minute (preserves signal level)
  • Analogue sensors → also retain max (detects transient spikes within bucket)
  • Digital sensors  → max (1 if the signal was ever active in the minute)
  • Count of raw 10s samples per minute → `n_samples` (for gap detection)

Gap handling:
  • After resampling, minutes with n_samples == 0 are genuine missing windows.
  • A boolean flag `is_gap` marks these rows.
  • The flag `gap_segment_id` groups consecutive non-gap rows into continuous
    operational segments so downstream models never cross gap boundaries.
  • We do NOT interpolate across gaps (doing so would fabricate sensor readings
    during e.g. maintenance shutdowns).

Operational mode features:
  • `motor_state`: categorical from Motor_current
      – 'off'       Motor_current ≈ 0A
      – 'offloaded' ≈ 4A
      – 'loaded'    ≈ 7A
      – 'starting'  ≈ 9A   (transient)
      – 'unknown'   anything that does not fit cleanly
  • `is_loaded`: bool — compressor working under full load
  • `load_fraction`: fraction of raw 10s samples in the minute where
    DV_eletric == 1 (outlet valve open = under load)

Usage:
    python -m src.data.preprocessor          # run full pipeline
    from src.data.preprocessor import run_preprocessing
    df_proc = run_preprocessing()
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANALOGUE_SENSORS,
    ARTIFACTS_DIR,
    DIGITAL_SENSORS,
    PROCESSED_DIR,
    RAW_CSV,
    RESAMPLE_FREQ,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Motor-current state boundaries (from documentation)
# Midpoints between documented nominal values: 0, 4, 7, 9 A
# ---------------------------------------------------------------------------
_MOTOR_BOUNDARIES = [
    (0.0,  1.5,  "off"),
    (1.5,  5.5,  "offloaded"),
    (5.5,  8.0,  "loaded"),
    (8.0,  12.0, "starting"),
]


def _classify_motor_state(current: float) -> str:
    """Return the documented operational state for a motor current reading."""
    for lo, hi, label in _MOTOR_BOUNDARIES:
        if lo <= current < hi:
            return label
    return "unknown"


def _assign_motor_state_series(s: pd.Series) -> pd.Series:
    """Vectorised motor-state classification."""
    conditions = [
        (s >= lo) & (s < hi) for lo, hi, _ in _MOTOR_BOUNDARIES
    ]
    choices = [label for _, _, label in _MOTOR_BOUNDARIES]
    result = np.select(conditions, choices, default="unknown")
    return pd.Series(result, index=s.index, dtype="category")


def load_and_validate(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Load the raw CSV and perform minimal structural validation.

    Returns a DataFrame indexed by timestamp (sorted, no duplicates).
    """
    from src.data.loader import load_raw

    df = load_raw(path, nrows=nrows)

    # Ensure correct dtypes after load
    for col in ANALOGUE_SENSORS:
        df[col] = df[col].astype("float32")
    for col in DIGITAL_SENSORS:
        df[col] = df[col].astype("int8")

    # Drop any duplicate timestamps (keep first occurrence)
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < n_before:
        logger.warning("Dropped %d duplicate timestamps", n_before - len(df))

    # Sort chronologically
    df = df.sort_index()
    logger.info("Loaded and validated %d rows", len(df))
    return df


def resample_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the 10s raw data to 1-minute resolution.

    Aggregation per bucket:
      - Analogue: mean (primary) and max (spike detection)
      - Digital: max (was the signal ever active?)
      - n_samples: count of raw rows in the bucket
      - load_fraction: fraction of rows where DV_eletric == 1

    Missing minutes (n_samples == 0) indicate operational gaps and are
    retained as NaN rows with is_gap=True.
    """
    logger.info("Resampling %d raw rows to 1-minute resolution ...", len(df))

    resampler = df.resample(RESAMPLE_FREQ)

    # Analogue means
    analogue_mean = resampler[ANALOGUE_SENSORS].mean()
    analogue_mean.columns = ANALOGUE_SENSORS  # already named correctly

    # Analogue maxima (for spike detection)
    analogue_max = resampler[ANALOGUE_SENSORS].max()
    analogue_max.columns = [f"{c}_max" for c in ANALOGUE_SENSORS]

    # Digital: max within minute (1 if ever active)
    digital_max = resampler[DIGITAL_SENSORS].max()

    # Count of raw 10s samples per 1-min bucket
    n_samples = resampler[ANALOGUE_SENSORS[0]].count().rename("n_samples")

    # Load fraction: fraction of DV_eletric==1 rows in the minute
    load_fraction = resampler["DV_eletric"].mean().rename("load_fraction")

    # Combine all columns
    resampled = pd.concat(
        [analogue_mean, analogue_max, digital_max, n_samples, load_fraction],
        axis=1,
    )

    # n_samples == 0 → genuine gap (no data in that minute)
    resampled["is_gap"] = resampled["n_samples"] == 0

    n_gap = int(resampled["is_gap"].sum())
    n_total = len(resampled)
    logger.info(
        "Resampled to %d 1-minute rows (%d gap rows, %.1f%% coverage)",
        n_total,
        n_gap,
        100 * (1 - n_gap / n_total),
    )
    return resampled


def add_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived operational-mode features.

    New columns:
      motor_state     : categorical state from Motor_current mean
      is_loaded       : bool, compressor under full load
      pressure_diff   : TP3 - TP2  (reservoir vs compressor pressure)
      reservoir_panel_diff: TP3 - Reservoirs  (should be ~0 in healthy system)
      h1_near_zero    : bool, H1 < 0.5 bar (cyclonic separator pressure low)
    """
    df = df.copy()

    # Motor state from mean current in the minute
    motor_mean = df["Motor_current"].fillna(0.0)
    df["motor_state"] = _assign_motor_state_series(motor_mean)

    # is_loaded: DV_eletric was active for ≥50% of the minute
    df["is_loaded"] = df["load_fraction"] >= 0.5

    # Pressure differentials
    df["pressure_diff"] = (df["TP3"] - df["TP2"]).astype("float32")
    df["reservoir_panel_diff"] = (df["TP3"] - df["Reservoirs"]).astype("float32")

    # H1 near-zero flag (strong failure indicator from Stage 1)
    df["h1_near_zero"] = df["H1"] < 0.5

    return df


def assign_gap_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a contiguous-segment ID to each non-gap row.

    gap_segment_id: integer, increments each time a gap is encountered.
    Rows where is_gap=True get segment_id = NaN.
    This allows downstream windows/features to stay within a single segment.
    """
    df = df.copy()
    # Segment breaks: either this row is a gap, or the previous row was a gap
    gap_starts = df["is_gap"] | df["is_gap"].shift(1, fill_value=False)
    segment_id = (~df["is_gap"] & gap_starts.shift(-1, fill_value=False)).cumsum()

    # Simpler approach: cumulative sum of gap→non-gap transitions
    was_gap = df["is_gap"].shift(1, fill_value=True)
    new_segment = (~df["is_gap"]) & was_gap
    seg_ids = new_segment.cumsum()
    seg_ids = seg_ids.where(~df["is_gap"], other=np.nan)
    df["gap_segment_id"] = seg_ids
    return df


def annotate_failure_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `failure_id` column marking rows that fall within a documented
    failure window (from config.yaml).

    Values:
      ''    → normal operation
      'F1'  → within documented failure window F1
      'F2'  → within documented failure window F2
      etc.
    """
    from src.config import FAILURE_EVENTS

    df = df.copy()
    df["failure_id"] = ""

    for event in FAILURE_EVENTS:
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "failure_id"] = event["id"]
        n = int(mask.sum())
        logger.info(
            "Failure %s: %d 1-min rows annotated (%s → %s)",
            event["id"], n, event["start"], event["end"],
        )

    return df


def run_preprocessing(
    nrows: int | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps:
      1. Load and validate raw CSV
      2. Resample to 1-minute resolution
      3. Add operational features
      4. Assign gap segment IDs
      5. Annotate documented failure windows
      6. Save to data/processed/processed_1min.parquet

    Parameters
    ----------
    nrows:        Optional row limit on raw CSV load (for testing).
    save_output:  If True, save result as Parquet.

    Returns
    -------
    Processed DataFrame at 1-minute resolution.
    """
    logger.info("Starting preprocessing pipeline …")

    # 1. Load
    df_raw = load_and_validate(RAW_CSV, nrows=nrows)

    # 2. Resample
    df_1min = resample_to_1min(df_raw)

    # 3. Operational features
    df_1min = add_operational_features(df_1min)

    # 4. Gap segments
    df_1min = assign_gap_segments(df_1min)

    # 5. Failure annotation
    df_1min = annotate_failure_windows(df_1min)

    logger.info(
        "Preprocessing complete: %d rows, %d columns",
        len(df_1min),
        len(df_1min.columns),
    )

    if save_output:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "processed_1min.parquet"
        df_1min.to_parquet(out_path, compression="snappy")
        logger.info("Saved processed data to %s", out_path)

    return df_1min


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_preprocessing()
