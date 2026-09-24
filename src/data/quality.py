"""
src/data/quality.py — Data quality checks for the MetroPT-3 dataset.

Generates a reproducible quality report covering:
  - Structural validation (row/column counts, dtypes)
  - Timestamp continuity and gap analysis
  - Sensor range validation against documented limits
  - Digital signal validity
  - Outlier flags (without silent deletion)
  - Saves a JSON report and a gap-annotation Parquet to data/processed/

Usage (CLI):
    python -m src.data.quality

Usage (API):
    from src.data.quality import run_quality_check
    report = run_quality_check()
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANALOGUE_SENSORS,
    ARTIFACTS_DIR,
    DIGITAL_SENSORS,
    EFFECTIVE_SAMPLING_S,
    PROCESSED_DIR,
    RAW_CSV,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Documented sensor ranges  (from official PDF documentation + physical sense)
# Values marked PHYSICAL are conservative physical-plausibility limits.
# Values marked DOCUMENTED come directly from the data description paper.
# ---------------------------------------------------------------------------
SENSOR_RANGES: dict[str, dict] = {
    # (min_valid, max_valid, source)
    "TP2":            {"min": -0.1,  "max": 12.0,  "unit": "bar", "source": "PHYSICAL"},
    "TP3":            {"min":  0.0,  "max": 12.0,  "unit": "bar", "source": "PHYSICAL"},
    "H1":             {"min": -0.1,  "max": 12.0,  "unit": "bar", "source": "PHYSICAL"},
    "DV_pressure":    {"min": -0.1,  "max": 10.0,  "unit": "bar", "source": "PHYSICAL"},
    "Reservoirs":     {"min":  0.0,  "max": 12.0,  "unit": "bar", "source": "PHYSICAL"},
    "Oil_temperature":{"min": 10.0,  "max": 110.0, "unit": "degC","source": "PHYSICAL"},
    "Motor_current":  {"min":  0.0,  "max": 12.0,  "unit": "A",   "source": "PHYSICAL"},
}

# Gap threshold: gaps > this many seconds flag a data-continuity break.
# We use 5× the effective sampling interval (10s × 5 = 50s) to be tolerant
# of the observed 9-12s jitter, but flag genuine missing stretches.
GAP_THRESHOLD_S: int = 50


def _check_structure(df: pd.DataFrame) -> dict:
    """Validate row/column counts and data types."""
    expected_cols = ANALOGUE_SENSORS + DIGITAL_SENSORS
    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "expected_sensor_cols": expected_cols,
        "missing_cols": missing_cols,
        "extra_cols": extra_cols,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "index_name": df.index.name,
        "index_dtype": str(df.index.dtype),
    }


def _check_nulls(df: pd.DataFrame) -> dict:
    """Count null values per column."""
    nulls = df.isnull().sum()
    return {
        "total_null_cells": int(nulls.sum()),
        "per_column": {c: int(v) for c, v in nulls.items() if v > 0},
        "any_nulls": bool(nulls.sum() > 0),
    }


def _check_duplicates(df: pd.DataFrame) -> dict:
    """
    Check for duplicate rows and duplicate timestamps.

    Note: rows with identical sensor values at *different* timestamps are
    expected in this dataset (steady-state compressor operation produces
    near-identical consecutive readings). Only duplicate timestamps indicate
    a genuine data integrity problem. 'duplicate_rows' is reported for
    completeness but is not flagged as an issue in the quality summary.
    """
    dup_rows = int(df.duplicated().sum())
    dup_ts = int(df.index.duplicated().sum())
    return {
        "duplicate_rows": dup_rows,
        "duplicate_rows_note": (
            "Rows with identical sensor values but different timestamps — "
            "expected for steady-state operation. Not a data quality issue."
        ),
        "duplicate_timestamps": dup_ts,
    }


def _check_gaps(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Analyse timestamp continuity.

    Returns a summary dict and a DataFrame of all gaps above GAP_THRESHOLD_S,
    annotated with the gap duration in seconds and hours.
    """
    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    gap_dist = diffs.value_counts().sort_index()

    # Gaps above threshold
    big = diffs[diffs > GAP_THRESHOLD_S]
    large_gaps = pd.DataFrame({
        "gap_end_ts":    big.index,
        "gap_seconds":   big.values,
    })
    large_gaps["gap_start_ts"] = large_gaps["gap_end_ts"] - pd.to_timedelta(
        large_gaps["gap_seconds"], unit="s"
    )
    large_gaps["gap_hours"] = (large_gaps["gap_seconds"] / 3600).round(3)
    large_gaps = large_gaps[
        ["gap_start_ts", "gap_end_ts", "gap_seconds", "gap_hours"]
    ].reset_index(drop=True)

    summary = {
        "total_gaps_above_threshold": int(len(large_gaps)),
        "gap_threshold_s": GAP_THRESHOLD_S,
        "dominant_gap_s": int(diffs.mode().iloc[0]),
        "min_gap_s": float(diffs.min()),
        "max_gap_s": float(diffs.max()),
        "mean_gap_s": float(diffs.mean().round(2)),
        "total_missing_time_hours": float(
            large_gaps["gap_hours"].sum().round(2)
        ),
        "gap_distribution_top10": {
            str(int(k)): int(v)
            for k, v in gap_dist.nlargest(10).items()
        },
    }
    return summary, large_gaps


def _check_sensor_ranges(df: pd.DataFrame) -> dict:
    """
    Validate analogue sensors against documented/physical range limits.
    Flags out-of-range readings without removing them.
    """
    results = {}
    for col, limits in SENSOR_RANGES.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        below = int((s < limits["min"]).sum())
        above = int((s > limits["max"]).sum())
        results[col] = {
            "unit": limits["unit"],
            "range_source": limits["source"],
            "expected_min": limits["min"],
            "expected_max": limits["max"],
            "observed_min": float(s.min().round(4)),
            "observed_max": float(s.max().round(4)),
            "observed_mean": float(s.mean().round(4)),
            "observed_std": float(s.std().round(4)),
            "n_below_min": below,
            "n_above_max": above,
            "range_ok": below == 0 and above == 0,
        }
    return results


def _check_digital_signals(df: pd.DataFrame) -> dict:
    """
    Validate digital sensors: must be 0 or 1 only, report activation rates
    and transition counts.
    """
    results = {}
    for col in DIGITAL_SENSORS:
        if col not in df.columns:
            continue
        s = df[col]
        unique_vals = sorted(s.unique().tolist())
        invalid = s[~s.isin([0, 1])].count()
        n = len(s)
        n_active = int((s == 1).sum())
        n_transitions = int((s.diff().abs() > 0).sum())
        results[col] = {
            "unique_values": [int(v) for v in unique_vals],
            "invalid_values": int(invalid),
            "n_active_1": n_active,
            "pct_active": round(100 * n_active / n, 2),
            "n_inactive_0": n - n_active,
            "pct_inactive": round(100 * (n - n_active) / n, 2),
            "n_transitions": n_transitions,
            "valid": invalid == 0 and set(unique_vals).issubset({0, 1}),
        }
    return results


def _flag_outliers(df: pd.DataFrame) -> dict:
    """
    Flag statistical outliers using IQR method on analogue sensors.
    Does NOT remove them — returns counts and examples only.
    IQR outlier = value < Q1 - 3*IQR  or  value > Q3 + 3*IQR  (3× is conservative).
    """
    results = {}
    for col in ANALOGUE_SENSORS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        flags = (s < lo) | (s > hi)
        outlier_vals = s[flags]
        results[col] = {
            "iqr_fence_low": float(round(lo, 4)),
            "iqr_fence_high": float(round(hi, 4)),
            "n_outliers": int(flags.sum()),
            "pct_outliers": float(round(100 * flags.mean(), 3)),
            # Store up to 5 example timestamps + values for inspection
            "examples": [
                {"ts": str(ts), "value": float(round(val, 4))}
                for ts, val in outlier_vals.head(5).items()
            ],
        }
    return results


def run_quality_check(
    df: pd.DataFrame | None = None,
    save_outputs: bool = True,
) -> dict:
    """
    Run the full quality check pipeline.

    Parameters
    ----------
    df:           Pre-loaded DataFrame (timestamp index). If None, loads from RAW_CSV.
    save_outputs: If True, saves JSON report and gap table to data/processed/.

    Returns
    -------
    report: dict with all quality check results.
    """
    if df is None:
        logger.info("Loading raw CSV for quality check …")
        from src.data.loader import load_raw
        df = load_raw(RAW_CSV)

    logger.info("Running quality checks on %d rows …", len(df))

    report: dict = {}
    report["structure"]      = _check_structure(df)
    report["nulls"]          = _check_nulls(df)
    report["duplicates"]     = _check_duplicates(df)
    report["gaps"], gap_df   = _check_gaps(df)
    report["sensor_ranges"]  = _check_sensor_ranges(df)
    report["digital_signals"]= _check_digital_signals(df)
    report["outliers"]       = _flag_outliers(df)

    # Overall pass/fail summary
    issues = []
    if report["nulls"]["any_nulls"]:
        issues.append("null_values_present")
    # Note: duplicate_rows is NOT flagged — same-value rows at different
    # timestamps are expected and benign (see _check_duplicates docstring).
    if report["duplicates"]["duplicate_timestamps"] > 0:
        issues.append("duplicate_timestamps")
    for col, r in report["sensor_ranges"].items():
        if not r["range_ok"]:
            issues.append(f"out_of_range:{col}")
    for col, r in report["digital_signals"].items():
        if not r["valid"]:
            issues.append(f"invalid_digital:{col}")

    report["summary"] = {
        "issues_found": issues,
        "n_issues": len(issues),
        "quality_ok": len(issues) == 0,
    }

    if save_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_path = PROCESSED_DIR / "quality_report.json"
        # Convert non-serialisable types before saving
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Quality report saved to %s", report_path)

        # Save gap table as Parquet for downstream use
        if len(gap_df) > 0:
            gap_path = PROCESSED_DIR / "timestamp_gaps.parquet"
            gap_df.to_parquet(gap_path)
            logger.info("Gap table (%d gaps) saved to %s", len(gap_df), gap_path)

    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    """Print a concise quality summary to stdout."""
    s = report["structure"]
    g = report["gaps"]
    sep = "=" * 60
    print("\n" + sep)
    print("  RailGuard -- Data Quality Report")
    print(sep)
    print(f"  Rows:            {s['n_rows']:,}")
    print(f"  Sensor columns:  {s['n_cols']}")
    print(f"  Null cells:      {report['nulls']['total_null_cells']}")
    print(f"  Duplicate rows:  {report['duplicates']['duplicate_rows']}")
    print(f"  Duplicate ts:    {report['duplicates']['duplicate_timestamps']}")
    print(f"  Dominant gap:    {g['dominant_gap_s']}s")
    print(f"  Gaps > {g['gap_threshold_s']}s:     {g['total_gaps_above_threshold']}")
    print(f"  Missing time:    {g['total_missing_time_hours']:.1f} hours")
    print()
    print("  Sensor range check:")
    for col, r in report["sensor_ranges"].items():
        flag = "!!" if not r["range_ok"] else "  "
        status = "OK" if r["range_ok"] else f"{flag}({r['n_below_min']} below, {r['n_above_max']} above)"
        print(f"    {col:<20} {r['observed_min']:8.3f} - {r['observed_max']:7.3f} {r['unit']:<4}  {status}")
    print()
    print("  Digital signal validity:")
    for col, r in report["digital_signals"].items():
        status = "OK" if r["valid"] else "!! INVALID VALUES"
        print(f"    {col:<20} active={r['pct_active']:5.1f}%  transitions={r['n_transitions']:6d}  {status}")
    print()
    issues = report["summary"]["issues_found"]
    if issues:
        print(f"  Issues found: {', '.join(issues)}")
    else:
        print("  No structural data quality issues detected.")
    print(sep + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_quality_check()
