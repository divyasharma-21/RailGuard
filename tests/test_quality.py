"""
tests/test_quality.py — Unit tests for the data quality module.

Tests use small synthetic DataFrames to validate correctness of
each quality check without touching the raw CSV file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import (
    _check_digital_signals,
    _check_duplicates,
    _check_nulls,
    _check_sensor_ranges,
    _check_structure,
    _flag_outliers,
    GAP_THRESHOLD_S,
    _check_gaps,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_raw_df(
    n_rows: int = 100,
    freq: str = "10s",
    start: str = "2020-02-01 00:00:00",
    add_nulls: bool = False,
    add_dups: bool = False,
    add_out_of_range: bool = False,
) -> pd.DataFrame:
    """Build a minimal synthetic raw DataFrame for quality-check testing."""
    idx = pd.date_range(start, periods=n_rows, freq=freq)
    rng = np.random.default_rng(0)

    data = {
        "TP2":            rng.uniform(0.0, 2.0, n_rows).astype("float32"),
        "TP3":            rng.uniform(8.5, 10.0, n_rows).astype("float32"),
        "H1":             rng.uniform(6.0, 9.0, n_rows).astype("float32"),
        "DV_pressure":    rng.uniform(-0.02, 0.05, n_rows).astype("float32"),
        "Reservoirs":     rng.uniform(8.5, 10.0, n_rows).astype("float32"),
        "Oil_temperature":rng.uniform(50.0, 75.0, n_rows).astype("float32"),
        "Motor_current":  rng.uniform(0.0, 7.5, n_rows).astype("float32"),
        "COMP":           rng.integers(0, 2, n_rows).astype("int8"),
        "DV_eletric":     rng.integers(0, 2, n_rows).astype("int8"),
        "Towers":         rng.integers(0, 2, n_rows).astype("int8"),
        "MPG":            rng.integers(0, 2, n_rows).astype("int8"),
        "LPS":            np.zeros(n_rows, dtype="int8"),
        "Pressure_switch":np.ones(n_rows, dtype="int8"),
        "Oil_level":      np.ones(n_rows, dtype="int8"),
        "Caudal_impulses":np.ones(n_rows, dtype="int8"),
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"

    if add_nulls:
        df.loc[df.index[5], "TP2"] = np.nan
        df.loc[df.index[10], "Oil_temperature"] = np.nan

    if add_dups:
        # Add a duplicate row by repeating row 0
        dup = df.iloc[[0]]
        df = pd.concat([df, dup])

    if add_out_of_range:
        # Force Oil_temperature to 120°C (above physical limit of 110)
        df.iloc[3, df.columns.get_loc("Oil_temperature")] = 120.0

    return df


# ── Structure tests ──────────────────────────────────────────────────────────

class TestCheckStructure:

    def test_reports_correct_row_count(self):
        df = _make_raw_df(n_rows=50)
        r = _check_structure(df)
        assert r["n_rows"] == 50

    def test_no_missing_cols(self):
        df = _make_raw_df()
        r = _check_structure(df)
        assert r["missing_cols"] == []

    def test_detects_missing_col(self):
        df = _make_raw_df().drop(columns=["TP2"])
        r = _check_structure(df)
        assert "TP2" in r["missing_cols"]

    def test_index_name(self):
        df = _make_raw_df()
        r = _check_structure(df)
        assert r["index_name"] == "timestamp"


# ── Null tests ───────────────────────────────────────────────────────────────

class TestCheckNulls:

    def test_no_nulls(self):
        df = _make_raw_df()
        r = _check_nulls(df)
        assert r["total_null_cells"] == 0
        assert r["any_nulls"] is False

    def test_detects_nulls(self):
        df = _make_raw_df(add_nulls=True)
        r = _check_nulls(df)
        assert r["any_nulls"] is True
        assert r["total_null_cells"] == 2
        assert "TP2" in r["per_column"]
        assert "Oil_temperature" in r["per_column"]


# ── Duplicate tests ───────────────────────────────────────────────────────────

class TestCheckDuplicates:

    def test_no_duplicates(self):
        df = _make_raw_df()
        r = _check_duplicates(df)
        assert r["duplicate_rows"] == 0
        assert r["duplicate_timestamps"] == 0

    def test_detects_duplicate_row(self):
        df = _make_raw_df(add_dups=True)
        r = _check_duplicates(df)
        assert r["duplicate_rows"] >= 1


# ── Gap tests ─────────────────────────────────────────────────────────────────

class TestCheckGaps:

    def test_regular_10s_no_large_gaps(self):
        df = _make_raw_df(n_rows=200, freq="10s")
        summary, gap_df = _check_gaps(df)
        assert summary["dominant_gap_s"] == 10
        assert summary["total_gaps_above_threshold"] == 0

    def test_single_large_gap_detected(self):
        # Build a series with a deliberate 5-min gap in the middle
        idx1 = pd.date_range("2020-02-01 00:00:00", periods=50, freq="10s")
        idx2 = pd.date_range("2020-02-01 00:10:00", periods=50, freq="10s")
        # gap between end of idx1 (00:08:10) and start of idx2 (00:10:00) = 110s
        idx  = idx1.append(idx2)
        rng  = np.random.default_rng(1)
        df   = pd.DataFrame(
            {"TP2": rng.uniform(0, 1, len(idx)).astype("float32"),
             "TP3": rng.uniform(8.5, 10.0, len(idx)).astype("float32"),
             "H1":  rng.uniform(6.0, 9.0, len(idx)).astype("float32"),
             "DV_pressure": rng.uniform(-0.02, 0.05, len(idx)).astype("float32"),
             "Reservoirs":  rng.uniform(8.5, 10.0, len(idx)).astype("float32"),
             "Oil_temperature": rng.uniform(50, 75, len(idx)).astype("float32"),
             "Motor_current": rng.uniform(0, 7, len(idx)).astype("float32"),
             "COMP": np.zeros(len(idx), dtype="int8"),
             "DV_eletric": np.zeros(len(idx), dtype="int8"),
             "Towers": np.zeros(len(idx), dtype="int8"),
             "MPG": np.zeros(len(idx), dtype="int8"),
             "LPS": np.zeros(len(idx), dtype="int8"),
             "Pressure_switch": np.ones(len(idx), dtype="int8"),
             "Oil_level": np.ones(len(idx), dtype="int8"),
             "Caudal_impulses": np.ones(len(idx), dtype="int8"),
             },
            index=idx,
        )
        df.index.name = "timestamp"
        summary, gap_df = _check_gaps(df)
        assert summary["total_gaps_above_threshold"] == 1
        assert len(gap_df) == 1
        assert gap_df["gap_seconds"].iloc[0] > GAP_THRESHOLD_S

    def test_gap_df_columns(self):
        df = _make_raw_df(n_rows=200, freq="10s")
        _, gap_df = _check_gaps(df)
        # Even empty gap_df should have the right columns
        for col in ["gap_start_ts", "gap_end_ts", "gap_seconds", "gap_hours"]:
            assert col in gap_df.columns


# ── Sensor range tests ────────────────────────────────────────────────────────

class TestCheckSensorRanges:

    def test_all_in_range(self):
        df = _make_raw_df()
        r = _check_sensor_ranges(df)
        for col, res in r.items():
            assert res["range_ok"], f"{col} should be in range but is not"

    def test_detects_above_max(self):
        df = _make_raw_df(add_out_of_range=True)
        r = _check_sensor_ranges(df)
        assert r["Oil_temperature"]["n_above_max"] >= 1
        assert r["Oil_temperature"]["range_ok"] is False

    def test_reports_observed_extremes(self):
        df = _make_raw_df()
        r = _check_sensor_ranges(df)
        for col in r:
            assert "observed_min" in r[col]
            assert "observed_max" in r[col]


# ── Digital signal tests ──────────────────────────────────────────────────────

class TestCheckDigitalSignals:

    def test_valid_binary_signals(self):
        df = _make_raw_df()
        r = _check_digital_signals(df)
        for col, res in r.items():
            assert res["valid"], f"{col} should be valid binary but is not"

    def test_detects_invalid_value(self):
        df = _make_raw_df()
        # Inject a non-binary value
        df["LPS"] = df["LPS"].astype(object)
        df.iloc[5, df.columns.get_loc("LPS")] = 2
        r = _check_digital_signals(df)
        assert r["LPS"]["invalid_values"] >= 1
        assert r["LPS"]["valid"] == False  # noqa: E712 (numpy bool needs ==)

    def test_activation_rate_computation(self):
        df = _make_raw_df(n_rows=100)
        # Force LPS=1 for exactly half the rows
        df["LPS"] = pd.array(
            [1] * 50 + [0] * 50, dtype="int8"
        )
        r = _check_digital_signals(df)
        assert r["LPS"]["pct_active"] == 50.0

    def test_transition_count(self):
        df = _make_raw_df(n_rows=10)
        # LPS alternates 0,1,0,1,...
        df["LPS"] = pd.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype="int8")
        r = _check_digital_signals(df)
        # Should count 9 transitions (each consecutive pair changes)
        assert r["LPS"]["n_transitions"] == 9


# ── Outlier flag tests ────────────────────────────────────────────────────────

class TestFlagOutliers:

    def test_no_outliers_in_normal_data(self):
        df = _make_raw_df(n_rows=200)
        r = _flag_outliers(df)
        # With uniform random data, there shouldn't be IQR-3× outliers
        for col, res in r.items():
            assert "n_outliers" in res
            assert "iqr_fence_low" in res
            assert "iqr_fence_high" in res

    def test_detects_extreme_spike(self):
        df = _make_raw_df(n_rows=300)
        # Inject an extreme spike in TP2
        df.iloc[150, df.columns.get_loc("TP2")] = 999.0
        r = _flag_outliers(df)
        assert r["TP2"]["n_outliers"] >= 1

    def test_example_list_length(self):
        df = _make_raw_df(n_rows=200)
        df.iloc[100, df.columns.get_loc("TP2")] = 999.0
        r = _flag_outliers(df)
        # examples list should be at most 5 items
        assert len(r["TP2"]["examples"]) <= 5
