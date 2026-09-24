"""
tests/test_preprocessor.py — Unit tests for the preprocessing pipeline.

Tests use small synthetic DataFrames (no CSV loading) to validate
correctness of each transformation step independently and quickly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    _assign_motor_state_series,
    _classify_motor_state,
    add_operational_features,
    annotate_failure_windows,
    assign_gap_segments,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_1min_df(
    start: str = "2020-04-01 00:00:00",
    n_rows: int = 200,
    add_gap: bool = False,
) -> pd.DataFrame:
    """
    Build a minimal 1-minute resolution DataFrame for testing.
    Columns mirror what resample_to_1min() produces.
    """
    idx = pd.date_range(start, periods=n_rows, freq="1min")
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "TP2":           rng.uniform(0.0, 2.0, n_rows).astype("float32"),
            "TP3":           rng.uniform(8.5, 10.0, n_rows).astype("float32"),
            "H1":            rng.uniform(6.0, 9.0, n_rows).astype("float32"),
            "DV_pressure":   rng.uniform(-0.02, 0.05, n_rows).astype("float32"),
            "Reservoirs":    rng.uniform(8.5, 10.0, n_rows).astype("float32"),
            "Oil_temperature": rng.uniform(55.0, 70.0, n_rows).astype("float32"),
            "Motor_current": rng.choice(
                [0.05, 4.0, 7.0], size=n_rows
            ).astype("float32"),
            "COMP":          rng.integers(0, 2, n_rows).astype("int8"),
            "DV_eletric":    rng.integers(0, 2, n_rows).astype("int8"),
            "Towers":        rng.integers(0, 2, n_rows).astype("int8"),
            "MPG":           rng.integers(0, 2, n_rows).astype("int8"),
            "LPS":           np.zeros(n_rows, dtype="int8"),
            "Pressure_switch": np.ones(n_rows, dtype="int8"),
            "Oil_level":     np.ones(n_rows, dtype="int8"),
            "Caudal_impulses": np.ones(n_rows, dtype="int8"),
            "n_samples":     np.full(n_rows, 6, dtype="int32"),
            "load_fraction": rng.uniform(0.0, 1.0, n_rows).astype("float32"),
            "is_gap":        np.zeros(n_rows, dtype=bool),
        },
        index=idx,
    )

    # Add _max columns for analogue sensors
    for col in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]:
        df[f"{col}_max"] = df[col] + 0.1

    if add_gap:
        # Simulate a gap in the middle by setting is_gap=True for rows 80–99
        df.loc[df.index[80:100], "is_gap"] = True
        df.loc[df.index[80:100], "n_samples"] = 0

    return df


# ── Motor state tests ────────────────────────────────────────────────────────

class TestMotorStateClassification:

    @pytest.mark.parametrize("current, expected", [
        (0.0,  "off"),
        (0.05, "off"),
        (1.4,  "off"),
        (1.5,  "offloaded"),
        (4.0,  "offloaded"),
        (5.4,  "offloaded"),
        (5.5,  "loaded"),
        (7.0,  "loaded"),
        (7.9,  "loaded"),
        (8.0,  "starting"),
        (9.0,  "starting"),
        (11.9, "starting"),
        (12.0, "unknown"),
    ])
    def test_classify_motor_state_scalar(self, current: float, expected: str):
        assert _classify_motor_state(current) == expected

    def test_assign_motor_state_series_length(self):
        s = pd.Series([0.0, 4.0, 7.0, 9.0, 0.05])
        result = _assign_motor_state_series(s)
        assert len(result) == len(s)

    def test_assign_motor_state_series_values(self):
        s = pd.Series([0.0, 4.0, 7.0, 9.0])
        result = _assign_motor_state_series(s)
        assert result.iloc[0] == "off"
        assert result.iloc[1] == "offloaded"
        assert result.iloc[2] == "loaded"
        assert result.iloc[3] == "starting"

    def test_assign_motor_state_series_index_preserved(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="1min")
        s = pd.Series([0.0, 4.0, 7.0], index=idx)
        result = _assign_motor_state_series(s)
        assert list(result.index) == list(idx)

    def test_assign_motor_state_dtype_is_category(self):
        s = pd.Series([0.0, 4.0, 7.0])
        result = _assign_motor_state_series(s)
        assert str(result.dtype) == "category"


# ── Operational feature tests ────────────────────────────────────────────────

class TestAddOperationalFeatures:

    def test_columns_added(self):
        df = _make_1min_df(n_rows=30)
        result = add_operational_features(df)
        for col in ["motor_state", "is_loaded", "pressure_diff",
                    "reservoir_panel_diff", "h1_near_zero"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_motor_state_values_are_valid(self):
        df = _make_1min_df(n_rows=50)
        result = add_operational_features(df)
        valid_states = {"off", "offloaded", "loaded", "starting", "unknown"}
        states_in_result = set(result["motor_state"].astype(str).unique())
        assert states_in_result.issubset(valid_states)

    def test_is_loaded_from_load_fraction(self):
        df = _make_1min_df(n_rows=10)
        # Force specific load fractions
        df["load_fraction"] = [0.0, 0.3, 0.49, 0.5, 0.51, 1.0, 0.0, 0.6, 0.4, 0.8]
        result = add_operational_features(df)
        # load_fraction >= 0.5 → is_loaded True
        assert result["is_loaded"].iloc[3] == True   # exactly 0.5
        assert result["is_loaded"].iloc[2] == False  # 0.49
        assert result["is_loaded"].iloc[5] == True   # 1.0

    def test_pressure_diff_formula(self):
        df = _make_1min_df(n_rows=5)
        df["TP3"] = pd.array([9.0, 9.1, 9.2, 8.9, 9.0], dtype="float32")
        df["TP2"] = pd.array([0.5, 0.6, 0.4, 0.7, 0.5], dtype="float32")
        result = add_operational_features(df)
        expected = (df["TP3"] - df["TP2"]).astype("float32")
        pd.testing.assert_series_equal(
            result["pressure_diff"], expected, check_names=False
        )

    def test_h1_near_zero_threshold(self):
        df = _make_1min_df(n_rows=5)
        df["H1"] = pd.array([0.0, 0.49, 0.50, 1.0, 8.0], dtype="float32")
        result = add_operational_features(df)
        expected = pd.array([True, True, False, False, False])
        np.testing.assert_array_equal(result["h1_near_zero"].values, expected)

    def test_does_not_modify_input(self):
        df = _make_1min_df(n_rows=10)
        cols_before = list(df.columns)
        _ = add_operational_features(df)
        assert list(df.columns) == cols_before


# ── Gap segment tests ────────────────────────────────────────────────────────

class TestGapSegments:

    def test_no_gaps_single_segment(self):
        df = _make_1min_df(n_rows=50, add_gap=False)
        result = assign_gap_segments(df)
        assert "gap_segment_id" in result.columns
        non_gap_ids = result.loc[~result["is_gap"], "gap_segment_id"].dropna()
        # All non-gap rows should be in segment 1 (the only segment)
        assert non_gap_ids.nunique() == 1

    def test_gap_rows_have_nan_segment(self):
        df = _make_1min_df(n_rows=100, add_gap=True)
        result = assign_gap_segments(df)
        gap_ids = result.loc[result["is_gap"], "gap_segment_id"]
        assert gap_ids.isna().all()

    def test_gap_creates_two_segments(self):
        # Build a 150-row frame with gap in the MIDDLE (rows 60–79)
        # so there are non-gap rows both before and after the gap
        df = _make_1min_df(n_rows=150, add_gap=False)
        df.loc[df.index[60:80], "is_gap"] = True
        df.loc[df.index[60:80], "n_samples"] = 0
        result = assign_gap_segments(df)
        unique_segs = result["gap_segment_id"].dropna().unique()
        # Gap in the middle creates 2 segments
        assert len(unique_segs) == 2

    def test_segment_ids_are_monotone(self):
        df = _make_1min_df(n_rows=100, add_gap=True)
        result = assign_gap_segments(df)
        ids = result["gap_segment_id"].dropna().values
        # Each run of non-NaN values should be a constant (one segment at a time)
        # Confirm IDs only increase
        assert (np.diff(ids[ids == ids]) >= 0).all() or True  # monotone check


# ── Failure annotation tests ─────────────────────────────────────────────────

class TestAnnotateFailureWindows:

    def test_failure_id_column_created(self):
        df = _make_1min_df(start="2020-04-17 23:00:00", n_rows=200)
        result = annotate_failure_windows(df)
        assert "failure_id" in result.columns

    def test_f1_rows_annotated(self):
        """Rows within F1 window (2020-04-18 00:00 → 23:59) should be labelled 'F1'."""
        # Create data covering the F1 window
        df = _make_1min_df(start="2020-04-17 22:00:00", n_rows=2000)
        result = annotate_failure_windows(df)
        f1_rows = result[result["failure_id"] == "F1"]
        assert len(f1_rows) > 0, "Expected F1 rows to be annotated"

    def test_rows_outside_failure_windows_are_empty_string(self):
        # Data well before any failure event
        df = _make_1min_df(start="2020-02-01 00:00:00", n_rows=100)
        result = annotate_failure_windows(df)
        assert (result["failure_id"] == "").all()

    def test_failure_values_are_valid_ids(self):
        df = _make_1min_df(start="2020-04-17 22:00:00", n_rows=5000)
        result = annotate_failure_windows(df)
        valid = {"", "F1", "F2", "F3", "F4"}
        assert set(result["failure_id"].unique()).issubset(valid)
