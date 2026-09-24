"""
tests/test_features.py — Unit tests for feature engineering and labeling.

Tests use small synthetic DataFrames to validate correctness without
running the full preprocessing pipeline or loading the raw CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    build_feature_matrix,
    build_model_ready,
    get_feature_columns,
    _rolling_within_segments,
    _trend_within_segments,
)
from src.features.labeling import (
    build_labels,
    get_anomaly_training_index,
    get_temporal_splits,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_processed_df(
    start: str = "2020-02-01 00:00:00",
    n_rows: int = 300,
    failure_period: tuple | None = None,
    add_gap: bool = False,
) -> pd.DataFrame:
    """
    Build a minimal 1-minute processed DataFrame for feature-engineering tests.
    All required columns from the preprocessor are included.
    """
    idx = pd.date_range(start, periods=n_rows, freq="1min")
    rng = np.random.default_rng(7)

    df = pd.DataFrame({
        "TP2":           rng.uniform(0.0, 2.0, n_rows).astype("float32"),
        "TP3":           rng.uniform(8.5, 10.0, n_rows).astype("float32"),
        "H1":            rng.uniform(6.0, 9.0, n_rows).astype("float32"),
        "DV_pressure":   rng.uniform(0.0, 0.05, n_rows).astype("float32"),
        "Reservoirs":    rng.uniform(8.5, 10.0, n_rows).astype("float32"),
        "Oil_temperature": rng.uniform(55.0, 70.0, n_rows).astype("float32"),
        "Motor_current": rng.choice([0.05, 4.0, 7.0], n_rows).astype("float32"),
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
        "motor_state":   pd.Categorical(["offloaded"] * n_rows),
        "is_loaded":     rng.integers(0, 2, n_rows).astype(bool),
        "pressure_diff": rng.uniform(-0.5, 0.5, n_rows).astype("float32"),
        "reservoir_panel_diff": rng.uniform(-0.1, 0.1, n_rows).astype("float32"),
        "h1_near_zero":  np.zeros(n_rows, dtype=bool),
    }, index=idx)

    # Add _max columns
    for col in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]:
        df[f"{col}_max"] = df[col] + 0.1

    # gap_segment_id
    seg_ids = np.ones(n_rows, dtype="float64")
    if add_gap:
        gap_rows = slice(100, 120)
        df.loc[df.index[gap_rows], "is_gap"] = True
        df.loc[df.index[gap_rows], "n_samples"] = 0
        seg_ids[120:] = 2.0
        seg_ids[100:120] = np.nan
    df["gap_segment_id"] = seg_ids

    # failure_id
    df["failure_id"] = ""
    if failure_period is not None:
        start_i, end_i = failure_period
        df.iloc[start_i:end_i, df.columns.get_loc("failure_id")] = "F1"

    return df


# ── _rolling_within_segments ──────────────────────────────────────────────────

class TestRollingWithinSegments:

    def test_returns_series_same_length(self):
        df = _make_processed_df(n_rows=100)
        result = _rolling_within_segments(
            df["H1"].astype("float64"), df["gap_segment_id"], window=10, stat="mean"
        )
        assert len(result) == 100

    def test_gap_rows_are_nan(self):
        df = _make_processed_df(n_rows=200, add_gap=True)
        result = _rolling_within_segments(
            df["H1"].astype("float64"), df["gap_segment_id"], window=10, stat="mean"
        )
        gap_values = result[df["is_gap"]].dropna()
        assert len(gap_values) == 0, "Gap rows should be NaN"

    def test_mean_is_float32(self):
        df = _make_processed_df(n_rows=50)
        result = _rolling_within_segments(
            df["H1"].astype("float64"), df["gap_segment_id"], window=5, stat="mean"
        )
        assert result.dtype == "float32"

    def test_max_gte_mean(self):
        df = _make_processed_df(n_rows=100)
        series = df["H1"].astype("float64")
        seg_ids = df["gap_segment_id"]
        means = _rolling_within_segments(series, seg_ids, window=10, stat="mean")
        maxes = _rolling_within_segments(series, seg_ids, window=10, stat="max")
        valid = ~means.isna() & ~maxes.isna()
        assert (maxes[valid] >= means[valid] - 1e-5).all()

    def test_min_lte_mean(self):
        df = _make_processed_df(n_rows=100)
        series = df["H1"].astype("float64")
        seg_ids = df["gap_segment_id"]
        means = _rolling_within_segments(series, seg_ids, window=10, stat="mean")
        mins  = _rolling_within_segments(series, seg_ids, window=10, stat="min")
        valid = ~means.isna() & ~mins.isna()
        assert (mins[valid] <= means[valid] + 1e-5).all()

    def test_does_not_cross_gap(self):
        """Values immediately after a gap should be NaN (window startup)."""
        df = _make_processed_df(n_rows=200, add_gap=True)
        result = _rolling_within_segments(
            df["H1"].astype("float64"), df["gap_segment_id"], window=30, stat="mean"
        )
        # Row 120 is the first row after the gap — within a window of 30,
        # fewer than min_periods (15) rows are available, so it should be NaN
        assert pd.isna(result.iloc[120])


# ── _trend_within_segments ────────────────────────────────────────────────────

class TestTrendWithinSegments:

    def test_returns_series_same_length(self):
        df = _make_processed_df(n_rows=100)
        result = _trend_within_segments(
            df["H1"].astype("float64"), df["gap_segment_id"], window=20
        )
        assert len(result) == 100

    def test_increasing_series_positive_slope(self):
        """A strictly increasing series should have positive trend."""
        idx = pd.date_range("2020-01-01", periods=100, freq="1min")
        series = pd.Series(np.arange(100, dtype="float64"), index=idx)
        seg_ids = pd.Series(np.ones(100), index=idx)
        result = _trend_within_segments(series, seg_ids, window=20)
        # After window fills, all slopes should be positive
        valid = result.dropna()
        assert (valid > 0).all()

    def test_constant_series_zero_slope(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="1min")
        series = pd.Series(np.full(100, 5.0), index=idx)
        seg_ids = pd.Series(np.ones(100), index=idx)
        result = _trend_within_segments(series, seg_ids, window=20)
        valid = result.dropna()
        assert (np.abs(valid) < 1e-6).all()

    def test_float32_output(self):
        df = _make_processed_df(n_rows=60)
        result = _trend_within_segments(
            df["H1"].astype("float64"), df["gap_segment_id"], window=15
        )
        assert result.dtype == "float32"


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_returns_dataframe(self):
        df = _make_processed_df(n_rows=100)
        result = build_feature_matrix(df)
        assert isinstance(result, pd.DataFrame)

    def test_same_length_as_input(self):
        df = _make_processed_df(n_rows=150)
        result = build_feature_matrix(df)
        assert len(result) == len(df)

    def test_same_index_as_input(self):
        df = _make_processed_df(n_rows=150)
        result = build_feature_matrix(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_contains_rolling_features(self):
        df = _make_processed_df(n_rows=200)
        result = build_feature_matrix(df)
        # Check for at least one rolling feature
        rolling_cols = [c for c in result.columns if "_30m_" in c or "_2h_" in c]
        assert len(rolling_cols) > 0, "Expected rolling feature columns"

    def test_contains_metadata_columns(self):
        df = _make_processed_df(n_rows=100)
        result = build_feature_matrix(df)
        for col in ["is_gap", "gap_segment_id", "failure_id"]:
            assert col in result.columns, f"Missing metadata column: {col}"

    def test_gap_rows_have_nan_features(self):
        df = _make_processed_df(n_rows=200, add_gap=True)
        result = build_feature_matrix(df)
        # Rolling features at gap rows should be NaN
        gap_rows = result[result["is_gap"]]
        feat_cols = get_feature_columns(result)
        nan_counts = gap_rows[feat_cols].isna().all(axis=0)
        # At least half the features should be all-NaN at gap rows
        assert nan_counts.mean() > 0.5


# ── build_model_ready ─────────────────────────────────────────────────────────

class TestBuildModelReady:

    def test_removes_gap_rows(self):
        df = _make_processed_df(n_rows=200, add_gap=True)
        feat_df = build_feature_matrix(df)
        X, meta = build_model_ready(feat_df)
        assert "is_gap" not in X.columns
        # Gap rows are excluded from X
        n_gap = int(df["is_gap"].sum())
        # X may have fewer rows than non-gap rows due to NaN dropout
        assert len(X) <= len(df) - n_gap

    def test_no_nan_in_X(self):
        df = _make_processed_df(n_rows=200)
        feat_df = build_feature_matrix(df)
        X, _ = build_model_ready(feat_df)
        assert not X.isna().any().any(), "X should not contain any NaN values"

    def test_meta_is_failure_id(self):
        df = _make_processed_df(n_rows=200, failure_period=(100, 130))
        feat_df = build_feature_matrix(df)
        _, meta = build_model_ready(feat_df)
        assert meta.name == "failure_id"
        assert "F1" in meta.values

    def test_returns_float32(self):
        df = _make_processed_df(n_rows=100)
        feat_df = build_feature_matrix(df)
        X, _ = build_model_ready(feat_df)
        assert X.dtypes.eq("float32").all(), "All X columns should be float32"


# ── build_labels ──────────────────────────────────────────────────────────────

class TestBuildLabels:

    def test_columns_created(self):
        df = _make_processed_df(n_rows=200, failure_period=(100, 130))
        result = build_labels(df)
        for col in ["is_failure", "is_atrisk", "is_excluded", "label", "label_strict"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_failure_rows_are_labelled(self):
        df = _make_processed_df(n_rows=200, failure_period=(100, 130))
        result = build_labels(df)
        # is_failure should be True where failure_id != ''
        failure_mask = result["failure_id"] == "F1"
        assert result.loc[failure_mask, "is_failure"].all()

    def test_label_is_binary(self):
        df = _make_processed_df(n_rows=200)
        result = build_labels(df)
        assert set(result["label"].unique()).issubset({0, 1})

    def test_label_strict_subset_of_label(self):
        """label_strict rows should be a subset of label rows."""
        df = _make_processed_df(n_rows=200, failure_period=(100, 130))
        result = build_labels(df)
        # Every label_strict=1 row should also have label=1
        strict_rows = result["label_strict"] == 1
        assert result.loc[strict_rows, "label"].all()

    def test_no_label_in_gap_rows(self):
        df = _make_processed_df(n_rows=200, add_gap=True)
        result = build_labels(df)
        # Gap rows should never be labelled positive
        gap_rows = result["is_gap"]
        assert (result.loc[gap_rows, "label"] == 0).all()


# ── get_temporal_splits ───────────────────────────────────────────────────────

class TestTemporalSplits:

    def test_splits_do_not_overlap(self):
        """
        Build a large enough dataset to span the split boundaries.
        Use a start date within the known split range.
        """
        # Build a df spanning Apr–Aug 2020 (covers train/val/test boundaries)
        start = "2020-04-01 00:00:00"
        n = 200000  # ~139 days at 1min
        idx = pd.date_range(start, periods=n, freq="1min")
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "H1":            rng.uniform(6.0, 9.0, n).astype("float32"),
            "TP2":           rng.uniform(0.0, 2.0, n).astype("float32"),
            "TP3":           rng.uniform(8.5, 10.0, n).astype("float32"),
            "DV_pressure":   rng.uniform(0.0, 0.05, n).astype("float32"),
            "Reservoirs":    rng.uniform(8.5, 10.0, n).astype("float32"),
            "Oil_temperature": rng.uniform(55.0, 70.0, n).astype("float32"),
            "Motor_current": rng.choice([0.05, 4.0, 7.0], n).astype("float32"),
            "COMP":          rng.integers(0, 2, n).astype("int8"),
            "DV_eletric":    rng.integers(0, 2, n).astype("int8"),
            "Towers":        np.ones(n, dtype="int8"),
            "MPG":           rng.integers(0, 2, n).astype("int8"),
            "LPS":           np.zeros(n, dtype="int8"),
            "Pressure_switch": np.ones(n, dtype="int8"),
            "Oil_level":     np.ones(n, dtype="int8"),
            "Caudal_impulses": np.ones(n, dtype="int8"),
            "n_samples":     np.full(n, 6, dtype="int32"),
            "load_fraction": rng.uniform(0.0, 1.0, n).astype("float32"),
            "is_gap":        np.zeros(n, dtype=bool),
            "motor_state":   pd.Categorical(["offloaded"] * n),
            "is_loaded":     np.zeros(n, dtype=bool),
            "pressure_diff": np.zeros(n, dtype="float32"),
            "reservoir_panel_diff": np.zeros(n, dtype="float32"),
            "h1_near_zero":  np.zeros(n, dtype=bool),
            "gap_segment_id": np.ones(n, dtype="float64"),
            "failure_id":    "",
        }, index=idx)
        for col in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                    "Oil_temperature", "Motor_current"]:
            df[f"{col}_max"] = df[col] + 0.1

        labeled = build_labels(df)
        train_idx, val_idx, test_idx = get_temporal_splits(labeled)

        # No overlap
        assert len(train_idx.intersection(val_idx)) == 0
        assert len(val_idx.intersection(test_idx)) == 0
        assert len(train_idx.intersection(test_idx)) == 0

    def test_train_before_val_before_test(self):
        """Check strict ordering: all train < all val < all test."""
        start = "2020-04-01 00:00:00"
        n = 200000
        idx = pd.date_range(start, periods=n, freq="1min")
        rng = np.random.default_rng(4)
        df = pd.DataFrame({
            "H1": rng.uniform(6.0, 9.0, n).astype("float32"),
            "TP2": rng.uniform(0.0, 2.0, n).astype("float32"),
            "TP3": rng.uniform(8.5, 10.0, n).astype("float32"),
            "DV_pressure": rng.uniform(0.0, 0.05, n).astype("float32"),
            "Reservoirs": rng.uniform(8.5, 10.0, n).astype("float32"),
            "Oil_temperature": rng.uniform(55.0, 70.0, n).astype("float32"),
            "Motor_current": rng.choice([0.05, 4.0, 7.0], n).astype("float32"),
            "COMP": rng.integers(0, 2, n).astype("int8"),
            "DV_eletric": rng.integers(0, 2, n).astype("int8"),
            "Towers": np.ones(n, dtype="int8"),
            "MPG": rng.integers(0, 2, n).astype("int8"),
            "LPS": np.zeros(n, dtype="int8"),
            "Pressure_switch": np.ones(n, dtype="int8"),
            "Oil_level": np.ones(n, dtype="int8"),
            "Caudal_impulses": np.ones(n, dtype="int8"),
            "n_samples": np.full(n, 6, dtype="int32"),
            "load_fraction": rng.uniform(0.0, 1.0, n).astype("float32"),
            "is_gap": np.zeros(n, dtype=bool),
            "motor_state": pd.Categorical(["offloaded"] * n),
            "is_loaded": np.zeros(n, dtype=bool),
            "pressure_diff": np.zeros(n, dtype="float32"),
            "reservoir_panel_diff": np.zeros(n, dtype="float32"),
            "h1_near_zero": np.zeros(n, dtype=bool),
            "gap_segment_id": np.ones(n, dtype="float64"),
            "failure_id": "",
        }, index=idx)
        for col in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]:
            df[f"{col}_max"] = df[col] + 0.1

        labeled = build_labels(df)
        train_idx, val_idx, test_idx = get_temporal_splits(labeled)

        if len(train_idx) and len(val_idx):
            assert train_idx.max() <= val_idx.min()
        if len(val_idx) and len(test_idx):
            assert val_idx.max() <= test_idx.min()


# ── Anomaly training index ─────────────────────────────────────────────────────

class TestAnomalyTrainingIndex:

    def test_only_february_march(self):
        """Anomaly training index should only include Feb–Mar 2020."""
        df = _make_processed_df(start="2020-02-01", n_rows=100)
        labeled = build_labels(df)
        idx = get_anomaly_training_index(labeled)
        assert idx.max() <= pd.Timestamp("2020-03-31 23:59:00")

    def test_no_failure_rows(self):
        df = _make_processed_df(
            start="2020-02-01", n_rows=200, failure_period=(50, 80)
        )
        labeled = build_labels(df)
        idx = get_anomaly_training_index(labeled)
        failure_mask = labeled.loc[idx, "failure_id"] != ""
        assert not failure_mask.any(), "Anomaly training index should not contain failure rows"

    def test_no_gap_rows(self):
        df = _make_processed_df(start="2020-02-01", n_rows=200, add_gap=True)
        labeled = build_labels(df)
        idx = get_anomaly_training_index(labeled)
        gap_mask = labeled.loc[idx, "is_gap"]
        assert not gap_mask.any(), "Anomaly training index should not contain gap rows"
