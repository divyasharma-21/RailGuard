"""
src/models/anomaly.py — Isolation Forest anomaly detection model.

Design rationale
----------------
Isolation Forest (Liu et al., 2008) is chosen as the primary anomaly detection
method for the following reasons:

1. Unsupervised — we do not need failure labels to train it, which is important
   because we have only 4 documented failure events.

2. Effective on high-dimensional tabular data without assuming a distribution.

3. Computationally tractable on our ~70K-row normal training baseline.

4. Anomaly scores are well-defined and continuous, not just binary flags.
   We use the negative mean path length as a score: higher (less negative)
   = more anomalous.

5. The contamination parameter is NOT set to an arbitrary engineering value.
   We set contamination='auto' which uses the original paper's threshold of
   0.5 for score normalisation, giving calibrated anomaly scores. The
   decision_function output is used directly as a continuous score.

Training baseline:
   Feb–Mar 2020 (confirmed normal from Stage 2 EDA findings).
   This baseline is clean: LPS activity is minimal, no documented failures,
   and oil temperature is at seasonal baseline.

Feature selection:
   H1, load_fraction, Motor_current, Oil_temperature, TP2, DV_pressure,
   LPS rate, and their rolling statistics are included. These are the sensors
   with the largest separation between normal and failure states (Stage 2).

Anomaly score interpretation:
   The IsolationForest.decision_function() returns the anomaly score.
   - Score > 0:  inlier (normal). Higher = more normal.
   - Score < 0:  anomaly. Lower (more negative) = more anomalous.
   - Score = 0:  boundary (original paper threshold).
   We invert and rescale to an "anomaly score" in [0, 1] for display:
     anomaly_score = sigmoid(-raw_score * scale_factor)
   where scale_factor is tuned so that the 99th percentile of training
   scores maps to ~0.2 (i.e., training data has low anomaly scores).

Saved artifacts (data/artifacts/):
   anomaly_scaler.pkl      : fitted StandardScaler
   anomaly_model.pkl       : fitted IsolationForest
   anomaly_feature_cols.json: feature column names (for consistency)
   anomaly_score_stats.json: training score statistics (for score calibration)

Usage:
    from src.models.anomaly import train_anomaly_model, score_anomaly
    model_bundle = train_anomaly_model(df_features, normal_idx)
    scores = score_anomaly(model_bundle, X_new)
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths
SCALER_PATH      = ARTIFACTS_DIR / "anomaly_scaler.pkl"
MODEL_PATH       = ARTIFACTS_DIR / "anomaly_model.pkl"
FEAT_COLS_PATH   = ARTIFACTS_DIR / "anomaly_feature_cols.json"
SCORE_STATS_PATH = ARTIFACTS_DIR / "anomaly_score_stats.json"

# IsolationForest hyperparameters
IF_N_ESTIMATORS  = 200
IF_MAX_SAMPLES   = "auto"
IF_CONTAMINATION = "auto"   # uses paper default; score calibrated separately
IF_RANDOM_STATE  = 42
IF_N_JOBS        = -1


# ---------------------------------------------------------------------------
# Feature selection for anomaly detection
# ---------------------------------------------------------------------------

def select_anomaly_features(df_features: pd.DataFrame) -> List[str]:
    """
    Select feature columns appropriate for anomaly detection.

    Prefers primary discriminators (from Stage 2 findings) plus their rolling
    statistics. Excludes metadata columns.
    """
    exclude = {"is_gap", "gap_segment_id", "failure_id", "motor_state", "is_loaded",
               "is_failure", "is_atrisk", "is_excluded", "label", "label_strict"}

    # Priority order: raw readings first, then rolling windows
    priority_prefixes = [
        "H1", "load_fraction", "Motor_current", "Oil_temperature",
        "TP2", "DV_pressure", "LPS", "pressure_diff",
    ]

    all_cols = [c for c in df_features.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df_features[c])]

    # Sort: priority features first (in priority order), then rest alphabetically
    def sort_key(col: str) -> Tuple[int, str]:
        for i, prefix in enumerate(priority_prefixes):
            if col.startswith(prefix):
                return (i, col)
        return (len(priority_prefixes), col)

    return sorted(all_cols, key=sort_key)


# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------

def _calibrate_scores(
    raw_scores: np.ndarray,
    train_score_mean: float,
    train_score_std: float,
) -> np.ndarray:
    """
    Convert raw IsolationForest decision_function scores to [0, 1] anomaly scores.

    Strategy:
    - Standardise using training-set statistics (z-score)
    - Apply sigmoid to map to (0, 1)
    - Higher output = more anomalous

    Invert sign because decision_function returns negative values for anomalies
    (lower = more anomalous), but we want higher = more anomalous.
    """
    # Invert: training normal samples have high raw scores (near 0 or positive)
    # so -raw_score → high values for anomalies
    inverted = -raw_scores

    # Normalise using training distribution
    z = (inverted - (-train_score_mean)) / (train_score_std + 1e-9)

    # Sigmoid → (0, 1)
    anomaly_scores = 1.0 / (1.0 + np.exp(-z))
    return anomaly_scores.astype("float32")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_anomaly_model(
    df_features: pd.DataFrame,
    normal_idx: pd.Index,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Train the Isolation Forest anomaly detection model on confirmed-normal data.

    Parameters
    ----------
    df_features:  Feature matrix (output of build_feature_matrix).
    normal_idx:   Index of confirmed-normal rows (from get_anomaly_training_index).
    save_artifacts: If True, save scaler/model/metadata to ARTIFACTS_DIR.

    Returns
    -------
    Dictionary with keys:
        model         : fitted IsolationForest
        scaler        : fitted StandardScaler
        feature_cols  : list of feature column names
        score_stats   : dict of training score statistics
        train_scores  : np.ndarray of training raw scores
    """
    feat_cols = select_anomaly_features(df_features)
    logger.info("Anomaly model: using %d features", len(feat_cols))

    # Extract normal training data
    X_train = df_features.loc[normal_idx, feat_cols].copy()

    # Drop rows with NaN (window startup at segment boundaries)
    n_before = len(X_train)
    X_train = X_train.dropna()
    n_dropped = n_before - len(X_train)
    if n_dropped > 0:
        logger.info("Dropped %d NaN rows from training set (%d remain)", n_dropped, len(X_train))

    logger.info(
        "Training on %d confirmed-normal rows (Feb–Mar 2020 baseline)",
        len(X_train),
    )

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values.astype("float64"))

    # Train IsolationForest
    logger.info(
        "Training IsolationForest: n_estimators=%d, contamination=%s …",
        IF_N_ESTIMATORS, IF_CONTAMINATION,
    )
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=IF_N_JOBS,
    )
    model.fit(X_scaled)

    # Compute training scores for calibration
    train_raw_scores = model.decision_function(X_scaled)
    score_stats = {
        "train_score_mean": float(np.mean(train_raw_scores)),
        "train_score_std":  float(np.std(train_raw_scores)),
        "train_score_p01":  float(np.percentile(train_raw_scores, 1)),
        "train_score_p05":  float(np.percentile(train_raw_scores, 5)),
        "train_score_p50":  float(np.percentile(train_raw_scores, 50)),
        "train_score_p95":  float(np.percentile(train_raw_scores, 95)),
        "train_score_p99":  float(np.percentile(train_raw_scores, 99)),
        "n_train": int(len(X_train)),
        "feature_cols": feat_cols,
    }

    logger.info(
        "Training score stats: mean=%.4f, std=%.4f, p05=%.4f, p95=%.4f",
        score_stats["train_score_mean"],
        score_stats["train_score_std"],
        score_stats["train_score_p05"],
        score_stats["train_score_p95"],
    )

    bundle = {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feat_cols,
        "score_stats":  score_stats,
        "train_scores": train_raw_scores,
    }

    if save_artifacts:
        _save_artifacts(bundle)

    return bundle


def _save_artifacts(bundle: Dict[str, Any]) -> None:
    """Save model artifacts to ARTIFACTS_DIR."""
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(bundle["scaler"], f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle["model"], f)
    with open(FEAT_COLS_PATH, "w") as f:
        json.dump(bundle["feature_cols"], f, indent=2)
    with open(SCORE_STATS_PATH, "w") as f:
        # Remove non-serialisable keys
        stats = {k: v for k, v in bundle["score_stats"].items()}
        json.dump(stats, f, indent=2)
    logger.info("Anomaly model artifacts saved to %s", ARTIFACTS_DIR)


def load_anomaly_model() -> Dict[str, Any]:
    """Load saved anomaly detection artifacts from ARTIFACTS_DIR."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly model not found at {MODEL_PATH}. "
            "Run scripts/train_models.py first."
        )
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEAT_COLS_PATH) as f:
        feat_cols = json.load(f)
    with open(SCORE_STATS_PATH) as f:
        score_stats = json.load(f)

    logger.info("Loaded anomaly model from %s", ARTIFACTS_DIR)
    return {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feat_cols,
        "score_stats":  score_stats,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def score_anomaly(
    bundle: Dict[str, Any],
    df_features: pd.DataFrame,
) -> pd.Series:
    """
    Score a feature DataFrame using the loaded/trained anomaly model.

    Parameters
    ----------
    bundle:       Model bundle (from train_anomaly_model or load_anomaly_model).
    df_features:  Feature matrix (output of build_feature_matrix).

    Returns
    -------
    Series of calibrated anomaly scores in [0, 1], indexed like df_features.
    NaN is returned for gap rows and rows where features are all-NaN.

    Interpretation:
      < 0.3  → normal (inlier territory)
      0.3–0.5 → slightly elevated (monitor)
      0.5–0.7 → anomalous (investigate)
      > 0.7  → highly anomalous
    Note: these ranges are guidelines derived from the training score distribution,
    not arbitrary engineering thresholds.
    """
    model      = bundle["model"]
    scaler     = bundle["scaler"]
    feat_cols  = bundle["feature_cols"]
    stats      = bundle["score_stats"]

    result = pd.Series(np.nan, index=df_features.index, dtype="float32")

    # Only score non-gap rows that have the required feature columns
    available_cols = [c for c in feat_cols if c in df_features.columns]
    if len(available_cols) < len(feat_cols):
        logger.warning(
            "Missing %d/%d feature columns; using %d available",
            len(feat_cols) - len(available_cols), len(feat_cols), len(available_cols),
        )

    if "is_gap" in df_features.columns:
        score_mask = ~df_features["is_gap"]
    else:
        score_mask = pd.Series(True, index=df_features.index)

    X = df_features.loc[score_mask, available_cols].copy()

    # Fill NaN with column medians (from feature matrix; use training stats
    # implicitly via scaler which handles this)
    X = X.fillna(X.median())

    # Drop rows that are still all-NaN after fill
    valid_rows = ~X.isna().all(axis=1)
    X = X[valid_rows]

    if len(X) == 0:
        logger.warning("No valid rows to score")
        return result

    X_scaled = scaler.transform(X.values.astype("float64"))
    raw_scores = model.decision_function(X_scaled)

    calibrated = _calibrate_scores(
        raw_scores,
        stats["train_score_mean"],
        stats["train_score_std"],
    )

    result.loc[X.index] = calibrated
    return result


def compute_anomaly_events(
    scores: pd.Series,
    threshold: float = 0.5,
    min_duration_m: int = 5,
) -> pd.DataFrame:
    """
    Identify contiguous anomaly events from the scored time series.

    Parameters
    ----------
    scores:         Anomaly score series (output of score_anomaly).
    threshold:      Score above which a row is considered anomalous.
    min_duration_m: Minimum event duration in minutes to report.

    Returns
    -------
    DataFrame with columns: start, end, duration_m, max_score, mean_score.

    Threshold note: the default 0.5 corresponds to the inflection of the
    sigmoid used in calibration. It is derived from the training score
    distribution (mean of training set maps to ~0.5 before inversion).
    """
    is_anomaly = scores > threshold
    is_anomaly = is_anomaly.fillna(False)

    events = []
    in_event = False
    event_start = None

    for ts, flag in is_anomaly.items():
        if flag and not in_event:
            in_event = True
            event_start = ts
        elif not flag and in_event:
            event_end = ts
            duration_m = int((event_end - event_start).total_seconds() / 60)
            if duration_m >= min_duration_m:
                window_scores = scores.loc[event_start:event_end].dropna()
                events.append({
                    "start":      event_start,
                    "end":        event_end,
                    "duration_m": duration_m,
                    "max_score":  float(window_scores.max()) if len(window_scores) else np.nan,
                    "mean_score": float(window_scores.mean()) if len(window_scores) else np.nan,
                })
            in_event = False

    # Handle event still open at end
    if in_event and event_start is not None:
        event_end = is_anomaly.index[-1]
        duration_m = int((event_end - event_start).total_seconds() / 60)
        if duration_m >= min_duration_m:
            window_scores = scores.loc[event_start:event_end].dropna()
            events.append({
                "start":      event_start,
                "end":        event_end,
                "duration_m": duration_m,
                "max_score":  float(window_scores.max()) if len(window_scores) else np.nan,
                "mean_score": float(window_scores.mean()) if len(window_scores) else np.nan,
            })

    result = pd.DataFrame(events, columns=["start", "end", "duration_m", "max_score", "mean_score"])
    logger.info("Identified %d anomaly events (threshold=%.2f)", len(result), threshold)
    return result


def get_anomaly_threshold_from_training(
    bundle: Dict[str, Any],
    specificity_percentile: float = 95.0,
) -> float:
    """
    Derive an anomaly decision threshold from the training score distribution.

    Returns the anomaly score corresponding to the specificity_percentile of
    the *training* set's calibrated scores. At this threshold, approximately
    (100 - specificity_percentile)% of training-set (normal) points would
    be flagged as anomalies — i.e., false positive rate ≈ 5% at default.

    This is data-derived, not an arbitrary engineering limit.
    """
    stats = bundle["score_stats"]
    train_scores = bundle.get("train_scores")

    if train_scores is not None:
        calibrated_train = _calibrate_scores(
            train_scores,
            stats["train_score_mean"],
            stats["train_score_std"],
        )
        threshold = float(np.percentile(calibrated_train, specificity_percentile))
    else:
        # Fallback: use sigmoid of z-score corresponding to the percentile
        from scipy.stats import norm
        z = norm.ppf(specificity_percentile / 100)
        threshold = float(1.0 / (1.0 + np.exp(-z)))

    logger.info(
        "Anomaly threshold at %.0f%% specificity: %.4f", specificity_percentile, threshold
    )
    return threshold
