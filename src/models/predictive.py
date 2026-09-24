"""
src/models/predictive.py — Supervised failure-risk predictive models.

Design rationale and limitations
---------------------------------
The dataset contains only 4 documented failure events. Before building any
supervised model, we must be transparent about what this means:

LIMITATION:
  4 labeled events is insufficient for robust supervised learning in the
  classical sense. Any supervised model trained here will be:
  (a) unable to generalise to failure types not represented in the 4 events
  (b) sensitive to the specific temporal patterns of these 4 air-leak events
  (c) evaluated on 1 held-out event (F4), which is a single data point

  We include supervised models because:
  - The class separation is very strong (Stage 2 EDA shows clear sensor
    divergence between normal and failure states)
  - The pre-failure degradation in F3 and F4 is measurable
  - Evaluation results (even on 1 test event) are informative about whether
    the learned patterns generalise at all

  HOWEVER: the anomaly detection model (anomaly.py) is the PRIMARY predictive
  component. The supervised models are secondary/validation tools.

Model choices:
  1. Logistic Regression (interpretable baseline):
     - Interpretable coefficients
     - Works well when features are strongly predictive (which they are here)
     - Uses class_weight='balanced' to address 2% imbalance
     - L2 regularisation to prevent overfitting given small positive class

  2. Gradient Boosting (HistGradientBoostingClassifier):
     - More powerful non-linear model
     - Handles mixed-scale features natively
     - class_weight not directly supported → use scale_pos_weight equivalent
       via sample_weight
     - Compared to LR to check if non-linearity helps on this data

Temporal cross-validation:
  Train on Feb–Apr 2020 (contains F1)
  Validate on May–Jun 2020 (contains F2, F3) — for hyperparameter selection
  Test on Jul–Aug 2020 (contains F4) — held out, reported once

  NO random shuffling. NO k-fold across time.
  This matches the Stage 2 recommendation.

Evaluation:
  Primary:   Precision-Recall AUC (PR-AUC)
  Secondary: Precision, Recall, F1 at operating threshold
             Confusion matrix
  Not used:  Accuracy (misleading at 2% imbalance)

Saved artifacts:
  predictive_lr_model.pkl     : fitted LogisticRegression
  predictive_gbt_model.pkl    : fitted HistGradientBoostingClassifier
  predictive_scaler.pkl       : fitted StandardScaler (for LR only)
  predictive_feature_cols.json: feature columns
  predictive_results.json     : evaluation metrics

Usage:
    from src.models.predictive import train_predictive_models, load_predictive_models
    results = train_predictive_models(df_features, df_labels)
    models  = load_predictive_models()
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths
LR_MODEL_PATH    = ARTIFACTS_DIR / "predictive_lr_model.pkl"
GBT_MODEL_PATH   = ARTIFACTS_DIR / "predictive_gbt_model.pkl"
PRED_SCALER_PATH = ARTIFACTS_DIR / "predictive_scaler.pkl"
FEAT_COLS_PATH   = ARTIFACTS_DIR / "predictive_feature_cols.json"
RESULTS_PATH     = ARTIFACTS_DIR / "predictive_results.json"

# Logistic Regression hyperparameters
LR_C             = 0.1       # regularisation (tuned conservatively given small positive class)
LR_MAX_ITER      = 2000
LR_SOLVER        = "lbfgs"

# GBT hyperparameters
GBT_MAX_ITER     = 300
GBT_MAX_LEAF_NODES = 31
GBT_LEARNING_RATE = 0.05
GBT_L2_REG        = 1.0
GBT_RANDOM_STATE  = 42

# Operating threshold for binary classification (derived from PR curve)
# Will be stored after training
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights for class-balanced training.

    Returns an array where minority-class samples have higher weight
    proportional to the class imbalance ratio.
    Used for GBT which does not support class_weight='balanced' directly.
    """
    n_total = len(y)
    n_pos   = int(y.sum())
    n_neg   = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.ones(n_total)
    w_pos = n_total / (2 * n_pos)
    w_neg = n_total / (2 * n_neg)
    weights = np.where(y == 1, w_pos, w_neg)
    return weights


def _find_pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.5,
) -> float:
    """
    Find the classification threshold that achieves at least target_precision
    while maximising recall.

    Returns the threshold with best F1 score where precision >= target.
    Falls back to 0.5 if no threshold meets the constraint.

    This is data-derived: not an arbitrary engineering limit.
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns n+1 elements for precision/recall
    # and n elements for thresholds (last precision/recall has no threshold)
    best_thresh = DEFAULT_THRESHOLD
    best_f1 = 0.0

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision and r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

    if best_f1 == 0.0:
        logger.warning(
            "No threshold achieves precision >= %.2f; using default %.2f",
            target_precision, DEFAULT_THRESHOLD,
        )

    return best_thresh


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_predictive_models(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Train Logistic Regression and Gradient Boosting failure-risk models.

    Parameters
    ----------
    df_features:  Feature matrix (from build_feature_matrix).
    df_labels:    DataFrame with 'label' column (from build_labels).
    train_idx:    Training set index (chronological first split).
    val_idx:      Validation set index.
    test_idx:     Test set index.
    save_artifacts: Save model artifacts to disk.

    Returns
    -------
    Dictionary with models, scalers, feature columns, and evaluation metrics.
    """
    from src.models.evaluation import evaluate_model

    # ── Feature selection ────────────────────────────────────────────────────
    # Use the same feature columns as anomaly detection (consistent preprocessing)
    from src.models.anomaly import select_anomaly_features
    feat_cols = select_anomaly_features(df_features)
    feat_cols = [c for c in feat_cols if c in df_features.columns]
    logger.info("Predictive model: using %d features", len(feat_cols))

    # ── Prepare splits ───────────────────────────────────────────────────────
    def _get_xy(idx: pd.Index) -> Tuple[pd.DataFrame, np.ndarray]:
        # Intersect: some idx rows may have been dropped from feature matrix
        common_idx = idx.intersection(df_features.index).intersection(df_labels.index)
        X = df_features.loc[common_idx, feat_cols].copy()
        y = df_labels.loc[common_idx, "label"].values.astype("int8")
        # Fill NaN: use column median from training set
        X = X.fillna(X.median())
        return X.astype("float32"), y

    X_train, y_train = _get_xy(train_idx)
    X_val,   y_val   = _get_xy(val_idx)
    X_test,  y_test  = _get_xy(test_idx)

    logger.info(
        "Splits: train=%d (pos=%d, %.1f%%), val=%d (pos=%d, %.1f%%), test=%d (pos=%d, %.1f%%)",
        len(y_train), y_train.sum(), 100 * y_train.mean(),
        len(y_val),   y_val.sum(),   100 * y_val.mean(),
        len(y_test),  y_test.sum(),  100 * y_test.mean(),
    )

    # ── Logistic Regression ──────────────────────────────────────────────────
    logger.info("Training Logistic Regression …")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train.values.astype("float64"))
    X_val_sc   = scaler.transform(X_val.values.astype("float64"))
    X_test_sc  = scaler.transform(X_test.values.astype("float64"))

    lr_model = LogisticRegression(
        C=LR_C,
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        solver=LR_SOLVER,
        random_state=42,
    )
    lr_model.fit(X_train_sc, y_train)

    lr_val_prob  = lr_model.predict_proba(X_val_sc)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test_sc)[:, 1]

    # Threshold derived from validation PR curve
    lr_threshold = _find_pr_threshold(y_val, lr_val_prob, target_precision=0.5)
    logger.info("LR operating threshold (from val PR curve): %.4f", lr_threshold)

    lr_val_metrics  = evaluate_model(y_val,  lr_val_prob,  threshold=lr_threshold, name="LR val")
    lr_test_metrics = evaluate_model(y_test, lr_test_prob, threshold=lr_threshold, name="LR test")

    # ── Gradient Boosting ────────────────────────────────────────────────────
    logger.info("Training HistGradientBoostingClassifier …")
    sample_weights = _compute_sample_weights(y_train)

    gbt_model = HistGradientBoostingClassifier(
        max_iter=GBT_MAX_ITER,
        max_leaf_nodes=GBT_MAX_LEAF_NODES,
        learning_rate=GBT_LEARNING_RATE,
        l2_regularization=GBT_L2_REG,
        random_state=GBT_RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=0,
    )
    gbt_model.fit(
        X_train.values.astype("float64"),
        y_train,
        sample_weight=sample_weights,
    )

    gbt_val_prob  = gbt_model.predict_proba(X_val.values.astype("float64"))[:, 1]
    gbt_test_prob = gbt_model.predict_proba(X_test.values.astype("float64"))[:, 1]

    gbt_threshold = _find_pr_threshold(y_val, gbt_val_prob, target_precision=0.5)
    logger.info("GBT operating threshold (from val PR curve): %.4f", gbt_threshold)

    gbt_val_metrics  = evaluate_model(y_val,  gbt_val_prob,  threshold=gbt_threshold, name="GBT val")
    gbt_test_metrics = evaluate_model(y_test, gbt_test_prob, threshold=gbt_threshold, name="GBT test")

    # ── Feature importance (LR coefficients) ────────────────────────────────
    lr_coefs = dict(zip(feat_cols, lr_model.coef_[0].tolist()))
    lr_coef_sorted = sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("Top 10 LR feature coefficients:")
    for fname, coef in lr_coef_sorted[:10]:
        logger.info("  %-35s  %+.4f", fname, coef)

    # ── Assemble results ─────────────────────────────────────────────────────
    results = {
        "lr": {
            "val_metrics":  lr_val_metrics,
            "test_metrics": lr_test_metrics,
            "threshold":    lr_threshold,
        },
        "gbt": {
            "val_metrics":  gbt_val_metrics,
            "test_metrics": gbt_test_metrics,
            "threshold":    gbt_threshold,
        },
        "feature_cols": feat_cols,
        "train_size": len(y_train),
        "val_size":   len(y_val),
        "test_size":  len(y_test),
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate":   float(y_val.mean()),
        "test_pos_rate":  float(y_test.mean()),
        "lr_coef_sorted": lr_coef_sorted,
        "limitation": (
            "Only 4 labeled failure events used. Supervised model is indicative "
            "only. Do not use for production decisions without more failure data."
        ),
    }

    bundle = {
        "lr_model":    lr_model,
        "gbt_model":   gbt_model,
        "scaler":      scaler,
        "feat_cols":   feat_cols,
        "lr_threshold": lr_threshold,
        "gbt_threshold": gbt_threshold,
        "results":     results,
        # Store test probabilities for PR curve plotting
        "y_test":      y_test,
        "lr_test_prob": lr_test_prob,
        "gbt_test_prob": gbt_test_prob,
        "y_val":       y_val,
        "lr_val_prob": lr_val_prob,
        "gbt_val_prob": gbt_val_prob,
    }

    if save_artifacts:
        _save_artifacts(bundle)

    return bundle


def _save_artifacts(bundle: Dict[str, Any]) -> None:
    """Persist model artifacts."""
    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(bundle["lr_model"], f)
    with open(GBT_MODEL_PATH, "wb") as f:
        pickle.dump(bundle["gbt_model"], f)
    with open(PRED_SCALER_PATH, "wb") as f:
        pickle.dump(bundle["scaler"], f)
    with open(FEAT_COLS_PATH, "w") as f:
        json.dump(bundle["feat_cols"], f, indent=2)

    # Save results (convert numpy types for JSON serialisation)
    def _to_json(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json(i) for i in obj]
        return obj

    with open(RESULTS_PATH, "w") as f:
        json.dump(_to_json(bundle["results"]), f, indent=2)

    # Also save test predictions for dashboard PR curves
    np.save(str(ARTIFACTS_DIR / "pred_y_test.npy"), bundle["y_test"])
    np.save(str(ARTIFACTS_DIR / "pred_lr_prob.npy"), bundle["lr_test_prob"])
    np.save(str(ARTIFACTS_DIR / "pred_gbt_prob.npy"), bundle["gbt_test_prob"])

    logger.info("Predictive model artifacts saved to %s", ARTIFACTS_DIR)


def load_predictive_models() -> Dict[str, Any]:
    """Load saved predictive model artifacts."""
    if not LR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Predictive models not found at {LR_MODEL_PATH}. "
            "Run scripts/train_models.py first."
        )
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(GBT_MODEL_PATH, "rb") as f:
        gbt_model = pickle.load(f)
    with open(PRED_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEAT_COLS_PATH) as f:
        feat_cols = json.load(f)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    y_test_path = ARTIFACTS_DIR / "pred_y_test.npy"
    y_test      = np.load(str(y_test_path)) if y_test_path.exists() else None
    lr_prob_path = ARTIFACTS_DIR / "pred_lr_prob.npy"
    lr_prob      = np.load(str(lr_prob_path)) if lr_prob_path.exists() else None
    gbt_prob_path = ARTIFACTS_DIR / "pred_gbt_prob.npy"
    gbt_prob      = np.load(str(gbt_prob_path)) if gbt_prob_path.exists() else None

    logger.info("Loaded predictive models from %s", ARTIFACTS_DIR)
    return {
        "lr_model":     lr_model,
        "gbt_model":    gbt_model,
        "scaler":       scaler,
        "feat_cols":    feat_cols,
        "results":      results,
        "lr_threshold": results["lr"]["threshold"],
        "gbt_threshold": results["gbt"]["threshold"],
        "y_test":       y_test,
        "lr_test_prob": lr_prob,
        "gbt_test_prob": gbt_prob,
    }


def predict_risk(
    bundle: Dict[str, Any],
    df_features: pd.DataFrame,
    model: str = "gbt",
) -> pd.Series:
    """
    Score new data with the predictive risk model.

    Parameters
    ----------
    bundle:      Model bundle from load_predictive_models() or train_predictive_models().
    df_features: Feature matrix.
    model:       'lr' or 'gbt'.

    Returns
    -------
    Series of risk probabilities in [0, 1].
    """
    feat_cols = bundle["feat_cols"]
    available = [c for c in feat_cols if c in df_features.columns]

    if "is_gap" in df_features.columns:
        score_mask = ~df_features["is_gap"]
    else:
        score_mask = pd.Series(True, index=df_features.index)

    result = pd.Series(np.nan, index=df_features.index, dtype="float32")
    X = df_features.loc[score_mask, available].copy().fillna(0.0).astype("float64")

    if model == "lr":
        X_sc = bundle["scaler"].transform(X.values)
        probs = bundle["lr_model"].predict_proba(X_sc)[:, 1]
    else:
        probs = bundle["gbt_model"].predict_proba(X.values)[:, 1]

    result.loc[X.index] = probs.astype("float32")
    return result
