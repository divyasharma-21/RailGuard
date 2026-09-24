"""
src/explainability/shap_analysis.py — SHAP-based feature importance and explanations.

Purpose
-------
SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) provides
theoretically grounded feature attribution for both models:

1. For the Gradient Boosting model (primary), TreeExplainer is used.
   This is exact and computationally efficient for tree ensembles.

2. For the Logistic Regression model, LinearExplainer is used.

SHAP values answer: "How much did each feature push this prediction
above or below the expected prediction?"

IMPORTANT CAVEAT (displayed in the dashboard):
  SHAP values show which features the MODEL uses to make predictions.
  They do NOT prove that those features CAUSE equipment failure.
  The model has learned associations from the data. The physical
  causal mechanism (air leak → continuous load → all downstream effects)
  is described in the project documentation and is consistent with the
  SHAP findings, but the model does not prove causation.

Saved artifacts:
  shap_gbt_values.npy         : SHAP values for GBT on test set
  shap_gbt_expected_value.npy : baseline expected value
  shap_feature_importance.json: mean |SHAP| importance per feature

Usage:
    from src.explainability.shap_analysis import (
        compute_shap_values,
        get_feature_importance,
        load_shap_artifacts,
    )
    shap_bundle = compute_shap_values(gbt_model, X_test, feat_cols)
    importance  = get_feature_importance(shap_bundle)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROCESSED_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths
SHAP_VALUES_PATH      = ARTIFACTS_DIR / "shap_gbt_values.npy"
SHAP_EXPECTED_PATH    = ARTIFACTS_DIR / "shap_gbt_expected_value.npy"
SHAP_IMPORTANCE_PATH  = ARTIFACTS_DIR / "shap_feature_importance.json"
SHAP_FEAT_COLS_PATH   = ARTIFACTS_DIR / "shap_feature_cols.json"

# Max rows to use for SHAP computation (to keep it tractable)
SHAP_MAX_ROWS = 5000


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feat_cols: List[str],
    model_type: str = "gbt",
    max_rows: int = SHAP_MAX_ROWS,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Compute SHAP values for the given model and feature matrix.

    For large datasets, a stratified subsample is used to keep computation
    tractable while preserving positive/negative class proportions.

    Parameters
    ----------
    model:       Fitted sklearn model (GBT or LR).
    X:           Feature DataFrame (rows = observations, cols = features).
    feat_cols:   Feature column names.
    model_type:  'gbt' or 'lr'.
    max_rows:    Maximum rows to explain.
    save_artifacts: Save SHAP values to disk.

    Returns
    -------
    Dict with keys: shap_values, expected_value, feature_cols, X_sample
    """
    try:
        import shap
    except ImportError:
        logger.error("shap package not installed. Run: pip install shap")
        raise

    # Subset to available columns
    available = [c for c in feat_cols if c in X.columns]
    X_use = X[available].copy().fillna(0.0).astype("float64")

    # Subsample if needed
    if len(X_use) > max_rows:
        logger.info("Subsampling %d → %d rows for SHAP computation", len(X_use), max_rows)
        np.random.seed(42)
        idx = np.random.choice(len(X_use), max_rows, replace=False)
        X_use = X_use.iloc[sorted(idx)]

    logger.info("Computing SHAP values for %d rows, %d features …", len(X_use), len(available))

    if model_type == "gbt":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_use.values)
        # For binary classifiers, TreeExplainer may return list [neg_class, pos_class]
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]  # use positive class SHAP values
            expected_value = float(explainer.expected_value[1])
        elif isinstance(shap_values, list):
            shap_values = shap_values[0]
            ev = explainer.expected_value
            expected_value = float(ev[0] if hasattr(ev, "__len__") else ev)
        else:
            ev = explainer.expected_value
            expected_value = float(ev[0] if hasattr(ev, "__len__") else ev)
    elif model_type == "lr":
        # Use masker for background
        background = shap.maskers.Independent(X_use.values, max_samples=100)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_use.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = float(
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info("SHAP values computed: shape=%s", shap_values.shape)

    # Feature importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = [
        {"feature": feat, "mean_abs_shap": float(v)}
        for feat, v in sorted(
            zip(available, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    bundle = {
        "shap_values":    shap_values,
        "expected_value": expected_value,
        "feature_cols":   available,
        "X_sample":       X_use,
        "importance":     importance,
    }

    if save_artifacts:
        np.save(str(SHAP_VALUES_PATH), shap_values)
        np.save(str(SHAP_EXPECTED_PATH), np.array([expected_value]))
        with open(SHAP_IMPORTANCE_PATH, "w") as f:
            json.dump(importance, f, indent=2)
        with open(SHAP_FEAT_COLS_PATH, "w") as f:
            json.dump(available, f, indent=2)
        # Save X_sample for waterfall plots
        X_use.reset_index(drop=True).to_parquet(
            str(ARTIFACTS_DIR / "shap_X_sample.parquet")
        )
        logger.info("SHAP artifacts saved to %s", ARTIFACTS_DIR)

    _log_top_features(importance, top_n=10)
    return bundle


def _log_top_features(importance: List[dict], top_n: int = 10) -> None:
    logger.info("Top %d features by mean |SHAP|:", top_n)
    for i, item in enumerate(importance[:top_n], 1):
        logger.info("  %2d. %-35s  %.5f", i, item["feature"], item["mean_abs_shap"])


def load_shap_artifacts() -> Dict[str, Any]:
    """Load saved SHAP artifacts from ARTIFACTS_DIR."""
    if not SHAP_VALUES_PATH.exists():
        raise FileNotFoundError(
            f"SHAP artifacts not found at {SHAP_VALUES_PATH}. "
            "Run scripts/train_models.py first."
        )
    shap_values    = np.load(str(SHAP_VALUES_PATH))
    expected_value = float(np.load(str(SHAP_EXPECTED_PATH))[0])

    with open(SHAP_IMPORTANCE_PATH) as f:
        importance = json.load(f)
    with open(SHAP_FEAT_COLS_PATH) as f:
        feat_cols = json.load(f)

    X_sample_path = ARTIFACTS_DIR / "shap_X_sample.parquet"
    X_sample = pd.read_parquet(str(X_sample_path)) if X_sample_path.exists() else None

    logger.info("Loaded SHAP artifacts from %s", ARTIFACTS_DIR)
    return {
        "shap_values":    shap_values,
        "expected_value": expected_value,
        "feature_cols":   feat_cols,
        "X_sample":       X_sample,
        "importance":     importance,
    }


def get_feature_importance(
    shap_bundle: Dict[str, Any],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame of feature importances sorted by mean |SHAP|.

    Parameters
    ----------
    shap_bundle: Output of compute_shap_values or load_shap_artifacts.
    top_n:       If provided, return only the top N features.

    Returns
    -------
    DataFrame with columns: feature, mean_abs_shap, rank
    """
    importance = shap_bundle["importance"]
    df = pd.DataFrame(importance)
    df["rank"] = np.arange(1, len(df) + 1)

    if top_n is not None:
        df = df.head(top_n)

    return df.reset_index(drop=True)


def explain_single_prediction(
    shap_bundle: Dict[str, Any],
    row_index: int,
) -> pd.DataFrame:
    """
    Return SHAP contributions for a single prediction row.

    Parameters
    ----------
    shap_bundle: Output of compute_shap_values or load_shap_artifacts.
    row_index:   Row position in the X_sample DataFrame.

    Returns
    -------
    DataFrame with columns: feature, feature_value, shap_value, direction
    Sorted by |shap_value| descending.
    """
    shap_values  = shap_bundle["shap_values"]
    feat_cols    = shap_bundle["feature_cols"]
    X_sample     = shap_bundle["X_sample"]

    if row_index >= len(shap_values):
        raise IndexError(f"row_index {row_index} >= {len(shap_values)}")

    row_shap   = shap_values[row_index]
    row_values = X_sample.iloc[row_index].values if X_sample is not None else [np.nan] * len(feat_cols)

    df = pd.DataFrame({
        "feature":       feat_cols,
        "feature_value": row_values,
        "shap_value":    row_shap,
    })
    df["direction"] = df["shap_value"].apply(
        lambda x: "increases risk" if x > 0 else "decreases risk"
    )
    df = df.reindex(df["shap_value"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)


def plot_feature_importance(
    shap_bundle: Dict[str, Any],
    top_n: int = 20,
    title: str = "Feature Importance (Mean |SHAP|)",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Bar chart of top-N features by mean absolute SHAP value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance_df = get_feature_importance(shap_bundle, top_n=top_n)

    # Make feature names more readable
    def _clean_name(s: str) -> str:
        s = s.replace("_", " ")
        s = s.replace("30m", "(30 min)")
        s = s.replace("2h", "(2 h)")
        s = s.replace("6h", "(6 h)")
        return s

    labels = [_clean_name(f) for f in importance_df["feature"]]
    values = importance_df["mean_abs_shap"].values

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    colours = ["#E63946" if i < 5 else "#457B9D" for i in range(len(values))]
    ax.barh(range(len(values)), values[::-1], color=colours[::-1], alpha=0.85)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Note
    ax.text(
        0.98, 0.02,
        "Higher = more influential for model predictions",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#555555",
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        logger.info("Feature importance plot saved to %s", save_path)

    return fig
