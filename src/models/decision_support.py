"""
src/models/decision_support.py — Maintenance decision support layer.

Purpose
-------
This module translates raw anomaly scores and risk probabilities into
actionable maintenance categories and evidence summaries.

IMPORTANT DISCLAIMER (shown in dashboard):
  This is a decision support tool, not an automatic maintenance diagnosis.
  All outputs should be reviewed by a qualified maintenance engineer before
  any action is taken. The system cannot account for external factors not
  present in sensor data.

Decision categories
-------------------
The three tiers are defined from the model output distribution, not from
manually chosen engineering thresholds:

  NORMAL       anomaly_score < low_threshold
               The sensor pattern is consistent with known-normal operation.

  MONITOR      low_threshold ≤ anomaly_score < high_threshold
               Slightly elevated anomaly score. Not sufficient for action
               but warrants continued observation.

  INVESTIGATE  anomaly_score ≥ high_threshold  OR  risk_prob ≥ risk_threshold
               Strong deviation from normal pattern. Investigation recommended.
               Does NOT mean a failure has occurred or is imminent.

Threshold derivation
--------------------
Thresholds are derived from the anomaly model's training score distribution:
  low_threshold  = 95th percentile of training scores (5% false positive rate)
  high_threshold = 99th percentile of training scores (1% false positive rate)

Risk probability threshold is derived from the validation PR curve:
  the operating threshold stored in predictive_results.json.

Evidence summary
----------------
For each row (or time window), the decision support layer reports:
  1. Decision category
  2. Primary evidence: which sensors/features contribute most to the score
  3. Recent sensor values (last 30 minutes)
  4. Comparison to training baseline statistics

Sensor evidence is computed from the feature matrix, not from SHAP values,
to avoid requiring SHAP at inference time. However, the SHAP feature importance
ranking is used to identify which features to highlight.

Usage:
    from src.models.decision_support import DecisionSupport
    ds = DecisionSupport.from_artifacts()
    category, evidence = ds.assess(anomaly_score, risk_prob, feature_row)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# ── Decision categories ───────────────────────────────────────────────────────

CATEGORY_NORMAL      = "Normal"
CATEGORY_MONITOR     = "Monitor"
CATEGORY_INVESTIGATE = "Investigate"

CATEGORY_COLOURS = {
    CATEGORY_NORMAL:      "#2A9D8F",   # teal
    CATEGORY_MONITOR:     "#E9C46A",   # amber
    CATEGORY_INVESTIGATE: "#E63946",   # red
}

CATEGORY_ICONS = {
    CATEGORY_NORMAL:      "✅",
    CATEGORY_MONITOR:     "⚠️",
    CATEGORY_INVESTIGATE: "🔴",
}


@dataclass
class DecisionResult:
    """Structured result from the decision support assessment."""
    category:        str
    anomaly_score:   float
    risk_prob:       float
    evidence:        List[Dict[str, Any]]   # [{feature, value, baseline_mean, deviation, label}]
    summary_text:    str
    thresholds_used: Dict[str, float]


@dataclass
class DecisionSupport:
    """
    Decision support system combining anomaly scores and risk probabilities.

    Attributes
    ----------
    anomaly_threshold_low:  Lower anomaly score threshold (below = Normal).
    anomaly_threshold_high: Upper anomaly score threshold (above = Investigate).
    risk_threshold:         Risk probability threshold for Investigate.
    feature_importance:     Ordered list of {feature, mean_abs_shap} dicts.
    baseline_stats:         Dict of {feature: {mean, std}} from normal training data.
    """
    anomaly_threshold_low:  float = 0.45
    anomaly_threshold_high: float = 0.55
    risk_threshold:         float = 0.5
    feature_importance:     List[Dict] = field(default_factory=list)
    baseline_stats:         Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path = ARTIFACTS_DIR) -> "DecisionSupport":
        """
        Construct a DecisionSupport instance from saved model artifacts.

        Loads:
        - Anomaly score thresholds from anomaly_score_stats.json
        - Risk threshold from predictive_results.json
        - Feature importance from shap_feature_importance.json
        - Baseline stats from anomaly_baseline_stats.json (if present)
        """
        score_stats_path = artifacts_dir / "anomaly_score_stats.json"
        pred_results_path = artifacts_dir / "predictive_results.json"
        shap_importance_path = artifacts_dir / "shap_feature_importance.json"
        baseline_stats_path  = artifacts_dir / "baseline_stats.json"

        # Anomaly thresholds from training distribution
        if score_stats_path.exists():
            with open(score_stats_path) as f:
                score_stats = json.load(f)
            # Derive thresholds from training calibrated score distribution
            # These are computed during training and stored
            low_thresh  = score_stats.get("alert_threshold_low",  0.45)
            high_thresh = score_stats.get("alert_threshold_high", 0.55)
        else:
            logger.warning("anomaly_score_stats.json not found; using defaults")
            low_thresh, high_thresh = 0.45, 0.55

        # Risk threshold from predictive model
        if pred_results_path.exists():
            with open(pred_results_path) as f:
                pred_results = json.load(f)
            # Use GBT threshold if available, else LR
            risk_thresh = pred_results.get("gbt", {}).get("threshold",
                          pred_results.get("lr", {}).get("threshold", 0.5))
        else:
            logger.warning("predictive_results.json not found; using default risk threshold")
            risk_thresh = 0.5

        # Feature importance
        feature_importance = []
        if shap_importance_path.exists():
            with open(shap_importance_path) as f:
                feature_importance = json.load(f)

        # Baseline stats
        baseline_stats = {}
        if baseline_stats_path.exists():
            with open(baseline_stats_path) as f:
                baseline_stats = json.load(f)

        ds = cls(
            anomaly_threshold_low=float(low_thresh),
            anomaly_threshold_high=float(high_thresh),
            risk_threshold=float(risk_thresh),
            feature_importance=feature_importance,
            baseline_stats=baseline_stats,
        )
        logger.info(
            "DecisionSupport loaded: thresholds=%.3f/%.3f, risk_thresh=%.3f",
            low_thresh, high_thresh, risk_thresh,
        )
        return ds

    def classify(
        self,
        anomaly_score: float,
        risk_prob: float = 0.0,
    ) -> str:
        """
        Classify a single observation into a decision category.

        Parameters
        ----------
        anomaly_score: Calibrated anomaly score in [0, 1].
        risk_prob:     Supervised risk probability in [0, 1] (optional).

        Returns
        -------
        One of CATEGORY_NORMAL, CATEGORY_MONITOR, CATEGORY_INVESTIGATE.
        """
        if np.isnan(anomaly_score):
            return CATEGORY_NORMAL

        if (anomaly_score >= self.anomaly_threshold_high or
                risk_prob >= self.risk_threshold):
            return CATEGORY_INVESTIGATE
        elif anomaly_score >= self.anomaly_threshold_low:
            return CATEGORY_MONITOR
        else:
            return CATEGORY_NORMAL

    def assess(
        self,
        anomaly_score: float,
        feature_row: Optional[pd.Series],
        risk_prob: float = 0.0,
        top_n_evidence: int = 5,
    ) -> DecisionResult:
        """
        Produce a full decision result with evidence for a single time point.

        Parameters
        ----------
        anomaly_score:  Calibrated anomaly score.
        feature_row:    Feature values for this time point (pd.Series).
        risk_prob:      Supervised risk probability (optional).
        top_n_evidence: Number of top contributing features to include.

        Returns
        -------
        DecisionResult with category, evidence, and summary text.
        """
        category = self.classify(anomaly_score, risk_prob)

        # Build evidence from feature values vs baseline
        evidence = []
        if feature_row is not None and self.feature_importance:
            for item in self.feature_importance[:top_n_evidence * 2]:
                feat = item["feature"]
                if feat not in feature_row.index:
                    continue
                val = float(feature_row.get(feat, np.nan))
                if np.isnan(val):
                    continue

                baseline = self.baseline_stats.get(feat, {})
                b_mean = baseline.get("mean", np.nan)
                b_std  = baseline.get("std",  np.nan)

                if not np.isnan(b_mean) and not np.isnan(b_std) and b_std > 0:
                    z_score = (val - b_mean) / b_std
                    deviation_label = _deviation_label(z_score, feat)
                else:
                    z_score = np.nan
                    deviation_label = "unknown baseline"

                evidence.append({
                    "feature":        feat,
                    "value":          val,
                    "baseline_mean":  b_mean,
                    "baseline_std":   b_std,
                    "z_score":        float(z_score) if not np.isnan(z_score) else None,
                    "deviation_label": deviation_label,
                    "shap_importance": item.get("mean_abs_shap", 0),
                })

                if len(evidence) >= top_n_evidence:
                    break

        summary_text = _build_summary(category, anomaly_score, risk_prob, evidence)

        return DecisionResult(
            category=category,
            anomaly_score=float(anomaly_score) if not np.isnan(anomaly_score) else 0.0,
            risk_prob=float(risk_prob) if not np.isnan(risk_prob) else 0.0,
            evidence=evidence,
            summary_text=summary_text,
            thresholds_used={
                "anomaly_low":  self.anomaly_threshold_low,
                "anomaly_high": self.anomaly_threshold_high,
                "risk":         self.risk_threshold,
            },
        )

    def classify_series(
        self,
        anomaly_scores: pd.Series,
        risk_probs: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Classify a time series of anomaly scores and risk probabilities.

        Returns a Series of decision category strings with the same index.
        """
        if risk_probs is None:
            risk_probs = pd.Series(0.0, index=anomaly_scores.index)

        categories = []
        for ts in anomaly_scores.index:
            a_score = float(anomaly_scores.loc[ts]) if ts in anomaly_scores.index else np.nan
            r_prob  = float(risk_probs.loc[ts]) if ts in risk_probs.index else 0.0
            categories.append(self.classify(a_score, r_prob))

        return pd.Series(categories, index=anomaly_scores.index)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _deviation_label(z_score: float, feature_name: str) -> str:
    """
    Convert a z-score to a human-readable deviation label.

    Based on empirical deviation ranges, not model-specific thresholds.
    """
    if np.isnan(z_score):
        return "unknown"

    # Key failure-indicator features: we know from Stage 2 what direction matters
    failure_direction_up = {
        "Motor_current", "Oil_temperature", "TP2", "DV_pressure", "load_fraction",
        "LPS",
    }
    failure_direction_down = {"H1"}

    direction = "high" if z_score > 0 else "low"
    abs_z = abs(z_score)

    if abs_z < 1.0:
        return "within normal range"
    elif abs_z < 2.0:
        severity = "slightly"
    elif abs_z < 3.5:
        severity = "notably"
    else:
        severity = "significantly"

    label = f"{severity} {direction} vs normal baseline"

    # Add failure-relevance note for key sensors
    base = feature_name.split("_30m")[0].split("_2h")[0].split("_6h")[0]
    if base in failure_direction_up and z_score > 2:
        label += " (elevated — failure-associated direction)"
    elif base in failure_direction_down and z_score < -2:
        label += " (depressed — failure-associated direction)"

    return label


def _build_summary(
    category: str,
    anomaly_score: float,
    risk_prob: float,
    evidence: List[Dict],
) -> str:
    """Build a plain-English summary of the decision."""
    if category == CATEGORY_NORMAL:
        return (
            f"Equipment is operating within normal parameters "
            f"(anomaly score: {anomaly_score:.3f}). "
            "No maintenance action required based on current sensor readings."
        )
    elif category == CATEGORY_MONITOR:
        top_feat = evidence[0]["feature"] if evidence else "unknown"
        top_feat_clean = top_feat.replace("_", " ")
        return (
            f"Slightly elevated anomaly score ({anomaly_score:.3f}) detected. "
            f"Primary contributing feature: {top_feat_clean}. "
            "Monitor trends over the next few hours. "
            "This threshold is exceeded by approximately 5% of normal operating periods, "
            "so this is not necessarily indicative of impending failure."
        )
    else:  # INVESTIGATE
        top_feats = [e["feature"].replace("_", " ") for e in evidence[:3]]
        feat_str = ", ".join(top_feats) if top_feats else "multiple sensors"
        return (
            f"Anomaly score {anomaly_score:.3f} exceeds the 99th percentile of the "
            f"normal baseline. Risk probability: {risk_prob:.1%}. "
            f"Primary contributing sensors: {feat_str}. "
            "Investigation is recommended. This system identifies unusual patterns — "
            "a qualified engineer should inspect the equipment and verify the sensor readings "
            "before scheduling maintenance."
        )


def compute_baseline_stats(
    df_features: pd.DataFrame,
    normal_idx: pd.Index,
    feat_cols: List[str],
) -> Dict[str, Dict]:
    """
    Compute baseline statistics (mean, std, percentiles) from confirmed-normal data.

    Saved to data/artifacts/baseline_stats.json for use by DecisionSupport.
    """
    available = [c for c in feat_cols if c in df_features.columns]
    X_normal = df_features.loc[normal_idx, available].dropna()

    stats = {}
    for col in available:
        s = X_normal[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "mean": float(s.mean()),
            "std":  float(s.std()),
            "p01":  float(s.quantile(0.01)),
            "p05":  float(s.quantile(0.05)),
            "p50":  float(s.quantile(0.50)),
            "p95":  float(s.quantile(0.95)),
            "p99":  float(s.quantile(0.99)),
        }

    out_path = ARTIFACTS_DIR / "baseline_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Baseline stats saved: %d features → %s", len(stats), out_path)
    return stats
