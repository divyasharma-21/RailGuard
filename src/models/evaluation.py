"""
src/models/evaluation.py — Model evaluation metrics for imbalanced classification.

All metrics are reported for the failure-detection problem where:
- Positive class (label=1): failure or pre-failure at-risk window
- Negative class (label=0): normal operation
- Class imbalance: ~2–4% positive

Primary metric: Precision-Recall AUC (PR-AUC)
  Unlike ROC-AUC, PR-AUC is informative under class imbalance because it
  focuses on the minority class. A random classifier achieves PR-AUC ≈ base
  rate (~0.02–0.04), so useful models must substantially exceed this.

Secondary metrics:
  Precision: of predicted positives, how many are actually failures?
  Recall:    of actual failures, how many were detected?
  F1:        harmonic mean of precision and recall

Not used as primary metric:
  Accuracy: would be ~98% for a trivial "always predict normal" classifier.
  ROC-AUC: less informative under severe class imbalance.

Usage:
    from src.models.evaluation import evaluate_model, plot_pr_curve
    metrics = evaluate_model(y_true, y_prob, threshold=0.4, name="GBT test")
    fig = plot_pr_curve(y_true, y_prob, label="GBT", title="Test Set PR Curve")
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

FIG_DIR = PROCESSED_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    name: str = "model",
) -> Dict[str, float]:
    """
    Compute classification metrics for a binary failure-detection model.

    Parameters
    ----------
    y_true:    True binary labels (0=normal, 1=failure/at-risk).
    y_prob:    Predicted probabilities for the positive class.
    threshold: Classification threshold for binary predictions.
    name:      Name for logging/reporting.

    Returns
    -------
    Dictionary with keys:
        pr_auc, roc_auc, precision, recall, f1,
        tp, fp, tn, fn,
        threshold, n_positive, n_negative, base_rate
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype("int8")

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    n_total = len(y_true)
    base_rate = n_pos / n_total if n_total > 0 else 0.0

    if n_pos == 0:
        logger.warning("%s: no positive samples — metrics are meaningless", name)
        return {
            "pr_auc": 0.0, "roc_auc": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "tn": n_neg, "fn": 0,
            "threshold": threshold,
            "n_positive": n_pos, "n_negative": n_neg, "base_rate": base_rate,
        }

    pr_auc  = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "pr_auc":      pr_auc,
        "roc_auc":     roc_auc,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "tp":          int(tp),
        "fp":          int(fp),
        "tn":          int(tn),
        "fn":          int(fn),
        "threshold":   threshold,
        "n_positive":  n_pos,
        "n_negative":  n_neg,
        "base_rate":   base_rate,
    }

    logger.info(
        "%s → PR-AUC=%.4f, ROC-AUC=%.4f, P=%.3f, R=%.3f, F1=%.3f | "
        "TP=%d, FP=%d, TN=%d, FN=%d (threshold=%.3f, base_rate=%.3f%%)",
        name, pr_auc, roc_auc, precision, recall, f1,
        int(tp), int(fp), int(tn), int(fn), threshold, 100 * base_rate,
    )
    return metrics


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob_dict: Dict[str, np.ndarray],
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for one or more models.

    Parameters
    ----------
    y_true:       True binary labels.
    y_prob_dict:  Dict of {model_name: y_prob} for each model to plot.
    title:        Plot title.
    save_path:    If provided, save figure to this path.

    Returns
    -------
    Matplotlib Figure object.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    base_rate = float(y_true.mean()) if len(y_true) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))

    colours = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    for i, (label, y_prob) in enumerate(y_prob_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        colour = colours[i % len(colours)]
        ax.plot(
            recall, precision,
            label=f"{label} (PR-AUC={ap:.3f})",
            lw=2, color=colour,
        )

    # Random classifier baseline
    ax.axhline(
        y=base_rate, color="#999999", linestyle="--", lw=1.5,
        label=f"Random classifier (PR-AUC={base_rate:.3f})"
    )

    ax.set_xlabel("Recall (sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        logger.info("PR curve saved to %s", save_path)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a normalised confusion matrix.

    Normalisation is by true class (rows), so values show recall per class.
    Both raw counts and percentages are shown.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Normal (0)", "Failure/At-Risk (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", fontsize=10,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def generate_evaluation_report(
    y_true: np.ndarray,
    y_prob_lr: np.ndarray,
    y_prob_gbt: np.ndarray,
    threshold_lr: float,
    threshold_gbt: float,
) -> Dict:
    """
    Generate a comprehensive evaluation report for both models.

    Returns a dictionary with all metrics and figures.
    """
    report = {}

    # Metrics
    report["lr_metrics"]  = evaluate_model(y_true, y_prob_lr,  threshold_lr,  "LR")
    report["gbt_metrics"] = evaluate_model(y_true, y_prob_gbt, threshold_gbt, "GBT")

    # PR curve
    pr_fig = plot_pr_curve(
        y_true,
        {"Logistic Regression": y_prob_lr, "GBT": y_prob_gbt},
        title="Precision-Recall Curve — Test Set (F4 event)",
        save_path=str(FIG_DIR / "pr_curve.png"),
    )
    report["pr_curve_fig"] = pr_fig

    # Confusion matrices
    y_pred_lr  = (y_prob_lr  >= threshold_lr).astype("int8")
    y_pred_gbt = (y_prob_gbt >= threshold_gbt).astype("int8")

    cm_lr_fig  = plot_confusion_matrix(
        y_true, y_pred_lr,  "Logistic Regression — Test Set",
        save_path=str(FIG_DIR / "cm_lr.png"),
    )
    cm_gbt_fig = plot_confusion_matrix(
        y_true, y_pred_gbt, "GBT — Test Set",
        save_path=str(FIG_DIR / "cm_gbt.png"),
    )
    report["cm_lr_fig"]  = cm_lr_fig
    report["cm_gbt_fig"] = cm_gbt_fig

    # Select primary model (GBT typically better on tabular data)
    if report["gbt_metrics"]["pr_auc"] >= report["lr_metrics"]["pr_auc"]:
        report["primary_model"] = "gbt"
    else:
        report["primary_model"] = "lr"

    logger.info(
        "Evaluation report: LR PR-AUC=%.4f, GBT PR-AUC=%.4f, primary=%s",
        report["lr_metrics"]["pr_auc"],
        report["gbt_metrics"]["pr_auc"],
        report["primary_model"],
    )
    return report
