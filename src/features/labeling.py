"""
src/features/labeling.py — Supervised failure-risk labeling.

Label strategy
--------------
With only 4 labeled failure events, a naive binary classifier would have
severe class imbalance (~2%) and very few positive examples (4 event groups).
This module implements a careful labeling strategy:

1. FAILURE label (is_failure=True):
   Rows that fall within the documented failure windows (failure_id != '').
   These are the ground-truth positive examples.

2. PRE-FAILURE AT-RISK label (is_atrisk=True):
   The `pre_failure_window_hours` period immediately before each failure start.
   These rows are plausibly degraded / at elevated risk.
   They are kept separate from the failure rows in the binary label so that
   the model can optionally distinguish failure onset from active failure.

3. POST-MAINTENANCE EXCLUSION:
   The `post_maintenance_cooldown_hours` window after each maintenance event
   is excluded from normal training samples to avoid including still-recovering
   system states in the "normal" class.

4. NORMAL label (is_failure=False, is_atrisk=False, not excluded):
   All remaining non-gap, non-excluded rows.

Binary target for supervised learning:
   label = 1 if is_failure OR is_atrisk else 0

   Rationale: including pre-failure windows as positives gives the model a
   chance to learn the pre-failure degradation pattern, not just the acute
   failure signature. Without this, the model cannot be useful for early
   warning.

Class imbalance note:
   With 4 events × 24h window + 4 × 24h pre-window, the positive class
   will be roughly 5,000–10,000 rows out of 250,000 non-gap rows = ~2–4%.
   This is handled in the model via class_weight='balanced' and evaluation
   via precision-recall AUC, not accuracy.

Temporal split:
   Train:     Feb 2020 – Apr 2020 (contains F1 at the end)
   Validate:  May 2020 – Jun 2020 (contains F2, F3)
   Test:      Jul 2020 – Aug 2020 (contains F4)

   This matches the Stage 2 recommendation for strict chronological splits.

Usage:
    from src.features.labeling import build_labels, get_temporal_splits
    df_labeled = build_labels(df_processed)
    train_idx, val_idx, test_idx = get_temporal_splits(df_labeled)
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    FAILURE_EVENTS,
    POST_MAINT_COOLDOWN_H,
    PRE_FAILURE_WINDOW_H,
)

logger = logging.getLogger(__name__)

# Temporal split boundaries (chronological, no shuffling)
TRAIN_END   = "2020-04-30 23:59:00"   # includes F1
VAL_START   = "2020-05-01 00:00:00"
VAL_END     = "2020-06-30 23:59:00"   # includes F2, F3
TEST_START  = "2020-07-01 00:00:00"
TEST_END    = "2020-08-31 23:59:00"   # includes F4


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add labeling columns to the processed 1-minute DataFrame.

    New columns added:
    ------------------
    is_failure     : bool — row is within a documented failure window
    is_atrisk      : bool — row is within the pre-failure window (not failure)
    is_excluded    : bool — row is in post-maintenance cooldown (exclude from normal training)
    label          : int  — 1 if is_failure or is_atrisk, else 0
    label_strict   : int  — 1 if is_failure only (for evaluation comparison)

    Parameters
    ----------
    df : Processed 1-minute DataFrame (must contain failure_id and is_gap).

    Returns
    -------
    DataFrame with the above columns added.
    """
    df = df.copy()

    df["is_failure"] = df["failure_id"] != ""
    df["is_atrisk"]  = False
    df["is_excluded"] = False

    for event in FAILURE_EVENTS:
        failure_start = pd.Timestamp(event["start"])
        maintenance   = pd.Timestamp(event["maintenance"])

        # Pre-failure at-risk window: [failure_start - window, failure_start)
        atrisk_start = failure_start - pd.Timedelta(hours=PRE_FAILURE_WINDOW_H)
        atrisk_mask  = (
            (df.index >= atrisk_start) &
            (df.index < failure_start) &
            (~df["is_failure"])
        )
        df.loc[atrisk_mask, "is_atrisk"] = True

        n_atrisk = int(atrisk_mask.sum())
        logger.info(
            "Event %s: %d at-risk rows labelled (%s to %s)",
            event["id"], n_atrisk, atrisk_start, failure_start,
        )

        # Post-maintenance exclusion window
        cooldown_end = maintenance + pd.Timedelta(hours=POST_MAINT_COOLDOWN_H)
        excl_mask    = (
            (df.index > maintenance) &
            (df.index <= cooldown_end) &
            (~df["is_failure"]) &
            (~df["is_atrisk"])
        )
        df.loc[excl_mask, "is_excluded"] = True
        logger.info(
            "Event %s: %d post-maintenance rows excluded (%s to %s)",
            event["id"], int(excl_mask.sum()), maintenance, cooldown_end,
        )

    # Binary label: 1 = at risk or in failure; 0 = normal
    df["label"] = ((df["is_failure"] | df["is_atrisk"]) & ~df["is_gap"]).astype("int8")

    # Strict label: only actual failure windows
    df["label_strict"] = (df["is_failure"] & ~df["is_gap"]).astype("int8")

    n_pos    = int(df["label"].sum())
    n_total  = int((~df["is_gap"]).sum())
    n_excl   = int(df["is_excluded"].sum())
    logger.info(
        "Labels: %d positive (%.2f%%), %d normal, %d excluded (of %d non-gap rows)",
        n_pos, 100 * n_pos / n_total if n_total else 0,
        n_total - n_pos - n_excl, n_excl, n_total,
    )
    return df


def get_temporal_splits(
    df: pd.DataFrame,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Return (train_idx, val_idx, test_idx) as DatetimeIndex objects.

    Splits are strictly chronological:
      Train:    up to TRAIN_END     (includes F1)
      Validate: VAL_START–VAL_END   (includes F2, F3)
      Test:     TEST_START–TEST_END (includes F4)

    Only non-gap, non-excluded rows are included.
    """
    mask_base = ~df["is_gap"]
    if "is_excluded" in df.columns:
        mask_base = mask_base & ~df["is_excluded"]

    train_mask = mask_base & (df.index <= TRAIN_END)
    val_mask   = mask_base & (df.index >= VAL_START) & (df.index <= VAL_END)
    test_mask  = mask_base & (df.index >= TEST_START) & (df.index <= TEST_END)

    train_idx = df.index[train_mask]
    val_idx   = df.index[val_mask]
    test_idx  = df.index[test_mask]

    logger.info(
        "Temporal splits: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    for split_name, idx, mask in [
        ("train", train_idx, train_mask),
        ("val",   val_idx,   val_mask),
        ("test",  test_idx,  test_mask),
    ]:
        if "label" in df.columns:
            n_pos = int(df.loc[idx, "label"].sum())
            logger.info(
                "  %s: %d rows, %d positive (%.2f%%)",
                split_name, len(idx), n_pos,
                100 * n_pos / len(idx) if len(idx) else 0,
            )

    return train_idx, val_idx, test_idx


def get_anomaly_training_index(df: pd.DataFrame) -> pd.Index:
    """
    Return the index of confirmed-normal rows for anomaly-detection training.

    Uses Feb–Mar 2020 (before any failure events) as the clean baseline.
    Gap rows, failure rows, and at-risk rows are excluded.

    This is the conservative normal baseline recommended in Stage 2.
    """
    if "is_atrisk" not in df.columns:
        df = build_labels(df)

    baseline_end = pd.Timestamp("2020-03-31 23:59:00")

    mask = (
        (~df["is_gap"]) &
        (~df["is_failure"]) &
        (~df["is_atrisk"]) &
        (df.index <= baseline_end)
    )
    idx = df.index[mask]
    logger.info(
        "Anomaly training baseline: %d rows (%s → %s)",
        len(idx), idx.min(), idx.max(),
    )
    return idx
