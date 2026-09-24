"""
scripts/train_models.py — End-to-end RailGuard model training pipeline.

Runs all model training steps in sequence:
  1. Load processed data (or run preprocessing if needed)
  2. Build feature matrix (gap-aware rolling features)
  3. Build labels (failure windows + pre-failure at-risk)
  4. Train anomaly detection model (Isolation Forest on Feb–Mar baseline)
  5. Train predictive models (Logistic Regression + GBT)
  6. Compute SHAP explanations
  7. Compute baseline statistics for decision support
  8. Save all artifacts to data/artifacts/
  9. Print evaluation summary

All outputs are saved to data/artifacts/. Raw data in data/raw/ is never
modified.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --skip-preprocessing
    python scripts/train_models.py --no-shap
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("railguard.train")


def main(skip_preprocessing: bool = False, no_shap: bool = False) -> None:
    from src.config import ARTIFACTS_DIR, PROCESSED_DIR
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("  RailGuard — Model Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load processed data ──────────────────────────────────────────
    logger.info("[1/8] Loading processed data …")
    processed_path = PROCESSED_DIR / "processed_1min.parquet"

    if not processed_path.exists() or not skip_preprocessing:
        if not processed_path.exists():
            logger.info("  Processed data not found. Running preprocessing pipeline …")
            from src.data.preprocessor import run_preprocessing
            df = run_preprocessing(save_output=True)
        else:
            from src.data.loader import load_parquet
            df = load_parquet(processed_path)
    else:
        from src.data.loader import load_parquet
        df = load_parquet(processed_path)

    logger.info("  Loaded %d rows × %d columns", len(df), len(df.columns))

    # ── Step 2: Build feature matrix ─────────────────────────────────────────
    logger.info("[2/8] Building feature matrix (gap-aware rolling windows) …")
    from src.features.engineering import build_feature_matrix
    df_features = build_feature_matrix(df)
    logger.info("  Feature matrix: %d rows × %d columns", *df_features.shape)

    # ── Step 3: Build labels ──────────────────────────────────────────────────
    logger.info("[3/8] Building failure labels …")
    from src.features.labeling import (
        build_labels,
        get_anomaly_training_index,
        get_temporal_splits,
    )
    df_labeled = build_labels(df)
    train_idx, val_idx, test_idx = get_temporal_splits(df_labeled)
    normal_idx = get_anomaly_training_index(df_labeled)

    logger.info(
        "  Labels: %d positive (%.2f%%) in %d non-gap rows",
        int(df_labeled["label"].sum()),
        100 * df_labeled["label"].mean(),
        int((~df_labeled["is_gap"]).sum()),
    )

    # ── Step 4: Train anomaly detection model ─────────────────────────────────
    logger.info("[4/8] Training Isolation Forest anomaly model …")
    from src.models.anomaly import (
        get_anomaly_threshold_from_training,
        score_anomaly,
        train_anomaly_model,
    )
    anomaly_bundle = train_anomaly_model(df_features, normal_idx, save_artifacts=True)

    # Compute and store decision thresholds
    low_thresh  = get_anomaly_threshold_from_training(anomaly_bundle, specificity_percentile=95.0)
    high_thresh = get_anomaly_threshold_from_training(anomaly_bundle, specificity_percentile=99.0)
    anomaly_bundle["score_stats"]["alert_threshold_low"]  = low_thresh
    anomaly_bundle["score_stats"]["alert_threshold_high"] = high_thresh

    # Update saved stats with thresholds
    score_stats_path = ARTIFACTS_DIR / "anomaly_score_stats.json"
    with open(score_stats_path, "w") as f:
        json.dump(anomaly_bundle["score_stats"], f, indent=2)
    logger.info("  Anomaly thresholds: low=%.4f, high=%.4f", low_thresh, high_thresh)

    # Score the full dataset and save
    logger.info("  Scoring full dataset with anomaly model …")
    anomaly_scores = score_anomaly(anomaly_bundle, df_features)
    anomaly_scores.to_frame("anomaly_score").to_parquet(
        str(ARTIFACTS_DIR / "anomaly_scores.parquet")
    )
    logger.info("  Anomaly scores saved")

    # ── Step 5: Baseline stats for decision support ────────────────────────────
    logger.info("[5/8] Computing baseline statistics …")
    from src.models.decision_support import compute_baseline_stats
    from src.models.anomaly import select_anomaly_features
    feat_cols = select_anomaly_features(df_features)
    baseline_stats = compute_baseline_stats(df_features, normal_idx, feat_cols)
    logger.info("  Baseline stats computed for %d features", len(baseline_stats))

    # ── Step 6: Train predictive models ────────────────────────────────────────
    logger.info("[6/8] Training supervised predictive models …")
    from src.models.predictive import train_predictive_models

    predictive_bundle = train_predictive_models(
        df_features=df_features,
        df_labels=df_labeled,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        save_artifacts=True,
    )
    results = predictive_bundle["results"]
    logger.info(
        "  LR   → val PR-AUC=%.4f, test PR-AUC=%.4f",
        results["lr"]["val_metrics"]["pr_auc"],
        results["lr"]["test_metrics"]["pr_auc"],
    )
    logger.info(
        "  GBT  → val PR-AUC=%.4f, test PR-AUC=%.4f",
        results["gbt"]["val_metrics"]["pr_auc"],
        results["gbt"]["test_metrics"]["pr_auc"],
    )

    # Score full dataset with GBT (primary supervised model)
    from src.models.predictive import predict_risk
    risk_scores = predict_risk(predictive_bundle, df_features, model="gbt")
    risk_scores.to_frame("risk_prob").to_parquet(
        str(ARTIFACTS_DIR / "risk_scores.parquet")
    )
    logger.info("  Risk scores saved")

    # ── Step 7: SHAP explanations ─────────────────────────────────────────────
    if not no_shap:
        logger.info("[7/8] Computing SHAP feature importances …")
        try:
            from src.explainability.shap_analysis import compute_shap_values, plot_feature_importance
            from src.features.engineering import get_feature_columns

            # Use the test set for SHAP (most representative for held-out behaviour)
            common_test = test_idx.intersection(df_features.index)
            X_test = df_features.loc[common_test, feat_cols].fillna(0.0)

            shap_bundle = compute_shap_values(
                model=predictive_bundle["gbt_model"],
                X=X_test,
                feat_cols=feat_cols,
                model_type="gbt",
                save_artifacts=True,
            )
            # Save feature importance plot
            fi_fig = plot_feature_importance(
                shap_bundle,
                top_n=20,
                title="RailGuard — Feature Importance (SHAP, GBT, Test Set)",
                save_path=str(PROCESSED_DIR / "figures" / "feature_importance_shap.png"),
            )
            logger.info("  SHAP analysis complete; top feature: %s",
                        shap_bundle["importance"][0]["feature"] if shap_bundle["importance"] else "N/A")
        except Exception as exc:
            logger.warning("SHAP computation failed: %s (continuing without SHAP)", exc)
    else:
        logger.info("[7/8] Skipping SHAP (--no-shap flag)")

    # ── Step 8: Evaluation report ─────────────────────────────────────────────
    logger.info("[8/8] Generating evaluation report …")
    from src.models.evaluation import generate_evaluation_report

    eval_report = generate_evaluation_report(
        y_true=predictive_bundle["y_test"],
        y_prob_lr=predictive_bundle["lr_test_prob"],
        y_prob_gbt=predictive_bundle["gbt_test_prob"],
        threshold_lr=predictive_bundle["lr_threshold"],
        threshold_gbt=predictive_bundle["gbt_threshold"],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Training complete in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("")
    logger.info("  EVALUATION RESULTS (test set — F4 event, Jul–Aug 2020)")
    logger.info("  ──────────────────────────────────────────────────────")
    for model_name in ("lr", "gbt"):
        m = results[model_name]["test_metrics"]
        logger.info(
            "  %-5s  PR-AUC=%.4f  Precision=%.3f  Recall=%.3f  F1=%.3f",
            model_name.upper(), m["pr_auc"], m["precision"], m["recall"], m["f1"],
        )
    logger.info("")
    logger.info("  Base rate (random classifier PR-AUC): %.4f",
                results["lr"]["test_metrics"]["base_rate"])
    logger.info("")
    logger.info(
        "  NOTE: Only 4 labeled failure events available. Supervised model "
        "results are indicative only. Anomaly detection is the primary "
        "production component."
    )
    logger.info("")
    logger.info("  Artifacts saved to: %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailGuard model training pipeline")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step (use existing data/processed/processed_1min.parquet)",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP computation (faster, but no explainability artifacts)",
    )
    args = parser.parse_args()
    main(skip_preprocessing=args.skip_preprocessing, no_shap=args.no_shap)
