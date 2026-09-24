"""
scripts/run_pipeline.py — Full RailGuard pipeline from raw CSV to dashboard-ready artifacts.

This is the single entry point to run the entire pipeline:
  1. Preprocessing (raw CSV → 1-minute Parquet)
  2. EDA and failure analysis figures (optional)
  3. Model training (feature engineering + anomaly + supervised + SHAP)

After running this script, launch the dashboard with:
    streamlit run src/dashboard/app.py

Usage:
    python scripts/run_pipeline.py                    # full pipeline
    python scripts/run_pipeline.py --skip-eda         # skip EDA figures
    python scripts/run_pipeline.py --skip-preprocessing  # use existing processed data
    python scripts/run_pipeline.py --no-shap          # skip SHAP (faster)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("railguard.pipeline")

PYTHON = sys.executable


def run_step(description: str, *args) -> None:
    """Run a Python command as a subprocess step."""
    logger.info("─" * 60)
    logger.info("STEP: %s", description)
    logger.info("─" * 60)
    t0 = time.time()
    result = subprocess.run([PYTHON] + list(args), cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("Step failed: %s (exit code %d)", description, result.returncode)
        sys.exit(result.returncode)
    logger.info("  Done in %.1f seconds", elapsed)


def main(
    skip_preprocessing: bool = False,
    skip_eda: bool = False,
    no_shap: bool = False,
) -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("  RailGuard — Full Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    if not skip_preprocessing:
        run_step("Preprocessing raw CSV → 1-minute Parquet",
                 "-m", "src.data.preprocessor")
    else:
        logger.info("SKIP: Preprocessing (using existing processed data)")

    # ── Step 2: EDA + Failure Analysis figures ────────────────────────────────
    if not skip_eda:
        run_step("EDA and failure analysis figures",
                 "scripts/run_stage2.py")
    else:
        logger.info("SKIP: EDA / failure analysis figures")

    # ── Step 3: Model training ─────────────────────────────────────────────────
    train_args = ["scripts/train_models.py", "--skip-preprocessing"]
    if no_shap:
        train_args.append("--no-shap")
    run_step("Feature engineering + anomaly detection + supervised models", *train_args)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Artifacts saved to: data/artifacts/")
    logger.info("  Figures saved to:   data/processed/figures/")
    logger.info("")
    logger.info("  To launch the dashboard:")
    logger.info("    streamlit run src/dashboard/app.py")
    logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailGuard full pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing (use existing processed_1min.parquet)")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA and failure analysis figure generation")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP computation (faster)")
    args = parser.parse_args()
    main(
        skip_preprocessing=args.skip_preprocessing,
        skip_eda=args.skip_eda,
        no_shap=args.no_shap,
    )
