"""
scripts/run_stage2.py — End-to-end Stage 2 pipeline runner.

Runs: data quality → preprocessing → EDA → failure analysis
Saves all outputs to data/processed/ and data/processed/figures/

Usage:
    python scripts/run_stage2.py
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import PROCESSED_DIR, RAW_CSV
from src.data.loader import load_parquet, load_raw
from src.data.quality import run_quality_check
from src.data.preprocessor import run_preprocessing
from src.analysis.eda import run_eda
from src.analysis.failure_analysis import run_failure_analysis


def main():
    logger.info("=" * 55)
    logger.info("  RailGuard Stage 2 — Full Pipeline")
    logger.info("=" * 55)

    # ── Step 1: Data quality ─────────────────────────────────
    logger.info("STEP 1: Data Quality Check")
    logger.info("Loading raw CSV …")
    df_raw = load_raw(RAW_CSV)
    run_quality_check(df_raw, save_outputs=True)

    # ── Step 2: Preprocessing ────────────────────────────────
    logger.info("STEP 2: Preprocessing → 1-minute resolution")
    proc_path = PROCESSED_DIR / "processed_1min.parquet"
    if proc_path.exists():
        logger.info("Processed file exists, loading from cache.")
        df = load_parquet(proc_path)
    else:
        df = run_preprocessing(save_output=True)

    logger.info(
        "Processed: %d rows, gap rows=%d (%.1f%% coverage)",
        len(df),
        int(df["is_gap"].sum()),
        100 * (1 - df["is_gap"].mean()),
    )

    # ── Step 3: EDA ───────────────────────────────────────────
    logger.info("STEP 3: Exploratory Data Analysis")
    figures = run_eda(df)
    logger.info("EDA: %d figures saved", len(figures))

    # ── Step 4: Failure analysis ──────────────────────────────
    logger.info("STEP 4: Failure Event Analysis")
    stats = run_failure_analysis(df)
    logger.info("Failure analysis: %d window summaries", len(stats))

    # ── Final summary ─────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("  Stage 2 complete.")
    logger.info("  Outputs in: %s", PROCESSED_DIR)
    logger.info("  Figures in: %s/figures/", PROCESSED_DIR)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
