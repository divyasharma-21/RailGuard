"""
src/config.py — Central configuration loader for RailGuard.
Reads config.yaml and exposes typed constants to the rest of the codebase.
"""
from __future__ import annotations

import yaml
from pathlib import Path

# Project root = two levels up from this file (src/config.py → src/ → root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config() -> dict:
    """Load and return the full config.yaml as a dict."""
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


# ── Load once at import time ──────────────────────────────────────────────────
_cfg = load_config()

# Paths
RAW_CSV       = PROJECT_ROOT / _cfg["paths"]["raw_csv"]
RAW_PDF       = PROJECT_ROOT / _cfg["paths"]["raw_pdf"]
PROCESSED_DIR = PROJECT_ROOT / _cfg["paths"]["processed_dir"]
ARTIFACTS_DIR = PROJECT_ROOT / _cfg["paths"]["artifacts_dir"]

# Sensor lists
ANALOGUE_SENSORS: list[str] = _cfg["sensors"]["analogue"]
DIGITAL_SENSORS:  list[str] = _cfg["sensors"]["digital"]
ALL_SENSORS:      list[str] = ANALOGUE_SENSORS + DIGITAL_SENSORS

# Failure events as a list of dicts
FAILURE_EVENTS: list[dict] = _cfg["failure_events"]

# Sampling
EFFECTIVE_SAMPLING_S: int = _cfg["dataset"]["effective_sampling_s"]
RESAMPLE_FREQ:        str = _cfg["anomaly"]["resample_freq"]

# Labeling
PRE_FAILURE_WINDOW_H:       int = _cfg["labeling"]["pre_failure_window_hours"]
POST_MAINT_COOLDOWN_H:      int = _cfg["labeling"]["post_maintenance_cooldown_hours"]
RUL_MAX_H:                  int = _cfg["labeling"]["rul_max_hours"]

# Documented thresholds
MPG_TRIGGER_BAR:  float = _cfg["pressure_thresholds"]["mpg_trigger_bar"]
LPS_TRIGGER_BAR:  float = _cfg["pressure_thresholds"]["lps_trigger_bar"]
