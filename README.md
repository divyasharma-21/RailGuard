# RailGuard — Predictive Maintenance Analytics System

> A technically credible, end-to-end predictive maintenance system for a metro rail air compressor, built on the publicly available **MetroPT-3** dataset from Porto Metro (2020).

---

## Overview

RailGuard demonstrates how sensor data from an industrial air compressor can be used to:
- **Detect anomalous operating patterns** using unsupervised machine learning
- **Assess failure risk** using a supervised model trained on documented failure events
- **Explain model predictions** using SHAP feature attribution
- **Support maintenance decisions** with evidence-based, calibrated alerts
- **Visualise everything** through a professional Streamlit dashboard

The system is honest about its limitations: with only 4 documented failure events, supervised learning cannot be fully validated, and the anomaly detection model is the primary predictive component.

---

## Real-World Problem Statement

Metro rail compressors are safety-critical components. Unexpected air pressure loss can disrupt pneumatic braking, door systems, and pantograph controls. Reactive maintenance (fix after failure) causes service disruption and higher repair costs. Predictive maintenance — flagging degradation before failure — reduces both.

This system addresses:

- **When should we schedule an inspection?** The anomaly model detects sensor patterns that differ from known-normal operation, giving early warning of potential problems.
- **How confident are we?** All scores are calibrated against the training baseline, and thresholds are data-derived rather than manually set.
- **What sensor is causing the concern?** SHAP feature attribution identifies which signals are driving elevated scores.
- **How do we act on this?** The decision support layer translates scores into Normal / Monitor / Investigate categories with plain-English evidence summaries.

---

## Dataset

| Property | Value |
|----------|-------|
| Name | MetroPT-3 — Metro do Porto Air Compressor Dataset |
| Source | UCI Machine Learning Repository |
| Period | 2020-02-01 → 2020-09-01 (213 days) |
| Raw rows | 1,516,948 at ~10-second effective resolution |
| Processed rows | 306,960 at 1-minute resolution |
| Sensors | 7 analogue + 8 digital (see below) |
| Documented failures | 4 air-leak events with maintenance records |
| Missing time | 909.5 hours (17.7%) — maintenance windows, shutdowns |

### Analogue Sensors

| Sensor | Unit | Description |
|--------|------|-------------|
| `TP2` | bar | Compressor pressure |
| `TP3` | bar | Pneumatic panel pressure |
| `H1` | bar | Cyclonic separator filter discharge pressure |
| `DV_pressure` | bar | Tower air-dryer discharge pressure drop |
| `Reservoirs` | bar | Downstream reservoir pressure |
| `Oil_temperature` | °C | Compressor oil temperature |
| `Motor_current` | A | Motor current (one phase of 3-phase) |

### Digital Signals

| Signal | Description |
|--------|-------------|
| `COMP` | Air intake valve (active = off/offloaded) |
| `DV_eletric` | Outlet valve (active = under load) |
| `Towers` | Tower selector (0=tower1, 1=tower2) |
| `MPG` | Load-start signal (triggers below 8.2 bar) |
| `LPS` | Low pressure switch (activates below 7 bar) |
| `Pressure_switch` | Tower discharge detection |
| `Oil_level` | Oil level alarm |
| `Caudal_impulses` | Air flow pulse counter |

### Documented Failure Events

| ID | Start | End | Type | Severity | Maintenance |
|----|-------|-----|------|----------|-------------|
| F1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air Leak | High stress | 2020-04-30 12:00 |
| F2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air Leak | High stress | 2020-04-30 12:00¹ |
| F3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air Leak | High stress | 2020-06-08 16:00 |
| F4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air Leak | High stress | 2020-07-16 00:00 |

¹ Same maintenance date as F1 in company records — likely a data entry anomaly.

---

## System Architecture

```
data/raw/MetroPT3(AirCompressor).csv
          │
          ▼ src/data/preprocessor.py
data/processed/processed_1min.parquet     (306,960 rows × 32 cols)
          │
          ▼ src/features/engineering.py
Feature Matrix                            (306,960 × 86 features, gap-aware rolling)
          │
          ├──► src/models/anomaly.py       Isolation Forest (normal baseline: Feb–Mar 2020)
          │         │ → data/artifacts/anomaly_model.pkl
          │         │ → data/artifacts/anomaly_scores.parquet
          │
          ├──► src/features/labeling.py    Binary labels + temporal splits
          │
          ├──► src/models/predictive.py    Logistic Regression + GBT
          │         │ → data/artifacts/predictive_lr_model.pkl
          │         │ → data/artifacts/predictive_gbt_model.pkl
          │         │ → data/artifacts/risk_scores.parquet
          │
          ├──► src/explainability/shap_analysis.py   SHAP (TreeExplainer)
          │         │ → data/artifacts/shap_gbt_values.npy
          │         │ → data/artifacts/shap_feature_importance.json
          │
          └──► src/models/decision_support.py        Normal/Monitor/Investigate
                    │
                    ▼
          src/dashboard/app.py            Streamlit web application
```

---

## Data Pipeline

### Stage 1 — Inspection
- Raw CSV structural validation: 1,516,948 rows, 15 sensor columns, 0 nulls
- Timestamp analysis: effective 10-second sampling (stored every 10th sample)
- 331 gaps > 50 seconds detected
- Failure event signatures measured from raw data

### Stage 2 — Preprocessing & EDA
- Resample to 1-minute resolution (mean for analogue, max for digital)
- Gap detection: minutes with n_samples=0 flagged as `is_gap=True`
- `gap_segment_id` groups consecutive non-gap rows into continuity segments
- Operational features added: `motor_state`, `is_loaded`, `load_fraction`, `pressure_diff`
- Failure windows annotated from config.yaml

### Stage 3 — Modelling (this stage)
- Gap-aware rolling features (30 min, 2 h, 6 h windows)
- Isolation Forest anomaly model trained on Feb–Mar 2020 baseline
- Supervised models trained with strict chronological splits
- SHAP explanations computed on test set
- Full scoring pipeline saves artifacts to `data/artifacts/`

---

## EDA Key Findings

1. **H1 is the strongest failure indicator**: collapses from ~8 bar to ~0 bar during all 4 failure events (97%+ reduction).
2. **DV_eletric reaches 97–100% during failure**: compressor locked under continuous load vs 9–31% in normal operation.
3. **Oil_temperature rises 8–19°C** above pre-failure baseline during failures.
4. **LPS activation correlates with severity**: minimal in F1/F2, elevated in F3, 33% active during F4.
5. **Progressive degradation is visible before F3 and F4**: DV_eletric already elevated at 17–31% in the 72-hour pre-failure window.
6. **TP3 ↔ Reservoirs: very high correlation (≈0.99)** in normal operation — both measure adjacent pressure points.

---

## Feature Engineering

### Rolling Windows (gap-aware)
All rolling statistics are computed **within each `gap_segment_id`** — never across gap boundaries. This prevents mixing readings from different operational sessions.

| Window | Size | Purpose |
|--------|------|---------|
| 30 min | 30 rows | Rapid within-cycle changes |
| 2 hours | 120 rows | Drift across compressor cycles |
| 6 hours | 360 rows | Slow thermal accumulation / pre-failure degradation |

### Statistics per window per feature
- **Mean** — signal level trend
- **Std** — variability (compressor cycling pattern changes before failure)
- **Min, Max** — extreme values within window

### Feature set (86 total)
- Primary discriminators: `H1`, `load_fraction`, `Motor_current`, `Oil_temperature`, `TP2`, `DV_pressure`
- Supporting: `LPS` rolling activation rate, `pressure_diff`, `reservoir_panel_diff`
- Trend features: 30-minute linear slope for `H1` and `load_fraction`

---

## Anomaly Detection

**Method:** `sklearn.ensemble.IsolationForest`

**Training baseline:** Feb–Mar 2020 (73,638 confirmed-normal rows — before any documented failure)

**Score calibration:** Sigmoid transformation of the `decision_function` output, normalised using training statistics. Higher = more anomalous. The training distribution maps cleanly to [0, 1].

**Decision thresholds (data-derived):**

| Category | Threshold | Percentile | False positive rate |
|----------|-----------|------------|---------------------|
| Monitor | ≥ 0.835 | 95th | ~5% of normal rows |
| Investigate | ≥ 0.988 | 99th | ~1% of normal rows |

**Observed performance on known failures:**
- All 4 failure events score > 0.98 (max possible range) during documented failure windows
- Pre-failure degradation visible in F3 and F4: scores begin rising 2–6 hours before onset
- Normal operation (Feb–Mar baseline): median score 0.49

---

## Predictive Modelling

### Label Strategy
- **Positive class (label=1):** failure windows + 24-hour pre-failure at-risk windows
- **Negative class (label=0):** all other non-gap, non-excluded rows
- **Excluded:** 24-hour post-maintenance cooldown windows

### Temporal Split (strictly chronological — no shuffling)
| Split | Period | Events | Rows | Positive rate |
|-------|--------|--------|------|---------------|
| Train | Feb–Apr 2020 | F1 | 106,824 | 2.47% |
| Validate | May–Jun 2020 | F2, F3 | 70,840 | 8.37% |
| Test | Jul–Aug 2020 | F4 | 72,015 | 1.79% |

### Models

**Logistic Regression (interpretable baseline)**
- `class_weight='balanced'`, L2 regularisation (C=0.1)
- Test PR-AUC: 0.0091 ← worse than random (base rate: 0.018)

**HistGradientBoostingClassifier**
- Sample-weighted for class imbalance, early stopping on validation
- Test PR-AUC: 0.0816 ← above base rate; learns some pattern but limited by 4 events

### Evaluation Metrics

Primary metric: **Precision-Recall AUC** (informative under 2% class imbalance)

| Model | Split | PR-AUC | Precision | Recall | F1 |
|-------|-------|--------|-----------|--------|----|
| LR | Validation | 0.1946 | 0.340 | 0.498 | 0.404 |
| LR | **Test** | 0.0091 | 0.000 | 0.000 | 0.000 |
| GBT | Validation | 0.1927 | 0.229 | 0.251 | 0.239 |
| GBT | **Test** | 0.0816 | 0.592 | 0.035 | 0.066 |

**Interpretation:** The GBT model detects *some* F4 failures with high precision but very low recall — it flags 45 true positives out of 1,292 (3.5% recall). The LR model completely fails on the test set. This is expected with only one test event and confirms that the supervised component is indicative only.

---

## Explainability

**Method:** SHAP TreeExplainer on the GBT model (test set, 5,000 rows subsampled)

### Top features by mean |SHAP| value

| Rank | Feature | Mean |SHAP| | Physical meaning |
|------|---------|------------|------------------|
| 1 | `H1_30m_max` | 0.378 | Peak cyclonic separator pressure in last 30 min — strongest failure signal |
| 2 | `TP2_30m_max` | 0.091 | Peak compressor pressure in last 30 min |
| 3 | `Oil_temperature_6h_min` | 0.085 | Minimum oil temperature over 6 hours |
| 4 | `Oil_temperature_30m_min` | 0.081 | Minimum oil temperature over 30 min |
| 5 | `TP2_30m_min` | 0.054 | Minimum compressor pressure in 30 min |
| 6–10 | Various rolling Oil_temp, Motor_current, H1_trend features | — | Supporting sensors |

**Caveat:** SHAP values show which features the model uses — they do **not** prove causation. The physical causal chain (air leak → continuous load → elevated temperature, pressure, and H1 collapse) is documented in the dataset description and is consistent with the SHAP findings, but the model does not prove this chain.

---

## Dashboard Features

Launch with: `streamlit run src/dashboard/app.py`

| Section | Description |
|---------|-------------|
| **A — Executive Overview** | Current health status, KPI cards, anomaly timeline, failure event table |
| **B — Sensor Analytics** | Interactive sensor selection, time-range filtering, distributions, correlations |
| **C — Anomaly Monitoring** | Anomaly score timeline, event detection, score distributions |
| **D — Failure Analysis** | Per-event sensor panels, pre/during/post statistics, LPS analysis |
| **E — Predictive Risk** | Model metrics, PR curve, confusion matrices, risk score timeline |
| **F — Maintenance Insights** | Decision categories, evidence table, recent sensor trends, feature importance |
| **G — About & Methodology** | Dataset info, architecture, limitations, dataset attribution |

---

## Project Structure

```
RailGuard/
├── config.yaml                    # Central configuration (sensors, failures, paths)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── MetroPT3(AirCompressor).csv   ← NEVER MODIFIED
│   ├── processed/
│   │   ├── processed_1min.parquet         ← 1-minute aggregated data
│   │   ├── eda_sensor_stats.csv
│   │   ├── failure_event_summary.csv
│   │   └── figures/                       ← EDA and analysis figures
│   └── artifacts/                         ← Trained model artifacts
│
├── src/
│   ├── config.py                  # Config loader
│   ├── data/
│   │   ├── loader.py              # Raw CSV and Parquet I/O
│   │   ├── preprocessor.py        # 1-minute aggregation pipeline
│   │   └── quality.py             # Data quality checks
│   ├── features/
│   │   ├── engineering.py         # Gap-aware rolling feature computation
│   │   └── labeling.py            # Failure labels and temporal splits
│   ├── models/
│   │   ├── anomaly.py             # Isolation Forest anomaly detection
│   │   ├── predictive.py          # LR + GBT supervised models
│   │   ├── evaluation.py          # PR-AUC, confusion matrix, PR curve
│   │   └── decision_support.py    # Normal/Monitor/Investigate categories
│   ├── explainability/
│   │   └── shap_analysis.py       # SHAP TreeExplainer
│   ├── analysis/
│   │   ├── eda.py                 # Exploratory data analysis
│   │   └── failure_analysis.py    # Per-failure event analysis
│   └── dashboard/
│       └── app.py                 # Streamlit dashboard (7 sections)
│
├── scripts/
│   ├── run_pipeline.py            # Full end-to-end pipeline
│   ├── train_models.py            # Model training only
│   └── run_stage2.py              # EDA + failure analysis figures
│
├── tests/
│   ├── test_preprocessor.py       # Preprocessing unit tests
│   ├── test_quality.py            # Data quality unit tests
│   ├── test_features.py           # Feature engineering + labeling tests
│   └── test_models.py             # Model loading + evaluation tests
│
└── docs/
    ├── stage1_findings.md
    ├── stage2_findings.md
    └── stage3_findings.md
```

---

## Installation

```bash
# Clone / download the repository
cd RailGuard

# Install dependencies
pip install -r requirements.txt
```

Python 3.10+ required.

---

## How to Run

### Option 1 — Full pipeline (preprocessing → models → artifacts)

```bash
python scripts/run_pipeline.py
```

This runs:
1. Preprocessing (raw CSV → 1-minute Parquet)
2. EDA and failure analysis figures
3. Feature engineering + anomaly model + supervised models + SHAP

Flags:
```bash
python scripts/run_pipeline.py --skip-preprocessing   # skip if processed data exists
python scripts/run_pipeline.py --skip-eda             # skip figure generation
python scripts/run_pipeline.py --no-shap              # skip SHAP (faster)
```

### Option 2 — Model training only (assumes preprocessing already done)

```bash
python scripts/train_models.py --skip-preprocessing
```

### Option 3 — Launch dashboard only (assumes artifacts exist)

```bash
streamlit run src/dashboard/app.py
```

### Option 4 — Run tests

```bash
python -m pytest tests/ -v
```

---

## Key Findings

1. **Anomaly detection is highly effective**: The Isolation Forest trained on Feb–Mar 2020 normal data scores all 4 failure events above 0.98 (vs. median 0.49 for normal rows). The separation is dramatic.

2. **H1 is the primary failure indicator**: The cyclonic separator pressure collapses >97% during all failure events. This is confirmed by both raw sensor analysis and SHAP feature attribution.

3. **Progressive degradation is detectable**: In F3 and F4, elevated load fractions and anomaly scores appear 2–6 hours before the documented failure start. This is the early-warning window that makes anomaly detection practically useful.

4. **Supervised learning is insufficient with 4 events**: The GBT model achieves test PR-AUC of 0.082 vs. random baseline of 0.018. Statistically meaningful, but with recall of 3.5% on the test event (F4), it cannot be used operationally without more failure data.

5. **Air leaks have a clear, consistent multi-sensor signature**: H1 collapse + continuous motor load (DV_eletric ~100%) + oil temperature elevation (+8–19°C) + rising LPS activity form a consistent pattern across all 4 events.

---

## Limitations

| Limitation | Impact | Mitigation in this project |
|-----------|--------|---------------------------|
| Only 4 labeled failure events | Supervised model cannot be statistically validated | Anomaly detection used as primary component; supervised model clearly labelled as indicative |
| All failures are air leaks | Models are not trained for other failure types | Clearly stated in dashboard; anomaly detection would still flag novel failure patterns |
| 17.7% missing data | Some failure precursors may be in gaps | Gaps flagged; no interpolation across gaps |
| Feb–Mar baseline may not be entirely normal | If early degradation began before April, baseline is slightly contaminated | Conservative approach: use only confirmed pre-failure period |
| Single metro system | Results may not generalise to different compressors | Documented as a single-system study |
| Correlation ≠ causation | SHAP importances show associations, not causal mechanisms | Explicitly noted in all explainability outputs and dashboard |

---

## Future Improvements

- Collect more failure events (across multiple compressors or over longer periods) to validate the supervised model
- Add autoencoder-based anomaly detection for comparison
- Implement online/streaming scoring for real-time monitoring
- Add maintenance log integration to improve post-maintenance cooldown detection
- Investigate residual life estimation (RUL) using F3 and F4 pre-failure degradation windows

---

## Dataset Attribution

**Veloso, B., Ribeiro, J., Pereira, P., & Gama, J. (2022).**  
*MetroPT: A Benchmark Dataset for Predictive Maintenance.*  
Applied Sciences, 12(12), 5897. https://doi.org/10.3390/app12125897

Dataset: UCI Machine Learning Repository — MetroPT-3 Dataset  
License: Creative Commons Attribution 4.0 (CC BY 4.0)

The raw data in `data/raw/MetroPT3(AirCompressor).csv` and `data/raw/Data Description_Metro.pdf` are the property of their respective owners and are used here for research and educational purposes under the CC BY 4.0 license with attribution.
