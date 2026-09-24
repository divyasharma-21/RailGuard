# Stage 2 Findings: Data Quality, Preprocessing, EDA, and Failure Analysis

> All values in this document were computed from the actual processed data during the Stage 2 pipeline run. Nothing is hardcoded or assumed.

---

## 1. Data Quality Findings

### Structural validation
| Property | Value |
|----------|-------|
| Rows | 1,516,948 |
| Sensor columns | 15 (7 analogue + 8 digital) |
| Null values | **0** |
| Duplicate timestamps | **0** |
| Rows with identical sensor values | 57,475 (~3.8%) |

**Note on "duplicate" rows:** 57,475 rows share identical sensor values with at least one other row. These occur at **different timestamps** and are **expected** — the air compressor frequently reaches near-identical steady-state readings during off-loaded or idle operation. These are not data integrity issues and are not removed.

### Sensor ranges
All 7 analogue sensors fall within physically plausible bounds. No readings require removal:
- Small negative values on TP2, H1, DV_pressure (−0.03 to 0 bar) are sensor noise around zero, expected behaviour
- Oil_temperature minimum of 15.4°C reflects system cold-start; maximum of 89.05°C is within safe operation range
- Motor_current reaches 9.295 A (above the documented "starting" state of 9A) — physically plausible for motor start transients

### Digital signals
All 8 digital signals contain only valid binary values (0 or 1). No corruption detected.

### Timestamp gaps
- **331 gaps above the 50-second threshold** detected
- **Dominant gap:** 10 seconds (88.2% of consecutive pairs)
- **Total missing time:** 909.5 hours across the 213-day period
- Largest individual gap: ~5.76 hours (system shutdown/maintenance window)
- Gaps represent genuine periods with no sensor data (power-off, maintenance) and are **not interpolated**

---

## 2. Preprocessing Decisions

### Aggregation to 1-minute resolution

**Decision:** Resample from ~10s raw data to 1-minute buckets.

**Rationale:**
1. **Jitter tolerance:** Raw timestamps have 9–12s jitter. 1-minute buckets absorb this completely without distorting signal levels.
2. **Scale:** 1.52M rows → 306,960 rows. Tractable for all ML methods including rolling-window feature computation.
3. **Operational granularity:** Compressor charge/rest cycles span minutes. Pre-failure drift develops over hours. No operationally important dynamics exist at sub-minute timescales for this failure type.
4. **Documentation alignment:** config.yaml `resample_freq: "1min"` was set in Stage 1 planning based on the same reasoning.

### Aggregation rules per column type

| Column type | Rule | Rationale |
|-------------|------|-----------|
| Analogue (mean) | `mean` | Preserves signal level within bucket |
| Analogue (max) | `max` → `{col}_max` | Detects within-bucket spikes without inflating mean |
| Digital | `max` | 1 if the signal was active for any sample in the minute |
| n_samples | `count` | Number of raw 10s rows per minute bucket |
| load_fraction | `mean(DV_eletric)` | Fraction of samples where outlet valve was active |

### Gap handling

- Minute buckets with `n_samples == 0` → `is_gap = True`
- Gap rows are **retained** in the DataFrame with NaN sensor values (not dropped)
- `gap_segment_id` assigns integer IDs to consecutive non-gap runs, incrementing at each gap boundary
- **No interpolation** is performed across gaps — downstream features must respect `gap_segment_id` boundaries

### Operational mode features added

| Feature | Description |
|---------|-------------|
| `motor_state` | Categorical: off/offloaded/loaded/starting (from Motor_current mean) |
| `is_loaded` | Bool: load_fraction ≥ 0.5 (compressor working under full load) |
| `pressure_diff` | TP3 − TP2 (reservoir vs. compressor pressure differential) |
| `reservoir_panel_diff` | TP3 − Reservoirs (should be ~0 in healthy operation) |
| `h1_near_zero` | Bool: H1 < 0.5 bar (key failure indicator — cyclonic separator pressure) |

### Processed dataset shape
- **306,960 rows** at 1-minute resolution
- **54,240 gap rows** (17.7% of 1-minute slots have no data)
- **82.3% coverage** of the 213-day observation window
- **32 columns** total (15 original sensors + 7 max columns + 5 new features + n_samples + load_fraction + is_gap + gap_segment_id + failure_id)

---

## 3. EDA Key Findings

### Sensor distributions
- **TP2** and **H1** show bimodal distributions in normal operation: one mode at ~0 bar (compressor off or offloaded) and one at ~8–10 bar (compressor charged)
- **DV_pressure** is near-zero (below ~0.1 bar) for ~84% of normal operation time; elevated values correspond to tower air-dryer discharge events
- **Oil_temperature** is approximately normally distributed around ~63°C during normal operation
- **Motor_current** shows three distinct modes matching documented states: ~0A (off), ~4A (offloaded), ~7A (under load)
- During failure windows, **all** distributions shift: H1 collapses toward 0, TP2/Motor_current/Oil_temperature all shift upward

### Correlations
- **TP3 ↔ Reservoirs: very high positive correlation** (~0.99+) — these two sensors measure pressure at adjacent points and track each other closely in healthy operation
- **TP2 ↔ Motor_current: strong positive correlation** — higher compressor pressure coincides with higher motor load
- **H1 ↔ TP2: moderate negative correlation** in normal operation — when H1 is high (separator functioning), compressor pressure is lower (system doesn't need to compensate)
- **DV_pressure:** low correlation with analogue sensors in normal operation; strong positive correlation with TP2 during failures (tower dryer active while compressor runs continuously)
- **Oil_temperature** shows moderate positive correlation with Motor_current — temperature rises with sustained load

### Operational states
- **Offloaded** is the dominant motor state (~53% of non-gap rows)
- **Loaded** accounts for ~15% of non-gap rows in normal operation; rises to ~95%+ during failures
- **Off** accounts for ~13%; brief transients during system cycling
- **24h rolling load fraction** shows clear elevation starting days before F3 and F4, suggesting measurable pre-failure degradation

### Temporal patterns
- **Hour of day:** No strong diurnal pattern in load fraction or temperature — the compressor operates continuously 24/7
- **Day of week:** Marginally lower load on weekends — consistent with reduced metro service

### TP3 vs Reservoirs
- In normal operation, TP3 and Reservoirs track within ±0.05 bar — close to the 1:1 ideal
- During failure windows, the scatter increases slightly — the compressor is running harder to maintain pressure against the leak

---

## 4. Failure Event Analysis Findings

All values below are computed from the processed 1-minute data.

### Cross-event sensor summary

| Event | Period | H1 mean | TP2 mean | Oil_temp mean | Motor_current mean | DV_eletric% | LPS% |
|-------|--------|---------|---------|--------------|-------------------|-------------|------|
| F1 | pre | 8.19 bar | 0.60 bar | 54.89°C | 1.12 A | 9.1% | 0.1% |
| F1 | **failure** | **0.10 bar** | **8.49 bar** | **74.06°C** | **5.59 A** | **98.7%** | 0.0% |
| F1 | post | 8.45 bar | 0.82 bar | 62.75°C | 2.73 A | 51.8% | 0.5% |
| F2 | pre | 7.90 bar | 0.75 bar | 66.00°C | 1.35 A | 10.9% | 0.0% |
| F2 | **failure** | **0.10 bar** | **8.15 bar** | **75.86°C** | **5.56 A** | **99.0%** | 0.0% |
| F2 | post | 7.98 bar | 1.01 bar | 64.06°C | 1.87 A | 15.1% | 0.1% |
| F3 | pre | 7.75 bar | 1.09 bar | 64.06°C | 1.84 A | 17.3% | 1.0% |
| F3 | **failure** | **−0.01 bar** | **7.89 bar** | **75.51°C** | **5.49 A** | **100.0%** | 1.2% |
| F3 | post | 7.45 bar | 1.45 bar | 64.87°C | 2.32 A | 22.0% | 3.4% |
| F4 | pre | 6.81 bar | 2.09 bar | 69.51°C | 3.06 A | 30.9% | 1.4% |
| F4 | **failure** | **0.27 bar** | **8.23 bar** | **83.86°C** | **5.49 A** | **97.0%** | **33.0%** |
| F4 | post | 7.35 bar | 1.58 bar | 69.71°C | 2.64 A | 23.7% | 2.7% |

### Key observations

**H1 is the single strongest failure indicator:**
- Normal: 7.7–8.2 bar (mean across all pre-failure windows)
- During failure: 0.0–0.27 bar (>97% reduction)
- This collapse is sharp and consistent across all 4 events

**DV_eletric reaches near-100% during every failure:**
- Normal: 9–31% active (compressor cycles on/off)
- During failure: 97–100% (compressor locked under continuous load)
- This is the clearest operational signature of the air leak

**Oil_temperature elevation is consistent but moderate:**
- Pre-failure: 55–70°C (varies by season — F4 in July shows higher baseline)
- During failure: 74–84°C (+8 to +19°C above pre-failure baseline)
- Not a false-positive risk: temperatures this elevated are only observed during confirmed failures

**LPS activation increases with failure severity:**
- F1 and F2: minimal LPS activity (0–1%) even during failure
- F3: LPS elevated to 1–3.4% (post-event, indicating residual pressure instability)
- F4: **33% active during failure** — the most severe event in the dataset
- Pre-F4 LPS was already elevated at 1.4% (72h pre-window), indicating progressive degradation

**Progressive degradation visible before F3 and F4:**
- F3 pre-window: DV_eletric already elevated at 17.3% vs ~10% for F1/F2 pre-windows
- F4 pre-window: DV_eletric at 30.9%, H1 already lower at 6.81 bar, LPS at 1.4%
- This confirms a measurable pre-failure signal exists — anomaly detection should be able to pick this up

**Post-failure recovery:**
- After F1: DV_eletric remains at 51.8% (system still stressed 48h post-event — maintenance not until Apr 30)
- After F2–F4: recovery is faster, consistent with prompt maintenance

---

## 5. Limitations

1. **Only 4 labeled failure events**: Statistical conclusions about sensor thresholds must be treated as indicative. A larger failure corpus would be needed to determine definitive operating limits.

2. **All failures are the same type (air leak)**: The system is calibrated only for this failure mode. Other failure types (e.g., oil system failure, bearing failure) are not represented and cannot be detected by models trained on this data alone.

3. **Temporal coverage**: Feb–Mar 2020 is used as the "normal" baseline. If system degradation began before March, the baseline may not be entirely clean.

4. **Missing time (909.5h)**: 17.7% of 1-minute slots have no data. These gaps may coincide with important operational events that were not recorded. The largest gaps align with maintenance windows, which is expected.

5. **Correlation ≠ causation**: The strong correlations between H1, TP2, DV_eletric, Motor_current during failure windows describe co-occurrence. The physical causal mechanism (air leak → continuous load → all downstream effects) is described in the documentation and is consistent with observations, but causality is not proven by the correlation alone.

6. **Load fraction and motor state feature quality**: These are computed from 1-minute means of binary/discrete signals. Within-minute variability (e.g., 30s loaded + 30s off) will produce intermediate values. This is appropriate for the intended use.

---

## 6. Recommendations for Stage 3

Based on the findings above, Stage 3 (feature engineering + anomaly detection) should:

1. **Use H1, DV_eletric (load_fraction), Motor_current, and Oil_temperature as the primary features** — these show the largest and most consistent separation between normal and failure states.

2. **Use Feb–Mar 2020 as the clean normal training baseline** for unsupervised anomaly detection. LPS activity before March 11 is minimal, confirming a genuinely clean period.

3. **Engineer rolling-window features** (30-min, 2h, 6h) of H1 mean, load_fraction, Motor_current, and Oil_temperature. Pre-failure degradation in F3 and F4 is only visible over multi-hour windows.

4. **Use gap_segment_id boundaries** when computing all rolling features — never roll across a gap.

5. **Use temporal cross-validation only**: Train on Feb–Apr, validate on May–Jul, test on Jul–Sep (or similar forward-rolling scheme). Never shuffle the time series.

6. **Treat class imbalance carefully**: Failure rows = 5,253 / 252,720 non-gap rows = **2.08%**. Use precision-recall AUC, not accuracy, as the primary evaluation metric.

7. **Consider multi-resolution features**: 1-minute resolution for fine signal changes + hourly aggregates for slower thermal drift.
