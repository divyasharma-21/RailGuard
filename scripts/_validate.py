"""Test that dashboard imports cleanly and data loading functions work."""
import sys
sys.path.insert(0, '.')

# Test imports
print("Testing imports...")
from src.config import ARTIFACTS_DIR, PROCESSED_DIR, FAILURE_EVENTS
import json, pandas as pd, numpy as np
print("  src.config OK")

from src.models.anomaly import load_anomaly_model
bundle = load_anomaly_model()
print(f"  anomaly model loaded: {len(bundle['feature_cols'])} features, thresholds={bundle['score_stats']['alert_threshold_low']:.3f}/{bundle['score_stats']['alert_threshold_high']:.3f}")

from src.models.predictive import load_predictive_models
pred = load_predictive_models()
print(f"  predictive models loaded: LR+GBT, feat={len(pred['feat_cols'])}")

from src.models.decision_support import DecisionSupport
ds = DecisionSupport.from_artifacts()
print(f"  decision support: low={ds.anomaly_threshold_low:.3f}, high={ds.anomaly_threshold_high:.3f}")

# Test data loading
df = pd.read_parquet(str(PROCESSED_DIR / 'processed_1min.parquet'))
a_scores = pd.read_parquet(str(ARTIFACTS_DIR / 'anomaly_scores.parquet'))
r_scores = pd.read_parquet(str(ARTIFACTS_DIR / 'risk_scores.parquet'))
print(f"  processed data: {df.shape}")
print(f"  anomaly scores: {a_scores.shape}, range=[{a_scores['anomaly_score'].min():.3f},{a_scores['anomaly_score'].max():.3f}]")
print(f"  risk scores: {r_scores.shape}, range=[{r_scores['risk_prob'].min():.4f},{r_scores['risk_prob'].max():.4f}]")

# Test that failure rows have high anomaly scores
failure_idx = df[df['failure_id'] != ''].index
fail_scores = a_scores.loc[failure_idx.intersection(a_scores.index), 'anomaly_score']
print(f"  failure mean anomaly score: {fail_scores.mean():.4f} (expected >0.97)")
assert fail_scores.mean() > 0.97, "Failure anomaly scores too low!"

# Test Streamlit imports won't fail — just check it parses
import ast
with open("src/dashboard/app.py", encoding="utf-8") as f:
    src = f.read()
ast.parse(src)
print("  dashboard app.py: syntax OK")

print("\nAll validation checks passed!")
