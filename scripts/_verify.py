"""Verify artifacts."""
import sys, os
sys.path.insert(0, '.')
from src.config import RAW_CSV, PROCESSED_DIR, ARTIFACTS_DIR
print('Raw CSV:', RAW_CSV.exists(), os.path.getsize(str(RAW_CSV)), 'bytes')
print('Processed parquet:', (PROCESSED_DIR / 'processed_1min.parquet').exists())
artifacts = [
    'anomaly_model.pkl','anomaly_scaler.pkl','anomaly_feature_cols.json',
    'anomaly_score_stats.json','anomaly_scores.parquet',
    'predictive_lr_model.pkl','predictive_gbt_model.pkl',
    'predictive_results.json','risk_scores.parquet',
    'shap_feature_importance.json','baseline_stats.json',
]
missing = [f for f in artifacts if not (ARTIFACTS_DIR/f).exists()]
print(f'Artifacts: {len(artifacts)-len(missing)}/{len(artifacts)} present')
print('Missing:', missing or 'None')
