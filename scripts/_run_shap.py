"""Quick SHAP test script."""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from src.data.loader import load_parquet
from src.config import PROCESSED_DIR
from src.features.engineering import build_feature_matrix
from src.features.labeling import build_labels, get_temporal_splits
from src.models.predictive import load_predictive_models
from src.explainability.shap_analysis import compute_shap_values, plot_feature_importance

df = load_parquet(PROCESSED_DIR / 'processed_1min.parquet')
df_features = build_feature_matrix(df)
pred = load_predictive_models()
feat_cols = pred['feat_cols']

df_labeled = build_labels(df)
train_idx, val_idx, test_idx = get_temporal_splits(df_labeled)
common_test = test_idx.intersection(df_features.index)
X_test = df_features.loc[common_test, feat_cols].fillna(0.0)

bundle = compute_shap_values(pred['gbt_model'], X_test, feat_cols, model_type='gbt', save_artifacts=True)
print('Top 10:', [(x['feature'], round(x['mean_abs_shap'],4)) for x in bundle['importance'][:10]])

fig = plot_feature_importance(
    bundle, top_n=20,
    title='RailGuard — Feature Importance (SHAP, GBT, Test Set)',
    save_path='data/processed/figures/feature_importance_shap.png',
)
print('Done.')
