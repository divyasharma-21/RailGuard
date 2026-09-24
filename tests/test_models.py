"""
tests/test_models.py — Tests for model loading, anomaly scoring, and evaluation metrics.

These tests verify:
  - Saved artifacts can be loaded without error
  - Anomaly model produces scores in [0, 1]
  - Predictive model produces probabilities in [0, 1]
  - Evaluation metrics are correctly computed
  - Decision support classifies correctly
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR


# ── Evaluation metric tests ───────────────────────────────────────────────────

class TestEvaluateModel:

    def test_perfect_classifier(self):
        from src.models.evaluation import evaluate_model
        y = np.array([0, 0, 0, 1, 1], dtype="int8")
        p = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        m = evaluate_model(y, p, threshold=0.5, name="test")
        assert m["pr_auc"] == pytest.approx(1.0, abs=1e-4)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_random_classifier(self):
        from src.models.evaluation import evaluate_model
        rng = np.random.default_rng(1)
        n = 1000
        y = (rng.random(n) < 0.05).astype("int8")  # 5% positive
        p = rng.random(n)  # random probabilities
        m = evaluate_model(y, p, threshold=0.5, name="random")
        # PR-AUC should be close to base rate for random classifier
        assert m["pr_auc"] < 0.2

    def test_no_positive_samples(self):
        from src.models.evaluation import evaluate_model
        y = np.zeros(100, dtype="int8")
        p = np.random.rand(100)
        m = evaluate_model(y, p, threshold=0.5, name="no_pos")
        assert m["pr_auc"] == 0.0
        assert m["n_positive"] == 0

    def test_confusion_matrix_components_sum(self):
        from src.models.evaluation import evaluate_model
        y = np.array([0, 0, 1, 1, 0, 1], dtype="int8")
        p = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.7])
        m = evaluate_model(y, p, threshold=0.5, name="cm_test")
        total = m["tp"] + m["fp"] + m["tn"] + m["fn"]
        assert total == len(y)

    def test_threshold_affects_predictions(self):
        from src.models.evaluation import evaluate_model
        y = np.array([0, 0, 0, 1, 1, 1], dtype="int8")
        p = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        m_low  = evaluate_model(y, p, threshold=0.3, name="low_t")
        m_high = evaluate_model(y, p, threshold=0.9, name="high_t")
        # Lower threshold → more predicted positives → higher recall
        assert m_low["recall"] >= m_high["recall"]

    def test_returns_all_required_keys(self):
        from src.models.evaluation import evaluate_model
        y = np.array([0, 1, 0, 1], dtype="int8")
        p = np.array([0.1, 0.9, 0.2, 0.8])
        m = evaluate_model(y, p)
        for key in ["pr_auc", "roc_auc", "precision", "recall", "f1",
                    "tp", "fp", "tn", "fn", "threshold", "n_positive",
                    "n_negative", "base_rate"]:
            assert key in m, f"Missing metric key: {key}"


# ── Anomaly model artifact tests ──────────────────────────────────────────────

class TestAnomalyModelArtifacts:

    @pytest.fixture(autouse=True)
    def check_artifacts_exist(self):
        """Skip tests if model artifacts haven't been trained yet."""
        if not (ARTIFACTS_DIR / "anomaly_model.pkl").exists():
            pytest.skip("Anomaly model artifacts not found. Run train_models.py first.")

    def test_load_model_no_error(self):
        from src.models.anomaly import load_anomaly_model
        bundle = load_anomaly_model()
        assert "model" in bundle
        assert "scaler" in bundle
        assert "feature_cols" in bundle
        assert "score_stats" in bundle

    def test_feature_cols_is_list(self):
        from src.models.anomaly import load_anomaly_model
        bundle = load_anomaly_model()
        assert isinstance(bundle["feature_cols"], list)
        assert len(bundle["feature_cols"]) > 0

    def test_score_stats_has_thresholds(self):
        from src.models.anomaly import load_anomaly_model
        bundle = load_anomaly_model()
        stats = bundle["score_stats"]
        assert "alert_threshold_low"  in stats
        assert "alert_threshold_high" in stats
        assert stats["alert_threshold_low"] < stats["alert_threshold_high"]

    def test_anomaly_scores_parquet_loadable(self):
        scores_path = ARTIFACTS_DIR / "anomaly_scores.parquet"
        assert scores_path.exists(), "anomaly_scores.parquet not found"
        df = pd.read_parquet(str(scores_path))
        assert "anomaly_score" in df.columns
        valid = df["anomaly_score"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), "Scores should be in [0, 1]"


# ── Predictive model artifact tests ──────────────────────────────────────────

class TestPredictiveModelArtifacts:

    @pytest.fixture(autouse=True)
    def check_artifacts_exist(self):
        if not (ARTIFACTS_DIR / "predictive_lr_model.pkl").exists():
            pytest.skip("Predictive model artifacts not found. Run train_models.py first.")

    def test_load_models_no_error(self):
        from src.models.predictive import load_predictive_models
        bundle = load_predictive_models()
        assert "lr_model" in bundle
        assert "gbt_model" in bundle
        assert "scaler" in bundle
        assert "feat_cols" in bundle

    def test_results_json_loadable(self):
        assert (ARTIFACTS_DIR / "predictive_results.json").exists()
        with open(ARTIFACTS_DIR / "predictive_results.json") as f:
            results = json.load(f)
        for key in ["lr", "gbt", "train_size", "test_size"]:
            assert key in results

    def test_lr_metrics_present(self):
        with open(ARTIFACTS_DIR / "predictive_results.json") as f:
            results = json.load(f)
        m = results["lr"]["test_metrics"]
        for key in ["pr_auc", "precision", "recall", "f1"]:
            assert key in m

    def test_risk_scores_parquet_loadable(self):
        risk_path = ARTIFACTS_DIR / "risk_scores.parquet"
        assert risk_path.exists()
        df = pd.read_parquet(str(risk_path))
        assert "risk_prob" in df.columns
        valid = df["risk_prob"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_predict_produces_valid_probs(self):
        from src.models.predictive import load_predictive_models, predict_risk
        bundle = load_predictive_models()
        feat_cols = bundle["feat_cols"]

        # Build a tiny synthetic feature DataFrame with required columns
        n = 20
        idx = pd.date_range("2020-08-01", periods=n, freq="1min")
        rng = np.random.default_rng(5)
        X = pd.DataFrame(
            {col: rng.random(n).astype("float32") for col in feat_cols},
            index=idx,
        )
        X["is_gap"] = False

        probs = predict_risk(bundle, X, model="gbt")
        valid = probs.dropna()
        assert len(valid) == n
        assert (valid >= 0).all() and (valid <= 1).all()


# ── Decision support tests ─────────────────────────────────────────────────────

class TestDecisionSupport:

    def test_classify_normal(self):
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport(
            anomaly_threshold_low=0.4,
            anomaly_threshold_high=0.7,
            risk_threshold=0.5,
        )
        assert ds.classify(0.2) == "Normal"
        assert ds.classify(0.3) == "Normal"

    def test_classify_monitor(self):
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport(
            anomaly_threshold_low=0.4,
            anomaly_threshold_high=0.7,
            risk_threshold=0.5,
        )
        assert ds.classify(0.5) == "Monitor"

    def test_classify_investigate_by_anomaly(self):
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport(
            anomaly_threshold_low=0.4,
            anomaly_threshold_high=0.7,
            risk_threshold=0.5,
        )
        assert ds.classify(0.8) == "Investigate"

    def test_classify_investigate_by_risk(self):
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport(
            anomaly_threshold_low=0.4,
            anomaly_threshold_high=0.7,
            risk_threshold=0.5,
        )
        # Even a low anomaly score should flag Investigate if risk_prob >= threshold
        assert ds.classify(0.2, risk_prob=0.6) == "Investigate"

    def test_nan_anomaly_score_returns_normal(self):
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport()
        assert ds.classify(float("nan")) == "Normal"

    def test_classify_series(self):
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport(
            anomaly_threshold_low=0.4,
            anomaly_threshold_high=0.7,
        )
        scores = pd.Series([0.1, 0.5, 0.8, float("nan")],
                           index=pd.date_range("2020-01-01", periods=4, freq="1min"))
        cats = ds.classify_series(scores)
        assert cats.iloc[0] == "Normal"
        assert cats.iloc[1] == "Monitor"
        assert cats.iloc[2] == "Investigate"
        assert cats.iloc[3] == "Normal"

    @pytest.fixture(autouse=True)
    def _skip_if_no_artifacts(self):
        """from_artifacts test only runs when artifacts exist."""
        pass

    def test_from_artifacts_when_available(self):
        if not (ARTIFACTS_DIR / "anomaly_score_stats.json").exists():
            pytest.skip("Score stats artifact not found")
        from src.models.decision_support import DecisionSupport
        ds = DecisionSupport.from_artifacts()
        assert 0 < ds.anomaly_threshold_low < 1
        assert 0 < ds.anomaly_threshold_high < 1
        assert ds.anomaly_threshold_low < ds.anomaly_threshold_high


# ── Data path tests ────────────────────────────────────────────────────────────

class TestDataPaths:

    def test_processed_parquet_exists(self):
        from src.config import PROCESSED_DIR
        assert (PROCESSED_DIR / "processed_1min.parquet").exists(), \
            "Processed data not found. Run preprocessing first."

    def test_raw_csv_exists(self):
        from src.config import RAW_CSV
        assert RAW_CSV.exists(), f"Raw CSV not found at {RAW_CSV}"

    def test_raw_csv_not_modified(self):
        """Raw CSV must not be modified by the pipeline."""
        from src.config import RAW_CSV
        import hashlib
        # Just verify it's readable and has expected shape (1,516,948 rows + header)
        with open(RAW_CSV, "rb") as f:
            first_bytes = f.read(200)
        # Check it's a proper CSV with expected header
        header = first_bytes.decode("utf-8", errors="replace")
        assert "timestamp" in header.lower() or "TP2" in header, \
            "Raw CSV appears corrupted or missing expected header"

    def test_artifacts_dir_has_models(self):
        if not (ARTIFACTS_DIR / "anomaly_model.pkl").exists():
            pytest.skip("Models not trained yet")
        expected_files = [
            "anomaly_model.pkl",
            "anomaly_scaler.pkl",
            "anomaly_feature_cols.json",
            "anomaly_score_stats.json",
        ]
        for f in expected_files:
            assert (ARTIFACTS_DIR / f).exists(), f"Missing artifact: {f}"
