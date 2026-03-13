"""
test_coverage_boost.py
======================
Targets files with lowest coverage to push overall from 58% → 80%+:
  - src/inference_engine.py    (0%  → ~85%)
  - src/universal_trainer.py   (0%  → ~80%)
  - src/shap_explainer.py      (21% → ~75%)
  - src/fraud_system.py        (59% → ~95%)

Place this file in tests/ and run:
    pytest tests/test_coverage_boost.py -v --tb=short
    pytest tests/ --cov=src --cov=Scripts --cov-report=term-missing
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_dataset():
    """100-row balanced dataset — both classes guaranteed."""
    np.random.seed(0)
    n = 100
    data = {"Time": np.random.uniform(0, 172800, n),
            "Amount": np.random.uniform(0, 500, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    data["Class"] = [0] * 80 + [1] * 20
    return pd.DataFrame(data)


@pytest.fixture
def trained_fraud_detector(small_dataset, tmp_path):
    """Fitted AutoMLFraudDetector using fast LR. Returns (detector, X_val, y_val, tmp_path)."""
    from src.model_selector import AutoModelSelector
    from src.fraud_system import AutoMLFraudDetector
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = small_dataset.drop("Class", axis=1)
    y = small_dataset["Class"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    selector = AutoModelSelector()
    selector.MODELS = {
        "LR": LogisticRegression(max_iter=200, class_weight="balanced")
    }
    detector = AutoMLFraudDetector(
        model_selector=selector,
        objective="f1",
        model_path=str(tmp_path / "model.pkl"),
        model_version="test_v1"
    )
    detector.fit(X_train, y_train, X_val, y_val)
    return detector, X_val, y_val, tmp_path


@pytest.fixture
def rf_pipeline_and_data(small_dataset):
    """Fitted RandomForest pipeline — required for SHAP TreeExplainer."""
    from src.model_selector import AutoModelSelector
    from sklearn.ensemble import RandomForestClassifier

    X = small_dataset.drop("Class", axis=1)
    y = small_dataset["Class"]
    selector = AutoModelSelector()
    selector.MODELS = {
        "RF": RandomForestClassifier(n_estimators=10, random_state=42,
                                     class_weight="balanced")
    }
    pipeline, _, _ = selector.train_models(X, y)
    return pipeline, X, y


# RF-backed trained_fraud_detector for InferenceEngine tests (SHAP needs a tree model)
@pytest.fixture
def trained_fraud_detector_rf(small_dataset, tmp_path):
    """Fitted AutoMLFraudDetector using RandomForest (required for SHAP TreeExplainer)."""
    from src.model_selector import AutoModelSelector
    from src.fraud_system import AutoMLFraudDetector
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = small_dataset.drop("Class", axis=1)
    y = small_dataset["Class"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    selector = AutoModelSelector()
    selector.MODELS = {
        "RF": RandomForestClassifier(n_estimators=10, random_state=42,
                                     class_weight="balanced")
    }
    detector = AutoMLFraudDetector(
        model_selector=selector,
        objective="f1",
        model_path=str(tmp_path / "rf_model.pkl"),
        model_version="test_rf_v1"
    )
    detector.fit(X_train, y_train, X_val, y_val)
    return detector, X_val, y_val, tmp_path


@pytest.fixture
def universal_df():
    """Small binary classification DataFrame for UniversalTrainer tests.
    Uses 180/20 split so minority_ratio = 0.1 — clearly < 0.2 threshold."""
    np.random.seed(7)
    n = 200
    df = pd.DataFrame({
        "num1":  np.random.randn(n),
        "num2":  np.random.uniform(0, 100, n),
        "cat1":  np.random.choice(["A", "B", "C"], n),
        "label": [0] * 180 + [1] * 20,   # 10% minority → is_imbalanced = True
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FRAUD SYSTEM  (fraud_system.py  59% → ~95%)
# ─────────────────────────────────────────────────────────────────────────────

class TestFraudSystemExtended:

    # ── evaluate() ────────────────────────────────────────────────────────────

    def test_evaluate_runs_successfully(self, trained_fraud_detector):
        detector, X_val, y_val, _ = trained_fraud_detector
        detector.evaluate(X_val, y_val)   # must not raise

    def test_evaluate_raises_when_not_fitted(self, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "m.pkl")
        )
        with pytest.raises(ValueError, match="not trained"):
            det.evaluate(MagicMock(), MagicMock())

    def test_evaluate_single_class_no_crash(self, trained_fraud_detector):
        """evaluate() logs a warning for single-class y_test — must not raise."""
        detector, X_val, _, _ = trained_fraud_detector
        detector.evaluate(X_val, np.zeros(len(X_val), dtype=int))

    # ── predict / predict_proba ───────────────────────────────────────────────

    def test_predict_proba_raises_when_not_fitted(self, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "m.pkl")
        )
        with pytest.raises(ValueError):
            det.predict_proba(pd.DataFrame([[1, 2]]))

    def test_predict_output_is_binary(self, trained_fraud_detector):
        detector, X_val, _, _ = trained_fraud_detector
        assert set(detector.predict(X_val)).issubset({0, 1})

    def test_predict_proba_values_in_range(self, trained_fraud_detector):
        detector, X_val, _, _ = trained_fraud_detector
        proba = detector.predict_proba(X_val)
        assert ((proba >= 0) & (proba <= 1)).all()

    # ── get_* accessors ───────────────────────────────────────────────────────

    def test_get_model_has_predict_proba(self, trained_fraud_detector):
        detector, _, _, _ = trained_fraud_detector
        assert hasattr(detector.get_model(), "predict_proba")

    def test_get_feature_names_matches_training_columns(self, trained_fraud_detector, small_dataset):
        detector, _, _, _ = trained_fraud_detector
        assert detector.get_feature_names() == list(small_dataset.drop("Class", axis=1).columns)

    def test_get_version_correct(self, trained_fraud_detector):
        detector, _, _, _ = trained_fraud_detector
        assert detector.get_version() == "test_v1"

    def test_get_threshold_in_valid_range(self, trained_fraud_detector):
        detector, _, _, _ = trained_fraud_detector
        assert 0.0 <= detector.get_threshold() <= 1.0

    # ── load() ────────────────────────────────────────────────────────────────

    def test_load_restores_threshold(self, trained_fraud_detector):
        detector, _, _, tmp_path = trained_fraud_detector
        original = detector.get_threshold()
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det2 = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "model.pkl")
        )
        det2.load()
        assert abs(det2.get_threshold() - original) < 1e-6

    def test_load_restores_feature_names(self, trained_fraud_detector):
        detector, _, _, tmp_path = trained_fraud_detector
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det2 = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "model.pkl")
        )
        det2.load()
        assert det2.get_feature_names() == detector.get_feature_names()

    def test_load_raises_file_not_found(self, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "missing.pkl")
        )
        with pytest.raises(FileNotFoundError):
            det.load()

    def test_auto_load_restores_model(self, trained_fraud_detector):
        """auto_load=True should restore the model automatically when file exists."""
        _, _, _, tmp_path = trained_fraud_detector
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det2 = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "model.pkl"),
            auto_load=True
        )
        assert det2.best_model is not None

    # ── objective paths ───────────────────────────────────────────────────────

    def test_cost_objective_sets_threshold(self, small_dataset, tmp_path):
        """objective='cost' should route through BusinessCostOptimizer."""
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}
        det = AutoMLFraudDetector(
            model_selector=selector,
            objective="cost",
            model_path=str(tmp_path / "cost_model.pkl")
        )
        det.fit(X_train, y_train, X_val, y_val)
        assert 0.0 <= det.get_threshold() <= 1.0

    def test_manual_threshold_overrides_optimizer(self, small_dataset, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}
        det = AutoMLFraudDetector(
            model_selector=selector,
            objective="f1",
            manual_threshold=0.42,
            model_path=str(tmp_path / "manual_model.pkl")
        )
        det.fit(X_train, y_train, X_val, y_val)
        assert abs(det.get_threshold() - 0.42) < 1e-6

    def test_fit_none_raises_value_error(self, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        det = AutoMLFraudDetector(
            model_selector=AutoModelSelector(),
            model_path=str(tmp_path / "m.pkl")
        )
        with pytest.raises(ValueError):
            det.fit(None, None, None, None)

    def test_saved_package_has_all_keys(self, trained_fraud_detector):
        _, _, _, tmp_path = trained_fraud_detector
        data = joblib.load(str(tmp_path / "model.pkl"))
        for key in ["model", "threshold", "objective", "cv_score",
                    "feature_names", "features", "model_version", "timestamp"]:
            assert key in data, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHAP EXPLAINER  (shap_explainer.py  21% → ~75%)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntelligentSHAP:

    # ── _build_explainer ──────────────────────────────────────────────────────

    def test_build_explainer_sets_feature_names(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s._build_explainer(X.sample(20, random_state=0))
        assert s.feature_names is not None and len(s.feature_names) > 0

    def test_build_explainer_sets_explainer_object(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s._build_explainer(X.sample(20, random_state=0))
        assert s.explainer is not None

    # ── _preprocess ───────────────────────────────────────────────────────────

    def test_preprocess_returns_dataframe(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s._build_explainer(X.sample(20, random_state=0))
        assert isinstance(s._preprocess(X.iloc[:5]), pd.DataFrame)

    def test_preprocess_no_nulls(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s._build_explainer(X.sample(20, random_state=0))
        assert s._preprocess(X.iloc[:5]).isnull().sum().sum() == 0

    # ── _shap_values ──────────────────────────────────────────────────────────

    def test_shap_values_returns_2d_array(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s._build_explainer(X.sample(20, random_state=0))
        vals = s._shap_values(s._preprocess(X.iloc[:5]))
        assert vals.ndim == 2 and vals.shape[0] == 5

    def test_shap_values_shape_matches_features(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s._build_explainer(X.sample(20, random_state=0))
        vals = s._shap_values(s._preprocess(X.iloc[:3]))
        assert vals.shape[1] == len(s.feature_names)

    # ── local_explanation_json ────────────────────────────────────────────────

    def test_local_explanation_json_keys(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        result = IntelligentSHAP(pipeline).local_explanation_json(X, index=0)
        for k in ["base_value", "top_positive_features", "top_negative_features"]:
            assert k in result

    def test_local_explanation_json_top_k_limit(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        result = IntelligentSHAP(pipeline).local_explanation_json(X, index=0, top_k=3)
        assert len(result["top_positive_features"]) <= 3
        assert len(result["top_negative_features"]) <= 3

    def test_local_explanation_json_base_value_is_float(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        result = IntelligentSHAP(pipeline).local_explanation_json(X, index=0)
        assert isinstance(result["base_value"], float)

    def test_local_explanation_json_feature_dicts_have_keys(self, rf_pipeline_and_data):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        result = IntelligentSHAP(pipeline).local_explanation_json(X, index=0)
        for item in result["top_positive_features"] + result["top_negative_features"]:
            assert "feature" in item and "impact" in item

    def test_local_explanation_json_reuses_explainer(self, rf_pipeline_and_data):
        """Second call must reuse the same explainer — not rebuild it."""
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        s = IntelligentSHAP(pipeline)
        s.local_explanation_json(X, index=0)
        first = s.explainer
        s.local_explanation_json(X, index=1)
        assert s.explainer is first

    # ── local_explanation (plot) ──────────────────────────────────────────────

    def test_local_explanation_saves_png(self, rf_pipeline_and_data, tmp_path, monkeypatch):
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        monkeypatch.chdir(tmp_path)
        IntelligentSHAP(pipeline).local_explanation(X, index=0)
        assert os.path.exists("reports/shap/waterfall_0.png")

    # ── global_explanation (plot) ─────────────────────────────────────────────

    def test_global_explanation_saves_both_plots(self, rf_pipeline_and_data, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")   # force non-interactive backend — no Tk needed
        from src.shap_explainer import IntelligentSHAP
        pipeline, X, _ = rf_pipeline_and_data
        monkeypatch.chdir(tmp_path)
        IntelligentSHAP(pipeline).global_explanation(X)
        assert os.path.exists("reports/shap/global_summary.png")
        assert os.path.exists("reports/shap/feature_importance_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INFERENCE ENGINE  (inference_engine.py  0% → ~85%)
# ─────────────────────────────────────────────────────────────────────────────

class TestFraudInferenceEngine:

    @pytest.fixture
    def inference_engine(self, trained_fraud_detector_rf):
        """Live FraudInferenceEngine backed by an RF model (SHAP TreeExplainer needs tree model)."""
        detector, X_val, _, model_tmp = trained_fraud_detector_rf

        metadata_path = str(model_tmp / "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "model_name":    "test-model",
                "model_version": "test_rf_v1",
                "threshold":     detector.get_threshold(),
                "features":      detector.get_feature_names(),
                "cv_score":      0.99,
            }, f)

        from src.inference_engine import FraudInferenceEngine
        engine = FraudInferenceEngine(
            model_path=str(model_tmp / "rf_model.pkl"),
            metadata_path=metadata_path
        )
        return engine, X_val, detector.get_feature_names()

    # ── constructor ───────────────────────────────────────────────────────────

    def test_engine_raises_if_model_missing(self, tmp_path):
        from src.inference_engine import FraudInferenceEngine
        with pytest.raises(FileNotFoundError):
            FraudInferenceEngine(
                model_path=str(tmp_path / "no_model.pkl"),
                metadata_path=str(tmp_path / "no_meta.json")
            )

    def test_engine_loads_without_metadata_file(self, trained_fraud_detector_rf):
        """Engine falls back gracefully when metadata JSON is absent."""
        _, _, _, model_tmp = trained_fraud_detector_rf
        from src.inference_engine import FraudInferenceEngine
        engine = FraudInferenceEngine(
            model_path=str(model_tmp / "rf_model.pkl"),
            metadata_path=str(model_tmp / "nonexistent_meta.json")
        )
        assert engine.expected_features is not None

    def test_engine_stores_expected_features(self, inference_engine):
        engine, _, feature_names = inference_engine
        assert engine.expected_features == feature_names

    # ── _validate_and_prepare_input ───────────────────────────────────────────

    def test_validate_raises_on_non_dict(self, inference_engine):
        engine, _, _ = inference_engine
        with pytest.raises(ValueError, match="dictionary"):
            engine._validate_and_prepare_input([1, 2, 3])

    def test_validate_raises_on_missing_features(self, inference_engine):
        engine, _, _ = inference_engine
        with pytest.raises(ValueError, match="Missing features"):
            engine._validate_and_prepare_input({"only_one_feature": 1.0})

    def test_validate_strips_extra_features(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        row["EXTRA_JUNK"] = 999.0
        df = engine._validate_and_prepare_input(row)
        assert "EXTRA_JUNK" not in df.columns

    def test_validate_returns_ordered_dataframe(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        df = engine._validate_and_prepare_input(row)
        assert list(df.columns) == feature_names

    # ── predict ───────────────────────────────────────────────────────────────

    def test_predict_returns_expected_top_keys(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        result = engine.predict(row)
        for k in ["model_info", "prediction_result", "explanation"]:
            assert k in result

    def test_predict_probability_in_range(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        prob = engine.predict(row)["prediction_result"]["probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_class_is_binary(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        assert engine.predict(row)["prediction_result"]["predicted_class"] in {0, 1}

    def test_predict_threshold_used_matches_engine(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        result = engine.predict(row)
        assert abs(result["prediction_result"]["threshold_used"] - engine.threshold) < 1e-6

    def test_predict_explanation_has_shap_keys(self, inference_engine):
        engine, X_val, feature_names = inference_engine
        row = dict(zip(feature_names, X_val.iloc[0].values))
        expl = engine.predict(row)["explanation"]
        for k in ["base_value", "top_positive_features", "top_negative_features"]:
            assert k in expl

    def test_predict_raises_on_missing_features(self, inference_engine):
        engine, _, _ = inference_engine
        with pytest.raises(ValueError):
            engine.predict({"bad_feature": 0.0})

    # ── get_metadata ──────────────────────────────────────────────────────────

    def test_get_metadata_returns_dict(self, inference_engine):
        engine, _, _ = inference_engine
        assert isinstance(engine.get_metadata(), dict)

    def test_get_metadata_has_features_key(self, inference_engine):
        engine, _, _ = inference_engine
        assert "features" in engine.get_metadata()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  UNIVERSAL TRAINER  (universal_trainer.py  0% → ~80%)
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetProfiler:

    def test_profile_returns_expected_keys(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        profile = DatasetProfiler().profile(universal_df, target_col="label")
        for key in ["n_rows", "n_cols", "n_numeric", "n_categorical",
                    "numeric_cols", "categorical_cols", "n_classes",
                    "minority_ratio", "is_imbalanced", "missing_cols",
                    "missing_pct", "has_missing", "needs_sampling"]:
            assert key in profile, f"Missing key: {key}"

    def test_profile_correct_row_count(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        assert DatasetProfiler().profile(universal_df, "label")["n_rows"] == len(universal_df)

    def test_profile_detects_imbalance(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        # 160 / 40 → minority_ratio = 0.2 → is_imbalanced = True
        assert DatasetProfiler().profile(universal_df, "label")["is_imbalanced"]

    def test_profile_raises_on_missing_target(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        with pytest.raises(ValueError, match="not found"):
            DatasetProfiler().profile(universal_df, target_col="nonexistent")

    def test_profile_detects_categorical_columns(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        assert "cat1" in DatasetProfiler().profile(universal_df, "label")["categorical_cols"]

    def test_profile_detects_numeric_columns(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        profile = DatasetProfiler().profile(universal_df, "label")
        assert "num1" in profile["numeric_cols"]
        assert "num2" in profile["numeric_cols"]

    def test_profile_no_missing_by_default(self, universal_df):
        from src.universal_trainer import DatasetProfiler
        assert DatasetProfiler().profile(universal_df, "label")["has_missing"] == False

    def test_profile_detects_missing_values(self):
        from src.universal_trainer import DatasetProfiler
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1, 0, 1]})
        profile = DatasetProfiler().profile(df, "b")
        assert profile["has_missing"] == True
        assert "a" in profile["missing_cols"]


class TestBuildPreprocessor:

    def test_returns_column_transformer(self):
        from src.universal_trainer import build_preprocessor
        from sklearn.compose import ColumnTransformer
        assert isinstance(build_preprocessor(["num1", "num2"], ["cat1"]), ColumnTransformer)

    def test_raises_when_no_columns(self):
        from src.universal_trainer import build_preprocessor
        with pytest.raises(ValueError, match="No usable columns"):
            build_preprocessor([], [])

    def test_works_with_only_numeric(self):
        from src.universal_trainer import build_preprocessor
        assert build_preprocessor(["num1", "num2"], []) is not None

    def test_works_with_only_categorical(self):
        from src.universal_trainer import build_preprocessor
        assert build_preprocessor([], ["cat1"]) is not None


class TestGetModels:

    def test_returns_four_models(self):
        from src.universal_trainer import get_models
        assert len(get_models(is_imbalanced=True, n_rows=1000)) == 4

    def test_all_models_have_fit(self):
        from src.universal_trainer import get_models
        for name, model in get_models(True, 1000).items():
            assert hasattr(model, "fit"), f"{name} missing .fit()"

    def test_large_dataset_reduces_estimators(self):
        from src.universal_trainer import get_models
        assert get_models(is_imbalanced=False, n_rows=300_000)["RandomForest"].n_estimators == 100

    def test_balanced_sets_class_weight(self):
        from src.universal_trainer import get_models
        assert get_models(is_imbalanced=True, n_rows=1000)["LogisticRegression"].class_weight == "balanced"


class TestRamUtilities:

    def test_get_available_ram_positive(self):
        from src.universal_trainer import get_available_ram_gb
        assert get_available_ram_gb() > 0

    def test_get_dataframe_ram_positive(self, universal_df):
        from src.universal_trainer import get_dataframe_ram_gb
        assert get_dataframe_ram_gb(universal_df) > 0

    def test_check_ram_safety_keys(self, universal_df):
        from src.universal_trainer import check_ram_safety
        result = check_ram_safety(universal_df)
        for key in ["available_gb", "dataframe_gb", "estimated_needed_gb", "is_safe"]:
            assert key in result


class TestUniversalTrainer:

    def test_fit_returns_metrics_dict(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        assert isinstance(
            UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(universal_df, "label"),
            dict
        )

    def test_fit_metrics_has_expected_keys(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        metrics = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(universal_df, "label")
        for key in ["best_model", "cv_roc_auc", "test_roc_auc",
                    "f1_score", "recall", "precision", "threshold"]:
            assert key in metrics, f"Missing metrics key: {key}"

    def test_fit_roc_auc_in_range(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        metrics = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(universal_df, "label")
        assert 0.0 <= metrics["cv_roc_auc"] <= 1.0
        assert 0.0 <= metrics["test_roc_auc"] <= 1.0

    def test_fit_saves_model_to_disk(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        path = str(tmp_path / "u.pkl")
        UniversalTrainer(model_save_path=path).fit(universal_df, "label")
        assert os.path.exists(path)

    def test_fit_raises_on_multiclass(self, tmp_path):
        from src.universal_trainer import UniversalTrainer
        df = pd.DataFrame({"f1": np.random.randn(90),
                           "label": [0]*30 + [1]*30 + [2]*30})
        with pytest.raises(ValueError, match="Binary"):
            UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(df, "label")

    def test_predict_proba_after_fit(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        trainer = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl"))
        trainer.fit(universal_df, "label")
        X = universal_df.drop("label", axis=1)
        proba = trainer.predict_proba(X)
        assert len(proba) == len(X)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predict_after_fit_is_binary(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        trainer = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl"))
        trainer.fit(universal_df, "label")
        assert set(trainer.predict(universal_df.drop("label", axis=1))).issubset({0, 1})

    def test_predict_proba_raises_before_fit(self, tmp_path):
        from src.universal_trainer import UniversalTrainer
        with pytest.raises(ValueError, match="not trained"):
            UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).predict_proba(
                pd.DataFrame([[1, 2]])
            )

    def test_save_and_load_roundtrip(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        path = str(tmp_path / "u.pkl")
        t1 = UniversalTrainer(model_save_path=path)
        t1.fit(universal_df, "label")
        t2 = UniversalTrainer(model_save_path=path)
        t2.load(path)
        assert t2.best_model_name == t1.best_model_name
        assert abs(t2.threshold - t1.threshold) < 1e-6

    def test_load_raises_file_not_found(self, tmp_path):
        from src.universal_trainer import UniversalTrainer
        with pytest.raises(FileNotFoundError):
            UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).load(
                str(tmp_path / "nonexistent.pkl")
            )

    def test_positive_label_string_encoding(self, tmp_path):
        """positive_label='Yes' must be encoded as 1."""
        from src.universal_trainer import UniversalTrainer
        np.random.seed(1)
        n = 100
        df = pd.DataFrame({
            "f1": np.random.randn(n), "f2": np.random.randn(n),
            "outcome": ["No"]*80 + ["Yes"]*20
        })
        metrics = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(
            df, target_col="outcome", positive_label="Yes"
        )
        assert metrics["best_model"] is not None

    def test_progress_callback_called_six_times(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        calls = []
        UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(
            universal_df, "label",
            progress_callback=lambda s, t, m: calls.append(s)
        )
        assert len(calls) >= 6

    def test_encode_target_binary_integers_unchanged(self, tmp_path):
        from src.universal_trainer import UniversalTrainer
        trainer = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl"))
        y = pd.Series([0, 1, 0, 1, 1])
        np.testing.assert_array_equal(trainer._encode_target(y, None), [0, 1, 0, 1, 1])

    def test_encode_target_with_positive_label(self, tmp_path):
        from src.universal_trainer import UniversalTrainer
        trainer = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl"))
        y = pd.Series(["No", "Yes", "No", "Yes"])
        np.testing.assert_array_equal(trainer._encode_target(y, "Yes"), [0, 1, 0, 1])

    def test_all_cv_scores_in_metrics(self, universal_df, tmp_path):
        from src.universal_trainer import UniversalTrainer
        metrics = UniversalTrainer(model_save_path=str(tmp_path / "u.pkl")).fit(universal_df, "label")
        assert "all_cv_scores" in metrics
        assert isinstance(metrics["all_cv_scores"], dict)
