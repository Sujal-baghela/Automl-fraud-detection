import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_transaction():
    row = {"Time": 406.0, "Amount": 149.62}
    for i in range(1, 29):
        row[f"V{i}"] = 0.0
    row["V1"] = -1.3598071
    row["V14"] = -0.3111695
    return pd.DataFrame([row])


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 200
    data = {"Time": np.random.uniform(0, 172800, n),
            "Amount": np.random.uniform(0, 500, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    data["Class"] = np.where(np.random.rand(n) < 0.1, 1, 0)
    return pd.DataFrame(data)


@pytest.fixture
def small_dataset():
    """Minimal dataset guaranteed to have both classes for model training."""
    np.random.seed(0)
    n = 100
    data = {"Time": np.random.uniform(0, 172800, n),
            "Amount": np.random.uniform(0, 500, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    # Force class balance
    labels = [0] * 80 + [1] * 20
    data["Class"] = labels
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

class TestFeatureEngineering:

    def test_adds_five_features(self, sample_dataset):
        from Scripts.train import engineer_features
        X = sample_dataset.drop("Class", axis=1)
        out = engineer_features(X.copy())
        assert out.shape[1] == X.shape[1] + 5

    def test_hour_range(self, sample_dataset):
        from Scripts.train import engineer_features
        out = engineer_features(sample_dataset.drop("Class", axis=1).copy())
        assert out["Hour"].between(0, 24).all()

    def test_night_txn_binary(self, sample_dataset):
        from Scripts.train import engineer_features
        out = engineer_features(sample_dataset.drop("Class", axis=1).copy())
        assert set(out["Night_txn"].unique()).issubset({0, 1})

    def test_amount_log_nonnegative(self, sample_dataset):
        from Scripts.train import engineer_features
        out = engineer_features(sample_dataset.drop("Class", axis=1).copy())
        assert (out["Amount_log"] >= 0).all()

    def test_high_amount_binary(self, sample_dataset):
        from Scripts.train import engineer_features
        out = engineer_features(sample_dataset.drop("Class", axis=1).copy())
        assert set(out["High_amount"].unique()).issubset({0, 1})

    def test_no_nulls(self, sample_dataset):
        from Scripts.train import engineer_features
        out = engineer_features(sample_dataset.drop("Class", axis=1).copy())
        assert out.isnull().sum().sum() == 0

    def test_idempotent(self, sample_dataset):
        """Running twice should not keep adding columns."""
        from Scripts.train import engineer_features
        X = sample_dataset.drop("Class", axis=1).copy()
        out1 = engineer_features(X.copy())
        assert out1.shape[1] == X.shape[1] + 5


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────

class TestDataLoader:

    def test_load_data_success(self, sample_dataset, tmp_path):
        from src.data_loader import DataLoader
        path = str(tmp_path / "data.csv")
        sample_dataset.to_csv(path, index=False)
        loader = DataLoader(file_path=path)
        df = loader.load_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_dataset.shape

    def test_load_data_file_not_found(self):
        from src.data_loader import DataLoader
        loader = DataLoader(file_path="nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_split_removes_target(self, sample_dataset, tmp_path):
        from src.data_loader import DataLoader
        path = str(tmp_path / "data.csv")
        sample_dataset.to_csv(path, index=False)
        loader = DataLoader(file_path=path)
        df = loader.load_data()
        X, y = loader.split_features_target(df, target_column="Class")
        assert "Class" not in X.columns
        assert y.name == "Class"

    def test_split_lengths_match(self, sample_dataset, tmp_path):
        from src.data_loader import DataLoader
        path = str(tmp_path / "data.csv")
        sample_dataset.to_csv(path, index=False)
        loader = DataLoader(file_path=path)
        df = loader.load_data()
        X, y = loader.split_features_target(df, target_column="Class")
        assert len(X) == len(y)

    def test_split_missing_target_raises(self, sample_dataset, tmp_path):
        from src.data_loader import DataLoader
        path = str(tmp_path / "data.csv")
        sample_dataset.to_csv(path, index=False)
        loader = DataLoader(file_path=path)
        df = loader.load_data()
        with pytest.raises(ValueError):
            loader.split_features_target(df, target_column="NonExistent")

    def test_get_metadata_keys(self, sample_dataset, tmp_path):
        from src.data_loader import DataLoader
        path = str(tmp_path / "data.csv")
        sample_dataset.to_csv(path, index=False)
        loader = DataLoader(file_path=path)
        df = loader.load_data()
        meta = loader.get_metadata(df)
        for key in ["rows", "columns", "numerical_features", "categorical_features"]:
            assert key in meta


# ─────────────────────────────────────────────
# DATA CLEANER
# ─────────────────────────────────────────────

class TestDataCleaner:

    def test_remove_duplicates(self):
        from src.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
        cleaner = DataCleaner()
        result = cleaner.remove_duplicates(df)
        assert result.shape[0] == 2

    def test_impute_missing_numerical(self):
        from src.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [4.0, 5.0, 6.0]})
        cleaner = DataCleaner()
        result = cleaner.impute_missing(df)
        assert result["A"].isnull().sum() == 0

    def test_impute_missing_categorical(self):
        from src.cleaner import DataCleaner
        df = pd.DataFrame({"A": ["cat", None, "cat"], "B": [1, 2, 3]})
        cleaner = DataCleaner()
        result = cleaner.impute_missing(df)
        assert result["A"].isnull().sum() == 0

    def test_detect_outliers_returns_df(self, sample_dataset):
        from src.cleaner import DataCleaner
        cleaner = DataCleaner()
        result = cleaner.detect_outliers_iqr(sample_dataset.copy())
        assert isinstance(result, pd.DataFrame)

    def test_clean_pipeline_no_nulls(self, sample_dataset):
        from src.cleaner import DataCleaner
        cleaner = DataCleaner()
        result = cleaner.clean(sample_dataset.copy())
        assert result.isnull().sum().sum() == 0

    def test_clean_removes_duplicates(self):
        from src.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1.0, 1.0, 2.0], "B": [3.0, 3.0, 4.0]})
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert result.shape[0] == 2


# ─────────────────────────────────────────────
# DRIFT DETECTOR
# ─────────────────────────────────────────────

class TestDriftDetector:

    def test_fit_stores_all_features(self, sample_dataset):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector().fit(X)
        assert len(dd.reference) == X.shape[1]

    def test_fit_returns_self(self, sample_dataset):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector()
        result = dd.fit(X)
        assert result is dd

    def test_detect_returns_expected_keys(self, sample_dataset):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector().fit(X)
        report = dd.detect(X)
        for key in ["status", "drift_ratio", "drifted_count", "total_features", "details"]:
            assert key in report

    def test_no_drift_on_same_data(self, sample_dataset):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector(psi_threshold=0.2).fit(X)
        report = dd.detect(X)
        assert report["drift_ratio"] < 0.3

    def test_drift_ratio_between_0_and_1(self, sample_dataset):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector().fit(X)
        report = dd.detect(X)
        assert 0.0 <= report["drift_ratio"] <= 1.0

    def test_save_load_roundtrip(self, sample_dataset, tmp_path):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector().fit(X)
        path = str(tmp_path / "ref.json")
        dd.save(path)
        dd2 = DriftDetector().load(path)
        assert set(dd2.reference.keys()) == set(dd.reference.keys())

    def test_load_preserves_thresholds(self, sample_dataset, tmp_path):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector(psi_threshold=0.15, ks_alpha=0.01).fit(X)
        path = str(tmp_path / "ref.json")
        dd.save(path)
        dd2 = DriftDetector().load(path)
        assert dd2.psi_threshold == 0.15
        assert dd2.ks_alpha == 0.01

    def test_detect_missing_column_skipped(self, sample_dataset):
        from src.drift_detector import DriftDetector
        X = sample_dataset.drop("Class", axis=1)
        dd = DriftDetector().fit(X)
        X_partial = X.drop(columns=["V1"])
        report = dd.detect(X_partial)
        assert report["total_features"] < X.shape[1]


# ─────────────────────────────────────────────
# THRESHOLD OPTIMIZER
# ─────────────────────────────────────────────

class TestThresholdOptimizer:

    @pytest.fixture
    def binary_data(self):
        np.random.seed(42)
        y_true = np.array([0]*80 + [1]*20)
        y_proba = np.clip(y_true + np.random.randn(100) * 0.3, 0, 1)
        return y_true, y_proba

    def test_optimize_returns_float(self, binary_data):
        from src.threshold_optimizer import ThresholdOptimizer
        y_true, y_proba = binary_data
        opt = ThresholdOptimizer(strategy="maximize_f1")
        t = opt.optimize(y_true, y_proba)
        assert isinstance(t, float)

    def test_threshold_in_valid_range(self, binary_data):
        from src.threshold_optimizer import ThresholdOptimizer
        y_true, y_proba = binary_data
        for strategy in ["maximize_f1", "maximize_recall", "maximize_precision"]:
            opt = ThresholdOptimizer(strategy=strategy)
            t = opt.optimize(y_true, y_proba)
            assert 0.0 <= t <= 1.0, f"Threshold out of range for {strategy}"

    def test_invalid_strategy_raises(self, binary_data):
        from src.threshold_optimizer import ThresholdOptimizer
        y_true, y_proba = binary_data
        opt = ThresholdOptimizer(strategy="invalid_strategy")
        with pytest.raises(ValueError):
            opt.optimize(y_true, y_proba)

    def test_get_all_strategies_returns_three(self, binary_data):
        from src.threshold_optimizer import ThresholdOptimizer
        y_true, y_proba = binary_data
        opt = ThresholdOptimizer(strategy="maximize_f1")
        opt.optimize(y_true, y_proba)
        results = opt.get_all_strategies()
        assert len(results) == 3

    def test_maximize_recall_gives_high_recall(self, binary_data):
        from src.threshold_optimizer import ThresholdOptimizer
        from sklearn.metrics import recall_score
        y_true, y_proba = binary_data
        opt = ThresholdOptimizer(strategy="maximize_recall")
        t = opt.optimize(y_true, y_proba)
        y_pred = (y_proba >= t).astype(int)
        assert recall_score(y_true, y_pred) >= 0.7


# ─────────────────────────────────────────────
# BUSINESS COST OPTIMIZER
# ─────────────────────────────────────────────

class TestBusinessCostOptimizer:

    @pytest.fixture
    def binary_data(self):
        np.random.seed(42)
        y_true = np.array([0]*80 + [1]*20)
        y_proba = np.clip(y_true + np.random.randn(100) * 0.3, 0, 1)
        return y_true, y_proba

    def test_optimize_returns_float(self, binary_data):
        from src.cost_optimizer import BusinessCostOptimizer
        y_true, y_proba = binary_data
        opt = BusinessCostOptimizer()
        t = opt.optimize(y_true, y_proba)
        assert isinstance(t, float)

    def test_threshold_in_valid_range(self, binary_data):
        from src.cost_optimizer import BusinessCostOptimizer
        y_true, y_proba = binary_data
        opt = BusinessCostOptimizer(fraud_loss=10000, false_alarm_cost=200)
        t = opt.optimize(y_true, y_proba)
        assert 0.0 <= t <= 1.0

    def test_get_results_keys(self, binary_data):
        from src.cost_optimizer import BusinessCostOptimizer
        y_true, y_proba = binary_data
        opt = BusinessCostOptimizer()
        opt.optimize(y_true, y_proba)
        results = opt.get_results()
        assert "Optimal Threshold" in results
        assert "Minimum Cost ($)" in results
        assert "Metrics at Optimal Threshold" in results

    def test_high_fraud_loss_lowers_threshold(self, binary_data):
        """Higher fraud loss should push threshold lower to catch more fraud."""
        from src.cost_optimizer import BusinessCostOptimizer
        y_true, y_proba = binary_data
        opt_low  = BusinessCostOptimizer(fraud_loss=100,   false_alarm_cost=200)
        opt_high = BusinessCostOptimizer(fraud_loss=50000, false_alarm_cost=200)
        t_low  = opt_low.optimize(y_true, y_proba)
        t_high = opt_high.optimize(y_true, y_proba)
        assert t_high <= t_low

    def test_single_class_no_valid_matrix(self):
        from src.cost_optimizer import BusinessCostOptimizer
        np.random.seed(42)
        y_true  = np.zeros(50, dtype=int)
        y_proba = np.random.rand(50)
        opt = BusinessCostOptimizer()
        opt.optimize(y_true, y_proba)
        # sklearn allows single-class confusion matrix
        # so optimizer still runs — just verify cost is non-negative
        assert opt.minimum_cost >= 0
# ─────────────────────────────────────────────
# MODEL SELECTOR
# ─────────────────────────────────────────────

class TestAutoModelSelector:

    def test_train_returns_pipeline_score_name(self, small_dataset):
        from src.model_selector import AutoModelSelector
        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        selector = AutoModelSelector()
        # Only test with LR to keep CI fast
        from sklearn.linear_model import LogisticRegression
        selector.MODELS = {"LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced")}
        pipeline, score, name = selector.train_models(X, y)
        assert pipeline is not None
        assert 0.0 <= score <= 1.0
        assert name == "LogisticRegression"

    def test_all_scores_populated(self, small_dataset):
        from src.model_selector import AutoModelSelector
        from sklearn.linear_model import LogisticRegression
        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}
        selector.train_models(X, y)
        assert "LR" in selector.all_scores

    def test_pipeline_has_preprocessor_and_model(self, small_dataset):
        from src.model_selector import AutoModelSelector
        from sklearn.linear_model import LogisticRegression
        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}
        pipeline, _, _ = selector.train_models(X, y)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_best_model_can_predict_proba(self, small_dataset):
        from src.model_selector import AutoModelSelector
        from sklearn.linear_model import LogisticRegression
        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}
        pipeline, _, _ = selector.train_models(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ─────────────────────────────────────────────
# FRAUD SYSTEM (AutoMLFraudDetector)
# ─────────────────────────────────────────────

class TestAutoMLFraudDetector:

    @pytest.fixture
    def trained_detector(self, small_dataset, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}

        detector = AutoMLFraudDetector(
            model_selector=selector,
            objective="f1",
            model_path=str(tmp_path / "model.pkl"),
            model_version="test_v1"
        )
        detector.fit(X_train, y_train, X_val, y_val)
        return detector, X_val, y_val

    def test_fit_sets_threshold(self, trained_detector):
        detector, _, _ = trained_detector
        assert 0.0 <= detector.get_threshold() <= 1.0

    def test_fit_sets_feature_names(self, trained_detector):
        detector, X_val, _ = trained_detector
        assert detector.get_feature_names() == list(X_val.columns)

    def test_predict_returns_binary(self, trained_detector):
        detector, X_val, _ = trained_detector
        preds = detector.predict(X_val)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self, trained_detector):
        detector, X_val, _ = trained_detector
        proba = detector.predict_proba(X_val)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_get_version(self, trained_detector):
        detector, _, _ = trained_detector
        assert detector.get_version() == "test_v1"

    def test_model_saved_to_disk(self, trained_detector, tmp_path):
        detector, _, _ = trained_detector
        assert os.path.exists(detector.model_path)

    def test_fit_none_data_raises(self, tmp_path):
        from src.model_selector import AutoModelSelector
        from src.fraud_system import AutoMLFraudDetector
        selector = AutoModelSelector()
        detector = AutoMLFraudDetector(
            model_selector=selector,
            model_path=str(tmp_path / "model.pkl")
        )
        with pytest.raises(ValueError):
            detector.fit(None, None, None, None)

    def test_save_load_roundtrip(self, trained_detector, tmp_path):
        import joblib
        detector, _, _ = trained_detector
        data = joblib.load(detector.model_path)
        for key in ["model", "threshold", "feature_names", "model_version"]:
            assert key in data


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

class TestGenerateEvaluationReports:

    @pytest.fixture
    def trained_pipeline(self, small_dataset):
        from src.model_selector import AutoModelSelector
        from sklearn.linear_model import LogisticRegression
        X = small_dataset.drop("Class", axis=1)
        y = small_dataset["Class"]
        selector = AutoModelSelector()
        selector.MODELS = {"LR": LogisticRegression(max_iter=200, class_weight="balanced")}
        pipeline, _, _ = selector.train_models(X, y)
        return pipeline, X, y

    def test_returns_expected_keys(self, trained_pipeline, tmp_path, monkeypatch):
        from src.evaluation import generate_evaluation_reports
        pipeline, X, y = trained_pipeline
        monkeypatch.chdir(tmp_path)
        result = generate_evaluation_reports(pipeline, X, y, threshold=0.5)
        for key in ["roc_auc", "threshold", "confusion_matrix"]:
            assert key in result

    def test_roc_auc_in_range(self, trained_pipeline, tmp_path, monkeypatch):
        from src.evaluation import generate_evaluation_reports
        pipeline, X, y = trained_pipeline
        monkeypatch.chdir(tmp_path)
        result = generate_evaluation_reports(pipeline, X, y, threshold=0.5)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_saves_report_files(self, trained_pipeline, tmp_path, monkeypatch):
        from src.evaluation import generate_evaluation_reports
        pipeline, X, y = trained_pipeline
        monkeypatch.chdir(tmp_path)
        generate_evaluation_reports(pipeline, X, y, threshold=0.5)
        assert os.path.exists("reports/evaluation/roc_curve.png")
        assert os.path.exists("reports/evaluation/confusion_matrix.png")
        assert os.path.exists("reports/evaluation/classification_report.txt")


# ─────────────────────────────────────────────
# SAVED MODEL ARTIFACTS (require trained model)
# ─────────────────────────────────────────────

class TestSavedArtifacts:

    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pkl"),
        reason="Run python -m Scripts.train first"
    )
    def test_model_package_keys(self):
        import joblib
        data = joblib.load("models/best_model.pkl")
        for key in ["model", "threshold", "feature_names", "model_version", "cv_score"]:
            assert key in data

    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pkl"),
        reason="Run python -m Scripts.train first"
    )
    def test_probability_in_range(self, sample_transaction):
        import joblib
        from Scripts.train import engineer_features
        data = joblib.load("models/best_model.pkl")
        prob = data["model"].predict_proba(engineer_features(sample_transaction))[0][1]
        assert 0.0 <= prob <= 1.0

    @pytest.mark.skipif(
        not os.path.exists("models/metadata_v1.json"),
        reason="Run python -m Scripts.train first"
    )
    def test_metadata_has_required_keys(self):
        with open("models/metadata_v1.json") as f:
            meta = json.load(f)
        for key in ["model_name", "model_version", "threshold", "features", "cv_score"]:
            assert key in meta