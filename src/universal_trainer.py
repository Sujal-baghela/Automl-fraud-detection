import os
import logging
import warnings
import numpy as np
import pandas as pd
import psutil

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# RAM UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def get_dataframe_ram_gb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e9


def check_ram_safety(df: pd.DataFrame) -> dict:
    """Check if dataset fits safely in RAM for training."""
    available = get_available_ram_gb()
    df_size   = get_dataframe_ram_gb(df)
    # Training needs ~4x the data size (copies, CV, model)
    needed    = df_size * 4
    safe      = needed < available * 0.8

    return {
        "available_gb": round(available, 2),
        "dataframe_gb": round(df_size, 3),
        "estimated_needed_gb": round(needed, 2),
        "is_safe": safe,
        "warning": None if safe else (
            f"Training may need ~{needed:.1f}GB RAM but only "
            f"{available:.1f}GB is available. Consider sampling."
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# DATASET PROFILER
# ──────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    """
    Auto-profiles any uploaded dataset.
    Detects column types, class balance, missing values, and data issues.
    """

    def profile(self, df: pd.DataFrame, target_col: str) -> dict:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        class_counts  = y.value_counts()
        n_classes     = len(class_counts)
        minority_ratio = class_counts.min() / len(y)

        missing_cols = X.columns[X.isnull().any()].tolist()
        missing_pct  = round(X.isnull().sum().sum() / X.size * 100, 2)

        # Detect high cardinality categoricals
        high_card_cols = [
            c for c in cat_cols
            if X[c].nunique() > 50
        ]

        profile = {
            "n_rows":          len(df),
            "n_cols":          len(X.columns),
            "n_numeric":       len(num_cols),
            "n_categorical":   len(cat_cols),
            "numeric_cols":    num_cols,
            "categorical_cols": cat_cols,
            "high_cardinality_cols": high_card_cols,
            "target_col":      target_col,
            "n_classes":       n_classes,
            "class_counts":    class_counts.to_dict(),
            "minority_ratio":  round(float(minority_ratio), 6),
            "is_imbalanced":   minority_ratio < 0.2,
            "missing_cols":    missing_cols,
            "missing_pct":     missing_pct,
            "has_missing":     len(missing_cols) > 0,
            "size_gb":         round(get_dataframe_ram_gb(df), 3),
            "needs_sampling":  len(df) > 500_000,
        }

        logger.info("Dataset profiled: %d rows | %d cols | %d classes | %.1f%% minority",
                    len(df), len(X.columns), n_classes, minority_ratio * 100)

        return profile


# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSAL PREPROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build a preprocessor for any mix of numeric and categorical columns."""
    transformers = []

    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False,
                                      max_categories=50))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise ValueError("No numeric or categorical columns found.")

    return ColumnTransformer(transformers=transformers)


# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSAL MODEL SELECTOR
# ──────────────────────────────────────────────────────────────────────────────

def get_models(is_imbalanced: bool, n_rows: int) -> dict:
    """
    Return appropriate models based on dataset characteristics.
    Uses class_weight for large/imbalanced datasets instead of SMOTE.
    """
    use_balanced = is_imbalanced

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced" if use_balanced else None,
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced" if use_balanced else None,
            n_jobs=-1,
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            is_unbalance=use_balanced,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            verbosity=0
        ),
    }

    # For very large datasets, reduce estimators for speed
    if n_rows > 200_000:
        models["RandomForest"].n_estimators = 100
        models["LightGBM"].n_estimators     = 200
        models["XGBoost"].n_estimators      = 100
        logger.info("Large dataset detected — reducing n_estimators for speed.")

    return models


# ──────────────────────────────────────────────────────────────────────────────
# UNIVERSAL TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class UniversalTrainer:
    """
    Train any binary classification dataset end-to-end.

    Usage:
        trainer = UniversalTrainer()
        result  = trainer.fit(df, target_col="Churn", positive_label="Yes")
        trainer.save("models/universal_model.pkl")
    """

    def __init__(self, model_save_path: str = "models/universal_model.pkl"):
        self.model_save_path  = model_save_path
        self.best_pipeline    = None
        self.best_model_name  = None
        self.best_score       = None
        self.threshold        = 0.5
        self.feature_names    = None
        self.target_col       = None
        self.positive_label   = None
        self.label_encoder    = None
        self.profile          = None
        self.all_scores       = {}
        self.metrics          = {}

    # ── Encode target to 0/1 ──────────────────────────────────────────────────
    def _encode_target(self, y: pd.Series, positive_label) -> np.ndarray:
        unique_vals = y.unique()

        # Already 0/1
        if set(unique_vals).issubset({0, 1}):
            return y.values.astype(int)

        # Binary with known positive label
        if positive_label is not None:
            encoded = (y == positive_label).astype(int)
            logger.info("Target encoded: '%s' → 1, others → 0", positive_label)
            return encoded.values

        # Binary — encode with LabelEncoder
        le = LabelEncoder()
        encoded = le.fit_transform(y)
        self.label_encoder = le
        logger.info("Target label-encoded: %s", dict(zip(le.classes_, le.transform(le.classes_))))
        return encoded

    # ── Sample large datasets ─────────────────────────────────────────────────
    def _maybe_sample(self, X: pd.DataFrame, y: np.ndarray,
                      max_rows: int = 500_000) -> tuple:
        if len(X) <= max_rows:
            return X, y

        logger.info("Sampling %d rows from %d for training.", max_rows, len(X))
        idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
        return X.iloc[idx].reset_index(drop=True), y[idx]

    # ── Main fit method ───────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        positive_label=None,
        test_size: float = 0.2,
        sample_if_large: bool = True,
        progress_callback=None   # optional callable(step, total, message)
    ) -> dict:

        def _progress(step, total, msg):
            logger.info("[%d/%d] %s", step, total, msg)
            if progress_callback:
                progress_callback(step, total, msg)

        total_steps = 6

        # ── Step 1: Profile ───────────────────────────────────────────────────
        _progress(1, total_steps, "Profiling dataset...")
        profiler     = DatasetProfiler()
        self.profile = profiler.profile(df, target_col)
        self.target_col    = target_col
        self.positive_label = positive_label

        if self.profile["n_classes"] != 2:
            raise ValueError(
                f"UniversalTrainer supports binary classification only. "
                f"Found {self.profile['n_classes']} classes: "
                f"{list(self.profile['class_counts'].keys())}"
            )

        # ── Step 2: Prepare data ──────────────────────────────────────────────
        _progress(2, total_steps, "Preparing data...")
        X = df.drop(columns=[target_col])
        y = self._encode_target(df[target_col], positive_label)

        # Drop high cardinality columns (>50 unique values) with warning
        high_card = self.profile["high_cardinality_cols"]
        if high_card:
            logger.warning("Dropping high-cardinality columns: %s", high_card)
            X = X.drop(columns=high_card, errors="ignore")

        self.feature_names = list(X.columns)
        num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

        # Sample if large
        if sample_if_large and self.profile["needs_sampling"]:
            X, y = self._maybe_sample(X, y)
            logger.info("Sampled to %d rows for training.", len(X))

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info("Split: train=%d | val=%d", len(X_train), len(X_val))

        # ── Step 3: Build preprocessor ────────────────────────────────────────
        _progress(3, total_steps, "Building preprocessor...")
        preprocessor = build_preprocessor(num_cols, cat_cols)

        # ── Step 4: Train models ──────────────────────────────────────────────
        _progress(4, total_steps, "Training models via cross-validation...")
        from sklearn.model_selection import cross_val_score

        models    = get_models(self.profile["is_imbalanced"], len(X_train))
        cv_folds  = 2 if len(X_train) > 200_000 else 3
        best_score, best_pipeline, best_name = -np.inf, None, None

        for name, model in models.items():
            logger.info("Training: %s", name)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model",        model)
            ])
            try:
                score = float(np.mean(cross_val_score(
                    pipeline, X_train, y_train,
                    cv=cv_folds, scoring="roc_auc", n_jobs=-1
                )))
            except Exception as e:
                logger.warning("Model %s failed CV: %s", name, e)
                score = 0.0

            logger.info("%s CV ROC-AUC: %.5f", name, score)
            self.all_scores[name] = round(score, 5)

            if score > best_score:
                best_score, best_pipeline, best_name = score, pipeline, name

        logger.info("Best model: %s (%.5f)", best_name, best_score)
        best_pipeline.fit(X_train, y_train)

        self.best_pipeline   = best_pipeline
        self.best_model_name = best_name
        self.best_score      = best_score

        # ── Step 5: Optimize threshold ────────────────────────────────────────
        _progress(5, total_steps, "Optimizing decision threshold...")
        from sklearn.metrics import precision_recall_curve

        y_proba_val = best_pipeline.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
        precision = precision[:-1]
        recall    = recall[:-1]
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

        if len(thresholds) > 0:
            self.threshold = float(thresholds[np.argmax(f1_scores)])
        else:
            self.threshold = 0.5

        logger.info("Optimal threshold (F1): %.5f", self.threshold)

        # ── Step 6: Evaluate ──────────────────────────────────────────────────
        _progress(6, total_steps, "Evaluating model...")
        y_pred = (y_proba_val >= self.threshold).astype(int)

        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = (cm.ravel() if cm.shape == (2,2) else (0, 0, 0, 0))

        self.metrics = {
            "best_model":   best_name,
            "cv_roc_auc":   round(best_score, 5),
            "test_roc_auc": round(roc_auc_score(y_val, y_proba_val), 5),
            "f1_score":     round(f1_score(y_val, y_pred), 5),
            "recall":       round(recall_score(y_val, y_pred), 5),
            "precision":    round(precision_score(y_val, y_pred), 5),
            "threshold":    round(self.threshold, 5),
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
            "all_cv_scores": self.all_scores,
        }

        logger.info("Results: ROC-AUC=%.5f | F1=%.5f | Recall=%.5f",
                    self.metrics["test_roc_auc"],
                    self.metrics["f1_score"],
                    self.metrics["recall"])

        # Save model
        self.save()

        return self.metrics

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return self.best_pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str = None):
        path = path or self.model_save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        package = {
            "pipeline":       self.best_pipeline,
            "threshold":      self.threshold,
            "feature_names":  self.feature_names,
            "target_col":     self.target_col,
            "positive_label": self.positive_label,
            "best_model_name": self.best_model_name,
            "metrics":        self.metrics,
            "profile":        self.profile,
            "all_scores":     self.all_scores,
            "label_encoder":  self.label_encoder,
        }
        joblib.dump(package, path)
        logger.info("Universal model saved to %s", path)

    def load(self, path: str = None):
        path = path or self.model_save_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        package = joblib.load(path)
        self.best_pipeline    = package["pipeline"]
        self.threshold        = package["threshold"]
        self.feature_names    = package["feature_names"]
        self.target_col       = package["target_col"]
        self.positive_label   = package["positive_label"]
        self.best_model_name  = package["best_model_name"]
        self.metrics          = package["metrics"]
        self.profile          = package["profile"]
        self.all_scores       = package["all_scores"]
        self.label_encoder    = package.get("label_encoder")
        logger.info("Universal model loaded from %s", path)
        return self