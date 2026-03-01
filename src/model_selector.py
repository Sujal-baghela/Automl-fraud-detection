import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# NamedColumnTransformer
# Always returns a named DataFrame instead of a numpy array.
# Required for LightGBM and XGBoost to receive named feature
# columns at every stage of the pipeline.
# ──────────────────────────────────────────────────────────────
class NamedColumnTransformer(ColumnTransformer):

    def transform(self, X):
        return pd.DataFrame(
            super().transform(X),
            columns=self.get_feature_names_out()
        )

    def fit_transform(self, X, y=None):
        return pd.DataFrame(
            super().fit_transform(X, y),
            columns=self.get_feature_names_out()
        )


class AutoModelSelector:

    # ──────────────────────────────────────────────────────────
    # 5 competing models — all tuned for imbalanced fraud data
    #
    # scale_pos_weight = 284315/492 = 578 (exact class ratio)
    # max_delta_step=1 — XGBoost official recommendation for
    # imbalanced classification, stabilizes gradient updates
    # ──────────────────────────────────────────────────────────
    MODELS = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        "LightGBM_Balanced": LGBMClassifier(
            n_estimators=500, scale_pos_weight=578,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "LightGBM_HighRecall": LGBMClassifier(
            n_estimators=500, is_unbalance=True,
            min_child_samples=5,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "XGBoost_HighRecall": XGBClassifier(
            n_estimators=300, scale_pos_weight=578,
            max_delta_step=1, random_state=42,
            n_jobs=-1, eval_metric="auc", verbosity=0
        ),
    }

    def __init__(self):
        self.best_model_name = None
        self.best_score      = None
        self.best_pipeline   = None
        self.all_scores      = {}   # stores every model's CV score for MLflow

    # ──────────────────────────────────────────────────────────
    # Build preprocessor — auto-detects column types
    # ──────────────────────────────────────────────────────────
    def _build_preprocessor(self, X):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        logger.info(f"Detected {len(num_cols)} numeric | {len(cat_cols)} categorical features")

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return NamedColumnTransformer(transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ])

    # ──────────────────────────────────────────────────────────
    # Train all models via CV, select best, refit on full
    # training data, return (pipeline, score, name)
    # ──────────────────────────────────────────────────────────
    def train_models(self, X_train, y_train):
        logger.info("Starting AutoML model selection...")

        preprocessor = self._build_preprocessor(X_train)
        cv_folds     = 2 if len(X_train) > 200000 else 3

        logger.info(f"Dataset: {len(X_train)} rows | CV folds: {cv_folds}")

        best_score, best_pipeline, best_name = -np.inf, None, None

        for name, model in self.MODELS.items():
            logger.info(f"Training: {name}")

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model",        model)
            ])

            score = float(np.mean(cross_val_score(
                pipeline, X_train, y_train,
                cv=cv_folds, scoring="roc_auc", n_jobs=-1
            )))

            logger.info(f"{name} CV ROC-AUC: {score:.5f}")

            # Store every score for MLflow logging
            self.all_scores[name] = round(score, 5)

            if score > best_score:
                best_score, best_pipeline, best_name = score, pipeline, name

        logger.info(f"Fitting best model on full data: {best_name}")
        best_pipeline.fit(X_train, y_train)

        self.best_model_name = best_name
        self.best_score      = best_score
        self.best_pipeline   = best_pipeline

        logger.info(f"Selected: {best_name} | CV ROC-AUC: {best_score:.5f}")

        return best_pipeline, best_score, best_name