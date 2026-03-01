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
    """
    Final production model selector.

    Models competing:
    - LogisticRegression  : fast baseline
    - RandomForest        : was best in Run 1 (90.8% recall)
                            now gets 35 features — may improve further
    - LightGBM_Balanced   : best in Run 4 (0.97564 test AUC)
    - LightGBM_HighRecall : LightGBM with native imbalance flag
    - XGBoost_HighRecall  : XGBoost with stable gradient updates

    Best model selected automatically by CV ROC-AUC.
    """

    MODELS = {
        # ── Baseline ───────────────────────────────────────────
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear"
        ),

        # ── RandomForest — Run 1 winner ────────────────────────
        # Upgraded from 200 to 500 trees.
        # Now receives 35 features instead of 30.
        # Feature engineering (Hour, Amount_zscore etc.) may
        # give it the signal needed to catch the 9 hard frauds.
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),

        # ── LightGBM variant 1 — Run 4 winner ─────────────────
        # scale_pos_weight=578 = 284315/492 exact class ratio
        "LightGBM_Balanced": LGBMClassifier(
            n_estimators=500,
            scale_pos_weight=578,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),

        # ── LightGBM variant 2 — internal imbalance flag ──────
        "LightGBM_HighRecall": LGBMClassifier(
            n_estimators=500,
            is_unbalance=True,
            min_child_samples=5,
            num_leaves=63,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),

        # ── XGBoost — stable imbalanced gradient updates ──────
        "XGBoost_HighRecall": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=578,
            min_child_weight=1,
            max_delta_step=1,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            verbosity=0
        ),
    }

    def __init__(self):
        self.best_model_name = None
        self.best_score      = None
        self.best_pipeline   = None

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

            logger.info(f"{name} ROC-AUC: {score:.5f}")

            if score > best_score:
                best_score, best_pipeline, best_name = score, pipeline, name

        logger.info(f"Fitting best model on full data: {best_name}")
        best_pipeline.fit(X_train, y_train)

        self.best_model_name = best_name
        self.best_score      = best_score
        self.best_pipeline   = best_pipeline

        logger.info(f"Selected: {best_name} | CV ROC-AUC: {best_score:.5f}")

        return best_pipeline, best_score, best_name