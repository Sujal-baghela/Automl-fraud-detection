import os
import joblib
import numpy as np
import datetime
import logging

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)


logger = logging.getLogger(__name__)


class AutoMLFraudDetector:

    def __init__(
        self,
        model_selector,
        objective="cost",
        fraud_loss=10000,
        false_alarm_cost=200,
        manual_threshold=None,
        model_path="models/best_model.pkl",
        model_version="v1",
        auto_load=False
    ):
        self.model_selector = model_selector
        self.objective = objective
        self.fraud_loss = fraud_loss
        self.false_alarm_cost = false_alarm_cost
        self.manual_threshold = manual_threshold
        self.model_path = model_path
        self.model_version = model_version

        self.best_model = None
        self.threshold = 0.5
        self.cv_score = None
        self.feature_names = None

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if auto_load and os.path.exists(self.model_path):
            self.load()

    # ======================================================
    # TRAIN SYSTEM
    # ======================================================
    def fit(self, X_train, y_train, X_val, y_val):

        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None.")

        logger.info("Training started at %s", datetime.datetime.now())

        # Save feature names
        self.feature_names = list(X_train.columns)

        # 1️⃣ Train models using CV
        # FIX: train_models() returns (pipeline, score, name) — unpack all three
        self.best_model, self.cv_score, best_name = self.model_selector.train_models(
            X_train, y_train
        )

        logger.info("Best model selected: %s", best_name)

        if not hasattr(self.best_model, "predict_proba"):
            raise ValueError("Selected model does not support predict_proba.")

        # 2️⃣ Predict on validation
        y_proba_val = self.best_model.predict_proba(X_val)[:, 1]

        # 3️⃣ Threshold Selection
        if self.manual_threshold is not None:
            self.threshold = self.manual_threshold
            logger.info("Using manual threshold: %.5f", self.threshold)

        else:
            if self.objective == "cost":
                from src.cost_optimizer import BusinessCostOptimizer

                optimizer = BusinessCostOptimizer(
                    fraud_loss=self.fraud_loss,
                    false_alarm_cost=self.false_alarm_cost
                )
                self.threshold = optimizer.optimize(y_val, y_proba_val)
                logger.info("Cost-optimized threshold: %.5f", self.threshold)
                logger.info("Cost optimizer results: %s", optimizer.get_results())

            else:
                from src.threshold_optimizer import ThresholdOptimizer

                optimizer = ThresholdOptimizer(
                    strategy=f"maximize_{self.objective}"
                )
                self.threshold = optimizer.optimize(y_val, y_proba_val)
                logger.info(
                    "Metric-optimized threshold (%s): %.5f",
                    self.objective, self.threshold
                )

        # 4️⃣ Save system — structured package matching inference_engine expectations
        model_package = {
            "model": self.best_model,
            "threshold": float(self.threshold),
            "objective": self.objective,
            "cv_score": float(self.cv_score) if self.cv_score else None,
            "fraud_loss": self.fraud_loss,
            "false_alarm_cost": self.false_alarm_cost,
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "features": self.feature_names,   # alias for inference_engine compatibility
            "timestamp": str(datetime.datetime.now())
        }

        joblib.dump(model_package, self.model_path)

        logger.info("Model trained & saved successfully to %s", self.model_path)
        logger.info("Selected Threshold: %.5f", self.threshold)
        logger.info("CV ROC-AUC: %.5f", self.cv_score)

    # ======================================================
    # EVALUATE SYSTEM
    # ======================================================
    def evaluate(self, X_test, y_test):

        if self.best_model is None:
            raise ValueError("Model not trained or loaded.")

        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        logger.info("===== FINAL TEST RESULTS =====")
        logger.info("Objective: %s", self.objective)
        logger.info("CV ROC-AUC: %.5f", self.cv_score)
        logger.info("Threshold Used: %.5f", self.threshold)

        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
            logger.info("Test ROC-AUC: %.5f", roc_auc)
        else:
            logger.warning("Test ROC-AUC not defined (single class present).")

        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
        logger.info(
            "Classification Report:\n%s",
            classification_report(y_test, y_pred)
        )

    # ======================================================
    # PREDICT PROBABILITY
    # ======================================================
    def predict_proba(self, X):

        if self.best_model is None:
            raise ValueError("Model not trained or loaded.")

        return self.best_model.predict_proba(X)[:, 1]

    # ======================================================
    # PREDICT CLASS (uses optimized threshold)
    # ======================================================
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    # ======================================================
    # LOAD SAVED SYSTEM
    # ======================================================
    def load(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Saved model not found at: {self.model_path}")

        data = joblib.load(self.model_path)

        self.best_model = data["model"]
        self.threshold = data["threshold"]
        self.objective = data.get("objective")
        self.cv_score = data.get("cv_score")
        self.feature_names = data.get("feature_names") or data.get("features")
        self.model_version = data.get("model_version", "unknown")

        logger.info("Model loaded successfully from %s", self.model_path)
        logger.info("Threshold: %.5f", self.threshold)
        logger.info("Model version: %s", self.model_version)

    # ======================================================
    # ACCESSORS FOR SHAP / INFERENCE / TRAIN SCRIPT
    # ======================================================
    def get_model(self):
        return self.best_model

    def get_feature_names(self):
        return self.feature_names

    def get_threshold(self):
        return self.threshold

    def get_version(self):
        return self.model_version
