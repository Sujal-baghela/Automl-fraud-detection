import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_selector import AutoModelSelector
from src.fraud_system import AutoMLFraudDetector
from src.data_loader import DataLoader
from src.cleaner import DataCleaner          # FIX: was imported but never used before
from src.evaluation import generate_evaluation_reports
from src.shap_explainer import IntelligentSHAP
from src.drift_detector import DriftDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 5 domain-aware features derived from Time and Amount.
    Always operates on a copy to avoid mutating the caller's DataFrame.
    """
    df = df.copy()   # FIX: never mutate the caller's DataFrame
    seconds_in_day      = 3600 * 24
    df["Hour"]          = (df["Time"] % seconds_in_day) / 3600
    df["Night_txn"]     = ((df["Hour"] < 6) | (df["Hour"] > 22)).astype(int)
    df["Amount_log"]    = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["High_amount"]   = (df["Amount"] > 1000).astype(int)
    return df


def main():
    try:
        # ── MLflow Setup ───────────────────────────────────────────────────────
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("fraud-detection")

        logger.info("Starting training pipeline...")

        # ── Load Dataset ───────────────────────────────────────────────────────
        loader = DataLoader(file_path="Data/creditcard.csv")
        df     = loader.load_data()
        logger.info("Dataset shape after load: %s", df.shape)

        # ── Clean Data ─────────────────────────────────────────────────────────
        # FIX: DataCleaner was imported but clean() was never called.
        # Must run BEFORE train/test split to avoid data leakage.
        cleaner = DataCleaner()
        df      = cleaner.clean(df)
        logger.info("Dataset shape after cleaning: %s", df.shape)

        # ── Feature Engineering ────────────────────────────────────────────────
        logger.info("Applying feature engineering...")
        df = engineer_features(df)
        logger.info("Shape after engineering: %s", df.shape)

        X, y = loader.split_features_target(df, target_column="Class")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("X_train: %s | X_val: %s", X_train.shape, X_val.shape)

        # ── SMOTE ──────────────────────────────────────────────────────────────
        logger.info("Applying SMOTE...")
        logger.info("BEFORE:\n%s", pd.Series(y_train).value_counts().to_string())

        resampler                    = SMOTE(random_state=42)
        X_train_res, y_train_res     = resampler.fit_resample(X_train, y_train)
        X_train_res                  = pd.DataFrame(X_train_res, columns=X_train.columns)
        y_train_res                  = pd.Series(y_train_res, name=y_train.name)

        logger.info("AFTER:\n%s", pd.Series(y_train_res).value_counts().to_string())
        logger.info("Resampled shape: %s", X_train_res.shape)

        # ── MLflow Run ─────────────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"automl-v5-{datetime.now().strftime('%H%M%S')}"):

            mlflow.log_params({
                "dataset_rows":     len(df),
                "features":         X_train.shape[1],
                "imbalance_method": "SMOTE",
                "fraud_loss":       10000,
                "false_alarm_cost": 200,
                "model_version":    "v5",
                "test_size":        0.2,
                "random_state":     42,
            })

            # ── Train ──────────────────────────────────────────────────────────
            selector = AutoModelSelector()

            detector = AutoMLFraudDetector(
                model_selector=selector,
                objective="cost",
                fraud_loss=10000,
                false_alarm_cost=200,
                model_path="models/best_model.pkl",
                model_version="v5"
            )

            detector.fit(X_train_res, y_train_res, X_val, y_val)
            logger.info("Training and threshold optimization completed.")

            detector.evaluate(X_val, y_val)

            # ── Log CV scores for all models ───────────────────────────────────
            if hasattr(selector, "all_scores"):
                for model_name, score in selector.all_scores.items():
                    mlflow.log_metric(f"cv_roc_auc_{model_name}", score)

            # ── Final Metrics ──────────────────────────────────────────────────
            from sklearn.metrics import (
                roc_auc_score, recall_score, precision_score,
                f1_score, confusion_matrix
            )

            best_pipeline = detector.get_model()
            threshold     = detector.get_threshold()

            y_prob = best_pipeline.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            missed_frauds = int(fn)
            false_alarms  = int(fp)
            business_cost = missed_frauds * 10000 + false_alarms * 200

            mlflow.log_metrics({
                "test_roc_auc":   round(roc_auc_score(y_val, y_prob), 5),
                "cv_roc_auc":     round(detector.cv_score, 5),
                "recall":         round(recall_score(y_val, y_pred), 5),
                "precision":      round(precision_score(y_val, y_pred), 5),
                "f1_score":       round(f1_score(y_val, y_pred), 5),
                "threshold":      round(threshold, 5),
                "missed_frauds":  missed_frauds,
                "false_alarms":   false_alarms,
                "business_cost":  business_cost,
                "true_positives": int(tp),
                "true_negatives": int(tn),
            })

            mlflow.set_tags({
                "best_model":          selector.best_model_name,
                "feature_count":       str(X_train.shape[1]),
                "smote_applied":       "True",
                "feature_engineering": "True",
            })

            logger.info(
                "MLflow metrics logged — Cost: $%s | Recall: %.3f",
                f"{business_cost:,}",
                recall_score(y_val, y_pred)
            )

            # ── Evaluation Reports ─────────────────────────────────────────────
            generate_evaluation_reports(
                pipeline=best_pipeline,
                X_test=X_val,
                y_test=y_val,
                threshold=threshold
            )

            for path, folder in [
                ("reports/evaluation/roc_curve.png",        "plots"),
                ("reports/evaluation/confusion_matrix.png", "plots"),
                ("reports/shap/feature_importance_bar.png", "plots"),
                ("reports/shap/global_summary.png",         "plots"),
            ]:
                if os.path.exists(path):
                    mlflow.log_artifact(path, folder)

            # ── SHAP ───────────────────────────────────────────────────────────
            logger.info("Generating SHAP explanation...")
            IntelligentSHAP(best_pipeline).global_explanation(X_train_res)
            logger.info("SHAP completed.")

            # ── Log Model ──────────────────────────────────────────────────────
            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path="model",
                registered_model_name="automl-fraud-detector"
            )

            # ── Drift Detector ─────────────────────────────────────────────────
            logger.info("Fitting drift detector on training data...")
            drift_detector = DriftDetector(psi_threshold=0.2, ks_alpha=0.05)
            drift_detector.fit(X_train_res)
            drift_detector.save("models/drift_reference.json")

            drift_report = drift_detector.detect(X_val)
            drift_detector.print_report(drift_report)

            mlflow.log_metrics({
                "drift_ratio":      drift_report["drift_ratio"],
                "drifted_features": drift_report["drifted_count"],
            })
            mlflow.log_artifact("models/drift_reference.json", "drift")
            logger.info("Drift detector ready.")

            # ── Metadata ───────────────────────────────────────────────────────
            os.makedirs("models", exist_ok=True)

            metadata = {
                "model_name":            type(best_pipeline.named_steps["model"]).__name__,
                "model_version":         detector.get_version(),
                "training_date":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_shape":         list(df.shape),
                "original_train_shape":  list(X_train.shape),
                "resampled_train_shape": list(X_train_res.shape),
                "features":              detector.get_feature_names(),
                "threshold":             threshold,
                "objective":             detector.objective,
                "cv_score":              detector.cv_score,
                "business_cost":         business_cost,
                "missed_frauds":         missed_frauds,
                "false_alarms":          false_alarms,
                "imbalance_handling": {
                    "technique":             "SMOTE",
                    "original_fraud_ratio":  round(float(y_train.sum()) / len(y_train), 6),
                    "resampled_fraud_ratio": round(float(y_train_res.sum()) / len(y_train_res), 6),
                }
            }

            with open("models/metadata_v1.json", "w") as f:
                json.dump(metadata, f, indent=4)

            mlflow.log_artifact("models/metadata_v1.json", "metadata")

            logger.info("Metadata saved. Pipeline completed successfully.")
            logger.info("MLflow run complete. View UI: mlflow ui --port 5000")

    except Exception as e:
        logger.exception("Training failed.")
        raise e


if __name__ == "__main__":
    main()