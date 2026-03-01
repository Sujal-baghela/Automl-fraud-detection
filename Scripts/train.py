import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_selector import AutoModelSelector
from src.fraud_system import AutoMLFraudDetector
from src.data_loader import DataLoader
from src.evaluation import generate_evaluation_reports
from src.shap_explainer import IntelligentSHAP


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ================================================================
# FEATURE ENGINEERING
# ================================================================
# Adds 5 derived features from Time and Amount.
# V1-V28 are already PCA-transformed and cannot be engineered.
#
# Hour          — fraud rates vary by hour of day
# Night_txn     — binary flag: transactions between 10pm-6am
# Amount_log    — log(1+Amount) compresses right skew
# Amount_zscore — how unusual this amount is (std deviations)
# High_amount   — binary flag: Amount > $1,000
#
# Applied before train/val split so statistics are consistent.
# ================================================================
def engineer_features(df):
    df = df.copy()

    seconds_in_day      = 3600 * 24
    df["Hour"]          = (df["Time"] % seconds_in_day) / 3600
    df["Night_txn"]     = ((df["Hour"] < 6) | (df["Hour"] > 22)).astype(int)
    df["Amount_log"]    = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["High_amount"]   = (df["Amount"] > 1000).astype(int)

    return df


def main():
    try:
        logger.info("Starting training pipeline...")

        # ---------- Load ----------
        loader = DataLoader(file_path="Data/creditcard.csv")
        df     = loader.load_data()
        logger.info(f"Dataset shape: {df.shape}")

        # ---------- Feature Engineering ----------
        logger.info("Applying feature engineering...")
        df = engineer_features(df)
        logger.info(
            f"Features after engineering: {df.shape[1] - 1} "
            f"(+5 new: Hour, Night_txn, Amount_log, Amount_zscore, High_amount)"
        )

        # ---------- Split ----------
        X, y = loader.split_features_target(df, target_column="Class")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"X_train: {X_train.shape} | X_val: {X_val.shape}")

        # ---------- SMOTE ----------
        # Standard SMOTE — produced best baseline in Run 1
        # (90.8% recall, $108,400 cost).
        # Now combined with 35 features — RandomForest gets
        # 5 new signals it didn't have in Run 1.
        logger.info("Applying SMOTE to training data...")
        logger.info(f"BEFORE:\n{pd.Series(y_train).value_counts().to_string()}")

        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
        y_train_res = pd.Series(y_train_res, name=y_train.name)

        logger.info(f"AFTER:\n{pd.Series(y_train_res).value_counts().to_string()}")
        logger.info(f"Resampled shape: {X_train_res.shape}")

        # ---------- Train ----------
        selector = AutoModelSelector()

        detector = AutoMLFraudDetector(
            model_selector=selector,
            objective="cost",
            fraud_loss=10000,
            false_alarm_cost=200,
            model_path="models/best_model.pkl",
            model_version="v5"          # final version
        )

        detector.fit(X_train_res, y_train_res, X_val, y_val)
        logger.info("Training and threshold optimization completed.")

        detector.evaluate(X_val, y_val)

        # ---------- Reports ----------
        best_pipeline = detector.get_model()

        generate_evaluation_reports(
            pipeline=best_pipeline,
            X_test=X_val,
            y_test=y_val,
            threshold=detector.get_threshold()
        )

        # ---------- SHAP ----------
        logger.info("Generating SHAP explanation...")
        IntelligentSHAP(best_pipeline).global_explanation(X_train)
        logger.info("SHAP completed.")

        # ---------- Metadata ----------
        os.makedirs("models", exist_ok=True)

        metadata = {
            "model_name":            type(best_pipeline.named_steps["model"]).__name__,
            "model_version":         detector.get_version(),
            "training_date":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_shape":         list(df.shape),
            "original_train_shape":  list(X_train.shape),
            "resampled_train_shape": list(X_train_res.shape),
            "features":              detector.get_feature_names(),
            "feature_engineering": [
                "Hour", "Night_txn",
                "Amount_log", "Amount_zscore", "High_amount"
            ],
            "threshold":             detector.get_threshold(),
            "objective":             detector.objective,
            "cv_score":              detector.cv_score,
            "fraud_loss":            10000,
            "false_alarm_cost":      200,
            "imbalance_handling": {
                "technique":             "SMOTE",
                "k_neighbors":           5,
                "original_fraud_ratio":  round(float(y_train.sum()) / len(y_train), 6),
                "resampled_fraud_ratio": round(float(y_train_res.sum()) / len(y_train_res), 6),
            }
        }

        with open("models/metadata_v1.json", "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Metadata saved. Pipeline completed successfully.")

    except Exception as e:
        logger.exception("Training failed.")
        raise e


if __name__ == "__main__":
    main()