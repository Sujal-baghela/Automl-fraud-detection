import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_selector import AutoModelSelector
from src.fraud_system import AutoMLFraudDetector
from src.data_loader import DataLoader
from src.evaluation import generate_evaluation_reports
from src.shap_explainer import IntelligentSHAP


# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------- MAIN TRAIN FUNCTION ----------------
def main():

    try:
        logger.info("Starting training pipeline...")

        # ---------- Load Dataset via DataLoader ----------
        data_path = "Data/creditcard.csv"

        loader = DataLoader(file_path=data_path)
        df = loader.load_data()

        logger.info(f"Dataset shape: {df.shape}")

        metadata_summary = loader.get_metadata(df)
        logger.info(f"Dataset metadata: {metadata_summary}")

        # ---------- Split Features / Target ----------
        X, y = loader.split_features_target(df, target_column="Class")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Train-validation split completed.")
        logger.info(f"X_train shape: {X_train.shape} | X_val shape: {X_val.shape}")

        # ---------- Initialize Selector & Detector ----------
        selector = AutoModelSelector()

        detector = AutoMLFraudDetector(
            model_selector=selector,
            objective="cost",           # Uses BusinessCostOptimizer
            fraud_loss=10000,
            false_alarm_cost=200,
            model_path="models/best_model.pkl",
            model_version="v1"
        )

        # ---------- Train + Optimize Threshold + Save ----------
        detector.fit(X_train, y_train, X_val, y_val)

        logger.info("Model training and threshold optimization completed.")

        # ---------- Evaluate ----------
        detector.evaluate(X_val, y_val)

        # ---------- Generate Evaluation Reports ----------
        best_pipeline = detector.get_model()

        generate_evaluation_reports(
            pipeline=best_pipeline,
            X_test=X_val,
            y_test=y_val,
            threshold=detector.get_threshold()
        )

        # ---------- Generate Global SHAP ----------
        logger.info("Generating global SHAP explanation...")
        shap_engine = IntelligentSHAP(best_pipeline)
        shap_engine.global_explanation(X_train)
        logger.info("SHAP explanation completed.")

        # ---------- Save Metadata ----------
        os.makedirs("models", exist_ok=True)

        metadata = {
            "model_name": type(best_pipeline.named_steps["model"]).__name__,
            "model_version": detector.get_version(),
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_shape": list(df.shape),
            "features": detector.get_feature_names(),
            "threshold": detector.get_threshold(),
            "objective": detector.objective,
            "cv_score": detector.cv_score
        }

        metadata_path = "models/metadata_v1.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Metadata saved to {metadata_path}")
        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.exception("Training failed.")
        raise e


if __name__ == "__main__":
    main()
