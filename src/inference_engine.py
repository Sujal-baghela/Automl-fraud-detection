import joblib
import json
import logging
import os
import pandas as pd

from src.shap_explainer import IntelligentSHAP


logger = logging.getLogger(__name__)


class FraudInferenceEngine:
    """
    Production-grade inference engine.

    Responsibilities:
    - Load trained model
    - Load metadata
    - Validate input schema
    - Perform prediction
    - Generate SHAP explanation
    - Structured logging
    """

    def __init__(
        self,
        model_path="models/best_model.pkl",
        metadata_path="models/metadata_v1.json"
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading trained model...")
        model_data = joblib.load(model_path)

        self.pipeline = model_data["model"]
        self.threshold = model_data["threshold"]
        self.objective = model_data.get("objective")
        self.cv_score = model_data.get("cv_score")

        logger.info("Model loaded successfully.")

        # -----------------------------
        # Load Metadata (Preferred)
        # -----------------------------
        if os.path.exists(metadata_path):
            logger.info("Loading metadata file...")
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

            self.expected_features = self.metadata["features"]

        else:
            logger.warning("Metadata file not found. Falling back to model data.")
            self.expected_features = model_data.get("features")

            self.metadata = {
                "threshold": self.threshold,
                "objective": self.objective,
                "cv_score": self.cv_score,
                "features": self.expected_features
            }

        if not self.expected_features:
            raise ValueError("Feature list missing. Model metadata corrupted.")

        logger.info(f"Expected feature count: {len(self.expected_features)}")

        # -----------------------------
        # Initialize SHAP Once
        # -----------------------------
        logger.info("Initializing SHAP engine...")
        self.shap_engine = IntelligentSHAP(self.pipeline)
        logger.info("Inference engine ready.")

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    def _validate_and_prepare_input(self, input_dict):

        if not isinstance(input_dict, dict):
            raise ValueError("Input must be a dictionary of feature_name: value.")

        input_features = set(input_dict.keys())
        expected_features = set(self.expected_features)

        # Missing feature check
        missing = expected_features - input_features
        if missing:
            logger.error(f"Missing features detected: {missing}")
            raise ValueError(f"Missing features: {missing}")

        # Remove unexpected features
        extra = input_features - expected_features
        if extra:
            logger.warning(f"Ignoring unexpected features: {extra}")

        # Keep only expected features
        filtered_input = {
            feature: input_dict[feature]
            for feature in self.expected_features
        }

        # Build ordered DataFrame
        df = pd.DataFrame([filtered_input])

        # Enforce column order
        df = df[self.expected_features]

        return df

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, input_dict):

        logger.info("Prediction request received.")

        try:
            df = self._validate_and_prepare_input(input_dict)

            prob = self.pipeline.predict_proba(df)[:, 1][0]
            prediction = int(prob >= self.threshold)

            logger.info(
                f"Prediction completed. Probability={prob:.5f}, Threshold={self.threshold}"
            )

            explanation = self.shap_engine.local_explanation_json(df, index=0)

            return {
                "model_info": {
                    "objective": self.objective,
                    "cv_score": self.cv_score,
                },
                "prediction_result": {
                    "probability": float(prob),
                    "threshold_used": float(self.threshold),
                    "predicted_class": prediction
                },
                "explanation": explanation
            }

        except Exception as e:
            logger.exception("Error during prediction.")
            raise e

    # ======================================================
    # GET METADATA (API use)
    # ======================================================

    def get_metadata(self):
        return self.metadata