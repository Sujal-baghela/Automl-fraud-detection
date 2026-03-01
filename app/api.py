import joblib
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.shap_explainer import IntelligentSHAP


# ======================================
# Logging Configuration
# ======================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================
# Initialize FastAPI App
# ======================================

app = FastAPI(
    title="AutoML-X Fraud Detection API",
    version="2.0",
    description="Production-grade REST API for Fraud Prediction with SHAP Explainability"
)

# ======================================
# Load Model Package at Startup
# FIX: fraud_system.py saves a dict — load it as such
# ======================================

try:
    model_data = joblib.load("models/best_model.pkl")

    pipeline       = model_data["model"]
    threshold      = model_data["threshold"]
    objective      = model_data.get("objective", "cost")
    cv_score       = model_data.get("cv_score")
    model_version  = model_data.get("model_version", "unknown")

    # Support both key names for feature list
    feature_names = (
        model_data.get("feature_names") or
        model_data.get("features")
    )

    if not feature_names:
        raise ValueError("Feature list missing from saved model package.")

    logger.info("Model package loaded successfully.")
    logger.info("Threshold: %.5f | Objective: %s | Version: %s",
                threshold, objective, model_version)

except Exception as e:
    logger.error("Error loading model: %s", e)
    raise e


# ======================================
# SHAP Engine Setup
# FIX: use IntelligentSHAP wrapper instead of raw shap.TreeExplainer
# ======================================

try:
    shap_engine = IntelligentSHAP(pipeline)
    logger.info("SHAP engine initialized.")
except Exception as e:
    logger.warning("SHAP engine could not be initialized: %s", e)
    shap_engine = None


# ======================================
# Input Schema
# ======================================

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ======================================
# Health Check Route
# ======================================

@app.get("/")
def health_check():
    return {
        "status": "API is running successfully",
        "model_version": model_version,
        "objective": objective,
        "cv_score": round(cv_score, 5) if cv_score else None,
        "threshold": round(threshold, 5)
    }


# ======================================
# Model Info Route
# ======================================

@app.get("/model-info")
def model_info():
    return {
        "model_version": model_version,
        "objective": objective,
        "cv_score": round(cv_score, 5) if cv_score else None,
        "threshold": round(threshold, 5),
        "feature_count": len(feature_names),
        "features": feature_names
    }


# ======================================
# Prediction Endpoint
# FIX: uses optimized threshold, not sklearn default 0.5
# FIX: risk level relative to threshold, not hardcoded constants
# FIX: uses IntelligentSHAP for explanation
# ======================================

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert request to DataFrame
        input_dict = transaction.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Align columns to training feature order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # ==============================
        # Prediction using optimized threshold
        # ==============================
        probability = float(pipeline.predict_proba(input_df)[0][1])
        prediction = int(probability >= threshold)

        # ==============================
        # Risk Classification
        # FIX: risk levels relative to optimized threshold, not arbitrary constants
        # ==============================
        low_boundary    = threshold * 0.25
        medium_boundary = threshold * 0.75

        if probability < low_boundary:
            risk_level = "Low"
        elif probability < medium_boundary:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # ==============================
        # SHAP Explanation via IntelligentSHAP
        # ==============================
        explanation = {}

        if shap_engine is not None:
            try:
                explanation = shap_engine.local_explanation_json(
                    input_df, index=0, top_k=5
                )
            except Exception as shap_error:
                logger.warning("SHAP explanation failed: %s", shap_error)
                explanation = {"error": "SHAP explanation unavailable"}

        # ==============================
        # Response
        # ==============================
        return {
            "model_info": {
                "version": model_version,
                "objective": objective,
                "cv_score": round(cv_score, 5) if cv_score else None,
            },
            "prediction_result": {
                "fraud_probability_percent": round(probability * 100, 6),
                "threshold_used": round(threshold, 5),
                "predicted_class": prediction,
                "label": "Fraud" if prediction == 1 else "Legitimate",
                "risk_level": risk_level
            },
            "explanation": explanation
        }

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# Batch Prediction Endpoint
# ======================================

class BatchTransactions(BaseModel):
    transactions: list[Transaction]


@app.post("/predict/batch")
def predict_batch(batch: BatchTransactions):
    try:
        records = [t.model_dump() for t in batch.transactions]
        input_df = pd.DataFrame(records)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        probabilities = pipeline.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            prob = float(prob)
            low_boundary    = threshold * 0.25
            medium_boundary = threshold * 0.75

            if prob < low_boundary:
                risk_level = "Low"
            elif prob < medium_boundary:
                risk_level = "Medium"
            else:
                risk_level = "High"

            results.append({
                "index": i,
                "fraud_probability_percent": round(prob * 100, 6),
                "predicted_class": int(pred),
                "label": "Fraud" if pred == 1 else "Legitimate",
                "risk_level": risk_level
            })

        return {
            "threshold_used": round(threshold, 5),
            "total_transactions": len(results),
            "fraud_count": int(sum(predictions)),
            "results": results
        }

    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
