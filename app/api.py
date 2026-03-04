import io
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.alerting import AlertManager
from src.monitor import ModelMonitor
from src.shap_explainer import IntelligentSHAP
from src.universal_trainer import UniversalTrainer

# ======================================
# Logging
# ======================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================
# App
# ======================================

app = FastAPI(
    title="AutoML-X API",
    version="3.0",
    description=(
        "Production-grade REST API — "
        "Fraud Detection (fixed schema) + "
        "Universal Trainer (any binary CSV)"
    ),
)

# ======================================
# ── FRAUD MODEL ───────────────────────
# Load best_model.pkl once at startup
# ======================================

try:
    model_data = joblib.load("models/best_model.pkl")

    pipeline      = model_data["model"]
    threshold     = model_data["threshold"]
    objective     = model_data.get("objective", "cost")
    cv_score      = model_data.get("cv_score")
    model_version = model_data.get("model_version", "unknown")
    feature_names = model_data.get("feature_names") or model_data.get("features")

    if not feature_names:
        raise ValueError("Feature list missing from saved model package.")

    logger.info("Fraud model loaded — threshold=%.5f | version=%s", threshold, model_version)

except Exception as e:
    logger.error("Failed to load fraud model: %s", e)
    raise e

# ── SHAP engine ───────────────────────
try:
    shap_engine = IntelligentSHAP(pipeline)
    logger.info("SHAP engine ready.")
except Exception as e:
    logger.warning("SHAP engine failed to init: %s", e)
    shap_engine = None

# ======================================
# ── UNIVERSAL MODEL ───────────────────
# Load universal_model.pkl if it exists
# (optional — only needed for /universal/predict)
# ======================================

universal_trainer = None
try:
    universal_trainer = UniversalTrainer()
    universal_trainer.load("models/universal_model.pkl")
    logger.info(
        "Universal model loaded — best=%s | threshold=%.5f",
        universal_trainer.best_model_name,
        universal_trainer.threshold,
    )
except Exception as e:
    logger.warning("Universal model not loaded (train one first): %s", e)

# ======================================
# ── MONITORING ────────────────────────
# ======================================

monitor       = ModelMonitor()
alert_manager = AlertManager()
logger.info("ModelMonitor and AlertManager ready.")


# ======================================================================
# INPUT SCHEMAS
# ======================================================================

class Transaction(BaseModel):
    """Fixed schema for Credit Card Fraud dataset (30 features)."""
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


class BatchTransactions(BaseModel):
    transactions: list[Transaction]


# ======================================================================
# ── GENERAL ROUTES ────────────────────────────────────────────────────
# ======================================================================

@app.get("/", tags=["General"])
def health_check():
    """API health check — shows both model statuses."""
    return {
        "status": "AutoML-X API is running",
        "fraud_model": {
            "version":   model_version,
            "objective": objective,
            "cv_score":  round(cv_score, 5) if cv_score else None,
            "threshold": round(threshold, 5),
        },
        "universal_model": {
            "loaded":     universal_trainer is not None,
            "best_model": universal_trainer.best_model_name if universal_trainer else None,
            "threshold":  round(universal_trainer.threshold, 5) if universal_trainer else None,
        },
    }


@app.get("/model-info", tags=["General"])
def model_info():
    """Fraud model metadata + feature list."""
    return {
        "model_version":  model_version,
        "objective":      objective,
        "cv_score":       round(cv_score, 5) if cv_score else None,
        "threshold":      round(threshold, 5),
        "feature_count":  len(feature_names),
        "features":       feature_names,
    }


# ======================================================================
# ── FRAUD DETECTION ROUTES ────────────────────────────────────────────
# ======================================================================

@app.post("/predict", tags=["Fraud Detection"])
def predict(transaction: Transaction):
    """
    Predict fraud for a single credit card transaction.
    Uses the fixed 30-feature fraud model with SHAP explanation.
    """
    try:
        input_dict = transaction.model_dump()
        input_df   = pd.DataFrame([input_dict])
        input_df   = input_df.reindex(columns=feature_names, fill_value=0)

        probability = float(pipeline.predict_proba(input_df)[0][1])
        prediction  = int(probability >= threshold)

        low_boundary    = threshold * 0.25
        medium_boundary = threshold * 0.75
        if probability < low_boundary:
            risk_level = "Low"
        elif probability < medium_boundary:
            risk_level = "Medium"
        else:
            risk_level = "High"

        explanation = {}
        if shap_engine is not None:
            try:
                explanation = shap_engine.local_explanation_json(
                    input_df, index=0, top_k=5
                )
            except Exception as shap_err:
                logger.warning("SHAP failed: %s", shap_err)
                explanation = {"error": "SHAP unavailable"}

        try:
            monitor.log_prediction(input_dict, prediction, probability)
        except Exception as mon_err:
            logger.warning("Monitor log failed: %s", mon_err)

        return {
            "model_info": {
                "version":   model_version,
                "objective": objective,
                "cv_score":  round(cv_score, 5) if cv_score else None,
            },
            "prediction_result": {
                "fraud_probability_percent": round(probability * 100, 6),
                "threshold_used":            round(threshold, 5),
                "predicted_class":           prediction,
                "label":                     "Fraud" if prediction == 1 else "Legitimate",
                "risk_level":                risk_level,
            },
            "explanation": explanation,
        }

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Fraud Detection"])
def predict_batch(batch: BatchTransactions):
    """Batch fraud prediction for multiple transactions."""
    try:
        records  = [t.model_dump() for t in batch.transactions]
        input_df = pd.DataFrame(records)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        probabilities = pipeline.predict_proba(input_df)[:, 1]
        predictions   = (probabilities >= threshold).astype(int)

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
                "index":                     i,
                "fraud_probability_percent": round(prob * 100, 6),
                "predicted_class":           int(pred),
                "label":                     "Fraud" if pred == 1 else "Legitimate",
                "risk_level":                risk_level,
            })

        try:
            for record, pred, prob in zip(records, predictions, probabilities):
                monitor.log_prediction(record, int(pred), float(prob))
        except Exception as mon_err:
            logger.warning("Monitor batch log failed: %s", mon_err)

        return {
            "threshold_used":     round(threshold, 5),
            "total_transactions": len(results),
            "fraud_count":        int(sum(predictions)),
            "results":            results,
        }

    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# ── UNIVERSAL TRAINER ROUTES ──────────────────────────────────────────
# ======================================================================

@app.post("/universal/train", tags=["Universal Trainer"])
async def universal_train(
    file: UploadFile = File(..., description="CSV file to train on"),
    target_col: str  = Form(..., description="Name of the target column"),
    positive_label: str = Form(None, description="Which value means positive (optional)"),
):
    """
    Train a binary classifier on ANY uploaded CSV dataset.

    - Accepts any CSV with a binary target column
    - Auto-detects numeric and categorical columns
    - Trains 4 models: LogisticRegression, RandomForest, LightGBM, XGBoost
    - Selects best model via cross-validation (ROC-AUC)
    - Optimizes decision threshold via F1
    - Saves model to models/universal_model.pkl
    - Returns full evaluation metrics

    Example with curl:
        curl -X POST http://localhost:7860/universal/train
             -F "file=@churn.csv"
             -F "target_col=Churn"
             -F "positive_label=Yes"
    """
    global universal_trainer

    # Read uploaded CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        logger.info("Universal train: loaded %d rows x %d cols from %s",
                    len(df), df.shape[1], file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Validate target column exists
    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found. "
                   f"Available columns: {df.columns.tolist()}"
        )

    # Train
    try:
        trainer = UniversalTrainer(model_save_path="models/universal_model.pkl")
        metrics = trainer.fit(
            df=df,
            target_col=target_col,
            positive_label=positive_label if positive_label else None,
        )
        universal_trainer = trainer
        logger.info("Universal training complete — best=%s ROC-AUC=%.5f",
                    metrics["best_model"], metrics["test_roc_auc"])

        return {
            "status":   "Training complete",
            "filename": file.filename,
            "dataset": {
                "rows":       len(df),
                "cols":       df.shape[1],
                "target_col": target_col,
            },
            "metrics": metrics,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Universal training failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/universal/model-info", tags=["Universal Trainer"])
def universal_model_info():
    """
    Info about the currently loaded universal model.
    Train one first via POST /universal/train.
    """
    if universal_trainer is None:
        raise HTTPException(
            status_code=404,
            detail="No universal model loaded. Train one via POST /universal/train"
        )
    return {
        "best_model":     universal_trainer.best_model_name,
        "threshold":      round(universal_trainer.threshold, 5),
        "target_col":     universal_trainer.target_col,
        "positive_label": universal_trainer.positive_label,
        "feature_count":  len(universal_trainer.feature_names),
        "features":       universal_trainer.feature_names,
        "metrics":        universal_trainer.metrics,
        "all_cv_scores":  universal_trainer.all_scores,
        "dataset_profile": {
            "n_rows":        universal_trainer.profile.get("n_rows"),
            "n_numeric":     universal_trainer.profile.get("n_numeric"),
            "n_categorical": universal_trainer.profile.get("n_categorical"),
            "is_imbalanced": universal_trainer.profile.get("is_imbalanced"),
            "minority_ratio": universal_trainer.profile.get("minority_ratio"),
        } if universal_trainer.profile else None,
    }


@app.post("/universal/predict", tags=["Universal Trainer"])
async def universal_predict(
    file: UploadFile = File(..., description="CSV with one row to predict"),
):
    """
    Predict using the trained universal model.

    Upload a CSV with a single row (or multiple rows for batch).
    Missing columns are filled with 0, extra columns are ignored.
    Target column is automatically excluded if present.

    Example:
        curl -X POST http://localhost:7860/universal/predict
             -F "file=@single_row.csv"
    """
    if universal_trainer is None:
        raise HTTPException(
            status_code=404,
            detail="No universal model loaded. Train one via POST /universal/train"
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Drop target column if present (user may upload full dataset row)
    if universal_trainer.target_col and universal_trainer.target_col in df.columns:
        df = df.drop(columns=[universal_trainer.target_col])

    try:
        probabilities = universal_trainer.predict_proba(df)
        predictions   = universal_trainer.predict(df)

        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            prob = float(prob)
            t    = universal_trainer.threshold
            results.append({
                "index":            i,
                "probability":      round(prob, 6),
                "probability_pct":  round(prob * 100, 4),
                "predicted_class":  int(pred),
                "label":            "POSITIVE" if pred == 1 else "NEGATIVE",
                "threshold_used":   round(t, 5),
            })

        return {
            "model":        universal_trainer.best_model_name,
            "target_col":   universal_trainer.target_col,
            "total_rows":   len(results),
            "positive_count": int(sum(predictions)),
            "positive_rate":  round(float(predictions.mean()) * 100, 2),
            "results":      results,
        }

    except Exception as e:
        logger.error("Universal predict failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/universal/predict/json", tags=["Universal Trainer"])
def universal_predict_json(data: dict):
    """
    Predict using the universal model with a JSON body instead of CSV.

    Send a single row as a flat JSON object:
        {"feature1": value1, "feature2": value2, ...}

    Missing features are filled with 0, extra keys are ignored.
    """
    if universal_trainer is None:
        raise HTTPException(
            status_code=404,
            detail="No universal model loaded. Train one via POST /universal/train"
        )

    try:
        # Remove target column key if accidentally included
        if universal_trainer.target_col in data:
            data.pop(universal_trainer.target_col)

        input_df    = pd.DataFrame([data])
        probability = float(universal_trainer.predict_proba(input_df)[0])
        prediction  = int(universal_trainer.predict(input_df)[0])
        t           = universal_trainer.threshold

        return {
            "model":           universal_trainer.best_model_name,
            "target_col":      universal_trainer.target_col,
            "probability":     round(probability, 6),
            "probability_pct": round(probability * 100, 4),
            "predicted_class": prediction,
            "label":           "POSITIVE" if prediction == 1 else "NEGATIVE",
            "threshold_used":  round(t, 5),
        }

    except Exception as e:
        logger.error("Universal predict JSON failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# ── MONITORING ROUTES ─────────────────────────────────────────────────
# ======================================================================

@app.get("/monitor/health", tags=["Monitoring"])
def monitor_health():
    """System health — DB status, prediction counts, model availability."""
    return monitor.check_health()


@app.get("/monitor/summary", tags=["Monitoring"])
def monitor_summary(last_n: int = 100):
    """Rolling fraud prediction statistics over last N requests."""
    return monitor.get_summary(last_n=last_n)


@app.get("/monitor/alerts", tags=["Monitoring"])
def monitor_alerts(last_n: int = 50):
    """Alert history from the SQLite alert log."""
    return {"alerts": monitor.get_alert_history(last_n=last_n)}


@app.post("/monitor/check-alerts", tags=["Monitoring"])
def monitor_check_alerts(last_n: int = 100):
    """Manually trigger alert rule evaluation against recent predictions."""
    fired = monitor.trigger_alerts(last_n=last_n)
    return {
        "fired_count": len(fired),
        "alerts": [
            {
                "name":     a.name,
                "severity": a.severity,
                "message":  a.message,
                "extra":    a.extra,
            }
            for a in fired
        ],
    }