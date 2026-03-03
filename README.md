# 🛡️ AutoML-X — Intelligent Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-009688?style=flat-square&logo=fastapi)
![RandomForest](https://img.shields.io/badge/RandomForest-AutoML-brightgreen?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-0.50.0-orange?style=flat-square)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced--Learn-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Competing-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![CI](https://github.com/Sujal-baghela/Automl-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujal-baghela/Automl-fraud-detection/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/Tests-53%20passing-brightgreen?style=flat-square)
![Coverage](https://img.shields.io/badge/Coverage-58%25-yellow?style=flat-square)

---

## 📌 Overview

**AutoML-X** is a production-ready, end-to-end machine learning system for credit card fraud detection. It automatically trains, evaluates, and selects the best model from 5 competing algorithms using cross-validation, applies feature engineering on raw transaction data, optimizes the decision threshold based on real business costs, and serves predictions through a FastAPI REST API with full SHAP explainability.

Built as a **Minor Project** at **Madhav Institute of Technology and Science**.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| 🏆 Best Model | Random Forest |
| 📈 CV ROC-AUC | **0.99999** |
| 📊 Test ROC-AUC | **0.97482** |
| 🎯 Frauds Caught | **88 / 98 (89.8% Recall)** |
| 💰 Minimum Business Cost | **$113,800** |
| ⚡ Optimal Threshold | **0.20603** (cost-optimized) |
| 🔢 Total Features | **35** (30 original + 5 engineered) |
| 🔁 Imbalance Handling | SMOTE (50/50 balanced) |
| 🧪 Test Suite | **53 tests passing** across 10 classes |
| 📋 CI/CD | GitHub Actions — lint + test on every push |

---

## 📊 Evaluation Reports

**ROC Curve (AUC = 0.97482)**

![ROC Curve](reports/evaluation/roc_curve.png)

**Confusion Matrix (Threshold = 0.20603)**

![Confusion Matrix](reports/evaluation/confusion_matrix.png)

---

## 🔍 SHAP Explainability

**Feature Importance — Mean |SHAP| (Top 20 Features)**

![SHAP Bar](reports/shap/feature_importance_bar.png)

**Feature Impact Direction — Beeswarm Plot**

![SHAP Beeswarm](reports/shap/global_summary.png)

> Top fraud indicators: `V14`, `V4`, `V12`, `V10`, `V3` — consistent with published research on this dataset.

---

## 🏗️ System Architecture

```
creditcard.csv
      │
      ▼
 DataLoader ──► imbalance detection (0.17% fraud rate)
      │
      ▼
 DataCleaner ──► remove duplicates, impute missing, log outliers
      │
      ▼
 Feature Engineering (+5 derived features)
      ├── Hour          (fraud varies by time of day)
      ├── Night_txn     (10pm–6am binary flag)
      ├── Amount_log    (log transform — compresses skew)
      ├── Amount_zscore (how unusual is this amount)
      └── High_amount   (Amount > $1,000 binary flag)
      │
      ▼
 SMOTE Oversampling ──► balanced training set (50/50)
      │
      ▼
 AutoModelSelector — 5 models compete via cross-validation
      ├── LogisticRegression    (ROC-AUC: 0.99812)
      ├── RandomForest          (ROC-AUC: 0.99999) ✅ selected
      ├── LightGBM_Balanced     (ROC-AUC: 0.99998)
      ├── LightGBM_HighRecall   (ROC-AUC: 0.99997)
      └── XGBoost_HighRecall    (ROC-AUC: 0.99993)
      │
      ▼
 BusinessCostOptimizer ──► threshold = 0.20603
      │                    cost = $113,800
      ▼
 best_model.pkl ──► FastAPI ──► /predict
      │                   └──► /predict/batch
      │                   └──► SHAP explanation per transaction
      ▼
 DriftDetector ──► PSI + KS test on new data
      │
      ▼
 MLflow ──► experiment tracking + model registry
```

---

## 🧪 Experimentation Journey

5 systematic experiments were run to optimize recall and business cost:

| Run | Technique | Model | Recall | Cost | Outcome |
|-----|-----------|-------|--------|------|---------|
| 1 | SMOTE | RandomForest (30 features) | 90.8% | $108,400 | Strong baseline |
| 2 | BorderlineSMOTE | LightGBM | 84.7% | $151,800 | Lower recall |
| 3 | SMOTETomek | LightGBM variants | 84.7% | $157,200 | No improvement |
| 4 | SMOTE + Feature Eng. | LightGBM_Balanced | 88.8% | $118,800 | Best Test AUC |
| **5** | **SMOTE + Feature Eng.** | **RandomForest (35 features)** | **89.8%** | **$113,800** | **✅ Final** |

Feature engineering caught **4 additional borderline frauds** that the model previously missed — proving domain-aware feature extraction adds measurable value beyond hyperparameter tuning alone.

---

## 🔧 Feature Engineering

The raw dataset has `V1–V28` (PCA-anonymized) plus `Time` and `Amount` (original). Five new features were derived:

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `Hour` | Hour of day (0–23) | Fraud rates vary by time of day |
| `Night_txn` | 1 if between 10pm–6am | Direct signal for suspicious hours |
| `Amount_log` | log(1 + Amount) | Compresses right skew for fair treatment |
| `Amount_zscore` | Standard deviations from mean | Flags unusually sized transactions |
| `High_amount` | 1 if Amount > $1,000 | Direct signal for high-value transactions |

---

## 💰 Business Cost Optimization

Instead of the default 0.5 threshold, `BusinessCostOptimizer` searches threshold values using actual predicted probabilities to minimize:

```
Total Cost = (Missed Frauds × $10,000) + (False Alarms × $200)
```

Missing a fraud costs **50× more** than a false alarm. The optimizer reflects this — accepting more false alarms to catch more real fraud.

```
Final result at threshold 0.20603:
  10 frauds missed  ×  $10,000  =  $100,000
  69 false alarms   ×    $200   =   $13,800
  ─────────────────────────────────────────
  Total minimum cost            =  $113,800
```

---

## 🧪 Test Suite

53 unit tests across 10 test classes — all passing in CI:

| Test Class | Tests | What It Covers |
|---|---|---|
| `TestFeatureEngineering` | 7 | Hour range, binary flags, no nulls, immutability |
| `TestDataLoader` | 6 | Load, split, missing target, metadata |
| `TestDataCleaner` | 6 | Duplicates, imputation, outlier detection |
| `TestDriftDetector` | 8 | Fit, detect, save/load, threshold preservation |
| `TestThresholdOptimizer` | 5 | All 3 strategies, invalid strategy, recall check |
| `TestBusinessCostOptimizer` | 5 | Cost minimization, asymmetric loss behavior |
| `TestAutoModelSelector` | 4 | Pipeline structure, CV scores, predict_proba |
| `TestAutoMLFraudDetector` | 8 | Fit, predict, save/load, version, None guard |
| `TestGenerateEvaluationReports` | 3 | ROC-AUC, file outputs, return keys |
| `TestSavedArtifacts` | 3 | Model package keys, metadata (skip if no model) |

**🟢 Run locally:**

```bash
pytest tests/ -v --tb=short
```

---

## 🔁 CI/CD Pipeline

Every push to `main` triggers two jobs automatically:

```
Push to main
      │
      ▼
 ┌─── Lint (ruff) ───────────────────────────────┐
 │  Checks code style across all .py files       │
 │  Fails fast on syntax/style errors            │
 └───────────────────────────────────────────────┘
      │ (only runs if lint passes)
      ▼
 ┌─── Test (pytest) ──────────────────────────────┐
 │  Runs 53 unit tests with coverage             │
 │  Uploads coverage.xml as artifact             │
 │  Uses pip caching for fast runs (~1m 30s)     │
 └────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
automl-x/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions — lint + test pipeline
│
├── app/
│   └── api.py                  # FastAPI REST API (single + batch prediction)
│
├── Scripts/
│   ├── __init__.py
│   └── train.py                # Training entry point + feature engineering
│
├── src/
│   ├── model_selector.py       # AutoML — 5 models compete via CV
│   ├── fraud_system.py         # Main orchestrator (fit, predict, save, load)
│   ├── cost_optimizer.py       # Business cost threshold optimization
│   ├── threshold_optimizer.py  # Metric-based threshold (F1, recall, precision)
│   ├── shap_explainer.py       # SHAP explainability (SHAP 0.50.0 compatible)
│   ├── inference_engine.py     # Production inference engine
│   ├── evaluation.py           # ROC curve, confusion matrix, report generation
│   ├── drift_detector.py       # PSI + KS test data drift detection
│   ├── data_loader.py          # Dataset loading and validation
│   └── cleaner.py              # Data cleaning utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared pytest path configuration
│   └── test_pipeline.py        # 53 unit tests across 10 classes
│
├── models/
│   ├── best_model.pkl          # Trained model package (gitignored)
│   ├── drift_reference.json    # Drift detector reference statistics
│   └── metadata_v1.json        # Model metadata and training config
│
├── reports/
│   ├── evaluation/             # ROC curve, confusion matrix, classification report
│   └── shap/                   # SHAP beeswarm and bar plots
│
├── Data/
│   └── .gitkeep                # Placeholder — add creditcard.csv here
│
├── .gitattributes              # Normalize line endings across OS
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

**1. Clone the repository**

```bash
git clone https://github.com/Sujal-baghela/Automl-fraud-detection.git
cd Automl-fraud-detection
```

**2. Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `Data/creditcard.csv`.

**5. Train the model**

```bash
python -m Scripts.train
```

**6. Start the API**

```bash
uvicorn app.api:app --reload --port 8000
```

**7. Open Swagger UI**

```
http://127.0.0.1:8000/docs
```

**8. View MLflow experiments**

```bash
mlflow ui --port 5000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model info |
| `GET` | `/model-info` | Full model metadata + feature list |
| `POST` | `/predict` | Single transaction prediction + SHAP |
| `POST` | `/predict/batch` | Batch prediction for multiple transactions |

**Sample Request**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0, "Amount": 149.62,
    "V1": -1.35, "V2": -0.07, "V3": 2.53,
    "V4": 1.37, "V5": -0.33, "V6": 0.46,
    "V7": 0.23, "V8": 0.09, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46,
    "V16": -0.47, "V17": 0.20, "V18": 0.02,
    "V19": 0.40, "V20": 0.25, "V21": -0.01,
    "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13,
    "V28": -0.02
  }'
```

**Sample Response**

```json
{
  "model_info": {
    "version": "v5",
    "objective": "cost",
    "cv_score": 0.99999
  },
  "prediction_result": {
    "fraud_probability_percent": 23.4,
    "threshold_used": 0.20603,
    "predicted_class": 1,
    "label": "Fraud",
    "risk_level": "High"
  },
  "explanation": {
    "base_value": 0.00173,
    "top_positive_features": [
      {"feature": "num__V14", "impact": 0.0842},
      {"feature": "num__V4",  "impact": 0.0631}
    ],
    "top_negative_features": [
      {"feature": "num__V12", "impact": -0.0521}
    ]
  }
}
```

---

## 🐳 Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# API will be available at http://localhost:8000
```

---

## 📦 Tech Stack

| Category | Library |
|----------|---------|
| ML Models | scikit-learn, LightGBM, XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP 0.50.0 |
| API | FastAPI, Uvicorn |
| Experiment Tracking | MLflow |
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| Testing | pytest, pytest-cov |
| Linting | Ruff |
| CI/CD | GitHub Actions |

---

## 📋 Dataset

**Credit Card Fraud Detection** — European cardholders, September 2013.

- 284,807 transactions over 2 days
- 492 frauds (0.17% — highly imbalanced)
- Features V1–V28 are PCA-transformed (anonymized)
- Features `Time` and `Amount` are original
- Binary target: `0` = Legitimate, `1` = Fraud

Source: [Kaggle — ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> ⚠️ Dataset not included due to size and licensing. Download from Kaggle and place at `Data/creditcard.csv`.

---

## 👥 Authors

**Sujal Baghela** — [@Sujal-baghela](https://github.com/Sujal-baghela)

**Sameer Bhilware** — [@SameerBhilware-ui](https://github.com/SameerBhilware-ui)

**Institution:** Madhav Institute of Technology and Science

---

## 📄 License

This project is licensed under the MIT License.