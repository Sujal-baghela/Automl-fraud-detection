# AutoML-X Project Summary
> Paste this entire file at the start of a new chat to restore full context.

## 🔗 GitHub
https://github.com/Sujal-baghela/Automl-fraud-detection

## 👤 Developer
- **Name:** Sujal Baghela
- **Co-author:** Sameer Bhilware
- **Institution:** Madhav Institute of Technology and Science (MITS)
- **Laptop:** Lenovo LOQ, Ryzen 5 7000, 24GB RAM, 512GB storage
- **OS:** Windows 11, PowerShell
- **Python:** 3.11, virtual environment at `venv\`
- **Project path:** `C:\Users\Lenovo\Documents\auto ml\automl-x`

## 📌 Project Overview
AutoML-X is a production-ready fraud detection system being redesigned into a **universal AutoML tool** that works on any binary classification dataset (any columns, any domain).

## 📁 Complete File Structure
```
automl-x/
├── .github/workflows/ci.yml     # GitHub Actions — lint (ruff) + test (pytest)
├── app/
│   ├── api.py                   # FastAPI REST API (fixed — _risk_level helper)
│   └── dashboard.py             # Streamlit dashboard (4 pages, flexible CSV)
├── Scripts/
│   ├── __init__.py
│   └── train.py                 # Training pipeline (fixed — cleaner integrated)
├── src/
│   ├── model_selector.py        # 5 models compete via CV
│   ├── fraud_system.py          # Main orchestrator
│   ├── cost_optimizer.py        # Business cost threshold optimization
│   ├── threshold_optimizer.py   # F1/recall/precision threshold
│   ├── shap_explainer.py        # SHAP explainability
│   ├── inference_engine.py      # Production inference
│   ├── evaluation.py            # ROC curve, confusion matrix
│   ├── drift_detector.py        # PSI + KS drift detection (fixed)
│   ├── data_loader.py           # CSV loading
│   └── cleaner.py               # Data cleaning
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # sys.path fix
│   └── test_pipeline.py         # 53 tests, 10 classes, ALL PASSING
├── models/
│   ├── best_model.pkl           # Trained model (gitignored)
│   ├── drift_reference.json     # Drift reference stats
│   └── metadata_v1.json         # Model metadata
├── reports/
│   ├── evaluation/              # roc_curve.png, confusion_matrix.png
│   └── shap/                    # feature_importance_bar.png, global_summary.png
├── Data/
│   └── .gitkeep                 # creditcard.csv goes here (gitignored)
├── .gitattributes               # LF line endings
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md                    # Updated with CI badge, full docs
```

## ✅ CI/CD Status
- **CI:** GitHub Actions — AutoML-X CI
- **Lint:** ruff — ✅ PASSING
- **Tests:** pytest — ✅ 53 PASSING, 2 skipped (need trained model)
- **Coverage:** 58%
- **Badge:** Live green badge on README

## 🐛 All Bugs Fixed
1. `cleaner.clean()` was never called in `train.py` — fixed
2. `engineer_features()` mutated caller's DataFrame — added `.copy()`
3. `api.py` had duplicated risk level logic — extracted to `_risk_level()` helper
4. `drift_detector._psi()` crashed on zero-variance features — guard added
5. `cost_optimizer` used fixed 200 thresholds — now uses actual probability values
6. `cleaner.impute_missing()` mutated df — added `.copy()`
7. `ci.yml` was missing 6 dependencies — fixed
8. `app.py` had bare `except:` and unnecessary f-strings — fixed
9. `drift_detector.py` had semicolon on same line — fixed

## 🧪 Test Classes
1. TestFeatureEngineering (7 tests)
2. TestDataLoader (6 tests)
3. TestDataCleaner (6 tests)
4. TestDriftDetector (8 tests)
5. TestThresholdOptimizer (5 tests)
6. TestBusinessCostOptimizer (5 tests)
7. TestAutoModelSelector (4 tests)
8. TestAutoMLFraudDetector (8 tests)
9. TestGenerateEvaluationReports (3 tests)
10. TestSavedArtifacts (3 tests, 2 skip without model)

## 🎯 Model Results
- Best Model: Random Forest
- CV ROC-AUC: 0.99999
- Test ROC-AUC: 0.97482
- Recall: 89.8% (88/98 frauds caught)
- Threshold: 0.20603 (cost-optimized)
- Business Cost: $113,800
- Features: 35 (30 original + 5 engineered)

## 📊 Streamlit Dashboard (app/dashboard.py)
4 pages:
1. 🏠 Overview — model stats, ROC curve, confusion matrix, SHAP plots
2. 🔍 Predict Transaction — single prediction with gauge + SHAP
3. 📂 Batch Prediction — upload ANY CSV, auto-fill missing cols with 0
4. 🌊 Drift Monitor — upload ANY CSV, analyze matching columns only

Run: `streamlit run app/dashboard.py`

## 🔄 Next Steps (IN ORDER)
### 1. ✅ DONE — CI/CD + Tests + Code Quality
### 2. ✅ DONE — Streamlit Dashboard
### 3. 🔨 IN PROGRESS — Universal AutoML Redesign
   - Make the system work for ANY binary classification dataset
   - Any columns, any domain (fraud, churn, medical, loan default etc.)
   - Support datasets up to 1GB (25.4GB RAM available)
   - Binary classification only (not multi-class)
   - Two dashboards: fraud-specific + universal
   - New files to create:
     - `src/universal_trainer.py` — dataset-agnostic training engine
     - `app/universal.py` — universal Streamlit dashboard
   - Behavior for large datasets (>500k rows): WARN USER and ask permission
   - Replace SMOTE with class_weight for datasets >500k rows
   - Remove hardcoded feature engineering for universal mode
   - Keep all existing fraud detection code untouched

### 4. MLflow Model Registry
### 5. Increase test coverage 58% → 80%
### 6. Monitoring + Alerting system
### 7. Deploy to cloud (Render/HuggingFace)

## 🛠️ Commands Reference
### In VENV (activate first: `venv\Scripts\activate`):
- Train model: `python -m Scripts.train`
- Run API: `uvicorn app.api:app --reload --port 8000`
- Run dashboard: `streamlit run app/dashboard.py`
- Run universal: `streamlit run app/universal.py`
- Run tests: `pytest tests/ -v --tb=short`
- MLflow UI: `mlflow ui --port 5000`

### Outside VENV (git commands):
- `git add .`
- `git commit -m "message"`
- `git push origin main`

## 📦 Key Dependencies
pytest, pytest-cov, ruff, streamlit, plotly, fastapi, uvicorn,
scikit-learn, lightgbm, xgboost, shap, mlflow, imbalanced-learn,
pandas, numpy, scipy, joblib, seaborn, matplotlib, psutil, chardet, pyarrow
