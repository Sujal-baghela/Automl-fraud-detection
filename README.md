💳 AutoML-X: Intelligent Fraud Detection System

A production-style automated machine learning pipeline for credit card fraud detection with:

Multi-model training & selection

Cross-validation optimization

SHAP explainability

Automated evaluation reporting

Logging + reproducibility

Modular architecture

Designed for industry-grade ML workflows.

📌 Problem Statement

Credit card fraud detection requires:

High recall (detect fraud cases)

High precision (avoid false alarms)

Model interpretability

Reproducible ML pipeline

This project builds a complete automated ML training system that selects the best model based on ROC-AUC and generates explainable outputs.

Dataset: Based on anonymized credit card transaction records.

⚙️ System Architecture

Training Pipeline:

Data Loading

Train / Validation Split

Automatic Feature Detection

Multi-Model Training

Cross-Validation Evaluation

Best Model Selection

SHAP Explainability

Evaluation Report Generation

Model Persistence

🧠 Models Evaluated

Logistic Regression

Random Forest

Gradient Boosting

LightGBM

Best model is selected automatically based on ROC-AUC.

📊 Model Performance

Best Model: LightGBM
Cross-Validation ROC-AUC: 0.97793

Evaluation artifacts generated automatically:

ROC Curve (reports/evaluation/roc_curve.png)

Confusion Matrix (reports/evaluation/confusion_matrix.png)

Classification Report (reports/evaluation/classification_report.txt)

SHAP Global Explanation (reports/shap/)

🔍 Explainability

Uses SHAP (TreeExplainer for tree-based models) to:

Identify top contributing features

Provide global feature importance visualization

Increase transparency for fraud detection decisions

📂 Project Structure
automl-x/
│
├── src/
│   ├── model_selector.py
│   ├── shap_explainer.py
│   ├── evaluation.py
│
├── Scripts/
│   ├── train.py
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── shap/
│   └── evaluation/
│
├── README.md
└── requirements.txt
🚀 Installation
git clone <your-repo-url>
cd automl-x
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
🏃 Run Training Pipeline

From project root:

python -m Scripts.train

Pipeline will:

Train multiple models

Select best model

Generate SHAP explainability

Generate evaluation reports

Save model to models/

📈 Why This Project Is Industry-Ready

Modular architecture

Logging enabled

Automated evaluation saving

Explainability included

Reproducible training

Clean separation of concerns

This mirrors real-world ML system design.

🔮 Future Improvements

REST API deployment (FastAPI)

Docker containerization

MLflow experiment tracking

Hyperparameter optimization (Optuna)

CI/CD integration

Cloud deployment (AWS/GCP/Azure)