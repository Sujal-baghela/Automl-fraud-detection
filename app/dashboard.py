import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.drift_detector import DriftDetector
from src.shap_explainer import IntelligentSHAP


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AutoML-X Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# LOAD MODEL & METADATA (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    data = joblib.load("models/best_model.pkl")
    return data

@st.cache_resource
def load_metadata():
    with open("models/metadata_v1.json") as f:
        return json.load(f)

@st.cache_resource
def load_drift_detector():
    dd = DriftDetector()
    dd.load("models/drift_reference.json")
    return dd

@st.cache_resource
def load_shap_engine(_pipeline):
    return IntelligentSHAP(_pipeline)

try:
    model_data    = load_model()
    metadata      = load_metadata()
    pipeline      = model_data["model"]
    threshold     = model_data["threshold"]
    feature_names = model_data.get("feature_names") or model_data.get("features")
    cv_score      = model_data.get("cv_score", 0)
    model_version = model_data.get("model_version", "v1")
    drift_detector = load_drift_detector()
    shap_engine    = load_shap_engine(pipeline)
    MODEL_LOADED   = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"❌ Could not load model: {e}")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/AutoML--X-Fraud%20Detection-blue?style=for-the-badge")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["🏠 Overview", "🔍 Predict Transaction", "📂 Batch Prediction", "🌊 Drift Monitor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📦 Model Info")
    st.markdown(f"**Version:** `{model_version}`")
    st.markdown(f"**Threshold:** `{threshold:.5f}`")
    st.markdown(f"**CV ROC-AUC:** `{cv_score:.5f}`")
    st.markdown(f"**Features:** `{len(feature_names)}`")
    st.markdown("---")
    st.caption("Built with ❤️ at MITS")


# ═══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🛡️ AutoML-X — Fraud Detection System")
    st.markdown("Production-ready AutoML pipeline with SHAP explainability and cost-optimized threshold.")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏆 Best Model",    metadata.get("model_name", "RandomForest"))
    col2.metric("📈 CV ROC-AUC",    f"{metadata.get('cv_score', 0):.5f}")
    col3.metric("⚡ Threshold",      f"{metadata.get('threshold', 0):.5f}")
    col4.metric("💰 Business Cost", f"${metadata.get('business_cost', 0):,}")
    col5.metric("🔢 Features",       len(metadata.get("features", [])))

    st.markdown("---")

    # Plots
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 ROC Curve")
        if os.path.exists("reports/evaluation/roc_curve.png"):
            st.image("reports/evaluation/roc_curve.png", use_container_width=True)
        else:
            st.info("Run training to generate ROC curve.")

    with col_right:
        st.subheader("🔲 Confusion Matrix")
        if os.path.exists("reports/evaluation/confusion_matrix.png"):
            st.image("reports/evaluation/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run training to generate confusion matrix.")

    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance")

    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        if os.path.exists("reports/shap/feature_importance_bar.png"):
            st.image("reports/shap/feature_importance_bar.png", use_container_width=True)
        else:
            st.info("Run training to generate SHAP plots.")

    with col_shap2:
        if os.path.exists("reports/shap/global_summary.png"):
            st.image("reports/shap/global_summary.png", use_container_width=True)
        else:
            st.info("Run training to generate SHAP plots.")

    st.markdown("---")
    st.subheader("🧪 Experimentation Journey")
    experiments = pd.DataFrame({
        "Run":       [1, 2, 3, 4, 5],
        "Technique": ["SMOTE", "BorderlineSMOTE", "SMOTETomek",
                      "SMOTE + Feature Eng.", "SMOTE + Feature Eng."],
        "Model":     ["RandomForest", "LightGBM", "LightGBM",
                      "LightGBM_Balanced", "RandomForest ✅"],
        "Recall":    ["90.8%", "84.7%", "84.7%", "88.8%", "89.8%"],
        "Cost":      ["$108,400", "$151,800", "$157,200", "$118,800", "$113,800"],
        "Outcome":   ["Strong baseline", "Lower recall", "No improvement",
                      "Best Test AUC", "✅ Final"]
    })
    st.dataframe(experiments, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — SINGLE TRANSACTION PREDICTOR
# ═══════════════════════════════════════════════════════

elif page == "🔍 Predict Transaction":
    st.title("🔍 Single Transaction Predictor")
    st.markdown("Enter transaction details below to get a fraud prediction with SHAP explanation.")
    st.markdown("---")

    # Input form
    with st.form("prediction_form"):
        st.subheader("📝 Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            time_val   = st.number_input("Time (seconds)", value=406.0, step=1.0)
        with col2:
            amount_val = st.number_input("Amount ($)", value=149.62, min_value=0.0, step=0.01)

        st.markdown("**PCA Features (V1 — V28)**")
        v_cols = st.columns(7)
        v_values = {}
        for i in range(1, 29):
            col_idx = (i - 1) % 7
            with v_cols[col_idx]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01,
                                                      format="%.4f",
                                                      label_visibility="visible")

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        # Build input
        input_dict = {"Time": time_val, "Amount": amount_val}
        input_dict.update(v_values)
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        probability = float(pipeline.predict_proba(input_df)[0][1])
        prediction  = int(probability >= threshold)

        # Risk level
        if probability < threshold * 0.25:
            risk = "Low"
            risk_color = "green"
        elif probability < threshold * 0.75:
            risk = "Medium"
            risk_color = "orange"
        else:
            risk = "High"
            risk_color = "red"

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        # Gauge meter
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(probability * 100, 2),
            title={"text": "Fraud Probability (%)"},
            delta={"reference": threshold * 100},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0,  threshold * 25],  "color": "#d4edda"},
                    {"range": [threshold * 25, threshold * 75], "color": "#fff3cd"},
                    {"range": [threshold * 75, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Result cards
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Prediction",    "🚨 FRAUD" if prediction == 1 else "✅ Legitimate")
        col2.metric("📊 Probability",   f"{probability * 100:.4f}%")
        col3.metric("⚠️ Risk Level",    risk)

        # SHAP explanation
        st.markdown("---")
        st.subheader("🔍 SHAP Explanation — Why This Prediction?")

        try:
            explanation = shap_engine.local_explanation_json(input_df, index=0, top_k=5)

            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.markdown("**🔴 Features pushing toward FRAUD**")
                pos_features = explanation.get("top_positive_features", [])
                if pos_features:
                    pos_df = pd.DataFrame(pos_features)
                    pos_df["impact"] = pos_df["impact"].round(4)
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No positive SHAP features.")

            with col_neg:
                st.markdown("**🟢 Features pushing toward LEGITIMATE**")
                neg_features = explanation.get("top_negative_features", [])
                if neg_features:
                    neg_df = pd.DataFrame(neg_features)
                    neg_df["impact"] = neg_df["impact"].round(4)
                    st.dataframe(neg_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No negative SHAP features.")

            st.caption(f"Base value (average model output): {explanation.get('base_value', 0):.5f}")

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")


# ═══════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════

elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV file of transactions to get fraud predictions for all rows.")
    st.markdown("---")

    st.info("📋 CSV must contain the same columns as training data: `Time`, `V1-V28`, `Amount`. `Class` column is optional.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df)} transactions | {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner("Running predictions..."):
                try:
                    # Drop target if present
                    X = df.drop(columns=["Class"], errors="ignore")
                    X = X.reindex(columns=feature_names, fill_value=0)

                    probabilities = pipeline.predict_proba(X)[:, 1]
                    predictions   = (probabilities >= threshold).astype(int)

                    results = df.copy()
                    results["fraud_probability_%"] = (probabilities * 100).round(4)
                    results["predicted_class"]     = predictions
                    results["label"]               = ["🚨 Fraud" if p == 1 else "✅ Legitimate"
                                                       for p in predictions]
                    results["risk_level"]          = [
                        "High"   if p >= threshold * 0.75 else
                        "Medium" if p >= threshold * 0.25 else "Low"
                        for p in probabilities
                    ]

                    # Summary metrics
                    st.markdown("---")
                    st.subheader("📊 Batch Results Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Transactions", len(results))
                    col2.metric("🚨 Frauds Detected",  int(predictions.sum()))
                    col3.metric("✅ Legitimate",        int((predictions == 0).sum()))
                    col4.metric("📊 Fraud Rate",        f"{predictions.mean()*100:.2f}%")

                    # Fraud distribution chart
                    fig = go.Figure(go.Pie(
                        labels=["Legitimate", "Fraud"],
                        values=[(predictions == 0).sum(), predictions.sum()],
                        hole=0.4,
                        marker_colors=["#28a745", "#dc3545"]
                    ))
                    fig.update_layout(title="Prediction Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    # Full results table
                    st.subheader("📋 Full Results")
                    st.dataframe(results, use_container_width=True)

                    # Download button
                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════
# PAGE 4 — DRIFT MONITOR
# ═══════════════════════════════════════════════════════

elif page == "🌊 Drift Monitor":
    st.title("🌊 Data Drift Monitor")
    st.markdown("Upload new transaction data to check if it has drifted from the training distribution.")
    st.markdown("---")

    st.info("📋 Upload a CSV with the same feature columns used during training. The system will compare it against the saved drift reference.")

    uploaded_file = st.file_uploader("Upload CSV for drift analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {len(df)} transactions")

        if st.button("🔍 Run Drift Analysis", use_container_width=True):
            with st.spinner("Analyzing drift..."):
                try:
                    X = df.drop(columns=["Class"], errors="ignore")
                    report = drift_detector.detect(X)

                    st.markdown("---")
                    st.subheader("📊 Drift Report")

                    # Overall status
                    status = report["status"]
                    if "NO DRIFT" in status:
                        st.success(f"{status}")
                    elif "MODERATE" in status:
                        st.warning(f"{status}")
                    else:
                        st.error(f"{status}")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Drifted Features",  report["drifted_count"])
                    col2.metric("Total Features",    report["total_features"])
                    col3.metric("Drift Ratio",       f"{report['drift_ratio']*100:.1f}%")

                    # PSI bar chart
                    details = report["details"]
                    if details:
                        features_list = list(details.keys())
                        psi_values    = [details[f]["psi"] for f in features_list]
                        colors        = [
                            "#dc3545" if details[f]["psi"] >= 0.2 else
                            "#ffc107" if details[f]["psi"] >= 0.1 else
                            "#28a745"
                            for f in features_list
                        ]

                        fig = go.Figure(go.Bar(
                            x=features_list,
                            y=psi_values,
                            marker_color=colors,
                            name="PSI Score"
                        ))
                        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                                      annotation_text="Moderate (0.1)")
                        fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                                      annotation_text="High (0.2)")
                        fig.update_layout(
                            title="PSI Score per Feature",
                            xaxis_title="Feature",
                            yaxis_title="PSI Score",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Detailed table
                    st.subheader("📋 Feature-level Drift Details")
                    detail_rows = []
                    for feat, d in details.items():
                        detail_rows.append({
                            "Feature": feat,
                            "PSI":     d["psi"],
                            "KS p-value": d["pvalue"],
                            "Status":  d["status"],
                            "Drifted": "Yes" if d["drifted"] else "No"
                        })
                    detail_df = pd.DataFrame(detail_rows).sort_values("PSI", ascending=False)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

                    if report["drifted_count"] > 0:
                        st.warning(f"⚠️ Consider retraining — {report['drifted_count']} features have drifted significantly.")
                    else:
                        st.success("✅ Model is stable — no retraining needed.")

                except Exception as e:
                    st.error(f"Drift analysis failed: {e}")