"""
app_fraud_local.py
══════════════════
AutoML-X Fraud Detection — LOCAL ONLY
Runs on port 8501. Never deployed to HuggingFace.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift_detector import DriftDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML-X Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0a0f; }

.metric-card  { background:#13131a; border:1px solid #1e1e2e; border-radius:12px; padding:20px; text-align:center; }
.metric-value { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#00ff88; }
.metric-label { font-size:.8rem; color:#666; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.fraud-alert  { background:linear-gradient(135deg,#1a0a0a,#2a0a0a); border:1px solid #ff4444; border-radius:12px; padding:24px; text-align:center; }
.safe-alert   { background:linear-gradient(135deg,#0a1a0a,#0a2a0a); border:1px solid #00ff88; border-radius:12px; padding:24px; text-align:center; }
.local-badge  { display:inline-block; background:#1a0a2a; border:1px solid #cc00ff; color:#cc00ff;
                padding:4px 14px; border-radius:20px; font-size:.75rem; font-weight:600;
                letter-spacing:1px; text-transform:uppercase; font-family:'Space Mono',monospace; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_fraud_model():
    return joblib.load("models/best_model.pkl")

@st.cache_resource
def load_metadata():
    try:
        with open("models/metadata_v1.json") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_resource
def load_drift_detector():
    try:
        if os.path.exists("models/drift_reference.json"):
            dd = DriftDetector()
            dd.load("models/drift_reference.json")
            return dd
    except Exception:
        pass
    return None


def engineer_features(df):
    seconds_in_day      = 3600 * 24
    df["Hour"]          = (df["Time"] % seconds_in_day) / 3600
    df["Night_txn"]     = ((df["Hour"] < 6) | (df["Hour"] > 22)).astype(int)
    df["Amount_log"]    = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-8)
    df["High_amount"]   = (df["Amount"] > 1000).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ AutoML-X")
    st.markdown("*Fraud Detection*")
    st.markdown('<span class="local-badge">🔒 Local Only</span>', unsafe_allow_html=True)
    st.divider()
    try:
        meta = load_metadata()
        if meta:
            st.markdown(f"**Model:** {meta.get('model_name', 'RandomForest')}")
            st.markdown(f"**Version:** {meta.get('model_version', 'v5')}")
            st.markdown(f"**Threshold:** {meta.get('threshold', 0.206)}")
    except Exception:
        pass
    st.markdown("**Test ROC-AUC:** 0.97482")
    st.markdown("**Recall:** 89.8%")
    st.markdown("**Business Cost:** $113,800")
    st.divider()
    page = st.radio("Navigation", [
        "🔍 Single Transaction",
        "📦 Batch Detection",
        "📊 Model Info",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("AutoML-X · v5.0 · Fraud · Local")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🛡️ AutoML-X Fraud Detection")
st.markdown(
    "*Production-grade credit card fraud detection with SHAP explainability* &nbsp;&nbsp;"
    '<span class="local-badge">🔒 Local Only</span>',
    unsafe_allow_html=True
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Single Transaction
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Single Transaction":
    st.markdown("### Enter Transaction Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        time_val   = st.number_input("Time (seconds)", value=406.0, step=1.0)
        amount_val = st.number_input("Amount ($)", value=149.62, min_value=0.0, step=0.01)

    with col2:
        st.markdown("**V1–V10**")
        v_vals = {}
        d1 = {1:-1.35, 2:-0.07, 3:2.53, 4:1.37, 5:-0.33,
              6:0.46,  7:0.23,  8:0.09, 9:0.36, 10:0.09}
        for i in range(1, 11):
            v_vals[f"V{i}"] = st.number_input(
                f"V{i}", value=d1.get(i, 0.0), format="%.4f", key=f"v{i}"
            )

    with col3:
        st.markdown("**V11–V20**")
        d2 = {11:-0.55, 12:-0.61, 13:-0.99, 14:-0.31, 15:1.46,
              16:-0.47, 17:0.20, 18:0.02, 19:0.40, 20:0.25}
        for i in range(11, 21):
            v_vals[f"V{i}"] = st.number_input(
                f"V{i}", value=d2.get(i, 0.0), format="%.4f", key=f"v{i}"
            )

    cols = st.columns(8)
    d3   = {21:-0.01, 22:0.27, 23:-0.11, 24:0.06,
            25:0.12,  26:-0.18, 27:0.13, 28:-0.02}
    for idx, i in enumerate(range(21, 29)):
        with cols[idx]:
            v_vals[f"V{i}"] = st.number_input(
                f"V{i}", value=d3.get(i, 0.0), format="%.4f", key=f"v{i}"
            )

    st.divider()
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        try:
            model_data          = load_fraud_model()
            pipeline, threshold = model_data["model"], model_data["threshold"]
            row                 = {"Time": time_val, "Amount": amount_val}
            row.update(v_vals)
            df       = engineer_features(pd.DataFrame([row]))
            prob     = pipeline.predict_proba(df)[0][1]
            is_fraud = prob >= threshold
            risk     = "🔴 HIGH" if prob > 0.7 else ("🟡 MEDIUM" if prob > 0.3 else "🟢 LOW")

            st.divider()
            if is_fraud:
                st.markdown(f"""<div class="fraud-alert">
                    <div style="font-size:3rem">🚨</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#ff4444">FRAUD DETECTED</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#ff4444;font-family:monospace">{prob*100:.1f}%</div>
                    <div style="color:#888">Fraud Probability | Risk: {risk}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="safe-alert">
                    <div style="font-size:3rem">✅</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#00ff88">LEGITIMATE</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#00ff88;font-family:monospace">{prob*100:.1f}%</div>
                    <div style="color:#888">Fraud Probability | Risk: {risk}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()
            st.markdown("### 🔍 Why this decision? (SHAP)")
            try:
                import shap
                explainer     = shap.TreeExplainer(pipeline.named_steps["model"])
                X_transformed = pipeline.named_steps["preprocessor"].transform(df)
                shap_values   = explainer.shap_values(X_transformed)
                sv = (shap_values[1][0]
                      if isinstance(shap_values, list)
                      else (shap_values[0] if shap_values.ndim == 3 else shap_values[0]))
                feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                shap_df    = pd.DataFrame({"feature": feat_names, "shap": sv})
                shap_df    = shap_df.reindex(
                    shap_df["shap"].abs().sort_values(ascending=False).index
                ).head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(shap_df["feature"], shap_df["shap"],
                        color=["#ff4444" if v > 0 else "#00ff88" for v in shap_df["shap"]])
                ax.set_xlabel("SHAP Value")
                ax.axvline(0, color="white", linewidth=0.5)
                fig.patch.set_facecolor("#13131a")
                ax.set_facecolor("#13131a")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.info(f"SHAP unavailable: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Detection
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📦 Batch Detection":
    st.markdown("### Upload CSV for Batch Fraud Detection")
    st.info("Upload a CSV with columns: Time, Amount, V1–V28")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        try:
            df_raw              = pd.read_csv(uploaded)
            model_data          = load_fraud_model()
            pipeline, threshold = model_data["model"], model_data["threshold"]
            df_eng              = engineer_features(df_raw.copy())
            probs               = pipeline.predict_proba(df_eng)[:, 1]
            preds               = (probs >= threshold).astype(int)

            df_raw["fraud_probability"] = (probs * 100).round(2)
            df_raw["prediction"]        = preds
            df_raw["label"]             = df_raw["prediction"].map(
                {0: "✅ Legitimate", 1: "🚨 Fraud"}
            )
            fraud_count = preds.sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",      f"{len(df_raw):,}")
            c2.metric("Fraud",      f"{fraud_count:,}")
            c3.metric("Fraud Rate", f"{fraud_count/len(df_raw)*100:.2f}%")
            c4.metric("Est. Risk",  f"${fraud_count * 10_000:,}")

            dd = load_drift_detector()
            if dd:
                st.divider()
                st.markdown("### 📡 Data Drift Check")
                report = dd.detect(df_eng)
                st.markdown(
                    f"**{report['status']}** — "
                    f"{report['drifted_count']}/{report['total_features']} features drifted"
                )

            st.divider()
            st.dataframe(
                df_raw[["fraud_probability", "label", "Amount", "Time"]]
                .sort_values("fraud_probability", ascending=False)
                .head(100),
                use_container_width=True
            )
            st.download_button(
                "⬇️ Download Results",
                df_raw.to_csv(index=False).encode(),
                "fraud_results.csv", "text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Info
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Info":
    st.markdown("### Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV ROC-AUC",    "0.99999")
    c2.metric("Test ROC-AUC",  "0.97482")
    c3.metric("Recall",        "89.8%")
    c4.metric("Business Cost", "$113,800")

    st.divider()
    st.markdown("### Experimentation Journey")
    st.dataframe(pd.DataFrame({
        "Run":       [1, 2, 3, 4, 5],
        "Technique": ["SMOTE", "BorderlineSMOTE", "SMOTETomek",
                      "SMOTE+FeatEng", "SMOTE+FeatEng"],
        "Model":     ["RandomForest", "LightGBM", "LightGBM",
                      "LightGBM", "RandomForest"],
        "Recall":    ["90.8%", "84.7%", "84.7%", "88.8%", "89.8%"],
        "Cost":      ["$108,400", "$151,800", "$157,200", "$118,800", "$113,800"],
        "Verdict":   ["Strong baseline", "Lower recall", "No improvement",
                      "Best AUC", "✅ Final"],
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Business Cost Formula")
    st.code("Total Cost = (Missed Frauds × $10,000) + (False Alarms × $200)")
    st.markdown(
        "At threshold **0.20603**: "
        "10 missed × $10,000 + 69 false alarms × $200 = **$113,800**"
    )