import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift_detector import DriftDetector
from src.universal_trainer import UniversalTrainer, DatasetProfiler, check_ram_safety

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML-X",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0a0f; }

.metric-card {
    background: #13131a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}
.metric-label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.fraud-alert {
    background: linear-gradient(135deg, #1a0a0a, #2a0a0a);
    border: 1px solid #ff4444;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.safe-alert {
    background: linear-gradient(135deg, #0a1a0a, #0a2a0a);
    border: 1px solid #00ff88;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 8px;
}
.mode-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-fraud { background: #1a1a2e; border: 1px solid #00ff88; color: #00ff88; }
.badge-universal { background: #1a1a2e; border: 1px solid #00aaff; color: #00aaff; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════

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

@st.cache_resource
def load_universal_model():
    try:
        trainer = UniversalTrainer()
        trainer.load("models/universal_model.pkl")
        return trainer
    except Exception:
        return None


# ── Feature Engineering (must match train.py) ─────────────────
def engineer_features(df):
    seconds_in_day      = 3600 * 24
    df["Hour"]          = (df["Time"] % seconds_in_day) / 3600
    df["Night_txn"]     = ((df["Hour"] < 6) | (df["Hour"] > 22)).astype(int)
    df["Amount_log"]    = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-8)
    df["High_amount"]   = (df["Amount"] > 1000).astype(int)
    return df


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ AutoML-X")
    st.markdown("*Intelligent ML Platform*")
    st.divider()

    mode = st.radio(
        "Select Mode",
        ["🛡️ Fraud Detection", "🤖 Universal Trainer"],
        label_visibility="collapsed"
    )

    st.divider()

    if mode == "🛡️ Fraud Detection":
        st.markdown('<span class="mode-badge badge-fraud">Fraud Mode</span>', unsafe_allow_html=True)
        st.markdown("")
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
            "📊 Model Info"
        ], label_visibility="collapsed")

    else:
        st.markdown('<span class="mode-badge badge-universal">Universal Mode</span>', unsafe_allow_html=True)
        st.markdown("")
        trainer = load_universal_model()
        if trainer:
            m = trainer.metrics
            st.markdown("### ✅ Model Loaded")
            st.markdown(f"**Model:** `{m.get('best_model', 'N/A')}`")
            st.markdown(f"**Target:** `{trainer.target_col}`")
            st.markdown(f"**ROC-AUC:** `{m.get('test_roc_auc', 0):.5f}`")
            st.markdown(f"**Threshold:** `{trainer.threshold:.5f}`")
        else:
            st.info("No model trained yet.\nGo to Upload → Train.")

        st.divider()
        page = st.radio("Navigation", [
            "📁 Upload & Configure",
            "🚀 Train Model",
            "📊 Results",
            "🔍 Predict",
            "📂 Batch Predict"
        ], label_visibility="collapsed")

    st.divider()
    st.caption("AutoML-X Platform | v3.0")


# ══════════════════════════════════════════════════════════════
# FRAUD DETECTION PAGES
# ══════════════════════════════════════════════════════════════

if mode == "🛡️ Fraud Detection":

    st.markdown("# 🛡️ AutoML-X Fraud Detection")
    st.markdown("*Production-grade credit card fraud detection with SHAP explainability*")
    st.divider()

    # ── PAGE: Single Transaction ──────────────────────────────
    if page == "🔍 Single Transaction":
        st.markdown("### Enter Transaction Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            time_val   = st.number_input("Time (seconds)", value=406.0, step=1.0)
            amount_val = st.number_input("Amount ($)", value=149.62, min_value=0.0, step=0.01)

        with col2:
            st.markdown("**V1 – V10**")
            v_vals = {}
            defaults1 = {1: -1.35, 2: -0.07, 3: 2.53, 4: 1.37, 5: -0.33,
                         6: 0.46, 7: 0.23, 8: 0.09, 9: 0.36, 10: 0.09}
            for i in range(1, 11):
                v_vals[f"V{i}"] = st.number_input(f"V{i}", value=defaults1.get(i, 0.0),
                                                   format="%.4f", key=f"v{i}")

        with col3:
            st.markdown("**V11 – V20**")
            defaults2 = {11: -0.55, 12: -0.61, 13: -0.99, 14: -0.31, 15: 1.46,
                         16: -0.47, 17: 0.20, 18: 0.02, 19: 0.40, 20: 0.25}
            for i in range(11, 21):
                v_vals[f"V{i}"] = st.number_input(f"V{i}", value=defaults2.get(i, 0.0),
                                                   format="%.4f", key=f"v{i}")

        cols = st.columns(8)
        defaults3 = {21: -0.01, 22: 0.27, 23: -0.11, 24: 0.06,
                     25: 0.12, 26: -0.18, 27: 0.13, 28: -0.02}
        for idx, i in enumerate(range(21, 29)):
            with cols[idx]:
                v_vals[f"V{i}"] = st.number_input(f"V{i}", value=defaults3.get(i, 0.0),
                                                   format="%.4f", key=f"v{i}")

        st.divider()

        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            try:
                model_data = load_fraud_model()
                pipeline   = model_data["model"]
                threshold  = model_data["threshold"]

                row = {"Time": time_val, "Amount": amount_val}
                row.update(v_vals)
                df  = pd.DataFrame([row])
                df  = engineer_features(df)

                prob     = pipeline.predict_proba(df)[0][1]
                is_fraud = prob >= threshold
                risk     = "🔴 HIGH" if prob > 0.7 else ("🟡 MEDIUM" if prob > 0.3 else "🟢 LOW")

                st.divider()
                if is_fraud:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <div style="font-size:3rem">🚨</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#ff4444;margin:8px 0">FRAUD DETECTED</div>
                        <div style="font-size:2.5rem;font-weight:700;color:#ff4444;font-family:monospace">{prob*100:.1f}%</div>
                        <div style="color:#888;font-size:0.9rem">Fraud Probability | Risk: {risk}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <div style="font-size:3rem">✅</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#00ff88;margin:8px 0">LEGITIMATE</div>
                        <div style="font-size:2.5rem;font-weight:700;color:#00ff88;font-family:monospace">{prob*100:.1f}%</div>
                        <div style="color:#888;font-size:0.9rem">Fraud Probability | Risk: {risk}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                st.markdown("### 🔍 Why this decision?")
                try:
                    import shap
                    explainer     = shap.TreeExplainer(pipeline.named_steps["model"])
                    X_transformed = pipeline.named_steps["preprocessor"].transform(df)
                    shap_values   = explainer.shap_values(X_transformed)

                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                    else:
                        sv = shap_values[0] if shap_values.ndim == 3 else shap_values[0]

                    feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                    shap_df    = pd.DataFrame({"feature": feat_names, "shap": sv})
                    shap_df    = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=False).index).head(10)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors  = ["#ff4444" if v > 0 else "#00ff88" for v in shap_df["shap"]]
                    ax.barh(shap_df["feature"], shap_df["shap"], color=colors)
                    ax.set_xlabel("SHAP Value")
                    ax.set_title("Top 10 Feature Contributions")
                    ax.axvline(0, color="white", linewidth=0.5)
                    fig.patch.set_facecolor("#13131a")
                    ax.set_facecolor("#13131a")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.title.set_color("white")
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ── PAGE: Batch Detection ─────────────────────────────────
    elif page == "📦 Batch Detection":
        st.markdown("### Upload CSV for Batch Fraud Detection")
        st.info("Upload a CSV with columns: Time, Amount, V1–V28")

        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            try:
                df_raw     = pd.read_csv(uploaded)
                model_data = load_fraud_model()
                pipeline   = model_data["model"]
                threshold  = model_data["threshold"]

                df_eng = engineer_features(df_raw.copy())
                probs  = pipeline.predict_proba(df_eng)[:, 1]
                preds  = (probs >= threshold).astype(int)

                df_raw["fraud_probability"] = (probs * 100).round(2)
                df_raw["prediction"]        = preds
                df_raw["label"]             = df_raw["prediction"].map({0: "✅ Legitimate", 1: "🚨 Fraud"})

                fraud_count = preds.sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",       f"{len(df_raw):,}")
                c2.metric("Fraud",       f"{fraud_count:,}")
                c3.metric("Fraud Rate",  f"{fraud_count/len(df_raw)*100:.2f}%")
                c4.metric("Est. Risk",   f"${fraud_count * 10000:,}")

                dd = load_drift_detector()
                if dd:
                    st.divider()
                    st.markdown("### 📡 Data Drift Check")
                    report = dd.detect(df_eng)
                    st.markdown(f"**{report['status']}** — {report['drifted_count']}/{report['total_features']} features drifted")

                st.divider()
                st.dataframe(
                    df_raw[["fraud_probability", "label", "Amount", "Time"]]
                    .sort_values("fraud_probability", ascending=False).head(100),
                    use_container_width=True
                )
                csv = df_raw.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv, "fraud_results.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

    # ── PAGE: Model Info ──────────────────────────────────────
    elif page == "📊 Model Info":
        st.markdown("### Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CV ROC-AUC",   "0.99999")
        c2.metric("Test ROC-AUC", "0.97482")
        c3.metric("Recall",       "89.8%")
        c4.metric("Business Cost","$113,800")

        st.divider()
        st.markdown("### Experimentation Journey")
        experiments = pd.DataFrame({
            "Run":       [1, 2, 3, 4, 5],
            "Technique": ["SMOTE", "BorderlineSMOTE", "SMOTETomek", "SMOTE+FeatEng", "SMOTE+FeatEng"],
            "Model":     ["RandomForest", "LightGBM", "LightGBM", "LightGBM", "RandomForest"],
            "Recall":    ["90.8%", "84.7%", "84.7%", "88.8%", "89.8%"],
            "Cost":      ["$108,400", "$151,800", "$157,200", "$118,800", "$113,800"],
            "Verdict":   ["Strong baseline", "Lower recall", "No improvement", "Best AUC", "✅ Final"],
        })
        st.dataframe(experiments, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Business Cost Formula")
        st.code("Total Cost = (Missed Frauds × $10,000) + (False Alarms × $200)")
        st.markdown("At threshold **0.20603**: 10 missed × $10,000 + 69 false alarms × $200 = **$113,800**")


# ══════════════════════════════════════════════════════════════
# UNIVERSAL TRAINER PAGES
# ══════════════════════════════════════════════════════════════

else:
    st.markdown("# 🤖 AutoML-X Universal Trainer")
    st.markdown("*Train on any binary classification dataset automatically*")
    st.divider()

    # ── PAGE: Upload & Configure ──────────────────────────────
    if page == "📁 Upload & Configure":
        st.subheader("Upload Your Dataset")
        st.markdown("Upload **any CSV** with a binary target column.")

        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.session_state["df"]       = df
                st.session_state["filename"] = uploaded.name
                st.success(f"✅ Loaded: **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} cols")

                ram = check_ram_safety(df)
                if not ram["is_safe"]:
                    st.warning(f"⚠️ {ram['warning']}")

                st.dataframe(df.head(10), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows",    f"{len(df):,}")
                c2.metric("Columns", df.shape[1])
                c3.metric("Missing", f"{df.isnull().sum().sum():,}")

                st.divider()
                st.subheader("⚙️ Configure")
                col1, col2 = st.columns(2)
                with col1:
                    target_col = st.selectbox("🎯 Target Column", df.columns.tolist(),
                                              index=len(df.columns)-1)
                    st.session_state["target_col"] = target_col
                with col2:
                    unique_vals = df[target_col].dropna().unique().tolist()
                    pos_raw = st.selectbox("✅ Positive Value",
                                           ["Auto-detect"] + [str(v) for v in unique_vals])
                    st.session_state["positive_label"] = None if pos_raw == "Auto-detect" else pos_raw

                if st.button("🔍 Analyze Dataset", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        profiler = DatasetProfiler()
                        profile  = profiler.profile(df, target_col)
                        st.session_state["profile"] = profile

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Numeric",     profile["n_numeric"])
                        c2.metric("Categorical", profile["n_categorical"])
                        c3.metric("Missing %",   f"{profile['missing_pct']}%")
                        c4.metric("Imbalanced",  "Yes ⚠️" if profile["is_imbalanced"] else "No ✅")

                        if profile["n_classes"] != 2:
                            st.error(f"❌ Found {profile['n_classes']} classes — only binary supported.")
                        else:
                            st.success("✅ Binary confirmed! Go to 🚀 Train Model.")

            except Exception as e:
                st.error(f"Failed: {e}")

    # ── PAGE: Train ───────────────────────────────────────────
    elif page == "🚀 Train Model":
        st.subheader("Train AutoML Model")

        if "df" not in st.session_state or st.session_state.get("df") is None:
            st.warning("⚠️ Upload a dataset first.")
            st.stop()

        df         = st.session_state["df"]
        target_col = st.session_state.get("target_col")
        pos_label  = st.session_state.get("positive_label")

        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset",      st.session_state.get("filename", "Uploaded"))
        c2.metric("Rows",         f"{len(df):,}")
        c3.metric("Target",       target_col or "Not set")

        st.info("Will train: LogisticRegression, RandomForest, LightGBM, XGBoost")

        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def update_progress(step, total, msg):
                progress_bar.progress(step / total)
                status_text.markdown(f"**Step {step}/{total}:** {msg}")

            try:
                trainer = UniversalTrainer(model_save_path="models/universal_model.pkl")
                metrics = trainer.fit(
                    df=df,
                    target_col=target_col,
                    positive_label=pos_label,
                    sample_if_large=len(df) > 500_000,
                    progress_callback=update_progress
                )
                st.session_state["u_trainer"] = trainer
                st.session_state["u_metrics"] = metrics
                st.session_state["u_trained"] = True
                progress_bar.progress(1.0)
                st.success("🎉 Training complete! Go to 📊 Results.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Best Model", metrics["best_model"])
                c2.metric("ROC-AUC",   f"{metrics['test_roc_auc']:.5f}")
                c3.metric("F1",        f"{metrics['f1_score']:.5f}")
                c4.metric("Recall",    f"{metrics['recall']:.5f}")

            except Exception as e:
                st.error(f"Training failed: {e}")

    # ── PAGE: Results ─────────────────────────────────────────
    elif page == "📊 Results":
        trainer = st.session_state.get("u_trainer") or load_universal_model()
        if trainer is None:
            st.warning("⚠️ No model trained yet.")
            st.stop()

        metrics = trainer.metrics
        st.subheader("Performance Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model",   metrics["best_model"])
        c2.metric("CV ROC-AUC",   f"{metrics['cv_roc_auc']:.5f}")
        c3.metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.5f}")
        c4.metric("F1 Score",     f"{metrics['f1_score']:.5f}")
        c5.metric("Recall",       f"{metrics['recall']:.5f}")

        st.divider()
        st.subheader("Model Competition")
        scores_df = pd.DataFrame({
            "Model":      list(metrics["all_cv_scores"].keys()),
            "CV ROC-AUC": list(metrics["all_cv_scores"].values())
        }).sort_values("CV ROC-AUC", ascending=False)
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

        st.divider()
        tp, tn = metrics["TP"], metrics["TN"]
        fp, fn = metrics["FP"], metrics["FN"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Threshold",  f"{metrics['threshold']:.5f}")
        c2.metric("Precision",  f"{metrics['precision']:.5f}")
        c3.metric("Recall",     f"{metrics['recall']:.5f}")

        st.markdown(f"**Confusion Matrix** — TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

        if os.path.exists("models/universal_model.pkl"):
            with open("models/universal_model.pkl", "rb") as f:
                st.download_button("⬇️ Download Model", f,
                                   "universal_model.pkl", "application/octet-stream",
                                   use_container_width=True)

    # ── PAGE: Single Predict ──────────────────────────────────
    elif page == "🔍 Predict":
        trainer = st.session_state.get("u_trainer") or load_universal_model()
        if trainer is None:
            st.warning("⚠️ Train a model first.")
            st.stop()

        st.subheader("Single Prediction")
        features = trainer.feature_names
        df_ref   = st.session_state.get("df")

        input_vals = {}
        cols_per_row = 4
        col_list = st.columns(cols_per_row)
        for i, feat in enumerate(features):
            with col_list[i % cols_per_row]:
                if df_ref is not None and feat in df_ref.columns:
                    sample = df_ref[feat].dropna()
                    if sample.dtype in [np.float64, np.int64, np.float32, np.int32]:
                        input_vals[feat] = st.number_input(feat, value=float(sample.median()), format="%.4f")
                    else:
                        input_vals[feat] = st.selectbox(feat, sample.unique().tolist())
                else:
                    input_vals[feat] = st.number_input(feat, value=0.0)

        if st.button("🚀 Predict", type="primary", use_container_width=True):
            try:
                input_df    = pd.DataFrame([input_vals])
                probability = float(trainer.predict_proba(input_df)[0])
                prediction  = int(probability >= trainer.threshold)
                t           = trainer.threshold

                if prediction == 1:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <div style="font-size:3rem">⚠️</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#ff4444">POSITIVE</div>
                        <div style="font-size:2.5rem;font-weight:700;color:#ff4444;font-family:monospace">{probability*100:.2f}%</div>
                        <div style="color:#888">Probability | Threshold: {t:.5f}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <div style="font-size:3rem">✅</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#00ff88">NEGATIVE</div>
                        <div style="font-size:2.5rem;font-weight:700;color:#00ff88;font-family:monospace">{probability*100:.2f}%</div>
                        <div style="color:#888">Probability | Threshold: {t:.5f}</div>
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ── PAGE: Batch Predict ───────────────────────────────────
    elif page == "📂 Batch Predict":
        trainer = st.session_state.get("u_trainer") or load_universal_model()
        if trainer is None:
            st.warning("⚠️ Train a model first.")
            st.stop()

        st.subheader("Batch Prediction")
        st.info(f"Required features: {', '.join(trainer.feature_names[:5])}... ({len(trainer.feature_names)} total)")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_new = pd.read_csv(uploaded)
            X_raw  = df_new.drop(columns=[trainer.target_col], errors="ignore")

            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Predicting..."):
                    try:
                        probabilities = trainer.predict_proba(X_raw)
                        predictions   = (probabilities >= trainer.threshold).astype(int)

                        results = df_new.copy()
                        results["probability_%"]   = (probabilities * 100).round(4)
                        results["predicted_class"] = predictions
                        results["label"]           = ["POSITIVE" if p == 1 else "NEGATIVE" for p in predictions]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total",    len(results))
                        c2.metric("Positive", int(predictions.sum()))
                        c3.metric("Rate",     f"{predictions.mean()*100:.2f}%")

                        st.dataframe(results, use_container_width=True)
                        csv = results.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Download Predictions", csv,
                                           "predictions.csv", "text/csv",
                                           use_container_width=True)

                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")