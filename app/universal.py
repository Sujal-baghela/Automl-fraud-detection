import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.universal_trainer import UniversalTrainer, DatasetProfiler, check_ram_safety


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AutoML-X Universal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

for key, default in {
    "df":           None,
    "trainer":      None,
    "metrics":      None,
    "trained":      False,
    "profile":      None,
    "target_col":   None,
    "positive_label": None,
    "filename":     None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AutoML-X Universal")
    st.markdown("Train on **any** binary classification dataset.")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📁 Upload & Configure",
         "🚀 Train Model",
         "📊 Results & Evaluation",
         "🔍 Predict",
         "📂 Batch Predict"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if st.session_state.trained and st.session_state.metrics:
        m = st.session_state.metrics
        st.markdown("### ✅ Model Trained")
        st.markdown(f"**Model:** `{m.get('best_model', 'N/A')}`")
        st.markdown(f"**ROC-AUC:** `{m.get('test_roc_auc', 0):.5f}`")
        st.markdown(f"**F1:** `{m.get('f1_score', 0):.5f}`")
        st.markdown(f"**Threshold:** `{m.get('threshold', 0.5):.5f}`")
    else:
        st.info("No model trained yet.\nGo to Upload → Train.")

    st.markdown("---")
    st.caption("AutoML-X Universal | Binary Classification")


# ═══════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & CONFIGURE
# ═══════════════════════════════════════════════════════

if page == "📁 Upload & Configure":
    st.title("📁 Upload Your Dataset")
    st.markdown("Upload **any CSV** with a binary target column. AutoML-X will handle the rest.")
    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    col_upload, col_path = st.columns([2, 1])

    with col_upload:
        st.subheader("Option A — Upload CSV (up to 200MB)")
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    with col_path:
        st.subheader("Option B — Large file (>200MB)")
        file_path = st.text_input(
            "Enter full file path",
            placeholder=r"C:\data\myfile.csv"
        )
        load_path = st.button("Load from path")

    # Load data
    df = None

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.filename = uploaded.name
            st.success(f"✅ Loaded: **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    elif load_path and file_path:
        try:
            df = pd.read_csv(file_path)
            st.session_state.filename = os.path.basename(file_path)
            st.success(f"✅ Loaded: **{file_path}** — {len(df):,} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    if df is not None:
        st.session_state.df = df

        # ── RAM Check ─────────────────────────────────────────────────────────
        ram = check_ram_safety(df)
        if not ram["is_safe"]:
            st.warning(f"⚠️ {ram['warning']}")
        else:
            st.success(f"✅ RAM OK — Dataset: {ram['dataframe_gb']}GB | Available: {ram['available_gb']}GB")

        st.markdown("---")

        # ── Preview ───────────────────────────────────────────────────────────
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows",    f"{len(df):,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", f"{df.isnull().sum().sum():,}")
        col4.metric("Size",    f"{ram['dataframe_gb']} GB")

        st.markdown("---")

        # ── Configure ─────────────────────────────────────────────────────────
        st.subheader("⚙️ Configure Training")

        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            target_col = st.selectbox(
                "🎯 Select Target Column (what to predict)",
               options=df.columns.tolist(),
               index=len(df.columns) - 1,
               key="target_col_select"
             )
            st.session_state.target_col = target_col

        with col_cfg2:
            if target_col:
                unique_vals = df[target_col].dropna().unique().tolist()
                if len(unique_vals) > 10:
                    st.warning(f"Target has {len(unique_vals)} unique values — are you sure this is binary?")
                pos_raw = st.selectbox(
                   "✅ Which value means POSITIVE? (fraud, churn, yes, 1...)",
                   options=["Auto-detect"] + [str(v) for v in unique_vals],
                   key="positive_label_select"
                )
                st.session_state.positive_label = None if pos_raw == "Auto-detect" else pos_raw

        # Large dataset sampling warning
        if len(df) > 500_000:
            st.markdown("---")
            st.warning(f"⚠️ Large dataset detected: **{len(df):,} rows**")
            sample_choice = st.radio(
                 "This dataset exceeds 500,000 rows. Training on full data may be slow.",
                 ["Sample 500,000 rows for training (faster, recommended)",
                 "Use full dataset (slower, may need more RAM)"],
                 key="sample_choice_radio"
                )
            st.session_state.sample_if_large = "Sample" in sample_choice
        else:
            sample_if_large = False

        st.markdown("---")

        # ── Profile dataset ───────────────────────────────────────────────────
        target_col = st.session_state.target_col
        if st.button("🔍 Analyze Dataset", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    profiler = DatasetProfiler()
                    profile  = profiler.profile(df, target_col)
                    st.session_state.profile     = profile
                    st.session_state.target_col  = target_col
                    st.session_state.positive_label = positive_label

                    st.subheader("📊 Dataset Analysis")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Numeric Cols",     profile["n_numeric"])
                    col2.metric("Categorical Cols", profile["n_categorical"])
                    col3.metric("Missing Values",   f"{profile['missing_pct']}%")
                    col4.metric("Imbalanced",       "Yes ⚠️" if profile["is_imbalanced"] else "No ✅")

                    # Class distribution
                    class_df = pd.DataFrame({
                        "Class": list(profile["class_counts"].keys()),
                        "Count": list(profile["class_counts"].values())
                    })
                    fig = px.bar(class_df, x="Class", y="Count",
                                 title="Class Distribution",
                                 color="Class")
                    st.plotly_chart(fig, use_container_width=True)

                    if profile["n_classes"] != 2:
                        st.error(f"❌ Found {profile['n_classes']} classes. AutoML-X only supports binary classification.")
                    else:
                        st.success("✅ Binary classification confirmed! Go to 🚀 Train Model.")

                    if profile["high_cardinality_cols"]:
                        st.warning(f"⚠️ High-cardinality columns will be dropped: {profile['high_cardinality_cols']}")

                    if profile["needs_sampling"]:
                        st.info(f"ℹ️ Dataset has {len(df):,} rows — will sample 500k for training unless you chose full dataset.")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")


# ═══════════════════════════════════════════════════════
# PAGE 2 — TRAIN
# ═══════════════════════════════════════════════════════

elif page == "🚀 Train Model":
    st.title("🚀 Train AutoML Model")
    st.markdown("AutoML-X will train 4 competing models and select the best one automatically.")
    st.markdown("---")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first on the Upload page.")
        st.stop()

    if st.session_state.target_col is None:
        st.warning("⚠️ Please configure your target column on the Upload page.")
        st.stop()

    df          = st.session_state.df
    target_col  = st.session_state.target_col
    pos_label   = st.session_state.positive_label
    profile     = st.session_state.profile

    # Show config summary
    st.subheader("📋 Training Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset",        st.session_state.filename or "Uploaded")
    col2.metric("Rows",           f"{len(df):,}")
    col3.metric("Target Column",  target_col)
    col4.metric("Positive Label", str(pos_label) if pos_label else "Auto")

    st.markdown("---")

    st.info("""
    **What will happen:**
    1. ✅ Dataset profiling & validation
    2. ✅ Auto-cleaning (missing values, column types)
    3. ✅ Train 4 models: LogisticRegression, RandomForest, LightGBM, XGBoost
    4. ✅ Select best model via cross-validation (ROC-AUC)
    5. ✅ Optimize decision threshold (F1)
    6. ✅ Generate evaluation metrics
    """)

    if st.button("🚀 Start Training", use_container_width=True, type="primary"):

        progress_bar = st.progress(0)
        status_text  = st.empty()
        log_area     = st.empty()
        logs         = []

        def update_progress(step, total, msg):
            progress_bar.progress(step / total)
            status_text.markdown(f"**Step {step}/{total}:** {msg}")
            logs.append(f"[{step}/{total}] {msg}")
            log_area.code("\n".join(logs))

        try:
            trainer = UniversalTrainer(
                model_save_path="models/universal_model.pkl"
            )

            metrics = trainer.fit(
                df=df,
                target_col=target_col,
                positive_label=pos_label,
                sample_if_large=len(df) > 500_000,
                progress_callback=update_progress
            )

            st.session_state.trainer  = trainer
            st.session_state.metrics  = metrics
            st.session_state.trained  = True

            progress_bar.progress(1.0)
            status_text.markdown("**✅ Training Complete!**")

            st.success("🎉 Training complete! Go to 📊 Results to see your model performance.")

            # Quick metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏆 Best Model",  metrics["best_model"])
            col2.metric("📈 ROC-AUC",     f"{metrics['test_roc_auc']:.5f}")
            col3.metric("🎯 F1 Score",    f"{metrics['f1_score']:.5f}")
            col4.metric("📊 Recall",      f"{metrics['recall']:.5f}")

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════
# PAGE 3 — RESULTS
# ═══════════════════════════════════════════════════════

elif page == "📊 Results & Evaluation":
    st.title("📊 Model Results & Evaluation")
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ No model trained yet. Please train a model first.")
        st.stop()

    metrics = st.session_state.metrics
    trainer = st.session_state.trainer

    # ── Key Metrics ───────────────────────────────────────────────────────────
    st.subheader("🎯 Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏆 Best Model",   metrics["best_model"])
    col2.metric("📈 CV ROC-AUC",   f"{metrics['cv_roc_auc']:.5f}")
    col3.metric("📊 Test ROC-AUC", f"{metrics['test_roc_auc']:.5f}")
    col4.metric("🎯 F1 Score",     f"{metrics['f1_score']:.5f}")
    col5.metric("📋 Recall",       f"{metrics['recall']:.5f}")

    st.markdown("---")

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.subheader("🏁 Model Competition Results")
    scores_df = pd.DataFrame({
        "Model":    list(metrics["all_cv_scores"].keys()),
        "CV ROC-AUC": list(metrics["all_cv_scores"].values())
    }).sort_values("CV ROC-AUC", ascending=False)

    fig = px.bar(
        scores_df, x="Model", y="CV ROC-AUC",
        title="All Models — CV ROC-AUC Comparison",
        color="CV ROC-AUC",
        color_continuous_scale="Greens",
        range_y=[max(0, scores_df["CV ROC-AUC"].min() - 0.05), 1.0]
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Random baseline (0.5)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix")
    tp = metrics["TP"]
    tn = metrics["TN"]
    fp = metrics["FP"]
    fn = metrics["FN"]

    cm_fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative",    "Actual Positive"],
        text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
        texttemplate="%{text}",
        colorscale="Blues"
    ))
    cm_fig.update_layout(title=f"Confusion Matrix (threshold={metrics['threshold']:.4f})", height=350)
    st.plotly_chart(cm_fig, use_container_width=True)

    # ── Threshold Info ────────────────────────────────────────────────────────
    st.subheader("⚡ Decision Threshold")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal Threshold", f"{metrics['threshold']:.5f}")
    col2.metric("Precision",         f"{metrics['precision']:.5f}")
    col3.metric("Recall",            f"{metrics['recall']:.5f}")

    st.markdown("---")

    # ── Download Model ────────────────────────────────────────────────────────
    st.subheader("💾 Download Trained Model")
    if os.path.exists("models/universal_model.pkl"):
        with open("models/universal_model.pkl", "rb") as f:
            st.download_button(
                "⬇️ Download universal_model.pkl",
                data=f,
                file_name="universal_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════
# PAGE 4 — SINGLE PREDICT
# ═══════════════════════════════════════════════════════

elif page == "🔍 Predict":
    st.title("🔍 Single Prediction")
    st.markdown("Enter values for each feature to get a prediction.")
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ Please train a model first.")
        st.stop()

    trainer  = st.session_state.trainer
    features = trainer.feature_names
    df_ref   = st.session_state.df

    with st.form("predict_form"):
        st.subheader("📝 Enter Feature Values")
        input_vals = {}
        cols = st.columns(4)

        for i, feat in enumerate(features):
            col = cols[i % 4]
            with col:
                # Use median as default for numeric, mode for categorical
                if df_ref is not None and feat in df_ref.columns:
                    sample_val = df_ref[feat].dropna()
                    if sample_val.dtype in [np.float64, np.int64, np.float32, np.int32]:
                        default = float(sample_val.median())
                        input_vals[feat] = st.number_input(feat, value=default, format="%.4f")
                    else:
                        options = sample_val.unique().tolist()
                        input_vals[feat] = st.selectbox(feat, options=options)
                else:
                    input_vals[feat] = st.number_input(feat, value=0.0)

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        try:
            input_df    = pd.DataFrame([input_vals])
            probability = float(trainer.predict_proba(input_df)[0])
            prediction  = int(probability >= trainer.threshold)
            threshold   = trainer.threshold

            # Gauge
            risk_color = "red" if probability >= threshold * 0.75 else \
                         "orange" if probability >= threshold * 0.25 else "green"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 2),
                title={"text": "Positive Class Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": risk_color},
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": threshold * 100
                    }
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction",   "✅ POSITIVE" if prediction == 1 else "❌ NEGATIVE")
            col2.metric("Probability",  f"{probability*100:.4f}%")
            col3.metric("Threshold",    f"{threshold:.5f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════
# PAGE 5 — BATCH PREDICT
# ═══════════════════════════════════════════════════════

elif page == "📂 Batch Predict":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV to get predictions for all rows.")
    st.markdown("---")

    if not st.session_state.trained:
        st.warning("⚠️ Please train a model first.")
        st.stop()

    trainer  = st.session_state.trainer
    features = trainer.feature_names

    st.info("📋 Upload any CSV — missing columns auto-filled with 0, extra columns ignored.")

    with st.expander("📋 Required feature columns"):
        st.write(features)

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.markdown(f"**Loaded:** {len(df_new):,} rows | {df_new.shape[1]} cols")

        X_raw   = df_new.drop(columns=[trainer.target_col], errors="ignore")
        matched = [c for c in features if c in X_raw.columns]
        missing = [c for c in features if c not in X_raw.columns]
        extra   = [c for c in X_raw.columns if c not in features]

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Matched", len(matched))
        col2.metric("⚠️ Missing (→0)", len(missing))
        col3.metric("🗑️ Ignored", len(extra))

        if missing:
            st.warning(f"Missing columns filled with 0: `{', '.join(missing[:10])}`")

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner("Predicting..."):
                try:
                    probabilities = trainer.predict_proba(X_raw)
                    predictions   = (probabilities >= trainer.threshold).astype(int)

                    results = df_new.copy()
                    results["probability_%"]   = (probabilities * 100).round(4)
                    results["predicted_class"] = predictions
                    results["label"]           = [
                        "POSITIVE" if p == 1 else "NEGATIVE"
                        for p in predictions
                    ]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total",    len(results))
                    col2.metric("Positive", int(predictions.sum()))
                    col3.metric("Rate",     f"{predictions.mean()*100:.2f}%")

                    fig = go.Figure(go.Pie(
                        labels=["Negative", "Positive"],
                        values=[(predictions==0).sum(), predictions.sum()],
                        hole=0.4,
                        marker_colors=["#28a745", "#dc3545"]
                    ))
                    fig.update_layout(title="Prediction Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(results, use_container_width=True)

                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")