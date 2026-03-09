"""
app_universal.py · AutoML-X v7.3 REFACTORED
AutoML-X Universal Trainer — Modular & Compact Edition
"""
import sys, os, gc, numpy as np, pandas as pd, streamlit as st, matplotlib, matplotlib.pyplot as plt
matplotlib.use("Agg")

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from src.universal_trainer import (
    UniversalTrainer, DatasetProfiler, DataQualityReport, load_csv_chunked, get_tier,
    TIER_TINY, TIER_SMALL, TIER_MEDIUM, TIER_LARGE, TIER_XLARGE, TIER_MASSIVE,
)
from ui_helpers import CSS, JM, ph, sec, render_tier, badge, render_col_table, render_tier_legend, apply_plot_style, no_model_msg, sidebar_model_status, TIER_META

# Optional modules
try:
    from src.shap_universal import UniversalSHAP
    _SHAP = True
except: _SHAP = False
try:
    from src.report_generator import generate_pdf_report
    _PDF = True
except: _PDF = False
try:
    from src.cost_optimizer import BusinessCostOptimizer
    _COST_OPT = True
except: _COST_OPT = False
try:
    from src.threshold_optimizer import ThresholdOptimizer
    _THR_OPT = True
except: _THR_OPT = False
try:
    from src.drift_detector import DriftDetector
    _DRIFT = True
except: _DRIFT = False
try:
    from src.evaluation import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix, plot_score_distribution, plot_threshold_strategies, get_classification_summary
    _EVAL = True
except: _EVAL = False

st.set_page_config(page_title="AutoML-X", page_icon="⬡", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

# Session State Guards
for key, val in [("_current_page", "Home"), ("df", None), ("u_trainer", None), ("target_col", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ━━━ SIDEBAR ━━━
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">&#x2B21;</div><div><span class="brand-name">AutoML-X</span>'
        '<span class="brand-ver">v7.3</span></div></div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:.75rem;color:#6a6a8a;margin-bottom:1rem;font-weight:300">Universal Binary Classifier</div>',
        unsafe_allow_html=True,
    )
    sidebar_model_status(st.session_state.get("u_trainer"))
    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    
    _pages = ["Home", "01 -- Upload", "02 -- Analyze", "03 -- Train", "04 -- Results", "05 -- Predict", "06 -- Batch"]
    _nav_target = st.session_state.pop("_nav", None)
    _nav_index = _pages.index(_nav_target) if _nav_target in _pages else _pages.index(st.session_state["_current_page"])
    page = st.radio("nav", _pages, index=_nav_index, label_visibility="collapsed", key="nav_radio")
    if page != st.session_state["_current_page"]:
        st.session_state["_current_page"] = page
        st.rerun()
    
    st.markdown('<div class="nav-section" style="margin-top:1.8rem">Tier System</div>', unsafe_allow_html=True)
    render_tier_legend()
    st.markdown(
        '<div class="privacy-notice"><div class="pn-title">&#x1F512; DATA PRIVACY</div>'
        "All processing in-memory. No data stored, logged, or transmitted.</div>",
        unsafe_allow_html=True,
    )
    if "df" in st.session_state or "u_trainer" in st.session_state:
        dsn = st.session_state.get("dataset_name", "")
        trained = "u_trainer" in st.session_state
        st.markdown(
            f'<div style="background:#07070f;border:1px solid rgba(52,211,153,.15);border-radius:8px;padding:.65rem .9rem;margin-top:.5rem;{JM};">'
            f'<div style="font-size:.55rem;color:#2a5a3a;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem">SESSION</div>'
            + (f'<div class="ms-row"><span class="ms-key">DATASET</span><span class="ms-val-green" style="font-size:.62rem">{dsn}</span></div>' if dsn else "")
            + f'<div class="ms-row"><span class="ms-key">STATUS</span><span class="{"ms-val-green" if trained else "ms-val-amber"}">'
            f'{"Model ready" if trained else "Not trained"}</span></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sidebar-footer">AutoML-X · HuggingFace Space</div>', unsafe_allow_html=True)

# ━━━ HOME ━━━
if page == "Home":
    active = st.session_state.get("u_trainer")
    st.markdown(
        '<div class="hero-wrap"><div class="hero-eyebrow">AutoML-X &nbsp;·&nbsp; Universal Binary Classifier</div>'
        '<div class="hero-title">Upload any CSV.<br>Get a trained <span>ML model</span><br>in minutes.</div>'
        '<div class="hero-sub">No code. No configuration. AutoML-X reads your data, selects the right algorithm, trains, evaluates, and explains — all automatically. Built for fintech, healthcare, HR, and credit risk.</div>'
        '<div class="hero-badges">'
        '<span class="hero-badge" style="color:#34d399;border-color:rgba(52,211,153,.25);background:rgba(52,211,153,.05)">✓ Auto model selection</span>'
        '<span class="hero-badge" style="color:#6366f1;border-color:rgba(99,102,241,.25);background:rgba(99,102,241,.05)">✓ 6 dataset tiers</span>'
        '<span class="hero-badge" style="color:#60a5fa;border-color:rgba(96,165,250,.25);background:rgba(96,165,250,.05)">✓ SHAP explainability</span>'
        '<span class="hero-badge" style="color:#fbbf24;border-color:rgba(251,191,36,.25);background:rgba(251,191,36,.05)">✓ Up to 2M rows</span>'
        '<span class="hero-badge" style="color:#c084fc;border-color:rgba(192,132,252,.25);background:rgba(192,132,252,.05)">✓ PDF report export</span>'
        '<span class="hero-badge" style="color:#f87171;border-color:rgba(248,113,113,.25);background:rgba(248,113,113,.05)">✓ Drift detection</span></div></div>',
        unsafe_allow_html=True,
    )
    cta1, _, __ = st.columns([1.2, 1.2, 3])
    with cta1:
        if st.button("Start with your CSV", type="primary", use_container_width=True):
            st.session_state["_nav"] = "01 -- Upload"
            st.rerun()

    sec("How it works")
    st.markdown(
        '<div class="flow-wrap">'
        '<div class="flow-step"><div class="flow-num">01</div><span class="flow-icon">📁</span>'
        '<div class="flow-title">Upload CSV</div><div class="flow-desc">Drop any binary classification dataset. Auto-detects column types, handles missing values, selects processing tier.</div></div>'
        '<div class="flow-arrow">→</div>'
        '<div class="flow-step"><div class="flow-num">02</div><span class="flow-icon">⚡</span>'
        '<div class="flow-title">Auto-Train</div><div class="flow-desc">Multiple models compared — Logistic Regression, LightGBM, XGBoost. Cross-validated, threshold-optimised. Best model selected.</div></div>'
        '<div class="flow-arrow">→</div>'
        '<div class="flow-step"><div class="flow-num">03</div><span class="flow-icon">📊</span>'
        '<div class="flow-title">Explain & Export</div><div class="flow-desc">SHAP values, ROC/PR curves, drift detection on new data. Download model (.pkl) or PDF report.</div></div></div>',
        unsafe_allow_html=True,
    )

# ━━━ UPLOAD ━━━
elif page == "01 -- Upload":
    ph("STEP 01 / 06", "Upload Dataset", "Drop a binary classification CSV. Auto-detection runs automatically.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        file_mb = uploaded.size / 1e6
        if file_mb > 50:
            st.markdown(f'<div class="info-panel-warn"><span style="{JM};font-size:.72rem;color:#fbbf24">△ Large file ({file_mb:.0f} MB) — chunked loader active</span></div>', unsafe_allow_html=True)
            with st.spinner("Loading…"):
                df = load_csv_chunked(uploaded, max_rows=None, chunk_size=100_000)
        else:
            df = pd.read_csv(uploaded)
        render_tier(get_tier(len(df)), len(df))
        st.session_state["df"] = df
        st.session_state.pop("profile", None)
        st.session_state.pop("u_trainer", None)
        st.session_state["dataset_name"] = uploaded.name
        st.success(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
        st.button("Next: Preview", key="next_upload")

# ━━━ ANALYZE ━━━
elif page == "02 -- Analyze":
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload a dataset first on Step 01.")
        st.stop()
    
    ph("STEP 02 / 06", "Data Analysis", "Profile, clean, and detect data quality issues.")
    df = st.session_state["df"]
    
    profiler = DatasetProfiler(df)
    profile = profiler.profile()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    sec("Column Analysis")
    render_col_table(profile)
    
    sec("Data Quality")
    quality = DataQualityReport(df, profile).generate()
    for issue in quality.get("issues", []):
        if issue["severity"] == "error":
            st.error(issue["message"])
        elif issue["severity"] == "warning":
            st.warning(issue["message"])
        else:
            st.info(issue["message"])

# ━━━ TRAIN ━━━
elif page == "03 -- Train":
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Upload dataset first.")
        st.stop()
    
    ph("STEP 03 / 06", "Train Model", "Select target & preprocessing, then train.")
    df = st.session_state["df"]
    target_col = st.selectbox("Target column (binary)", df.columns, index=len(df.columns)-1)
    st.session_state["target_col"] = target_col
    
    col1, col2 = st.columns(2)
    with col1:
        valid_pct = st.slider("Validation split %", 10, 40, 20)
    with col2:
        pos_raw = st.selectbox("Positive class", ["Auto-detect"] + sorted(df[target_col].unique().astype(str).tolist()))
        st.session_state["positive_label"] = None if pos_raw == "Auto-detect" else pos_raw
    
    if st.button("Train Model", type="primary", use_container_width=True):
        with st.spinner("Training…"):
            trainer = UniversalTrainer()
            metrics = trainer.fit(df, target_col, val_pct=valid_pct/100)
            st.session_state["u_trainer"] = trainer
            st.session_state["drift_detector"] = DriftDetector() if _DRIFT else None
            if _DRIFT and trainer:
                st.session_state["drift_detector"].fit(trainer.X_train_processed, trainer.feature_names)
                st.session_state["drift_features"] = trainer.feature_names
            st.success("✓ Training complete!")
            st.json(metrics)

# ━━━ RESULTS ━━━
elif page == "04 -- Results":
    trainer = st.session_state.get("u_trainer")
    if trainer is None:
        no_model_msg()
        st.stop()
    
    ph("STEP 04 / 06", "Results & Evaluation", "View metrics, ROC curves, SHAP explanations.")
    metrics = trainer.metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.5f}")
    with col2:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.5f}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.5f}")
    with col4:
        st.metric("Precision", f"{metrics.get('precision', 0):.5f}")
    
    sec("ROC Curve")
    if _EVAL and "val_probs" in metrics:
        fig = plot_roc_curve(np.array(metrics["val_labels"]), np.array(metrics["val_probs"]), trainer.best_model_name)
        if fig:
            st.pyplot(fig)
            plt.close()

# ━━━ PREDICT ━━━
elif page == "05 -- Predict":
    trainer = st.session_state.get("u_trainer")
    if trainer is None:
        no_model_msg()
        st.stop()
    
    ph("STEP 05 / 06", "Single Prediction", "Enter data to score one sample.")
    inputs = {}
    for feat in trainer.feature_names[:5]:
        inputs[feat] = st.number_input(feat)
    
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([inputs])
        prob = trainer.predict_proba(X)[0]
        st.markdown(f'<div class="fraud-alert" style="background:rgba(52,211,153,.05);border-color:#34d399"><div class="result-label">Prediction</div><div class="result-prob" style="color:#34d399">{prob*100:.2f}%</div></div>', unsafe_allow_html=True)

# ━━━ BATCH ━━━
elif page == "06 -- Batch":
    trainer = st.session_state.get("u_trainer")
    if trainer is None:
        no_model_msg()
        st.stop()
    
    ph("STEP 06 / 06", "Batch Prediction", "Score a full CSV file.")
    uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
    if uploaded:
        df_new = pd.read_csv(uploaded)
        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(df_new):,} rows…"):
                probs = trainer.predict_proba(df_new)
                results = pd.DataFrame({
                    "probability": [f"{p*100:.4f}" for p in probs],
                    "prediction": ["POSITIVE" if p >= trainer.threshold else "NEGATIVE" for p in probs],
                })
                st.markdown(f'<div class="stat-row stat-row-3"><div class="stat-card"><span class="stat-val">{len(results):,}</span><span class="stat-lbl">Total scored</span></div></div>', unsafe_allow_html=True)
                st.dataframe(results.head(50))
                st.download_button("Download Predictions", results.to_csv(index=False), "predictions.csv")

st.markdown('<div class="sidebar-footer">AutoML-X · HuggingFace Space</div>', unsafe_allow_html=True)
