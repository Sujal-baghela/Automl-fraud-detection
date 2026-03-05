"""
app_universal.py
════════════════
AutoML-X Universal Trainer — PUBLIC HuggingFace Space
Runs on port 7860. No fraud detection code here.
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.universal_trainer import (
    UniversalTrainer, DatasetProfiler,
    ColumnTypeDetector, ComplexityDetector,
    check_ram_safety, load_csv_chunked,
    get_tier, TIER_LABELS, TIER_STRATEGY,
    TIER_TINY, TIER_SMALL, TIER_MEDIUM,
    TIER_LARGE, TIER_XLARGE, TIER_MASSIVE,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML-X Universal Trainer",
    page_icon="🤖",
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
.metric-value { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; color:#00aaff; }
.metric-label { font-size:.8rem; color:#666; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.pos-alert    { background:linear-gradient(135deg,#1a0a0a,#2a0a0a); border:1px solid #ff4444; border-radius:12px; padding:24px; text-align:center; }
.neg-alert    { background:linear-gradient(135deg,#0a1a0a,#0a2a0a); border:1px solid #00ff88; border-radius:12px; padding:24px; text-align:center; }

.col-badge   { display:inline-block; padding:2px 10px; border-radius:12px; font-size:.72rem; font-weight:600; margin:2px; font-family:'Space Mono',monospace; }
.col-numeric { background:#0a2a1a; border:1px solid #00cc66; color:#00cc66; }
.col-cat     { background:#1a1a0a; border:1px solid #ffcc00; color:#ffcc00; }
.col-id      { background:#2a0a0a; border:1px solid #ff4444; color:#ff4444; }
.col-date    { background:#0a1a2a; border:1px solid #00aaff; color:#00aaff; }
.col-text    { background:#2a0a2a; border:1px solid #cc00ff; color:#cc00ff; }

.tier-banner { border-radius:10px; padding:14px 20px; margin:10px 0; }
.tier-0      { background:#0a1a0a; border:1px solid #00cc66; }
.tier-1      { background:#0a1a12; border:1px solid #00cc88; }
.tier-2      { background:#0a1220; border:1px solid #00aaff; }
.tier-3      { background:#1a120a; border:1px solid #ffaa00; }
.tier-4      { background:#1a0a0a; border:1px solid #ff6600; }
.tier-5      { background:#2a0a0a; border:1px solid #ff2222; }

.complexity-box { background:#13131a; border:1px solid #1e1e2e; border-radius:10px; padding:16px 20px; margin:8px 0; }
.info-chip { display:inline-block; background:#1a1a2e; border-radius:6px; padding:3px 8px; font-size:.75rem; color:#aaa; margin:2px; font-family:'Space Mono',monospace; }
.profile-row      { display:flex; align-items:center; padding:6px 0; border-bottom:1px solid #1a1a2a; }
.profile-col-name { font-family:'Space Mono',monospace; font-size:.8rem; color:#ccc; flex:2; }
.profile-col-type { flex:1; }
.profile-col-stats{ flex:3; font-size:.78rem; color:#888; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_universal_model():
    try:
        trainer = UniversalTrainer()
        trainer.load("models/universal_model.pkl")
        return trainer
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

TIER_ICONS = {
    TIER_TINY:    ("🔵", "#00cc66", "Tiny"),
    TIER_SMALL:   ("🟢", "#00cc88", "Small"),
    TIER_MEDIUM:  ("🔷", "#00aaff", "Medium"),
    TIER_LARGE:   ("🟠", "#ffaa00", "Large"),
    TIER_XLARGE:  ("🔶", "#ff6600", "XLarge"),
    TIER_MASSIVE: ("🔴", "#ff2222", "Massive"),
}

def render_tier_banner(tier: int, n_rows: int = None):
    icon, color, name = TIER_ICONS[tier]
    label    = TIER_LABELS[tier]
    strategy = TIER_STRATEGY[tier]
    rows_str = f" · {n_rows:,} rows" if n_rows else ""
    st.markdown(f"""
    <div class="tier-banner tier-{tier}">
      <span style="font-size:1.1rem;font-weight:700;color:{color}">{icon} Dataset Tier: {name}</span>
      <span style="color:#888;font-size:.85rem;margin-left:12px">{label}{rows_str}</span>
      <div style="color:#aaa;font-size:.78rem;margin-top:4px;font-family:'Space Mono',monospace">
        Strategy → {strategy}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_col_badge(col_type: str) -> str:
    cls_map   = {"numeric":"col-numeric","categorical":"col-cat",
                 "id_dropped":"col-id","date":"col-date","text":"col-text"}
    label_map = {"numeric":"NUM","categorical":"CAT",
                 "id_dropped":"ID ✕","date":"DATE ✕","text":"TEXT ✕"}
    css   = cls_map.get(col_type, "col-numeric")
    label = label_map.get(col_type, col_type.upper())
    return f'<span class="col-badge {css}">{label}</span>'


def render_profile_table(profile: dict):
    col_stats = profile.get("col_stats", {})
    if not col_stats:
        return
    rows_html = ""
    for col, stats in col_stats.items():
        col_type = stats.get("type", "unknown")
        badge    = render_col_badge(col_type)
        if col_type == "numeric":
            detail = (
                f'mean={stats.get("mean","?")} &nbsp;|&nbsp; '
                f'std={stats.get("std","?")} &nbsp;|&nbsp; '
                f'min={stats.get("min","?")} &nbsp;|&nbsp; '
                f'max={stats.get("max","?")} &nbsp;|&nbsp; '
                f'skew={stats.get("skew","?")} &nbsp;|&nbsp; '
                f'{stats.get("missing_pct",0):.1f}% missing'
            )
        elif col_type == "categorical":
            top     = stats.get("top_values", {})
            top_str = " · ".join(f"{k}({v})" for k, v in list(top.items())[:3])
            detail  = (f'{stats.get("n_unique","?")} unique &nbsp;|&nbsp; '
                       f'top: {top_str} &nbsp;|&nbsp; '
                       f'{stats.get("missing_pct",0):.1f}% missing')
        else:
            detail = stats.get("note", col_type)
        rows_html += f"""
        <div class="profile-row">
          <div class="profile-col-name">{col}</div>
          <div class="profile-col-type">{badge}</div>
          <div class="profile-col-stats">{detail}</div>
        </div>"""
    st.markdown(
        f'<div style="max-height:420px;overflow-y:auto;padding:4px 0">{rows_html}</div>',
        unsafe_allow_html=True
    )


def render_complexity_box(cplx: dict):
    c = cplx.get("complexity", "unknown")
    color_map = {"linear":"#00aaff","nonlinear":"#ff8800",
                 "mixed":"#00ff88","unknown":"#888"}
    icon_map  = {"linear":"📐","nonlinear":"🌳","mixed":"⚖️","unknown":"❓"}
    color = color_map.get(c, "#888")
    icon  = icon_map.get(c, "❓")
    lr    = cplx.get("lr_score",  "N/A")
    lgb   = cplx.get("lgb_score", "N/A")
    rec   = cplx.get("recommended", "All models")
    note  = cplx.get("note", "")
    st.markdown(f"""
    <div class="complexity-box" style="border-color:{color}">
      <div style="font-size:1.05rem;font-weight:700;color:{color};margin-bottom:6px">
        {icon} Complexity: <span style="text-transform:uppercase">{c}</span>
      </div>
      <div style="font-size:.82rem;color:#aaa;margin-bottom:6px">{note}</div>
      <span class="info-chip">LR AUC: {lr}</span>
      <span class="info-chip">LGB AUC: {lgb}</span>
      <span class="info-chip">→ {rec}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 AutoML-X")
    st.markdown("*Universal Trainer*")
    st.divider()

    trainer_loaded = load_universal_model()
    if trainer_loaded:
        m = trainer_loaded.metrics
        st.markdown("### ✅ Model Loaded")
        st.markdown(f"**Model:** `{m.get('best_model','N/A')}`")
        st.markdown(f"**Target:** `{trainer_loaded.target_col}`")
        st.markdown(f"**ROC-AUC:** `{m.get('test_roc_auc',0):.5f}`")
        st.markdown(f"**Tier:** `{m.get('tier_label','?')}`")
        cplx = m.get("complexity", {})
        if cplx:
            st.markdown(f"**Complexity:** `{cplx.get('complexity','?')}`")
    else:
        st.info("No model trained yet.")

    st.divider()
    page = st.radio("Navigation", [
        "📁 Upload & Configure",
        "🚀 Train Model",
        "📊 Results",
        "🔍 Predict",
        "📂 Batch Predict",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("AutoML-X Platform · v5.0 · Universal")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🤖 AutoML-X Universal Trainer")
st.markdown("*Train any binary classification dataset — 100 rows to 2M+ rows — automatically*")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Upload & Configure
# ══════════════════════════════════════════════════════════════════════════════

if page == "📁 Upload & Configure":
    st.subheader("📁 Upload Your Dataset")

    st.markdown("""
    <div style="background:#0d0d1a;border:1px solid #1e1e2e;border-radius:10px;padding:14px 18px;margin-bottom:16px">
    <div style="font-family:'Space Mono',monospace;font-size:.72rem;color:#555;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px">Dataset Size Tiers — Auto-selected Strategy</div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;font-size:.78rem">
      <span style="color:#00cc66">🔵 <b>Tiny</b> &lt;1K · 4 models · 5-fold CV</span>
      <span style="color:#00cc88">🟢 <b>Small</b> 1K–50K · 4 models · 5-fold CV</span>
      <span style="color:#00aaff">🔷 <b>Medium</b> 50K–200K · 4 models · 3-fold CV</span>
      <span style="color:#ffaa00">🟠 <b>Large</b> 200K–500K · 3 models · 2-fold CV</span>
      <span style="color:#ff6600">🔶 <b>XLarge</b> 500K–2M · LGB+LR · 2-fold on 200K sample</span>
      <span style="color:#ff2222">🔴 <b>Massive</b> 2M+ · LGB only · no CV · chunked predict</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.75rem;color:#888;margin-bottom:12px">
    <span class="col-badge col-numeric">NUM</span> Numeric feature &nbsp;
    <span class="col-badge col-cat">CAT</span> Categorical &nbsp;
    <span class="col-badge col-id">ID ✕</span> Auto-dropped &nbsp;
    <span class="col-badge col-date">DATE ✕</span> Auto-dropped &nbsp;
    <span class="col-badge col-text">TEXT ✕</span> Auto-dropped
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose CSV file (any size)", type=["csv"])

    if uploaded:
        file_size_mb = uploaded.size / 1e6
        is_large     = file_size_mb > 50
        st.markdown(f"📎 **{uploaded.name}** — {file_size_mb:.1f} MB")

        col1, col2 = st.columns(2)
        with col1:
            if is_large:
                max_rows = st.slider(
                    "Max rows to load",
                    min_value=100_000, max_value=2_000_000,
                    value=2_000_000, step=100_000,
                    format="%d rows",
                    help="Tier-aware training handles everything from here."
                )
            else:
                max_rows = None
                st.success(f"✅ Small file ({file_size_mb:.1f} MB) — loading everything")
        with col2:
            target_hint = st.text_input(
                "Target column (for stratified sampling on huge files)",
                value="", placeholder="e.g. Churn, fraud, label"
            )
            target_hint = target_hint.strip() or None

        if st.button("📂 Load Dataset", type="primary", use_container_width=True):
            placeholder = st.empty()

            def chunk_cb(msg):
                placeholder.info(f"⏳ {msg}")

            with st.spinner("Loading…"):
                try:
                    if is_large:
                        df = load_csv_chunked(
                            uploaded, max_rows=max_rows,
                            chunk_size=100_000, target_col=target_hint,
                            progress_callback=chunk_cb
                        )
                    else:
                        df = pd.read_csv(uploaded)

                    placeholder.empty()
                    st.session_state["df"]       = df
                    st.session_state["filename"] = uploaded.name
                    st.session_state.pop("profile", None)

                    tier = get_tier(len(df))
                    st.success(
                        f"✅ Loaded **{uploaded.name}** — "
                        f"{len(df):,} rows × {df.shape[1]} cols"
                    )
                    render_tier_banner(tier, len(df))

                except Exception as e:
                    placeholder.empty()
                    st.error(f"Load failed: {e}")

    if "df" in st.session_state and st.session_state["df"] is not None:
        df   = st.session_state["df"]
        ram  = check_ram_safety(df)
        tier = get_tier(len(df))

        if not ram["is_safe"]:
            st.warning(f"⚠️ {ram['warning']}")
        else:
            st.success(
                f"✅ RAM OK · dataset {ram['dataframe_gb']:.3f} GB "
                f"· {ram['available_gb']:.1f} GB free"
            )

        render_tier_banner(tier, len(df))
        st.dataframe(df.head(8), use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",    f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", f"{df.isnull().sum().sum():,}")
        c4.metric("Size",    f"{ram['dataframe_gb']:.3f} GB")

        st.divider()
        st.subheader("⚙️ Configure")
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("🎯 Target Column", df.columns.tolist(),
                                      index=len(df.columns) - 1)
            st.session_state["target_col"] = target_col
        with col2:
            unique_vals = df[target_col].dropna().unique().tolist()
            pos_raw = st.selectbox("✅ Positive Value",
                                   ["Auto-detect"] + [str(v) for v in unique_vals])
            st.session_state["positive_label"] = (
                None if pos_raw == "Auto-detect" else pos_raw
            )

        st.divider()
        if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
            with st.spinner("Running smart analysis…"):
                try:
                    profiler = DatasetProfiler()
                    profile  = profiler.profile(df, target_col)
                    st.session_state["profile"] = profile
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "profile" in st.session_state and st.session_state["profile"]:
            profile = st.session_state["profile"]
            st.divider()
            st.subheader("📊 Dataset Analysis Report")

            render_tier_banner(profile["tier"], profile["n_rows"])
            st.caption(f"Training strategy: **{profile['tier_strategy']}**")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Numeric",     profile["n_numeric"])
            c2.metric("Categorical", profile["n_categorical"])
            c3.metric("IDs Dropped", profile["n_id_dropped"])
            c4.metric("Dates Drop.", profile["n_date_dropped"])
            c5.metric("Missing %",   f"{profile['missing_pct']}%")
            c6.metric("Imbalanced",  "Yes ⚠️" if profile["is_imbalanced"] else "No ✅")

            st.divider()
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**🎯 Class Distribution**")
                class_df = pd.DataFrame.from_dict(
                    profile["class_counts"], orient="index", columns=["Count"]
                )
                class_df["Pct"] = (
                    class_df["Count"] / class_df["Count"].sum() * 100
                ).round(2)
                st.dataframe(class_df, use_container_width=True)
                if profile["n_classes"] != 2:
                    st.error(f"❌ {profile['n_classes']} classes — only binary supported.")
                elif profile["is_imbalanced"]:
                    st.warning(
                        f"⚠️ Imbalanced: minority = "
                        f"{profile['minority_ratio']*100:.1f}% "
                        f"— class_weight='balanced' will be applied"
                    )
                else:
                    st.success("✅ Binary & balanced. Ready to train!")
            with cb:
                try:
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    labels  = [str(k) for k in profile["class_counts"].keys()]
                    values  = list(profile["class_counts"].values())
                    colors  = ["#00ff88", "#ff4444"] if len(values) >= 2 else ["#00aaff"]
                    ax.bar(labels, values, color=colors[:len(values)])
                    ax.set_title("Class Distribution", color="white")
                    fig.patch.set_facecolor("#13131a")
                    ax.set_facecolor("#13131a")
                    ax.tick_params(colors="white")
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#333")
                    st.pyplot(fig)
                    plt.close()
                except Exception:
                    pass

            if profile.get("high_corr_pairs"):
                st.divider()
                st.markdown("**⚠️ Highly Correlated Features (>0.95)**")
                st.dataframe(
                    pd.DataFrame(profile["high_corr_pairs"],
                                 columns=["Feature A", "Feature B", "Correlation"]),
                    use_container_width=True, hide_index=True
                )

            st.divider()
            st.markdown("**📋 Per-Column Profile**")
            note_parts = []
            if profile["id_cols"]:
                note_parts.append(
                    f'<span class="info-chip">🗑️ Dropping {len(profile["id_cols"])} ID col(s): '
                    f'{", ".join(profile["id_cols"][:4])}'
                    f'{"…" if len(profile["id_cols"]) > 4 else ""}</span>'
                )
            if profile["date_cols"]:
                note_parts.append(
                    f'<span class="info-chip">📅 {len(profile["date_cols"])} date col(s) dropped</span>'
                )
            if profile["text_cols"]:
                note_parts.append(
                    f'<span class="info-chip">📝 {len(profile["text_cols"])} text col(s) dropped</span>'
                )
            if note_parts:
                st.markdown(" ".join(note_parts), unsafe_allow_html=True)
                st.markdown("")

            render_profile_table(profile)

            num_cols = profile["numeric_cols"]
            if num_cols:
                st.divider()
                st.markdown("**📈 Numeric Feature Distributions (top 8)**")
                show_cols = num_cols[:8]
                n    = len(show_cols)
                nc_  = min(4, n)
                nr_  = (n + nc_ - 1) // nc_
                fig, axes = plt.subplots(nr_, nc_, figsize=(4 * nc_, 2.5 * nr_))
                axes_flat = np.array(axes).flatten() if n > 1 else [axes]
                for i, col in enumerate(show_cols):
                    ax   = axes_flat[i]
                    data = df[col].dropna().sample(
                        min(50_000, df[col].notna().sum()), random_state=42
                    )
                    ax.hist(data, bins=30, color="#00aaff", alpha=0.7, edgecolor="none")
                    ax.set_title(col, color="white", fontsize=8)
                    ax.set_facecolor("#0d0d1a")
                    ax.tick_params(colors="#666", labelsize=6)
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#222")
                for j in range(n, len(axes_flat)):
                    axes_flat[j].set_visible(False)
                fig.patch.set_facecolor("#13131a")
                fig.tight_layout(pad=1.0)
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Train Model
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚀 Train Model":
    st.subheader("🚀 Train AutoML Model")

    if "df" not in st.session_state or st.session_state.get("df") is None:
        st.warning("⚠️ Upload a dataset first.")
        st.stop()

    df         = st.session_state["df"]
    target_col = st.session_state.get("target_col")
    pos_label  = st.session_state.get("positive_label")

    if not target_col:
        st.warning("⚠️ Set the target column first.")
        st.stop()

    tier = get_tier(len(df))
    render_tier_banner(tier, len(df))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset",   st.session_state.get("filename", "Uploaded"))
    c2.metric("Rows",      f"{len(df):,}")
    c3.metric("Target",    target_col)
    c4.metric("Pos.Label", pos_label or "Auto")

    st.divider()
    with st.expander("ℹ️ Auto-selected training strategy for this tier", expanded=True):
        strategies = {
            TIER_TINY:    "**Tiny (<1K)** → All 4 models (LR+RF+LGB+XGB) · 5-fold CV · full data.",
            TIER_SMALL:   "**Small (1K–50K)** → All 4 models · 5-fold CV · full data.",
            TIER_MEDIUM:  "**Medium (50K–200K)** → All 4 models · 3-fold CV · reduced n_estimators.",
            TIER_LARGE:   "**Large (200K–500K)** → LR + LGB + XGB · 2-fold CV · RF dropped (too slow).",
            TIER_XLARGE:  "**XLarge (500K–2M)** → LGB + LR · 2-fold CV on 200K subsample → full fit on 500K.",
            TIER_MASSIVE: "**Massive (2M+)** → LightGBM only · no CV · 500K train sample · chunked 100K inference.",
        }
        st.markdown(strategies[tier])
        st.markdown("""
        **All tiers include:**
        - ✅ Smart column type detection (IDs/dates/text auto-dropped)
        - ✅ Complexity detection (linear vs non-linear → picks best models)
        - ✅ Optimal threshold search (max F1)
        - ✅ Chunked predict for large inference batches
        """)

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        log_area     = st.empty()
        log_lines    = []

        def update_progress(step, total, msg):
            progress_bar.progress(step / total)
            status_text.markdown(f"**Step {step}/{total}:** {msg}")
            log_lines.append(f"[{step}/{total}] {msg}")
            log_area.code("\n".join(log_lines[-8:]))

        try:
            trainer = UniversalTrainer(model_save_path="models/universal_model.pkl")
            metrics = trainer.fit(
                df=df,
                target_col=target_col,
                positive_label=pos_label,
                progress_callback=update_progress
            )
            st.session_state["u_trainer"] = trainer
            st.session_state["u_metrics"] = metrics
            progress_bar.progress(1.0)
            log_area.empty()
            status_text.empty()

            st.success(
                f"🎉 Training complete! "
                f"Best model: **{metrics['best_model']}** · "
                f"ROC-AUC: **{metrics['test_roc_auc']:.5f}**"
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Best Model", metrics["best_model"])
            c2.metric("ROC-AUC",    f"{metrics['test_roc_auc']:.5f}")
            c3.metric("F1",         f"{metrics['f1_score']:.5f}")
            c4.metric("Recall",     f"{metrics['recall']:.5f}")
            c5.metric("Features",   metrics.get("n_features_used", "?"))

            cplx = metrics.get("complexity")
            if cplx:
                st.divider()
                render_complexity_box(cplx)

            dropped = metrics.get("dropped_cols", [])
            if dropped:
                st.info(
                    f"🗑️ Auto-dropped {len(dropped)} col(s): "
                    f"`{'`, `'.join(dropped[:6])}`"
                    f"{'…' if len(dropped) > 6 else ''}"
                )

            st.info("👉 Go to **📊 Results** for full breakdown.")

        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Results
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Results":
    trainer = st.session_state.get("u_trainer") or load_universal_model()
    if trainer is None:
        st.warning("⚠️ No model trained yet.")
        st.stop()

    metrics = trainer.metrics
    st.subheader("📊 Performance Metrics")

    render_tier_banner(metrics.get("tier", 0), metrics.get("n_rows_total"))
    st.caption(f"Strategy used: **{metrics.get('tier_strategy','N/A')}**")
    st.caption(
        f"Trained on {metrics.get('n_train',0):,} rows · "
        f"evaluated on {metrics.get('n_val',0):,} rows"
    )

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model",   metrics["best_model"])
    c2.metric("CV ROC-AUC",   f"{metrics['cv_roc_auc']:.5f}")
    c3.metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.5f}")
    c4.metric("F1 Score",     f"{metrics['f1_score']:.5f}")
    c5.metric("Recall",       f"{metrics['recall']:.5f}")

    cplx = metrics.get("complexity")
    if cplx:
        st.divider()
        render_complexity_box(cplx)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**🏆 Model Competition**")
        scores_df = pd.DataFrame({
            "Model":      list(metrics["all_cv_scores"].keys()),
            "CV ROC-AUC": list(metrics["all_cv_scores"].values()),
        }).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)

        def hl(row):
            return ["background-color:#0a2a1a;color:#00ff88" if row.name == 0 else "" for _ in row]
        st.dataframe(scores_df.style.apply(hl, axis=1),
                     use_container_width=True, hide_index=True)
    with cb:
        try:
            fig, ax = plt.subplots(figsize=(4, 3))
            ms = scores_df.sort_values("CV ROC-AUC")
            colors = ["#00ff88" if i == len(ms) - 1 else "#00aaff" for i in range(len(ms))]
            ax.barh(ms["Model"], ms["CV ROC-AUC"], color=colors, height=0.5)
            ax.set_xlim(max(0, ms["CV ROC-AUC"].min() - 0.05), 1.0)
            ax.set_xlabel("CV ROC-AUC", color="white")
            fig.patch.set_facecolor("#13131a")
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#222")
            st.pyplot(fig)
            plt.close()
        except Exception:
            pass

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", f"{metrics['threshold']:.5f}")
    c2.metric("Precision", f"{metrics['precision']:.5f}")
    c3.metric("Recall",    f"{metrics['recall']:.5f}")
    c4.metric("F1",        f"{metrics['f1_score']:.5f}")
    tp, tn = metrics["TP"], metrics["TN"]
    fp, fn = metrics["FP"], metrics["FN"]
    st.markdown(f"**Confusion Matrix** · TP:`{tp}` TN:`{tn}` FP:`{fp}` FN:`{fn}`")

    dropped = metrics.get("dropped_cols", [])
    if dropped:
        st.divider()
        st.markdown(f"**🗑️ Auto-dropped:** `{'`, `'.join(dropped)}`")

    st.divider()
    if os.path.exists("models/universal_model.pkl"):
        with open("models/universal_model.pkl", "rb") as f:
            st.download_button(
                "⬇️ Download Model", f,
                "universal_model.pkl", "application/octet-stream",
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Single Predict
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Predict":
    trainer = st.session_state.get("u_trainer") or load_universal_model()
    if trainer is None:
        st.warning("⚠️ Train a model first.")
        st.stop()

    st.subheader("🔍 Single Prediction")
    features   = trainer.feature_names
    df_ref     = st.session_state.get("df")
    input_vals = {}
    col_list   = st.columns(4)
    for i, feat in enumerate(features):
        with col_list[i % 4]:
            if df_ref is not None and feat in df_ref.columns:
                sample = df_ref[feat].dropna()
                if sample.dtype in [np.float64, np.int64, np.float32, np.int32]:
                    input_vals[feat] = st.number_input(
                        feat, value=float(sample.median()), format="%.4f"
                    )
                else:
                    input_vals[feat] = st.selectbox(feat, sample.unique().tolist())
            else:
                input_vals[feat] = st.number_input(feat, value=0.0)

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        try:
            probability = float(trainer.predict_proba(pd.DataFrame([input_vals]))[0])
            prediction  = int(probability >= trainer.threshold)
            t           = trainer.threshold
            if prediction == 1:
                st.markdown(f"""<div class="pos-alert">
                    <div style="font-size:3rem">⚠️</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#ff4444">POSITIVE</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#ff4444;font-family:monospace">{probability*100:.2f}%</div>
                    <div style="color:#888">Probability | Threshold: {t:.5f}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="neg-alert">
                    <div style="font-size:3rem">✅</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#00ff88">NEGATIVE</div>
                    <div style="font-size:2.5rem;font-weight:700;color:#00ff88;font-family:monospace">{probability*100:.2f}%</div>
                    <div style="color:#888">Probability | Threshold: {t:.5f}</div>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Predict
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Batch Predict":
    trainer = st.session_state.get("u_trainer") or load_universal_model()
    if trainer is None:
        st.warning("⚠️ Train a model first.")
        st.stop()

    st.subheader("📂 Batch Prediction")
    st.info(
        f"Required features: `{'`, `'.join(trainer.feature_names[:5])}`… "
        f"({len(trainer.feature_names)} total) · "
        f"Chunked prediction (100K rows/chunk) — safe for any file size"
    )

    uploaded = st.file_uploader("Upload CSV (any size)", type=["csv"])
    if uploaded:
        file_size_mb = uploaded.size / 1e6
        if file_size_mb > 50:
            st.info(f"⚡ Large file ({file_size_mb:.0f} MB) — chunked loader active")
            with st.spinner("Loading…"):
                df_new = load_csv_chunked(uploaded, max_rows=None, chunk_size=100_000)
        else:
            df_new = pd.read_csv(uploaded)

        tier_inf = get_tier(len(df_new))
        st.success(f"Loaded {len(df_new):,} rows")
        render_tier_banner(tier_inf, len(df_new))

        X_raw = df_new.drop(columns=[trainer.target_col], errors="ignore")

        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Predicting {len(X_raw):,} rows (chunked 100K/batch)…"):
                try:
                    probabilities = trainer.predict_proba(X_raw)
                    predictions   = (probabilities >= trainer.threshold).astype(int)

                    results = df_new.copy()
                    results["probability_%"]   = (probabilities * 100).round(4)
                    results["predicted_class"] = predictions
                    results["label"] = [
                        "POSITIVE" if p == 1 else "NEGATIVE" for p in predictions
                    ]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total",    f"{len(results):,}")
                    c2.metric("Positive", f"{int(predictions.sum()):,}")
                    c3.metric("Rate",     f"{predictions.mean()*100:.2f}%")

                    st.dataframe(results, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Predictions",
                        results.to_csv(index=False).encode(),
                        "predictions.csv", "text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")