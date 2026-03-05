"""
app_universal.py  ·  AutoML-X v6.0
════════════════════════════════════
AutoML-X Universal Trainer — PUBLIC HuggingFace Space
Runs on port 7860.
Professional dark UI — industrial SaaS aesthetic.
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use("Agg")

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from src.universal_trainer import (
    UniversalTrainer, DatasetProfiler, DataQualityReport,
    SmartDataCleaner, ColumnTypeDetector, ComplexityDetector,
    check_ram_safety, load_csv_chunked,
    get_tier, TIER_LABELS, TIER_STRATEGY,
    TIER_TINY, TIER_SMALL, TIER_MEDIUM,
    TIER_LARGE, TIER_XLARGE, TIER_MASSIVE,
)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML-X",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080810 !important;
    color: #c8c8d8;
}
.main { background: #080810 !important; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d0d18; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 2px; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #05050e !important;
    border-right: 1px solid #13132a !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Hide Streamlit chrome ─────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Typography ────────────────────────────────────────────────────────── */
.brand-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #e8e8ff 0%, #7878ff 50%, #4040cc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #44445a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    color: #e8e8ff;
    letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
}
.page-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #44445a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── Cards ─────────────────────────────────────────────────────────────── */
.card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f0f22 100%);
    border: 1px solid #22224a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card-accent::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4040cc, #7878ff, #40ccaa);
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-grid { display: grid; gap: 0.8rem; }
.metric-grid-2 { grid-template-columns: repeat(2, 1fr); }
.metric-grid-3 { grid-template-columns: repeat(3, 1fr); }
.metric-grid-4 { grid-template-columns: repeat(4, 1fr); }
.metric-grid-5 { grid-template-columns: repeat(5, 1fr); }

.m-card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.m-card:hover { border-color: #3030aa; }
.m-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #7878ff;
    line-height: 1.2;
    display: block;
}
.m-val-lg { font-size: 2rem; }
.m-val-green { color: #40ccaa; }
.m-val-amber { color: #ccaa40; }
.m-val-red   { color: #cc4040; }
.m-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #44445a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.3rem;
    display: block;
}

/* ── Tier badges ───────────────────────────────────────────────────────── */
.tier-strip {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    margin: 0.8rem 0;
    border-left: 3px solid;
}
.tier-0 { background: #080f10; border-color: #40ccaa; }
.tier-1 { background: #080f10; border-color: #40cc88; }
.tier-2 { background: #08100f; border-color: #4088ff; }
.tier-3 { background: #100d08; border-color: #ccaa40; }
.tier-4 { background: #100808; border-color: #cc6640; }
.tier-5 { background: #100808; border-color: #cc4040; }

.tier-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
}
.tier-detail {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #55556a;
}
.tier-strategy {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #44445a;
    margin-top: 0.15rem;
}

/* ── Column type badges ────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin: 2px;
}
.b-num  { background: #0a1f14; border: 1px solid #40cc88; color: #40cc88; }
.b-cat  { background: #1a1800; border: 1px solid #ccaa40; color: #ccaa40; }
.b-id   { background: #1a0808; border: 1px solid #cc4040; color: #cc4040; }
.b-date { background: #080f1a; border: 1px solid #4088ff; color: #4088ff; }
.b-text { background: #180a1a; border: 1px solid #aa40cc; color: #aa40cc; }

/* ── Complexity block ──────────────────────────────────────────────────── */
.complexity-panel {
    background: #0d0d1a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border-left: 3px solid;
    margin: 0.8rem 0;
}
.cplx-linear    { border-color: #4088ff; }
.cplx-nonlinear { border-color: #cc8840; }
.cplx-mixed     { border-color: #40ccaa; }
.cplx-unknown   { border-color: #55556a; }

.cplx-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.3rem;
}
.cplx-note {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #55556a;
    margin-bottom: 0.6rem;
}
.chip {
    display: inline-block;
    background: #13132a;
    border: 1px solid #22224a;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #7878aa;
    margin: 2px;
}

/* ── Quality report ────────────────────────────────────────────────────── */
.quality-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #111120;
    font-size: 0.82rem;
}
.q-icon { font-size: 0.9rem; flex-shrink: 0; margin-top: 1px; }
.q-msg  { color: #9898aa; flex: 1; font-family: 'DM Sans', sans-serif; }

/* ── Column profile table ──────────────────────────────────────────────── */
.col-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid #0f0f1e;
}
.col-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #c8c8d8;
    flex: 0 0 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.col-type-cell { flex: 0 0 80px; }
.col-stats {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #55556a;
    flex: 1;
}
.col-miss {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #7878aa;
    flex: 0 0 80px;
    text-align: right;
}

/* ── Result cards ──────────────────────────────────────────────────────── */
.result-positive {
    background: linear-gradient(135deg, #100808, #1a0a0a);
    border: 1px solid #cc4040;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #081008, #0a180a);
    border: 1px solid #40cc88;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 0.4rem;
}
.result-prob {
    font-family: 'DM Mono', monospace;
    font-size: 3rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #44445a;
    letter-spacing: 1px;
}

/* ── Nav pills in sidebar ──────────────────────────────────────────────── */
[data-testid="stRadio"] > div {
    gap: 0.3rem;
}
[data-testid="stRadio"] label {
    border-radius: 6px !important;
    padding: 0.45rem 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #55556a !important;
    cursor: pointer;
    transition: all 0.15s;
    border: 1px solid transparent !important;
}
[data-testid="stRadio"] label:hover {
    color: #9898cc !important;
    background: #0d0d1a !important;
    border-color: #22224a !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    color: #a0a0ff !important;
    background: #0f0f22 !important;
    border-color: #3030aa !important;
}

/* ── Streamlit overrides ───────────────────────────────────────────────── */
.stButton > button {
    background: #1a1a33 !important;
    color: #9898ff !important;
    border: 1px solid #2a2a55 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #22224a !important;
    border-color: #4444aa !important;
    color: #bbbbff !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2020aa, #3030cc) !important;
    color: #e8e8ff !important;
    border-color: #4040ff !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2828bb, #4040dd) !important;
}
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
.stProgress > div > div { background: linear-gradient(90deg, #3030aa, #6060ff) !important; border-radius: 2px; }
.stSelectbox > div > div,
.stTextInput > div > div { background: #0d0d1a !important; border-color: #1a1a2e !important; color: #c8c8d8 !important; border-radius: 8px !important; }
.stFileUploader { background: #0d0d1a !important; border: 1px dashed #22224a !important; border-radius: 10px !important; }
label { color: #7878aa !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
.stAlert { border-radius: 8px !important; }
hr { border-color: #111120 !important; }
.stExpander { background: #0d0d1a !important; border: 1px solid #1a1a2e !important; border-radius: 10px !important; }
.stExpander summary { font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; color: #7878aa !important; }

/* ── Progress bar log ──────────────────────────────────────────────────── */
.log-box {
    background: #05050e;
    border: 1px solid #111120;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #55558a;
    max-height: 180px;
    overflow-y: auto;
    line-height: 1.8;
}
.log-box .log-active { color: #7878ff; }
.log-box .log-done   { color: #40cc88; }

/* ── Sidebar model status ──────────────────────────────────────────────── */
.model-status-card {
    background: #0a0a18;
    border: 1px solid #1a1a30;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
}
.ms-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.2rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
}
.ms-key { color: #44445a; }
.ms-val { color: #9898cc; font-weight: 500; }

/* ── Step indicator ────────────────────────────────────────────────────── */
.step-bar {
    display: flex;
    gap: 0.4rem;
    margin-bottom: 1.5rem;
    align-items: center;
}
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #1a1a2e;
    border: 1px solid #22224a;
    flex-shrink: 0;
}
.step-dot.active { background: #4040cc; border-color: #6060ff; }
.step-dot.done   { background: #40cc88; border-color: #40cc88; }
.step-line { flex: 1; height: 1px; background: #111120; }
.step-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #44445a; }

/* ── Section header ────────────────────────────────────────────────────── */
.sec-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #c8c8e8;
    margin: 1.2rem 0 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #111120;
    margin-left: 0.5rem;
}

/* ── Upload zone ───────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background: #0d0d1a !important;
    border: 1px dashed #22224a !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #4444aa !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TIER_META = {
    TIER_TINY:    ("#40ccaa", "TINY",    "< 1K rows"),
    TIER_SMALL:   ("#40cc88", "SMALL",   "1K – 50K rows"),
    TIER_MEDIUM:  ("#4088ff", "MEDIUM",  "50K – 200K rows"),
    TIER_LARGE:   ("#ccaa40", "LARGE",   "200K – 500K rows"),
    TIER_XLARGE:  ("#cc6640", "XLARGE",  "500K – 2M rows"),
    TIER_MASSIVE: ("#cc4040", "MASSIVE", "2M+ rows"),
}

def render_tier(tier: int, n_rows: int = None):
    color, name, rng = TIER_META[tier]
    strategy = TIER_STRATEGY[tier]
    row_str  = f" — {n_rows:,} rows" if n_rows else ""
    st.markdown(f"""
    <div class="tier-strip tier-{tier}">
        <div>
            <div class="tier-name" style="color:{color}">⬡ {name}</div>
            <div class="tier-detail">{rng}{row_str}</div>
        </div>
        <div style="flex:1"></div>
        <div class="tier-strategy">{strategy}</div>
    </div>
    """, unsafe_allow_html=True)


def badge(col_type: str) -> str:
    m = {
        "numeric":    ("b-num",  "NUM"),
        "categorical":("b-cat",  "CAT"),
        "id_dropped": ("b-id",   "ID"),
        "date":       ("b-date", "DATE"),
        "text":       ("b-text", "TEXT"),
    }
    cls, lbl = m.get(col_type, ("b-num", col_type.upper()[:4]))
    return f'<span class="badge {cls}">{lbl}</span>'


def render_complexity(cplx: dict):
    c      = cplx.get("complexity", "unknown")
    cls_m  = {"linear":"cplx-linear","nonlinear":"cplx-nonlinear",
               "mixed":"cplx-mixed","unknown":"cplx-unknown"}
    icon_m = {"linear":"◈","nonlinear":"◉","mixed":"◫","unknown":"◌"}
    col_m  = {"linear":"#4088ff","nonlinear":"#cc8840","mixed":"#40ccaa","unknown":"#55556a"}
    css    = cls_m.get(c, "cplx-unknown")
    icon   = icon_m.get(c, "◌")
    color  = col_m.get(c, "#55556a")
    lr     = cplx.get("lr_score",  "—")
    lgb    = cplx.get("lgb_score", "—")
    rec    = cplx.get("recommended", "All models")
    note   = cplx.get("note", "")
    st.markdown(f"""
    <div class="complexity-panel {css}">
        <div class="cplx-title" style="color:{color}">{icon} COMPLEXITY: {c.upper()}</div>
        <div class="cplx-note">{note}</div>
        <span class="chip">LR AUC {lr}</span>
        <span class="chip">LGB AUC {lgb}</span>
        <span class="chip">→ {rec}</span>
    </div>
    """, unsafe_allow_html=True)


def render_col_table(profile: dict):
    col_stats = profile.get("col_stats", {})
    if not col_stats:
        return
    html = '<div style="max-height:460px;overflow-y:auto">'
    for col, stats in col_stats.items():
        ctype = stats.get("type", "numeric")
        b     = badge(ctype)
        miss  = stats.get("missing_pct", 0)
        miss_color = "#cc4040" if miss > 20 else "#7878aa"

        if ctype == "numeric":
            mean_v = stats.get("mean", "—")
            std_v  = stats.get("std",  "—")
            skew_v = stats.get("skew", "—")
            min_v  = stats.get("min",  "—")
            max_v  = stats.get("max",  "—")
            detail = f"μ={mean_v} σ={std_v} skew={skew_v} [{min_v} … {max_v}]"
        elif ctype == "categorical":
            n_u   = stats.get("n_unique", "?")
            tops  = stats.get("top_values", {})
            top_s = " · ".join(f"{k}" for k in list(tops.keys())[:3])
            detail = f"{n_u} unique → {top_s}"
        else:
            detail = stats.get("note", ctype)

        html += f"""
        <div class="col-row">
            <div class="col-name" title="{col}">{col}</div>
            <div class="col-type-cell">{b}</div>
            <div class="col-stats">{detail}</div>
            <div class="col-miss" style="color:{miss_color}">{miss:.1f}% miss</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_quality_report(quality: dict):
    issues = quality.get("issues", [])
    if not issues:
        st.markdown("""
        <div class="card" style="border-color:#40cc88">
            <span style="color:#40cc88;font-family:'Syne',sans-serif;font-weight:700">
                ✓ NO ISSUES DETECTED
            </span>
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#55556a;margin-left:1rem">
                Dataset looks clean and ready to train
            </span>
        </div>
        """, unsafe_allow_html=True)
        return
    icon_map = {"error": "✕", "warning": "△", "info": "○"}
    col_map  = {"error": "#cc4040", "warning": "#ccaa40", "info": "#4088ff"}
    html = '<div class="card" style="padding:0.8rem 1.2rem">'
    for issue in issues:
        sev   = issue.get("severity", "info")
        icon  = icon_map.get(sev, "○")
        color = col_map.get(sev, "#4088ff")
        msg   = issue.get("message", "")
        html += f"""
        <div class="quality-row">
            <div class="q-icon" style="color:{color}">{icon}</div>
            <div class="q-msg">{msg}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def sidebar_model_status(trainer):
    if trainer is None:
        st.markdown("""
        <div class="model-status-card">
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#33334a;
                        text-align:center;padding:0.5rem">NO MODEL LOADED</div>
        </div>
        """, unsafe_allow_html=True)
        return
    m = trainer.metrics
    rows = [
        ("MODEL",   m.get("best_model", "—")),
        ("TARGET",  trainer.target_col or "—"),
        ("AUC",     f"{m.get('test_roc_auc',0):.5f}"),
        ("F1",      f"{m.get('f1_score',0):.5f}"),
        ("TIER",    m.get("tier_label","—").split()[0]),
    ]
    cplx = m.get("complexity", {})
    if cplx:
        rows.append(("COMPLEXITY", cplx.get("complexity","—").upper()))
    html = '<div class="model-status-card">'
    html += '<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;color:#33334a;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.6rem">LOADED MODEL</div>'
    for k, v in rows:
        html += f'<div class="ms-row"><span class="ms-key">{k}</span><span class="ms-val">{v}</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────

def apply_plot_style(fig, ax_or_axes):
    fig.patch.set_facecolor("#080810")
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#44445a", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a1a2e")
        ax.xaxis.label.set_color("#55556a")
        ax.yaxis.label.set_color("#55556a")
        ax.title.set_color("#9898cc")


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_saved_model():
    try:
        t = UniversalTrainer()
        t.load("models/universal_model.pkl")
        return t
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="brand-title">AutoML-X</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">UNIVERSAL TRAINER · v6.0</div>', unsafe_allow_html=True)

    saved_trainer = load_saved_model()
    active_trainer = st.session_state.get("u_trainer") or saved_trainer
    sidebar_model_status(active_trainer)

    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("", [
        "01 — Upload",
        "02 — Analyze",
        "03 — Train",
        "04 — Results",
        "05 — Predict",
        "06 — Batch",
    ], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    # Tier legend
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#33334a;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">
    TIER SYSTEM
    </div>
    """, unsafe_allow_html=True)
    tier_rows = [
        ("#40ccaa", "TINY",    "< 1K"),
        ("#40cc88", "SMALL",   "1K–50K"),
        ("#4088ff", "MEDIUM",  "50K–200K"),
        ("#ccaa40", "LARGE",   "200K–500K"),
        ("#cc6640", "XLARGE",  "500K–2M"),
        ("#cc4040", "MASSIVE", "2M+"),
    ]
    legend_html = ""
    for color, name, rng in tier_rows:
        legend_html += f"""
        <div style="display:flex;align-items:center;gap:0.5rem;padding:0.2rem 0">
            <div style="width:6px;height:6px;border-radius:50%;background:{color};flex-shrink:0"></div>
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{color}">{name}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#33334a">{rng}</span>
        </div>"""
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:1rem;left:0;right:0;text-align:center;
                font-family:'DM Mono',monospace;font-size:0.6rem;color:#22223a">
    AutoML-X Platform · HuggingFace
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 01 — Upload
# ─────────────────────────────────────────────────────────────────────────────

if page == "01 — Upload":
    st.markdown('<div class="page-title">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">CSV · ANY SIZE · AUTO-CLEANED</div>', unsafe_allow_html=True)

    # ── Upload area ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Drop your CSV here or click to browse", type=["csv"])

    if uploaded:
        file_mb  = uploaded.size / 1e6
        is_large = file_mb > 50

        st.markdown(f"""
        <div class="card">
            <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#9898cc">
                ◈ {uploaded.name}
            </span>
            <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#44445a;margin-left:1rem">
                {file_mb:.1f} MB
                {"· chunked loader active" if is_large else "· full load"}
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if is_large:
                max_rows = st.slider(
                    "Row limit", min_value=100_000, max_value=2_000_000,
                    value=2_000_000, step=100_000, format="{:,}",
                    help="Hard cap on rows loaded. Tier-aware training handles the rest."
                )
            else:
                max_rows = None
                st.markdown("""
                <div class="card" style="padding:0.7rem 1rem">
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#40cc88">
                        ✓ Small file — loading all rows
                    </span>
                </div>""", unsafe_allow_html=True)
        with col2:
            target_hint = st.text_input(
                "Target column (optional — for stratified sampling)",
                value="", placeholder="Churn, fraud, label…"
            )
            target_hint = target_hint.strip() or None

        if st.button("⬡ Load Dataset", type="primary", use_container_width=True):
            ph = st.empty()
            prog = st.progress(0)

            def cb(msg):
                ph.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#7878aa;
                            padding:0.4rem 0">{msg}</div>""", unsafe_allow_html=True)

            with st.spinner(""):
                try:
                    if is_large:
                        df = load_csv_chunked(
                            uploaded, max_rows=max_rows, chunk_size=100_000,
                            target_col=target_hint, progress_callback=cb
                        )
                    else:
                        df = pd.read_csv(uploaded)
                        cb(f"Loaded {len(df):,} rows")

                    prog.progress(1.0)
                    ph.empty()
                    prog.empty()

                    st.session_state["df"]        = df
                    st.session_state["filename"]  = uploaded.name
                    st.session_state.pop("profile", None)
                    st.session_state.pop("quality", None)

                    tier = get_tier(len(df))
                    color, name, _ = TIER_META[tier]

                    st.markdown(f"""
                    <div class="card-accent">
                        <div style="display:flex;align-items:center;gap:1.5rem">
                            <div>
                                <div style="font-family:'Syne',sans-serif;font-weight:700;
                                            font-size:1.1rem;color:#e8e8ff">
                                    Dataset Loaded
                                </div>
                                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                                            color:#44445a;margin-top:0.2rem">
                                    {uploaded.name}
                                </div>
                            </div>
                            <div style="display:flex;gap:1.5rem;margin-left:auto">
                                <div style="text-align:center">
                                    <div style="font-family:'DM Mono',monospace;font-size:1.3rem;
                                                color:#7878ff">{len(df):,}</div>
                                    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                                                color:#44445a;letter-spacing:1px">ROWS</div>
                                </div>
                                <div style="text-align:center">
                                    <div style="font-family:'DM Mono',monospace;font-size:1.3rem;
                                                color:#7878ff">{df.shape[1]}</div>
                                    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                                                color:#44445a;letter-spacing:1px">COLS</div>
                                </div>
                                <div style="text-align:center">
                                    <div style="font-family:'DM Mono',monospace;font-size:1.3rem;
                                                color:{color}">{name}</div>
                                    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                                                color:#44445a;letter-spacing:1px">TIER</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    render_tier(tier, len(df))

                except Exception as e:
                    ph.empty(); prog.empty()
                    st.error(f"Load failed: {e}")

    # ── Preview if loaded ─────────────────────────────────────────────────────
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.markdown('<div class="sec-header">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(6), use_container_width=True)

        ram  = check_ram_safety(df)
        tier = get_tier(len(df))

        st.markdown('<div class="sec-header">Configure Target</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target column", df.columns.tolist(),
                                      index=len(df.columns)-1)
            st.session_state["target_col"] = target_col
        with col2:
            unique_vals = df[target_col].dropna().unique().tolist()
            pos_raw = st.selectbox("Positive class value",
                                   ["Auto-detect"] + [str(v) for v in unique_vals])
            st.session_state["positive_label"] = (
                None if pos_raw == "Auto-detect" else pos_raw
            )

        if not ram["is_safe"]:
            st.warning(f"⚠️ {ram['warning']}")

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#44445a;
                    margin-top:0.5rem;text-align:right">
            RAM: {ram['dataframe_gb']:.3f} GB used · {ram['available_gb']:.1f} GB free
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.info("👉 Go to **02 — Analyze** to inspect data quality before training.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 02 — Analyze
# ─────────────────────────────────────────────────────────────────────────────

elif page == "02 — Analyze":
    st.markdown('<div class="page-title">Dataset Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">PROFILING · QUALITY REPORT · DISTRIBUTIONS</div>', unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("⚠️ Upload a dataset first (page 01).")
        st.stop()

    df         = st.session_state["df"]
    target_col = st.session_state.get("target_col", df.columns[-1])
    tier       = get_tier(len(df))
    render_tier(tier, len(df))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",    f"{len(df):,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", f"{df.isnull().sum().sum():,}")
    col4.metric("Target",  target_col)

    # ── Quick quality scan ────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Data Quality Scan</div>', unsafe_allow_html=True)
    dqr     = DataQualityReport()
    quality = dqr.assess(df, target_col)
    st.session_state["quality"] = quality

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    quality_color = {"excellent":"#40cc88","good":"#40cc88","fair":"#ccaa40","poor":"#cc4040"}
    qc = quality_color.get(quality["overall_quality"], "#7878aa")
    qcol1.markdown(f"""
    <div class="m-card"><span class="m-val" style="color:{qc}">{quality['overall_quality'].upper()}</span>
    <span class="m-lbl">QUALITY</span></div>""", unsafe_allow_html=True)
    qcol2.markdown(f"""
    <div class="m-card"><span class="m-val m-val-red">{quality['n_errors']}</span>
    <span class="m-lbl">ERRORS</span></div>""", unsafe_allow_html=True)
    qcol3.markdown(f"""
    <div class="m-card"><span class="m-val m-val-amber">{quality['n_warnings']}</span>
    <span class="m-lbl">WARNINGS</span></div>""", unsafe_allow_html=True)
    qcol4.markdown(f"""
    <div class="m-card"><span class="m-val" style="color:#4088ff">{quality['n_info']}</span>
    <span class="m-lbl">INFO</span></div>""", unsafe_allow_html=True)

    render_quality_report(quality)

    # ── Full profile ──────────────────────────────────────────────────────────
    if st.button("⬡ Run Full Profile", type="primary", use_container_width=True):
        with st.spinner("Profiling…"):
            try:
                profiler = DatasetProfiler()
                profile  = profiler.profile(df, target_col)
                st.session_state["profile"] = profile
            except Exception as e:
                st.error(f"Profile failed: {e}")

    if "profile" in st.session_state:
        profile = st.session_state["profile"]

        # Counts
        st.markdown('<div class="sec-header">Feature Overview</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid metric-grid-5" style="margin-bottom:1rem">
            <div class="m-card"><span class="m-val m-val-green">{profile['n_numeric']}</span>
                <span class="m-lbl">NUMERIC</span></div>
            <div class="m-card"><span class="m-val m-val-amber">{profile['n_categorical']}</span>
                <span class="m-lbl">CATEGORICAL</span></div>
            <div class="m-card"><span class="m-val m-val-red">{profile['n_id_dropped']}</span>
                <span class="m-lbl">IDs DROPPED</span></div>
            <div class="m-card"><span class="m-val" style="color:#4088ff">{profile['n_date_dropped']}</span>
                <span class="m-lbl">DATES DROPPED</span></div>
            <div class="m-card"><span class="m-val" style="color:#aa40cc">{profile['n_text_dropped']}</span>
                <span class="m-lbl">TEXT DROPPED</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Target distribution
        st.markdown('<div class="sec-header">Target Distribution</div>', unsafe_allow_html=True)
        tcol1, tcol2 = st.columns([1, 2])
        with tcol1:
            class_df = pd.DataFrame.from_dict(
                profile["class_counts"], orient="index", columns=["Count"]
            )
            class_df["Pct"] = (class_df["Count"] / class_df["Count"].sum() * 100).round(2)
            st.dataframe(class_df, use_container_width=True)
            if profile["is_imbalanced"]:
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#ccaa40;
                            margin-top:0.4rem">
                    △ Imbalanced — minority {profile['minority_ratio']*100:.1f}%
                    <br>class_weight='balanced' will be applied
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#40cc88;
                            margin-top:0.4rem">✓ Balanced classes</div>""",
                            unsafe_allow_html=True)
        with tcol2:
            try:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                labels  = [str(k) for k in profile["class_counts"].keys()]
                values  = list(profile["class_counts"].values())
                colors  = ["#4088ff", "#cc4040"] if len(values) >= 2 else ["#4088ff"]
                bars    = ax.bar(labels, values, color=colors[:len(values)],
                                 width=0.5, edgecolor="none")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                            f"{val:,}", ha="center", va="bottom",
                            color="#7878aa", fontsize=8, fontfamily="monospace")
                ax.set_title("Class Counts", fontsize=9)
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0.5)
                st.pyplot(fig); plt.close()
            except Exception:
                pass

        # High correlations
        if profile.get("high_corr_pairs"):
            st.markdown('<div class="sec-header">⚠ High Correlations ( > 0.95 )</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(profile["high_corr_pairs"],
                             columns=["Feature A", "Feature B", "Correlation"]),
                use_container_width=True, hide_index=True
            )

        # Column profile
        st.markdown('<div class="sec-header">Column Profile</div>', unsafe_allow_html=True)
        render_col_table(profile)

        # Distributions
        num_cols = profile["numeric_cols"]
        if num_cols:
            st.markdown('<div class="sec-header">Numeric Distributions (top 8)</div>',
                        unsafe_allow_html=True)
            show = num_cols[:8]
            nc_  = min(4, len(show))
            nr_  = (len(show) + nc_ - 1) // nc_
            fig, axes = plt.subplots(nr_, nc_, figsize=(4.5*nc_, 2.5*nr_))
            flat = np.array(axes).flatten() if len(show) > 1 else [axes]
            for i, col in enumerate(show):
                ax   = flat[i]
                data = df[col].dropna().sample(min(50_000, df[col].notna().sum()), random_state=42)
                ax.hist(data, bins=30, color="#4040cc", alpha=0.8, edgecolor="none")
                ax.set_title(col, fontsize=8)
            for j in range(len(show), len(flat)):
                flat[j].set_visible(False)
            apply_plot_style(fig, flat)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 03 — Train
# ─────────────────────────────────────────────────────────────────────────────

elif page == "03 — Train":
    st.markdown('<div class="page-title">Train Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">AUTO-ML · TIER-AWARE · SMART CLEANING</div>', unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("⚠️ Upload a dataset first (page 01).")
        st.stop()

    df         = st.session_state["df"]
    target_col = st.session_state.get("target_col")
    pos_label  = st.session_state.get("positive_label")

    if not target_col:
        st.warning("⚠️ Set target column on page 01.")
        st.stop()

    tier = get_tier(len(df))
    render_tier(tier, len(df))

    # ── Config ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Training Configuration</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        clean_data = st.checkbox("Smart data cleaning", value=True,
                                 help="Remove duplicates, cap outliers, coerce types")
    with cfg2:
        outlier_method = st.selectbox("Outlier method", ["iqr", "zscore", "none"],
                                      help="IQR: robust, Z-score: normal distributions")
    with cfg3:
        test_size = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05,
                              format="%.0f%%",
                              help="Fraction of data for evaluation")
        test_size = float(f"{test_size:.2f}")

    # Strategy info
    strategies = {
        TIER_TINY:    "4 models (LR · RF · LGB · XGB) · 5-fold CV · full data",
        TIER_SMALL:   "4 models (LR · RF · LGB · XGB) · 5-fold CV · full data",
        TIER_MEDIUM:  "4 models (LR · RF · LGB · XGB) · 3-fold CV",
        TIER_LARGE:   "3 models (LR · LGB · XGB) · 2-fold CV",
        TIER_XLARGE:  "2 models (LGB · LR) · 2-fold CV on 200K subsample",
        TIER_MASSIVE: "1 model (LGB) · no CV · 500K train sample · chunked inference",
    }
    st.markdown(f"""
    <div class="card" style="margin:0.5rem 0 1rem">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#44445a;
                     letter-spacing:1px">AUTO STRATEGY → </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#9898cc">
            {strategies[tier]}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    if st.button("⬡ Start Training", type="primary", use_container_width=True):
        prog     = st.progress(0)
        log_ph   = st.empty()
        log_lines = []
        step_map = {}

        def on_progress(step, total, msg):
            prog.progress(step / total)
            log_lines.append((step, total, msg))
            # Build log HTML
            html = '<div class="log-box">'
            for s, t, m in log_lines[-12:]:
                cls = "log-active" if s == log_lines[-1][0] else "log-done"
                html += f'<div class="{cls}">[{s}/{t}] {m}</div>'
            html += "</div>"
            log_ph.markdown(html, unsafe_allow_html=True)

        try:
            trainer = UniversalTrainer(model_save_path="models/universal_model.pkl")
            metrics = trainer.fit(
                df=df,
                target_col=target_col,
                positive_label=pos_label,
                test_size=test_size,
                clean_data=clean_data,
                outlier_method=outlier_method,
                progress_callback=on_progress
            )
            st.session_state["u_trainer"] = trainer
            st.session_state["u_metrics"] = metrics

            prog.progress(1.0)
            log_ph.empty()

            # ── Success card ─────────────────────────────────────────────────
            st.markdown(f"""
            <div class="card-accent" style="margin-top:1rem">
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            font-size:1.2rem;color:#40cc88;margin-bottom:0.8rem">
                    ✓ Training Complete
                </div>
                <div class="metric-grid metric-grid-5">
                    <div class="m-card">
                        <span class="m-val m-val-green">{metrics['best_model']}</span>
                        <span class="m-lbl">BEST MODEL</span>
                    </div>
                    <div class="m-card">
                        <span class="m-val">{metrics['test_roc_auc']:.4f}</span>
                        <span class="m-lbl">ROC-AUC</span>
                    </div>
                    <div class="m-card">
                        <span class="m-val">{metrics['f1_score']:.4f}</span>
                        <span class="m-lbl">F1</span>
                    </div>
                    <div class="m-card">
                        <span class="m-val">{metrics['recall']:.4f}</span>
                        <span class="m-lbl">RECALL</span>
                    </div>
                    <div class="m-card">
                        <span class="m-val">{metrics.get('n_features_used','?')}</span>
                        <span class="m-lbl">FEATURES</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cplx = metrics.get("complexity")
            if cplx:
                render_complexity(cplx)

            # Cleaning summary
            cr = metrics.get("cleaning_report", {})
            if cr and cr.get("changes"):
                with st.expander(f"⬡ Cleaning applied — {len(cr['changes'])} change(s)"):
                    for ch in cr["changes"]:
                        st.markdown(f"""
                        <div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                                    color:#7878aa;padding:0.3rem 0">✓ {ch}</div>
                        """, unsafe_allow_html=True)

            dropped = metrics.get("dropped_cols", [])
            if dropped:
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                            color:#44445a;margin-top:0.5rem">
                    Dropped {len(dropped)} cols: {', '.join(f'`{c}`' for c in dropped[:8])}
                    {'…' if len(dropped) > 8 else ''}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.info("→ Go to **04 — Results** for full performance breakdown.")

        except Exception as e:
            prog.empty(); log_ph.empty()
            st.error(f"Training failed: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 04 — Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == "04 — Results":
    st.markdown('<div class="page-title">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">PERFORMANCE · COMPETITION · BREAKDOWN</div>', unsafe_allow_html=True)

    trainer = st.session_state.get("u_trainer") or load_saved_model()
    if trainer is None:
        st.warning("⚠️ No trained model found.")
        st.stop()

    m = trainer.metrics
    render_tier(m.get("tier", 0), m.get("n_rows_total"))

    # ── Core metrics ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid metric-grid-5" style="margin-bottom:1rem">
        <div class="m-card">
            <span class="m-val m-val-green">{m.get('best_model','—')}</span>
            <span class="m-lbl">WINNER</span>
        </div>
        <div class="m-card">
            <span class="m-val">{m.get('cv_roc_auc',0):.5f}</span>
            <span class="m-lbl">CV ROC-AUC</span>
        </div>
        <div class="m-card">
            <span class="m-val">{m.get('test_roc_auc',0):.5f}</span>
            <span class="m-lbl">TEST ROC-AUC</span>
        </div>
        <div class="m-card">
            <span class="m-val">{m.get('f1_score',0):.5f}</span>
            <span class="m-lbl">F1 SCORE</span>
        </div>
        <div class="m-card">
            <span class="m-val">{m.get('recall',0):.5f}</span>
            <span class="m-lbl">RECALL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Complexity
    cplx = m.get("complexity")
    if cplx:
        render_complexity(cplx)

    # ── Model competition ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Model Competition</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 2])
    scores = m.get("all_cv_scores", {})

    with rc1:
        scores_df = pd.DataFrame({
            "Model": list(scores.keys()),
            "CV ROC-AUC": list(scores.values()),
        }).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)

        def hl_winner(row):
            style = "background:#0a1f10;color:#40cc88" if row.name == 0 else ""
            return [style] * len(row)
        st.dataframe(scores_df.style.apply(hl_winner, axis=1),
                     use_container_width=True, hide_index=True)

    with rc2:
        if scores:
            try:
                fig, ax = plt.subplots(figsize=(5, 3))
                ms = scores_df.sort_values("CV ROC-AUC")
                colors = ["#40cc88" if i == len(ms)-1 else "#4040cc"
                          for i in range(len(ms))]
                bars = ax.barh(ms["Model"], ms["CV ROC-AUC"],
                               color=colors, height=0.5, edgecolor="none")
                for bar, val in zip(bars, ms["CV ROC-AUC"]):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                            f"{val:.4f}", va="center", color="#7878aa",
                            fontsize=8, fontfamily="monospace")
                ax.set_xlim(max(0, ms["CV ROC-AUC"].min() - 0.05), 1.02)
                ax.set_xlabel("CV ROC-AUC", fontsize=8)
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0.5)
                st.pyplot(fig); plt.close()
            except Exception:
                pass

    # ── Threshold & confusion ─────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Threshold & Confusion</div>', unsafe_allow_html=True)
    th1, th2, th3, th4, th5 = st.columns(5)
    th1.metric("Threshold", f"{m.get('threshold',0.5):.5f}")
    th2.metric("Precision", f"{m.get('precision',0):.5f}")
    th3.metric("Recall",    f"{m.get('recall',0):.5f}")
    th4.metric("F1",        f"{m.get('f1_score',0):.5f}")
    th5.metric("Features",  m.get("n_features_used", "?"))

    tp, tn = m.get("TP", 0), m.get("TN", 0)
    fp, fn = m.get("FP", 0), m.get("FN", 0)
    total  = tp + tn + fp + fn or 1

    try:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        cm_data = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm_data, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred NEG","Pred POS"], fontsize=8, color="#7878aa")
        ax.set_yticklabels(["Act NEG","Act POS"],   fontsize=8, color="#7878aa")
        labels_cm = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels_cm[i][j], ha="center", va="center",
                        fontsize=9, color="#e8e8ff", fontfamily="monospace", fontweight="bold")
        ax.set_title("Confusion Matrix", fontsize=9)
        apply_plot_style(fig, ax)
        fig.tight_layout(pad=0.5)
        col_cm, _ = st.columns([1, 2])
        with col_cm:
            st.pyplot(fig); plt.close()
    except Exception:
        st.markdown(f"TP: `{tp}` TN: `{tn}` FP: `{fp}` FN: `{fn}`")

    # ── Training info ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Training Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <div class="metric-grid metric-grid-4">
            <div><span class="ms-key" style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#44445a;text-transform:uppercase">Strategy</span>
                 <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#9898cc;margin-top:0.3rem">
                    {m.get('tier_strategy','—')}</div></div>
            <div><span class="ms-key" style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#44445a;text-transform:uppercase">Train rows</span>
                 <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#9898cc;margin-top:0.3rem">
                    {m.get('n_train',0):,}</div></div>
            <div><span class="ms-key" style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#44445a;text-transform:uppercase">Val rows</span>
                 <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#9898cc;margin-top:0.3rem">
                    {m.get('n_val',0):,}</div></div>
            <div><span class="ms-key" style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#44445a;text-transform:uppercase">Total rows</span>
                 <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#9898cc;margin-top:0.3rem">
                    {m.get('n_rows_total',0):,}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cleaning report
    cr = m.get("cleaning_report", {})
    if cr and cr.get("changes"):
        with st.expander(f"⬡ Data Cleaning Report — {len(cr['changes'])} action(s)"):
            for ch in cr["changes"]:
                st.markdown(f"""<div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                    color:#7878aa;padding:0.2rem 0">✓ {ch}</div>""",
                            unsafe_allow_html=True)

    # Download
    st.markdown('<div class="sec-header">Export</div>', unsafe_allow_html=True)
    if os.path.exists("models/universal_model.pkl"):
        with open("models/universal_model.pkl", "rb") as f:
            st.download_button(
                "⬇ Download Model (.pkl)",
                f, "universal_model.pkl", "application/octet-stream",
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 05 — Predict
# ─────────────────────────────────────────────────────────────────────────────

elif page == "05 — Predict":
    st.markdown('<div class="page-title">Single Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">INTERACTIVE · REAL-TIME INFERENCE</div>', unsafe_allow_html=True)

    trainer = st.session_state.get("u_trainer") or load_saved_model()
    if trainer is None:
        st.warning("⚠️ Train a model first.")
        st.stop()

    features = trainer.feature_names
    df_ref   = st.session_state.get("df")

    st.markdown(f"""
    <div class="card">
        <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#55556a">
            Model: <span style="color:#9898cc">{trainer.best_model_name}</span>
            &nbsp;·&nbsp;
            {len(features)} features
            &nbsp;·&nbsp;
            Threshold: <span style="color:#9898cc">{trainer.threshold:.5f}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Feature Inputs</div>', unsafe_allow_html=True)
    input_vals = {}
    cols = st.columns(4)
    for i, feat in enumerate(features):
        with cols[i % 4]:
            if df_ref is not None and feat in df_ref.columns:
                sample = df_ref[feat].dropna()
                if sample.dtype in [np.float64, np.int64, np.float32, np.int32]:
                    input_vals[feat] = st.number_input(feat, value=float(sample.median()),
                                                       format="%.4f", key=f"f_{feat}")
                else:
                    uvals = sample.unique().tolist()
                    input_vals[feat] = st.selectbox(feat, uvals, key=f"f_{feat}")
            else:
                input_vals[feat] = st.number_input(feat, value=0.0, key=f"f_{feat}")

    st.markdown("")
    if st.button("⬡ Run Prediction", type="primary", use_container_width=True):
        try:
            prob = float(trainer.predict_proba(pd.DataFrame([input_vals]))[0])
            pred = int(prob >= trainer.threshold)
            t    = trainer.threshold

            if pred == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-label" style="color:#cc4040">POSITIVE</div>
                    <div class="result-prob" style="color:#cc4040">{prob*100:.2f}%</div>
                    <div class="result-meta">PROBABILITY · THRESHOLD {t:.5f}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-label" style="color:#40cc88">NEGATIVE</div>
                    <div class="result-prob" style="color:#40cc88">{prob*100:.2f}%</div>
                    <div class="result-meta">PROBABILITY · THRESHOLD {t:.5f}</div>
                </div>""", unsafe_allow_html=True)

            # Gauge
            try:
                fig, ax = plt.subplots(figsize=(5, 0.6))
                ax.barh([0], [1], color="#1a1a2e", height=0.4, edgecolor="none")
                bar_color = "#cc4040" if pred == 1 else "#40cc88"
                ax.barh([0], [prob], color=bar_color, height=0.4, edgecolor="none", alpha=0.9)
                ax.axvline(t, color="#7878aa", linewidth=1.5, linestyle="--", alpha=0.8)
                ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
                ax.axis("off")
                ax.text(prob, 0.35, f"{prob:.3f}", ha="center", fontsize=8,
                        color=bar_color, fontfamily="monospace")
                ax.text(t, -0.45, f"threshold\n{t:.3f}", ha="center", fontsize=7,
                        color="#55556a", fontfamily="monospace")
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0)
                st.pyplot(fig); plt.close()
            except Exception:
                pass

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: 06 — Batch
# ─────────────────────────────────────────────────────────────────────────────

elif page == "06 — Batch":
    st.markdown('<div class="page-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">CHUNKED · ANY SIZE · CSV OUTPUT</div>', unsafe_allow_html=True)

    trainer = st.session_state.get("u_trainer") or load_saved_model()
    if trainer is None:
        st.warning("⚠️ Train a model first.")
        st.stop()

    st.markdown(f"""
    <div class="card">
        <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#55556a">
            Model: <span style="color:#9898cc">{trainer.best_model_name}</span>
            &nbsp;·&nbsp;
            {len(trainer.feature_names)} required features
            &nbsp;·&nbsp;
            Chunked inference (100K rows/batch)
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Required features"):
        feat_html = " ".join(
            f'<span class="badge b-num">{f}</span>'
            for f in trainer.feature_names
        )
        st.markdown(feat_html, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV for batch inference", type=["csv"])

    if uploaded:
        file_mb  = uploaded.size / 1e6
        is_large = file_mb > 50

        if is_large:
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#ccaa40;
                        margin-bottom:0.5rem">
                △ Large file ({file_mb:.0f} MB) — chunked loader active
            </div>""", unsafe_allow_html=True)
            with st.spinner("Loading…"):
                df_new = load_csv_chunked(uploaded, max_rows=None, chunk_size=100_000)
        else:
            df_new = pd.read_csv(uploaded)

        tier_inf = get_tier(len(df_new))
        render_tier(tier_inf, len(df_new))
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#40cc88;
                    margin-bottom:0.5rem">
            ✓ Loaded {len(df_new):,} rows
        </div>""", unsafe_allow_html=True)

        X_raw = df_new.drop(columns=[trainer.target_col], errors="ignore")

        if st.button("⬡ Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Predicting {len(X_raw):,} rows…"):
                try:
                    probs = trainer.predict_proba(X_raw)
                    preds = (probs >= trainer.threshold).astype(int)

                    results = df_new.copy()
                    results["probability"]     = (probs * 100).round(4)
                    results["predicted_class"] = preds
                    results["label"] = ["POSITIVE" if p == 1 else "NEGATIVE" for p in preds]

                    pos_rate = preds.mean() * 100
                    st.markdown(f"""
                    <div class="metric-grid metric-grid-3" style="margin:1rem 0">
                        <div class="m-card"><span class="m-val">{len(results):,}</span>
                            <span class="m-lbl">TOTAL ROWS</span></div>
                        <div class="m-card"><span class="m-val m-val-red">{int(preds.sum()):,}</span>
                            <span class="m-lbl">POSITIVE</span></div>
                        <div class="m-card"><span class="m-val">{pos_rate:.2f}%</span>
                            <span class="m-lbl">POS RATE</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Distribution plot
                    try:
                        fig, ax = plt.subplots(figsize=(6, 2.5))
                        ax.hist(probs, bins=50, color="#4040cc", alpha=0.8, edgecolor="none")
                        ax.axvline(trainer.threshold, color="#cc4040", linewidth=1.5,
                                   linestyle="--", label=f"threshold={trainer.threshold:.3f}")
                        ax.set_xlabel("Probability", fontsize=8)
                        ax.set_title("Score Distribution", fontsize=9)
                        ax.legend(fontsize=7)
                        apply_plot_style(fig, ax)
                        fig.tight_layout(pad=0.5)
                        st.pyplot(fig); plt.close()
                    except Exception:
                        pass

                    st.dataframe(results.head(200), use_container_width=True)
                    st.download_button(
                        "⬇ Download Predictions CSV",
                        results.to_csv(index=False).encode(),
                        "automlx_predictions.csv", "text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")