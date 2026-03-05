"""
app_universal.py  ·  AutoML-X v7.0
════════════════════════════════════
AutoML-X Universal Trainer — PUBLIC HuggingFace Space
Runs on port 7860. Design: Obsidian Terminal — Linear/Vercel aesthetic.
"""

import sys, os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="AutoML-X", page_icon="⬡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500&display=swap');

*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0a0a0f!important;color:#d4d4e8}
.main{background:#0a0a0f!important}
.block-container{padding:2rem 3rem 5rem!important;max-width:1440px!important}
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2a2a3e;border-radius:2px}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stDecoration"]{display:none}
.stDeployButton{display:none}

[data-testid="stSidebar"]{background:#07070d!important;border-right:1px solid rgba(99,102,241,.12)!important}
[data-testid="stSidebar"] .block-container{padding:2rem 1.4rem!important}

.brand{display:flex;align-items:center;gap:10px;margin-bottom:.3rem}
.brand-logo{width:32px;height:32px;background:linear-gradient(135deg,#6366f1,#818cf8);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.brand-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.3rem;color:#f1f1ff;letter-spacing:-.5px}
.brand-ver{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#6366f1;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);border-radius:4px;padding:1px 6px;margin-left:2px}

.nav-section{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3a3a5c;letter-spacing:2.5px;text-transform:uppercase;margin:1.5rem 0 .5rem}
[data-testid="stRadio"]>div{gap:2px!important}
[data-testid="stRadio"] label{border-radius:7px!important;padding:.5rem .8rem!important;font-family:'Inter',sans-serif!important;font-size:.82rem!important;font-weight:400!important;color:#6b6b8a!important;cursor:pointer;transition:all .15s ease!important;border:1px solid transparent!important;margin:0!important}
[data-testid="stRadio"] label:hover{color:#c4c4e0!important;background:rgba(99,102,241,.06)!important}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){color:#a5b4fc!important;background:rgba(99,102,241,.1)!important;border-color:rgba(99,102,241,.2)!important;font-weight:500!important}

.page-header{border-bottom:1px solid #13132a;padding-bottom:1.2rem;margin-bottom:2rem}
.page-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem}
.page-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2rem;color:#f1f1ff;letter-spacing:-.8px;line-height:1.1;margin:0}
.page-desc{font-family:'Inter',sans-serif;font-size:.88rem;color:#6b6b8a;margin-top:.4rem;font-weight:300}

.section-label{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#4a4a6a;letter-spacing:2px;text-transform:uppercase;margin:1.8rem 0 .8rem;display:flex;align-items:center;gap:8px}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#13132a,transparent)}

.stat-row{display:grid;gap:12px;margin-bottom:1.5rem}
.stat-row-3{grid-template-columns:repeat(3,1fr)}
.stat-row-4{grid-template-columns:repeat(4,1fr)}
.stat-row-5{grid-template-columns:repeat(5,1fr)}
.stat-row-6{grid-template-columns:repeat(6,1fr)}

.stat-card{background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;transition:border-color .2s}
.stat-card:hover{border-color:rgba(99,102,241,.3)}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:500;color:#a5b4fc;line-height:1.1;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-val-green{color:#34d399}
.stat-val-amber{color:#fbbf24}
.stat-val-red{color:#f87171}
.stat-val-blue{color:#60a5fa}
.stat-val-purple{color:#c084fc}
.stat-lbl{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:2px;margin-top:.35rem;display:block}

.info-panel{background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem}
.info-panel-accent{background:linear-gradient(135deg,#0e0e1a,#101020);border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;position:relative;overflow:hidden}
.info-panel-accent::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#818cf8,#34d399)}
.info-panel-success{background:rgba(52,211,153,.04);border:1px solid rgba(52,211,153,.2);border-radius:10px;padding:1rem 1.3rem;margin-bottom:.8rem}
.info-panel-warn{background:rgba(251,191,36,.04);border:1px solid rgba(251,191,36,.2);border-radius:10px;padding:.9rem 1.2rem;margin-bottom:.6rem}
.info-panel-error{background:rgba(248,113,113,.04);border:1px solid rgba(248,113,113,.2);border-radius:10px;padding:.9rem 1.2rem;margin-bottom:.6rem}

.tier-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:6px;border:1px solid;font-family:'JetBrains Mono',monospace;font-size:.72rem;margin-bottom:1.2rem}

.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:500;letter-spacing:.5px;margin:2px}
.b-num{background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);color:#34d399}
.b-cat{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.25);color:#fbbf24}
.b-id{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.25);color:#f87171}
.b-date{background:rgba(96,165,250,.08);border:1px solid rgba(96,165,250,.25);color:#60a5fa}
.b-text{background:rgba(192,132,252,.08);border:1px solid rgba(192,132,252,.25);color:#c084fc}

.col-row{display:flex;align-items:center;gap:1rem;padding:.7rem 0;border-bottom:1px solid #0f0f1e}
.col-name{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#c4c4e0;flex:0 0 180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.col-type-cell{flex:0 0 70px}
.col-stats{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#4a4a6a;flex:1}
.col-miss{font-family:'JetBrains Mono',monospace;font-size:.68rem;flex:0 0 80px;text-align:right}

.q-row{display:flex;align-items:flex-start;gap:10px;padding:.7rem 0;border-bottom:1px solid #0f0f1e;font-size:.83rem}
.q-icon{font-size:.85rem;flex-shrink:0;margin-top:1px}
.q-msg{color:#8888aa;font-family:'Inter',sans-serif;flex:1;font-weight:300}

.cplx-panel{background:#0e0e1a;border-radius:10px;padding:1.2rem 1.5rem;border-left:3px solid;margin:.8rem 0}
.cplx-linear{border-color:#60a5fa}
.cplx-nonlinear{border-color:#fbbf24}
.cplx-mixed{border-color:#34d399}
.cplx-unknown{border-color:#4a4a6a}
.cplx-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1rem;margin-bottom:.3rem}
.cplx-note{font-family:'Inter',sans-serif;font-size:.78rem;color:#6b6b8a;margin-bottom:.6rem;font-weight:300}
.chip{display:inline-block;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.15);border-radius:4px;padding:2px 8px;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#7c7caa;margin:2px}

.log-terminal{background:#06060c;border:1px solid #13132a;border-radius:8px;padding:1rem 1.2rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;max-height:200px;overflow-y:auto;line-height:1.9}
.log-terminal .log-done{color:#34d399}
.log-terminal .log-active{color:#a5b4fc}

.result-pos{background:rgba(248,113,113,.04);border:1px solid rgba(248,113,113,.3);border-radius:12px;padding:2.5rem;text-align:center}
.result-neg{background:rgba(52,211,153,.04);border:1px solid rgba(52,211,153,.3);border-radius:12px;padding:2.5rem;text-align:center}
.result-label{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:600;letter-spacing:4px;text-transform:uppercase;margin-bottom:.5rem}
.result-prob{font-family:'JetBrains Mono',monospace;font-size:3.5rem;font-weight:600;line-height:1;margin-bottom:.6rem}
.result-meta{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3a3a5c;letter-spacing:1.5px}

/* ── SIDEBAR MODEL CARD — REDESIGNED ──────────────────────────────────────── */
.model-card{
    background: linear-gradient(135deg, #0e0e1c, #11111f);
    border: 1px solid rgba(99,102,241,.35);
    border-radius: 10px;
    padding: .85rem 1rem;
    margin: .8rem 0;
    position: relative;
    overflow: hidden;
}
.model-card::before{
    content:'';
    position:absolute;
    top:0;left:0;right:0;
    height:2px;
    background:linear-gradient(90deg,#6366f1,#818cf8,#34d399);
}
.model-card-header{
    font-family:'JetBrains Mono',monospace;
    font-size:.58rem;
    color:#6366f1;
    letter-spacing:2.5px;
    text-transform:uppercase;
    margin-bottom:.65rem;
    display:flex;
    align-items:center;
    gap:6px;
}
.model-card-header::before{
    content:'';
    display:inline-block;
    width:5px;height:5px;
    border-radius:50%;
    background:#34d399;
    box-shadow:0 0 6px #34d399;
    flex-shrink:0;
}
.model-card-empty{
    background:#0a0a14;
    border:1px dashed #1a1a2e;
    border-radius:8px;
    padding:.9rem 1rem;
    margin:.8rem 0;
    text-align:center;
    font-family:'JetBrains Mono',monospace;
    font-size:.65rem;
    color:#2a2a4a;
    letter-spacing:1.5px;
}
.ms-row{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:.28rem 0;
    font-family:'JetBrains Mono',monospace;
    font-size:.68rem;
    border-bottom: 1px solid rgba(255,255,255,.03);
}
.ms-row:last-child{border-bottom:none}
/* KEY: was #3a3a5c (nearly invisible) — now proper muted label */
.ms-key{color:#7070a0;letter-spacing:.5px}
/* VAL: was #a5b4fc (too dim) — now bright white-blue for readability */
.ms-val{color:#e0e4ff;font-weight:500}
.ms-val-green{color:#34d399;font-weight:500}
.ms-val-amber{color:#fbbf24;font-weight:500}

/* ── TIER SYSTEM LEGEND — REDESIGNED ──────────────────────────────────────── */
.tier-legend{
    background:#0b0b16;
    border:1px solid rgba(99,102,241,.18);
    border-radius:8px;
    padding:.65rem .75rem;
    margin-top:.3rem;
}
.tier-legend-row{
    display:flex;
    align-items:center;
    gap:8px;
    padding:.3rem .2rem;
    border-radius:4px;
    transition:background .15s;
}
.tier-legend-row:hover{background:rgba(255,255,255,.03)}
.tier-dot{
    width:7px;height:7px;
    border-radius:50%;
    flex-shrink:0;
}
/* TIER NAME: was {color}88 (50% opacity = washed out) — now full color */
.tier-name{
    font-family:'JetBrains Mono',monospace;
    font-size:.63rem;
    font-weight:600;
    letter-spacing:.5px;
    flex:1;
}
/* RANGE: was #2a2a4a (near black on dark bg) — now readable muted */
.tier-range{
    font-family:'JetBrains Mono',monospace;
    font-size:.6rem;
    color:#5a5a80;
    text-align:right;
}

.stButton>button{background:#0e0e1a!important;color:#a5b4fc!important;border:1px solid #2a2a4a!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-size:.83rem!important;font-weight:500!important;padding:.55rem 1.4rem!important;transition:all .15s ease!important}
.stButton>button:hover{background:rgba(99,102,241,.08)!important;border-color:rgba(99,102,241,.4)!important;color:#c4d0ff!important}
.stButton>button[kind="primary"]{background:#6366f1!important;color:#fff!important;border-color:#6366f1!important;font-weight:600!important}
.stButton>button[kind="primary"]:hover{background:#4f52e8!important;border-color:#4f52e8!important;box-shadow:0 0 20px rgba(99,102,241,.3)!important}

.stSelectbox>div>div,.stTextInput>div>div,.stNumberInput>div>div{background:#0e0e1a!important;border-color:#1a1a2e!important;color:#d4d4e8!important;border-radius:8px!important}
[data-testid="stFileUploadDropzone"]{background:#0e0e1a!important;border:1px dashed #2a2a4a!important;border-radius:10px!important}
[data-testid="stFileUploadDropzone"]:hover{border-color:rgba(99,102,241,.4)!important;background:rgba(99,102,241,.02)!important}
label{color:#6b6b8a!important;font-family:'Inter',sans-serif!important;font-size:.78rem!important;font-weight:400!important;letter-spacing:0!important;text-transform:none!important}

.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#818cf8)!important;border-radius:2px!important}
[data-testid="stMetric"]{background:#0e0e1a!important;border:1px solid #1a1a2e!important;border-radius:10px!important;padding:.9rem 1.2rem!important}
[data-testid="stMetricLabel"]{font-family:'JetBrains Mono',monospace!important;font-size:.6rem!important;color:#3a3a5c!important;text-transform:uppercase!important;letter-spacing:2px!important}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;font-size:1.5rem!important;color:#a5b4fc!important}
.stExpander{background:#0e0e1a!important;border:1px solid #1a1a2e!important;border-radius:10px!important}
.stExpander summary{font-family:'Inter',sans-serif!important;font-size:.82rem!important;color:#6b6b8a!important}
div[data-testid="stDataFrame"]{border-radius:8px;overflow:hidden}
.stAlert{border-radius:8px!important;font-family:'Inter',sans-serif!important}
hr{border-color:#13132a!important;margin:1rem 0!important}

.next-step{background:rgba(99,102,241,.05);border:1px solid rgba(99,102,241,.15);border-radius:8px;padding:.85rem 1.2rem;font-family:'Inter',sans-serif;font-size:.83rem;color:#8888cc;margin-top:1rem}
.next-step strong{color:#a5b4fc}
.sidebar-footer{position:fixed;bottom:1.2rem;font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#2a2a4a;letter-spacing:1px}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_META = {
    TIER_TINY:    ("#34d399","TINY",   "< 1K rows"),
    TIER_SMALL:   ("#34d399","SMALL",  "1K – 50K rows"),
    TIER_MEDIUM:  ("#60a5fa","MEDIUM", "50K – 200K rows"),
    TIER_LARGE:   ("#fbbf24","LARGE",  "200K – 500K rows"),
    TIER_XLARGE:  ("#fb923c","X-LARGE","500K – 2M rows"),
    TIER_MASSIVE: ("#f87171","MASSIVE","2M+ rows"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def ph(eyebrow, title, desc=""):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        {"" if not desc else f'<div class="page-desc">{desc}</div>'}
    </div>""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def render_tier(tier, n_rows=None):
    color, name, rng = TIER_META[tier]
    row_str = f" · {n_rows:,} rows" if n_rows else ""
    st.markdown(f"""
    <div class="tier-badge" style="color:{color};border-color:{color}30;background:{color}08">
        <span style="font-weight:600">{name}</span>
        <span style="color:{color}80">{rng}{row_str}</span>
        <span style="color:{color}50">·</span>
        <span style="color:{color}70;font-size:.65rem">{TIER_STRATEGY[tier]}</span>
    </div>""", unsafe_allow_html=True)

def badge(col_type):
    m = {"numeric":("b-num","NUM"),"categorical":("b-cat","CAT"),
         "id_dropped":("b-id","ID"),"date":("b-date","DATE"),"text":("b-text","TXT")}
    cls, lbl = m.get(col_type, ("b-num", col_type.upper()[:3]))
    return f'<span class="badge {cls}">{lbl}</span>'

def render_complexity(cplx):
    c = cplx.get("complexity","unknown")
    cls_m = {"linear":"cplx-linear","nonlinear":"cplx-nonlinear","mixed":"cplx-mixed","unknown":"cplx-unknown"}
    col_m = {"linear":"#60a5fa","nonlinear":"#fbbf24","mixed":"#34d399","unknown":"#4a4a6a"}
    icon_m= {"linear":"◈","nonlinear":"◉","mixed":"◫","unknown":"◌"}
    st.markdown(f"""
    <div class="cplx-panel {cls_m.get(c,'cplx-unknown')}">
        <div class="cplx-title" style="color:{col_m.get(c,'#4a4a6a')}">{icon_m.get(c,'◌')} {c.upper()} COMPLEXITY</div>
        <div class="cplx-note">{cplx.get('note','')}</div>
        <span class="chip">LR {cplx.get('lr_score','—')}</span>
        <span class="chip">LGB {cplx.get('lgb_score','—')}</span>
        <span class="chip">-> {cplx.get('recommended','All models')}</span>
    </div>""", unsafe_allow_html=True)

def render_col_table(profile):
    col_stats = profile.get("col_stats", {})
    if not col_stats: return
    html = '<div style="max-height:500px;overflow-y:auto;padding-right:4px">'
    for col, stats in col_stats.items():
        ctype = stats.get("type","numeric")
        miss  = stats.get("missing_pct", 0)
        mc    = "#f87171" if miss > 20 else "#4a4a6a" if miss > 0 else "#2a2a4a"
        if ctype == "numeric":
            detail = f"mu={stats.get('mean','--')}  sigma={stats.get('std','--')}  skew={stats.get('skew','--')}  [{stats.get('min','--')} ... {stats.get('max','--')}]"
        elif ctype == "categorical":
            tops   = stats.get("top_values", {})
            detail = f"{stats.get('n_unique','?')} unique -> {' · '.join(list(tops.keys())[:3])}"
        else:
            detail = stats.get("note", ctype)
        html += f"""
        <div class="col-row">
            <div class="col-name" title="{col}">{col}</div>
            <div class="col-type-cell">{badge(ctype)}</div>
            <div class="col-stats">{detail}</div>
            <div class="col-miss" style="color:{mc}">{"--" if miss == 0 else f"{miss:.1f}%"}</div>
        </div>"""
    st.markdown(html + "</div>", unsafe_allow_html=True)

def render_quality_report(quality):
    issues = quality.get("issues", [])
    if not issues:
        st.markdown("""<div class="info-panel-success">
            <span style="color:#34d399;font-family:'Space Grotesk',sans-serif;
                         font-weight:600;font-size:.9rem">✓ No issues — dataset looks clean</span>
        </div>""", unsafe_allow_html=True)
        return
    icon_m  = {"error":"✕","warning":"△","info":"○"}
    color_m = {"error":"#f87171","warning":"#fbbf24","info":"#60a5fa"}
    panel_m = {"error":"info-panel-error","warning":"info-panel-warn","info":"info-panel"}
    html = ""
    for issue in issues:
        sev = issue.get("severity","info")
        html += f"""
        <div class="{panel_m.get(sev,'info-panel')}" style="margin-bottom:.5rem">
            <div class="q-row" style="border:none;padding:0">
                <div class="q-icon" style="color:{color_m.get(sev,'#60a5fa')}">{icon_m.get(sev,'○')}</div>
                <div class="q-msg">{issue.get('message','')}</div>
            </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

def sidebar_model_status(trainer):
    if trainer is None:
        st.markdown('<div class="model-card-empty">NO MODEL LOADED</div>', unsafe_allow_html=True)
        return
    m = trainer.metrics
    best_model = m.get("best_model", "--")
    target     = trainer.target_col or "--"
    auc        = f"{m.get('test_roc_auc', 0):.4f}"
    f1         = f"{m.get('f1_score', 0):.4f}"
    tier_raw   = m.get("tier_label", "--").split()[0]
    cplx       = m.get("complexity", {})
    cplx_val   = cplx.get("complexity", "--").upper() if cplx else "--"

    # Colour-code AUC value
    auc_float = m.get("test_roc_auc", 0)
    auc_cls   = "ms-val-green" if auc_float >= 0.85 else "ms-val-amber" if auc_float >= 0.70 else "ms-val"

    rows = [
        ("MODEL",  best_model, "ms-val"),
        ("TARGET", target,     "ms-val"),
        ("AUC",    auc,        auc_cls),
        ("F1",     f1,         "ms-val"),
        ("TIER",   tier_raw,   "ms-val"),
    ]
    if cplx:
        rows.append(("CPLX", cplx_val, "ms-val"))

    html = '<div class="model-card"><div class="model-card-header">ACTIVE MODEL</div>'
    for k, v, cls in rows:
        html += f'<div class="ms-row"><span class="ms-key">{k}</span><span class="{cls}">{v}</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_tier_legend():
    """Redesigned tier legend — full-color names, readable range labels."""
    tiers = [
        ("#34d399", "TINY",    "< 1K"),
        ("#34d399", "SMALL",   "1K-50K"),
        ("#60a5fa", "MEDIUM",  "50K-200K"),
        ("#fbbf24", "LARGE",   "200K-500K"),
        ("#fb923c", "X-LARGE", "500K-2M"),
        ("#f87171", "MASSIVE", "2M+"),
    ]
    rows = ""
    for color, name, rng in tiers:
        rows += f"""
        <div class="tier-legend-row">
            <div class="tier-dot" style="background:{color};box-shadow:0 0 5px {color}80"></div>
            <span class="tier-name" style="color:{color}">{name}</span>
            <span class="tier-range">{rng}</span>
        </div>"""
    st.markdown(f'<div class="tier-legend">{rows}</div>', unsafe_allow_html=True)

def apply_plot_style(fig, ax_or_axes):
    fig.patch.set_facecolor("#0a0a0f")
    axes = ax_or_axes if isinstance(ax_or_axes,(list,np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#0e0e1a")
        ax.tick_params(colors="#3a3a5c", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#1a1a2e")
        ax.xaxis.label.set_color("#4a4a6a")
        ax.yaxis.label.set_color("#4a4a6a")
        ax.title.set_color("#8888aa")

@st.cache_resource
def load_saved_model():
    try:
        t = UniversalTrainer(); t.load("models/universal_model.pkl"); return t
    except: return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-logo">&#x2B21;</div>
        <div><span class="brand-name">AutoML-X</span><span class="brand-ver">v7</span></div>
    </div>
    <div style="font-family:'Inter',sans-serif;font-size:.75rem;color:#4a4a6a;
                margin-bottom:1rem;font-weight:300">Universal Binary Classifier</div>
    """, unsafe_allow_html=True)

    saved_trainer  = load_saved_model()
    active_trainer = st.session_state.get("u_trainer") or saved_trainer
    sidebar_model_status(active_trainer)

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["01 -- Upload","02 -- Analyze","03 -- Train",
                                    "04 -- Results","05 -- Predict","06 -- Batch"],
                    label_visibility="collapsed")

    st.markdown('<div class="nav-section" style="margin-top:1.8rem">Tier System</div>', unsafe_allow_html=True)
    render_tier_legend()

    st.markdown('<div class="sidebar-footer">AutoML-X · HuggingFace Space</div>', unsafe_allow_html=True)

# ── PAGE 01 — Upload ──────────────────────────────────────────────────────────
if page == "01 -- Upload":
    ph("STEP 01 / 06","Upload Dataset","Drop any CSV -- up to 2GB, any number of rows. Auto-cleaned before training.")
    uploaded = st.file_uploader("Drop your CSV here or click to browse", type=["csv"])
    if uploaded:
        file_mb = uploaded.size / 1e6; is_large = file_mb > 50
        st.markdown(f"""<div class="info-panel" style="display:flex;align-items:center;gap:1.5rem">
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:.95rem;color:#d4d4e8">{uploaded.name}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#4a4a6a;margin-top:2px">
                    {file_mb:.1f} MB &nbsp;·&nbsp;
                    {"<span style='color:#fbbf24'>chunked loader</span>" if is_large else "<span style='color:#34d399'>full load</span>"}
                </div>
            </div></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            max_rows = st.slider("Row limit",100_000,2_000_000,2_000_000,100_000,format="{:,}") if is_large else None
        with c2:
            encoding = st.selectbox("Encoding",["utf-8","latin-1","iso-8859-1","utf-16"],index=0)
        if st.button("Load Dataset", type="primary", use_container_width=True):
            p2 = st.progress(0); ph2 = st.empty()
            try:
                ph2.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:.75rem;color:#6366f1;padding:.5rem 0">Loading...</div>', unsafe_allow_html=True)
                p2.progress(0.3)
                df = load_csv_chunked(uploaded,max_rows=max_rows,chunk_size=100_000,encoding=encoding) if is_large else pd.read_csv(uploaded,encoding=encoding)
                p2.progress(0.9)
                st.session_state["df"] = df
                st.session_state.pop("profile",None); st.session_state.pop("u_trainer",None)
                p2.progress(1.0); ph2.empty(); p2.empty()
                tier = get_tier(len(df)); color,name,_ = TIER_META[tier]
                st.markdown(f"""<div class="info-panel-accent" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem">
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.1rem;color:#34d399">✓ Dataset Loaded</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#4a4a6a;margin-top:3px">{uploaded.name}</div>
                    </div>
                    <div style="display:flex;gap:2rem">
                        <div style="text-align:center"><div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#a5b4fc">{len(df):,}</div><div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3a3a5c;letter-spacing:2px">ROWS</div></div>
                        <div style="text-align:center"><div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#a5b4fc">{df.shape[1]}</div><div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3a3a5c;letter-spacing:2px">COLS</div></div>
                        <div style="text-align:center"><div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:{color}">{name}</div><div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3a3a5c;letter-spacing:2px">TIER</div></div>
                    </div></div>""", unsafe_allow_html=True)
                render_tier(tier, len(df))
            except Exception as e:
                ph2.empty(); p2.empty(); st.error(f"Load failed: {e}")
    if "df" in st.session_state:
        df = st.session_state["df"]
        sec("Preview"); st.dataframe(df.head(6), use_container_width=True)
        ram = check_ram_safety(df); tier = get_tier(len(df))
        sec("Configure Target")
        c1,c2 = st.columns(2)
        with c1:
            target_col = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns)-1)
            st.session_state["target_col"] = target_col
        with c2:
            unique_vals = df[target_col].dropna().unique().tolist()
            pos_raw = st.selectbox("Positive class value", ["Auto-detect"]+[str(v) for v in unique_vals])
            st.session_state["positive_label"] = None if pos_raw == "Auto-detect" else pos_raw
        if not ram["is_safe"]: st.warning(f"⚠️ {ram['warning']}")
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;color:#3a3a5c;text-align:right;margin-top:.3rem">RAM: {ram["dataframe_gb"]:.3f} GB used · {ram["available_gb"]:.1f} GB free</div>', unsafe_allow_html=True)
        st.markdown('<div class="next-step">-> Next: go to <strong>02 -- Analyze</strong> to inspect data quality</div>', unsafe_allow_html=True)

# ── PAGE 02 — Analyze ─────────────────────────────────────────────────────────
elif page == "02 -- Analyze":
    ph("STEP 02 / 06","Dataset Analysis","Data quality scan, feature profiling, distributions and correlations.")
    if "df" not in st.session_state: st.warning("⚠️ Upload a dataset first."); st.stop()
    df = st.session_state["df"]; target_col = st.session_state.get("target_col", df.columns[-1])
    render_tier(get_tier(len(df)), len(df))
    sec("Overview")
    miss_total = df.isnull().sum().sum()
    st.markdown(f"""<div class="stat-row stat-row-4">
        <div class="stat-card"><span class="stat-val">{len(df):,}</span><span class="stat-lbl">Rows</span></div>
        <div class="stat-card"><span class="stat-val">{df.shape[1]}</span><span class="stat-lbl">Columns</span></div>
        <div class="stat-card"><span class="stat-val {'stat-val-amber' if miss_total>0 else 'stat-val-green'}">{miss_total:,}</span><span class="stat-lbl">Missing values</span></div>
        <div class="stat-card"><span class="stat-val" style="font-size:1rem;padding-top:.3rem">{target_col}</span><span class="stat-lbl">Target column</span></div>
    </div>""", unsafe_allow_html=True)
    sec("Data Quality Scan")
    dqr = DataQualityReport(); quality = dqr.assess(df, target_col)
    st.session_state["quality"] = quality
    qc_color = {"excellent":"#34d399","good":"#34d399","fair":"#fbbf24","poor":"#f87171"}
    qc = qc_color.get(quality["overall_quality"],"#a5b4fc")
    st.markdown(f"""<div class="stat-row stat-row-4" style="margin-bottom:1rem">
        <div class="stat-card"><span class="stat-val" style="color:{qc};font-size:1.1rem;padding-top:.2rem">{quality['overall_quality'].upper()}</span><span class="stat-lbl">Overall quality</span></div>
        <div class="stat-card"><span class="stat-val stat-val-red">{quality['n_errors']}</span><span class="stat-lbl">Errors</span></div>
        <div class="stat-card"><span class="stat-val stat-val-amber">{quality['n_warnings']}</span><span class="stat-lbl">Warnings</span></div>
        <div class="stat-card"><span class="stat-val stat-val-blue">{quality['n_info']}</span><span class="stat-lbl">Info</span></div>
    </div>""", unsafe_allow_html=True)
    render_quality_report(quality)
    if st.button("Run Full Profile", type="primary", use_container_width=True):
        with st.spinner("Profiling dataset..."):
            try:
                profile = DatasetProfiler().profile(df, target_col)
                st.session_state["profile"] = profile
            except Exception as e: st.error(f"Profile failed: {e}")
    if "profile" in st.session_state:
        profile = st.session_state["profile"]
        sec("Feature Overview")
        st.markdown(f"""<div class="stat-row stat-row-5" style="margin-bottom:1.5rem">
            <div class="stat-card"><span class="stat-val stat-val-green">{profile['n_numeric']}</span><span class="stat-lbl">Numeric</span></div>
            <div class="stat-card"><span class="stat-val stat-val-amber">{profile['n_categorical']}</span><span class="stat-lbl">Categorical</span></div>
            <div class="stat-card"><span class="stat-val stat-val-red">{profile['n_id_dropped']}</span><span class="stat-lbl">IDs dropped</span></div>
            <div class="stat-card"><span class="stat-val stat-val-blue">{profile['n_date_dropped']}</span><span class="stat-lbl">Dates dropped</span></div>
            <div class="stat-card"><span class="stat-val stat-val-purple">{profile['n_text_dropped']}</span><span class="stat-lbl">Text dropped</span></div>
        </div>""", unsafe_allow_html=True)
        sec("Target Distribution")
        tc1,tc2 = st.columns([1,2])
        with tc1:
            class_df = pd.DataFrame.from_dict(profile["class_counts"],orient="index",columns=["Count"])
            class_df["Pct"] = (class_df["Count"]/class_df["Count"].sum()*100).round(2)
            st.dataframe(class_df, use_container_width=True)
            if profile["is_imbalanced"]:
                st.markdown(f'<div class="info-panel-warn" style="margin-top:.5rem"><span style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#fbbf24">△ Imbalanced -- minority {profile["minority_ratio"]*100:.1f}%<br><span style="color:#6b6b8a">class_weight=\'balanced\' applied</span></span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-panel-success" style="margin-top:.5rem"><span style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#34d399">✓ Balanced classes</span></div>', unsafe_allow_html=True)
        with tc2:
            try:
                fig,ax = plt.subplots(figsize=(5,2.8))
                labels = [str(k) for k in profile["class_counts"].keys()]
                values = list(profile["class_counts"].values())
                colors = ["#6366f1","#f87171"] if len(values)>=2 else ["#6366f1"]
                bars = ax.bar(labels,values,color=colors[:len(values)],width=.45,edgecolor="none")
                for bar,val in zip(bars,values):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*.01,f"{val:,}",ha="center",va="bottom",color="#6b6b8a",fontsize=8,fontfamily="monospace")
                ax.set_title("Class Distribution",fontsize=9)
                apply_plot_style(fig,ax); fig.tight_layout(pad=.5)
                st.pyplot(fig); plt.close()
            except: pass
        if profile.get("high_corr_pairs"):
            sec("High Correlations ( > 0.95 )")
            st.dataframe(pd.DataFrame(profile["high_corr_pairs"],columns=["Feature A","Feature B","Correlation"]),use_container_width=True,hide_index=True)
        sec("Column Profile"); render_col_table(profile)
        num_cols = profile["numeric_cols"]
        if num_cols:
            sec("Numeric Distributions (top 8)")
            show = num_cols[:8]; nc_ = min(4,len(show)); nr_ = (len(show)+nc_-1)//nc_
            fig,axes = plt.subplots(nr_,nc_,figsize=(4.5*nc_,2.8*nr_))
            flat = np.array(axes).flatten() if len(show)>1 else [axes]
            for i,col in enumerate(show):
                ax = flat[i]
                data = df[col].dropna().sample(min(50_000,df[col].notna().sum()),random_state=42)
                ax.hist(data,bins=30,color="#6366f1",alpha=.85,edgecolor="none")
                ax.set_title(col,fontsize=8)
            for j in range(len(show),len(flat)): flat[j].set_visible(False)
            apply_plot_style(fig,flat); fig.tight_layout(pad=1.0)
            st.pyplot(fig); plt.close()

# ── PAGE 03 — Train ───────────────────────────────────────────────────────────
elif page == "03 -- Train":
    ph("STEP 03 / 06","Train Model","AutoML selects the best algorithm for your dataset size and complexity.")
    if "df" not in st.session_state: st.warning("⚠️ Upload a dataset first."); st.stop()
    df = st.session_state["df"]; target_col = st.session_state.get("target_col")
    pos_label = st.session_state.get("positive_label")
    if not target_col: st.warning("⚠️ Set target column on page 01."); st.stop()
    tier = get_tier(len(df)); render_tier(tier, len(df))
    sec("Training Configuration")
    c1,c2,c3 = st.columns(3)
    with c1: clean_data = st.checkbox("Smart data cleaning",value=True)
    with c2: outlier_method = st.selectbox("Outlier method",["iqr","zscore","none"])
    with c3: test_size = float(f"{st.slider('Validation split',.1,.4,.2,.05,format='%.0f%%'):.2f}")
    strategies = {
        TIER_TINY:"4 models · LR / RF / LGB / XGB · 5-fold CV",
        TIER_SMALL:"4 models · LR / RF / LGB / XGB · 5-fold CV",
        TIER_MEDIUM:"4 models · LR / RF / LGB / XGB · 3-fold CV",
        TIER_LARGE:"3 models · LR / LGB / XGB · 2-fold CV",
        TIER_XLARGE:"2 models · LGB / LR · 2-fold CV on 200K sample",
        TIER_MASSIVE:"1 model · LGB · no CV · 500K sample",
    }
    st.markdown(f"""<div class="info-panel" style="margin:.5rem 0 1.2rem">
        <span style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#3a3a5c;letter-spacing:1px">AUTO STRATEGY</span>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#a5b4fc;margin-top:4px">{strategies[tier]}</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Start Training", type="primary", use_container_width=True):
        prog = st.progress(0); log_ph = st.empty(); log_lines = []
        def on_progress(step, total, msg):
            prog.progress(step/total); log_lines.append((step,total,msg))
            html = '<div class="log-terminal">'
            for s,t,m in log_lines[-14:]:
                cls = "log-active" if s==log_lines[-1][0] else "log-done"
                html += f'<div class="{cls}">{"-->" if cls=="log-active" else "✓"} [{s}/{t}] {m}</div>'
            log_ph.markdown(html+"</div>", unsafe_allow_html=True)
        try:
            trainer = UniversalTrainer(model_save_path="models/universal_model.pkl")
            metrics = trainer.fit(df=df,target_col=target_col,positive_label=pos_label,
                                  test_size=test_size,clean_data=clean_data,
                                  outlier_method=outlier_method,progress_callback=on_progress)
            st.session_state["u_trainer"] = trainer; st.session_state["u_metrics"] = metrics
            prog.progress(1.0); log_ph.empty()
            st.markdown(f"""<div class="info-panel-accent" style="margin-top:1.2rem">
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.1rem;color:#34d399;margin-bottom:1rem">✓ Training Complete</div>
                <div class="stat-row stat-row-5">
                    <div class="stat-card"><span class="stat-val stat-val-green" style="font-size:1rem;padding-top:.2rem">{metrics['best_model']}</span><span class="stat-lbl">Best model</span></div>
                    <div class="stat-card"><span class="stat-val">{metrics['test_roc_auc']:.4f}</span><span class="stat-lbl">ROC-AUC</span></div>
                    <div class="stat-card"><span class="stat-val">{metrics['f1_score']:.4f}</span><span class="stat-lbl">F1 Score</span></div>
                    <div class="stat-card"><span class="stat-val">{metrics['recall']:.4f}</span><span class="stat-lbl">Recall</span></div>
                    <div class="stat-card"><span class="stat-val">{metrics.get('n_features_used','?')}</span><span class="stat-lbl">Features</span></div>
                </div></div>""", unsafe_allow_html=True)
            cr = metrics.get("cleaning_report",{})
            if cr and cr.get("changes"):
                with st.expander(f"Data Cleaning Report -- {len(cr['changes'])} actions"):
                    for ch in cr["changes"]: st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#6b6b8a;padding:2px 0">✓ {ch}</div>', unsafe_allow_html=True)
            dropped = metrics.get("dropped_cols",[])
            if dropped: st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.7rem;color:#3a3a5c;margin-top:.5rem">Dropped {len(dropped)} col(s): {", ".join(f"`{c}`" for c in dropped[:8])}{"..." if len(dropped)>8 else ""}</div>', unsafe_allow_html=True)
            st.markdown('<div class="next-step">-> Go to <strong>04 -- Results</strong> for full performance breakdown</div>', unsafe_allow_html=True)
        except Exception as e:
            prog.empty(); log_ph.empty(); st.error(f"Training failed: {e}")
            import traceback
            with st.expander("Traceback"): st.code(traceback.format_exc())

# ── PAGE 04 — Results ─────────────────────────────────────────────────────────
elif page == "04 -- Results":
    ph("STEP 04 / 06","Results","Model performance, leaderboard, confusion matrix and training details.")
    trainer = st.session_state.get("u_trainer") or load_saved_model()
    if trainer is None: st.warning("⚠️ No trained model found."); st.stop()
    m = trainer.metrics; render_tier(m.get("tier",0), m.get("n_rows_total"))
    sec("Performance")
    st.markdown(f"""<div class="stat-row stat-row-5" style="margin-bottom:1.5rem">
        <div class="stat-card"><span class="stat-val stat-val-green" style="font-size:1.1rem;padding-top:.3rem">{m.get('best_model','--')}</span><span class="stat-lbl">Winner</span></div>
        <div class="stat-card"><span class="stat-val">{m.get('cv_roc_auc',0):.5f}</span><span class="stat-lbl">CV ROC-AUC</span></div>
        <div class="stat-card"><span class="stat-val">{m.get('test_roc_auc',0):.5f}</span><span class="stat-lbl">Test ROC-AUC</span></div>
        <div class="stat-card"><span class="stat-val">{m.get('f1_score',0):.5f}</span><span class="stat-lbl">F1 Score</span></div>
        <div class="stat-card"><span class="stat-val">{m.get('recall',0):.5f}</span><span class="stat-lbl">Recall</span></div>
    </div>""", unsafe_allow_html=True)
    cplx = m.get("complexity")
    if cplx: render_complexity(cplx)
    sec("Model Leaderboard")
    rc1,rc2 = st.columns([1,2]); scores = m.get("all_cv_scores",{})
    with rc1:
        scores_df = pd.DataFrame({"Model":list(scores.keys()),"CV ROC-AUC":[round(v,5) for v in scores.values()]}).sort_values("CV ROC-AUC",ascending=False).reset_index(drop=True)
        st.dataframe(scores_df.style.apply(lambda r:["background-color:#0a1f10;color:#34d399" if r.name==0 else ""]*len(r),axis=1),use_container_width=True,hide_index=True)
    with rc2:
        if scores:
            try:
                fig,ax = plt.subplots(figsize=(5,3)); ms = scores_df.sort_values("CV ROC-AUC")
                colors = ["#34d399" if i==len(ms)-1 else "#6366f1" for i in range(len(ms))]
                bars = ax.barh(ms["Model"],ms["CV ROC-AUC"],color=colors,height=.45,edgecolor="none")
                for bar,val in zip(bars,ms["CV ROC-AUC"]):
                    ax.text(val+.001,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",color="#6b6b8a",fontsize=8,fontfamily="monospace")
                ax.set_xlim(max(0,ms["CV ROC-AUC"].min()-.05),1.02); ax.set_xlabel("CV ROC-AUC",fontsize=8)
                apply_plot_style(fig,ax); fig.tight_layout(pad=.5); st.pyplot(fig); plt.close()
            except: pass
    sec("Threshold & Confusion Matrix")
    t1,t2,t3,t4,t5 = st.columns(5)
    t1.metric("Threshold",f"{m.get('threshold',.5):.5f}"); t2.metric("Precision",f"{m.get('precision',0):.5f}")
    t3.metric("Recall",f"{m.get('recall',0):.5f}"); t4.metric("F1",f"{m.get('f1_score',0):.5f}")
    t5.metric("Features",m.get("n_features_used","?"))
    tp,tn,fp,fn = m.get("TP",0),m.get("TN",0),m.get("FP",0),m.get("FN",0)
    try:
        fig,ax = plt.subplots(figsize=(3.5,3))
        ax.imshow(np.array([[tn,fp],[fn,tp]]),cmap="Blues",aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred NEG","Pred POS"],fontsize=8,color="#6b6b8a")
        ax.set_yticklabels(["Act NEG","Act POS"],fontsize=8,color="#6b6b8a")
        for i,row in enumerate([[f"TN\n{tn:,}",f"FP\n{fp:,}"],[f"FN\n{fn:,}",f"TP\n{tp:,}"]]):
            for j,lbl in enumerate(row): ax.text(j,i,lbl,ha="center",va="center",fontsize=9,color="#f1f1ff",fontfamily="monospace",fontweight="bold")
        ax.set_title("Confusion Matrix",fontsize=9); apply_plot_style(fig,ax); fig.tight_layout(pad=.5)
        col_cm,_ = st.columns([1,2])
        with col_cm: st.pyplot(fig); plt.close()
    except: st.markdown(f"TP:`{tp}` TN:`{tn}` FP:`{fp}` FN:`{fn}`")
    sec("Training Info")
    st.markdown(f"""<div class="info-panel"><div class="stat-row stat-row-4">
        <div><div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Strategy</div><div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get('tier_strategy','--')}</div></div>
        <div><div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Train rows</div><div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get('n_train',0):,}</div></div>
        <div><div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Val rows</div><div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get('n_val',0):,}</div></div>
        <div><div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Total rows</div><div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get('n_rows_total',0):,}</div></div>
    </div></div>""", unsafe_allow_html=True)
    cr = m.get("cleaning_report",{})
    if cr and cr.get("changes"):
        with st.expander(f"Data Cleaning Report -- {len(cr['changes'])} action(s)"):
            for ch in cr["changes"]: st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#6b6b8a;padding:2px 0">✓ {ch}</div>', unsafe_allow_html=True)
    sec("Export")
    if os.path.exists("models/universal_model.pkl"):
        with open("models/universal_model.pkl","rb") as f:
            st.download_button("Download Model (.pkl)",f,"universal_model.pkl","application/octet-stream",use_container_width=True)

# ── PAGE 05 — Predict ─────────────────────────────────────────────────────────
elif page == "05 -- Predict":
    ph("STEP 05 / 06","Single Prediction","Fill in feature values and get an instant probability score.")
    trainer = st.session_state.get("u_trainer") or load_saved_model()
    if trainer is None: st.warning("⚠️ Train a model first."); st.stop()
    features = trainer.feature_names; df_ref = st.session_state.get("df")
    st.markdown(f"""<div class="info-panel" style="display:flex;gap:2rem;flex-wrap:wrap">
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Model</span><div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.best_model_name}</div></div>
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Features</span><div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#a5b4fc;margin-top:3px">{len(features)}</div></div>
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Threshold</span><div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.threshold:.5f}</div></div>
    </div>""", unsafe_allow_html=True)
    sec("Feature Inputs")
    input_vals = {}; cols = st.columns(4)
    for i,feat in enumerate(features):
        with cols[i%4]:
            if df_ref is not None and feat in df_ref.columns:
                sample = df_ref[feat].dropna()
                if sample.dtype in [np.float64,np.int64,np.float32,np.int32]:
                    input_vals[feat] = st.number_input(feat,value=float(sample.median()),format="%.4f",key=f"f_{feat}")
                else:
                    input_vals[feat] = st.selectbox(feat,sample.unique().tolist(),key=f"f_{feat}")
            else:
                input_vals[feat] = st.number_input(feat,value=0.0,key=f"f_{feat}")
    st.markdown("")
    if st.button("Run Prediction", type="primary", use_container_width=True):
        try:
            prob = float(trainer.predict_proba(pd.DataFrame([input_vals]))[0])
            pred = int(prob >= trainer.threshold); t = trainer.threshold
            if pred==1:
                st.markdown(f'<div class="result-pos"><div class="result-label" style="color:#f87171">Positive</div><div class="result-prob" style="color:#f87171">{prob*100:.2f}%</div><div class="result-meta">PROBABILITY · THRESHOLD {t:.5f}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-neg"><div class="result-label" style="color:#34d399">Negative</div><div class="result-prob" style="color:#34d399">{prob*100:.2f}%</div><div class="result-meta">PROBABILITY · THRESHOLD {t:.5f}</div></div>', unsafe_allow_html=True)
            try:
                fig,ax = plt.subplots(figsize=(6,.65))
                bar_color = "#f87171" if pred==1 else "#34d399"
                ax.barh([0],[1],color="#13132a",height=.35,edgecolor="none")
                ax.barh([0],[prob],color=bar_color,height=.35,edgecolor="none",alpha=.9)
                ax.axvline(t,color="#6366f1",linewidth=1.5,linestyle="--",alpha=.8)
                ax.set_xlim(0,1); ax.set_ylim(-.5,.5); ax.axis("off")
                ax.text(prob,.32,f"{prob:.3f}",ha="center",fontsize=8,color=bar_color,fontfamily="monospace")
                ax.text(t,-.42,f"threshold {t:.3f}",ha="center",fontsize=7,color="#6366f1",fontfamily="monospace")
                apply_plot_style(fig,ax); fig.tight_layout(pad=0); st.pyplot(fig); plt.close()
            except: pass
        except Exception as e: st.error(f"Prediction failed: {e}")

# ── PAGE 06 — Batch ───────────────────────────────────────────────────────────
elif page == "06 -- Batch":
    ph("STEP 06 / 06","Batch Prediction","Score a full CSV file. Chunked inference handles any size.")
    trainer = st.session_state.get("u_trainer") or load_saved_model()
    if trainer is None: st.warning("⚠️ Train a model first."); st.stop()
    st.markdown(f"""<div class="info-panel" style="display:flex;gap:2rem;flex-wrap:wrap">
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Model</span><div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.best_model_name}</div></div>
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Required features</span><div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#a5b4fc;margin-top:3px">{len(trainer.feature_names)}</div></div>
        <div><span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Batch size</span><div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#a5b4fc;margin-top:3px">100K rows</div></div>
    </div>""", unsafe_allow_html=True)
    with st.expander("Required feature columns"):
        st.markdown(" ".join(f'<span class="badge b-num">{f}</span>' for f in trainer.feature_names), unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV for batch inference", type=["csv"])
    if uploaded:
        file_mb = uploaded.size/1e6; is_large = file_mb > 50
        if is_large:
            st.markdown(f'<div class="info-panel-warn"><span style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#fbbf24">△ Large file ({file_mb:.0f} MB) -- chunked loader active</span></div>', unsafe_allow_html=True)
            with st.spinner("Loading..."): df_new = load_csv_chunked(uploaded,max_rows=None,chunk_size=100_000)
        else: df_new = pd.read_csv(uploaded)
        render_tier(get_tier(len(df_new)), len(df_new))
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;color:#34d399;margin-bottom:.8rem">✓ Loaded {len(df_new):,} rows</div>', unsafe_allow_html=True)
        X_raw = df_new.drop(columns=[trainer.target_col],errors="ignore")
        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(X_raw):,} rows..."):
                try:
                    probs = trainer.predict_proba(X_raw); preds = (probs>=trainer.threshold).astype(int)
                    results = df_new.copy(); results["probability"]=(probs*100).round(4)
                    results["predicted_class"]=preds; results["label"]=["POSITIVE" if p==1 else "NEGATIVE" for p in preds]
                    pos_rate = preds.mean()*100
                    sec("Batch Results")
                    st.markdown(f"""<div class="stat-row stat-row-3" style="margin-bottom:1.2rem">
                        <div class="stat-card"><span class="stat-val">{len(results):,}</span><span class="stat-lbl">Total rows</span></div>
                        <div class="stat-card"><span class="stat-val stat-val-red">{int(preds.sum()):,}</span><span class="stat-lbl">Positive</span></div>
                        <div class="stat-card"><span class="stat-val">{pos_rate:.2f}%</span><span class="stat-lbl">Positive rate</span></div>
                    </div>""", unsafe_allow_html=True)
                    try:
                        fig,ax = plt.subplots(figsize=(6,2.8))
                        ax.hist(probs,bins=50,color="#6366f1",alpha=.85,edgecolor="none")
                        ax.axvline(trainer.threshold,color="#f87171",linewidth=1.5,linestyle="--",label=f"threshold {trainer.threshold:.3f}")
                        ax.set_xlabel("Probability",fontsize=8); ax.set_title("Score Distribution",fontsize=9); ax.legend(fontsize=7)
                        apply_plot_style(fig,ax); fig.tight_layout(pad=.5); st.pyplot(fig); plt.close()
                    except: pass
                    st.dataframe(results.head(200), use_container_width=True)
                    st.download_button("Download Predictions CSV",results.to_csv(index=False).encode(),"automlx_predictions.csv","text/csv",use_container_width=True)
                except Exception as e: st.error(f"Batch prediction failed: {e}")