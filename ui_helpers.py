"""UI Helpers & Constants — app_universal.py refactor"""
import streamlit as st
import numpy as np

JM = "font-family:'JetBrains Mono',monospace"

TIER_META = {
    (1): ("#34d399", "TINY",    "< 1K rows"),
    (2): ("#34d399", "SMALL",   "1K – 50K rows"),
    (3): ("#60a5fa", "MEDIUM",  "50K – 200K rows"),
    (4): ("#fbbf24", "LARGE",   "200K – 500K rows"),
    (5): ("#fb923c", "X-LARGE", "500K – 2M rows"),
    (6): ("#f87171", "MASSIVE", "2M+ rows"),
}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0a0a0f!important;color:#d4d4e8}
.main{background:#0a0a0f!important}
.block-container{padding:2rem 3rem 5rem!important;max-width:1440px!important}
::-webkit-scrollbar{width:3px;height:3px}::-webkit-scrollbar-thumb{background:#2a2a3e;border-radius:2px}
#MainMenu,footer,header,.stDeployButton,[data-testid="stDecoration"]{visibility:hidden;display:none}
[data-testid="stSidebar"]{background:#07070d!important;border-right:1px solid rgba(99,102,241,.12)!important}
[data-testid="stSidebar"] .block-container{padding:2rem 1.4rem!important}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:.3rem}
.brand-logo{width:32px;height:32px;background:linear-gradient(135deg,#6366f1,#818cf8);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.brand-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.3rem;color:#f1f1ff;letter-spacing:-.5px}
.brand-ver{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#6366f1;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);border-radius:4px;padding:1px 6px;margin-left:2px}
.nav-section{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#5a5a7c;letter-spacing:2.5px;text-transform:uppercase;margin:1.5rem 0 .5rem}
[data-testid="stRadio"]>div{gap:2px!important}
[data-testid="stRadio"] label{border-radius:7px!important;padding:.5rem .8rem!important;font-family:'Inter',sans-serif!important;font-size:.82rem!important;font-weight:400!important;color:#8a8aaa!important;cursor:pointer;transition:all .15s ease!important;border:1px solid transparent!important;margin:0!important}
[data-testid="stRadio"] label:hover{color:#c4c4e0!important;background:rgba(99,102,241,.06)!important}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){color:#a5b4fc!important;background:rgba(99,102,241,.1)!important;border-color:rgba(99,102,241,.2)!important;font-weight:500!important}
.page-header{border-bottom:1px solid #13132a;padding-bottom:1.2rem;margin-bottom:2rem}
.page-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem}
.page-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2rem;color:#f1f1ff;letter-spacing:-.8px;line-height:1.1;margin:0}
.page-desc{font-family:'Inter',sans-serif;font-size:.88rem;color:#8b8baa;margin-top:.4rem;font-weight:300}
.section-label{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#6a6a8a;letter-spacing:2px;text-transform:uppercase;margin:1.8rem 0 .8rem;display:flex;align-items:center;gap:8px}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#13132a,transparent)}
.stat-row{display:grid;gap:12px;margin-bottom:1.5rem}
.stat-row-3{grid-template-columns:repeat(3,1fr)}.stat-row-4{grid-template-columns:repeat(4,1fr)}.stat-row-5{grid-template-columns:repeat(5,1fr)}
.stat-card{background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.1rem 1.3rem;transition:border-color .2s}
.stat-card:hover{border-color:rgba(99,102,241,.3)}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:500;color:#a5b4fc;line-height:1.1;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-val-green{color:#34d399}.stat-val-amber{color:#fbbf24}.stat-val-red{color:#f87171}.stat-val-blue{color:#60a5fa}.stat-val-purple{color:#c084fc}
.stat-lbl{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#5a5a7c;text-transform:uppercase;letter-spacing:2px;margin-top:.35rem;display:block}
.info-panel{background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem}
.info-panel-accent{background:linear-gradient(135deg,#0e0e1a,#101020);border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;position:relative;overflow:hidden}
.info-panel-accent::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#818cf8,#34d399)}
.info-panel-success{background:rgba(52,211,153,.04);border:1px solid rgba(52,211,153,.2);border-radius:10px;padding:1rem 1.3rem;margin-bottom:.8rem}
.info-panel-warn{background:rgba(251,191,36,.04);border:1px solid rgba(251,191,36,.2);border-radius:10px;padding:.9rem 1.2rem;margin-bottom:.6rem}
.info-panel-error{background:rgba(248,113,113,.04);border:1px solid rgba(248,113,113,.2);border-radius:10px;padding:.9rem 1.2rem;margin-bottom:.6rem}
.tier-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:6px;border:1px solid;font-family:'JetBrains Mono',monospace;font-size:.72rem;margin-bottom:1.2rem}
.tier-legend{background:#0b0b16;border:1px solid rgba(99,102,241,.18);border-radius:8px;padding:.65rem .75rem;margin-top:.3rem}
.tier-legend-row{display:flex;align-items:center;gap:8px;padding:.3rem .2rem;border-radius:4px}
.tier-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.tier-name{font-family:'JetBrains Mono',monospace;font-size:.63rem;font-weight:600;letter-spacing:.5px;flex:1}
.tier-range{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#5a5a80;text-align:right}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:500;letter-spacing:.5px;margin:2px}
.b-num{background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);color:#34d399}
.b-cat{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.25);color:#fbbf24}
.b-id{background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.25);color:#f87171}
.b-date{background:rgba(96,165,250,.08);border:1px solid rgba(96,165,250,.25);color:#60a5fa}
.b-text{background:rgba(192,132,252,.08);border:1px solid rgba(192,132,252,.25);color:#c084fc}
.col-row{display:flex;align-items:center;gap:1rem;padding:.7rem 0;border-bottom:1px solid #0f0f1e}
.col-name{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#c4c4e0;flex:0 0 180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.col-type-cell{flex:0 0 70px}.col-stats{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#6a6a8a;flex:1}
.col-miss{font-family:'JetBrains Mono',monospace;font-size:.68rem;flex:0 0 80px;text-align:right}
.cplx-panel{background:#0e0e1a;border-radius:10px;padding:1.2rem 1.5rem;border-left:3px solid;margin:.8rem 0}
.cplx-linear{border-color:#60a5fa}.cplx-nonlinear{border-color:#fbbf24}.cplx-mixed{border-color:#34d399}.cplx-unknown{border-color:#4a4a6a}
.cplx-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1rem;margin-bottom:.3rem}
.cplx-note{font-family:'Inter',sans-serif;font-size:.78rem;color:#8b8baa;margin-bottom:.6rem;font-weight:300}
.chip{display:inline-block;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.15);border-radius:4px;padding:2px 8px;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#7c7caa;margin:2px}
.log-terminal{background:#06060c;border:1px solid #13132a;border-radius:8px;padding:1rem 1.2rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;max-height:200px;overflow-y:auto;line-height:1.9}
.log-terminal .log-done{color:#34d399}.log-terminal .log-active{color:#a5b4fc}
.result-neg{background:rgba(52,211,153,.04);border:1px solid rgba(52,211,153,.3);border-radius:12px;padding:2.5rem;text-align:center}
.result-label{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:600;letter-spacing:4px;text-transform:uppercase;margin-bottom:.5rem}
.result-prob{font-family:'JetBrains Mono',monospace;font-size:3.5rem;font-weight:600;line-height:1;margin-bottom:.6rem}
.result-meta{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#5a5a7c;letter-spacing:1.5px}
.model-card{background:linear-gradient(135deg,#0e0e1c,#11111f);border:1px solid rgba(99,102,241,.35);border-radius:10px;padding:.85rem 1rem;margin:.8rem 0;position:relative;overflow:hidden}
.model-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#818cf8,#34d399)}
.model-card-header{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#6366f1;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.65rem;display:flex;align-items:center;gap:6px}
.model-card-header::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#34d399;box-shadow:0 0 6px #34d399;flex-shrink:0}
.model-card-empty{background:#0a0a14;border:1px dashed #1a1a2e;border-radius:8px;padding:.9rem 1rem;margin:.8rem 0;text-align:center;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#4a4a6a;letter-spacing:1.5px}
.ms-row{display:flex;justify-content:space-between;align-items:center;padding:.28rem 0;font-family:'JetBrains Mono',monospace;font-size:.68rem;border-bottom:1px solid rgba(255,255,255,.03)}
.ms-row:last-child{border-bottom:none}
.ms-key{color:#9090c0;letter-spacing:.5px}.ms-val{color:#e0e4ff;font-weight:500}.ms-val-green{color:#34d399;font-weight:500}.ms-val-amber{color:#fbbf24;font-weight:500}
.privacy-notice{background:rgba(52,211,153,.03);border:1px solid rgba(52,211,153,.12);border-radius:8px;padding:.7rem .9rem;margin:.8rem 0;font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#5a7a6a;line-height:1.6}
.privacy-notice .pn-title{color:#34d399;font-size:.62rem;font-weight:600;letter-spacing:1.5px;margin-bottom:.35rem;display:flex;align-items:center;gap:5px}
.shap-panel{background:#0b0b18;border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:1.2rem 1.5rem;margin:.8rem 0}
.shap-title{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:600;color:#a5b4fc;margin-bottom:.3rem}
.shap-subtitle{font-family:'Inter',sans-serif;font-size:.75rem;color:#6a6a8a;font-weight:300;margin-bottom:1rem}
.shap-feat-row{display:flex;align-items:center;gap:10px;padding:.35rem 0;border-bottom:1px solid rgba(255,255,255,.03);font-family:'JetBrains Mono',monospace;font-size:.68rem}
.shap-feat-row:last-child{border-bottom:none}
.shap-feat-name{color:#aaaacc;flex:0 0 200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.shap-feat-bar-wrap{flex:1;height:6px;background:#13132a;border-radius:3px}
.shap-feat-bar{height:6px;border-radius:3px}
.shap-feat-val{flex:0 0 60px;text-align:right}
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
[data-testid="stMetricLabel"]{font-family:'JetBrains Mono',monospace!important;font-size:.6rem!important;color:#5a5a7c!important;text-transform:uppercase!important;letter-spacing:2px!important}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;font-size:1.5rem!important;color:#a5b4fc!important}
hr{border-color:#13132a!important;margin:1rem 0!important}
div[data-testid="stDataFrame"]{border-radius:8px;overflow:hidden}
.next-step{background:rgba(99,102,241,.05);border:1px solid rgba(99,102,241,.15);border-radius:8px;padding:.85rem 1.2rem;font-family:'Inter',sans-serif;font-size:.83rem;color:#8888cc;margin-top:1rem}
.next-step strong{color:#a5b4fc}
.sidebar-footer{position:fixed;bottom:1.2rem;font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#2a2a4a;letter-spacing:1px}
.metric-explain-row{display:flex;gap:1rem;padding:.65rem 0;border-bottom:1px solid #0f0f1e;align-items:flex-start}
.metric-explain-row:last-child{border-bottom:none}
.me-name{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:600;color:#a5b4fc;flex:0 0 100px;padding-top:1px}
.me-score{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#34d399;flex:0 0 70px;padding-top:1px}
.me-desc{font-family:'Inter',sans-serif;font-size:.75rem;color:#6b6b8a;font-weight:300;line-height:1.5;flex:1}
@keyframes alert-pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}50%{box-shadow:0 0 0 10px rgba(239,68,68,0)}}
.fraud-alert{background:linear-gradient(135deg,#1a0505,#200808);border:1.5px solid #ef4444;border-radius:12px;padding:2rem 2.5rem;text-align:center;animation:alert-pulse 2s ease-in-out infinite;margin-bottom:1rem}
.fraud-alert-tag{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#fca5a5;letter-spacing:4px;text-transform:uppercase;margin-bottom:.6rem}
.fraud-alert-prob{font-family:'JetBrains Mono',monospace;font-size:3.8rem;font-weight:700;color:#ef4444;line-height:1;margin-bottom:.5rem}
.fraud-alert-meta{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#7f1d1d;letter-spacing:1.5px}
.hero-wrap{position:relative;padding:3.5rem 0 2.5rem;overflow:hidden}
.hero-wrap::before{content:'';position:absolute;top:-80px;left:-100px;width:500px;height:500px;background:radial-gradient(circle,rgba(99,102,241,.07) 0%,transparent 70%);pointer-events:none}
.hero-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#6366f1;letter-spacing:3px;text-transform:uppercase;margin-bottom:.9rem;display:flex;align-items:center;gap:8px}
.hero-eyebrow::before{content:'';display:inline-block;width:18px;height:1px;background:#6366f1}
.hero-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:3rem;color:#f1f1ff;letter-spacing:-1.5px;line-height:1.08;margin:0 0 1rem}
.hero-title span{color:#6366f1}
.hero-sub{font-family:'Inter',sans-serif;font-size:1rem;color:#6b6b8a;font-weight:300;line-height:1.6;max-width:560px;margin-bottom:2rem}
.hero-badges{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:2.5rem}
.hero-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:20px;font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:500;border:1px solid}
.flow-wrap{display:flex;align-items:stretch;gap:0;margin:2rem 0}
.flow-step{flex:1;background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.4rem 1.2rem;transition:border-color .2s}
.flow-step:hover{border-color:rgba(99,102,241,.35)}
.flow-num{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:600;color:rgba(99,102,241,.15);line-height:1;margin-bottom:.5rem}
.flow-icon{font-size:1.4rem;margin-bottom:.6rem;display:block}
.flow-title{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:600;color:#c4c4e0;margin-bottom:.3rem}
.flow-desc{font-family:'Inter',sans-serif;font-size:.75rem;color:#4a4a6a;font-weight:300;line-height:1.5}
.flow-arrow{display:flex;align-items:center;padding:0 .6rem;color:#2a2a4a;font-size:1.1rem;flex-shrink:0}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:2rem 0}
.kpi-card{background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.2rem 1.3rem;text-align:center;transition:border-color .2s}
.kpi-card:hover{border-color:rgba(99,102,241,.3)}
.kpi-val{font-family:'JetBrains Mono',monospace;font-size:1.7rem;font-weight:600;line-height:1.1;display:block;margin-bottom:.3rem}
.kpi-lbl{font-family:'Inter',sans-serif;font-size:.72rem;color:#4a4a6a;font-weight:300}
.domain-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:1rem 0}
.domain-card{background:#0b0b16;border:1px solid #13132a;border-radius:8px;padding:1rem 1.1rem;transition:border-color .2s,background .2s}
.domain-card:hover{background:#0e0e1c;border-color:rgba(99,102,241,.25)}
.domain-icon{font-size:1.3rem;margin-bottom:.4rem;display:block}
.domain-name{font-family:'Space Grotesk',sans-serif;font-size:.82rem;font-weight:600;color:#c4c4e0;margin-bottom:.2rem}
.domain-desc{font-family:'Inter',sans-serif;font-size:.7rem;color:#4a4a6a;font-weight:300;line-height:1.45}
.authors-bar{display:flex;align-items:center;gap:1.5rem;padding:1rem 1.4rem;background:#0b0b16;border:1px solid #13132a;border-radius:8px;margin-top:2rem;flex-wrap:wrap}
.authors-label{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#5a5a7c;letter-spacing:2px;text-transform:uppercase;flex-shrink:0}
.author-chip{display:inline-flex;align-items:center;gap:5px;font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#a5b4fc}
.author-chip::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#6366f1}
.links-row{display:flex;gap:10px;margin-left:auto;flex-wrap:wrap}
.ext-link{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:5px;border:1px solid #1a1a2e;font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#8b8baa;text-decoration:none;transition:border-color .15s,color .15s}
.ext-link:hover{border-color:rgba(99,102,241,.4);color:#a5b4fc}
.batch-info{background:rgba(96,165,250,.04);border:1px solid rgba(96,165,250,.18);border-left:3px solid #60a5fa;border-radius:8px;padding:.85rem 1.2rem;margin-bottom:1rem;font-family:'Inter',sans-serif;font-size:.8rem;color:#8b8baa;line-height:1.55}
.batch-info strong{color:#60a5fa;font-weight:500}
.drift-row{display:flex;align-items:center;gap:10px;padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.03);font-family:'JetBrains Mono',monospace;font-size:.68rem}
.drift-row:last-child{border-bottom:none}
.q-row{display:flex;align-items:flex-start;gap:10px;padding:.7rem 0;border-bottom:1px solid #0f0f1e;font-size:.83rem}
.q-icon{font-size:.85rem;flex-shrink:0;margin-top:1px}
.q-msg{color:#aaaacc;font-family:'Inter',sans-serif;flex:1;font-weight:300}
</style>"""

def ph(eyebrow, title, desc=""):
    """Page header"""
    st.markdown(
        f'<div class="page-header"><div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        + (f'<div class="page-desc">{desc}</div>' if desc else "") + "</div>",
        unsafe_allow_html=True,
    )

def sec(label):
    """Section label"""
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def render_tier(tier, n_rows=None):
    """Render tier badge"""
    from src.universal_trainer import TIER_STRATEGY
    color, name, rng = TIER_META.get(tier, ("#888", "?", "?"))
    row_str = f" · {n_rows:,} rows" if n_rows else ""
    st.markdown(
        f'<div class="tier-badge" style="color:{color};border-color:{color}30;background:{color}08">'
        f'<span style="font-weight:600">{name}</span>'
        f'<span style="color:{color}80">{rng}{row_str}</span>'
        f'<span style="color:{color}70;font-size:.65rem"> · {TIER_STRATEGY.get(tier, "")}</span></div>',
        unsafe_allow_html=True,
    )

def badge(col_type):
    """Column type badge"""
    m = {"numeric":("b-num","NUM"), "categorical":("b-cat","CAT"), "id_dropped":("b-id","ID"), "date":("b-date","DATE"), "text":("b-text","TXT")}
    cls, lbl = m.get(col_type, ("b-num", col_type.upper()[:3]))
    return f'<span class="badge {cls}">{lbl}</span>'

def render_col_table(profile):
    """Render column statistics table"""
    col_stats = profile.get("col_stats", {})
    if not col_stats:
        return
    html = '<div style="max-height:500px;overflow-y:auto;padding-right:4px">'
    for col, stats in col_stats.items():
        ctype = stats.get("type", "numeric")
        miss  = stats.get("missing_pct", 0)
        mc    = "#f87171" if miss > 20 else "#4a4a6a" if miss > 0 else "#2a2a4a"
        if ctype == "numeric":
            detail = f"μ={stats.get('mean','--')} σ={stats.get('std','--')} [{stats.get('min','--')}…{stats.get('max','--')}]"
        elif ctype == "categorical":
            tops   = stats.get("top_values", {})
            detail = f"{stats.get('n_unique','?')} unique → {' · '.join(list(tops.keys())[:3])}"
        else:
            detail = stats.get("note", ctype)
        html += f'<div class="col-row"><div class="col-name" title="{col}">{col}</div><div class="col-type-cell">{badge(ctype)}</div><div class="col-stats">{detail}</div><div class="col-miss" style="color:{mc}">{"--" if miss==0 else f"{miss:.1f}%"}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

def render_tier_legend():
    """Render tier legend in sidebar"""
    tiers = [("#34d399","TINY","< 1K"), ("#34d399","SMALL","1K-50K"), ("#60a5fa","MEDIUM","50K-200K"), ("#fbbf24","LARGE","200K-500K"), ("#fb923c","X-LARGE","500K-2M"), ("#f87171","MASSIVE","2M+")]
    rows = "".join(f'<div class="tier-legend-row"><div class="tier-dot" style="background:{c};box-shadow:0 0 5px {c}80"></div><span class="tier-name" style="color:{c}">{n}</span><span class="tier-range">{r}</span></div>' for c, n, r in tiers)
    st.markdown(f'<div class="tier-legend">{rows}</div>', unsafe_allow_html=True)

def apply_plot_style(fig, axes):
    """Apply custom matplotlib style"""
    fig.patch.set_facecolor("#0a0a0f")
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in np.array(ax_list).flatten():
        ax.set_facecolor("#0e0e1a")
        ax.tick_params(colors="#3a3a5c", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a1a2e")
        ax.xaxis.label.set_color("#4a4a6a")
        ax.yaxis.label.set_color("#4a4a6a")
        ax.title.set_color("#8888aa")

def no_model_msg():
    """Display 'no model trained' message"""
    st.markdown(
        '<div class="info-panel-warn" style="margin-top:1rem">'
        f'<span style="{JM};font-size:.8rem;color:#fbbf24">'
        "△ No model trained in this session.<br>"
        '<span style="font-size:.72rem;color:#6b6b8a">Go to '
        '<strong style="color:#a5b4fc">03 — Train</strong> and run training first.</span>'
        "</span></div>",
        unsafe_allow_html=True,
    )

def sidebar_model_status(trainer):
    """Show trainer status in sidebar"""
    if trainer is None:
        st.markdown(
            f'<div style="background:#0a0a14;border:1px dashed #1a1a2e;border-radius:8px;'
            f'padding:.6rem .8rem;margin:.8rem 0;{JM};font-size:.62rem;color:#2a2a4a;letter-spacing:1px">'
            "▫ No model trained</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="model-card">'
            f'<div class="model-card-header">{trainer.best_model_name}</div>'
            f'<div class="ms-row"><span class="ms-key">ROC-AUC</span>'
            f'<span class="ms-val-green">{trainer.metrics.get("test_roc_auc",0):.5f}</span></div>'
            f'<div class="ms-row"><span class="ms-key">RECALL</span>'
            f'<span class="ms-val-green">{trainer.metrics.get("recall",0):.3f}</span></div>'
            f'<div class="ms-row"><span class="ms-key">THRESHOLD</span>'
            f'<span class="ms-val">{trainer.threshold:.4f}</span></div></div>',
            unsafe_allow_html=True,
        )
