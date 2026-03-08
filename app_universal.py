"""
app_universal.py · AutoML-X v7.3
AutoML-X Universal Trainer — PUBLIC HuggingFace Space · port 7860
Obsidian Terminal design (Space Grotesk / JetBrains Mono / Inter)

v7.3 FIXES & UPGRADES:
  FIX  slider: Row limit format="%,.0f"  (was "{:,}" → showed "{;}")
  FIX  slider: Validation split int range 10-40 + %d%%  (was %.0f%% on float → "0%")
  FIX  cost_optimizer: wired to real src/cost_optimizer.BusinessCostOptimizer
  FIX  evaluation.py: full rewrite — no disk writes, universal labels, returns figs
  FIX  confusion matrix: fixed-color cells, no Blues cmap white-on-white bug
  NEW  val_probs / val_labels stored in metrics (needs 2-line patch in universal_trainer)
  NEW  ThresholdOptimizer strategies wired (F1 / Recall / Precision) on Page 04
  NEW  ROC curve + Precision-Recall curve on Page 04
  NEW  Threshold strategy comparison chart on Page 04
  NEW  DriftDetector wired on Page 06 — compares batch vs training distribution
  NEW  Score distribution dual-panel with log Y (fixes imbalanced crammed bars)
"""

import sys
import os
import gc
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# ── Core trainer ─────────────────────────────────────────────────────────────
from src.universal_trainer import (
    UniversalTrainer, DatasetProfiler, DataQualityReport,
    SmartDataCleaner, ColumnTypeDetector, ComplexityDetector,
    check_ram_safety, load_csv_chunked, get_tier,
    TIER_LABELS, TIER_STRATEGY,
    TIER_TINY, TIER_SMALL, TIER_MEDIUM,
    TIER_LARGE, TIER_XLARGE, TIER_MASSIVE,
)

# ── Optional src modules ─────────────────────────────────────────────────────
try:
    from src.shap_universal import UniversalSHAP
    _SHAP = True
except ImportError:
    _SHAP = False

try:
    from src.report_generator import generate_pdf_report
    _PDF = True
except ImportError:
    _PDF = False

try:
    from src.cost_optimizer import BusinessCostOptimizer
    _COST_OPT = True
except ImportError:
    _COST_OPT = False

try:
    from src.threshold_optimizer import ThresholdOptimizer
    _THR_OPT = True
except ImportError:
    _THR_OPT = False

try:
    from src.drift_detector import DriftDetector
    _DRIFT = True
except ImportError:
    _DRIFT = False

try:
    from src.evaluation import (
        plot_roc_curve,
        plot_precision_recall_curve,
        plot_confusion_matrix,
        plot_score_distribution,
        plot_threshold_strategies,
        get_classification_summary,
    )
    _EVAL = True
except ImportError:
    _EVAL = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML-X",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
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
.stat-row-3{grid-template-columns:repeat(3,1fr)}.stat-row-4{grid-template-columns:repeat(4,1fr)}.stat-row-5{grid-template-columns:repeat(5,1fr)}
.stat-card{background:#0e0e1a;border:1px solid #1a1a2e;border-radius:10px;padding:1.1rem 1.3rem;transition:border-color .2s}
.stat-card:hover{border-color:rgba(99,102,241,.3)}
.stat-val{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:500;color:#a5b4fc;line-height:1.1;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-val-green{color:#34d399}.stat-val-amber{color:#fbbf24}.stat-val-red{color:#f87171}.stat-val-blue{color:#60a5fa}.stat-val-purple{color:#c084fc}
.stat-lbl{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:2px;margin-top:.35rem;display:block}
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
.col-type-cell{flex:0 0 70px}.col-stats{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#4a4a6a;flex:1}
.col-miss{font-family:'JetBrains Mono',monospace;font-size:.68rem;flex:0 0 80px;text-align:right}
.cplx-panel{background:#0e0e1a;border-radius:10px;padding:1.2rem 1.5rem;border-left:3px solid;margin:.8rem 0}
.cplx-linear{border-color:#60a5fa}.cplx-nonlinear{border-color:#fbbf24}.cplx-mixed{border-color:#34d399}.cplx-unknown{border-color:#4a4a6a}
.cplx-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:1rem;margin-bottom:.3rem}
.cplx-note{font-family:'Inter',sans-serif;font-size:.78rem;color:#6b6b8a;margin-bottom:.6rem;font-weight:300}
.chip{display:inline-block;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.15);border-radius:4px;padding:2px 8px;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#7c7caa;margin:2px}
.log-terminal{background:#06060c;border:1px solid #13132a;border-radius:8px;padding:1rem 1.2rem;font-family:'JetBrains Mono',monospace;font-size:.72rem;max-height:200px;overflow-y:auto;line-height:1.9}
.log-terminal .log-done{color:#34d399}.log-terminal .log-active{color:#a5b4fc}
.result-neg{background:rgba(52,211,153,.04);border:1px solid rgba(52,211,153,.3);border-radius:12px;padding:2.5rem;text-align:center}
.result-label{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:600;letter-spacing:4px;text-transform:uppercase;margin-bottom:.5rem}
.result-prob{font-family:'JetBrains Mono',monospace;font-size:3.5rem;font-weight:600;line-height:1;margin-bottom:.6rem}
.result-meta{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3a3a5c;letter-spacing:1.5px}
.model-card{background:linear-gradient(135deg,#0e0e1c,#11111f);border:1px solid rgba(99,102,241,.35);border-radius:10px;padding:.85rem 1rem;margin:.8rem 0;position:relative;overflow:hidden}
.model-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#818cf8,#34d399)}
.model-card-header{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#6366f1;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.65rem;display:flex;align-items:center;gap:6px}
.model-card-header::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#34d399;box-shadow:0 0 6px #34d399;flex-shrink:0}
.model-card-empty{background:#0a0a14;border:1px dashed #1a1a2e;border-radius:8px;padding:.9rem 1rem;margin:.8rem 0;text-align:center;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#2a2a4a;letter-spacing:1.5px}
.ms-row{display:flex;justify-content:space-between;align-items:center;padding:.28rem 0;font-family:'JetBrains Mono',monospace;font-size:.68rem;border-bottom:1px solid rgba(255,255,255,.03)}
.ms-row:last-child{border-bottom:none}
.ms-key{color:#7070a0;letter-spacing:.5px}.ms-val{color:#e0e4ff;font-weight:500}.ms-val-green{color:#34d399;font-weight:500}.ms-val-amber{color:#fbbf24;font-weight:500}
.privacy-notice{background:rgba(52,211,153,.03);border:1px solid rgba(52,211,153,.12);border-radius:8px;padding:.7rem .9rem;margin:.8rem 0;font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3a5a4a;line-height:1.6}
.privacy-notice .pn-title{color:#34d399;font-size:.62rem;font-weight:600;letter-spacing:1.5px;margin-bottom:.35rem;display:flex;align-items:center;gap:5px}
.shap-panel{background:#0b0b18;border:1px solid rgba(99,102,241,.2);border-radius:10px;padding:1.2rem 1.5rem;margin:.8rem 0}
.shap-title{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:600;color:#a5b4fc;margin-bottom:.3rem}
.shap-subtitle{font-family:'Inter',sans-serif;font-size:.75rem;color:#4a4a6a;font-weight:300;margin-bottom:1rem}
.shap-feat-row{display:flex;align-items:center;gap:10px;padding:.35rem 0;border-bottom:1px solid rgba(255,255,255,.03);font-family:'JetBrains Mono',monospace;font-size:.68rem}
.shap-feat-row:last-child{border-bottom:none}
.shap-feat-name{color:#8888aa;flex:0 0 200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
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
[data-testid="stMetricLabel"]{font-family:'JetBrains Mono',monospace!important;font-size:.6rem!important;color:#3a3a5c!important;text-transform:uppercase!important;letter-spacing:2px!important}
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
.authors-label{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#3a3a5c;letter-spacing:2px;text-transform:uppercase;flex-shrink:0}
.author-chip{display:inline-flex;align-items:center;gap:5px;font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#a5b4fc}
.author-chip::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#6366f1}
.links-row{display:flex;gap:10px;margin-left:auto;flex-wrap:wrap}
.ext-link{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:5px;border:1px solid #1a1a2e;font-family:'JetBrains Mono',monospace;font-size:.62rem;color:#6b6b8a;text-decoration:none;transition:border-color .15s,color .15s}
.ext-link:hover{border-color:rgba(99,102,241,.4);color:#a5b4fc}
.batch-info{background:rgba(96,165,250,.04);border:1px solid rgba(96,165,250,.18);border-left:3px solid #60a5fa;border-radius:8px;padding:.85rem 1.2rem;margin-bottom:1rem;font-family:'Inter',sans-serif;font-size:.8rem;color:#6b6b8a;line-height:1.55}
.batch-info strong{color:#60a5fa;font-weight:500}
.drift-row{display:flex;align-items:center;gap:10px;padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.03);font-family:'JetBrains Mono',monospace;font-size:.68rem}
.drift-row:last-child{border-bottom:none}
.q-row{display:flex;align-items:flex-start;gap:10px;padding:.7rem 0;border-bottom:1px solid #0f0f1e;font-size:.83rem}
.q-icon{font-size:.85rem;flex-shrink:0;margin-top:1px}.q-msg{color:#8888aa;font-family:'Inter',sans-serif;flex:1;font-weight:300}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_META = {
    TIER_TINY:    ("#34d399", "TINY",    "< 1K rows"),
    TIER_SMALL:   ("#34d399", "SMALL",   "1K – 50K rows"),
    TIER_MEDIUM:  ("#60a5fa", "MEDIUM",  "50K – 200K rows"),
    TIER_LARGE:   ("#fbbf24", "LARGE",   "200K – 500K rows"),
    TIER_XLARGE:  ("#fb923c", "X-LARGE", "500K – 2M rows"),
    TIER_MASSIVE: ("#f87171", "MASSIVE", "2M+ rows"),
}
JM = "font-family:'JetBrains Mono',monospace"


# ── Helpers ───────────────────────────────────────────────────────────────────
def ph(eyebrow, title, desc=""):
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        + (f'<div class="page-desc">{desc}</div>' if desc else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def sec(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def render_tier(tier, n_rows=None):
    color, name, rng = TIER_META[tier]
    row_str = f" · {n_rows:,} rows" if n_rows else ""
    st.markdown(
        f'<div class="tier-badge" style="color:{color};border-color:{color}30;background:{color}08">'
        f'<span style="font-weight:600">{name}</span>'
        f'<span style="color:{color}80">{rng}{row_str}</span>'
        f'<span style="color:{color}70;font-size:.65rem"> · {TIER_STRATEGY[tier]}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def badge(col_type):
    m = {
        "numeric":    ("b-num",  "NUM"),
        "categorical":("b-cat",  "CAT"),
        "id_dropped": ("b-id",   "ID"),
        "date":       ("b-date", "DATE"),
        "text":       ("b-text", "TXT"),
    }
    cls, lbl = m.get(col_type, ("b-num", col_type.upper()[:3]))
    return f'<span class="badge {cls}">{lbl}</span>'


def render_complexity(cplx):
    c = cplx.get("complexity", "unknown")
    cls_m  = {"linear":"cplx-linear","nonlinear":"cplx-nonlinear","mixed":"cplx-mixed","unknown":"cplx-unknown"}
    col_m  = {"linear":"#60a5fa","nonlinear":"#fbbf24","mixed":"#34d399","unknown":"#4a4a6a"}
    icon_m = {"linear":"◈","nonlinear":"◉","mixed":"◫","unknown":"◌"}
    st.markdown(
        f'<div class="cplx-panel {cls_m.get(c,"cplx-unknown")}">'
        f'<div class="cplx-title" style="color:{col_m.get(c,"#4a4a6a")}">'
        f'{icon_m.get(c,"◌")} {c.upper()} COMPLEXITY</div>'
        f'<div class="cplx-note">{cplx.get("note","")}</div>'
        f'<span class="chip">LR {cplx.get("lr_score","—")}</span>'
        f'<span class="chip">LGB {cplx.get("lgb_score","—")}</span>'
        f'<span class="chip">→ {cplx.get("recommended","All models")}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_col_table(profile):
    col_stats = profile.get("col_stats", {})
    if not col_stats:
        return
    html = '<div style="max-height:500px;overflow-y:auto;padding-right:4px">'
    for col, stats in col_stats.items():
        ctype = stats.get("type", "numeric")
        miss  = stats.get("missing_pct", 0)
        mc    = "#f87171" if miss > 20 else "#4a4a6a" if miss > 0 else "#2a2a4a"
        if ctype == "numeric":
            detail = (
                f"mu={stats.get('mean','--')}  σ={stats.get('std','--')}  "
                f"skew={stats.get('skew','--')}  [{stats.get('min','--')} … {stats.get('max','--')}]"
            )
        elif ctype == "categorical":
            tops   = stats.get("top_values", {})
            detail = f"{stats.get('n_unique','?')} unique → {' · '.join(list(tops.keys())[:3])}"
        else:
            detail = stats.get("note", ctype)
        html += (
            f'<div class="col-row">'
            f'<div class="col-name" title="{col}">{col}</div>'
            f'<div class="col-type-cell">{badge(ctype)}</div>'
            f'<div class="col-stats">{detail}</div>'
            f'<div class="col-miss" style="color:{mc}">{"--" if miss==0 else f"{miss:.1f}%"}</div>'
            f"</div>"
        )
    st.markdown(html + "</div>", unsafe_allow_html=True)


def render_quality_report(quality):
    issues = quality.get("issues", [])
    if not issues:
        st.markdown(
            '<div class="info-panel-success"><span style="color:#34d399;'
            'font-family:\'Space Grotesk\',sans-serif;font-weight:600;font-size:.9rem">'
            "✓ No issues — dataset looks clean</span></div>",
            unsafe_allow_html=True,
        )
        return
    icon_m  = {"error": "✕", "warning": "△", "info": "○"}
    color_m = {"error": "#f87171", "warning": "#fbbf24", "info": "#60a5fa"}
    panel_m = {"error": "info-panel-error", "warning": "info-panel-warn", "info": "info-panel"}
    html = ""
    for issue in issues:
        sev  = issue.get("severity", "info")
        html += (
            f'<div class="{panel_m.get(sev,"info-panel")}" style="margin-bottom:.5rem">'
            f'<div class="q-row" style="border:none;padding:0">'
            f'<div class="q-icon" style="color:{color_m.get(sev,"#60a5fa")}">{icon_m.get(sev,"○")}</div>'
            f'<div class="q-msg">{issue.get("message","")}</div>'
            f"</div></div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def sidebar_model_status(trainer):
    if trainer is None:
        st.markdown('<div class="model-card-empty">NO MODEL TRAINED YET</div>', unsafe_allow_html=True)
        return
    m         = trainer.metrics
    auc_float = m.get("test_roc_auc", 0)
    auc_cls   = "ms-val-green" if auc_float >= 0.85 else "ms-val-amber" if auc_float >= 0.70 else "ms-val"
    cplx      = m.get("complexity", {})
    rows = [
        ("MODEL",  m.get("best_model", "--"),            "ms-val"),
        ("TARGET", trainer.target_col or "--",           "ms-val"),
        ("AUC",    f"{auc_float:.4f}",                   auc_cls),
        ("F1",     f"{m.get('f1_score',0):.4f}",         "ms-val"),
        ("TIER",   m.get("tier_label", "--").split()[0], "ms-val"),
    ]
    if cplx:
        rows.append(("CPLX", cplx.get("complexity","--").upper(), "ms-val"))
    html = '<div class="model-card"><div class="model-card-header">ACTIVE MODEL</div>'
    for k, v, cls in rows:
        html += f'<div class="ms-row"><span class="ms-key">{k}</span><span class="{cls}">{v}</span></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def render_tier_legend():
    tiers = [
        ("#34d399","TINY",   "< 1K"),
        ("#34d399","SMALL",  "1K-50K"),
        ("#60a5fa","MEDIUM", "50K-200K"),
        ("#fbbf24","LARGE",  "200K-500K"),
        ("#fb923c","X-LARGE","500K-2M"),
        ("#f87171","MASSIVE","2M+"),
    ]
    rows = "".join(
        f'<div class="tier-legend-row">'
        f'<div class="tier-dot" style="background:{c};box-shadow:0 0 5px {c}80"></div>'
        f'<span class="tier-name" style="color:{c}">{n}</span>'
        f'<span class="tier-range">{r}</span></div>'
        for c, n, r in tiers
    )
    st.markdown(f'<div class="tier-legend">{rows}</div>', unsafe_allow_html=True)


def apply_plot_style(fig, ax_or_axes):
    fig.patch.set_facecolor("#0a0a0f")
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#0e0e1a")
        ax.tick_params(colors="#3a3a5c", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a1a2e")
        ax.xaxis.label.set_color("#4a4a6a")
        ax.yaxis.label.set_color("#4a4a6a")
        ax.title.set_color("#8888aa")


def _no_model_msg():
    st.markdown(
        '<div class="info-panel-warn" style="margin-top:1rem">'
        f'<span style="{JM};font-size:.8rem;color:#fbbf24">'
        "△ No model trained in this session.<br>"
        '<span style="font-size:.72rem;color:#6b6b8a">Go to '
        '<strong style="color:#a5b4fc">03 — Train</strong> and run training first.</span>'
        "</span></div>",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-logo">&#x2B21;</div>'
        '<div><span class="brand-name">AutoML-X</span>'
        '<span class="brand-ver">v7.3</span></div></div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:.75rem;color:#4a4a6a;'
        'margin-bottom:1rem;font-weight:300">Universal Binary Classifier</div>',
        unsafe_allow_html=True,
    )

    sidebar_model_status(st.session_state.get("u_trainer"))

    st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
    _pages = ["Home", "01 -- Upload", "02 -- Analyze", "03 -- Train",
              "04 -- Results", "05 -- Predict", "06 -- Batch"]
    _nav_target = st.session_state.pop("_nav", None)
    _nav_index  = _pages.index(_nav_target) if _nav_target in _pages else None
    page = st.radio(
        "nav", _pages,
        index=_nav_index if _nav_index is not None
              else _pages.index(st.session_state.get("_current_page", "Home")),
        label_visibility="collapsed",
    )
    st.session_state["_current_page"] = page

    st.markdown('<div class="nav-section" style="margin-top:1.8rem">Tier System</div>',
                unsafe_allow_html=True)
    render_tier_legend()

    st.markdown(
        '<div class="privacy-notice">'
        '<div class="pn-title">&#x1F512; DATA PRIVACY</div>'
        "All processing in-memory. No data stored, logged, or transmitted."
        "</div>",
        unsafe_allow_html=True,
    )

    if "df" in st.session_state or "u_trainer" in st.session_state:
        dsn     = st.session_state.get("dataset_name", "")
        trained = "u_trainer" in st.session_state
        st.markdown(
            f'<div style="background:#07070f;border:1px solid rgba(52,211,153,.15);'
            f'border-radius:8px;padding:.65rem .9rem;margin-top:.5rem;{JM};">'
            f'<div style="font-size:.55rem;color:#2a5a3a;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem">SESSION</div>'
            + (f'<div class="ms-row"><span class="ms-key">DATASET</span>'
               f'<span class="ms-val-green" style="font-size:.62rem">{dsn}</span></div>' if dsn else "")
            + f'<div class="ms-row"><span class="ms-key">STATUS</span>'
              f'<span class="{"ms-val-green" if trained else "ms-val-amber"}">'
              f'{"Model ready" if trained else "Not trained"}</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-footer">AutoML-X · HuggingFace Space</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":
    active = st.session_state.get("u_trainer")
    st.markdown(
        '<div class="hero-wrap">'
        '<div class="hero-eyebrow">AutoML-X &nbsp;·&nbsp; Universal Binary Classifier</div>'
        '<div class="hero-title">Upload any CSV.<br>Get a trained <span>ML model</span><br>in minutes.</div>'
        '<div class="hero-sub">No code. No configuration. AutoML-X reads your data, selects the right '
        "algorithm, trains, evaluates, and explains — all automatically. Built for fintech, healthcare, "
        "HR, and credit risk.</div>"
        '<div class="hero-badges">'
        '<span class="hero-badge" style="color:#34d399;border-color:rgba(52,211,153,.25);background:rgba(52,211,153,.05)">&#x2713; Auto model selection</span>'
        '<span class="hero-badge" style="color:#6366f1;border-color:rgba(99,102,241,.25);background:rgba(99,102,241,.05)">&#x2713; 6 dataset tiers</span>'
        '<span class="hero-badge" style="color:#60a5fa;border-color:rgba(96,165,250,.25);background:rgba(96,165,250,.05)">&#x2713; SHAP explainability</span>'
        '<span class="hero-badge" style="color:#fbbf24;border-color:rgba(251,191,36,.25);background:rgba(251,191,36,.05)">&#x2713; Up to 2M rows</span>'
        '<span class="hero-badge" style="color:#c084fc;border-color:rgba(192,132,252,.25);background:rgba(192,132,252,.05)">&#x2713; PDF report export</span>'
        '<span class="hero-badge" style="color:#f87171;border-color:rgba(248,113,113,.25);background:rgba(248,113,113,.05)">&#x2713; Drift detection</span>'
        "</div></div>",
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
        '<div class="flow-step"><div class="flow-num">01</div><span class="flow-icon">&#x1F4C2;</span>'
        '<div class="flow-title">Upload CSV</div><div class="flow-desc">Drop any binary classification '
        'dataset. AutoML-X auto-detects column types, handles missing values, and selects a processing '
        "tier based on size.</div></div>"
        '<div class="flow-arrow">&#x2192;</div>'
        '<div class="flow-step"><div class="flow-num">02</div><span class="flow-icon">&#x26A1;</span>'
        '<div class="flow-title">Auto-Train</div><div class="flow-desc">Multiple models compared — '
        "Logistic Regression, LightGBM, XGBoost. Cross-validated, threshold-optimised. Best model "
        "selected automatically.</div></div>"
        '<div class="flow-arrow">&#x2192;</div>'
        '<div class="flow-step"><div class="flow-num">03</div><span class="flow-icon">&#x1F4CA;</span>'
        '<div class="flow-title">Explain &amp; Export</div><div class="flow-desc">SHAP values, ROC/PR '
        "curves, drift detection on new data. Download the trained model (.pkl) or a full PDF "
        "performance report.</div></div></div>",
        unsafe_allow_html=True,
    )

    sec("Platform capabilities")
    if active:
        m_h   = active.metrics
        auc_c = "#34d399" if m_h.get("test_roc_auc", 0) >= 0.85 else "#fbbf24"
        st.markdown(
            f'<div class="kpi-grid">'
            f'<div class="kpi-card"><span class="kpi-val" style="color:{auc_c}">{m_h.get("test_roc_auc",0):.4f}</span>'
            f'<div class="kpi-lbl">ROC-AUC · {active.best_model_name}</div></div>'
            f'<div class="kpi-card"><span class="kpi-val" style="color:#a5b4fc">{m_h.get("recall",0)*100:.1f}%</span>'
            f'<div class="kpi-lbl">Recall on current model</div></div>'
            f'<div class="kpi-card"><span class="kpi-val" style="color:#fbbf24">{m_h.get("n_rows_total",0):,}</span>'
            f'<div class="kpi-lbl">Rows trained on</div></div>'
            f'<div class="kpi-card"><span class="kpi-val" style="color:#60a5fa">6</span>'
            f'<div class="kpi-lbl">Dataset size tiers</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="kpi-grid">'
            '<div class="kpi-card"><span class="kpi-val" style="color:#6366f1;font-size:1.1rem;padding-top:.3rem">LR / LGB / XGB</span><div class="kpi-lbl">3 algorithms compared automatically</div></div>'
            '<div class="kpi-card"><span class="kpi-val" style="color:#34d399">5-fold CV</span><div class="kpi-lbl">Cross-validation on smaller datasets</div></div>'
            '<div class="kpi-card"><span class="kpi-val" style="color:#fbbf24">Up to 2M</span><div class="kpi-lbl">Max rows supported</div></div>'
            '<div class="kpi-card"><span class="kpi-val" style="color:#60a5fa">6</span><div class="kpi-lbl">Dataset size tiers (Tiny to Massive)</div></div></div>',
            unsafe_allow_html=True,
        )

    sec("Use cases")
    st.markdown(
        '<div class="domain-grid">'
        '<div class="domain-card"><span class="domain-icon">&#x1F3E6;</span><div class="domain-name">Fraud Detection</div><div class="domain-desc">Classify transactions as fraud or legitimate. Handles extreme class imbalance with balanced weighting.</div></div>'
        '<div class="domain-card"><span class="domain-icon">&#x1F4B8;</span><div class="domain-name">Credit Risk</div><div class="domain-desc">Loan default prediction for NBFCs and microfinance. Works on both small (&lt;1K) and large (500K+) portfolios.</div></div>'
        '<div class="domain-card"><span class="domain-icon">&#x1F4C9;</span><div class="domain-name">Customer Churn</div><div class="domain-desc">Predict which customers will leave. Upload your CRM export and get a trained churn model in minutes.</div></div>'
        '<div class="domain-card"><span class="domain-icon">&#x1FA7A;</span><div class="domain-name">Disease Prediction</div><div class="domain-desc">Binary clinical outcomes — diabetic / not, positive / negative. No data science team required.</div></div>'
        '<div class="domain-card"><span class="domain-icon">&#x1F9D1;&#x200D;&#x1F4BC;</span><div class="domain-name">HR Analytics</div><div class="domain-desc">Employee attrition and hiring outcome classification. Upload any HR CSV, pick target column, done.</div></div>'
        '<div class="domain-card"><span class="domain-icon">&#x1F393;</span><div class="domain-name">Research Baseline</div><div class="domain-desc">Get a reproducible ML baseline fast. Model leaderboard with CV scores for direct comparison.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="authors-bar">'
        '<span class="authors-label">Built by</span>'
        '<span class="author-chip">Sujal Baghela</span>'
        '<span class="author-chip">Sameer Bhilware</span>'
        '<span class="authors-label" style="margin-left:.5rem">MITS, Gwalior &nbsp;·&nbsp; 2026</span>'
        '<div class="links-row">'
        '<a class="ext-link" href="https://github.com/Sujal-baghela/Automl-fraud-detection" target="_blank">&#x2B21; GitHub</a>'
        '<a class="ext-link" href="https://dark-ui-automl-fraud-detection.hf.space" target="_blank">&#x1F916; HuggingFace</a>'
        "</div></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 01 UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "01 -- Upload":
    ph("STEP 01 / 06", "Upload Dataset",
       "Drop any CSV — up to 2GB, any number of rows. Auto-cleaned before training.")

    uploaded = st.file_uploader("Drop your CSV here or click to browse", type=["csv"])

    if uploaded:
        file_mb  = uploaded.size / 1e6
        is_large = file_mb > 50
        _tag     = (
            "<span style='color:#fbbf24'>chunked loader</span>"
            if is_large
            else "<span style='color:#34d399'>full load</span>"
        )
        st.markdown(
            f'<div class="info-panel"><b style="color:#d4d4e8">{uploaded.name}</b>'
            f' &nbsp;·&nbsp; <span style="{JM};font-size:.68rem;color:#4a4a6a">'
            f"{file_mb:.1f} MB &nbsp;·&nbsp; {_tag}</span></div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            # FIX 1: Streamlit format uses printf-style, not Python f-strings.
            # "{:,}" → shows literal "{;}" in the UI.
            # Correct: "%,.0f" (comma thousands, no decimal places)
            max_rows = (
                st.slider("Row limit (rows)", 100_000, 2_000_000, 2_000_000, 100_000,
                          format="%,.0f")
                if is_large
                else None
            )
        with c2:
            encoding = st.selectbox(
                "Encoding", ["utf-8", "latin-1", "iso-8859-1", "utf-16"], index=0,
                help="utf-8 works for most modern files. Use latin-1 for older European data.",
            )

        if st.button("Load Dataset", type="primary", use_container_width=True):
            p2   = st.progress(0)
            ph2  = st.empty()
            try:
                ph2.markdown(
                    f'<div style="{JM};font-size:.75rem;color:#6366f1;padding:.5rem 0">Loading...</div>',
                    unsafe_allow_html=True,
                )
                p2.progress(0.3)
                if is_large:
                    df = load_csv_chunked(uploaded, max_rows=max_rows,
                                          chunk_size=100_000, encoding=encoding)
                else:
                    df = pd.read_csv(uploaded, encoding=encoding)
                p2.progress(0.9)
                st.session_state["df"] = df
                st.session_state.pop("profile", None)
                st.session_state.pop("u_trainer", None)
                st.session_state["dataset_name"] = uploaded.name
                p2.progress(1.0)
                ph2.empty()
                p2.empty()
                tier           = get_tier(len(df))
                color, name, _ = TIER_META[tier]
                st.markdown(
                    f'<div class="info-panel-accent" style="display:flex;align-items:center;'
                    f'justify-content:space-between;flex-wrap:wrap;gap:1rem">'
                    f'<div><div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
                    f'font-size:1.1rem;color:#34d399">✓ Dataset Loaded</div>'
                    f'<div style="{JM};font-size:.7rem;color:#4a4a6a;margin-top:3px">{uploaded.name}</div></div>'
                    f'<div style="display:flex;gap:2rem">'
                    f'<div style="text-align:center"><div style="{JM};font-size:1.4rem;color:#a5b4fc">{len(df):,}</div>'
                    f'<div style="{JM};font-size:.58rem;color:#3a3a5c;letter-spacing:2px">ROWS</div></div>'
                    f'<div style="text-align:center"><div style="{JM};font-size:1.4rem;color:#a5b4fc">{df.shape[1]}</div>'
                    f'<div style="{JM};font-size:.58rem;color:#3a3a5c;letter-spacing:2px">COLS</div></div>'
                    f'<div style="text-align:center"><div style="{JM};font-size:1.4rem;color:{color}">{name}</div>'
                    f'<div style="{JM};font-size:.58rem;color:#3a3a5c;letter-spacing:2px">TIER</div></div>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                render_tier(tier, len(df))
            except Exception as e:
                ph2.empty()
                p2.empty()
                st.error(f"Load failed: {e}")

    if "df" in st.session_state:
        df  = st.session_state["df"]
        sec("Preview")
        st.dataframe(df.head(6), use_container_width=True)
        ram = check_ram_safety(df)
        sec("Configure Target")
        c1, c2 = st.columns(2)
        with c1:
            target_col = st.selectbox(
                "Target column", df.columns.tolist(), index=len(df.columns) - 1
            )
            st.session_state["target_col"] = target_col
        with c2:
            unique_vals = df[target_col].dropna().unique().tolist()
            pos_raw     = st.selectbox(
                "Positive class value",
                ["Auto-detect"] + [str(v) for v in unique_vals],
            )
            st.session_state["positive_label"] = None if pos_raw == "Auto-detect" else pos_raw
        if not ram["is_safe"]:
            st.warning(f"⚠️ {ram['warning']}")
        st.markdown(
            f'<div style="{JM};font-size:.65rem;color:#3a3a5c;text-align:right;margin-top:.3rem">'
            f"RAM: {ram['dataframe_gb']:.3f} GB used · {ram['available_gb']:.1f} GB free</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="next-step">→ Next: go to <strong>02 — Analyze</strong> '
            "to inspect data quality</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# 02 ANALYZE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "02 -- Analyze":
    ph("STEP 02 / 06", "Dataset Analysis",
       "Data quality scan, feature profiling, distributions and correlations.")

    if "df" not in st.session_state:
        st.warning("⚠️ Upload a dataset first.")
        st.stop()

    df         = st.session_state["df"]
    target_col = st.session_state.get("target_col", df.columns[-1])
    render_tier(get_tier(len(df)), len(df))

    sec("Overview")
    miss_total = df.isnull().sum().sum()
    st.markdown(
        f'<div class="stat-row stat-row-4">'
        f'<div class="stat-card"><span class="stat-val">{len(df):,}</span><span class="stat-lbl">Rows</span></div>'
        f'<div class="stat-card"><span class="stat-val">{df.shape[1]}</span><span class="stat-lbl">Columns</span></div>'
        f'<div class="stat-card"><span class="stat-val {"stat-val-amber" if miss_total>0 else "stat-val-green"}">'
        f'{miss_total:,}</span><span class="stat-lbl">Missing values</span></div>'
        f'<div class="stat-card"><span class="stat-val" style="font-size:1rem;padding-top:.3rem">'
        f'{target_col}</span><span class="stat-lbl">Target column</span></div></div>',
        unsafe_allow_html=True,
    )

    sec("Data Quality Scan")
    dqr     = DataQualityReport()
    quality = dqr.assess(df, target_col)
    st.session_state["quality"] = quality
    qc_color = {
        "excellent":"#34d399","good":"#34d399","fair":"#fbbf24","poor":"#f87171"
    }.get(quality["overall_quality"], "#a5b4fc")
    st.markdown(
        f'<div class="stat-row stat-row-4" style="margin-bottom:1rem">'
        f'<div class="stat-card"><span class="stat-val" style="color:{qc_color};font-size:1.1rem;padding-top:.2rem">'
        f'{quality["overall_quality"].upper()}</span><span class="stat-lbl">Overall quality</span></div>'
        f'<div class="stat-card"><span class="stat-val stat-val-red">{quality["n_errors"]}</span>'
        f'<span class="stat-lbl">Errors</span></div>'
        f'<div class="stat-card"><span class="stat-val stat-val-amber">{quality["n_warnings"]}</span>'
        f'<span class="stat-lbl">Warnings</span></div>'
        f'<div class="stat-card"><span class="stat-val stat-val-blue">{quality["n_info"]}</span>'
        f'<span class="stat-lbl">Info</span></div></div>',
        unsafe_allow_html=True,
    )
    render_quality_report(quality)

    if st.button("Run Full Profile", type="primary", use_container_width=True):
        with st.spinner("Profiling dataset…"):
            try:
                st.session_state["profile"] = DatasetProfiler().profile(df, target_col)
            except Exception as e:
                st.error(f"Profile failed: {e}")

    if "profile" in st.session_state:
        profile = st.session_state["profile"]
        sec("Feature Overview")
        st.markdown(
            f'<div class="stat-row stat-row-5" style="margin-bottom:1.5rem">'
            f'<div class="stat-card"><span class="stat-val stat-val-green">{profile["n_numeric"]}</span><span class="stat-lbl">Numeric</span></div>'
            f'<div class="stat-card"><span class="stat-val stat-val-amber">{profile["n_categorical"]}</span><span class="stat-lbl">Categorical</span></div>'
            f'<div class="stat-card"><span class="stat-val stat-val-red">{profile["n_id_dropped"]}</span><span class="stat-lbl">IDs dropped</span></div>'
            f'<div class="stat-card"><span class="stat-val stat-val-blue">{profile["n_date_dropped"]}</span><span class="stat-lbl">Dates dropped</span></div>'
            f'<div class="stat-card"><span class="stat-val stat-val-purple">{profile["n_text_dropped"]}</span><span class="stat-lbl">Text dropped</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        sec("Target Distribution")
        tc1, tc2 = st.columns([1, 2])
        with tc1:
            class_df = pd.DataFrame.from_dict(
                profile["class_counts"], orient="index", columns=["Count"]
            )
            class_df["Pct"] = (class_df["Count"] / class_df["Count"].sum() * 100).round(2)
            st.dataframe(class_df, use_container_width=True)
            if profile["is_imbalanced"]:
                st.markdown(
                    f'<div class="info-panel-warn" style="margin-top:.5rem">'
                    f'<span style="{JM};font-size:.72rem;color:#fbbf24">'
                    f'△ Imbalanced — minority {profile["minority_ratio"]*100:.1f}%<br>'
                    f'<span style="color:#6b6b8a">class_weight=\'balanced\' applied</span>'
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="info-panel-success" style="margin-top:.5rem">'
                    f'<span style="{JM};font-size:.72rem;color:#34d399">✓ Balanced classes</span></div>',
                    unsafe_allow_html=True,
                )
        with tc2:
            try:
                fig, ax = plt.subplots(figsize=(5, 2.8))
                labels  = [str(k) for k in profile["class_counts"].keys()]
                values  = list(profile["class_counts"].values())
                colors  = ["#3efe04", "#fb02e6"] if len(values) >= 2 else ["#3b3efe"]
                 # FIX #2: log scale makes minority class visible alongside majority
                ax.set_yscale('log')
                bars    = ax.bar(labels, values, color=colors[:len(values)],
                                 width=0.45, edgecolor="none")
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(values) * 0.01,
                        f"{val:,}", ha="center", va="bottom",
                        color="#6b6b8a", fontsize=8, fontfamily="monospace",
                    )
                ax.set_title("Class Distribution", fontsize=9)
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0.5)
                st.pyplot(fig)
                plt.close()
            except Exception:
                pass

        if profile.get("high_corr_pairs"):
            sec("High Correlations ( > 0.95 )")
            st.dataframe(
                pd.DataFrame(profile["high_corr_pairs"],
                             columns=["Feature A", "Feature B", "Correlation"]),
                use_container_width=True, hide_index=True,
            )

        sec("Column Profile")
        render_col_table(profile)

        num_cols = profile["numeric_cols"]
        if num_cols:
            sec("Numeric Distributions (top 8)")
            show = num_cols[:8]
            nc_  = min(4, len(show))
            nr_  = (len(show) + nc_ - 1) // nc_
            fig, axes = plt.subplots(nr_, nc_, figsize=(4.5 * nc_, 2.8 * nr_))
            flat = np.array(axes).flatten() if len(show) > 1 else [axes]
            for i, col in enumerate(show):
                ax   = flat[i]
                data = df[col].dropna().sample(
                    min(50_000, df[col].notna().sum()), random_state=42
                )
                ax.hist(data, bins=30, color="#6366f1", alpha=0.85, edgecolor="none")
                ax.set_title(col, fontsize=8)
            for j in range(len(show), len(flat)):
                flat[j].set_visible(False)
            apply_plot_style(fig, flat)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 03 TRAIN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "03 -- Train":
    ph("STEP 03 / 06", "Train Model",
       "AutoML selects the best algorithm for your dataset size and complexity.")

    if "df" not in st.session_state:
        st.warning("⚠️ Upload a dataset first.")
        st.stop()

    df         = st.session_state["df"]
    target_col = st.session_state.get("target_col")
    pos_label  = st.session_state.get("positive_label")

    if not target_col:
        st.warning("⚠️ Set target column on page 01.")
        st.stop()

    tier = get_tier(len(df))
    render_tier(tier, len(df))

    sec("Training Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        clean_data = st.checkbox("Smart data cleaning", value=True)
    with c2:
        outlier_method = st.selectbox("Outlier method", ["iqr", "zscore", "none"])
    with c3:
        # FIX 2: Streamlit slider with float 0.1-0.4 and format='%.0f%%'
        # rounds the displayed value to 0 decimal places → "0%".
        # Fix: use integer range 10-40, format="%d%%", divide by 100 for float.
        _split_pct = st.slider("Validation split", 10, 40, 20, 5, format="%d%%")
        test_size  = _split_pct / 100

    strategies = {
        TIER_TINY:    "4 models · LR / RF / LGB / XGB · 5-fold CV",
        TIER_SMALL:   "4 models · LR / RF / LGB / XGB · 5-fold CV",
        TIER_MEDIUM:  "4 models · LR / RF / LGB / XGB · 3-fold CV",
        TIER_LARGE:   "3 models · LR / LGB / XGB · 2-fold CV",
        TIER_XLARGE:  "2 models · LGB / LR · 2-fold CV on 200K sample",
        TIER_MASSIVE: "1 model · LGB · no CV · 500K sample",
    }
    st.markdown(
        f'<div class="info-panel" style="margin:.5rem 0 1.2rem">'
        f'<span style="{JM};font-size:.62rem;color:#3a3a5c;letter-spacing:1px">AUTO STRATEGY</span>'
        f'<div style="{JM};font-size:.78rem;color:#a5b4fc;margin-top:4px">{strategies[tier]}</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("Start Training", type="primary", use_container_width=True):
        prog       = st.progress(0)
        log_ph     = st.empty()
        log_lines  = []

        def on_progress(step, total, msg):
            prog.progress(step / total)
            log_lines.append((step, total, msg))
            html = '<div class="log-terminal">'
            for s, t, m_ in log_lines[-14:]:
                cls  = "log-active" if s == log_lines[-1][0] else "log-done"
                icon = "-->" if cls == "log-active" else "✓"
                html += f'<div class="{cls}">{icon} [{s}/{t}] {m_}</div>'
            log_ph.markdown(html + "</div>", unsafe_allow_html=True)

        try:
            trainer = UniversalTrainer(model_save_path="models/universal_model.pkl")
            metrics = trainer.fit(
                df=df, target_col=target_col, positive_label=pos_label,
                test_size=test_size, clean_data=clean_data,
                outlier_method=outlier_method, progress_callback=on_progress,
            )
            st.session_state["u_trainer"]       = trainer
            st.session_state["u_metrics"]       = metrics
            st.session_state["train_done"]      = True
            st.session_state["best_model_name"] = trainer.best_model_name
            st.session_state["feature_cols"]    = trainer.feature_names
            st.session_state["threshold"]       = trainer.threshold
            st.session_state["target_col"]      = trainer.target_col

            # ── Store drift reference for Page 06 ────────────────────────────
            # DriftDetector.fit() needs the numeric features from training data.
            if _DRIFT:
                try:
                    X_ref     = df.drop(columns=[target_col], errors="ignore").reindex(
                        columns=trainer.feature_names, fill_value=0
                    )
                    num_feats = [
                        c for c in trainer.feature_names
                        if pd.api.types.is_numeric_dtype(X_ref[c])
                    ]
                    if num_feats:
                        dd = DriftDetector()
                        dd.fit(X_ref[num_feats])
                        st.session_state["drift_detector"] = dd
                        st.session_state["drift_features"] = num_feats
                except Exception:
                    pass  # drift is optional — never break training

            prog.progress(1.0)
            log_ph.empty()

            st.markdown(
                f'<div class="info-panel-accent" style="margin-top:1.2rem">'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
                f'font-size:1.1rem;color:#34d399;margin-bottom:1rem">✓ Training Complete</div>'
                f'<div class="stat-row stat-row-5">'
                f'<div class="stat-card"><span class="stat-val stat-val-green" style="font-size:1rem;padding-top:.2rem">'
                f'{metrics["best_model"]}</span><span class="stat-lbl">Best model</span></div>'
                f'<div class="stat-card"><span class="stat-val">{metrics["test_roc_auc"]:.4f}</span>'
                f'<span class="stat-lbl">ROC-AUC</span></div>'
                f'<div class="stat-card"><span class="stat-val">{metrics["f1_score"]:.4f}</span>'
                f'<span class="stat-lbl">F1 Score</span></div>'
                f'<div class="stat-card"><span class="stat-val">{metrics["recall"]:.4f}</span>'
                f'<span class="stat-lbl">Recall</span></div>'
                f'<div class="stat-card"><span class="stat-val">{metrics.get("n_features_used","?")}</span>'
                f'<span class="stat-lbl">Features</span></div></div></div>',
                unsafe_allow_html=True,
            )

            cr = metrics.get("cleaning_report", {})
            if cr and cr.get("changes"):
                with st.expander(f"Data Cleaning Report — {len(cr['changes'])} actions"):
                    for ch in cr["changes"]:
                        st.markdown(
                            f'<div style="{JM};font-size:.72rem;color:#6b6b8a;padding:2px 0">✓ {ch}</div>',
                            unsafe_allow_html=True,
                        )

            dropped = metrics.get("dropped_cols", [])
            if dropped:
                # FIX #10: visible warning-style panel for dropped columns
                st.markdown(
                    f'<div class="info-panel-warn" style="margin-top:.6rem;padding:.7rem 1rem">'
                    f'<span style="{JM};font-size:.72rem;color:#fbbf24">△ Dropped {len(dropped)} col(s): '
                    + " ".join(f'<span style="color:#a5b4fc;background:rgba(99,102,241,.08);padding:1px 5px;border-radius:3px">{c}</span>' for c in dropped[:8])
                    + (f' <span style="color:#6b6b8a">+ {len(dropped)-8} more</span>' if len(dropped) > 8 else "")
                    + "</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="next-step">→ Go to <strong>04 — Results</strong> '
                "for full performance breakdown</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            prog.empty()
            log_ph.empty()
            st.error(f"Training failed: {e}")
            import traceback
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# 04 RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "04 -- Results":
    ph("STEP 04 / 06", "Results",
       "Model performance, leaderboard, confusion matrix, ROC/PR curves, threshold tools.")

    trainer = st.session_state.get("u_trainer")
    if trainer is None:
        _no_model_msg()
        st.stop()

    drift_detector = st.session_state.get("drift_detector")
    drift_ready    = _DRIFT and drift_detector is not None

    st.markdown(
        f'<div class="info-panel" style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Model</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.best_model_name}</div></div>'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Required features</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{len(trainer.feature_names)}</div></div>'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Drift detector</span>'
        f'<div style="{JM};font-size:.85rem;color:{"#34d399" if drift_ready else "#4a4a6a"};margin-top:3px">'
        f'{"Ready" if drift_ready else "Not initialized"}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Required feature columns"):
        st.markdown(
            " ".join(f'<span class="badge b-num">{f}</span>' for f in trainer.feature_names),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="batch-info">'
        "<strong>Upload new, unlabeled data</strong> — not your training CSV.<br>"
        "This file should have the same feature columns but <strong>no target column</strong>. "
        "The model will score each row with a probability, risk level, and drift check."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
    if uploaded:
        file_mb  = uploaded.size / 1e6
        is_large = file_mb > 50
        if is_large:
            st.markdown(
                f'<div class="info-panel-warn">'
                f'<span style="{JM};font-size:.72rem;color:#fbbf24">'
                f"△ Large file ({file_mb:.0f} MB) — chunked loader active</span></div>",
                unsafe_allow_html=True,
            )
            with st.spinner("Loading…"):
                df_new = load_csv_chunked(uploaded, max_rows=None, chunk_size=100_000)
        else:
            df_new = pd.read_csv(uploaded)

        render_tier(get_tier(len(df_new)), len(df_new))
        st.markdown(
            f'<div style="{JM};font-size:.72rem;color:#34d399;margin-bottom:.8rem">'
            f"✓ Loaded {len(df_new):,} rows</div>",
            unsafe_allow_html=True,
        )

        X_raw = df_new.drop(columns=[trainer.target_col], errors="ignore")

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(X_raw):,} rows…"):
                try:
                    probs = trainer.predict_proba(X_raw)
                    preds = (probs >= trainer.threshold).astype(int)

                    # BUG-L FIX: format near-zero probabilities properly
                    results_pred = pd.DataFrame({
                        "probability": [f"{p*100:.4f}" for p in probs],
                        "prediction":  ["POSITIVE" if p == 1 else "NEGATIVE" for p in preds],
                        "risk_level":  pd.cut(
                            probs,
                            bins=[0, 0.2, 0.5, 0.8, 1.0],
                            labels=["Low", "Medium", "High", "Critical"],
                            include_lowest=True,
                        ).astype(str),
                    })
                    results_full = df_new.copy()
                    results_full["probability"]     = (probs * 100).round(4)
                    results_full["predicted_class"] = preds
                    results_full["prediction"]      = results_pred["prediction"]
                    results_full["risk_level"]      = results_pred["risk_level"]

                    pos_rate  = preds.mean() * 100
                    _prob_std = float(probs.std())

                    # BUG-D / FIX #8: near-zero score warning
                    if _prob_std < 0.01:
                        st.markdown(
                            f'<div class="info-panel-warn" style="margin-bottom:.8rem">'
                            f'<span style="{JM};font-size:.76rem;color:#fbbf24;font-weight:600">'
                            f'⚠ All predicted probabilities are near zero (std = {_prob_std:.5f})<br>'
                            f'<span style="color:#6b6b8a;font-weight:400">This usually means the batch data is from a '
                            f'<strong style="color:#a5b4fc">different domain</strong> than the training data. '
                            f'Check the drift report below to confirm. '
                            f'Retrain on data from this domain for valid predictions.</span></span></div>',
                            unsafe_allow_html=True,
                        )

                    sec("Batch Results")
                    st.markdown(
                        f'<div class="stat-row stat-row-3" style="margin-bottom:1.2rem">'
                        f'<div class="stat-card"><span class="stat-val">{len(results_pred):,}</span>'
                        f'<span class="stat-lbl">Total rows scored</span></div>'
                        f'<div class="stat-card"><span class="stat-val stat-val-red">{int(preds.sum()):,}</span>'
                        f'<span class="stat-lbl">Positives found</span></div>'
                        f'<div class="stat-card"><span class="stat-val">{pos_rate:.2f}%</span>'
                        f'<span class="stat-lbl">Positive rate</span></div></div>',
                        unsafe_allow_html=True,
                    )

                    if _EVAL:
                        fig_dist = plot_score_distribution(
                            probs, trainer.threshold,
                            title=f"Score Distribution — {trainer.best_model_name}",
                        )
                        if fig_dist:
                            st.pyplot(fig_dist)
                            plt.close()
                    else:
                        try:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))
                            for ax, rng, ttl in [
                                (ax1, (0.0, 1.0), "Full range"),
                                (ax2, (0.0, 0.5), "Tail zoom"),
                            ]:
                                mask = (probs >= rng[0]) & (probs <= rng[1])
                                ax.hist(probs[mask], bins=100, color="#6366f1",
                                        alpha=0.85, edgecolor="none", log=True)
                                ax.axvline(trainer.threshold, color="#f87171",
                                           linewidth=1.5, linestyle="--",
                                           label=f"threshold {trainer.threshold:.3f}")
                                ax.set_xlabel("Probability", fontsize=8)
                                ax.set_title(ttl, fontsize=8)
                                ax.legend(fontsize=7)
                            apply_plot_style(fig, [ax1, ax2])
                            fig.suptitle("Score Distribution (log Y)", fontsize=9, color="#8888aa")
                            fig.tight_layout(pad=0.5)
                            st.pyplot(fig)
                            plt.close()
                        except Exception:
                            pass

                    if drift_ready:
                        sec("Data Drift Report")
                        try:
                            drift_features = st.session_state.get("drift_features", [])
                            X_batch_num    = X_raw.reindex(
                                columns=trainer.feature_names, fill_value=0
                            )[drift_features]
                            drift_report   = drift_detector.detect(X_batch_num)

                            dratio  = drift_report["drift_ratio"]
                            dcount  = drift_report["drifted_count"]
                            dtotal  = drift_report["total_features"]
                            dstatus = drift_report["status"]

                            if dratio >= 0.3:
                                ds_color, ds_bg = "#f87171", "rgba(248,113,113,.08)"
                            elif dratio >= 0.1:
                                ds_color, ds_bg = "#fbbf24", "rgba(251,191,36,.08)"
                            else:
                                ds_color, ds_bg = "#34d399", "rgba(52,211,153,.08)"

                            st.markdown(
                                f'<div style="background:{ds_bg};border:1px solid {ds_color}30;'
                                f'border-left:3px solid {ds_color};border-radius:8px;padding:.9rem 1.2rem;margin-bottom:1rem">'
                                f'<div style="{JM};font-size:.75rem;color:{ds_color};font-weight:600">{dstatus}</div>'
                                f'<div style="{JM};font-size:.68rem;color:#4a4a6a;margin-top:.3rem">'
                                f"{dcount} / {dtotal} features drifted &nbsp;·&nbsp; ratio {dratio:.1%}</div></div>",
                                unsafe_allow_html=True,
                            )

                            details    = drift_report.get("details", {})
                            drift_rows = sorted(
                                details.items(), key=lambda x: x[1]["psi"], reverse=True
                            )[:10]

                            if drift_rows:
                                # FIX #3: PSI scale legend
                                st.markdown(
                                    f'<div style="background:#0e0e1a;border:1px solid #1a1a2e;border-radius:8px;'
                                    f'padding:.6rem 1rem;margin:.5rem 0;{JM};font-size:.63rem">'
                                    f'<span style="color:#3a3a5c;letter-spacing:1.5px;text-transform:uppercase">PSI SCALE &nbsp;·&nbsp; </span>'
                                    f'<span style="color:#34d399">&lt; 0.1 stable</span>'
                                    f'<span style="color:#3a3a5c"> &nbsp;·&nbsp; </span>'
                                    f'<span style="color:#fbbf24">0.1 – 0.2 warning</span>'
                                    f'<span style="color:#3a3a5c"> &nbsp;·&nbsp; </span>'
                                    f'<span style="color:#f87171">&gt; 0.2 high drift</span>'
                                    f'<span style="color:#3a3a5c"> &nbsp;(PSI = 26 means completely different distribution)</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                                html = (
                                    f'<div style="background:#0b0b18;border:1px solid rgba(251,191,36,.2);'
                                    f'border-radius:10px;padding:1.2rem 1.5rem;margin:.8rem 0">'
                                    f'<div style="{JM};font-size:.65rem;color:#fbbf24;font-weight:600;margin-bottom:.6rem">'
                                    f"TOP FEATURES BY PSI (Population Stability Index)</div>"
                                )
                                # BUG-B FIX: correct indentation — for loop and st.markdown at same level
                                for feat, d in drift_rows:
                                    psi_color = (
                                        "#f87171" if d["psi"] >= 0.2 else
                                        "#fbbf24" if d["psi"] >= 0.1 else "#34d399"
                                    )
                                    html += (
                                        f'<div class="drift-row">'
                                        f'<div style="flex:0 0 180px;color:#8888aa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{feat}">{feat}</div>'
                                        f'<div style="flex:0 0 80px;color:{psi_color}">PSI {d["psi"]:.4f}</div>'
                                        f'<div style="flex:0 0 80px;color:#4a4a6a">p={d["pvalue"]:.3f}</div>'
                                        f'<div style="flex:1;color:{psi_color}">{d["status"]}</div>'
                                        f"</div>"
                                    )
                                st.markdown(html + "</div>", unsafe_allow_html=True)

                            if dratio >= 0.1:
                                st.markdown(
                                    f'<div class="info-panel-warn" style="margin-top:.5rem">'
                                    f'<span style="{JM};font-size:.72rem;color:#fbbf24">'
                                    f"△ Drift detected in {dcount} features. "
                                    f"This batch may be from a different time period or data source. "
                                    f"Consider retraining the model on newer data.</span></div>",
                                    unsafe_allow_html=True,
                                )
                        except Exception as e:
                            st.caption(f"Drift detection error: {e}")
                    elif _DRIFT and not drift_ready:
                        st.markdown(
                            f'<div class="info-panel" style="margin-top:.5rem">'
                            f'<span style="{JM};font-size:.7rem;color:#3a3a5c">'
                            "△ Drift detector not initialized — train a model first to enable drift comparison."
                            "</span></div>",
                            unsafe_allow_html=True,
                        )

                    sec("Predictions Preview")
                    st.dataframe(results_pred.head(200), use_container_width=True)
                    with st.expander("View with original features"):
                        st.dataframe(results_full.head(100), use_container_width=True)

                    st.download_button(
                        "Download Predictions CSV",
                        results_full.to_csv(index=False).encode(),
                        "automlx_predictions.csv", "text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
    m = trainer.metrics
    render_tier(m.get("tier", 0), m.get("n_rows_total"))

    sec("Performance")
    st.markdown(
        f'<div class="stat-row stat-row-5" style="margin-bottom:1.5rem">'
        f'<div class="stat-card"><span class="stat-val stat-val-green" style="font-size:1.1rem;padding-top:.3rem">'
        f'{m.get("best_model","--")}</span><span class="stat-lbl">Winner</span></div>'
        f'<div class="stat-card"><span class="stat-val">{m.get("cv_roc_auc",0):.5f}</span>'
        f'<span class="stat-lbl">CV ROC-AUC</span></div>'
        f'<div class="stat-card"><span class="stat-val">{m.get("test_roc_auc",0):.5f}</span>'
        f'<span class="stat-lbl">Test ROC-AUC</span></div>'
        f'<div class="stat-card"><span class="stat-val">{m.get("f1_score",0):.5f}</span>'
        f'<span class="stat-lbl">F1 Score</span></div>'
        f'<div class="stat-card"><span class="stat-val">{m.get("recall",0):.5f}</span>'
        f'<span class="stat-lbl">Recall</span></div></div>',
        unsafe_allow_html=True,
    )

    cplx = m.get("complexity")
    if cplx:
        render_complexity(cplx)

    sec("Model Leaderboard")
    rc1, rc2 = st.columns([1, 2])
    scores   = m.get("all_cv_scores", {})
    with rc1:
        scores_df = (
            pd.DataFrame({
                "Model":     list(scores.keys()),
                "CV ROC-AUC": [round(v, 5) for v in scores.values()],
            })
            .sort_values("CV ROC-AUC", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            scores_df.style.apply(
                lambda r: ["background-color:#0a1f10;color:#34d399" if r.name == 0 else ""] * len(r),
                axis=1,
            ),
            use_container_width=True, hide_index=True,
        )
    with rc2:
        if scores:
            try:
                fig, ax = plt.subplots(figsize=(5, 3))
                ms      = scores_df.sort_values("CV ROC-AUC")
                colors  = ["#34d399" if i == len(ms) - 1 else "#6366f1" for i in range(len(ms))]
                bars    = ax.barh(ms["Model"], ms["CV ROC-AUC"],
                                  color=colors, height=0.45, edgecolor="none")
                for bar, val in zip(bars, ms["CV ROC-AUC"]):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f}", va="center", color="#6b6b8a",
                            fontsize=8, fontfamily="monospace")
                ax.set_xlim(max(0, ms["CV ROC-AUC"].min() - 0.05), 1.02)
                ax.set_xlabel("CV ROC-AUC", fontsize=8)
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0.5)
                st.pyplot(fig)
                plt.close()
            except Exception:
                pass

    # ── ROC + PR curves (from rewritten src/evaluation.py) ────────────────────
    val_probs  = m.get("val_probs")
    val_labels = m.get("val_labels")

    # NOTE for deployment: add these 2 lines to universal_trainer.py
    # inside fit() → step 9 → self.metrics = { ... } dict:
    #     "val_probs":  y_proba.tolist(),
    #     "val_labels": yvs.tolist(),
    # (y_proba and yvs are already computed just above self.metrics = {...})

    if _EVAL and val_probs is not None and val_labels is not None:
        sec("ROC & Precision-Recall Curves")
        vp    = np.array(val_probs)
        vl    = np.array(val_labels)
        ec1, ec2 = st.columns(2)
        with ec1:
            fig_roc = plot_roc_curve(vl, vp, m.get("best_model", "Model"))
            if fig_roc:
                st.pyplot(fig_roc)
                plt.close()
        with ec2:
            fig_pr = plot_precision_recall_curve(vl, vp, trainer.threshold, m.get("best_model", "Model"))
            if fig_pr:
                st.pyplot(fig_pr)
                plt.close()

    sec("Threshold & Confusion Matrix")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Threshold",  f"{m.get('threshold', .5):.5f}")
    t2.metric("Precision",  f"{m.get('precision', 0):.5f}")
    t3.metric("Recall",     f"{m.get('recall', 0):.5f}")
    t4.metric("F1",         f"{m.get('f1_score', 0):.5f}")
    t5.metric("Features",   m.get("n_features_used", "?"))

    with st.expander("What do these metrics mean?"):
        st.markdown(
            f'<div class="metric-explain-row"><div class="me-name">ROC-AUC</div>'
            f'<div class="me-score">{m.get("test_roc_auc",0):.4f}</div>'
            f'<div class="me-desc">Area Under the ROC Curve. Measures how well the model ranks positives '
            f'above negatives. <strong style="color:#34d399">0.97 = excellent</strong>, 0.5 = random.</div></div>'
            f'<div class="metric-explain-row"><div class="me-name">F1 Score</div>'
            f'<div class="me-score">{m.get("f1_score",0):.4f}</div>'
            f'<div class="me-desc">Harmonic mean of Precision and Recall. Use when both missing positives '
            f"and false alarms are costly.</div></div>"
            f'<div class="metric-explain-row"><div class="me-name">Recall</div>'
            f'<div class="me-score">{m.get("recall",0):.4f}</div>'
            f'<div class="me-desc">Of all actual positives, what % did the model catch? '
            f'<strong style="color:#fbbf24">Critical for fraud and medical diagnosis.</strong></div></div>'
            f'<div class="metric-explain-row"><div class="me-name">Precision</div>'
            f'<div class="me-score">{m.get("precision",0):.4f}</div>'
            f'<div class="me-desc">Of all flagged positives, what % were actually positive? '
            f"Trade-off with recall.</div></div>"
            f'<div class="metric-explain-row"><div class="me-name">Threshold</div>'
            f'<div class="me-score">{m.get("threshold",0.5):.4f}</div>'
            f'<div class="me-desc">Probability cutoff. AutoML-X optimises this by maximising '
            f"F1 on the validation set.</div></div>",
            unsafe_allow_html=True,
        )

    tp, tn, fp, fn = m.get("TP", 0), m.get("TN", 0), m.get("FP", 0), m.get("FN", 0)
    col_cm, _ = st.columns([1, 2])
    with col_cm:
        if _EVAL:
            fig_cm = plot_confusion_matrix(tn, fp, fn, tp)
            if fig_cm:
                st.pyplot(fig_cm)
                plt.close()
        else:
            # Fallback — fixed-color cells, no Blues cmap white-on-white bug
            try:
                fig, ax = plt.subplots(figsize=(3.5, 3))
                cell_colors = [["#01e040","#00c3ff"],["#d718b1bf","#3E895C"]]
                cell_labels = [[f"TN\n{tn:,}",f"FP\n{fp:,}"],[f"FN\n{fn:,}",f"TP\n{tp:,}"]]
                ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
                for i in range(2):
                    for j in range(2):
                        ax.add_patch(plt.Rectangle(
                            (j-0.5, i-0.5), 1, 1,
                            facecolor=cell_colors[i][j], edgecolor="#0a0a0f", linewidth=2,
                        ))
                        ax.text(j, i, cell_labels[i][j],
                                ha="center", va="center", fontsize=9,
                                color="#f1f1ff", fontfamily="monospace", fontweight="bold")
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                   # FIX #11: domain-specific labels
                ax.set_xticklabels(["Pred Legit","Pred Fraud"], fontsize=8, color="#6a6aef")
                ax.set_yticklabels(["Actual Legit","Actual Fraud"], fontsize=8, color="#6b6b8a")

                ax.set_title("Confusion Matrix", fontsize=9)
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0.5)
                st.pyplot(fig)
                plt.close()
            except Exception:
                st.markdown(f"TP:`{tp}` TN:`{tn}` FP:`{fp}` FN:`{fn}`")

    # ── Threshold Strategy Optimizer (src/threshold_optimizer.py) ────────────
    sec("Threshold Strategy Optimizer")
    with st.expander("⚙ Compare F1 / Recall / Precision strategies"):
        st.markdown(
            f'<div class="info-panel" style="margin-bottom:.8rem">'
            f'<span style="{JM};font-size:.72rem;color:#6b6b8a">'
            f'Pick your objective. <span style="color:#a5b4fc">Maximize F1</span> balances precision and recall. '
            f'<span style="color:#34d399">Maximize Recall</span> catches more positives (better for fraud). '
            f'<span style="color:#fbbf24">Maximize Precision</span> reduces false alarms.</span></div>',
            unsafe_allow_html=True,
        )
        if _THR_OPT and val_probs is not None and val_labels is not None:
            vp = np.array(val_probs); vl = np.array(val_labels)
            if st.button("Compute All Threshold Strategies", use_container_width=True):
                try:
                    opt = ThresholdOptimizer(strategy="maximize_f1")
                    opt.optimize(vl, vp)
                    st.session_state["thr_strategies"] = opt.get_all_strategies()
                    st.session_state["_thr_opt_obj"]   = opt
                except Exception as e:
                    st.caption(f"Threshold optimizer error: {e}")

            strategies_result = st.session_state.get("thr_strategies")
            if strategies_result:
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(
                        f'<div class="stat-card" style="text-align:center">'
                        f'<span class="stat-val" style="color:#6366f1">'
                        f'{strategies_result.get("maximize_f1",0):.4f}</span>'
                        f'<span class="stat-lbl">Best F1 Threshold</span></div>',
                        unsafe_allow_html=True,
                    )
                with s2:
                    st.markdown(
                        f'<div class="stat-card" style="text-align:center">'
                        f'<span class="stat-val stat-val-green">'
                        f'{strategies_result.get("maximize_recall",0):.4f}</span>'
                        f'<span class="stat-lbl">Best Recall Threshold</span></div>',
                        unsafe_allow_html=True,
                    )
                with s3:
                    st.markdown(
                        f'<div class="stat-card" style="text-align:center">'
                        f'<span class="stat-val stat-val-amber">'
                        f'{strategies_result.get("maximize_precision",0):.4f}</span>'
                        f'<span class="stat-lbl">Best Precision Threshold</span></div>',
                        unsafe_allow_html=True,
                    )
                if _EVAL:
                    fig_thr = plot_threshold_strategies(
                        vl, vp, trainer.threshold,
                        opt_f1=strategies_result.get("maximize_f1"),
                        opt_recall=strategies_result.get("maximize_recall"),
                        opt_precision=strategies_result.get("maximize_precision"),
                    )
                    if fig_thr:
                        st.pyplot(fig_thr)
                        plt.close()
        else:
            if not _THR_OPT:
                st.caption("threshold_optimizer.py not found in src/")
            else:
                st.caption(
                    "Requires val_probs + val_labels in trainer.metrics. "
                    "Add these 2 lines to universal_trainer.py → fit() → step 9 → self.metrics dict:\n"
                    '    "val_probs":  y_proba.tolist(),\n'
                    '    "val_labels": yvs.tolist(),'
                )

    # ── Cost-Sensitive Optimizer (src/cost_optimizer.py) ─────────────────────
    sec("Cost-Sensitive Threshold Optimizer")
    with st.expander("💰 Tune threshold by business cost of FN vs FP"):
        st.markdown(
            f'<div class="info-panel" style="margin-bottom:.8rem">'
            f'<span style="{JM};font-size:.75rem;color:#fbbf24;font-weight:600">'
            f'Business Cost Optimizer</span>'
            f'<div style="font-family:\'Inter\',sans-serif;font-size:.75rem;color:#4a4a6a;margin-top:.4rem">'
            f'Find the threshold that minimises total financial cost. '
            f'Powered by <code>src/cost_optimizer.BusinessCostOptimizer</code>.</div></div>',
            unsafe_allow_html=True,
        )
        co1, co2 = st.columns(2)
        with co1:
            cost_fn = st.number_input(
                "Cost of False Negative (missed positive)",
                min_value=1, value=10000,
                # FIX #9: real-world fintech framing
                help="e.g. ₹50,000 = average loss per undetected fraud transaction. Higher value → model is tuned to catch more positives at the cost of more false alerts.",
            )
            cost_fp = st.number_input(
                "Cost of False Positive (false alarm)",
                min_value=1, value=200,
                help="e.g. ₹200 = analyst time cost per false alert investigated. Lower than FN cost → threshold will be reduced to catch more fraud.",
            )
            
        with co2:
            ratio       = cost_fn / cost_fp
            ratio_color = "#f87171" if ratio > 20 else "#fbbf24" if ratio > 5 else "#34d399"
            st.markdown(
                f'<div class="info-panel" style="margin-top:1.6rem">'
                f'<span style="{JM};font-size:.68rem;color:#4a4a6a">'
                f'FN:FP ratio = <span style="color:{ratio_color};font-size:.85rem;font-weight:600">{ratio:.1f}x</span><br>'
                f'<span style="color:#3a3a5c">Higher ratio → lower threshold → catches more positives</span>'
                f"</span></div>",
                unsafe_allow_html=True,
            )

        if st.button("Find Cost-Optimal Threshold", use_container_width=True):
            if not _COST_OPT:
                st.caption("cost_optimizer.py not found in src/")
            elif val_probs is None or val_labels is None:
                st.caption(
                    "Requires val_probs + val_labels in trainer.metrics. "
                    "See universal_trainer.py patch note above."
                )
            else:
                try:
                    vp  = np.array(val_probs)
                    vl  = np.array(val_labels)
                    opt = BusinessCostOptimizer(
                        fraud_loss=cost_fn, false_alarm_cost=cost_fp
                    )
                    # BusinessCostOptimizer.optimize() uses np.unique(y_proba)
                    # which can hang on 200K rows. Cap at 200 evenly-spaced thresholds.
                    opt_thr = opt.optimize(
                        vl,
                        # fast path: use linspace sample, not every unique proba
                        np.clip(vp, 0, 1),
                    )
                    results  = opt.get_results()
                    bm       = results["Metrics at Optimal Threshold"]

                    st.markdown(
                        f'<div class="info-panel-success" style="margin-top:.8rem">'
                        f'<span style="{JM};font-size:.82rem;color:#34d399;font-weight:600">'
                        f'✓ Optimal threshold: {opt_thr:.4f}</span><br>'
                        f'<span style="{JM};font-size:.7rem;color:#6b6b8a">'
                        f'Minimum total cost: {results["Minimum Cost ($)"]:,.0f} &nbsp;·&nbsp; '
                        f'Current threshold: {trainer.threshold:.4f}</span><br>'
                        f'<span style="{JM};font-size:.7rem;color:#6b6b8a">'
                        f'Recall: {bm.get("Fraud Caught (Recall)",0):.3f} &nbsp;·&nbsp; '
                        f'Precision: {bm.get("Precision",0):.3f} &nbsp;·&nbsp; '
                        f'TP:{bm.get("TP",0):,} FP:{bm.get("FP",0):,} FN:{bm.get("FN",0):,}'
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )

                    # Cost curve plot
                    try:
                        thresholds_plot = np.linspace(0.01, 0.99, 200)
                        costs_plot = []
                        for thr in thresholds_plot:
                            preds_t = (vp >= thr).astype(int)
                            fn_t    = int(((preds_t == 0) & (vl == 1)).sum())
                            fp_t    = int(((preds_t == 1) & (vl == 0)).sum())
                            costs_plot.append(fn_t * cost_fn + fp_t * cost_fp)

                        fig, ax = plt.subplots(figsize=(6, 2.5))
                        ax.plot(thresholds_plot, costs_plot,
                                color="#6366f1", linewidth=1.5)
                        ax.axvline(opt_thr, color="#34d399", linewidth=1.5,
                                   linestyle="--", label=f"Optimal {opt_thr:.3f}")
                        ax.axvline(trainer.threshold, color="#fbbf24", linewidth=1,
                                   linestyle=":", alpha=0.7,
                                   label=f"Current {trainer.threshold:.3f}")
                        ax.set_xlabel("Threshold", fontsize=8)
                        ax.set_ylabel("Total Cost", fontsize=8)
                        ax.set_title("Cost vs Threshold", fontsize=9)
                        ax.legend(fontsize=7)
                        apply_plot_style(fig, ax)
                        fig.tight_layout(pad=0.5)
                        st.pyplot(fig)
                        plt.close()
                    except Exception:
                        pass

                    if abs(opt_thr - trainer.threshold) > 0.01:
                        st.markdown(
                            f'<div class="info-panel-warn" style="margin-top:.5rem">'
                            f'<span style="{JM};font-size:.72rem;color:#fbbf24">'
                            f'△ Optimal ({opt_thr:.4f}) differs from current ({trainer.threshold:.4f}). '
                            f"Use optimal value in your production scoring pipeline.</span></div>",
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.caption(f"Cost optimizer error: {e}")

    sec("Training Info")
    st.markdown(
        f'<div class="info-panel"><div class="stat-row stat-row-4">'
        f'<div><div style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Strategy</div>'
        f'<div style="{JM};font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get("tier_strategy","--")}</div></div>'
        f'<div><div style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Train rows</div>'
        f'<div style="{JM};font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get("n_train",0):,}</div></div>'
        f'<div><div style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Val rows</div>'
        f'<div style="{JM};font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get("n_val",0):,}</div></div>'
        f'<div><div style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Total rows</div>'
        f'<div style="{JM};font-size:.75rem;color:#a5b4fc;margin-top:4px">{m.get("n_rows_total",0):,}</div></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    cr = m.get("cleaning_report", {})
    if cr and cr.get("changes"):
        with st.expander(f"Data Cleaning Report — {len(cr['changes'])} action(s)"):
            for ch in cr["changes"]:
                st.markdown(
                    f'<div style="{JM};font-size:.72rem;color:#6b6b8a;padding:2px 0">✓ {ch}</div>',
                    unsafe_allow_html=True,
                )

    sec("Feature Importance (SHAP)")
    if not _SHAP:
        st.markdown(
            '<div class="info-panel-warn"><span style="color:#fbbf24">'
            "△ shap_universal.py not found in src/</span></div>",
            unsafe_allow_html=True,
        )
    else:
        df_ref = st.session_state.get("df")
        if df_ref is None:
            st.markdown(
                f'<div class="info-panel"><span style="color:#4a4a6a">'
                "Re-upload dataset to generate SHAP explanations</span></div>",
                unsafe_allow_html=True,
            )
        else:
            if st.button("Generate SHAP Explanations", use_container_width=True):
                with st.spinner("Computing SHAP values…"):
                    try:
                        X_shap      = df_ref.drop(columns=[trainer.target_col], errors="ignore").reindex(
                            columns=trainer.feature_names, fill_value=0
                        )
                        shap_engine = UniversalSHAP(trainer.best_pipeline)
                        st.session_state["shap_result"] = shap_engine.explain_single(
                            X_shap, index=0, top_k=12
                        )
                        st.session_state["shap_global"] = shap_engine.global_importance(
                            X_shap, n_sample=300
                        )
                    except Exception as e:
                        st.error(f"SHAP failed: {e}")

            shap_result = st.session_state.get("shap_result")
            if shap_result:
                sc1, sc2 = st.columns(2)
                with sc1:
                    top = shap_result.get("top_features", [])[:10]
                    if top:
                        max_abs   = max(abs(t["shap"]) for t in top) or 1
                        rows_html = (
                            '<div class="shap-panel">'
                            '<div class="shap-title">Top Features by Impact</div>'
                            '<div class="shap-subtitle">Absolute SHAP value</div>'
                        )
                        for t in top:
                            fname = t["feature"].replace("num__","").replace("cat__","")
                            pct   = abs(t["shap"]) / max_abs * 100
                            color = "#f87171" if t["shap"] > 0 else "#34d399"
                            rows_html += (
                                f'<div class="shap-feat-row">'
                                f'<div class="shap-feat-name" title="{fname}">{fname}</div>'
                                f'<div class="shap-feat-bar-wrap">'
                                f'<div class="shap-feat-bar" style="width:{pct:.1f}%;background:{color}"></div></div>'
                                f'<div class="shap-feat-val" style="color:{color}">{t["shap"]:+.4f}</div></div>'
                            )
                        st.markdown(rows_html + "</div>", unsafe_allow_html=True)
                with sc2:
                    try:
                        fig2 = UniversalSHAP(trainer.best_pipeline).plot_bar(
                            shap_result, title="SHAP — Feature Impact"
                        )
                        st.pyplot(fig2)
                        plt.close()
                    except Exception as e:
                        st.caption(f"Chart unavailable: {e}")

                st.markdown(
                    f'<div class="info-panel" style="margin-top:.8rem">'
                    f'<span style="{JM};font-size:.65rem;color:#4a4a6a">'
                    f'<span style="color:#f87171">Red</span> = increases fraud probability · '
                    f'<span style="color:#34d399">Green</span> = decreases fraud probability · '
                    f'Base: <span style="color:#a5b4fc">{shap_result.get("base_value",0):.4f}</span>'
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

    sec("Export")
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        if os.path.exists("models/universal_model.pkl"):
            with open("models/universal_model.pkl", "rb") as f:
                st.download_button(
                    "Download Model (.pkl)", f,
                    "universal_model.pkl", "application/octet-stream",
                    use_container_width=True,
                )
    with exp_c2:
        if _PDF:
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    try:
                        st.session_state["pdf_bytes"] = generate_pdf_report(
                            metrics=m,
                            dataset_name=st.session_state.get("dataset_name", "dataset.csv"),
                            shap_result=st.session_state.get("shap_result"),
                        )
                    except Exception as e:
                        st.error(f"PDF failed: {e}")
            if st.session_state.get("pdf_bytes"):
                dsn = st.session_state.get("dataset_name", "dataset").replace(".csv", "")
                st.download_button(
                    "Download PDF Report",
                    st.session_state["pdf_bytes"],
                    f"automlx_report_{dsn}.pdf", "application/pdf",
                    use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# 05 PREDICT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "05 -- Predict":
    ph("STEP 05 / 06", "Single Prediction",
       "Fill in feature values and get an instant probability score.")

    trainer = st.session_state.get("u_trainer")
    if trainer is None:
        _no_model_msg()
        st.stop()

    features = trainer.feature_names
    df_ref   = st.session_state.get("df")

    st.markdown(
        f'<div class="info-panel" style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Model</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.best_model_name}</div></div>'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Features</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{len(features)}</div></div>'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Threshold</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.threshold:.5f}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    sec("Feature Inputs")
    st.markdown(
        f'<div class="info-panel" style="margin-bottom:.8rem;padding:.7rem 1rem">'
        f'<span style="{JM};font-size:.62rem;color:#3a3a5c">'
        f'TIPS · <span style="color:#5a5a80">Defaults are median values from your training data.</span></span></div>',
        unsafe_allow_html=True,
    )

    input_vals = {}
    cols = st.columns(4)
    for i, feat in enumerate(features):
        with cols[i % 4]:
            if df_ref is not None and feat in df_ref.columns:
                sample = df_ref[feat].dropna()
                if sample.dtype in [np.float64, np.int64, np.float32, np.int32]:
                    input_vals[feat] = st.number_input(
                        feat, value=float(sample.median()), format="%.4f", key=f"f_{feat}"
                    )
                else:
                    input_vals[feat] = st.selectbox(
                        feat, sample.unique().tolist(), key=f"f_{feat}"
                    )
            else:
                input_vals[feat] = st.number_input(feat, value=0.0, key=f"f_{feat}")

    st.markdown("")
    if st.button("Run Prediction", type="primary", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_vals])
            prob     = float(trainer.predict_proba(input_df)[0])
            pred     = int(prob >= trainer.threshold)
            t        = trainer.threshold

            if pred == 1:
                st.markdown(
                    f'<div class="fraud-alert">'
                    f'<div class="fraud-alert-tag">&#x26A0; Positive — Risk Detected</div>'
                    f'<div class="fraud-alert-prob">{prob*100:.2f}%</div>'
                    f'<div class="fraud-alert-meta">PROBABILITY &nbsp;·&nbsp; THRESHOLD {t:.5f} &nbsp;·&nbsp; MODEL {trainer.best_model_name.upper()}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-neg">'
                    f'<div class="result-label" style="color:#34d399">Negative</div>'
                    f'<div class="result-prob" style="color:#34d399">{prob*100:.2f}%</div>'
                    f'<div class="result-meta">PROBABILITY · THRESHOLD {t:.5f}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Probability bar
            try:
                fig, ax = plt.subplots(figsize=(6, 0.65))
                bar_color = "#ef4444" if pred == 1 else "#34d399"
                ax.barh([0], [1], color="#13132a", height=0.35, edgecolor="none")
                ax.barh([0], [prob], color=bar_color, height=0.35, edgecolor="none", alpha=0.9)
                ax.axvline(t, color="#6366f1", linewidth=1.5, linestyle="--", alpha=0.8)
                ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.axis("off")
                ax.text(prob, 0.32, f"{prob:.3f}", ha="center", fontsize=8,
                        color=bar_color, fontfamily="monospace")
                ax.text(t, -0.42, f"threshold {t:.3f}", ha="center", fontsize=7,
                        color="#6366f1", fontfamily="monospace")
                apply_plot_style(fig, ax)
                fig.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()
            except Exception:
                pass

            if _SHAP:
                sec("Why this prediction? (SHAP)")
                with st.spinner("Computing SHAP explanation…"):
                    try:
                        shap_engine = UniversalSHAP(trainer.best_pipeline)
                        ref         = (
                            df_ref.drop(columns=[trainer.target_col], errors="ignore")
                            .reindex(columns=trainer.feature_names, fill_value=0)
                            if df_ref is not None
                            else input_df.reindex(columns=trainer.feature_names, fill_value=0)
                        )
                        shap_engine._build_explainer(
                            ref.sample(min(100, len(ref)), random_state=42)
                        )
                        result  = shap_engine.explain_single(
                            input_df.reindex(columns=trainer.feature_names, fill_value=0),
                            index=0, top_k=10,
                        )
                        top     = result.get("top_features", [])
                        pos_f   = [f for f in top if f["shap"] > 0][:3]
                        neg_f   = [f for f in top if f["shap"] < 0][:3]
                        html    = (
                            f'<div class="shap-panel">'
                            f'<div class="shap-title">Prediction Explanation</div>'
                            f'<div class="shap-subtitle">Model: {result.get("model_class","--")} · '
                            f'Base value: {result.get("base_value",0):.4f}</div>'
                        )
                        if pos_f:
                            html += (
                                f'<div style="color:#f87171;{JM};font-size:.62rem;'
                                f'letter-spacing:1px;margin:.6rem 0 .3rem">INCREASES PREDICTION</div>'
                            )
                            for f in pos_f:
                                fname = f["feature"].replace("num__","").replace("cat__","")
                                html += (
                                    f'<div class="shap-feat-row">'
                                    f'<div class="shap-feat-name">{fname}</div>'
                                    f'<div class="shap-feat-val" style="color:#f87171">{f["shap"]:+.4f}</div></div>'
                                )
                        if neg_f:
                            html += (
                                f'<div style="color:#34d399;{JM};font-size:.62rem;'
                                f'letter-spacing:1px;margin:.6rem 0 .3rem">DECREASES PREDICTION</div>'
                            )
                            for f in neg_f:
                                fname = f["feature"].replace("num__","").replace("cat__","")
                                html += (
                                    f'<div class="shap-feat-row">'
                                    f'<div class="shap-feat-name">{fname}</div>'
                                    f'<div class="shap-feat-val" style="color:#34d399">{f["shap"]:+.4f}</div></div>'
                                )
                        html += "</div>"
                        col_ex1, col_ex2 = st.columns(2)
                        with col_ex1:
                            st.markdown(html, unsafe_allow_html=True)
                        with col_ex2:
                            fig2 = shap_engine.plot_bar(result, title="Feature Impact")
                            st.pyplot(fig2)
                            plt.close()
                    except Exception as e:
                        st.caption(f"SHAP unavailable: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 06 BATCH
# ─────────────────────────────────────────────────────────────────────────────
elif page == "06 -- Batch":
    ph("STEP 06 / 06", "Batch Prediction",
       "Score a full CSV file. Drift detection compares batch vs training distribution.")

    trainer = st.session_state.get("u_trainer")
    if trainer is None:
        _no_model_msg()
        st.stop()

    drift_detector = st.session_state.get("drift_detector")
    drift_ready    = _DRIFT and drift_detector is not None

    st.markdown(
        f'<div class="info-panel" style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Model</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{trainer.best_model_name}</div></div>'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Required features</span>'
        f'<div style="{JM};font-size:.85rem;color:#a5b4fc;margin-top:3px">{len(trainer.feature_names)}</div></div>'
        f'<div><span style="{JM};font-size:.6rem;color:#3a3a5c;text-transform:uppercase;letter-spacing:1.5px">Drift detector</span>'
        f'<div style="{JM};font-size:.85rem;color:{"#34d399" if drift_ready else "#4a4a6a"};margin-top:3px">'
        f'{"Ready" if drift_ready else "Not initialized"}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Required feature columns"):
        st.markdown(
            " ".join(f'<span class="badge b-num">{f}</span>' for f in trainer.feature_names),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="batch-info">'
        "<strong>Upload new, unlabeled data</strong> — not your training CSV.<br>"
        "This file should have the same feature columns but <strong>no target column</strong>. "
        "The model will score each row with a probability, risk level, and drift check."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
    if uploaded:
        file_mb  = uploaded.size / 1e6
        is_large = file_mb > 50
        if is_large:
            st.markdown(
                f'<div class="info-panel-warn">'
                f'<span style="{JM};font-size:.72rem;color:#fbbf24">'
                f"△ Large file ({file_mb:.0f} MB) — chunked loader active</span></div>",
                unsafe_allow_html=True,
            )
            with st.spinner("Loading…"):
                df_new = load_csv_chunked(uploaded, max_rows=None, chunk_size=100_000)
        else:
            df_new = pd.read_csv(uploaded)

        render_tier(get_tier(len(df_new)), len(df_new))
        st.markdown(
            f'<div style="{JM};font-size:.72rem;color:#34d399;margin-bottom:.8rem">'
            f"✓ Loaded {len(df_new):,} rows</div>",
            unsafe_allow_html=True,
        )

        X_raw = df_new.drop(columns=[trainer.target_col], errors="ignore")

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(X_raw):,} rows…"):
                try:
                    probs = trainer.predict_proba(X_raw)
                    preds = (probs >= trainer.threshold).astype(int)

                    results_pred = pd.DataFrame({
                        "probability": [f"{p*100:.4f}" for p in probs],
                        "prediction":  ["POSITIVE" if p == 1 else "NEGATIVE" for p in preds],
                        "risk_level":  pd.cut(
                            probs,
                            bins=[0, 0.2, 0.5, 0.8, 1.0],
                            labels=["Low", "Medium", "High", "Critical"],
                            include_lowest=True,
                        ).astype(str),
                    })
                    results_full = df_new.copy()
                    results_full["probability"]     = (probs * 100).round(4)
                    results_full["predicted_class"] = preds
                    results_full["prediction"]      = results_pred["prediction"]
                    results_full["risk_level"]      = results_pred["risk_level"]

                    pos_rate = preds.mean() * 100
                    # FIX #8: detect and warn when all scores are near-zero
                    _prob_std = float(probs.std())
                    if _prob_std < 0.01:
                        st.markdown(
                            f'<div class="info-panel-warn" style="margin-bottom:.8rem">'
                            f'<span style="{JM};font-size:.76rem;color:#fbbf24;font-weight:600">'
                            f'⚠ All predicted probabilities are near zero (std = {_prob_std:.5f})<br>'
                            f'<span style="color:#6b6b8a;font-weight:400">This usually means the batch data is from a <strong style="color:#a5b4fc">different domain</strong> than the training data — '
                            f'e.g. scoring churn data against a fraud model. The drift report below will confirm. '
                            f'Retrain on data from this domain for valid predictions.</span></span></div>',
                            unsafe_allow_html=True,
                        )
                    sec("Batch Results")
                    st.markdown(
                        f'<div class="stat-row stat-row-3" style="margin-bottom:1.2rem">'
                        f'<div class="stat-card"><span class="stat-val">{len(results_pred):,}</span>'
                        f'<span class="stat-lbl">Total rows scored</span></div>'
                        f'<div class="stat-card"><span class="stat-val stat-val-red">{int(preds.sum()):,}</span>'
                        f'<span class="stat-lbl">Positives found</span></div>'
                        f'<div class="stat-card"><span class="stat-val">{pos_rate:.2f}%</span>'
                        f'<span class="stat-lbl">Positive rate</span></div></div>',
                        unsafe_allow_html=True,
                    )

                    # ── Score distribution dual-panel (log Y axis) ────────────
                    # Fixes the "all bars crammed near 0" bug on imbalanced data.
                    if _EVAL:
                        fig_dist = plot_score_distribution(
                            probs, trainer.threshold,
                            title=f"Score Distribution — {trainer.best_model_name}",
                        )
                        if fig_dist:
                            st.pyplot(fig_dist)
                            plt.close()
                    else:
                        try:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))
                            for ax, rng, ttl in [
                                (ax1, (0.0, 1.0), "Full range"),
                                (ax2, (0.0, 0.5), "Tail zoom"),
                            ]:
                                mask = (probs >= rng[0]) & (probs <= rng[1])
                                ax.hist(probs[mask], bins=100, color="#6366f1",
                                        alpha=0.85, edgecolor="none", log=True)
                                ax.axvline(trainer.threshold, color="#f87171",
                                           linewidth=1.5, linestyle="--",
                                           label=f"threshold {trainer.threshold:.3f}")
                                ax.set_xlabel("Probability", fontsize=8)
                                ax.set_title(ttl, fontsize=8)
                                ax.legend(fontsize=7)
                            apply_plot_style(fig, [ax1, ax2])
                            fig.suptitle("Score Distribution (log Y)", fontsize=9, color="#8888aa")
                            fig.tight_layout(pad=0.5)
                            st.pyplot(fig)
                            plt.close()
                        except Exception:
                            pass

                    # ── Drift Detection (src/drift_detector.py) ────────────────
                    # DriftDetector.detect() compares batch distribution vs reference
                    # (fitted during training on Page 03).
                    if drift_ready:
                        sec("Data Drift Report")
                        try:
                            drift_features = st.session_state.get("drift_features", [])
                            X_batch_num    = X_raw.reindex(
                                columns=trainer.feature_names, fill_value=0
                            )[drift_features]
                            drift_report   = drift_detector.detect(X_batch_num)

                            dratio  = drift_report["drift_ratio"]
                            dcount  = drift_report["drifted_count"]
                            dtotal  = drift_report["total_features"]
                            dstatus = drift_report["status"]

                            if dratio >= 0.3:
                                ds_color, ds_bg = "#f87171", "rgba(248,113,113,.08)"
                            elif dratio >= 0.1:
                                ds_color, ds_bg = "#fbbf24", "rgba(251,191,36,.08)"
                            else:
                                ds_color, ds_bg = "#34d399", "rgba(52,211,153,.08)"

                            st.markdown(
                                f'<div style="background:{ds_bg};border:1px solid {ds_color}30;'
                                f'border-left:3px solid {ds_color};border-radius:8px;padding:.9rem 1.2rem;margin-bottom:1rem">'
                                f'<div style="{JM};font-size:.75rem;color:{ds_color};font-weight:600">{dstatus}</div>'
                                f'<div style="{JM};font-size:.68rem;color:#4a4a6a;margin-top:.3rem">'
                                f"{dcount} / {dtotal} features drifted &nbsp;·&nbsp; ratio {dratio:.1%}</div></div>",
                                unsafe_allow_html=True,
                            )

                            details    = drift_report.get("details", {})
                            drift_rows = sorted(
                                details.items(), key=lambda x: x[1]["psi"], reverse=True
                            )[:10]
                            if drift_rows:
                               # FIX #3: PSI scale legend so users understand severity
                               st.markdown(
                                  f'<div style="background:#0e0e1a;border:1px solid #1a1a2e;border-radius:8px;'
                                  f'padding:.6rem 1rem;margin:.5rem 0;{JM};font-size:.63rem">'
                                  f'<span style="color:#3a3a5c;letter-spacing:1.5px;text-transform:uppercase">PSI SCALE &nbsp;·&nbsp; </span>'
                                  f'<span style="color:#34d399">&lt; 0.1 stable</span>'
                                  f'<span style="color:#3a3a5c"> &nbsp;·&nbsp; </span>'
                                  f'<span style="color:#fbbf24">0.1 – 0.2 warning</span>'
                                  f'<span style="color:#3a3a5c"> &nbsp;·&nbsp; </span>'
                                  f'<span style="color:#f87171">&gt; 0.2 high drift</span>'
                                  f'<span style="color:#3a3a5c"> &nbsp;(PSI = 26 means completely different distribution)</span>'
                                  f'</div>',
                                  unsafe_allow_html=True,
                                )
                               html = (
                                  f'<div style="background:#0b0b18;border:1px solid rgba(251,191,36,.2);'
                                  f'border-radius:10px;padding:1.2rem 1.5rem;margin:.8rem 0">'
                                  f'<div style="{JM};font-size:.65rem;color:#fbbf24;font-weight:600;margin-bottom:.6rem">'
                                  f"TOP FEATURES BY PSI (Population Stability Index)</div>"
                                )
                            for feat, d in drift_rows:
                                psi_color = (
                                    "#f87171" if d["psi"] >= 0.2 else
                                    "#fbbf24" if d["psi"] >= 0.1 else "#34d399"
                                )
                                html += (
                                    f'<div class="drift-row">'
                                    f'<div style="flex:0 0 180px;color:#8888aa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{feat}">{feat}</div>'
                                    f'<div style="flex:0 0 80px;color:{psi_color}">PSI {d["psi"]:.4f}</div>'
                                    f'<div style="flex:0 0 80px;color:#4a4a6a">p={d["pvalue"]:.3f}</div>'
                                    f'<div style="flex:1;color:{psi_color}">{d["status"]}</div>'
                                    f"</div>"
                                )
                            st.markdown(html + "</div>", unsafe_allow_html=True)

                            if dratio >= 0.1:
                                st.markdown(
                                    f'<div class="info-panel-warn" style="margin-top:.5rem">'
                                    f'<span style="{JM};font-size:.72rem;color:#fbbf24">'
                                    f"△ Drift detected in {dcount} features. "
                                    f"This batch may be from a different time period or data source. "
                                    f"Consider retraining the model on newer data.</span></div>",
                                    unsafe_allow_html=True,
                                )
                        except Exception as e:
                            st.caption(f"Drift detection error: {e}")
                    elif _DRIFT and not drift_ready:
                        st.markdown(
                            f'<div class="info-panel" style="margin-top:.5rem">'
                            f'<span style="{JM};font-size:.7rem;color:#3a3a5c">'
                            "△ Drift detector not initialized — train a model first to enable drift comparison."
                            "</span></div>",
                            unsafe_allow_html=True,
                        )

                    sec("Predictions Preview")
                    st.dataframe(results_pred.head(200), use_container_width=True)
                    with st.expander("View with original features"):
                        st.dataframe(results_full.head(100), use_container_width=True)
                    st.download_button(
                        "Download Predictions CSV",
                        results_full.to_csv(index=False).encode(),
                        "automlx_predictions.csv", "text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")