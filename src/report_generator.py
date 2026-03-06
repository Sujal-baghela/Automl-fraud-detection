"""
src/report_generator.py  --  AutoML-X v7.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates a professional PDF report after training.
Uses matplotlib + PdfPages (no external deps beyond matplotlib).

Output: bytes (so Streamlit can st.download_button it directly)
"""

import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = "#0a0a0f"
CARD     = "#0e0e1a"
BORDER   = "#1a1a2e"
INDIGO   = "#6366f1"
GREEN    = "#34d399"
RED      = "#f87171"
AMBER    = "#fbbf24"
BLUE     = "#60a5fa"
TEXT     = "#d4d4e8"
MUTED    = "#6b6b8a"
DIM      = "#3a3a5c"


def _set_page_style(fig):
    fig.patch.set_facecolor(BG)


def _hline(fig, y, xmin=0.08, xmax=0.92, color=BORDER, lw=0.6):
    """Draw a horizontal rule across the figure using a Line2D artist."""
    line = Line2D([xmin, xmax], [y, y], transform=fig.transFigure,
                  color=color, linewidth=lw, clip_on=False)
    fig.add_artist(line)


def _card(ax, facecolor=CARD, edgecolor=BORDER):
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(0.6)


def _page_cover(pdf, metrics: dict, dataset_name: str):
    fig = plt.figure(figsize=(8.27, 11.69))
    _set_page_style(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(BG)

    # top accent bar
    ax.axhline(0.96, color=INDIGO, linewidth=3, xmin=0.08, xmax=0.92)

    # logo text (ASCII-safe)
    ax.text(0.5, 0.88, "[A]", ha="center", va="center",
            fontsize=42, color=INDIGO, fontfamily="monospace", fontweight="bold")

    ax.text(0.5, 0.81, "AutoML-X", ha="center", va="center",
            fontsize=28, fontweight="bold", color=TEXT)
    ax.text(0.5, 0.77, "Model Performance Report", ha="center", va="center",
            fontsize=13, color=MUTED)

    # divider
    ax.axhline(0.73, color=BORDER, linewidth=0.6, xmin=0.15, xmax=0.85)

    # dataset info
    ax.text(0.5, 0.68, dataset_name, ha="center", va="center",
            fontsize=11, color=BLUE, fontfamily="monospace")
    ax.text(0.5, 0.64,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}",
            ha="center", va="center", fontsize=9, color=DIM)

    # key metrics boxes
    kpi_items = [
        ("Best Model",  metrics.get("best_model", "--"),               INDIGO),
        ("ROC-AUC",     f"{metrics.get('test_roc_auc', 0):.4f}",      GREEN),
        ("F1 Score",    f"{metrics.get('f1_score', 0):.4f}",           BLUE),
        ("Recall",      f"{metrics.get('recall', 0):.4f}",             AMBER),
    ]
    box_w = 0.18
    starts = [0.5 - 1.5 * (box_w + 0.03) + i * (box_w + 0.03) for i in range(4)]
    for (lbl, val, color), x in zip(kpi_items, starts):
        rect = mpatches.FancyBboxPatch(
            (x, 0.50), box_w, 0.09,
            boxstyle="round,pad=0.01",
            facecolor=CARD, edgecolor=color, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + box_w/2, 0.575, val, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, fontfamily="monospace")
        ax.text(x + box_w/2, 0.513, lbl, ha="center", va="center",
                fontsize=7, color=MUTED)

    # tier + strategy
    tier_lbl = metrics.get("tier_label", "--")
    strategy = metrics.get("tier_strategy", "--")
    ax.text(0.5, 0.44, f"Tier: {tier_lbl}", ha="center", fontsize=9,
            color=AMBER, fontfamily="monospace")
    ax.text(0.5, 0.40, strategy, ha="center", fontsize=8, color=DIM)

    # footer
    ax.axhline(0.05, color=BORDER, linewidth=0.6, xmin=0.08, xmax=0.92)
    ax.text(0.5, 0.025, "AutoML-X  |  Universal Binary Classifier  |  HuggingFace Space",
            ha="center", va="center", fontsize=7, color=DIM)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_metrics(pdf, metrics: dict):
    fig = plt.figure(figsize=(8.27, 11.69))
    _set_page_style(fig)

    # title block
    fig.text(0.08, 0.94, "STEP 01", fontsize=7, color=INDIGO,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.08, 0.91, "Performance Metrics", fontsize=16,
             fontweight="bold", color=TEXT)
    fig.text(0.08, 0.875, "Full evaluation results on held-out test set",
             fontsize=9, color=MUTED)
    _hline(fig, 0.86)

    # ── KPI grid (6 boxes, 3 per row) ─────────────────────────────────────
    grid_items = [
        ("CV ROC-AUC",   f"{metrics.get('cv_roc_auc', 0):.5f}",   GREEN),
        ("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0):.5f}",  GREEN),
        ("F1 Score",     f"{metrics.get('f1_score', 0):.5f}",      BLUE),
        ("Recall",       f"{metrics.get('recall', 0):.5f}",        AMBER),
        ("Precision",    f"{metrics.get('precision', 0):.5f}",     INDIGO),
        ("Threshold",    f"{metrics.get('threshold', 0.5):.5f}",   MUTED),
    ]
    # Draw boxes directly on a dedicated axes to avoid transform issues
    ax_kpi = fig.add_axes([0.05, 0.69, 0.90, 0.16])
    ax_kpi.set_xlim(0, 1); ax_kpi.set_ylim(0, 1); ax_kpi.axis("off")
    ax_kpi.set_facecolor(BG)
    cols_n, rows_n = 3, 2
    cell_w = 1.0 / cols_n
    cell_h = 1.0 / rows_n
    pad = 0.02
    for idx, (lbl, val, color) in enumerate(grid_items):
        col_i = idx % cols_n
        row_i = idx // cols_n
        x = col_i * cell_w + pad
        y = (1 - (row_i + 1) * cell_h) + pad
        w = cell_w - 2 * pad
        h = cell_h - 2 * pad
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.01", facecolor=CARD,
            edgecolor=color, linewidth=0.8)
        ax_kpi.add_patch(rect)
        ax_kpi.text(x + w/2, y + h * 0.65, val,
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=color, fontfamily="monospace")
        ax_kpi.text(x + w/2, y + h * 0.18, lbl,
                    ha="center", va="center", fontsize=7, color=MUTED)

    # ── Confusion matrix ───────────────────────────────────────────────────
    tp = metrics.get("TP", 0); tn = metrics.get("TN", 0)
    fp = metrics.get("FP", 0); fn = metrics.get("FN", 0)
    ax_cm = fig.add_axes([0.08, 0.38, 0.33, 0.26])
    _card(ax_cm)
    cm_data = np.array([[tn, fp], [fn, tp]], dtype=float)
    ax_cm.imshow(cm_data, cmap="Blues", aspect="auto")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred NEG", "Pred POS"], fontsize=7, color=MUTED)
    ax_cm.set_yticklabels(["Act NEG", "Act POS"],  fontsize=7, color=MUTED)
    for i, (row_vals, row_colors) in enumerate([
        ([f"TN\n{tn:,}", f"FP\n{fp:,}"], [GREEN, RED]),
        ([f"FN\n{fn:,}", f"TP\n{tp:,}"], [RED,   GREEN]),
    ]):
        for j, (lbl, c) in enumerate(zip(row_vals, row_colors)):
            ax_cm.text(j, i, lbl, ha="center", va="center",
                       fontsize=8, color=TEXT, fontfamily="monospace",
                       fontweight="bold")
    ax_cm.set_title("Confusion Matrix", fontsize=8, color=MUTED, pad=6)
    ax_cm.tick_params(colors=MUTED)

    # ── Model leaderboard ──────────────────────────────────────────────────
    scores = metrics.get("all_cv_scores", {})
    if scores:
        ax_lb = fig.add_axes([0.50, 0.38, 0.42, 0.26])
        _card(ax_lb)
        names  = list(scores.keys())
        vals   = list(scores.values())
        best_i = vals.index(max(vals))
        colors = [GREEN if i == best_i else INDIGO for i in range(len(names))]
        ax_lb.barh(names, vals, color=colors, edgecolor="none", height=0.5)
        for i, (name, val) in enumerate(zip(names, vals)):
            ax_lb.text(val + 0.002, i, f"{val:.4f}", va="center",
                       fontsize=6.5, color=MUTED, fontfamily="monospace")
        ax_lb.set_xlim(max(0, min(vals) - 0.05), 1.02)
        ax_lb.set_title("Model Leaderboard (CV AUC)", fontsize=8, color=MUTED, pad=6)
        ax_lb.tick_params(colors=MUTED, labelsize=7)
        ax_lb.set_xlabel("CV ROC-AUC", fontsize=7, color=DIM)
        for spine in ax_lb.spines.values():
            spine.set_edgecolor(BORDER)

    # ── Training info text ─────────────────────────────────────────────────
    _hline(fig, 0.36)
    fig.text(0.08, 0.34, "TRAINING INFO", fontsize=7, color=DIM,
             fontfamily="monospace", fontweight="bold")
    info_items = [
        f"Strategy  :  {metrics.get('tier_strategy', '--')}",
        f"Tier      :  {metrics.get('tier_label', '--')}",
        f"Train rows:  {metrics.get('n_train', 0):,}    |    Val rows: {metrics.get('n_val', 0):,}",
        f"Total rows:  {metrics.get('n_rows_total', 0):,}    |    Features used: {metrics.get('n_features_used', '?')}",
        f"Best model:  {metrics.get('best_model', '--')}",
    ]
    for i, txt in enumerate(info_items):
        fig.text(0.08, 0.315 - i * 0.025, txt, fontsize=8,
                 color=TEXT, fontfamily="monospace")

    # footer
    _hline(fig, 0.05)
    fig.text(0.5, 0.025, "AutoML-X  |  Universal Binary Classifier",
             ha="center", fontsize=7, color=DIM)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_shap(pdf, shap_result: dict, title: str = "Feature Importance (SHAP)"):
    fig = plt.figure(figsize=(8.27, 11.69))
    _set_page_style(fig)

    fig.text(0.08, 0.94, "STEP 02", fontsize=7, color=INDIGO,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.08, 0.91, "Model Explainability", fontsize=16,
             fontweight="bold", color=TEXT)
    fig.text(0.08, 0.875,
             "SHAP values show which features pushed this prediction higher or lower.",
             fontsize=9, color=MUTED)
    _hline(fig, 0.86)

    top = shap_result.get("top_features", [])[:12]
    if not top:
        fig.text(0.5, 0.5, "SHAP not available for this model",
                 ha="center", color=MUTED, fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    names  = [t["feature"].replace("num__", "").replace("cat__", "")
              for t in top]
    values = [t["shap"] for t in top]
    bar_colors = [RED if v > 0 else GREEN for v in values]

    ax = fig.add_axes([0.08, 0.24, 0.84, 0.58])
    _card(ax)
    ax.barh(names[::-1], values[::-1], color=bar_colors[::-1],
            edgecolor="none", height=0.6)
    ax.axvline(0, color=DIM, linewidth=0.8)
    ax.set_xlabel("SHAP value  (red = increases prediction, green = decreases)",
                  fontsize=7, color=MUTED)
    ax.set_title(title, fontsize=9, color=MUTED, pad=6)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    # value labels on bars
    patches = ax.patches
    rev_vals = values[::-1]
    for bar, val in zip(patches, rev_vals):
        offset = 0.002 if val >= 0 else -0.002
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha,
                fontsize=6.5, color="#a5b4fc", fontfamily="monospace")

    # base value note
    bv = shap_result.get("base_value", None)
    model_cls = shap_result.get("model_class", "--")
    if bv is not None:
        fig.text(0.08, 0.21,
                 f"Base value (model avg output): {bv:.4f}  |  Model: {model_cls}",
                 fontsize=8, color=DIM, fontfamily="monospace")

    # legend
    pos_patch = mpatches.Patch(color=RED,   label="Increases prediction")
    neg_patch = mpatches.Patch(color=GREEN, label="Decreases prediction")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              fontsize=7, facecolor=CARD, edgecolor=BORDER,
              labelcolor=TEXT)

    _hline(fig, 0.05)
    fig.text(0.5, 0.025, "AutoML-X  |  Universal Binary Classifier",
             ha="center", fontsize=7, color=DIM)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_privacy(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    _set_page_style(fig)

    fig.text(0.08, 0.94, "STEP 03", fontsize=7, color=INDIGO,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.08, 0.91, "Privacy & Methodology", fontsize=16,
             fontweight="bold", color=TEXT)
    _hline(fig, 0.86)

    sections = [
        ("DATA PRIVACY", GREEN, [
            "No data is stored or transmitted. All processing happens in-memory.",
            "Your CSV is cleared from the server immediately after the session ends.",
            "No personally identifiable information is logged or retained.",
            "This tool is for analysis only -- outputs should be reviewed by a professional.",
        ]),
        ("HOW IT WORKS", BLUE, [
            "1. Upload CSV -> auto-detected column types, smart cleaning applied.",
            "2. Tier system selects training strategy based on dataset size (6 tiers).",
            "3. Multiple models trained with cross-validation, best selected by AUC.",
            "4. Threshold optimised by maximising F1 on held-out validation set.",
            "5. SHAP values computed to explain which features drove each prediction.",
        ]),
        ("MODEL LIMITATIONS", AMBER, [
            "Binary classification only. Multi-class targets are not supported.",
            "Results depend on data quality. Noisy or biased data produces biased models.",
            "This is a general-purpose AutoML tool, not a domain-specific system.",
            "Always validate predictions with domain expertise before acting on them.",
        ]),
    ]

    y = 0.82
    for title, color, bullets in sections:
        fig.text(0.08, y, title, fontsize=8, color=color,
                 fontfamily="monospace", fontweight="bold")
        y -= 0.028
        for bullet in bullets:
            fig.text(0.10, y, f"- {bullet}", fontsize=8.5, color=TEXT)
            y -= 0.030
        y -= 0.018

    _hline(fig, 0.05)
    fig.text(0.5, 0.025,
             "AutoML-X  |  Universal Binary Classifier  |  HuggingFace Space",
             ha="center", fontsize=7, color=DIM)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf_report(
    metrics: dict,
    dataset_name: str = "Dataset",
    shap_result: dict = None,
) -> bytes:
    """
    Generate a PDF report and return as bytes for st.download_button.

    Args:
        metrics:      trainer.metrics dict from UniversalTrainer
        dataset_name: filename of the uploaded CSV
        shap_result:  output of UniversalSHAP.explain_single() -- optional

    Returns:
        bytes: PDF file contents
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _page_cover(pdf, metrics, dataset_name)
        _page_metrics(pdf, metrics)
        if shap_result:
            _page_shap(pdf, shap_result,
                       title=f"Feature Importance -- {metrics.get('best_model','Model')}")
        _page_privacy(pdf)

        d = pdf.infodict()
        d["Title"]   = "AutoML-X Model Report"
        d["Author"]  = "AutoML-X"
        d["Subject"] = f"Binary Classification Report -- {dataset_name}"
        d["Creator"] = "AutoML-X v7.1"

    return buf.getvalue()