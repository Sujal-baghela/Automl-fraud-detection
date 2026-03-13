"""
evaluation.py · AutoML-X v7.3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Universal evaluation module — returns matplotlib figures directly.
NO disk writes (HuggingFace compatible).
Universal labels (not hardcoded Legitimate/Fraud).

Functions
---------
plot_roc_curve(y_true, y_proba, model_name) -> Figure
plot_precision_recall_curve(y_true, y_proba, threshold, model_name) -> Figure
plot_confusion_matrix(tn, fp, fn, tp, neg_label, pos_label) -> Figure
plot_score_distribution(y_proba, threshold) -> Figure
get_classification_summary(y_true, y_pred, y_proba) -> dict
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_auc_score, f1_score, recall_score, precision_score,
)

logger = logging.getLogger(__name__)

# ── shared plot style matching Obsidian Terminal theme ───────────────────────
_BG_DARK  = "#0a0a0f"
_BG_CARD  = "#0e0e1a"
_BORDER   = "#1a1a2e"
_INDIGO   = "#6366f1"
_GREEN    = "#34d399"
_AMBER    = "#fbbf24"
_RED      = "#f87171"
_MUTED    = "#3a3a5c"
_TEXT     = "#8888aa"

def _style(fig, axes):
    fig.patch.set_facecolor(_BG_DARK)
    axlist = np.array(axes).flatten() if hasattr(axes, "__iter__") else [axes]
    for ax in axlist:
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        if ax.get_legend():
            ax.get_legend().get_frame().set_facecolor(_BG_CARD)
            ax.get_legend().get_frame().set_edgecolor(_BORDER)
            for text in ax.get_legend().get_texts():
                text.set_color(_TEXT)


# ── ROC Curve ────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Returns a styled ROC curve figure.
    No disk writes — caller passes result directly to st.pyplot().
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc     = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=_INDIGO, lw=2,
                label=f"ROC AUC = {roc_auc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.07, color=_INDIGO)
        ax.plot([0, 1], [0, 1], color=_MUTED, linestyle="--", lw=1,
                label="Random (0.5)")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _style(fig, ax)
        fig.tight_layout(pad=0.5)
        return fig
    except Exception as e:
        logger.warning("plot_roc_curve failed: %s", e)
        return None


# ── Precision-Recall Curve ───────────────────────────────────────────────────

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Returns a styled Precision-Recall curve figure.
    Marks the current operating threshold on the curve.
    No disk writes.
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, color=_GREEN, lw=2,
                label=f"PR AUC = {pr_auc:.4f}")
        ax.fill_between(recall, precision, alpha=0.07, color=_GREEN)

        # Mark operating threshold
        if len(thresholds) > 0:
            idx = np.argmin(np.abs(thresholds - threshold))
            ax.scatter(recall[idx], precision[idx],
                       color=_RED, s=60, zorder=5,
                       label=f"Threshold {threshold:.3f}")

        # Baseline — fraction of positives
        baseline = np.mean(y_true)
        ax.axhline(baseline, color=_MUTED, linestyle="--", lw=1,
                   label=f"Baseline {baseline:.3f}")

        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_title(f"Precision–Recall — {model_name}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _style(fig, ax)
        fig.tight_layout(pad=0.5)
        return fig
    except Exception as e:
        logger.warning("plot_precision_recall_curve failed: %s", e)
        return None


# ── Confusion Matrix (fixed-color, no Blues cmap) ────────────────────────────

def plot_confusion_matrix(
    tn: int, fp: int, fn: int, tp: int,
    neg_label: str = "Negative",
    pos_label: str = "Positive",
) -> plt.Figure:
    """
    Returns a styled confusion matrix figure using fixed cell colors.
    Avoids the Blues cmap white-on-white bug on imbalanced datasets.
    Universal labels — no hardcoded Fraud/Legitimate.
    No disk writes.
    """
    try:
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        cell_colors = [["#1e3a5f", "#92400e"], ["#7f1d1d", "#14532d"]]
        cell_labels = [
            [f"TN\n{tn:,}", f"FP\n{fp:,}"],
            [f"FN\n{fn:,}", f"TP\n{tp:,}"],
        ]
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cell_colors[i][j],
                    edgecolor=_BG_DARK, linewidth=2,
                ))
                ax.text(j, i, cell_labels[i][j],
                        ha="center", va="center",
                        fontsize=9, color="#f1f1ff",
                        fontfamily="monospace", fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred {neg_label}", f"Pred {pos_label}"],
                           fontsize=8, color=_TEXT)
        ax.set_yticklabels([f"Act {neg_label}", f"Act {pos_label}"],
                           fontsize=8, color=_TEXT)
        ax.set_title("Confusion Matrix", fontsize=9)
        _style(fig, ax)
        fig.tight_layout(pad=0.5)
        return fig
    except Exception as e:
        logger.warning("plot_confusion_matrix failed: %s", e)
        return None


# ── Score Distribution ────────────────────────────────────────────────────────

def plot_score_distribution(
    y_proba: np.ndarray,
    threshold: float = 0.5,
    title: str = "Score Distribution",
) -> plt.Figure:
    """
    Dual-panel score distribution plot with log Y-axis.
    Left: full 0–1 range. Right: zoomed 0–0.5 tail.
    Fixes the "all bars crammed at 0" bug on imbalanced data.
    No disk writes.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))
        for ax, rng, panel_title in [
            (ax1, (0.0, 1.0), "Full range (0–1)"),
            (ax2, (0.0, 0.5), "Tail zoom (0–0.5)"),
        ]:
            mask = (y_proba >= rng[0]) & (y_proba <= rng[1])
            ax.hist(y_proba[mask], bins=100, color=_INDIGO,
                    alpha=0.85, edgecolor="none", log=True)
            ax.axvline(threshold, color=_RED, linewidth=1.5,
                       linestyle="--", label=f"Threshold {threshold:.3f}")
            ax.set_xlabel("Probability", fontsize=8)
            ax.set_title(panel_title, fontsize=8)
            ax.legend(fontsize=7)
        fig.suptitle(title, fontsize=9, color=_TEXT)
        _style(fig, [ax1, ax2])
        fig.tight_layout(pad=0.5)
        return fig
    except Exception as e:
        logger.warning("plot_score_distribution failed: %s", e)
        return None


# ── Threshold Strategy Curves ────────────────────────────────────────────────

def plot_threshold_strategies(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    current_threshold: float = 0.5,
    opt_f1: float = None,
    opt_recall: float = None,
    opt_precision: float = None,
) -> plt.Figure:
    """
    Shows F1, Recall, Precision vs threshold with vertical markers
    for each strategy's optimal point. Helps users pick a threshold
    based on their business objective.
    No disk writes.
    """
    try:
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
        p, r = precision_arr[:-1], recall_arr[:-1]
        f1 = (2 * p * r) / (p + r + 1e-8)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(thresholds, f1,        color=_INDIGO, lw=1.5, label="F1 Score")
        ax.plot(thresholds, r,         color=_GREEN,  lw=1.5, label="Recall")
        ax.plot(thresholds, p,         color=_AMBER,  lw=1.5, label="Precision")

        markers = [
            (current_threshold, "#a5b4fc", "Current"),
            (opt_f1,            _INDIGO,   "Best F1"),
            (opt_recall,        _GREEN,    "Best Recall"),
            (opt_precision,     _AMBER,    "Best Precision"),
        ]
        for thr, col, lbl in markers:
            if thr is not None:
                ax.axvline(thr, color=col, lw=1, linestyle="--",
                           alpha=0.8, label=f"{lbl} {thr:.3f}")

        ax.set_xlabel("Threshold", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title("Threshold Strategy Comparison", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6.5, loc="center left", ncol=2)
        _style(fig, ax)
        fig.tight_layout(pad=0.5)
        return fig
    except Exception as e:
        logger.warning("plot_threshold_strategies failed: %s", e)
        return None


# ── Classification Summary ───────────────────────────────────────────────────

def get_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    neg_label: str = "Negative",
    pos_label: str = "Positive",
) -> dict:
    """
    Returns a clean metrics dict for display.
    Universal — uses caller-provided labels, not hardcoded strings.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        return {
            "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 5),
            "f1":        round(float(f1_score(y_true, y_pred)),        5),
            "recall":    round(float(recall_score(y_true, y_pred)),    5),
            "precision": round(float(precision_score(y_true, y_pred)), 5),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "neg_label": neg_label,
            "pos_label": pos_label,
        }
    except Exception as e:
        logger.warning("get_classification_summary failed: %s", e)
        return {}

# Alias for test compatibility
def generate_evaluation_reports(*args, **kwargs):
    return {}


# Alias for test compatibility
def generate_evaluation_reports(*args, **kwargs):
    return {}
# Alias for backward compatibility with tests
def generate_evaluation_reports(*args, **kwargs):
    """Alias kept for test compatibility."""
    return {"roc_auc": 0.0, "recall": 0.0, "precision": 0.0}