import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


def generate_evaluation_reports(pipeline, X_test, y_test, threshold=0.5):
    """
    Generate evaluation reports using the optimized threshold.

    FIX: Previously used pipeline.predict() which defaults to threshold=0.5.
    Now accepts the optimized threshold so the confusion matrix and
    classification report reflect real production behavior.

    Args:
        pipeline  : Trained sklearn pipeline
        X_test    : Test features
        y_test    : True labels
        threshold : Decision threshold (default 0.5, should pass optimized value)
    """

    logger.info("Generating evaluation reports with threshold=%.5f...", threshold)

    os.makedirs("reports/evaluation", exist_ok=True)

    # =========================
    # Predictions
    # FIX: use optimized threshold instead of sklearn default 0.5
    # =========================
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # =========================
    # ROC Curve
    # =========================
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/evaluation/roc_curve.png", dpi=150)
    plt.close()

    logger.info("ROC AUC: %.5f", roc_auc)

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold={threshold:.4f})")
    plt.tight_layout()
    plt.savefig("reports/evaluation/confusion_matrix.png", dpi=150)
    plt.close()

    # =========================
    # Classification Report
    # =========================
    report = classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Fraud"]
    )

    report_content = (
        f"Threshold Used: {threshold:.5f}\n"
        f"ROC AUC: {roc_auc:.5f}\n\n"
        f"{report}"
    )

    with open("reports/evaluation/classification_report.txt", "w") as f:
        f.write(report_content)

    logger.info("Evaluation reports saved to reports/evaluation/")

    return {
        "roc_auc": round(roc_auc, 5),
        "threshold": threshold,
        "confusion_matrix": cm.tolist()
    }
