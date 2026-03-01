import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class IntelligentSHAP:
    """
    SHAP wrapper for tree-based sklearn pipelines.
    Supports RandomForest, LightGBM, GradientBoosting.
    Compatible with SHAP 0.50.0+ and NamedColumnTransformer.
    """

    def __init__(self, pipeline):
        self.preprocessor  = pipeline.named_steps["preprocessor"]
        self.model         = pipeline.named_steps["model"]
        self.explainer     = None
        self.feature_names = None

    # --------------------------------------------------
    # Transform input — handles NamedColumnTransformer
    # which already returns a DataFrame (no double-wrap)
    # --------------------------------------------------
    def _preprocess(self, X):
        result = self.preprocessor.transform(X)
        if isinstance(result, pd.DataFrame):
            return result.reset_index(drop=True)
        return pd.DataFrame(result, columns=self.feature_names)

    # --------------------------------------------------
    # Extract 2D SHAP values for fraud class (index 1)
    # Uses .shap_values() — stable across SHAP versions
    # --------------------------------------------------
    def _shap_values(self, X_processed):
        raw = self.explainer.shap_values(X_processed)
        if isinstance(raw, list):
            return np.array(raw[1])           # [class_0, class_1] → take fraud
        if raw.ndim == 3:
            return raw[:, :, 1]               # (samples, features, classes) → class 1
        return raw                            # already 2D

    # --------------------------------------------------
    # Build TreeExplainer once on first call
    # --------------------------------------------------
    def _build_explainer(self, X_sample):
        logger.info("Building SHAP explainer...")
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP explainer ready.")

    # --------------------------------------------------
    # Global explanation — beeswarm + bar chart
    # --------------------------------------------------
    def global_explanation(self, X):
        logger.info("Generating global SHAP explanation...")

        X_sample = X.sample(min(500, len(X)), random_state=42)
        if self.explainer is None:
            self._build_explainer(X_sample)

        X_proc    = self._preprocess(X_sample)
        shap_vals = self._shap_values(X_proc)

        logger.info(f"SHAP values shape: {shap_vals.shape} | Features: {len(self.feature_names)}")

        os.makedirs("reports/shap", exist_ok=True)

        for plot_type, filename, title in [
            ("dot", "global_summary.png",        "SHAP Feature Impact — Fraud Class"),
            ("bar", "feature_importance_bar.png", "SHAP Feature Importance — Fraud Class"),
        ]:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_vals,
                X_proc.values,
                feature_names=list(self.feature_names),
                plot_type=plot_type,
                show=False,
                max_display=20
            )
            plt.title(title, fontsize=13)
            plt.tight_layout()
            plt.savefig(f"reports/shap/{filename}", bbox_inches="tight", dpi=150)
            plt.close()

        logger.info("Global SHAP plots saved to reports/shap/")

    # --------------------------------------------------
    # Local explanation — waterfall for one transaction
    # --------------------------------------------------
    def local_explanation(self, X, index=0):
        logger.info(f"Generating local explanation for index {index}")

        if self.explainer is None:
            self._build_explainer(X.sample(min(500, len(X)), random_state=42))

        X_proc    = self._preprocess(X.iloc[[index]])
        shap_vals = self._shap_values(X_proc)

        expected  = self.explainer.expected_value
        base_val  = float(expected[1] if isinstance(expected, (list, np.ndarray)) else expected)

        os.makedirs("reports/shap", exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=base_val,
                data=X_proc.iloc[0].values,
                feature_names=list(self.feature_names)
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"reports/shap/waterfall_{index}.png", bbox_inches="tight", dpi=150)
        plt.close()

        logger.info(f"Waterfall plot saved for index {index}")

    # --------------------------------------------------
    # JSON explanation — for API response
    # --------------------------------------------------
    def local_explanation_json(self, X, index=0, top_k=5):
        if self.explainer is None:
            self._build_explainer(X.sample(min(500, len(X)), random_state=42))

        X_proc    = self._preprocess(X.iloc[[index]])
        shap_vals = self._shap_values(X_proc)

        expected   = self.explainer.expected_value
        base_value = float(expected[1] if isinstance(expected, (list, np.ndarray)) else expected)

        impacts = sorted(
            [{"feature": f, "impact": float(v)}
             for f, v in zip(self.feature_names, shap_vals[0])],
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        return {
            "base_value":           base_value,
            "top_positive_features": [f for f in impacts if f["impact"] > 0][:top_k],
            "top_negative_features": [f for f in impacts if f["impact"] < 0][:top_k],
        }
