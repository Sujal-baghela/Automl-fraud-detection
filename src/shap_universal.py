"""
src/shap_universal.py  --  AutoML-X v7.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Universal SHAP wrapper for UniversalTrainer pipelines.
Extends IntelligentSHAP with auto-fallback for non-tree models (LR).

Model support:
  - LightGBM, XGBoost, RandomForest  -> TreeExplainer (fast, exact)
  - LogisticRegression                -> LinearExplainer (fast, exact)
  - Any other                         -> shap.Explainer  (auto, slower)
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Tree-based model class names that work with TreeExplainer
_TREE_MODELS = {
    "LGBMClassifier", "XGBClassifier", "RandomForestClassifier",
    "GradientBoostingClassifier", "ExtraTreesClassifier",
    "DecisionTreeClassifier", "HistGradientBoostingClassifier",
}
_LINEAR_MODELS = {
    "LogisticRegression", "LinearSVC", "SGDClassifier", "RidgeClassifier",
}


def _get_model_class(model) -> str:
    return type(model).__name__


class UniversalSHAP:
    """
    SHAP explainer for UniversalTrainer pipelines.
    Auto-selects TreeExplainer / LinearExplainer / Explainer
    based on the model type inside the pipeline.
    """

    def __init__(self, pipeline):
        self.pipeline      = pipeline
        self.preprocessor  = pipeline.named_steps["preprocessor"]
        self.model         = pipeline.named_steps["model"]
        self.explainer     = None
        self.feature_names = None
        self._model_class  = _get_model_class(self.model)
        self._is_tree      = self._model_class in _TREE_MODELS
        self._is_linear    = self._model_class in _LINEAR_MODELS

    # ── preprocessing ────────────────────────────────────────────────────────
    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        result = self.preprocessor.transform(X)
        if isinstance(result, pd.DataFrame):
            return result.reset_index(drop=True)
        return pd.DataFrame(result, columns=self.feature_names)

    # ── extract feature names from preprocessor ───────────────────────────────
    def _get_feature_names(self) -> list:
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            # Fallback: number the features
            try:
                n = self.preprocessor.transform(
                    pd.DataFrame([[0] * len(self.preprocessor.feature_names_in_)],
                                 columns=self.preprocessor.feature_names_in_)
                ).shape[1]
                return [f"feature_{i}" for i in range(n)]
            except Exception:
                return [f"feature_{i}" for i in range(100)]

    # ── build explainer once ─────────────────────────────────────────────────
    def _build_explainer(self, X_sample: pd.DataFrame):
        self.feature_names = self._get_feature_names()
        X_proc = self._preprocess(X_sample)

        if self._is_tree:
            logger.info("SHAP: Using TreeExplainer for %s", self._model_class)
            self.explainer = shap.TreeExplainer(self.model)
        elif self._is_linear:
            logger.info("SHAP: Using LinearExplainer for %s", self._model_class)
            self.explainer = shap.LinearExplainer(
                self.model, X_proc,
                feature_perturbation="interventional"
            )
        else:
            logger.info("SHAP: Using auto Explainer for %s", self._model_class)
            self.explainer = shap.Explainer(self.model, X_proc)

    # ── extract 2D SHAP values for positive class ─────────────────────────────
    def _shap_values_2d(self, X_proc: pd.DataFrame) -> np.ndarray:
        raw = self.explainer.shap_values(X_proc)
        if isinstance(raw, list):
            return np.array(raw[1])       # [class_0, class_1] -> fraud
        if hasattr(raw, "ndim") and raw.ndim == 3:
            return raw[:, :, 1]           # (samples, features, classes)
        return raw                        # already 2D

    # ── public: top-N features for one row ───────────────────────────────────
    def explain_single(self, X: pd.DataFrame, index: int = 0,
                       top_k: int = 10) -> dict:
        """
        Returns dict with:
          model_class, feature_names, shap_values (all),
          top_features (sorted by |shap|),
          base_value
        """
        sample = X.sample(min(200, len(X)), random_state=42)
        if self.explainer is None:
            self._build_explainer(sample)

        X_proc    = self._preprocess(X.iloc[[index]])
        shap_vals = self._shap_values_2d(X_proc)

        # base value
        ev = self.explainer.expected_value
        base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

        sv = shap_vals[0]
        pairs = sorted(
            [{"feature": f, "shap": float(v), "abs": abs(float(v))}
             for f, v in zip(self.feature_names, sv)],
            key=lambda x: x["abs"], reverse=True
        )

        return {
            "model_class":  self._model_class,
            "base_value":   base_value,
            "top_features": pairs[:top_k],
            "all_shap":     sv.tolist(),
            "feature_names": list(self.feature_names),
        }

    # ── public: bar chart figure (for st.pyplot) ──────────────────────────────
    def plot_bar(self, explain_result: dict, title: str = "Feature Importance (SHAP)",
                 max_display: int = 10, bg: str = "#0a0a0f") -> plt.Figure:
        top = explain_result["top_features"][:max_display]
        names  = [t["feature"].replace("num__", "").replace("cat__", "") for t in top]
        values = [t["shap"] for t in top]
        colors = ["#f87171" if v > 0 else "#34d399" for v in values]

        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.42)))
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                       edgecolor="none", height=0.6)
        ax.axvline(0, color="#3a3a5c", linewidth=0.8)
        ax.set_xlabel("SHAP value  (red = increases risk, green = decreases risk)",
                      fontsize=7, color="#6b6b8a")
        ax.set_title(title, fontsize=9, color="#8888aa")

        # value labels
        for bar, val in zip(bars[::-1], values):
            ax.text(val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=7, color="#a5b4fc", fontfamily="monospace")

        fig.patch.set_facecolor(bg)
        ax.set_facecolor("#0e0e1a")
        ax.tick_params(colors="#6b6b8a", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a1a2e")
        ax.xaxis.label.set_color("#4a4a6a")
        fig.tight_layout(pad=0.8)
        return fig

    # ── public: global importance from training data sample ──────────────────
    def global_importance(self, X: pd.DataFrame,
                          n_sample: int = 300) -> dict:
        sample = X.sample(min(n_sample, len(X)), random_state=42)
        if self.explainer is None:
            self._build_explainer(sample)

        X_proc    = self._preprocess(sample)
        shap_vals = self._shap_values_2d(X_proc)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        pairs = sorted(
            [{"feature": f, "importance": float(v)}
             for f, v in zip(self.feature_names, mean_abs)],
            key=lambda x: x["importance"], reverse=True
        )
        return {"features": pairs, "n_samples": len(sample)}