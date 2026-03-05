"""
universal_trainer.py  ·  AutoML-X v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles ANY dataset size end-to-end:

  Tier 0  Tiny    :     0 – 1 000 rows  → All 4 models · 5-fold CV · no sampling
  Tier 1  Small   :  1K  – 50K  rows   → All 4 models · 5-fold CV
  Tier 2  Medium  :  50K – 200K rows   → All 4 models · 3-fold CV
  Tier 3  Large   :  200K– 500K rows   → LR+LGB+XGB  · 2-fold CV
  Tier 4  XLarge  :  500K–2M   rows   → LGB+LR only  · 2-fold CV on 200K sample
  Tier 5  Massive :  2M+       rows   → LGB only     · no CV · direct 80/20 split
                                         chunked predict for inference
"""

import os
import gc
import logging
import warnings
import numpy as np
import pandas as pd
import psutil

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, confusion_matrix, precision_recall_curve
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TIER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TIER_TINY    = 0
TIER_SMALL   = 1
TIER_MEDIUM  = 2
TIER_LARGE   = 3
TIER_XLARGE  = 4
TIER_MASSIVE = 5

TIER_LABELS = {
    TIER_TINY:    "Tiny    (<1K rows)",
    TIER_SMALL:   "Small   (1K – 50K rows)",
    TIER_MEDIUM:  "Medium  (50K – 200K rows)",
    TIER_LARGE:   "Large   (200K – 500K rows)",
    TIER_XLARGE:  "XLarge  (500K – 2M rows)",
    TIER_MASSIVE: "Massive (2M+ rows)",
}

TIER_STRATEGY = {
    TIER_TINY:    "All 4 models · 5-fold CV · full data",
    TIER_SMALL:   "All 4 models · 5-fold CV · full data",
    TIER_MEDIUM:  "All 4 models · 3-fold CV · full data",
    TIER_LARGE:   "LR + LGB + XGB · 2-fold CV · full data",
    TIER_XLARGE:  "LGB + LR · 2-fold CV on 200K sample · full fit",
    TIER_MASSIVE: "LGB only · no CV · 80/20 split on 500K sample · chunked predict",
}

def get_tier(n_rows: int) -> int:
    if n_rows < 1_000:      return TIER_TINY
    if n_rows < 50_000:     return TIER_SMALL
    if n_rows < 200_000:    return TIER_MEDIUM
    if n_rows < 500_000:    return TIER_LARGE
    if n_rows < 2_000_000:  return TIER_XLARGE
    return TIER_MASSIVE


# ─────────────────────────────────────────────────────────────────────────────
# RAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9

def get_dataframe_ram_gb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e9

def check_ram_safety(df: pd.DataFrame) -> dict:
    available = get_available_ram_gb()
    df_size   = get_dataframe_ram_gb(df)
    needed    = df_size * 4
    safe      = needed < available * 0.8
    tier      = get_tier(len(df))
    return {
        "available_gb":        round(available, 2),
        "dataframe_gb":        round(df_size, 3),
        "estimated_needed_gb": round(needed, 2),
        "is_safe":             safe,
        "tier":                TIER_LABELS[tier],
        "strategy":            TIER_STRATEGY[tier],
        "warning": None if safe else (
            f"Training may need ~{needed:.1f} GB RAM but only "
            f"{available:.1f} GB available. Tier-aware sampling will apply."
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED CSV LOADER  ─  any file size
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_chunked(
    file_obj,
    max_rows: int = None,          # None = load everything (up to RAM limit)
    chunk_size: int = 100_000,
    target_col: str = None,
    progress_callback=None
) -> pd.DataFrame:
    """
    Stream any CSV in chunks. Never crashes on large files.
    - Stops early if RAM drops below 300 MB.
    - Stratified-samples if file exceeds max_rows.
    - Returns clean DataFrame ready for training.
    """
    chunks     = []
    total_rows = 0
    chunk_num  = 0

    reader = pd.read_csv(file_obj, chunksize=chunk_size, low_memory=False)

    for chunk in reader:
        chunk_num  += 1
        total_rows += len(chunk)

        if progress_callback:
            progress_callback(
                f"Reading chunk {chunk_num} · {total_rows:,} rows so far…"
            )

        chunks.append(chunk)

        if get_available_ram_gb() < 0.3:
            logger.warning("RAM critically low — stopping at %d rows", total_rows)
            if progress_callback:
                progress_callback(
                    f"⚠️ RAM critically low — stopped at {total_rows:,} rows"
                )
            break

    if not chunks:
        raise ValueError("No data could be read from the file.")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    if progress_callback:
        tier_lbl = TIER_LABELS[get_tier(len(df))]
        progress_callback(
            f"Concatenated {len(df):,} rows · {df.shape[1]} cols  [{tier_lbl}]"
        )

    # Stratified downsample only when caller supplies a hard limit
    if max_rows and len(df) > max_rows:
        if progress_callback:
            progress_callback(
                f"Sampling {max_rows:,} from {len(df):,} rows (stratified)…"
            )
        if target_col and target_col in df.columns:
            try:
                frac = max_rows / len(df)
                df = (
                    df.groupby(target_col, group_keys=False)
                      .apply(lambda x: x.sample(frac=frac, random_state=42))
                      .reset_index(drop=True)
                )
                if len(df) > max_rows:
                    df = df.sample(max_rows, random_state=42).reset_index(drop=True)
            except Exception:
                df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        else:
            df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    if progress_callback:
        progress_callback(
            f"✅ Loaded {len(df):,} rows × {df.shape[1]} cols  "
            f"[{TIER_LABELS[get_tier(len(df))]}]"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SMART COLUMN TYPE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ColumnTypeDetector:
    ID_PATTERNS = [
        "id", "_id", "id_", "uuid", "guid", "key", "index",
        "rownum", "row_num", "record", "seq", "sequence",
        "customerid", "userid", "accountid", "transactionid",
        "customer_id", "user_id", "account_id", "transaction_id",
        "phone", "mobile", "ssn", "passport",
    ]
    DATE_PATTERNS = [
        "date", "time", "timestamp", "created", "updated",
        "datetime", "dt", "year", "month", "day", "dob", "birth",
    ]
    TEXT_THRESHOLD  = 50
    HIGH_CARD_RATIO = 0.5

    def detect(self, df: pd.DataFrame, target_col: str = None) -> dict:
        id_cols, date_cols, text_cols = [], [], []
        numeric_cols, cat_cols        = [], []
        n_rows = len(df)

        for col in df.columns:
            if col == target_col:
                continue

            col_lower = col.lower().replace(" ", "_").replace("-", "_")
            series    = df[col].dropna()

            if len(series) == 0:
                id_cols.append(col)
                continue

            n_unique = series.nunique()
            dtype    = series.dtype

            is_id_name       = any(p in col_lower for p in self.ID_PATTERNS)
            is_high_card_int = (
                dtype in [np.int64, np.int32, np.float64] and
                n_unique > min(0.9 * n_rows, 10_000)
            )
            is_high_card_str = (
                dtype == object and
                n_unique / max(n_rows, 1) > self.HIGH_CARD_RATIO and
                n_unique > 100
            )

            if is_id_name or is_high_card_int or is_high_card_str:
                if any(p in col_lower for p in self.DATE_PATTERNS):
                    date_cols.append(col)
                else:
                    id_cols.append(col)
                continue

            if any(p in col_lower for p in self.DATE_PATTERNS):
                date_cols.append(col)
                continue

            if dtype == object:
                try:
                    pd.to_datetime(series.head(50), infer_datetime_format=True,
                                   errors="raise")
                    date_cols.append(col)
                    continue
                except Exception:
                    pass
                avg_len = series.astype(str).str.len().mean()
                if avg_len > self.TEXT_THRESHOLD and n_unique > 50:
                    text_cols.append(col)
                else:
                    cat_cols.append(col)
                continue

            if dtype in [np.int64, np.int32, np.float64, np.float32]:
                if n_unique <= 10 and dtype in [np.int64, np.int32]:
                    cat_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            if dtype == bool or set(series.unique()).issubset({0, 1, True, False}):
                cat_cols.append(col)
                continue

            numeric_cols.append(col)

        return {
            "id_cols":      id_cols,
            "date_cols":    date_cols,
            "text_cols":    text_cols,
            "numeric_cols": numeric_cols,
            "cat_cols":     cat_cols,
            "feature_cols": numeric_cols + cat_cols,
            "dropped_cols": id_cols + text_cols + date_cols,
        }


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityDetector:
    def detect(self, X: pd.DataFrame, y: np.ndarray,
               preprocessor, n_sample: int = 5_000) -> dict:
        try:
            n   = min(n_sample, len(X))
            idx = np.random.RandomState(42).choice(len(X), n, replace=False)
            Xs, ys = X.iloc[idx], y[idx]
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xs, ys, test_size=0.3, random_state=42, stratify=ys
            )
            lr_pipe = Pipeline([("pre", preprocessor),
                                ("clf", LogisticRegression(max_iter=500, n_jobs=-1))])
            lgb_pipe = Pipeline([("pre", preprocessor),
                                 ("clf", LGBMClassifier(n_estimators=50,
                                          random_state=42, verbose=-1, n_jobs=-1))])
            lr_pipe.fit(X_tr, y_tr)
            lgb_pipe.fit(X_tr, y_tr)
            lr_auc  = roc_auc_score(y_va, lr_pipe.predict_proba(X_va)[:,1])
            lgb_auc = roc_auc_score(y_va, lgb_pipe.predict_proba(X_va)[:,1])
            gap     = lgb_auc - lr_auc
            if   gap >  0.05: complexity, rec = "nonlinear", "LightGBM / RandomForest"
            elif gap < -0.03: complexity, rec = "linear",    "LogisticRegression"
            else:             complexity, rec = "mixed",     "All models"
            return {
                "complexity":  complexity,
                "recommended": rec,
                "lr_score":    round(lr_auc, 4),
                "lgb_score":   round(lgb_auc, 4),
                "gap":         round(gap, 4),
                "note": (f"LGB({lgb_auc:.3f}) vs LR({lr_auc:.3f}) "
                         f"gap={gap:+.3f} → {complexity}"),
            }
        except Exception as e:
            return {"complexity": "unknown", "recommended": "All models",
                    "note": f"Auto-detect failed: {e}",
                    "lr_score": None, "lgb_score": None, "gap": None}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROFILER
# ─────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    def profile(self, df: pd.DataFrame, target_col: str) -> dict:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        X  = df.drop(columns=[target_col])
        y  = df[target_col]

        detector  = ColumnTypeDetector()
        col_types = detector.detect(df, target_col)

        num_cols  = col_types["numeric_cols"]
        cat_cols  = col_types["cat_cols"]
        id_cols   = col_types["id_cols"]
        date_cols = col_types["date_cols"]
        text_cols = col_types["text_cols"]

        class_counts   = y.value_counts()
        n_classes      = len(class_counts)
        minority_ratio = class_counts.min() / max(len(y), 1)
        missing_pct    = round(X.isnull().sum().sum() / max(X.size, 1) * 100, 2)

        # Per-column stats on a sample (fast on huge datasets)
        sample_df = df.sample(min(50_000, len(df)), random_state=42)
        col_stats = {}

        for col in num_cols:
            s = sample_df[col].dropna()
            col_stats[col] = {
                "type":        "numeric",
                "missing":     int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().mean() * 100, 1),
                "mean":        round(float(s.mean()),    4) if len(s) else None,
                "std":         round(float(s.std()),     4) if len(s) else None,
                "min":         round(float(s.min()),     4) if len(s) else None,
                "max":         round(float(s.max()),     4) if len(s) else None,
                "median":      round(float(s.median()),  4) if len(s) else None,
                "skew":        round(float(s.skew()),    3) if len(s) > 2 else None,
            }
        for col in cat_cols:
            s  = sample_df[col].dropna()
            vc = s.value_counts()
            col_stats[col] = {
                "type":        "categorical",
                "missing":     int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().mean() * 100, 1),
                "n_unique":    int(df[col].nunique()),
                "top_values":  vc.head(5).to_dict(),
            }
        for col in id_cols:
            col_stats[col] = {"type": "id_dropped",
                              "note": "Auto-detected as ID — will be dropped",
                              "n_unique": int(df[col].nunique())}
        for col in date_cols:
            col_stats[col] = {"type": "date",
                              "note": "Detected as date — will be dropped"}
        for col in text_cols:
            col_stats[col] = {"type": "text",
                              "note": "Detected as free text — will be dropped"}

        # High-correlation check
        high_corr_pairs = []
        if len(num_cols) >= 2:
            try:
                corr  = sample_df[num_cols].corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                pairs = [
                    (c, r, round(upper.loc[r, c], 3))
                    for c in upper.columns for r in upper.index
                    if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.95
                ]
                high_corr_pairs = sorted(pairs, key=lambda x: -x[2])[:5]
            except Exception:
                pass

        n_rows = len(df)
        tier   = get_tier(n_rows)

        return {
            "n_rows":            n_rows,
            "n_cols":            len(X.columns),
            "tier":              tier,
            "tier_label":        TIER_LABELS[tier],
            "tier_strategy":     TIER_STRATEGY[tier],
            "n_numeric":         len(num_cols),
            "n_categorical":     len(cat_cols),
            "n_id_dropped":      len(id_cols),
            "n_date_dropped":    len(date_cols),
            "n_text_dropped":    len(text_cols),
            "numeric_cols":      num_cols,
            "categorical_cols":  cat_cols,
            "id_cols":           id_cols,
            "date_cols":         date_cols,
            "text_cols":         text_cols,
            "high_cardinality_cols": id_cols,   # backward compat
            "target_col":        target_col,
            "n_classes":         n_classes,
            "class_counts":      class_counts.to_dict(),
            "minority_ratio":    round(float(minority_ratio), 6),
            "is_imbalanced":     minority_ratio < 0.2,
            "missing_pct":       missing_pct,
            "has_missing":       missing_pct > 0,
            "high_corr_pairs":   high_corr_pairs,
            "col_stats":         col_stats,
            "size_gb":           round(get_dataframe_ram_gb(df), 3),
            "needs_sampling":    n_rows > 500_000,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False, max_categories=50)),
        ]), cat_cols))
    if not transformers:
        raise ValueError("No usable columns after type detection.")
    return ColumnTransformer(transformers=transformers)


# ─────────────────────────────────────────────────────────────────────────────
# TIER-AWARE MODEL + CV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_models_for_tier(tier: int, is_imbalanced: bool,
                        complexity: str) -> dict:
    bal  = is_imbalanced
    nonl = complexity in ("nonlinear", "mixed", "unknown")

    # ── Tier 0 & 1: Tiny / Small ─────────────────────────────────────────────
    if tier <= TIER_SMALL:
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=3000, class_weight="balanced" if bal else None, n_jobs=-1),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced" if bal else None,
                n_jobs=-1, random_state=42),
            "LightGBM": LGBMClassifier(
                n_estimators=300, is_unbalance=bal, random_state=42,
                n_jobs=-1, verbose=-1),
            "XGBoost": XGBClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
                eval_metric="auc", verbosity=0),
        }

    # ── Tier 2: Medium ────────────────────────────────────────────────────────
    if tier == TIER_MEDIUM:
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=2000, class_weight="balanced" if bal else None, n_jobs=-1),
            "RandomForest": RandomForestClassifier(
                n_estimators=150, class_weight="balanced" if bal else None,
                n_jobs=-1, random_state=42, max_depth=15),
            "LightGBM": LGBMClassifier(
                n_estimators=200, is_unbalance=bal, random_state=42,
                n_jobs=-1, verbose=-1),
            "XGBoost": XGBClassifier(
                n_estimators=150, random_state=42, n_jobs=-1,
                eval_metric="auc", verbosity=0),
        }

    # ── Tier 3: Large ─────────────────────────────────────────────────────────
    if tier == TIER_LARGE:
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced" if bal else None, n_jobs=-1),
            "LightGBM": LGBMClassifier(
                n_estimators=200, is_unbalance=bal, random_state=42,
                n_jobs=-1, verbose=-1, num_leaves=63),
            "XGBoost": XGBClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                eval_metric="auc", verbosity=0, tree_method="hist"),
        }

    # ── Tier 4: XLarge (500K–2M) ─────────────────────────────────────────────
    if tier == TIER_XLARGE:
        return {
            "LightGBM": LGBMClassifier(
                n_estimators=300, is_unbalance=bal, random_state=42,
                n_jobs=-1, verbose=-1, num_leaves=127,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
            "LogisticRegression": LogisticRegression(
                max_iter=500, class_weight="balanced" if bal else None,
                solver="saga", n_jobs=-1),
        }

    # ── Tier 5: Massive (2M+) ─────────────────────────────────────────────────
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=500, is_unbalance=bal, random_state=42,
            n_jobs=-1, verbose=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.6, colsample_bytree=0.7,
            max_bin=127),
    }


def get_cv_config(tier: int) -> dict:
    """
    Returns:
      use_cv     – whether to run cross_val_score
      folds      – number of CV folds
      cv_sample  – if set, subsample X to this size before CV
    """
    configs = {
        TIER_TINY:    {"use_cv": True,  "folds": 5, "cv_sample": None},
        TIER_SMALL:   {"use_cv": True,  "folds": 5, "cv_sample": None},
        TIER_MEDIUM:  {"use_cv": True,  "folds": 3, "cv_sample": None},
        TIER_LARGE:   {"use_cv": True,  "folds": 2, "cv_sample": None},
        TIER_XLARGE:  {"use_cv": True,  "folds": 2, "cv_sample": 200_000},
        TIER_MASSIVE: {"use_cv": False, "folds": 0, "cv_sample": None},
    }
    return configs[tier]


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class UniversalTrainer:
    """
    Train ANY binary classification dataset, any size, automatically.

    Tier 0  <1K       → full grid · 5-fold CV · no sampling
    Tier 1  1K–50K    → full grid · 5-fold CV
    Tier 2  50K–200K  → full grid · 3-fold CV
    Tier 3  200K–500K → 3-model  · 2-fold CV
    Tier 4  500K–2M   → 2-model  · 2-fold CV on 200K subsample
    Tier 5  2M+       → LGB only · no CV · 500K train sample · chunked predict
    """

    def __init__(self, model_save_path: str = "models/universal_model.pkl"):
        self.model_save_path  = model_save_path
        self.best_pipeline    = None
        self.best_model_name  = None
        self.best_score       = None
        self.threshold        = 0.5
        self.feature_names    = None
        self.target_col       = None
        self.positive_label   = None
        self.label_encoder    = None
        self.profile          = None
        self.col_types        = None
        self.complexity_info  = None
        self.tier             = None
        self.all_scores       = {}
        self.metrics          = {}

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _encode_target(self, y: pd.Series, positive_label) -> np.ndarray:
        unique_vals = y.unique()
        if set(unique_vals).issubset({0, 1}):
            return y.values.astype(int)
        if positive_label is not None:
            try:
                pos = type(y.iloc[0])(positive_label)
            except Exception:
                pos = positive_label
            return (y == pos).astype(int).values
        le = LabelEncoder()
        encoded = le.fit_transform(y)
        self.label_encoder = le
        return encoded

    def _stratified_sample(self, X: pd.DataFrame, y: np.ndarray,
                           max_rows: int) -> tuple:
        if len(X) <= max_rows:
            return X, y
        rng     = np.random.RandomState(42)
        classes = np.unique(y)
        idxs    = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            n_take  = max(1, int(max_rows * len(cls_idx) / len(y)))
            idxs.extend(rng.choice(cls_idx, min(n_take, len(cls_idx)), replace=False))
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        idxs = idxs[:max_rows]
        return X.iloc[idxs].reset_index(drop=True), y[idxs]

    # ─── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        positive_label=None,
        test_size: float = 0.2,
        sample_if_large: bool = True,   # kept for API compat
        progress_callback=None
    ) -> dict:

        def _p(step, total, msg):
            logger.info("[%d/%d] %s", step, total, msg)
            if progress_callback:
                progress_callback(step, total, msg)

        T = 8   # total steps

        # ── 1  Profile ────────────────────────────────────────────────────────
        _p(1, T, "Profiling dataset & detecting column types…")
        profiler     = DatasetProfiler()
        self.profile = profiler.profile(df, target_col)
        self.tier    = self.profile["tier"]
        tier_lbl     = TIER_LABELS[self.tier]
        self.target_col     = target_col
        self.positive_label = positive_label

        _p(1, T, f"Tier detected: {tier_lbl}  ·  strategy: {TIER_STRATEGY[self.tier]}")

        if self.profile["n_classes"] != 2:
            raise ValueError(
                f"Binary classification only. "
                f"Found {self.profile['n_classes']} classes."
            )

        # ── 2  Prepare features ───────────────────────────────────────────────
        _p(2, T, "Dropping IDs / dates / text columns…")
        drop_cols = (
            self.profile["id_cols"] +
            self.profile["date_cols"] +
            self.profile["text_cols"]
        )
        X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
        y = self._encode_target(df[target_col], positive_label)

        num_cols = [c for c in self.profile["numeric_cols"]    if c in X.columns]
        cat_cols = [c for c in self.profile["categorical_cols"] if c in X.columns]
        self.feature_names = list(X.columns)

        if not self.feature_names:
            raise ValueError("No usable features after type detection.")

        _p(2, T, f"{len(num_cols)} numeric + {len(cat_cols)} categorical features"
                 f" | dropped {len(drop_cols)} cols")

        # ── 3  Preprocessor ───────────────────────────────────────────────────
        _p(3, T, "Building preprocessing pipeline…")
        preprocessor = build_preprocessor(num_cols, cat_cols)

        # ── 4  Complexity detection (always fast — 5K sample) ─────────────────
        _p(4, T, "Detecting problem complexity (linear vs non-linear)…")
        detector          = ComplexityDetector()
        self.complexity_info = detector.detect(X, y, preprocessor)
        complexity         = self.complexity_info["complexity"]
        _p(4, T, f"Complexity → {complexity}  |  {self.complexity_info['note']}")

        # ── 5  Tier-aware split ───────────────────────────────────────────────
        _p(5, T, f"Splitting data [{tier_lbl}]…")

        if self.tier == TIER_MASSIVE:
            # Sample 500K for training; evaluation on a 100K held-out set
            X_s, y_s = self._stratified_sample(X, y, max_rows=500_000)
            X_train, X_val, y_train, y_val = train_test_split(
                X_s, y_s, test_size=test_size, random_state=42, stratify=y_s
            )
            del X_s, y_s, X, y
            gc.collect()
            _p(5, T, f"Massive: training on {len(X_train):,} rows "
                     f"(stratified 500K sample from {self.profile['n_rows']:,})")

        elif self.tier == TIER_XLARGE:
            X_s, y_s = self._stratified_sample(X, y, max_rows=500_000)
            X_train, X_val, y_train, y_val = train_test_split(
                X_s, y_s, test_size=test_size, random_state=42, stratify=y_s
            )
            del X_s, y_s, X, y
            gc.collect()
            _p(5, T, f"XLarge: training on {len(X_train):,} rows "
                     f"(stratified 500K sample)")

        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            if self.tier >= TIER_LARGE:
                del X, y
                gc.collect()
            _p(5, T, f"Train {len(X_train):,} · Val {len(X_val):,}")

        # ── 6  Model training (tier-aware) ────────────────────────────────────
        _p(6, T, "Training models…")
        models  = get_models_for_tier(self.tier, self.profile["is_imbalanced"], complexity)
        cv_cfg  = get_cv_config(self.tier)
        use_cv  = cv_cfg["use_cv"]
        n_folds = cv_cfg["folds"]
        cv_samp = cv_cfg["cv_sample"]

        best_score, best_pipeline, best_name = -np.inf, None, None

        for i, (name, model) in enumerate(models.items(), 1):
            _p(6, T, f"  [{i}/{len(models)}] {name}…")

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model",        model),
            ])

            if use_cv:
                Xcv = X_train
                ycv = y_train
                if cv_samp and len(X_train) > cv_samp:
                    Xcv, ycv = self._stratified_sample(X_train, y_train, cv_samp)
                try:
                    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                             random_state=42)
                    scores = cross_val_score(
                        pipeline, Xcv, ycv,
                        cv=skf, scoring="roc_auc", n_jobs=-1
                    )
                    score  = float(np.mean(scores))
                    logger.info("%s CV(%d) AUC: %.5f ±%.5f",
                                name, n_folds, score, float(np.std(scores)))
                except Exception as e:
                    logger.warning("%s CV failed: %s", name, e)
                    score = 0.0

                self.all_scores[name] = round(score, 5)
                if score > best_score:
                    best_score, best_pipeline, best_name = score, pipeline, name

            else:
                # Massive tier: no CV, fit directly
                try:
                    pipeline.fit(X_train, y_train)
                    score = roc_auc_score(
                        y_val, pipeline.predict_proba(X_val)[:,1]
                    )
                    logger.info("%s (no-CV) val AUC: %.5f", name, score)
                except Exception as e:
                    logger.warning("%s fit failed: %s", name, e)
                    score = 0.0
                self.all_scores[name] = round(score, 5)
                if score > best_score:
                    best_score, best_pipeline, best_name = score, pipeline, name

        logger.info("Best: %s (score=%.5f)", best_name, best_score)

        # Final fit on full train set (CV tiers only)
        if use_cv:
            _p(6, T, f"Final fit: {best_name} on {len(X_train):,} rows…")
            best_pipeline.fit(X_train, y_train)

        self.best_pipeline   = best_pipeline
        self.best_model_name = best_name
        self.best_score      = best_score

        # ── 7  Threshold optimisation ─────────────────────────────────────────
        _p(7, T, "Optimising decision threshold (max F1)…")

        # Use at most 200K for threshold search (fast)
        if len(X_val) > 200_000:
            Xvs, yvs = self._stratified_sample(X_val, y_val, 200_000)
        else:
            Xvs, yvs = X_val, y_val

        y_proba = best_pipeline.predict_proba(Xvs)[:,1]
        prec, rec, thresholds = precision_recall_curve(yvs, y_proba)
        pr, re = prec[:-1], rec[:-1]
        f1s    = (2 * pr * re) / (pr + re + 1e-8)
        self.threshold = float(thresholds[np.argmax(f1s)]) if len(thresholds) > 0 else 0.5
        _p(7, T, f"Optimal threshold: {self.threshold:.5f}")

        # ── 8  Evaluate ───────────────────────────────────────────────────────
        _p(8, T, "Evaluating…")
        y_pred = (y_proba >= self.threshold).astype(int)
        cm     = confusion_matrix(yvs, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        self.metrics = {
            "best_model":      best_name,
            "cv_roc_auc":      round(best_score, 5),
            "test_roc_auc":    round(roc_auc_score(yvs, y_proba), 5),
            "f1_score":        round(f1_score(yvs, y_pred), 5),
            "recall":          round(recall_score(yvs, y_pred), 5),
            "precision":       round(precision_score(yvs, y_pred), 5),
            "threshold":       round(self.threshold, 5),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "all_cv_scores":   self.all_scores,
            "complexity":      self.complexity_info,
            "dropped_cols":    drop_cols,
            "n_features_used": len(self.feature_names),
            "tier":            self.tier,
            "tier_label":      tier_lbl,
            "tier_strategy":   TIER_STRATEGY[self.tier],
            "n_train":         len(X_train),
            "n_val":           len(Xvs),
            "n_rows_total":    self.profile["n_rows"],
        }

        logger.info(
            "Done · ROC-AUC=%.5f · F1=%.5f · Recall=%.5f",
            self.metrics["test_roc_auc"],
            self.metrics["f1_score"],
            self.metrics["recall"],
        )

        self.save()
        return self.metrics

    # ── Predict (chunked for massive inference) ───────────────────────────────

    def predict_proba(self, X: pd.DataFrame,
                      chunk_size: int = 100_000) -> np.ndarray:
        if self.best_pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")
        X = X.reindex(columns=self.feature_names, fill_value=0)
        if len(X) <= chunk_size:
            return self.best_pipeline.predict_proba(X)[:,1]
        results = []
        for start in range(0, len(X), chunk_size):
            results.append(
                self.best_pipeline.predict_proba(X.iloc[start:start+chunk_size])[:,1]
            )
        return np.concatenate(results)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str = None):
        path = path or self.model_save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pipeline":        self.best_pipeline,
            "threshold":       self.threshold,
            "feature_names":   self.feature_names,
            "target_col":      self.target_col,
            "positive_label":  self.positive_label,
            "best_model_name": self.best_model_name,
            "metrics":         self.metrics,
            "profile":         self.profile,
            "col_types":       self.col_types,
            "complexity_info": self.complexity_info,
            "all_scores":      self.all_scores,
            "label_encoder":   self.label_encoder,
            "tier":            self.tier,
        }, path)
        logger.info("Model saved → %s", path)

    def load(self, path: str = None):
        path = path or self.model_save_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        pkg = joblib.load(path)
        self.best_pipeline    = pkg["pipeline"]
        self.threshold        = pkg["threshold"]
        self.feature_names    = pkg["feature_names"]
        self.target_col       = pkg["target_col"]
        self.positive_label   = pkg["positive_label"]
        self.best_model_name  = pkg["best_model_name"]
        self.metrics          = pkg["metrics"]
        self.profile          = pkg["profile"]
        self.col_types        = pkg.get("col_types")
        self.complexity_info  = pkg.get("complexity_info")
        self.all_scores       = pkg["all_scores"]
        self.label_encoder    = pkg.get("label_encoder")
        self.tier             = pkg.get("tier")
        return self