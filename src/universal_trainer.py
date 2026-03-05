"""
universal_trainer.py  ·  AutoML-X v6.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles ANY dataset size, ANY data quality, end-to-end.

Fixes in v6.1:
  - load_csv_chunked: added `encoding` parameter (fixes "unexpected keyword argument 'encoding'")
  - DataQualityReport.assess: try/except guards on string column detection
  - DatasetProfiler.profile: guards for empty num/cat lists, 0-row edge cases
  - ColumnTypeDetector.detect: pd.to_datetime now uses errors="coerce" (not "raise")
"""

import os
import gc
import re
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
    TIER_SMALL:   "Small   (1K - 50K rows)",
    TIER_MEDIUM:  "Medium  (50K - 200K rows)",
    TIER_LARGE:   "Large   (200K - 500K rows)",
    TIER_XLARGE:  "XLarge  (500K - 2M rows)",
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
# FIX v6.1: added `encoding` parameter — was missing, caused crash on page 01
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_chunked(
    file_obj,
    max_rows: int = None,
    chunk_size: int = 100_000,
    target_col: str = None,
    encoding: str = "utf-8",          # FIX: was missing entirely
    progress_callback=None
) -> pd.DataFrame:
    chunks     = []
    total_rows = 0
    chunk_num  = 0

    # FIX: pass encoding to pd.read_csv
    reader = pd.read_csv(
        file_obj,
        chunksize=chunk_size,
        low_memory=False,
        encoding=encoding,             # FIX: now correctly forwarded
        encoding_errors="replace",     # FIX: graceful fallback for bad bytes
    )

    for chunk in reader:
        chunk_num  += 1
        total_rows += len(chunk)

        if progress_callback:
            progress_callback(
                f"Reading chunk {chunk_num} · {total_rows:,} rows so far..."
            )

        chunks.append(chunk)

        if max_rows and total_rows >= max_rows:
            if progress_callback:
                progress_callback(f"Row limit {max_rows:,} reached — stopping.")
            break

        if get_available_ram_gb() < 0.3:
            logger.warning("RAM critically low — stopping at %d rows", total_rows)
            if progress_callback:
                progress_callback(
                    f"RAM critically low — stopped at {total_rows:,} rows"
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

    if max_rows and len(df) > max_rows:
        if progress_callback:
            progress_callback(
                f"Sampling {max_rows:,} from {len(df):,} rows (stratified)..."
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
            f"Loaded {len(df):,} rows x {df.shape[1]} cols  "
            f"[{TIER_LABELS[get_tier(len(df))]}]"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SMART DATA CLEANER
# ─────────────────────────────────────────────────────────────────────────────

class SmartDataCleaner:
    """
    Automatically cleans messy raw data before training.
    """

    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        missing_drop_threshold: float = 0.95,
        min_unique_ratio: float = 0.0,
    ):
        self.outlier_method          = outlier_method
        self.outlier_threshold       = outlier_threshold
        self.missing_drop_threshold  = missing_drop_threshold
        self.min_unique_ratio        = min_unique_ratio
        self.report: dict            = {}

    @staticmethod
    def _is_string_col(series: pd.Series) -> bool:
        """True for both legacy object dtype and pandas 2.2+ StringDtype."""
        try:
            return series.dtype == object or pd.api.types.is_string_dtype(series)
        except Exception:
            return series.dtype == object

    def clean(self, df: pd.DataFrame, target_col: str = None) -> tuple:
        report = {
            "rows_before": len(df),
            "cols_before": df.shape[1],
            "changes": [],
        }
        df = df.copy()

        # 1. Strip column name whitespace
        df.columns = [str(c).strip() for c in df.columns]

        # 2. Remove duplicate rows
        n_before = len(df)
        df = df.drop_duplicates()
        n_dup = n_before - len(df)
        if n_dup > 0:
            report["changes"].append(f"Removed {n_dup:,} duplicate rows")
        report["duplicates_removed"] = n_dup

        # 3. Drop columns with >threshold% missing (excluding target)
        high_missing = []
        for col in df.columns:
            if col == target_col:
                continue
            miss_rate = df[col].isnull().mean()
            if miss_rate > self.missing_drop_threshold:
                high_missing.append(col)
        if high_missing:
            df = df.drop(columns=high_missing)
            report["changes"].append(
                f"Dropped {len(high_missing)} cols with >{self.missing_drop_threshold*100:.0f}% missing: "
                f"{', '.join(high_missing[:5])}{'...' if len(high_missing) > 5 else ''}"
            )
        report["high_missing_cols_dropped"] = high_missing

        # 4. Drop constant columns
        constant_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                constant_cols.append(col)
        if constant_cols:
            df = df.drop(columns=constant_cols)
            report["changes"].append(
                f"Dropped {len(constant_cols)} constant cols: "
                f"{', '.join(constant_cols[:5])}{'...' if len(constant_cols) > 5 else ''}"
            )
        report["constant_cols_dropped"] = constant_cols

        # 5. Boolean coercion (yes/no, true/false, y/n -> 1/0)
        bool_map = {
            "yes": 1, "no": 0, "true": 1, "false": 0,
            "y": 1, "n": 0, "t": 1, "f": 0,
            "1": 1, "0": 0, "1.0": 1, "0.0": 0,
        }
        bool_coerced = []
        for col in df.columns:
            if col == target_col:
                continue
            try:
                if self._is_string_col(df[col]):
                    sample = df[col].dropna().astype(str).str.lower().unique()
                    if set(sample).issubset(set(bool_map.keys())) and len(sample) <= 4:
                        df[col] = df[col].astype(str).str.lower().map(bool_map)
                        bool_coerced.append(col)
            except Exception:
                pass
        if bool_coerced:
            report["changes"].append(
                f"Coerced {len(bool_coerced)} boolean cols (yes/no->1/0): "
                f"{', '.join(bool_coerced[:5])}"
            )
        report["bool_cols_coerced"] = bool_coerced

        # 6. Numeric-string coercion (e.g. "1,234" -> 1234, "$5.00" -> 5.0)
        numeric_coerced = []
        for col in df.columns:
            if col == target_col:
                continue
            try:
                if self._is_string_col(df[col]):
                    sample = df[col].dropna().head(200).astype(str)
                    cleaned = sample.str.replace(r"[$,€£¥%\s]", "", regex=True)
                    converted = pd.to_numeric(cleaned, errors="coerce")
                    success_rate = converted.notna().mean()
                    if success_rate >= 0.85:
                        df[col] = (
                            df[col].astype(str)
                            .str.replace(r"[$,€£¥%\s]", "", regex=True)
                        )
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        numeric_coerced.append(col)
            except Exception:
                pass
        if numeric_coerced:
            report["changes"].append(
                f"Coerced {len(numeric_coerced)} numeric-string cols: "
                f"{', '.join(numeric_coerced[:5])}"
            )
        report["numeric_strings_coerced"] = numeric_coerced

        # 7. Strip whitespace from string columns
        ws_stripped = []
        for col in df.columns:
            if col == target_col:
                continue
            try:
                if self._is_string_col(df[col]):
                    before = df[col].astype(str).str.strip()
                    if not (before == df[col].astype(str)).all():
                        df[col] = df[col].astype(str).str.strip()
                        ws_stripped.append(col)
            except Exception:
                pass
        if ws_stripped:
            report["changes"].append(
                f"Stripped whitespace from {len(ws_stripped)} string cols"
            )
        report["whitespace_stripped"] = ws_stripped

        # 8. Outlier capping (numeric cols only, excluding target)
        outlier_capped = []
        if self.outlier_method != "none":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in num_cols:
                num_cols.remove(target_col)
            for col in num_cols:
                s = df[col].dropna()
                if len(s) < 10:
                    continue
                try:
                    if self.outlier_method == "iqr":
                        q1, q3 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q3 - q1
                        if iqr == 0:
                            continue
                        lo = q1 - self.outlier_threshold * iqr
                        hi = q3 + self.outlier_threshold * iqr
                    else:  # zscore
                        mu, sigma = s.mean(), s.std()
                        if sigma == 0:
                            continue
                        lo = mu - self.outlier_threshold * sigma
                        hi = mu + self.outlier_threshold * sigma
                    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
                    if n_out > 0:
                        df[col] = df[col].clip(lower=lo, upper=hi)
                        outlier_capped.append(f"{col}({n_out})")
                except Exception:
                    pass
            if outlier_capped:
                report["changes"].append(
                    f"Capped outliers in {len(outlier_capped)} numeric cols "
                    f"(method={self.outlier_method})"
                )
        report["outliers_capped"] = outlier_capped

        report["rows_after"] = len(df)
        report["cols_after"] = df.shape[1]
        self.report = report
        return df, report


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY REPORT
# FIX v6.1: try/except guards around string detection to prevent Analyze crash
# ─────────────────────────────────────────────────────────────────────────────

class DataQualityReport:
    """Generates a pre-cleaning quality assessment."""

    def assess(self, df: pd.DataFrame, target_col: str = None) -> dict:
        issues = []
        warnings_list = []
        info_list = []

        n_rows, n_cols = df.shape

        # Guard: empty dataframe
        if n_rows == 0:
            return {
                "issues": [{"type": "empty", "severity": "error",
                            "message": "Dataset is empty — no rows to analyze."}],
                "n_errors": 1, "n_warnings": 0, "n_info": 0,
                "has_blockers": True, "overall_quality": "poor",
            }

        # Duplicate rows
        try:
            n_dup = df.duplicated().sum()
            if n_dup > 0:
                issues.append({
                    "type": "duplicates",
                    "severity": "warning",
                    "message": f"{n_dup:,} duplicate rows ({n_dup/n_rows*100:.1f}%) — will be removed",
                    "count": int(n_dup),
                })
        except Exception:
            pass

        # Missing values per column
        try:
            miss_counts = df.isnull().sum()
            high_miss   = miss_counts[miss_counts / n_rows > 0.3]
            for col, cnt in high_miss.items():
                if col == target_col:
                    continue
                pct = cnt / n_rows * 100
                sev = "error" if pct > 70 else "warning"
                issues.append({
                    "type": "high_missing",
                    "severity": sev,
                    "message": f"'{col}': {pct:.1f}% missing values",
                    "count": int(cnt),
                    "col": col,
                })
        except Exception:
            pass

        # Constant columns
        for col in df.columns:
            if col == target_col:
                continue
            try:
                if df[col].nunique(dropna=True) <= 1:
                    issues.append({
                        "type": "constant_col",
                        "severity": "warning",
                        "message": f"'{col}' has only 1 unique value — will be dropped",
                        "col": col,
                    })
            except Exception:
                pass

        # FIX v6.1: wrapped entire string column detection in try/except
        # Previously this could crash on mixed-type or extension-dtype columns
        str_cols = []
        for col in df.columns:
            try:
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    str_cols.append(col)
            except Exception:
                pass

        for col in str_cols:
            if col == target_col:
                continue
            try:
                sample = df[col].dropna().head(100).astype(str)
                cleaned = sample.str.replace(r"[$,€£¥%\s]", "", regex=True)
                conv = pd.to_numeric(cleaned, errors="coerce")
                if conv.notna().mean() >= 0.85:
                    info_list.append({
                        "type": "numeric_string",
                        "severity": "info",
                        "message": f"'{col}' looks numeric but stored as string — will be coerced",
                        "col": col,
                    })
            except Exception:
                pass

        # Target column issues
        if target_col and target_col in df.columns:
            try:
                n_target_miss = df[target_col].isnull().sum()
                if n_target_miss > 0:
                    issues.append({
                        "type": "target_missing",
                        "severity": "error",
                        "message": f"Target '{target_col}' has {n_target_miss:,} missing values — rows will be dropped",
                        "count": int(n_target_miss),
                    })
                n_classes = df[target_col].nunique()
                if n_classes > 2:
                    issues.append({
                        "type": "multiclass",
                        "severity": "error",
                        "message": f"Target '{target_col}' has {n_classes} classes — only binary supported",
                        "count": n_classes,
                    })
                elif n_classes == 1:
                    issues.append({
                        "type": "single_class",
                        "severity": "error",
                        "message": f"Target '{target_col}' has only 1 class — cannot train",
                    })
            except Exception:
                pass

        # High cardinality categoricals
        for col in str_cols:
            if col == target_col:
                continue
            try:
                n_unique = df[col].nunique()
                if n_unique > 50 and n_unique / n_rows < 0.5:
                    warnings_list.append({
                        "type": "high_cardinality",
                        "severity": "warning",
                        "message": f"'{col}' has {n_unique} unique values — may slow training",
                        "col": col,
                    })
            except Exception:
                pass

        all_issues = issues + warnings_list + info_list
        n_errors   = sum(1 for x in all_issues if x["severity"] == "error")
        n_warnings = sum(1 for x in all_issues if x["severity"] == "warning")
        n_info     = sum(1 for x in all_issues if x["severity"] == "info")

        return {
            "issues":        all_issues,
            "n_errors":      n_errors,
            "n_warnings":    n_warnings,
            "n_info":        n_info,
            "has_blockers":  n_errors > 0,
            "overall_quality": (
                "poor"   if n_errors > 0 else
                "fair"   if n_warnings > 3 else
                "good"   if n_warnings > 0 else
                "excellent"
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SMART COLUMN TYPE DETECTOR
# FIX v6.1: pd.to_datetime now uses errors="coerce" instead of errors="raise"
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

            is_id_name = any(p in col_lower for p in self.ID_PATTERNS)

            # Only flag INT columns as ID by cardinality — never float columns
            is_high_card_int = (
                dtype in [np.int64, np.int32] and
                n_unique > min(0.9 * n_rows, 10_000)
            )
            is_high_card_str = (
                (dtype == object or pd.api.types.is_string_dtype(series)) and
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

            if dtype == object or pd.api.types.is_string_dtype(series):
                # FIX v6.1: use errors="coerce" instead of errors="raise"
                # "raise" would crash on ANY column with even one unparseable value
                try:
                    parsed = pd.to_datetime(series.head(50), errors="coerce")
                    # Only classify as date if >80% parsed successfully
                    if parsed.notna().mean() > 0.8:
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
                         f"gap={gap:+.3f} -> {complexity}"),
            }
        except Exception as e:
            return {"complexity": "unknown", "recommended": "All models",
                    "note": f"Auto-detect failed: {e}",
                    "lr_score": None, "lgb_score": None, "gap": None}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROFILER
# FIX v6.1: guards for empty num/cat lists, 0-row datasets, corr() edge cases
# ─────────────────────────────────────────────────────────────────────────────

class DatasetProfiler:
    def profile(self, df: pd.DataFrame, target_col: str) -> dict:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        # FIX: guard against empty dataframe
        if len(df) == 0:
            raise ValueError("Dataset is empty — cannot profile 0 rows.")

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

        # FIX: safe sample size
        sample_size = min(50_000, max(1, len(df)))
        sample_df   = df.sample(sample_size, random_state=42)
        col_stats   = {}

        for col in num_cols:
            try:
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
            except Exception as e:
                col_stats[col] = {"type": "numeric", "note": f"Stats failed: {e}",
                                  "missing": 0, "missing_pct": 0}

        for col in cat_cols:
            try:
                s  = sample_df[col].dropna()
                vc = s.value_counts()
                col_stats[col] = {
                    "type":        "categorical",
                    "missing":     int(df[col].isnull().sum()),
                    "missing_pct": round(df[col].isnull().mean() * 100, 1),
                    "n_unique":    int(df[col].nunique()),
                    "top_values":  vc.head(5).to_dict(),
                }
            except Exception as e:
                col_stats[col] = {"type": "categorical", "note": f"Stats failed: {e}",
                                  "missing": 0, "missing_pct": 0}

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

        # FIX v6.1: guard corr() — requires at least 2 numeric cols with valid data
        high_corr_pairs = []
        if len(num_cols) >= 2:
            try:
                valid_num = [c for c in num_cols if c in sample_df.columns
                             and sample_df[c].notna().sum() > 1]
                if len(valid_num) >= 2:
                    corr  = sample_df[valid_num].corr().abs()
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

        dqr     = DataQualityReport()
        quality = dqr.assess(df, target_col)

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
            "high_cardinality_cols": id_cols,
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
            "quality_report":    quality,
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

def get_models_for_tier(tier: int, is_imbalanced: bool, complexity: str) -> dict:
    bal  = is_imbalanced

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

    return {
        "LightGBM": LGBMClassifier(
            n_estimators=500, is_unbalance=bal, random_state=42,
            n_jobs=-1, verbose=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.6, colsample_bytree=0.7,
            max_bin=127),
    }


def get_cv_config(tier: int) -> dict:
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
    """Train ANY binary classification dataset, any size, automatically."""

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
        self.cleaning_report  = {}

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

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        positive_label=None,
        test_size: float = 0.2,
        sample_if_large: bool = True,
        clean_data: bool = True,
        outlier_method: str = "iqr",
        progress_callback=None
    ) -> dict:

        def _p(step, total, msg):
            logger.info("[%d/%d] %s", step, total, msg)
            if progress_callback:
                progress_callback(step, total, msg)

        T = 9

        _p(1, T, "Smart data cleaning...")
        if target_col in df.columns:
            n_before = len(df)
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                _p(1, T, f"Dropped {n_dropped:,} rows with missing target")

        if clean_data:
            cleaner = SmartDataCleaner(outlier_method=outlier_method)
            df, self.cleaning_report = cleaner.clean(df, target_col)
            for ch in self.cleaning_report.get("changes", []):
                _p(1, T, f"  {ch}")
        else:
            self.cleaning_report = {"changes": [], "skipped": True}

        _p(2, T, "Profiling dataset & detecting column types...")
        profiler     = DatasetProfiler()
        self.profile = profiler.profile(df, target_col)
        self.tier    = self.profile["tier"]
        tier_lbl     = TIER_LABELS[self.tier]
        self.target_col     = target_col
        self.positive_label = positive_label

        _p(2, T, f"Tier: {tier_lbl}  ·  Strategy: {TIER_STRATEGY[self.tier]}")

        if self.profile["n_classes"] != 2:
            raise ValueError(
                f"Binary classification only. "
                f"Found {self.profile['n_classes']} classes: "
                f"{list(self.profile['class_counts'].keys())}"
            )

        _p(3, T, "Dropping IDs / dates / text columns...")
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

        _p(3, T, f"{len(num_cols)} numeric + {len(cat_cols)} categorical "
                 f"| dropped {len(drop_cols)} cols")

        _p(4, T, "Building preprocessing pipeline...")
        preprocessor = build_preprocessor(num_cols, cat_cols)

        _p(5, T, "Detecting problem complexity...")
        detector          = ComplexityDetector()
        self.complexity_info = detector.detect(X, y, preprocessor)
        complexity         = self.complexity_info["complexity"]
        _p(5, T, f"Complexity -> {complexity}  |  {self.complexity_info['note']}")

        _p(6, T, f"Splitting data [{tier_lbl}]...")
        if self.tier == TIER_MASSIVE:
            X_s, y_s = self._stratified_sample(X, y, max_rows=500_000)
            X_train, X_val, y_train, y_val = train_test_split(
                X_s, y_s, test_size=test_size, random_state=42, stratify=y_s)
            del X_s, y_s, X, y; gc.collect()
        elif self.tier == TIER_XLARGE:
            X_s, y_s = self._stratified_sample(X, y, max_rows=500_000)
            X_train, X_val, y_train, y_val = train_test_split(
                X_s, y_s, test_size=test_size, random_state=42, stratify=y_s)
            del X_s, y_s, X, y; gc.collect()
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)
            if self.tier >= TIER_LARGE:
                del X, y; gc.collect()

        _p(6, T, f"Train {len(X_train):,} · Val {len(X_val):,}")

        _p(7, T, "Training models (tier-aware)...")
        models  = get_models_for_tier(self.tier, self.profile["is_imbalanced"], complexity)
        cv_cfg  = get_cv_config(self.tier)
        use_cv  = cv_cfg["use_cv"]
        n_folds = cv_cfg["folds"]
        cv_samp = cv_cfg["cv_sample"]

        best_score, best_pipeline, best_name = -np.inf, None, None

        for i, (name, model) in enumerate(models.items(), 1):
            _p(7, T, f"  [{i}/{len(models)}] Training {name}...")
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            if use_cv:
                Xcv, ycv = X_train, y_train
                if cv_samp and len(X_train) > cv_samp:
                    Xcv, ycv = self._stratified_sample(X_train, y_train, cv_samp)
                try:
                    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                    scores = cross_val_score(pipeline, Xcv, ycv,
                                            cv=skf, scoring="roc_auc", n_jobs=-1)
                    score  = float(np.mean(scores))
                except Exception as e:
                    logger.warning("%s CV failed: %s", name, e); score = 0.0
                self.all_scores[name] = round(score, 5)
                if score > best_score:
                    best_score, best_pipeline, best_name = score, pipeline, name
            else:
                try:
                    pipeline.fit(X_train, y_train)
                    score = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:,1])
                except Exception as e:
                    logger.warning("%s fit failed: %s", name, e); score = 0.0
                self.all_scores[name] = round(score, 5)
                if score > best_score:
                    best_score, best_pipeline, best_name = score, pipeline, name

        if use_cv:
            _p(7, T, f"Final fit: {best_name} on {len(X_train):,} rows...")
            best_pipeline.fit(X_train, y_train)

        self.best_pipeline   = best_pipeline
        self.best_model_name = best_name
        self.best_score      = best_score

        _p(8, T, "Optimising decision threshold (max F1)...")
        if len(X_val) > 200_000:
            Xvs, yvs = self._stratified_sample(X_val, y_val, 200_000)
        else:
            Xvs, yvs = X_val, y_val

        y_proba = best_pipeline.predict_proba(Xvs)[:,1]
        prec, rec, thresholds = precision_recall_curve(yvs, y_proba)
        pr, re = prec[:-1], rec[:-1]
        f1s    = (2 * pr * re) / (pr + re + 1e-8)
        self.threshold = float(thresholds[np.argmax(f1s)]) if len(thresholds) > 0 else 0.5
        _p(8, T, f"Optimal threshold: {self.threshold:.5f}")

        _p(9, T, "Evaluating...")
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
            "cleaning_report": self.cleaning_report,
        }

        self.save()
        return self.metrics

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

    def save(self, path: str = None):
        path = path or self.model_save_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
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
            "cleaning_report": self.cleaning_report,
        }, path)
        logger.info("Model saved -> %s", path)

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
        self.cleaning_report  = pkg.get("cleaning_report", {})
        return self