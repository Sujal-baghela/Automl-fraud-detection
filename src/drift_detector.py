import json
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift using KS Test and PSI (Population Stability Index).
    PSI < 0.1 = No drift | 0.1-0.2 = Moderate | >0.2 = High drift
    """

    def __init__(self, psi_threshold=0.2, ks_alpha=0.05):
        self.psi_threshold = psi_threshold
        self.ks_alpha      = ks_alpha
        self.reference     = {}

    def fit(self, X: pd.DataFrame):
        for col in X.columns:
            self.reference[col] = {
                "mean":   float(X[col].mean()),
                "sample": X[col].dropna().sample(min(500, len(X)), random_state=42).tolist()
            }
        logger.info(f"Drift detector fitted on {len(self.reference)} features.")
        return self

    def _psi(self, ref, cur, bins=10):
        bp      = np.linspace(np.percentile(ref, 1), np.percentile(ref, 99), bins + 1)
        r, _    = np.histogram(ref, bins=bp)
        c, _    = np.histogram(cur, bins=bp)
        r_pct   = (r + 1e-6) / len(ref)
        c_pct   = (c + 1e-6) / len(cur)
        return float(abs(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct))))

    def detect(self, X: pd.DataFrame) -> dict:
        drifted = []
        details = {}

        for col in self.reference:
            if col not in X.columns:
                continue
            ref = np.array(self.reference[col]["sample"])
            cur = X[col].dropna().values
            if len(cur) < 10:
                continue

            psi            = round(self._psi(ref, cur), 4)
            _, ks_pvalue   = stats.ks_2samp(ref, cur)
            is_drifted     = psi >= self.psi_threshold or ks_pvalue < self.ks_alpha

            if is_drifted:
                drifted.append(col)

            details[col] = {
                "psi":      psi,
                "pvalue":   round(float(ks_pvalue), 4),
                "drifted":  is_drifted,
                "status":   "🔴 HIGH" if psi >= 0.2 else ("🟡 MODERATE" if psi >= 0.1 else "🟢 OK"),
            }

        ratio  = len(drifted) / len(details) if details else 0
        status = (
            "🔴 HIGH DRIFT — Consider retraining" if ratio >= 0.3 else
            "🟡 MODERATE DRIFT — Monitor closely" if ratio >= 0.1 else
            "🟢 NO DRIFT — Model is stable"
        )

        report = {
            "status":           status,
            "drift_ratio":      round(ratio, 3),
            "drifted_count":    len(drifted),
            "total_features":   len(details),
            "drifted_features": drifted[:10],
            "details":          details,
        }

        logger.info(f"Drift detection: {status} | {len(drifted)}/{len(details)} features drifted")
        return report

    def print_report(self, report: dict):
        print("\n" + "="*55)
        print("          DATA DRIFT REPORT")
        print("="*55)
        print(f"  Status  : {report['status']}")
        print(f"  Drifted : {report['drifted_count']}/{report['total_features']} features")
        print("\n  Top features by PSI:")
        top = sorted(report["details"].items(), key=lambda x: x[1]["psi"], reverse=True)[:5]
        for feat, d in top:
            print(f"    {d['status']}  {feat}: PSI={d['psi']}")
        print("="*55 + "\n")

    def save(self, path="models/drift_reference.json"):
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"reference": self.reference,
                       "psi_threshold": self.psi_threshold,
                       "ks_alpha": self.ks_alpha}, f)
        logger.info(f"Drift reference saved to {path}")

    def load(self, path="models/drift_reference.json"):
        with open(path) as f:
            d = json.load(f)
        self.reference     = d["reference"]
        self.psi_threshold = d["psi_threshold"]
        self.ks_alpha      = d["ks_alpha"]
        return self