"""
monitor.py — ModelMonitor for AutoML-X fraud detection system.

Integrates with:
  - src/drift_detector.py   → DriftDetector.detect(X: pd.DataFrame) → dict
      returns: {drift_ratio, drifted_count, total_features, drifted_features, status, details}
  - src/inference_engine.py → FraudInferenceEngine.get_metadata() → dict
      returns: {threshold, objective, cv_score, features, ...}

Storage: SQLite  (models/monitor.db)
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH                  = Path("models/monitor.db")
DEFAULT_FRAUD_RATE_THRESHOLD     = 0.30   # >30% fraud rate in window → CRITICAL
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.60   # avg probability < 60%     → WARNING
DEFAULT_MIN_PREDICTIONS          = 10     # need at least N preds before alerting
DEFAULT_DRIFT_RATIO_THRESHOLD    = 0.10   # ≥10% features drifted     → WARNING
                                          # matches DriftDetector's "MODERATE" boundary


@dataclass
class AlertRule:
    name: str
    severity: str          # INFO | WARNING | CRITICAL
    message: str
    fired: bool = False
    fired_at: Optional[str] = None
    extra: dict = field(default_factory=dict)


class ModelMonitor:
    """
    Real-time monitor for fraud prediction traffic.

    Usage in app/api.py
    -------------------
        monitor = ModelMonitor()

        @app.post("/predict")
        def predict(transaction: Transaction):
            ...
            monitor.log_prediction(input_dict, prediction, probability)
            ...

    Parameters
    ----------
    db_path                   : SQLite file path (auto-created)
    fraud_rate_threshold      : triggers CRITICAL if fraud_rate exceeds this
    low_confidence_threshold  : triggers WARNING if avg_probability below this
    min_predictions_for_alerts: skip alerting until window has this many rows
    drift_ratio_threshold     : triggers WARNING when DriftDetector.detect()
                                returns drift_ratio >= this value
    drift_detector            : DriftDetector instance (optional)
    inference_engine          : FraudInferenceEngine instance (optional, for health)
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        fraud_rate_threshold: float = DEFAULT_FRAUD_RATE_THRESHOLD,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        min_predictions_for_alerts: int = DEFAULT_MIN_PREDICTIONS,
        drift_ratio_threshold: float = DEFAULT_DRIFT_RATIO_THRESHOLD,
        drift_detector=None,
        inference_engine=None,
    ):
        self.db_path = Path(db_path)
        self.fraud_rate_threshold = fraud_rate_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.min_predictions_for_alerts = min_predictions_for_alerts
        self.drift_ratio_threshold = drift_ratio_threshold
        self.drift_detector = drift_detector
        self.inference_engine = inference_engine
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT    NOT NULL,
                    input_json  TEXT,
                    prediction  INTEGER NOT NULL,
                    probability REAL    NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_log (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT    NOT NULL,
                    name     TEXT    NOT NULL,
                    severity TEXT    NOT NULL,
                    message  TEXT    NOT NULL,
                    extra    TEXT
                )
                """
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        input_data: Any,
        prediction: int,
        probability: float,
    ) -> int:
        """
        Persist one prediction record.

        In api.py, call after computing probability/prediction:

            probability = float(pipeline.predict_proba(input_df)[0][1])
            prediction  = int(probability >= threshold)
            monitor.log_prediction(transaction.model_dump(), prediction, probability)

        Parameters
        ----------
        input_data  : transaction dict (Transaction.model_dump() from api.py)
        prediction  : 0 or 1
        probability : raw float from predict_proba (not %-scaled)

        Returns the new row id.
        """
        ts = datetime.now().isoformat()
        try:
            input_json = json.dumps(input_data)
        except (TypeError, ValueError):
            input_json = str(input_data)

        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO predictions (ts, input_json, prediction, probability) VALUES (?,?,?,?)",
                (ts, input_json, int(prediction), float(probability)),
            )
            row_id = cursor.lastrowid

        logger.debug("Logged prediction id=%s  pred=%s  prob=%.4f", row_id, prediction, probability)
        return row_id

    def get_summary(self, last_n: int = 100) -> dict:
        """
        Rolling statistics over the last *last_n* predictions.

        Returns
        -------
        {total_count, fraud_count, fraud_rate,
         avg_probability, min_probability, max_probability,
         window_start, window_end}
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT prediction, probability, ts FROM predictions ORDER BY id DESC LIMIT ?",
                (last_n,),
            ).fetchall()

        if not rows:
            return {
                "total_count":     0,
                "fraud_count":     0,
                "fraud_rate":      0.0,
                "avg_probability": 0.0,
                "min_probability": 0.0,
                "max_probability": 0.0,
                "window_start":    None,
                "window_end":      None,
            }

        preds      = [r["prediction"] for r in rows]
        probs      = [r["probability"] for r in rows]
        timestamps = [r["ts"] for r in rows]
        total      = len(preds)

        return {
            "total_count":     total,
            "fraud_count":     sum(preds),
            "fraud_rate":      sum(preds) / total,
            "avg_probability": sum(probs) / total,
            "min_probability": min(probs),
            "max_probability": max(probs),
            "window_start":    min(timestamps),
            "window_end":      max(timestamps),
        }

    def get_total_predictions(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM predictions").fetchone()
        return row["cnt"]

    def check_health(self) -> dict:
        """
        System health snapshot.

        Returns
        -------
        {status, db_ok, total_predictions, last_prediction_ts,
         uptime_seconds, drift_available, inference_available,
         model_version}   ← model_version from FraudInferenceEngine.get_metadata()
        """
        t0     = time.monotonic()
        status = "healthy"
        db_ok  = False
        last_ts = None
        total   = 0

        try:
            total = self.get_total_predictions()
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT ts FROM predictions ORDER BY id DESC LIMIT 1"
                ).fetchone()
            last_ts = row["ts"] if row else None
            db_ok = True
        except Exception as exc:
            logger.error("Health-check DB error: %s", exc)
            status = "unhealthy"

        elapsed = time.monotonic() - t0
        if db_ok and elapsed > 2.0:
            status = "degraded"

        health = {
            "status":              status,
            "db_ok":               db_ok,
            "total_predictions":   total,
            "last_prediction_ts":  last_ts,
            "uptime_seconds":      elapsed,
            "drift_available":     self.drift_detector is not None,
            "inference_available": self.inference_engine is not None,
            "model_version":       "unknown",
        }

        # FraudInferenceEngine.get_metadata() → {"threshold", "objective",
        #                                         "cv_score", "features", ...}
        if self.inference_engine is not None:
            try:
                meta = self.inference_engine.get_metadata()
                health["model_version"] = meta.get("model_version", "unknown")
            except Exception:
                pass

        return health

    def trigger_alerts(
        self,
        last_n: int = 100,
        drift_df: Optional[pd.DataFrame] = None,
    ) -> list:
        """
        Evaluate alert rules and return fired AlertRule objects.

        Parameters
        ----------
        last_n    : rolling window for prediction stats
        drift_df  : DataFrame of recent transactions to pass to
                    DriftDetector.detect(X: pd.DataFrame).
                    If None, drift check is skipped.

        Alert rules
        -----------
        1. high_fraud_rate  CRITICAL
           → fraud_rate > fraud_rate_threshold

        2. low_confidence   WARNING
           → avg_probability < low_confidence_threshold
             (model is uncertain; may signal distribution shift)

        3. drift_detected   WARNING
           → DriftDetector.detect(drift_df)["drift_ratio"]
             >= drift_ratio_threshold
             Threshold aligns with DriftDetector's own 0.1 MODERATE boundary.

        All fired alerts are persisted to alert_log table.
        """
        summary = self.get_summary(last_n=last_n)
        fired: list[AlertRule] = []

        if summary["total_count"] < self.min_predictions_for_alerts:
            logger.debug(
                "trigger_alerts: %d predictions, need %d — skipping",
                summary["total_count"], self.min_predictions_for_alerts,
            )
            return fired

        # ── Rule 1: High fraud rate ────────────────────────────────────
        if summary["fraud_rate"] > self.fraud_rate_threshold:
            fired.append(AlertRule(
                name="high_fraud_rate",
                severity="CRITICAL",
                message=(
                    f"Fraud rate {summary['fraud_rate']:.1%} exceeds threshold "
                    f"{self.fraud_rate_threshold:.1%} "
                    f"({summary['fraud_count']}/{summary['total_count']} transactions)."
                ),
                fired=True,
                fired_at=datetime.now().isoformat(),
                extra={
                    "fraud_rate":  summary["fraud_rate"],
                    "fraud_count": summary["fraud_count"],
                    "total_count": summary["total_count"],
                    "threshold":   self.fraud_rate_threshold,
                    "window":      last_n,
                },
            ))

        # ── Rule 2: Low model confidence ──────────────────────────────
        if summary["avg_probability"] < self.low_confidence_threshold:
            fired.append(AlertRule(
                name="low_confidence",
                severity="WARNING",
                message=(
                    f"Average prediction probability {summary['avg_probability']:.3f} "
                    f"is below confidence threshold {self.low_confidence_threshold}. "
                    f"Model may be uncertain."
                ),
                fired=True,
                fired_at=datetime.now().isoformat(),
                extra={
                    "avg_probability": summary["avg_probability"],
                    "threshold":       self.low_confidence_threshold,
                },
            ))

        # ── Rule 3: Data drift via DriftDetector.detect() ─────────────
        # DriftDetector.detect(X: pd.DataFrame) returns:
        #   {
        #     "status":           "🔴 HIGH DRIFT..." | "🟡 MODERATE..." | "🟢 NO DRIFT...",
        #     "drift_ratio":      float,   ← fraction of features that drifted
        #     "drifted_count":    int,
        #     "total_features":   int,
        #     "drifted_features": [str, ...],
        #     "details":          {feature: {psi, pvalue, drifted, status}, ...}
        #   }
        if self.drift_detector is not None and drift_df is not None:
            try:
                report = self.drift_detector.detect(drift_df)
                drift_ratio = float(report.get("drift_ratio", 0.0))
                if drift_ratio >= self.drift_ratio_threshold:
                    fired.append(AlertRule(
                        name="drift_detected",
                        severity="WARNING",
                        message=(
                            f"Data drift detected: {report.get('drifted_count', '?')}/"
                            f"{report.get('total_features', '?')} features drifted "
                            f"(ratio {drift_ratio:.1%}). "
                            f"Status: {report.get('status', 'unknown')}"
                        ),
                        fired=True,
                        fired_at=datetime.now().isoformat(),
                        extra={
                            "drift_ratio":      drift_ratio,
                            "drifted_count":    report.get("drifted_count"),
                            "total_features":   report.get("total_features"),
                            "drifted_features": report.get("drifted_features", []),
                            "status":           report.get("status"),
                            "threshold":        self.drift_ratio_threshold,
                        },
                    ))
            except Exception as exc:
                logger.warning("Drift check failed: %s", exc)

        for alert in fired:
            self._log_alert(alert)

        return fired

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_alert(self, alert: AlertRule) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO alert_log (ts, name, severity, message, extra) VALUES (?,?,?,?,?)",
                (
                    alert.fired_at or datetime.now().isoformat(),
                    alert.name,
                    alert.severity,
                    alert.message,
                    json.dumps(alert.extra),
                ),
            )

    def get_alert_history(self, last_n: int = 50) -> list[dict]:
        """Return the last *last_n* fired alerts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alert_log ORDER BY id DESC LIMIT ?", (last_n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_predictions(self) -> int:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM predictions")
            return cursor.rowcount

    def clear_alerts(self) -> int:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM alert_log")
            return cursor.rowcount