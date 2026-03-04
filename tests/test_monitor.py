"""
test_monitor.py — Full test suite for ModelMonitor and AlertManager.

Mocks match the real interfaces:
  DriftDetector.detect(X: pd.DataFrame) → {drift_ratio, drifted_count,
                                            total_features, drifted_features,
                                            status, details}
  FraudInferenceEngine.get_metadata()   → {threshold, objective, cv_score,
                                            features, model_version}

Run:
    pytest tests/test_monitor.py -v --tb=short
    pytest tests/test_monitor.py --cov=src/monitor --cov=src/alerting --cov-report=term-missing
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import pandas as pd
import pytest

from src.monitor import (
    AlertRule,
    ModelMonitor,
    DEFAULT_FRAUD_RATE_THRESHOLD,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_PREDICTIONS,
    DEFAULT_DRIFT_RATIO_THRESHOLD,
)
from src.alerting import AlertManager, SEVERITY_LEVELS, SLACK_WEBHOOK_ENV


# ======================================================================
# Helpers — realistic DriftDetector.detect() return values
# ======================================================================

def make_drift_report(drift_ratio: float = 0.0, drifted_count: int = 0, total: int = 10):
    """Mimics DriftDetector.detect() return structure exactly."""
    if drift_ratio >= 0.3:
        status = "🔴 HIGH DRIFT — Consider retraining"
    elif drift_ratio >= 0.1:
        status = "🟡 MODERATE DRIFT — Monitor closely"
    else:
        status = "🟢 NO DRIFT — Model is stable"
    return {
        "status":           status,
        "drift_ratio":      round(drift_ratio, 3),
        "drifted_count":    drifted_count,
        "total_features":   total,
        "drifted_features": [f"V{i}" for i in range(drifted_count)],
        "details":          {},
    }


def make_drift_detector(drift_ratio: float = 0.0, drifted_count: int = 0):
    """Returns a MagicMock that behaves like DriftDetector."""
    mock = MagicMock()
    mock.detect.return_value = make_drift_report(drift_ratio, drifted_count)
    return mock


def make_inference_engine(model_version: str = "v1.0.0"):
    """Returns a MagicMock that behaves like FraudInferenceEngine."""
    mock = MagicMock()
    mock.get_metadata.return_value = {
        "threshold":     0.47,
        "objective":     "cost",
        "cv_score":      0.98,
        "features":      ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        "model_version": model_version,
    }
    return mock


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def monitor(tmp_db):
    """Minimal ModelMonitor — no external dependencies."""
    return ModelMonitor(db_path=tmp_db, min_predictions_for_alerts=2)


@pytest.fixture
def alert_manager(tmp_path):
    return AlertManager(
        log_file=str(tmp_path / "alerts.jsonl"),
        dedup_cooldown_seconds=300,
    )


# ======================================================================
# ModelMonitor — initialisation
# ======================================================================

class TestModelMonitorInit:
    def test_creates_db_file(self, tmp_db):
        ModelMonitor(db_path=tmp_db)
        assert tmp_db.exists()

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "monitor.db"
        ModelMonitor(db_path=deep)
        assert deep.exists()

    def test_creates_predictions_table(self, tmp_db):
        ModelMonitor(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "predictions" in tables

    def test_creates_alert_log_table(self, tmp_db):
        ModelMonitor(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "alert_log" in tables

    def test_default_thresholds(self, tmp_db):
        m = ModelMonitor(db_path=tmp_db)
        assert m.fraud_rate_threshold     == DEFAULT_FRAUD_RATE_THRESHOLD
        assert m.low_confidence_threshold == DEFAULT_LOW_CONFIDENCE_THRESHOLD
        assert m.min_predictions_for_alerts == DEFAULT_MIN_PREDICTIONS
        assert m.drift_ratio_threshold    == DEFAULT_DRIFT_RATIO_THRESHOLD

    def test_custom_thresholds(self, tmp_db):
        m = ModelMonitor(
            db_path=tmp_db,
            fraud_rate_threshold=0.5,
            low_confidence_threshold=0.4,
            min_predictions_for_alerts=5,
            drift_ratio_threshold=0.2,
        )
        assert m.fraud_rate_threshold       == 0.5
        assert m.low_confidence_threshold   == 0.4
        assert m.min_predictions_for_alerts == 5
        assert m.drift_ratio_threshold      == 0.2

    def test_accepts_drift_detector(self, tmp_db):
        dd = make_drift_detector()
        m  = ModelMonitor(db_path=tmp_db, drift_detector=dd)
        assert m.drift_detector is dd

    def test_accepts_inference_engine(self, tmp_db):
        ie = make_inference_engine()
        m  = ModelMonitor(db_path=tmp_db, inference_engine=ie)
        assert m.inference_engine is ie


# ======================================================================
# ModelMonitor — log_prediction
# ======================================================================

class TestLogPrediction:
    def test_returns_integer_id(self, monitor):
        row_id = monitor.log_prediction({"Amount": 100.0}, 0, 0.12)
        assert isinstance(row_id, int) and row_id >= 1

    def test_sequential_ids(self, monitor):
        id1 = monitor.log_prediction({}, 0, 0.1)
        id2 = monitor.log_prediction({}, 1, 0.9)
        assert id2 > id1

    def test_persists_prediction(self, monitor, tmp_db):
        monitor.log_prediction({"V1": 1.5}, 1, 0.85)
        conn = sqlite3.connect(tmp_db)
        row  = conn.execute("SELECT prediction FROM predictions").fetchone()
        conn.close()
        assert row[0] == 1

    def test_persists_probability(self, monitor, tmp_db):
        monitor.log_prediction({}, 0, 0.42)
        conn = sqlite3.connect(tmp_db)
        row  = conn.execute("SELECT probability FROM predictions").fetchone()
        conn.close()
        assert abs(row[0] - 0.42) < 1e-6

    def test_persists_transaction_dict(self, monitor, tmp_db):
        """Realistic input: Transaction.model_dump() from api.py."""
        tx = {"Time": 0.0, "Amount": 149.62, "V1": -1.35, "V2": -0.07}
        monitor.log_prediction(tx, 0, 0.03)
        conn  = sqlite3.connect(tmp_db)
        row   = conn.execute("SELECT input_json FROM predictions").fetchone()
        conn.close()
        parsed = json.loads(row[0])
        assert parsed["Amount"] == 149.62

    def test_handles_non_serialisable_input(self, monitor):
        """Falls back to str() for non-JSON types — must not raise."""
        row_id = monitor.log_prediction(object(), 0, 0.1)
        assert row_id >= 1

    def test_handles_list_input(self, monitor):
        row_id = monitor.log_prediction([1.0, 2.0, 3.0], 0, 0.05)
        assert row_id >= 1

    def test_persists_iso_timestamp(self, monitor, tmp_db):
        monitor.log_prediction({}, 0, 0.1)
        conn = sqlite3.connect(tmp_db)
        row  = conn.execute("SELECT ts FROM predictions").fetchone()
        conn.close()
        datetime.fromisoformat(row[0])   # raises if not valid ISO


# ======================================================================
# ModelMonitor — get_summary
# ======================================================================

class TestGetSummary:
    def test_empty_db(self, monitor):
        s = monitor.get_summary()
        assert s["total_count"] == 0
        assert s["fraud_rate"]  == 0.0
        assert s["window_start"] is None

    def test_total_count(self, monitor):
        for _ in range(5):
            monitor.log_prediction({}, 0, 0.1)
        assert monitor.get_summary()["total_count"] == 5

    def test_fraud_rate(self, monitor):
        monitor.log_prediction({}, 1, 0.9)
        monitor.log_prediction({}, 1, 0.8)
        monitor.log_prediction({}, 0, 0.2)
        monitor.log_prediction({}, 0, 0.1)
        s = monitor.get_summary()
        assert s["fraud_count"] == 2
        assert abs(s["fraud_rate"] - 0.5) < 1e-6

    def test_avg_probability(self, monitor):
        monitor.log_prediction({}, 0, 0.2)
        monitor.log_prediction({}, 0, 0.4)
        assert abs(monitor.get_summary()["avg_probability"] - 0.3) < 1e-6

    def test_min_max_probability(self, monitor):
        monitor.log_prediction({}, 0, 0.1)
        monitor.log_prediction({}, 1, 0.9)
        s = monitor.get_summary()
        assert abs(s["min_probability"] - 0.1) < 1e-6
        assert abs(s["max_probability"] - 0.9) < 1e-6

    def test_last_n_window(self, monitor):
        for _ in range(10):
            monitor.log_prediction({}, 0, 0.05)
        for _ in range(5):
            monitor.log_prediction({}, 1, 0.95)
        s = monitor.get_summary(last_n=5)
        assert s["total_count"] == 5
        assert s["fraud_count"] == 5

    def test_window_timestamps(self, monitor):
        monitor.log_prediction({}, 0, 0.1)
        monitor.log_prediction({}, 0, 0.2)
        s = monitor.get_summary()
        assert s["window_start"] is not None
        assert s["window_end"]   is not None


# ======================================================================
# ModelMonitor — get_total_predictions
# ======================================================================

class TestGetTotalPredictions:
    def test_zero_initially(self, monitor):
        assert monitor.get_total_predictions() == 0

    def test_counts_all_rows(self, monitor):
        for _ in range(7):
            monitor.log_prediction({}, 0, 0.1)
        assert monitor.get_total_predictions() == 7


# ======================================================================
# ModelMonitor — check_health
# ======================================================================

class TestCheckHealth:
    def test_required_keys_present(self, monitor):
        h = monitor.check_health()
        for k in ("status", "db_ok", "total_predictions",
                  "last_prediction_ts", "uptime_seconds",
                  "drift_available", "inference_available", "model_version"):
            assert k in h

    def test_healthy_on_empty_db(self, monitor):
        h = monitor.check_health()
        assert h["status"]             == "healthy"
        assert h["db_ok"]              is True
        assert h["last_prediction_ts"] is None
        assert h["model_version"]      == "unknown"

    def test_total_predictions_reflected(self, monitor):
        monitor.log_prediction({}, 0, 0.1)
        monitor.log_prediction({}, 1, 0.9)
        assert monitor.check_health()["total_predictions"] == 2

    def test_last_prediction_ts_set(self, monitor):
        monitor.log_prediction({}, 0, 0.3)
        assert monitor.check_health()["last_prediction_ts"] is not None

    def test_drift_available_false_by_default(self, monitor):
        assert monitor.check_health()["drift_available"] is False

    def test_drift_available_true_when_set(self, tmp_db):
        m = ModelMonitor(db_path=tmp_db, drift_detector=make_drift_detector())
        assert m.check_health()["drift_available"] is True

    def test_inference_available_false_by_default(self, monitor):
        assert monitor.check_health()["inference_available"] is False

    def test_inference_available_true_when_set(self, tmp_db):
        m = ModelMonitor(db_path=tmp_db, inference_engine=make_inference_engine())
        assert m.check_health()["inference_available"] is True

    def test_model_version_from_get_metadata(self, tmp_db):
        """FraudInferenceEngine.get_metadata() is the source of model_version."""
        ie = make_inference_engine(model_version="v2.5.1")
        m  = ModelMonitor(db_path=tmp_db, inference_engine=ie)
        h  = m.check_health()
        assert h["model_version"] == "v2.5.1"
        ie.get_metadata.assert_called_once()

    def test_model_version_unknown_when_get_metadata_raises(self, tmp_db):
        ie = MagicMock()
        ie.get_metadata.side_effect = RuntimeError("metadata unavailable")
        m  = ModelMonitor(db_path=tmp_db, inference_engine=ie)
        assert m.check_health()["model_version"] == "unknown"

    def test_model_version_unknown_when_key_missing(self, tmp_db):
        ie = make_inference_engine()
        # remove model_version from returned dict
        ie.get_metadata.return_value = {"threshold": 0.4}
        m  = ModelMonitor(db_path=tmp_db, inference_engine=ie)
        assert m.check_health()["model_version"] == "unknown"

    def test_unhealthy_on_db_error(self, tmp_db):
        m = ModelMonitor(db_path=tmp_db)
        with patch.object(m, "get_total_predictions", side_effect=Exception("boom")):
            h = m.check_health()
        assert h["status"] == "unhealthy"
        assert h["db_ok"]  is False

    def test_degraded_when_db_slow(self, tmp_db):
        m = ModelMonitor(db_path=tmp_db)
        with patch("time.monotonic", side_effect=[0.0, 3.0]):
            h = m.check_health()
        assert h["status"] == "degraded"


# ======================================================================
# ModelMonitor — trigger_alerts
# ======================================================================

class TestTriggerAlerts:
    def test_no_alerts_below_min_predictions(self, monitor):
        monitor.log_prediction({}, 1, 0.99)   # only 1; min=2
        assert monitor.trigger_alerts() == []

    def test_no_alerts_on_normal_traffic(self, monitor):
        monitor.log_prediction({}, 0, 0.8)
        monitor.log_prediction({}, 0, 0.9)
        assert monitor.trigger_alerts() == []

    # ── Rule 1: high_fraud_rate ─────────────────────────────────────

    def test_high_fraud_rate_fires_critical(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        alerts = monitor.trigger_alerts()
        names  = [a.name for a in alerts]
        assert "high_fraud_rate" in names

    def test_high_fraud_rate_severity_is_critical(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        alert = next(a for a in monitor.trigger_alerts() if a.name == "high_fraud_rate")
        assert alert.severity == "CRITICAL"

    def test_high_fraud_rate_extra_has_fraud_count(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        alert = next(a for a in monitor.trigger_alerts() if a.name == "high_fraud_rate")
        assert "fraud_count"  in alert.extra
        assert "total_count"  in alert.extra
        assert "fraud_rate"   in alert.extra
        assert "threshold"    in alert.extra

    # ── Rule 2: low_confidence ──────────────────────────────────────

    def test_low_confidence_fires_warning(self):
        with tempfile.TemporaryDirectory() as td:
            m = ModelMonitor(
                db_path=Path(td) / "m.db",
                low_confidence_threshold=0.95,   # very high → triggers
                min_predictions_for_alerts=2,
            )
            m.log_prediction({}, 0, 0.1)
            m.log_prediction({}, 0, 0.1)
            alerts = m.trigger_alerts()
        assert any(a.name == "low_confidence" for a in alerts)

    def test_low_confidence_severity_is_warning(self):
        with tempfile.TemporaryDirectory() as td:
            m = ModelMonitor(
                db_path=Path(td) / "m.db",
                low_confidence_threshold=0.95,
                min_predictions_for_alerts=2,
            )
            m.log_prediction({}, 0, 0.1)
            m.log_prediction({}, 0, 0.1)
            alert = next(a for a in m.trigger_alerts() if a.name == "low_confidence")
        assert alert.severity == "WARNING"

    # ── Rule 3: drift_detected ──────────────────────────────────────

    def test_drift_alert_requires_drift_df(self, tmp_db):
        """drift_df=None → no drift check even if detector is set."""
        m = ModelMonitor(
            db_path=tmp_db,
            drift_detector=make_drift_detector(drift_ratio=0.5),
            min_predictions_for_alerts=2,
        )
        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.9)
        alerts = m.trigger_alerts(drift_df=None)  # explicit None
        assert all(a.name != "drift_detected" for a in alerts)

    def test_drift_alert_fires_above_threshold(self, tmp_db):
        """DriftDetector.detect() returns drift_ratio >= 0.1 → alert."""
        dd = make_drift_detector(drift_ratio=0.35, drifted_count=4)
        m  = ModelMonitor(
            db_path=tmp_db,
            drift_detector=dd,
            drift_ratio_threshold=0.10,
            min_predictions_for_alerts=2,
        )
        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.9)
        df = pd.DataFrame([[0.0] * 30], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
        alerts = m.trigger_alerts(drift_df=df)
        assert any(a.name == "drift_detected" for a in alerts)
        # Verify DriftDetector.detect was called with the DataFrame
        dd.detect.assert_called_once_with(df)

    def test_drift_alert_not_fired_below_threshold(self, tmp_db):
        """drift_ratio 0.05 < threshold 0.10 → no alert."""
        dd = make_drift_detector(drift_ratio=0.05, drifted_count=0)
        m  = ModelMonitor(
            db_path=tmp_db,
            drift_detector=dd,
            drift_ratio_threshold=0.10,
            min_predictions_for_alerts=2,
        )
        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.9)
        df = pd.DataFrame([[0.0] * 30], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
        alerts = m.trigger_alerts(drift_df=df)
        assert all(a.name != "drift_detected" for a in alerts)

    def test_drift_alert_extra_matches_detect_output(self, tmp_db):
        """extra fields should map 1-to-1 from DriftDetector.detect() keys."""
        dd = make_drift_detector(drift_ratio=0.30, drifted_count=3)
        m  = ModelMonitor(
            db_path=tmp_db,
            drift_detector=dd,
            drift_ratio_threshold=0.10,
            min_predictions_for_alerts=2,
        )
        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.9)
        df = pd.DataFrame([[0.0] * 30], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
        alert = next(a for a in m.trigger_alerts(drift_df=df) if a.name == "drift_detected")
        assert alert.extra["drift_ratio"]     == 0.30
        assert alert.extra["drifted_count"]   == 3
        assert alert.extra["total_features"]  == 10
        assert "drifted_features" in alert.extra
        assert "status"           in alert.extra

    def test_drift_alert_status_string_in_message(self, tmp_db):
        """Status string from DriftDetector.detect() appears in alert message."""
        dd = make_drift_detector(drift_ratio=0.35, drifted_count=4)
        m  = ModelMonitor(
            db_path=tmp_db,
            drift_detector=dd,
            drift_ratio_threshold=0.10,
            min_predictions_for_alerts=2,
        )
        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.9)
        df = pd.DataFrame([[0.0] * 30], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
        alerts = m.trigger_alerts(drift_df=df)
        drift_alert = next(a for a in alerts if a.name == "drift_detected")
        assert "HIGH DRIFT" in drift_alert.message or "MODERATE" in drift_alert.message

    def test_drift_exception_does_not_crash(self, tmp_db):
        dd = MagicMock()
        dd.detect.side_effect = RuntimeError("PSI calc failed")
        m  = ModelMonitor(db_path=tmp_db, drift_detector=dd, min_predictions_for_alerts=2)
        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.8)
        df = pd.DataFrame([[0.0]])
        alerts = m.trigger_alerts(drift_df=df)  # must not raise
        assert isinstance(alerts, list)

    def test_multiple_rules_can_fire_together(self):
        with tempfile.TemporaryDirectory() as td:
            dd = make_drift_detector(drift_ratio=0.5, drifted_count=5)
            m  = ModelMonitor(
                db_path=Path(td) / "m.db",
                fraud_rate_threshold=0.1,
                low_confidence_threshold=0.95,
                drift_ratio_threshold=0.1,
                min_predictions_for_alerts=2,
                drift_detector=dd,
            )
            m.log_prediction({}, 1, 0.1)
            m.log_prediction({}, 1, 0.1)
            df = pd.DataFrame([[0.0] * 30], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
            alerts = m.trigger_alerts(drift_df=df)
        names = [a.name for a in alerts]
        assert "high_fraud_rate" in names
        assert "low_confidence"  in names
        assert "drift_detected"  in names

    def test_fired_alerts_persisted_to_db(self, monitor, tmp_db):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        monitor.trigger_alerts()
        conn  = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM alert_log").fetchone()[0]
        conn.close()
        assert count >= 1

    def test_alert_fired_at_is_iso(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        alert = monitor.trigger_alerts()[0]
        datetime.fromisoformat(alert.fired_at)


# ======================================================================
# ModelMonitor — alert history & clear helpers
# ======================================================================

class TestAlertHistoryAndClear:
    def test_get_alert_history_empty(self, monitor):
        assert monitor.get_alert_history() == []

    def test_get_alert_history_returns_dicts(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        monitor.trigger_alerts()
        history = monitor.get_alert_history()
        assert len(history) > 0
        assert isinstance(history[0], dict)
        assert "name" in history[0]
        assert "severity" in history[0]

    def test_last_n_limits_history(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        monitor.trigger_alerts()
        monitor.trigger_alerts()
        assert len(monitor.get_alert_history(last_n=1)) <= 1

    def test_clear_predictions(self, monitor):
        monitor.log_prediction({}, 0, 0.1)
        monitor.log_prediction({}, 0, 0.1)
        deleted = monitor.clear_predictions()
        assert deleted == 2
        assert monitor.get_total_predictions() == 0

    def test_clear_alerts(self, monitor):
        for _ in range(4):
            monitor.log_prediction({}, 1, 0.95)
        monitor.trigger_alerts()
        monitor.clear_alerts()
        assert monitor.get_alert_history() == []

    def test_db_rollback_on_bad_query(self, tmp_db):
        m = ModelMonitor(db_path=tmp_db)
        with pytest.raises(Exception):
            with m._conn() as conn:
                conn.execute("INSERT INTO nonexistent_table VALUES (1)")


# ======================================================================
# AlertManager — severity filtering
# ======================================================================

class TestAlertManagerSeverity:
    def test_info_passes_info_min(self, alert_manager):
        alert_manager.min_severity = "INFO"
        a = AlertRule(name="t", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        assert alert_manager.send_alert(a)["sent"] is True

    def test_info_blocked_by_warning_min(self, alert_manager):
        alert_manager.min_severity = "WARNING"
        a = AlertRule(name="t", severity="INFO", message="x", fired=True)
        r = alert_manager.send_alert(a)
        assert r["sent"] is False
        assert "severity" in r["skipped_reason"]

    def test_critical_passes_warning_min(self, alert_manager):
        alert_manager.min_severity = "WARNING"
        a = AlertRule(name="crit", severity="CRITICAL", message="x", fired=True, fired_at=datetime.now().isoformat())
        assert alert_manager.send_alert(a)["sent"] is True

    def test_warning_blocked_by_critical_min(self, alert_manager):
        alert_manager.min_severity = "CRITICAL"
        a = AlertRule(name="warn", severity="WARNING", message="x", fired=True)
        r = alert_manager.send_alert(a)
        assert r["sent"] is False

    def test_unknown_severity_treated_as_zero(self, alert_manager):
        alert_manager.min_severity = "INFO"
        a = AlertRule(name="unk", severity="UNKNOWN", message="x", fired=True, fired_at=datetime.now().isoformat())
        assert alert_manager.send_alert(a)["sent"] is True


# ======================================================================
# AlertManager — deduplication
# ======================================================================

class TestAlertManagerDedup:
    def test_first_send_passes(self, alert_manager):
        a = AlertRule(name="d1", severity="WARNING", message="x", fired=True, fired_at=datetime.now().isoformat())
        assert alert_manager.send_alert(a)["sent"] is True

    def test_second_within_cooldown_blocked(self, alert_manager):
        alert_manager.dedup_cooldown_seconds = 300
        a = AlertRule(name="d2", severity="WARNING", message="x", fired=True, fired_at=datetime.now().isoformat())
        alert_manager.send_alert(a)
        r = alert_manager.send_alert(a)
        assert r["sent"] is False
        assert "duplicate" in r["skipped_reason"]

    def test_zero_cooldown_always_sends(self, alert_manager):
        alert_manager.dedup_cooldown_seconds = 0
        a = AlertRule(name="d3", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        alert_manager.send_alert(a)
        assert alert_manager.send_alert(a)["sent"] is True

    def test_different_names_not_deduped(self, alert_manager):
        a1 = AlertRule(name="alpha", severity="INFO", message="a", fired=True, fired_at=datetime.now().isoformat())
        a2 = AlertRule(name="beta",  severity="INFO", message="b", fired=True, fired_at=datetime.now().isoformat())
        alert_manager.send_alert(a1)
        assert alert_manager.send_alert(a2)["sent"] is True

    def test_clear_cache_allows_resend(self, alert_manager):
        a = AlertRule(name="clear_test", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        alert_manager.send_alert(a)
        alert_manager.clear_dedup_cache()
        assert alert_manager.send_alert(a)["sent"] is True

    def test_send_count_increments(self, alert_manager):
        alert_manager.dedup_cooldown_seconds = 0
        a = AlertRule(name="cnt", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        alert_manager.send_alert(a)
        alert_manager.send_alert(a)
        cache = alert_manager.get_dedup_cache()
        assert cache["cnt"]["send_count"] == 2

    def test_get_dedup_cache_keys(self, alert_manager):
        a = AlertRule(name="snap", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        alert_manager.send_alert(a)
        snap = alert_manager.get_dedup_cache()["snap"]
        assert "last_sent" in snap
        assert "send_count" in snap


# ======================================================================
# AlertManager — log file channel
# ======================================================================

class TestAlertManagerLogFile:
    def test_log_file_created(self, tmp_path):
        log = str(tmp_path / "alerts.jsonl")
        mgr = AlertManager(log_file=log)
        a   = AlertRule(name="lf1", severity="INFO", message="hello", fired=True, fired_at=datetime.now().isoformat())
        mgr.send_alert(a)
        assert Path(log).exists()

    def test_log_file_valid_json(self, tmp_path):
        log = str(tmp_path / "alerts.jsonl")
        mgr = AlertManager(log_file=log)
        a   = AlertRule(name="lf2", severity="WARNING", message="check",
                        fired=True, fired_at=datetime.now().isoformat(),
                        extra={"fraud_rate": 0.45, "fraud_count": 45, "total_count": 100})
        mgr.send_alert(a)
        record = json.loads(Path(log).read_text().strip())
        assert record["name"]                   == "lf2"
        assert record["extra"]["fraud_rate"]    == 0.45

    def test_log_file_appends(self, tmp_path):
        log = str(tmp_path / "alerts.jsonl")
        mgr = AlertManager(log_file=log)
        for name in ("x1", "x2", "x3"):
            a = AlertRule(name=name, severity="INFO", message="m", fired=True, fired_at=datetime.now().isoformat())
            mgr.send_alert(a)
        lines = Path(log).read_text().strip().splitlines()
        assert len(lines) == 3

    def test_log_file_in_channels(self, tmp_path):
        mgr = AlertManager(log_file=str(tmp_path / "x.jsonl"))
        a   = AlertRule(name="ch", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        assert "log_file" in mgr.send_alert(a)["channels"]

    def test_log_file_write_failure_handled(self):
        mgr = AlertManager(log_file="/root/no_permission/alerts.jsonl")
        a   = AlertRule(name="fail", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        # must not raise — monitoring failures shouldn't surface
        result = mgr.send_alert(a)
        assert isinstance(result, dict)


# ======================================================================
# AlertManager — Slack channel
# ======================================================================

class TestAlertManagerSlack:
    def _make_mgr(self, tmp_path, webhook="https://hooks.slack.com/fake"):
        return AlertManager(
            slack_webhook_url=webhook,
            log_file=str(tmp_path / "sl.jsonl"),
        )

    def _mock_urlopen(self, status=200):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__  = MagicMock(return_value=False)
        mock_resp.status    = status
        return mock_resp

    def test_slack_not_in_channels_without_webhook(self, tmp_path):
        mgr = AlertManager(log_file=str(tmp_path / "x.jsonl"), slack_webhook_url=None)
        a   = AlertRule(name="ns", severity="INFO", message="x", fired=True, fired_at=datetime.now().isoformat())
        assert "slack" not in mgr.send_alert(a)["channels"]

    def test_slack_in_channels_on_200(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(name="s200", severity="WARNING", message="boom", fired=True, fired_at=datetime.now().isoformat())
        with patch("urllib.request.urlopen") as mock_uo:
            mock_uo.return_value = self._mock_urlopen(200)
            result = mgr.send_alert(a)
        assert "slack" in result["channels"]

    def test_slack_not_in_channels_on_500(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(name="s500", severity="CRITICAL", message="x", fired=True, fired_at=datetime.now().isoformat())
        with patch("urllib.request.urlopen") as mock_uo:
            mock_uo.return_value = self._mock_urlopen(500)
            result = mgr.send_alert(a)
        assert "slack" not in result["channels"]

    def test_slack_url_error_handled(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(name="surl", severity="CRITICAL", message="x", fired=True, fired_at=datetime.now().isoformat())
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            result = mgr.send_alert(a)
        assert "slack" not in result["channels"]

    def test_slack_unexpected_error_handled(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(name="sexp", severity="CRITICAL", message="x", fired=True, fired_at=datetime.now().isoformat())
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = mgr.send_alert(a)
        assert "slack" not in result["channels"]

    def test_slack_webhook_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(SLACK_WEBHOOK_ENV, "https://hooks.slack.com/env")
        mgr = AlertManager(log_file=str(tmp_path / "env.jsonl"))
        assert mgr.slack_webhook_url == "https://hooks.slack.com/env"

    def test_slack_no_webhook_send_returns_false(self, alert_manager):
        """_send_to_slack direct call returns False with no webhook."""
        alert_manager.slack_webhook_url = None
        a = AlertRule(name="x", severity="INFO", message="x", fired=True)
        assert alert_manager._send_to_slack(a) is False

    def test_slack_payload_contains_fraud_rate_extra(self, tmp_path):
        """When alert.extra has fraud_rate, Slack payload should include it."""
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(
            name="high_fraud_rate", severity="CRITICAL",
            message="Fraud spike", fired=True,
            fired_at=datetime.now().isoformat(),
            extra={"fraud_rate": 0.45, "fraud_count": 45, "total_count": 100,
                   "threshold": 0.30, "window": 100},
        )
        captured_payloads = []
        def fake_urlopen(req, timeout=None):
            captured_payloads.append(json.loads(req.data))
            return self._mock_urlopen(200)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            mgr.send_alert(a)
        assert len(captured_payloads) == 1
        fields = captured_payloads[0]["attachments"][0]["fields"]
        field_titles = [f["title"] for f in fields]
        assert "Fraud Rate" in field_titles

    def test_slack_payload_contains_drift_ratio_extra(self, tmp_path):
        """When alert.extra has drift_ratio, Slack payload includes it."""
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(
            name="drift_detected", severity="WARNING",
            message="Drift alert", fired=True,
            fired_at=datetime.now().isoformat(),
            extra={"drift_ratio": 0.3, "drifted_count": 9, "total_features": 30,
                   "drifted_features": ["V1", "V2"], "status": "🔴 HIGH DRIFT",
                   "threshold": 0.1},
        )
        captured = []
        def fake_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            return self._mock_urlopen(200)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            mgr.send_alert(a)
        fields = captured[0]["attachments"][0]["fields"]
        assert any(f["title"] == "Drift Ratio" for f in fields)

    def test_slack_payload_contains_avg_confidence(self, tmp_path):
        """When alert.extra has avg_probability, Slack shows it."""
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(
            name="low_confidence", severity="WARNING",
            message="Confidence low", fired=True,
            fired_at=datetime.now().isoformat(),
            extra={"avg_probability": 0.42, "threshold": 0.60},
        )
        captured = []
        def fake_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            return self._mock_urlopen(200)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            mgr.send_alert(a)
        fields = captured[0]["attachments"][0]["fields"]
        assert any(f["title"] == "Avg Confidence" for f in fields)

    def test_all_severity_colours_covered(self, tmp_path):
        """INFO/WARNING/CRITICAL each use a distinct colour — must not crash."""
        for sev in ("INFO", "WARNING", "CRITICAL"):
            mgr = self._make_mgr(tmp_path)
            a   = AlertRule(name=f"col_{sev}", severity=sev, message="x",
                            fired=True, fired_at=datetime.now().isoformat())
            with patch("urllib.request.urlopen") as mock_uo:
                mock_uo.return_value = self._mock_urlopen(200)
                mgr.send_alert(a)   # should not raise

    def test_unknown_severity_uses_fallback_colour(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        a   = AlertRule(name="unk", severity="UNKNOWN", message="x",
                        fired=True, fired_at=datetime.now().isoformat())
        with patch("urllib.request.urlopen") as mock_uo:
            mock_uo.return_value = self._mock_urlopen(200)
            mgr.send_alert(a)   # should not raise


# ======================================================================
# AlertManager — send_all
# ======================================================================

class TestSendAll:
    def test_returns_list_per_alert(self, alert_manager, tmp_path):
        alert_manager.log_file = str(tmp_path / "all.jsonl")
        alerts = [
            AlertRule(name="a1", severity="INFO",    message="1", fired=True, fired_at=datetime.now().isoformat()),
            AlertRule(name="a2", severity="WARNING",  message="2", fired=True, fired_at=datetime.now().isoformat()),
            AlertRule(name="a3", severity="CRITICAL", message="3", fired=True, fired_at=datetime.now().isoformat()),
        ]
        results = alert_manager.send_all(alerts)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_empty_list(self, alert_manager):
        assert alert_manager.send_all([]) == []


# ======================================================================
# Thread safety
# ======================================================================

class TestThreadSafety:
    def test_concurrent_log_predictions(self, monitor):
        errors = []
        def log_many():
            try:
                for _ in range(20):
                    monitor.log_prediction({"Amount": 1.0}, 0, 0.1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_many) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == []
        assert monitor.get_total_predictions() == 100

    def test_concurrent_alert_sends(self, alert_manager, tmp_path):
        alert_manager.log_file              = str(tmp_path / "conc.jsonl")
        alert_manager.dedup_cooldown_seconds = 0
        errors = []
        def send_many():
            try:
                for i in range(10):
                    a = AlertRule(name=f"t{i}", severity="INFO", message="x",
                                  fired=True, fired_at=datetime.now().isoformat())
                    alert_manager.send_alert(a)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=send_many) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []


# ======================================================================
# Full integration: monitor → trigger_alerts → alert_manager → log file
# ======================================================================

class TestEndToEndIntegration:
    def test_fraud_spike_pipeline(self, tmp_path):
        db  = tmp_path / "e2e.db"
        log = str(tmp_path / "e2e.jsonl")
        dd  = make_drift_detector(drift_ratio=0.0)
        ie  = make_inference_engine("v3.1.0")
        m   = ModelMonitor(
            db_path=db,
            fraud_rate_threshold=0.5,
            min_predictions_for_alerts=3,
            drift_detector=dd,
            inference_engine=ie,
        )
        mgr = AlertManager(log_file=log, dedup_cooldown_seconds=0)

        # Simulate fraud spike: 4/4 = 100% fraud
        for _ in range(4):
            m.log_prediction({"Amount": 500.0, "V1": -3.0}, 1, 0.97)

        alerts  = m.trigger_alerts()
        results = mgr.send_all(alerts)

        assert len(alerts) >= 1
        assert any(r["sent"] for r in results)
        assert Path(log).exists()

        # All lines must be valid JSON
        for line in Path(log).read_text().strip().splitlines():
            record = json.loads(line)
            assert "name" in record and "severity" in record

    def test_drift_spike_pipeline(self, tmp_path):
        db  = tmp_path / "drift.db"
        log = str(tmp_path / "drift.jsonl")
        dd  = make_drift_detector(drift_ratio=0.35, drifted_count=3)
        m   = ModelMonitor(
            db_path=db,
            drift_ratio_threshold=0.10,
            min_predictions_for_alerts=2,
            drift_detector=dd,
        )
        mgr = AlertManager(log_file=log, dedup_cooldown_seconds=0)

        m.log_prediction({}, 0, 0.8)
        m.log_prediction({}, 0, 0.9)
        df = pd.DataFrame(
            [[0.0] * 30],
            columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        )
        alerts  = m.trigger_alerts(drift_df=df)
        results = mgr.send_all(alerts)

        drift_alerts = [a for a in alerts if a.name == "drift_detected"]
        assert len(drift_alerts) == 1
        assert any(r["sent"] for r in results)

    def test_health_reflects_inference_engine_version(self, tmp_path):
        db = tmp_path / "hv.db"
        ie = make_inference_engine("v9.9.9")
        m  = ModelMonitor(db_path=db, inference_engine=ie)
        assert m.check_health()["model_version"] == "v9.9.9"