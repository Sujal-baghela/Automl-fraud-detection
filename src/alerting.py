"""
alerting.py — AlertManager for AutoML-X fraud detection system.

Routes fired AlertRule objects to:
  - Log file  (always, newline-delimited JSON)
  - Slack incoming webhook (optional)

Features
--------
- Severity levels:  INFO | WARNING | CRITICAL
- Alert dedup:      same alert won't re-fire within cooldown window (thread-safe)
- Zero extra deps:  Slack uses stdlib urllib.request
- Webhook URL from constructor or AUTOML_SLACK_WEBHOOK env var
"""

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.monitor import AlertRule

logger = logging.getLogger(__name__)

SEVERITY_LEVELS     = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
SLACK_WEBHOOK_ENV   = "AUTOML_SLACK_WEBHOOK"
DEFAULT_COOLDOWN_S  = 300   # 5 minutes between identical alerts


@dataclass
class SentAlert:
    name: str
    severity: str
    last_sent: datetime
    send_count: int = 1


class AlertManager:
    """
    Routes AlertRule objects to configured output channels.

    Parameters
    ----------
    slack_webhook_url     : Slack incoming webhook URL.
                            Falls back to AUTOML_SLACK_WEBHOOK env var.
    log_file              : path to append alert JSON lines.
                            Default: "models/alerts.jsonl"
    min_severity          : drop alerts below this level.
                            One of "INFO", "WARNING", "CRITICAL".
    dedup_cooldown_seconds: same alert_name won't fire again within this window.
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        log_file: str = "models/alerts.jsonl",
        min_severity: str = "INFO",
        dedup_cooldown_seconds: int = DEFAULT_COOLDOWN_S,
    ):
        self.slack_webhook_url     = slack_webhook_url or os.getenv(SLACK_WEBHOOK_ENV)
        self.log_file              = log_file
        self.min_severity          = min_severity
        self.dedup_cooldown_seconds = dedup_cooldown_seconds

        # thread-safe dedup cache: alert_name → SentAlert
        self._dedup_cache: dict[str, SentAlert] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, alert: AlertRule) -> dict:
        """
        Route one alert to all configured channels.

        Returns
        -------
        {"sent": bool, "channels": [str, ...], "skipped_reason": str | None}
        """
        if not self._meets_severity(alert.severity):
            return {
                "sent":           False,
                "channels":       [],
                "skipped_reason": f"severity {alert.severity} below minimum {self.min_severity}",
            }

        if self._is_duplicate(alert):
            return {
                "sent":           False,
                "channels":       [],
                "skipped_reason": "duplicate within cooldown window",
            }

        channels_sent: list[str] = []

        # Log file — always attempted
        if self._send_to_log_file(alert):
            channels_sent.append("log_file")

        # Slack — only if webhook configured
        if self.slack_webhook_url:
            if self._send_to_slack(alert):
                channels_sent.append("slack")

        self._mark_sent(alert)

        return {"sent": True, "channels": channels_sent, "skipped_reason": None}

    def send_all(self, alerts: list[AlertRule]) -> list[dict]:
        """Send a list of alerts; returns one result dict per alert."""
        return [self.send_alert(a) for a in alerts]

    def get_dedup_cache(self) -> dict:
        """Return a snapshot of the current dedup cache (debug / testing)."""
        with self._lock:
            return {
                k: {
                    "name":       v.name,
                    "severity":   v.severity,
                    "last_sent":  v.last_sent.isoformat(),
                    "send_count": v.send_count,
                }
                for k, v in self._dedup_cache.items()
            }

    def clear_dedup_cache(self) -> None:
        """Reset the dedup cache."""
        with self._lock:
            self._dedup_cache.clear()

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    def _send_to_log_file(self, alert: AlertRule) -> bool:
        """Append one JSON line to the alert log file."""
        record = {
            "ts":       alert.fired_at or datetime.now().isoformat(),
            "name":     alert.name,
            "severity": alert.severity,
            "message":  alert.message,
            "extra":    alert.extra,
        }
        try:
            dir_part = os.path.dirname(self.log_file)
            if dir_part:
                os.makedirs(dir_part, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logger.info("[AlertManager] %s — %s", alert.severity, alert.name)
            return True
        except Exception as exc:
            logger.error("[AlertManager] Log file write failed: %s", exc)
            return False

    def _send_to_slack(self, alert: AlertRule) -> bool:
        """
        POST a formatted message to the Slack incoming webhook.

        Message format mirrors the AutoML-X project style:
          - colour-coded attachment (green/yellow/red)
          - shows fraud probability / drift ratio from alert.extra when present
        """
        if not self.slack_webhook_url:
            return False

        emoji = {
            "INFO":     ":information_source:",
            "WARNING":  ":warning:",
            "CRITICAL": ":rotating_light:",
        }.get(alert.severity, ":bell:")

        colour = {
            "INFO":     "#36a64f",
            "WARNING":  "#ffcc00",
            "CRITICAL": "#ff0000",
        }.get(alert.severity, "#cccccc")

        # Build context-aware extra fields shown in Slack message
        extra_fields = []
        if alert.extra:
            # fraud rate alert
            if "fraud_rate" in alert.extra:
                extra_fields.append({
                    "title": "Fraud Rate",
                    "value": f"{alert.extra['fraud_rate']:.1%}  "
                             f"({alert.extra.get('fraud_count', '?')}/"
                             f"{alert.extra.get('total_count', '?')} txns)",
                    "short": True,
                })
            # drift alert
            if "drift_ratio" in alert.extra:
                extra_fields.append({
                    "title": "Drift Ratio",
                    "value": f"{alert.extra['drift_ratio']:.1%}  "
                             f"({alert.extra.get('drifted_count', '?')}/"
                             f"{alert.extra.get('total_features', '?')} features)",
                    "short": True,
                })
            # confidence alert
            if "avg_probability" in alert.extra:
                extra_fields.append({
                    "title": "Avg Confidence",
                    "value": f"{alert.extra['avg_probability']:.3f}",
                    "short": True,
                })

        payload = {
            "text": f"{emoji} *AutoML-X Fraud Monitor* [{alert.severity}]",
            "attachments": [{
                "color": colour,
                "fields": [
                    {"title": "Alert",    "value": alert.name,     "short": True},
                    {"title": "Severity", "value": alert.severity, "short": True},
                    {"title": "Message",  "value": alert.message,  "short": False},
                    *extra_fields,
                ],
                "footer": f"AutoML-X | {alert.fired_at or datetime.now().isoformat()}",
            }],
        }

        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self.slack_webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("[AlertManager] Slack sent: %s", alert.name)
                    return True
                logger.warning("[AlertManager] Slack status %s", resp.status)
                return False
        except urllib.error.URLError as exc:
            logger.error("[AlertManager] Slack URLError: %s", exc)
            return False
        except Exception as exc:
            logger.error("[AlertManager] Slack unexpected error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Dedup helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, alert: AlertRule) -> bool:
        with self._lock:
            cached = self._dedup_cache.get(alert.name)
            if cached is None:
                return False
            elapsed = (datetime.now() - cached.last_sent).total_seconds()
            return elapsed < self.dedup_cooldown_seconds

    def _mark_sent(self, alert: AlertRule) -> None:
        with self._lock:
            cached = self._dedup_cache.get(alert.name)
            if cached:
                cached.last_sent  = datetime.now()
                cached.send_count += 1
            else:
                self._dedup_cache[alert.name] = SentAlert(
                    name=alert.name,
                    severity=alert.severity,
                    last_sent=datetime.now(),
                )

    def _meets_severity(self, severity: str) -> bool:
        return SEVERITY_LEVELS.get(severity, 0) >= SEVERITY_LEVELS.get(self.min_severity, 0)