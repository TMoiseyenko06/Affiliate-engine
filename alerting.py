"""Alerting for the pipeline.

Alerts are always persisted to the ``alerts`` DB table and appended to a flat
alert file (``ALERT_FILE_PATH``). Optionally, if ``ALERT_WEBHOOK_URL`` is set,
the alert is also POSTed to that webhook (Slack-compatible ``{"text": ...}``).

Triggers covered here:
- ``alert``                : generic alert (used by orchestrator on skips)
- ``check_deadman_switch`` : no successful post in the configured window
- ``alert_budget_capped``  : a daily API budget cap was hit
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from config import CONFIG
from db import Database, utcnow

logger = logging.getLogger("affiliate_engine.alerting")


def alert(message: str, db: Optional[Database] = None) -> None:
    """Record an alert to the DB, the alert file, and (optionally) a webhook."""
    logger.warning("ALERT: %s", message)
    timestamp = utcnow().isoformat()
    line = f"{timestamp}\t{message}\n"

    # 1. Flat file (never let file IO crash the caller).
    try:
        with open(CONFIG.alert_file_path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        logger.error("Failed to write alert file: %s", exc)

    # 2. DB.
    if db is not None:
        try:
            db.add_alert(message)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to write alert to DB: %s", exc)

    # 3. Optional webhook.
    if CONFIG.alert_webhook_url:
        _post_webhook(message)


def _post_webhook(message: str) -> None:
    try:
        import requests

        requests.post(
            CONFIG.alert_webhook_url,
            json={"text": f"[affiliate-engine] {message}"},
            timeout=CONFIG.http_timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Failed to post alert webhook: %s", exc)


def alert_skip(post_attempt_id: str, reason: str, db: Optional[Database] = None) -> None:
    alert(f"Cycle {post_attempt_id} SKIPPED after repeated verifier failure: {reason}", db)


def alert_budget_capped(provider: str, db: Optional[Database] = None) -> None:
    alert(f"Daily API budget cap hit for provider '{provider}'.", db)


def check_deadman_switch(db: Database) -> bool:
    """Alert if no successful post within ``DEADMAN_HOURS``.

    Returns True if an alert was raised (i.e. the switch tripped).
    """
    last = db.last_successful_post_time()
    threshold = utcnow() - timedelta(hours=CONFIG.deadman_hours)
    if last is None:
        alert(
            f"Dead-man's-switch: no successful post has EVER been recorded "
            f"(threshold {CONFIG.deadman_hours}h).",
            db,
        )
        return True
    if last < threshold:
        alert(
            f"Dead-man's-switch: no successful post since {last.isoformat()} "
            f"(> {CONFIG.deadman_hours}h ago).",
            db,
        )
        return True
    return False
