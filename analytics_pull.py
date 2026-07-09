"""Separate periodic pull from Zernio's Analytics API into the ``performance``
table. Intended to run on its own cron schedule (e.g. once daily), independent
of the posting cycle.

Pinterest's own API is never called — this pipeline posts and reads analytics
entirely through Zernio (https://zernio.com). One request covers the whole
lookback window (``GET /v1/analytics?platform=pinterest&fromDate=&toDate=``),
not one request per pin.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from agents.clients import ApiError, ZernioClient
from config import CONFIG
from db import Database, get_db, utcnow

logger = logging.getLogger("affiliate_engine.analytics")


def _extract_metric(entry: Dict[str, Any], *keys: str) -> int:
    """Best-effort extraction of a metric from a single analytics-post entry.

    Zernio's docs confirm impressions/saves/clicks are available but don't
    fully specify field names, so this checks a nested ``metrics`` dict (if
    present) and the entry itself, tolerating either casing/naming.
    """
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else entry
    for key in keys:
        for candidate_key in (key, key.lower(), key.upper()):
            value = metrics.get(candidate_key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return 0


def _post_identifier(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("postId", "post_id", "_id", "id"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def pull_all(db: Database = None, lookback_days: Optional[int] = None) -> int:
    """Pull analytics for every post with a pin_id, via a single Zernio call
    covering the whole lookback window. Returns the count of posts updated.
    """
    db = db or get_db()
    lookback_days = lookback_days if lookback_days is not None else CONFIG.analytics_lookback_days
    posts = db.posts_with_pins()
    if not posts:
        return 0

    now = utcnow()
    from_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")

    client = ZernioClient()
    try:
        analytics = client.get_analytics(from_date=from_date, to_date=to_date)
    except ApiError as exc:
        logger.error("Zernio analytics pull failed for %s..%s: %s", from_date, to_date, exc)
        return 0

    entries: List[Dict[str, Any]] = analytics.get("posts") or []
    by_pin_id = {pid: entry for entry in entries if isinstance(entry, dict) and (pid := _post_identifier(entry))}

    pulled = 0
    for post in posts:
        pin_id = post.get("pin_id")
        entry = by_pin_id.get(pin_id) if pin_id else None
        if entry is None:
            continue
        saves = _extract_metric(entry, "saves", "save")
        clicks = _extract_metric(entry, "clicks", "click", "pinClick")
        try:
            db.record_performance(post["id"], saves, clicks)
            pulled += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to record performance for post %s: %s", post["id"], exc)
    return pulled
