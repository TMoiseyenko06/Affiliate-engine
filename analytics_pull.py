"""Separate periodic pull from Pinterest Analytics API into the ``performance``
table. Intended to run on its own cron schedule (e.g. once daily), independent
of the posting cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from agents.clients import ApiError, PinterestClient
from db import Database, get_db

logger = logging.getLogger("affiliate_engine.analytics")


def _extract_metric(analytics: Dict[str, Any], key: str) -> int:
    """Best-effort extraction of a metric total from a v5 analytics response.

    The v5 shape nests metrics under ``all.daily_metrics`` or ``all.summary_metrics``;
    we sum whatever daily values we find, tolerating shape drift.
    """
    try:
        all_bucket = analytics.get("all", analytics)
        summary = all_bucket.get("summary_metrics") or {}
        if key in summary and summary[key] is not None:
            return int(summary[key])
        total = 0
        for day in all_bucket.get("daily_metrics", []):
            metrics = day.get("metrics", {})
            if metrics.get(key) is not None:
                total += int(metrics[key])
        return total
    except (AttributeError, TypeError, ValueError):
        return 0


def pull_all(db: Database = None) -> int:
    """Pull analytics for every post that has a pin_id. Returns the count pulled."""
    db = db or get_db()
    client = PinterestClient()
    posts = db.posts_with_pins()
    pulled = 0
    for post in posts:
        pin_id = post.get("pin_id")
        if not pin_id:
            continue
        try:
            analytics = client.get_pin_analytics(pin_id)
        except ApiError as exc:
            logger.error("Analytics pull failed for pin %s: %s", pin_id, exc)
            continue
        saves = _extract_metric(analytics, "SAVE")
        clicks = _extract_metric(analytics, "PIN_CLICK")
        try:
            db.record_performance(post["id"], saves, clicks)
            pulled += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to record performance for post %s: %s", post["id"], exc)
    return pulled
