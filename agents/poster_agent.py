"""Poster agent — pure code, no LLM.

Formats verified content into a Pinterest API v5 create-pin request, posts it,
captures the response (pin ID, URL), and writes it to the DB. On API error it
logs the full error and does NOT silently retry more than once.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from config import AFFILIATE_CONTENT_TYPES, CONFIG
from db import Database
from .clients import ApiError, PinterestClient

logger = logging.getLogger("affiliate_engine.poster")


def _pin_url(response: Dict[str, Any]) -> Optional[str]:
    pin_id = response.get("id")
    # v5 responses don't always include a canonical URL; construct one.
    if pin_id:
        return f"https://www.pinterest.com/pin/{pin_id}/"
    return None


def post_pin(
    content_type: str,
    copy: Dict[str, Any],
    image_bytes: bytes,
    subject: Dict[str, Any],
    db: Optional[Database] = None,
    board_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the pin and persist the result.

    Returns {"posted": bool, "pin_id": ..., "pin_url": ..., "post_id": ...,
    "error": ...}. Retries at most once on API error, then gives up.
    """
    from db import get_db

    db = db or get_db()
    board_id = board_id or CONFIG.pinterest_board_id
    product_id_or_topic = subject.get("product_id") or subject.get("topic") or "unknown"

    if not board_id:
        return {"posted": False, "error": "No board_id configured", "post_id": None}

    client = PinterestClient()
    title = copy.get("title", "")
    description = copy.get("description", "")
    link = copy.get("link") or copy.get("link_or_null")

    last_error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None

    # At most two attempts total (initial + one retry).
    for attempt in range(2):
        try:
            response = client.create_pin(
                board_id=board_id,
                title=title,
                description=description,
                image_bytes=image_bytes,
                link=link,
            )
            last_error = None
            break
        except ApiError as exc:
            last_error = str(exc)
            logger.error("Pinterest post attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                break  # do not retry more than once

    if response is None:
        # Record the failed attempt.
        post_id = db.create_post(
            content_type=content_type,
            product_id_or_topic=product_id_or_topic,
            board_id=board_id,
            status="failed",
        )
        return {"posted": False, "error": last_error, "post_id": post_id}

    pin_id = response.get("id")
    pin_url = _pin_url(response)
    post_id = db.create_post(
        content_type=content_type,
        product_id_or_topic=product_id_or_topic,
        board_id=board_id,
        status="posted",
        pin_id=pin_id,
        pin_url=pin_url,
    )
    # Mark product used only after a confirmed successful post.
    if content_type in AFFILIATE_CONTENT_TYPES and subject.get("product_id"):
        db.mark_product_used(subject["product_id"])

    return {
        "posted": True,
        "pin_id": pin_id,
        "pin_url": pin_url,
        "post_id": post_id,
        "error": None,
    }
