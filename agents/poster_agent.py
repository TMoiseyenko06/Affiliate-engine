"""Poster agent — pure code, no LLM.

Formats verified content into a Pinterest post and publishes it via Zernio
(https://zernio.com), a third-party scheduler — NOT Pinterest's own API v5
directly. Captures the response (post ID, URL) and writes it to the DB. On
API error it logs the full error and does NOT silently retry more than once.

Two-step flow (Zernio requires a public image URL, unlike Pinterest's own API
which accepts inline base64 bytes): upload the image once to get a public
URL, then attempt create-post up to twice reusing that same URL (Zernio's
temp upload storage is valid for the retry window; no need to re-upload).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from config import AFFILIATE_CONTENT_TYPES, CONFIG
from db import Database
from .clients import ApiError, ZernioClient

logger = logging.getLogger("affiliate_engine.poster")


def post_pin(
    content_type: str,
    copy: Dict[str, Any],
    image_bytes: bytes,
    subject: Dict[str, Any],
    db: Optional[Database] = None,
    board_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload the image and create the pin via Zernio; persist the result.

    Returns {"posted": bool, "pin_id": ..., "pin_url": ..., "post_id": ...,
    "error": ...}. Retries create-post at most once on API error, then gives up.
    """
    from db import get_db

    db = db or get_db()
    board_id = board_id or CONFIG.pinterest_board_id
    product_id_or_topic = subject.get("product_id") or subject.get("topic") or "unknown"

    if not board_id:
        return {"posted": False, "error": "No board_id configured", "post_id": None}

    client = ZernioClient()
    title = copy.get("title", "")
    description = copy.get("description", "")
    link = copy.get("link") or copy.get("link_or_null")

    last_error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None

    try:
        image_public_url = client.upload_media(image_bytes)
    except ApiError as exc:
        logger.error("Zernio media upload failed: %s", exc)
        post_id = db.create_post(
            content_type=content_type,
            product_id_or_topic=product_id_or_topic,
            board_id=board_id,
            status="failed",
        )
        return {"posted": False, "error": str(exc), "post_id": post_id}

    # At most two create-post attempts total (initial + one retry).
    for attempt in range(2):
        try:
            response = client.create_pinterest_post(
                title=title,
                description=description,
                image_public_url=image_public_url,
                board_id=board_id,
                link=link,
            )
            last_error = None
            break
        except ApiError as exc:
            last_error = str(exc)
            logger.error("Zernio Pinterest post attempt %d failed: %s", attempt + 1, exc)
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

    pin_id = ZernioClient.extract_post_id(response)
    pin_url = ZernioClient.extract_pin_url(response)
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
