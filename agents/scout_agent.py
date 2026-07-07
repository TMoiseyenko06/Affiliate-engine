"""Scout agent — finds a product candidate for the configured niche.

Pulls candidate products/trends, then uses a cheap/fast OpenRouter model to
rank and filter them. Excludes any product used within the reuse lookback
window (queried from the DB).

Returns structured JSON:
    {product_id, title, category, commission_rate, trend_signal, source_url}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from config import CONFIG
from db import Database
from .clients import ApiError, OpenRouterClient

logger = logging.getLogger("affiliate_engine.scout")


SYSTEM_PROMPT = (
    "You are a product-scouting analyst for a Pinterest affiliate marketing "
    "operation. You rank candidate Amazon products for a given niche by their "
    "likely Pinterest performance and affiliate value. You always respond with "
    "a single JSON object and nothing else."
)


def _fetch_candidates(niche: str) -> List[Dict[str, Any]]:
    """Fetch raw candidate products for the niche.

    If ``PRODUCT_SOURCE_URL`` is configured, fetch JSON from it. Otherwise fall
    back to a small built-in seed list so the pipeline is runnable without an
    external data source (useful for dry-runs / development).
    """
    if CONFIG.product_source_url:
        try:
            resp = requests.get(
                CONFIG.product_source_url,
                params={"niche": niche},
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data = data.get("products", [])
            if isinstance(data, list):
                return data
            logger.warning("Product source returned unexpected shape; using seeds")
        except (requests.RequestException, ValueError) as exc:
            logger.error("Failed to fetch product candidates: %s", exc)
            # Fall through to seeds rather than crashing the cycle.

    return _seed_candidates(niche)


def _seed_candidates(niche: str) -> List[Dict[str, Any]]:
    """Deterministic seed candidates keyed loosely by niche."""
    base = [
        {
            "product_id": "B08XYZHOME1",
            "title": "Stackable Clear Storage Bins with Lids (6-pack)",
            "category": "home_organization",
            "commission_rate": 0.04,
            "trend_signal": "rising",
            "source_url": "https://www.amazon.com/dp/B08XYZHOME1",
        },
        {
            "product_id": "B09ABCKITCH2",
            "title": "Bamboo Drawer Organizer Set",
            "category": "kitchen",
            "commission_rate": 0.03,
            "trend_signal": "steady",
            "source_url": "https://www.amazon.com/dp/B09ABCKITCH2",
        },
        {
            "product_id": "B07DEFDESK3",
            "title": "Minimalist Desk Cable Management Tray",
            "category": "home_office",
            "commission_rate": 0.045,
            "trend_signal": "rising",
            "source_url": "https://www.amazon.com/dp/B07DEFDESK3",
        },
        {
            "product_id": "B06GHILABEL4",
            "title": "Reusable Pantry Label Set (150 labels)",
            "category": "home_organization",
            "commission_rate": 0.05,
            "trend_signal": "hot",
            "source_url": "https://www.amazon.com/dp/B06GHILABEL4",
        },
    ]
    return base


def scout_product(
    niche: Optional[str] = None,
    db: Optional[Database] = None,
    extra_context: str = "",
) -> Dict[str, Any]:
    """Return a single ranked product candidate not recently used.

    Raises ``ApiError`` if scouting cannot produce a usable candidate.
    """
    from db import get_db

    db = db or get_db()
    niche = niche or CONFIG.primary_niche()

    candidates = _fetch_candidates(niche)
    excluded = set(db.products_used_within(CONFIG.reuse_lookback_days))
    fresh = [c for c in candidates if c.get("product_id") not in excluded]

    if not fresh:
        raise ApiError(
            f"No fresh product candidates for niche '{niche}' "
            f"(all {len(candidates)} excluded within {CONFIG.reuse_lookback_days}d)"
        )

    # Ask the cheap model to rank; degrade gracefully to a deterministic pick.
    ranked = _rank_with_llm(niche, fresh, db, extra_context)
    if ranked is None:
        ranked = _deterministic_pick(fresh)

    # Best-effort real product image for the creative step; None is fine.
    from amazon_scraper import fetch_product_image_url

    ranked["image_url"] = fetch_product_image_url(ranked.get("source_url", ""))

    # Persist the chosen product so it counts toward reuse exclusion.
    db.upsert_product(
        product_id=ranked["product_id"],
        title=ranked.get("title"),
        commission_rate=ranked.get("commission_rate"),
        source=ranked.get("source_url"),
    )
    return ranked


def _deterministic_pick(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the highest commission_rate as a fallback ranking."""
    signal_weight = {"hot": 3, "rising": 2, "steady": 1}
    return max(
        candidates,
        key=lambda c: (
            signal_weight.get(str(c.get("trend_signal", "")).lower(), 0),
            c.get("commission_rate", 0) or 0,
        ),
    )


def _rank_with_llm(
    niche: str,
    candidates: List[Dict[str, Any]],
    db: Database,
    extra_context: str,
) -> Optional[Dict[str, Any]]:
    """Use OpenRouter to choose the best candidate. Returns None on failure."""
    client = OpenRouterClient(db=db)
    user_prompt = (
        f"Niche: {niche}\n"
        f"Rank these Amazon product candidates for Pinterest affiliate performance "
        f"and pick the single best one.\n\n"
        f"Candidates JSON:\n{json.dumps(candidates, indent=2)}\n\n"
    )
    if extra_context:
        user_prompt += f"Additional context (address this):\n{extra_context}\n\n"
    user_prompt += (
        "Respond with a JSON object for the winning candidate ONLY, with exactly "
        "these keys: product_id, title, category, commission_rate, trend_signal, "
        "source_url. Copy values from the chosen candidate; do not invent products."
    )

    try:
        result = client.chat_json(
            model=CONFIG.scout_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
        )
    except (ApiError, ValueError) as exc:
        logger.error("Scout LLM ranking failed, falling back to heuristic: %s", exc)
        return None

    # Validate the LLM actually returned one of our candidates.
    valid_ids = {c["product_id"] for c in candidates}
    if result.get("product_id") not in valid_ids:
        logger.warning("Scout LLM returned unknown product_id; using heuristic")
        return None

    # Backfill any missing fields from the source candidate to be safe.
    source = next(c for c in candidates if c["product_id"] == result["product_id"])
    for key in ("title", "category", "commission_rate", "trend_signal", "source_url"):
        result.setdefault(key, source.get(key))
    return result
