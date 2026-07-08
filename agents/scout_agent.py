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
from .clients import ApiError, OpenRouterClient, resolve_asin

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
    """Fallback candidates used only when PRODUCT_SOURCE_URL is not configured.

    These are REAL, verified ASINs (checked against live Amazon listings), not
    invented placeholders — earlier versions of this list used made-up ASINs
    that don't resolve to real products, which produced hallucinated copy and
    broken affiliate links. Each entry carries a real ``features`` list so the
    copywriter has actual product facts to write from instead of a bare title.

    This is still a small, static, hand-checked set meant to make the pipeline
    runnable out of the box — listings go stale (delisted, repriced). For
    ongoing production use, configure PRODUCT_SOURCE_URL to point at a real,
    current product feed (ideally backed by the Amazon Product Advertising API).
    """
    base = [
        {
            "product_id": "B0CJFY5FM9",
            "title": "ThreeKin 230 Black Spice Labels - Waterproof & Oil-Resistant Pre-Printed Stickers",
            "category": "home_organization",
            "commission_rate": 0.04,
            "trend_signal": "rising",
            "source_url": "https://www.amazon.com/dp/B0CJFY5FM9",
            "features": [
                "230 pre-printed spice labels",
                "waterproof and oil-resistant",
                "BPA-free",
                "easy-clean, no-residue removal",
            ],
        },
        {
            "product_id": "B09V5MBSDX",
            "title": "Amazon Basics Plastic Storage Containers with Secure Latching Lids, Set of 10",
            "category": "home_organization",
            "commission_rate": 0.04,
            "trend_signal": "steady",
            "source_url": "https://www.amazon.com/dp/B09V5MBSDX",
            "features": [
                "set of 10 stackable bins",
                "5 quart capacity each",
                "secure latching lids",
                "clear body with grey lids",
            ],
        },
        {
            "product_id": "B07DFBSTFR",
            "title": "IRIS USA 20-Pack Storage Bins with Lids, 6 Quart, Clear Stackable Containers",
            "category": "home_organization",
            "commission_rate": 0.045,
            "trend_signal": "rising",
            "source_url": "https://www.amazon.com/dp/B07DFBSTFR",
            "features": [
                "20-pack, 6 quart each",
                "clear see-through stackable containers",
                "latching lids",
                "BPA-free plastic",
            ],
        },
        {
            "product_id": "B088WCT93C",
            "title": "ROYAL CRAFT WOOD 5-Piece Bamboo Drawer Organizer Set",
            "category": "kitchen",
            "commission_rate": 0.03,
            "trend_signal": "steady",
            "source_url": "https://www.amazon.com/dp/B088WCT93C",
            "features": [
                "5-piece nesting bamboo tray set",
                "multi-use: kitchen, bathroom, office, makeup, jewelry",
                "natural bamboo construction",
                "fits standard drawer widths",
            ],
        },
    ]
    return base


def scout_product(
    niche: Optional[str] = None,
    db: Optional[Database] = None,
    extra_context: str = "",
) -> Dict[str, Any]:
    """Return a single ranked product candidate not already used (per
    ``CONFIG.product_reuse_mode`` — permanent or a rolling cooldown).

    The exclusion check is always a single local DB query turned into an
    in-memory set, done BEFORE any ranking or image-fetching calls — no
    external API (OpenRouter ranking, ScraperAPI image fetch) is ever called
    per-candidate-until-fresh; both are called at most once, only for the
    single already-chosen winner.

    Raises ``ApiError`` if scouting cannot produce a usable candidate (only
    possible in cooldown mode, or permanent mode with the fallback disabled).
    """
    from db import get_db

    db = db or get_db()
    niche = niche or CONFIG.primary_niche()

    if CONFIG.test_force_product_url:
        return _forced_test_product(niche, db)

    candidates = _fetch_candidates(niche)
    if CONFIG.product_reuse_mode == "permanent":
        excluded = set(db.all_used_product_ids())
    else:
        excluded = set(db.products_used_within(CONFIG.reuse_lookback_days))
    fresh = [c for c in candidates if c.get("product_id") not in excluded]

    if not fresh:
        fresh = _handle_exhausted_catalog(niche, candidates, db)

    # Ask the cheap model to rank; degrade gracefully to a deterministic pick.
    ranked = _rank_with_llm(niche, fresh, db, extra_context)
    if ranked is None:
        ranked = _deterministic_pick(fresh)

    # Best-effort real product image for the creative step; None is fine.
    # Scraping is confirmed blocked by Amazon (see config.py) so this normally
    # returns None immediately; TEST_PRODUCT_IMAGE_URL is a manual override
    # for exercising the reference-image flow without a working scraper.
    from amazon_scraper import fetch_product_image_url

    ranked["image_url"] = fetch_product_image_url(ranked.get("source_url", "")) or CONFIG.test_product_image_url

    # Persist the chosen product so it counts toward reuse exclusion.
    db.upsert_product(
        product_id=ranked["product_id"],
        title=ranked.get("title"),
        commission_rate=ranked.get("commission_rate"),
        source=ranked.get("source_url"),
    )
    return ranked


def _handle_exhausted_catalog(
    niche: str, candidates: List[Dict[str, Any]], db: Database
) -> List[Dict[str, Any]]:
    """Every candidate for this niche has already been used.

    In cooldown mode, or permanent mode with the fallback disabled, this is a
    hard stop (raises). In permanent mode with the fallback enabled (the
    default), fall back to re-posting the single least-recently-used
    candidate rather than stalling the pipeline entirely, and raise an alert
    so it's obvious the product catalog needs topping up. Still just one
    local DB query — no extra API calls.
    """
    if CONFIG.product_reuse_mode == "permanent" and CONFIG.product_reuse_fallback_enabled:
        candidate_ids = [c["product_id"] for c in candidates if c.get("product_id")]
        lru_id = db.least_recently_used_product_id(candidate_ids)
        if lru_id:
            from alerting import alert

            alert(
                f"Product catalog exhausted for niche '{niche}': all {len(candidates)} "
                f"candidates have already been posted (PRODUCT_REUSE_MODE=permanent). "
                f"Falling back to re-posting the least-recently-used product ({lru_id}). "
                f"Add more products to PRODUCT_SOURCE_URL or the seed list.",
                db,
            )
            return [c for c in candidates if c.get("product_id") == lru_id]

    raise ApiError(
        f"No fresh product candidates for niche '{niche}' (all {len(candidates)} "
        f"excluded; mode={CONFIG.product_reuse_mode})"
    )


def _forced_test_product(niche: str, db: Database) -> Dict[str, Any]:
    """Return TEST_FORCE_PRODUCT_URL as the product, bypassing ranking/reuse.

    Testing-only escape hatch: lets you repeatedly exercise the pipeline
    against one specific real listing instead of whatever the ranking step
    would otherwise pick. If the URL's ASIN matches a seed candidate, that
    candidate's real feature data is used instead of a bare stub.
    """
    url = CONFIG.test_force_product_url
    asin = resolve_asin(url)
    logger.warning(
        "TEST_FORCE_PRODUCT_URL is set — using forced product %s instead of "
        "normal scouting. Unset it for real operation.",
        asin or url,
    )

    seed_match = next(
        (c for c in _seed_candidates(niche) if c.get("product_id") == asin), None
    )
    if seed_match:
        product = dict(seed_match)
    else:
        product_id = asin or f"TEST-{abs(hash(url)) % 100000}"
        title = CONFIG.test_force_product_title or f"Test product ({product_id})"
        product = {
            "product_id": product_id,
            "title": title,
            "category": niche,
            "commission_rate": None,
            "trend_signal": "forced-test",
            "source_url": url,
        }

    # Use the canonical amazon.com/dp/ASIN form, not a short link — appending
    # ?tag= to a short link often silently fails to carry the Associates tag
    # through the redirect (shorteners frequently drop unrecognized params).
    if asin:
        product["source_url"] = f"https://www.amazon.com/dp/{asin}"

    from amazon_scraper import fetch_product_image_url

    product["image_url"] = fetch_product_image_url(url) or CONFIG.test_product_image_url
    db.upsert_product(
        product_id=product["product_id"],
        title=product["title"],
        commission_rate=product["commission_rate"],
        source=product["source_url"],
    )
    return product


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
        'Respond with a JSON object containing ONLY the winning candidate\'s '
        '"product_id" — do not repeat or restate any other fields.'
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

    # The LLM chooses which product wins; we never trust it to reproduce that
    # product's data (name, features, price, etc.) — pulling the untouched
    # original candidate avoids any risk of the LLM mangling real product facts.
    valid_ids = {c["product_id"] for c in candidates}
    if result.get("product_id") not in valid_ids:
        logger.warning("Scout LLM returned unknown product_id; using heuristic")
        return None

    source = next(c for c in candidates if c["product_id"] == result["product_id"])
    return dict(source)
