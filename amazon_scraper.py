"""Best-effort fetch of a product listing's primary image.

INTERIM MEASURE: this is a stopgap until the official Amazon Product
Advertising API (PA-API 5.0) is wired up, which requires an approved
Associates account (qualifying recent sales) and AWS-style request signing.

Two strategies, tried in order:
1. ScraperAPI's structured Amazon product endpoint (SCRAPERAPI_KEY) — a
   third-party proxy/scraping service used because PA-API access isn't
   available yet. Returns real product JSON rather than HTML to parse.
2. Direct HTTP scrape of the listing page (SCRAPE_PRODUCT_IMAGES) — CONFIRMED
   BLOCKED by Amazon's bot wall in testing from two independent networks;
   kept as a fallback in case that ever changes, off by default.

Designed to be trivially replaceable: swap the body of
``fetch_product_image_url`` for a PA-API call later without touching any
caller (scout_agent stores whatever URL — or None — comes back).

Never raises: any failure (network, parsing, disabled via config) returns
None, and callers fall back to pure text-to-image generation.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, Optional

import requests

from config import CONFIG

logger = logging.getLogger("affiliate_engine.amazon_scraper")

_TIMEOUT_SECONDS = 15
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_OG_IMAGE_RE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_HIRES_RE = re.compile(r'"hiRes":"(https:[^"]+)"')
_LARGE_RE = re.compile(r'"large":"(https:[^"]+)"')


def fetch_product_image_url(product_url: str) -> Optional[str]:
    """Return the listing's primary image URL, or None if unavailable."""
    if not product_url:
        return None

    if CONFIG.scraperapi_key:
        url = _fetch_via_scraperapi(product_url)
        if url:
            return url

    if CONFIG.scrape_product_images:
        return _fetch_via_direct_scrape(product_url)

    return None


def _fetch_via_scraperapi(product_url: str) -> Optional[str]:
    """Fetch via ScraperAPI's structured Amazon product endpoint.

    Per https://docs.scraperapi.com: GET {base}/structured/amazon/product
    with asin, country, api_key query params; returns product JSON.
    The exact shape of the "images" field is UNVERIFIED (ScraperAPI's docs
    didn't show a full example response when this was built) — parsing
    below tolerates several plausible shapes and logs the raw response keys
    on failure so a wrong guess is easy to diagnose and fix.
    """
    from agents.clients import resolve_asin

    asin = resolve_asin(product_url)
    if not asin:
        logger.warning("Could not resolve an ASIN from %s for ScraperAPI lookup", product_url)
        return None

    try:
        resp = requests.get(
            f"{CONFIG.scraperapi_base_url.rstrip('/')}/structured/amazon/product",
            params={"asin": asin, "country": CONFIG.scraperapi_country, "api_key": CONFIG.scraperapi_key},
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as exc:
        body = (exc.response.text or "")[:500] if exc.response is not None else ""
        logger.warning("ScraperAPI HTTP error for ASIN %s: %s :: %s", asin, exc, body)
        return None
    except (requests.RequestException, ValueError) as exc:
        logger.warning("ScraperAPI request failed for ASIN %s: %s", asin, exc)
        return None

    url = _extract_scraperapi_image(data)
    if url:
        logger.info("Fetched product image via ScraperAPI for ASIN %s", asin)
        return url

    logger.warning(
        "ScraperAPI response for ASIN %s had no recognizable image field; top-level keys: %s",
        asin, list(data.keys()) if isinstance(data, dict) else type(data),
    )
    return None


def _extract_scraperapi_image(data: Any) -> Optional[str]:
    """Tolerantly pull an image URL out of ScraperAPI's structured response."""
    if not isinstance(data, dict):
        return None
    images = data.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("url", "image", "link", "large", "hi_res"):
                if first.get(key):
                    return first[key]
    if isinstance(images, str):
        return images
    for key in ("main_image", "image", "image_url", "primary_image"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict):
            for subkey in ("url", "link"):
                if val.get(subkey):
                    return val[subkey]
    return None


def _fetch_via_direct_scrape(product_url: str) -> Optional[str]:
    """Direct HTTP scrape — CONFIRMED BLOCKED by Amazon's bot wall in
    testing; kept only for anyone who has their own working workaround."""
    try:
        resp = requests.get(
            product_url,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        page = resp.text
    except requests.RequestException as exc:
        logger.warning("Direct product image scrape failed for %s: %s", product_url, exc)
        return None

    for pattern in (_OG_IMAGE_RE, _HIRES_RE, _LARGE_RE):
        match = pattern.search(page)
        if match:
            url = html.unescape(match.group(1))
            logger.info("Scraped product image for %s", product_url)
            return url

    logger.warning("No product image found on page: %s", product_url)
    return None
