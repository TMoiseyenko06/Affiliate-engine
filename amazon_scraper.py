"""Best-effort scraper for a product listing's primary image.

INTERIM MEASURE: this is a stopgap until the official Amazon Product
Advertising API (PA-API 5.0) is wired up, which requires an approved
Associates account (qualifying recent sales) and AWS-style request signing.
Scraping a listing page is against Amazon's Terms of Service and fragile to
markup changes — this module exists so the pipeline can be tested with real
product imagery today, not as a durable production strategy.

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
from typing import Optional

import requests

from config import CONFIG

logger = logging.getLogger("affiliate_engine.amazon_scraper")

_TIMEOUT_SECONDS = 10
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
    """Return the listing's primary image URL, or None if unavailable.

    Tries the ``og:image`` meta tag first (stable across most redesigns),
    then falls back to Amazon's embedded image-gallery JSON. Swallows all
    errors — this is a best-effort enhancement, never a hard requirement.
    """
    if not CONFIG.scrape_product_images:
        return None
    if not product_url:
        return None

    try:
        resp = requests.get(
            product_url,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        page = resp.text
    except requests.RequestException as exc:
        logger.warning("Product image scrape failed for %s: %s", product_url, exc)
        return None

    for pattern in (_OG_IMAGE_RE, _HIRES_RE, _LARGE_RE):
        match = pattern.search(page)
        if match:
            url = html.unescape(match.group(1))
            logger.info("Scraped product image for %s", product_url)
            return url

    logger.warning("No product image found on page: %s", product_url)
    return None
