"""Copywriter agent — generates Pinterest copy via an OpenRouter Claude model.

The system prompt differs by content_type:
- affiliate : product-anchored, SEO keyword-rich, MUST include the verbatim
              Amazon Associates disclosure and a correctly Associates-tagged
              link (no cloaking / no redirects).
- organic   : niche value content only — no product mention, no CTA, no link.

Output (validated):
    {title, description, keywords[], disclosure_text_or_null, link_or_null}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

from config import (
    CONFIG,
    PINTEREST_DESCRIPTION_MAX,
    PINTEREST_TITLE_MAX,
)
from db import Database
from .clients import ApiError, OpenRouterClient

logger = logging.getLogger("affiliate_engine.copywriter")


def build_associates_link(source_url: str, tag: str) -> str:
    """Return the product URL with the Associates ``tag`` query param set.

    This is a transparent, non-cloaked affiliate link: the destination is the
    real Amazon product page with the ``tag=`` parameter appended.
    """
    parsed = urlparse(source_url)
    query = dict(parse_qsl(parsed.query))
    query["tag"] = tag
    new_query = urlencode(query)
    return urlunparse(parsed._replace(query=new_query))


AFFILIATE_SYSTEM_PROMPT = (
    "You are an expert Pinterest affiliate copywriter for the niche '{niche}'. "
    "You write keyword-rich, Pinterest-SEO-optimized copy that is product-anchored "
    "and honest — never invent product features that are not supported by the "
    "provided product data.\n"
    "HARD RULES:\n"
    "1. The 'description' field MUST contain this disclosure text VERBATIM, "
    "exactly once: \"{disclosure}\"\n"
    "2. The 'link' field MUST be exactly this URL (do not alter, shorten, cloak, "
    "or redirect it): {link}\n"
    "3. 'title' must be <= {title_max} characters. 'description' must be <= "
    "{desc_max} characters (including the disclosure).\n"
    "4. Provide 5-10 SEO keywords relevant to the product and niche.\n"
    "Respond with a single JSON object and nothing else, with keys: "
    "title, description, keywords, disclosure_text_or_null, link_or_null."
)

ORGANIC_SYSTEM_PROMPT = (
    "You are an expert Pinterest content creator for the niche '{niche}'. "
    "You write high-value, inspirational or educational content for this niche.\n"
    "HARD RULES:\n"
    "1. Do NOT mention any specific product or brand.\n"
    "2. Do NOT include any call-to-action, sales language, or a link "
    "(no 'shop', 'buy', 'click', 'link in bio', etc.).\n"
    "3. 'disclosure_text_or_null' and 'link_or_null' MUST be null.\n"
    "4. 'title' must be <= {title_max} characters. 'description' must be <= "
    "{desc_max} characters.\n"
    "5. Provide 5-10 SEO keywords relevant to the niche.\n"
    "Respond with a single JSON object and nothing else, with keys: "
    "title, description, keywords, disclosure_text_or_null, link_or_null."
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def write_copy(
    content_type: str,
    subject: Dict[str, Any],
    angle: str = "",
    db: Optional[Database] = None,
    extra_context: str = "",
) -> Dict[str, Any]:
    """Generate and validate Pinterest copy.

    ``subject`` is the product dict (affiliate) or a topic dict (organic).
    Raises ``ApiError`` / ``ValueError`` on failure.
    """
    from db import get_db

    db = db or get_db()

    if content_type == "affiliate":
        return _write_affiliate(subject, angle, db, extra_context)
    elif content_type == "organic":
        return _write_organic(subject, angle, db, extra_context)
    raise ValueError(f"Unknown content_type: {content_type}")


def _write_affiliate(
    product: Dict[str, Any], angle: str, db: Database, extra_context: str
) -> Dict[str, Any]:
    niche = product.get("category") or CONFIG.primary_niche()
    link = build_associates_link(product["source_url"], CONFIG.amazon_associates_tag)
    disclosure = CONFIG.affiliate_disclosure_text

    system_prompt = AFFILIATE_SYSTEM_PROMPT.format(
        niche=niche,
        disclosure=disclosure,
        link=link,
        title_max=PINTEREST_TITLE_MAX,
        desc_max=PINTEREST_DESCRIPTION_MAX,
    )
    user_prompt = (
        f"Product data:\n{json.dumps(product, indent=2)}\n\n"
        f"Angle: {angle or 'general appeal'}\n\n"
    )
    if extra_context:
        user_prompt += f"Fix this problem from a previous attempt:\n{extra_context}\n\n"
    user_prompt += "Write the Pinterest pin copy now."

    result = _call(system_prompt, user_prompt, db)

    # Deterministically enforce the invariants regardless of what the LLM did.
    result["link"] = link
    result["link_or_null"] = link
    result.setdefault("disclosure_text_or_null", disclosure)
    if not result.get("disclosure_text_or_null"):
        result["disclosure_text_or_null"] = disclosure

    result["title"] = _truncate(str(result.get("title", "")), PINTEREST_TITLE_MAX)

    description = str(result.get("description", ""))
    if disclosure not in description:
        # Guarantee the verbatim disclosure is present.
        joiner = " " if description and not description.endswith((" ", "\n")) else ""
        description = f"{description}{joiner}{disclosure}"
    result["description"] = _truncate(description, PINTEREST_DESCRIPTION_MAX)
    # If truncation dropped the disclosure, prioritise the disclosure.
    if disclosure not in result["description"]:
        head_room = PINTEREST_DESCRIPTION_MAX - len(disclosure) - 1
        head = _truncate(description, max(head_room, 0)).rstrip()
        result["description"] = f"{head} {disclosure}".strip()[:PINTEREST_DESCRIPTION_MAX]

    result.setdefault("keywords", [])
    return result


def _write_organic(
    topic: Dict[str, Any], angle: str, db: Database, extra_context: str
) -> Dict[str, Any]:
    niche = topic.get("niche") or CONFIG.primary_niche()
    system_prompt = ORGANIC_SYSTEM_PROMPT.format(
        niche=niche,
        title_max=PINTEREST_TITLE_MAX,
        desc_max=PINTEREST_DESCRIPTION_MAX,
    )
    user_prompt = (
        f"Topic: {topic.get('topic', niche)}\n"
        f"Angle: {angle or 'inspirational value content'}\n\n"
    )
    if extra_context:
        user_prompt += f"Fix this problem from a previous attempt:\n{extra_context}\n\n"
    user_prompt += "Write the Pinterest pin copy now."

    result = _call(system_prompt, user_prompt, db)

    # Enforce organic invariants.
    result["disclosure_text_or_null"] = None
    result["link_or_null"] = None
    result["link"] = None
    result["title"] = _truncate(str(result.get("title", "")), PINTEREST_TITLE_MAX)
    result["description"] = _truncate(str(result.get("description", "")), PINTEREST_DESCRIPTION_MAX)
    result.setdefault("keywords", [])
    return result


def _call(system_prompt: str, user_prompt: str, db: Database) -> Dict[str, Any]:
    client = OpenRouterClient(db=db)
    result = client.chat_json(
        model=CONFIG.copywriter_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.8,
    )
    if not isinstance(result, dict):
        raise ValueError("Copywriter did not return a JSON object")
    return result
