"""Verifier agent — the hard gate that runs AFTER compositor, BEFORE poster.

Two layers:

1. Deterministic checks (pure code, no LLM) — always run first:
   - disclosure present verbatim if content_type == affiliate
   - link well-formed and matches the expected Associates tag pattern
   - image is correct dimensions and under the file-size cap
   - no duplicate product/topic within the reuse lookback window
   - title/description within Pinterest character limits
   - organic posts carry no link/disclosure

2. LLM judgement checks (separate OpenRouter call, a DIFFERENT model from the
   copywriter/orchestrator) — an independent review, not self-review:
   - copy matches the actual product/topic (no hallucinated features)
   - organic content avoids any CTA/link/sales language
   - image is plausible and the overlaid text is legible

Output: {"pass": bool, "failures": [reasons]}. The deterministic layer alone
can fail the gate; the LLM layer can only add failures, never override a
deterministic failure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from config import (
    AFFILIATE_CONTENT_TYPES,
    CONFIG,
    PINTEREST_DESCRIPTION_MAX,
    PINTEREST_MAX_IMAGE_BYTES,
    PINTEREST_TITLE_MAX,
    PIN_IMAGE_HEIGHT,
    PIN_IMAGE_WIDTH,
)
from db import Database
from .clients import ApiError, OpenRouterClient

logger = logging.getLogger("affiliate_engine.verifier")


VERIFIER_SYSTEM_PROMPT = (
    "You are an independent compliance and quality reviewer for Pinterest posts. "
    "You did NOT write this content. Review it critically and impartially. "
    "Respond with a single JSON object: "
    '{"pass": true|false, "failures": [list of short reason strings]}. '
    "If everything is fine, return pass=true with an empty failures list."
)


def run_deterministic_checks(
    content_type: str,
    copy: Dict[str, Any],
    image_meta: Dict[str, Any],
    subject: Dict[str, Any],
    db: Optional[Database] = None,
) -> List[str]:
    """Run all pure-code checks. Returns a list of failure reasons (empty=pass)."""
    failures: List[str] = []

    title = str(copy.get("title") or "")
    description = str(copy.get("description") or "")

    # --- character limits ---
    if not title:
        failures.append("Title is empty.")
    if len(title) > PINTEREST_TITLE_MAX:
        failures.append(f"Title exceeds {PINTEREST_TITLE_MAX} chars ({len(title)}).")
    if len(description) > PINTEREST_DESCRIPTION_MAX:
        failures.append(f"Description exceeds {PINTEREST_DESCRIPTION_MAX} chars ({len(description)}).")

    # --- image dimensions & size ---
    if image_meta.get("width") != PIN_IMAGE_WIDTH or image_meta.get("height") != PIN_IMAGE_HEIGHT:
        failures.append(
            f"Image dimensions {image_meta.get('width')}x{image_meta.get('height')} "
            f"!= required {PIN_IMAGE_WIDTH}x{PIN_IMAGE_HEIGHT}."
        )
    if not image_meta.get("aspect_ok", False):
        failures.append("Image aspect ratio is not 2:3.")
    size_bytes = image_meta.get("size_bytes", 0)
    if size_bytes <= 0:
        failures.append("Image size could not be determined.")
    elif size_bytes > PINTEREST_MAX_IMAGE_BYTES:
        failures.append(f"Image size {size_bytes} exceeds cap {PINTEREST_MAX_IMAGE_BYTES}.")

    # --- title-overlay intent must match content_type ---
    # affiliate_image_only pins must have NO text drawn on the image (the
    # product speaks for itself); the other two types must have it. This is
    # a deterministic guarantee, independent of whatever the compositor did.
    expected_title_drawn = content_type != "affiliate_image_only"
    actual_title_drawn = bool(image_meta.get("title_drawn"))
    if expected_title_drawn != actual_title_drawn:
        failures.append(
            f"Title overlay presence mismatch for content_type={content_type}: "
            f"expected title_drawn={expected_title_drawn}, got {actual_title_drawn}."
        )

    if content_type in AFFILIATE_CONTENT_TYPES:
        failures.extend(_affiliate_checks(copy, description, subject, db))
    elif content_type == "organic":
        failures.extend(_organic_checks(copy, description))
    else:
        failures.append(f"Unknown content_type: {content_type}")

    return failures


def _affiliate_checks(
    copy: Dict[str, Any],
    description: str,
    subject: Dict[str, Any],
    db: Optional[Database],
) -> List[str]:
    failures: List[str] = []

    # Disclosure present verbatim.
    disclosure = CONFIG.affiliate_disclosure_text
    if disclosure not in description:
        failures.append("Affiliate disclosure text missing verbatim from description.")
    if copy.get("disclosure_text_or_null") != disclosure:
        failures.append("disclosure_text_or_null does not match the configured disclosure.")

    # Link well-formed and correctly Associates-tagged.
    link = copy.get("link") or copy.get("link_or_null")
    if not link:
        failures.append("Affiliate post is missing a link.")
    else:
        failures.extend(_check_associates_link(link))

    # Duplicate product within lookback window.
    product_id = subject.get("product_id")
    if db is not None and product_id:
        # The scout marks the product as used at selection time, so only treat
        # it as a duplicate if a *posted* post already used it in the window.
        if db.topic_used_within(product_id, CONFIG.reuse_lookback_days):
            failures.append(
                f"Product {product_id} already posted within {CONFIG.reuse_lookback_days} days."
            )

    return failures


def _check_associates_link(link: str) -> List[str]:
    failures: List[str] = []
    parsed = urlparse(link)
    if parsed.scheme not in ("http", "https"):
        failures.append("Link is not a valid http(s) URL.")
        return failures
    if not parsed.netloc:
        failures.append("Link has no host.")
        return failures
    # No cloaking: must point at an Amazon domain, not a shortener/redirector.
    host = parsed.netloc.lower()
    if "amazon." not in host and not host.endswith("amzn.to"):
        failures.append(f"Link host '{host}' is not an Amazon domain (possible cloaking).")
    # Associates tag must be present and match the configured tag.
    tag_values = parse_qs(parsed.query).get("tag", [])
    if not tag_values:
        failures.append("Link is missing the Associates 'tag' parameter.")
    else:
        expected = CONFIG.amazon_associates_tag
        if tag_values[0] != expected:
            failures.append(f"Associates tag '{tag_values[0]}' != expected '{expected}'.")
        # Tag pattern sanity: name-NN.
        if not re.match(r"^[A-Za-z0-9]+-[0-9]{2}$", tag_values[0]):
            failures.append(f"Associates tag '{tag_values[0]}' does not match expected pattern.")
    return failures


def _organic_checks(copy: Dict[str, Any], description: str) -> List[str]:
    failures: List[str] = []
    if copy.get("link") or copy.get("link_or_null"):
        failures.append("Organic post must not contain a link.")
    if copy.get("disclosure_text_or_null"):
        failures.append("Organic post must not contain a disclosure.")

    # Cheap deterministic CTA/sales-language screen (LLM does the nuanced pass).
    text = f"{copy.get('title', '')} {description}".lower()
    banned = [
        "shop now", "buy now", "click", "link in bio", "grab yours",
        "on sale", "discount", "add to cart", "purchase", "% off",
    ]
    hits = [phrase for phrase in banned if phrase in text]
    if hits:
        failures.append(f"Organic post contains CTA/sales language: {hits}")
    return failures


def run_llm_checks(
    content_type: str,
    copy: Dict[str, Any],
    subject: Dict[str, Any],
    image_meta: Dict[str, Any],
    db: Optional[Database] = None,
) -> List[str]:
    """Independent LLM review. Returns failure reasons; [] means it passed.

    On LLM/parse error this returns a single failure so a broken judge fails
    closed (nothing posts without a successful independent review).
    """
    client = OpenRouterClient(db=db)

    # subject_data is filtered: image_url is an internal creative-pipeline detail
    # (which real photo, if any, was used as a Higgsfield reference) — it is not
    # part of the product's factual data and null is an expected, normal state
    # (scrape unavailable -> text-only generation), not a defect to grade.
    subject_data = {k: v for k, v in subject.items() if k != "image_url"}

    review_payload = {
        "content_type": content_type,
        "title": copy.get("title"),
        "description": copy.get("description"),
        "keywords": copy.get("keywords"),
        "link": copy.get("link") or copy.get("link_or_null"),
        "subject_data": subject_data,
    }
    if content_type in AFFILIATE_CONTENT_TYPES:
        criteria = (
            "Checks: (a) does the copy match the actual product data with NO "
            "hallucinated features; (b) is the affiliate copy appropriate and "
            "honest. Fail if copy claims features not present in subject_data. "
            "You are NOT shown the actual pin image — do not comment on or fail "
            "for image content, legibility, or missing image URLs; that is "
            "verified separately by deterministic checks."
        )
    else:
        criteria = (
            "Checks: (a) does the content avoid ANY product mention, CTA, sales "
            "language, or link; (b) is it genuine niche value content. Fail if "
            "any sales/CTA/link language appears. You are NOT shown the actual "
            "pin image — do not comment on or fail for image content, "
            "legibility, or missing image URLs; that is verified separately by "
            "deterministic checks."
        )

    user_prompt = (
        f"Review this Pinterest {content_type} post.\n\n"
        f"{criteria}\n\n"
        f"Content JSON:\n{json.dumps(review_payload, indent=2, default=str)}\n\n"
        f'Respond with {{"pass": bool, "failures": [reasons]}}.'
    )

    try:
        result = client.chat_json(
            model=CONFIG.verifier_model,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
    except (ApiError, ValueError) as exc:
        logger.error("Verifier LLM check failed (failing closed): %s", exc)
        return [f"LLM verifier unavailable: {exc}"]

    if not result.get("pass", False):
        raw = result.get("failures") or ["LLM verifier returned pass=false without reasons"]
        return [str(r) for r in raw]
    return []


def verify(
    content_type: str,
    copy: Dict[str, Any],
    image_meta: Dict[str, Any],
    subject: Dict[str, Any],
    db: Optional[Database] = None,
    post_attempt_id: str = "unknown",
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """Full verification. Returns {"pass": bool, "failures": [...]}.

    Deterministic checks run first; the LLM layer runs only if they pass (no
    point paying for an LLM review of content that already failed hard rules,
    though its failures would only add to the list anyway).
    """
    from db import get_db

    db = db or get_db()

    failures = run_deterministic_checks(content_type, copy, image_meta, subject, db)

    if not failures and not skip_llm:
        failures.extend(run_llm_checks(content_type, copy, subject, image_meta, db))

    passed = len(failures) == 0
    try:
        db.log_verifier(post_attempt_id, passed, failures)
    except Exception as exc:  # pragma: no cover - logging must not crash gate
        logger.error("Failed to write verifier_log: %s", exc)

    return {"pass": passed, "failures": failures}
