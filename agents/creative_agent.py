"""Creative agent — builds an image-gen prompt and calls Higgsfield.

Requests a 1000x1500 (2:3) image. Affiliate mode produces product/lifestyle
context imagery; organic mode produces mood/aesthetic imagery for the niche.

Returns the path to the saved raw image plus its bytes.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import requests

from config import CONFIG, PIN_IMAGE_HEIGHT, PIN_IMAGE_WIDTH
from db import Database
from .clients import ApiError, HiggsfieldClient, OpenRouterClient

logger = logging.getLogger("affiliate_engine.creative")


# The compositor overlays a gradient scrim + stroke-outlined text directly on
# top of the image (see compositor.py) — it does NOT need pre-reserved blank
# space to stay legible. Asking the image model for "negative space" produces
# exactly the flat/empty-looking product-on-white shots we don't want, so
# these prompts explicitly forbid it and demand a full-bleed real scene.
AFFILIATE_SCENE_SYSTEM_PROMPT = (
    "You are an expert product photographer and creative director for Pinterest "
    "lifestyle content. You never describe plain studio product shots on white "
    "or plain colored backgrounds — you always imagine a specific, realistic "
    "scene that demonstrates the product being actively used or clearly doing "
    "its job in context.\n\n"
    "Two requirements must BOTH be true at once, and neither may compromise the "
    "other:\n"
    "1. THE PRODUCT MUST BE THE UNMISTAKABLE HERO OF THE SHOT. A viewer glancing "
    "at a small thumbnail for one second must instantly understand what the "
    "product is and what it does. It must be large, prominent, well-lit, and in "
    "sharp focus — the clear center of attention, not one item lost among many. "
    "Never bury it in background clutter or make it small relative to the frame.\n"
    "2. The image must NOT look like an advertisement or product catalog photo. "
    "It should look like an authentic, candid photo a real person took while "
    "genuinely using the thing — natural framing, real (but minimal and "
    "purposeful) context around it, phone-camera-like realism rather than "
    "studio lighting or a glossy commercial finish.\n\n"
    "Reconcile these by keeping the supporting scene SIMPLE: one or two "
    "purposeful context elements (e.g. the specific food being seasoned, the "
    "open fridge it's organizing), not a busy or cluttered tabletop full of "
    "unrelated objects — clutter that competes with the product for attention "
    "defeats the entire point. Avoid depending on small text/lettering being "
    "legible anywhere in the scene (text rendering in AI-generated images is "
    "unreliable and often garbled) — show the product's real printed design "
    "only if it's large and central; otherwise favor the action or the "
    "organized result over close-up readable text.\n\n"
    "Before writing the prompt, think about the specific person who would stop "
    "scrolling for this: what relatable moment or 'before/after' contrast would "
    "make them go 'wait, I need to see this' — while still leaving no doubt "
    "about what's being shown.\n\n"
    "Concrete examples of the target balance: a single labeled spice jar, large "
    "and in sharp focus, with a hand actively sprinkling from it onto food on a "
    "stovetop, a couple of other matching labeled jars softly visible nearby; "
    "storage bins as the clear main subject, shown full of food inside an open "
    "refrigerator with the door frame visible for context but nothing else "
    "competing for attention; a drawer organizer as the clear main subject, "
    "shown filled with neatly arranged utensils in an open drawer. Respond with "
    'a single JSON object: {"scene_prompt": "..."}. The scene_prompt must be a '
    "vivid, concrete image-generation prompt (not meta-commentary) for a "
    "photorealistic text-to-image model, vertical 2:3 format. It must "
    "explicitly state the product is the large, sharply-focused main subject, "
    "and must explicitly rule out: plain/white/studio backgrounds, floating "
    "isolated product shots on empty backgrounds, and anything that reads as a "
    "staged commercial/advertisement — but must equally rule out busy, "
    "cluttered, or distracting scenes that obscure the product."
)

ORGANIC_SCENE_SYSTEM_PROMPT = (
    "You are an expert photographer and creative director for Pinterest mood/"
    "aesthetic content. You never describe plain, empty, or minimalist studio "
    "compositions — you imagine a specific, richly detailed real-world scene "
    "that captures the feeling of the topic, full of authentic props and "
    "environment (not floating objects on a plain background). Respond with a "
    "single JSON object: {\"scene_prompt\": \"...\"}. The scene_prompt must be "
    "a vivid, concrete image-generation prompt (not meta-commentary) for a "
    "photorealistic text-to-image model, vertical 2:3 format, no text or "
    "logos in the image. It must explicitly rule out plain/empty backgrounds "
    "and negative space — the scene should fill the entire frame."
)


def _static_fallback_prompt(content_type: str, copy: Dict[str, Any], subject: Dict[str, Any]) -> str:
    """Used only if the scene-reasoning LLM call fails — degrade, don't crash."""
    keywords = ", ".join(copy.get("keywords", [])[:6])
    if content_type == "affiliate":
        title = subject.get("title", copy.get("title", ""))
        category = subject.get("category", CONFIG.primary_niche())
        features = ", ".join(subject.get("features", [])[:4])
        return (
            f"Candid, authentic-looking photo where '{title}' is the large, sharply "
            f"-focused main subject, actively being used in a real, lived-in "
            f"{category} setting — like a real person's phone photo, NOT a staged "
            f"advertisement or product catalog shot, NOT a white or empty "
            f"background, NOT professional studio lighting. Keep supporting context "
            f"minimal and purposeful (one or two relevant elements only) so the "
            f"product stays the unmistakable focal point — avoid busy or cluttered "
            f"scenes that compete for attention. Features to reflect: {features}. "
            f"Themes: {keywords}. Vertical 2:3 format, soft natural light."
        )
    niche = subject.get("niche") or CONFIG.primary_niche()
    return (
        f"Photorealistic, richly detailed lifestyle scene capturing the feeling of "
        f"'{niche}' — full of real, authentic props and environment, NOT a plain "
        f"or empty background. Full-bleed scene filling the entire frame. "
        f"Themes: {keywords}. Vertical 2:3 format, soft natural light, no text, no logos."
    )


def _build_scene_prompt_llm(
    content_type: str, copy: Dict[str, Any], subject: Dict[str, Any], db: Optional[Database]
) -> Optional[str]:
    """Ask an LLM to reason about a concrete in-use scene. None on any failure."""
    client = OpenRouterClient(db=db)
    if content_type == "affiliate":
        system_prompt = AFFILIATE_SCENE_SYSTEM_PROMPT
        user_prompt = (
            f"Product: {subject.get('title', copy.get('title', ''))}\n"
            f"Category/niche: {subject.get('category', CONFIG.primary_niche())}\n"
            f"Real product features: {subject.get('features', [])}\n"
            f"Pinterest copy keywords: {copy.get('keywords', [])}\n\n"
            "Describe the specific real-world in-use scene now."
        )
    else:
        system_prompt = ORGANIC_SCENE_SYSTEM_PROMPT
        user_prompt = (
            f"Niche: {subject.get('niche', CONFIG.primary_niche())}\n"
            f"Topic: {subject.get('topic', '')}\n"
            f"Pinterest copy keywords: {copy.get('keywords', [])}\n\n"
            "Describe the specific real-world scene now."
        )

    try:
        result = client.chat_json(
            model=CONFIG.creative_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.9,
        )
    except (ApiError, ValueError) as exc:
        logger.warning("Creative scene-prompt LLM call failed, using static fallback: %s", exc)
        return None

    scene_prompt = result.get("scene_prompt")
    if not scene_prompt or not isinstance(scene_prompt, str):
        logger.warning("Creative scene-prompt LLM returned no usable scene_prompt; using static fallback")
        return None
    return scene_prompt


def build_image_prompt(
    content_type: str, copy: Dict[str, Any], subject: Dict[str, Any], db: Optional[Database] = None
) -> str:
    """Compose an image-generation prompt, reasoning about a concrete in-use
    scene via an LLM call. Degrades to a static (but still in-use-oriented)
    prompt if that call fails, so a bad/missing key never blocks a cycle.
    """
    scene_prompt = _build_scene_prompt_llm(content_type, copy, subject, db)
    return scene_prompt or _static_fallback_prompt(content_type, copy, subject)


def _download_reference_image(url: str) -> Optional[bytes]:
    """Fetch a real product photo. Never raises; None on any failure so the
    caller can fall back to generation rather than blocking the cycle."""
    try:
        resp = requests.get(url, timeout=CONFIG.http_timeout_seconds)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        logger.warning("Failed to download reference image %s: %s", url, exc)
        return None


def _save_image(content_type: str, image_bytes: bytes, output_dir: Optional[str], tag: str) -> str:
    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"creative_{content_type}_{os.getpid()}_{abs(hash(tag)) % 10000}.png")
    try:
        with open(path, "wb") as fh:
            fh.write(image_bytes)
    except OSError as exc:
        raise ApiError(f"Failed to save image: {exc}") from exc
    return path


def generate_creative(
    content_type: str,
    copy: Dict[str, Any],
    subject: Dict[str, Any],
    db: Optional[Database] = None,
    output_dir: Optional[str] = None,
    mock: bool = False,
) -> Tuple[str, bytes]:
    """Generate the base image; return (path, bytes). Raises ``ApiError``.

    When a real product photo is available (``subject['image_url']``, affiliate
    only), it is used DIRECTLY as the base image rather than asking a
    generative model to recreate the product — no text-to-image or
    reference-guided generation can reliably preserve an exact product's
    shape, color, and label design, and showing a product that doesn't match
    what's actually sold at the link is a real misrepresentation risk, not
    just a style issue. AI generation is only used when no real photo exists.

    When ``mock`` is True and no real photo is available, a locally drawn
    1000x1500 placeholder is produced instead (for testing the pipeline
    without a Higgsfield key). A mock image is intended only for ``--dry-run``.
    """
    from db import get_db

    db = db or get_db()

    reference_image_url = subject.get("image_url") if content_type == "affiliate" else None
    if reference_image_url:
        image_bytes = _download_reference_image(reference_image_url)
        if image_bytes is not None:
            logger.info("Using real product photo directly (no AI generation): %s", reference_image_url)
            path = _save_image(content_type, image_bytes, output_dir, reference_image_url)
            return path, image_bytes
        logger.warning("Falling back to AI generation since the reference photo could not be fetched")

    prompt = build_image_prompt(content_type, copy, subject, db=db)
    logger.info("Image prompt: %s", prompt)

    if mock:
        image_bytes = _mock_image(content_type, copy, subject)
        logger.warning("MOCK image generated (no Higgsfield call) — dry-run only")
    else:
        client = HiggsfieldClient(db=db)
        image_bytes = client.generate_image(prompt, PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT)

    path = _save_image(content_type, image_bytes, output_dir, prompt)
    return path, image_bytes


def _mock_image(content_type: str, copy: Dict[str, Any], subject: Dict[str, Any]) -> bytes:
    """Draw a 1000x1500 placeholder so the compositor/verifier have a real image."""
    import io

    from PIL import Image, ImageDraw

    # Distinct background per mode so it's obvious at a glance.
    bg = (206, 214, 224) if content_type == "affiliate" else (214, 224, 210)
    img = Image.new("RGB", (PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT), bg)
    draw = ImageDraw.Draw(img)
    label = subject.get("title") or subject.get("topic") or content_type
    lines = [
        "MOCK IMAGE (no Higgsfield)",
        f"mode: {content_type}",
        str(label)[:40],
    ]
    y = PIN_IMAGE_HEIGHT // 2 - 40
    for ln in lines:
        draw.text((60, y), ln, fill=(60, 60, 60))
        y += 28
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
