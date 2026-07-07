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
    "its job in context. Think concretely: who is using it (if anyone), where, "
    "doing what, with what other real props and environment around it. For "
    "example: labeled spice jars shown on a spice rack or in a cabinet next to "
    "other labeled jars and food, or a hand actively sprinkling spice from one "
    "onto food while cooking; storage bins shown full of food inside an open, "
    "organized refrigerator; a drawer organizer shown filled with neatly "
    "arranged utensils in an open drawer. Respond with a single JSON object: "
    '{"scene_prompt": "..."}. The scene_prompt must be a vivid, concrete '
    "image-generation prompt (not meta-commentary) for a photorealistic "
    "text-to-image model, vertical 2:3 format. It must explicitly rule out: "
    "plain/white/studio backgrounds, floating isolated product shots, empty "
    "negative space, and minimalist compositions. The scene should fill the "
    "entire frame with a real, lived-in environment."
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
            f"Photorealistic lifestyle photo showing '{title}' actively being used "
            f"in a real {category} setting, in context with other relevant real "
            f"objects around it — NOT a plain studio product shot, NOT a white or "
            f"empty background. Full-bleed scene filling the entire frame. "
            f"Features to reflect: {features}. Themes: {keywords}. "
            f"Vertical 2:3 format, soft natural light."
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


def generate_creative(
    content_type: str,
    copy: Dict[str, Any],
    subject: Dict[str, Any],
    db: Optional[Database] = None,
    output_dir: Optional[str] = None,
    mock: bool = False,
) -> Tuple[str, bytes]:
    """Generate the base image; return (path, bytes). Raises ``ApiError``.

    When ``mock`` is True, no image API is called: a locally drawn 1000x1500
    placeholder is produced instead (for testing the pipeline without a
    Higgsfield key). A mock image is intended only for ``--dry-run``.
    """
    from db import get_db

    db = db or get_db()
    prompt = build_image_prompt(content_type, copy, subject, db=db)
    logger.info("Image prompt: %s", prompt)

    if mock:
        image_bytes = _mock_image(content_type, copy, subject)
        logger.warning("MOCK image generated (no Higgsfield call) — dry-run only")
    else:
        client = HiggsfieldClient(db=db)
        # Affiliate posts anchor generation on the real product photo when one
        # was found (see amazon_scraper); organic has no product to reference.
        reference_image_url = subject.get("image_url") if content_type == "affiliate" else None
        image_bytes = client.generate_image(
            prompt, PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT, reference_image_url=reference_image_url
        )

    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"creative_{content_type}_{os.getpid()}_{abs(hash(prompt)) % 10000}.png")
    try:
        with open(path, "wb") as fh:
            fh.write(image_bytes)
    except OSError as exc:
        raise ApiError(f"Failed to save generated image: {exc}") from exc
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
