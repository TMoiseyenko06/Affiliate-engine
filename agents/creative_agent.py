"""Creative agent — produces the base 1000x1500 (2:3) pin image.

Affiliate mode, when a real product photo is available, edits that photo
into a generated in-use scene (OpenRouter image-editing model) while
preserving the product's exact appearance, falling back to the unedited
photo if editing fails, and to full Higgsfield text-to-image generation only
when no real photo exists at all. Organic mode always uses Higgsfield
text-to-image for mood/aesthetic imagery.

Returns the path to the saved image plus its bytes.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import requests

from config import CONFIG, PIN_IMAGE_HEIGHT, PIN_IMAGE_WIDTH
from db import Database
from .clients import ApiError, BudgetError, HiggsfieldClient, OpenRouterClient

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
    "Three requirements must ALL be true at once, and none may compromise the "
    "others:\n"
    "1. THE PRODUCT MUST BE THE UNMISTAKABLE HERO OF THE SHOT. A viewer glancing "
    "at a small thumbnail for one second must instantly understand what the "
    "product is and what it does. It must be large, prominent, well-lit, and in "
    "sharp focus — the clear center of attention, not one item lost among many. "
    "Never bury it in background clutter or make it small relative to the frame.\n"
    "2. The image must NOT look like an advertisement or product catalog photo. "
    "It should look like an authentic, candid photo a real person took while "
    "genuinely using the thing — natural framing, real (but minimal and "
    "purposeful) context around it, phone-camera-like realism rather than "
    "studio lighting or a glossy commercial finish.\n"
    "3. THE SCENE MUST MATCH THE EXACT NARRATIVE IN THE PROVIDED TITLE/"
    "DESCRIPTION, not a different plausible use-case you invent yourself. The "
    "copy has already committed to a specific pain point, moment, or angle — "
    "the image is illustrating THAT story. If the copy is about a parent "
    "digging through bins for a missing toy, show that search, not an "
    "unrelated tidy-closet scene; if it's about labeling spice jars while "
    "cooking, show that, not a pantry restock. Inventing a different scenario "
    "than the copy — even a plausible one — breaks the pin.\n\n"
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
    "environment (not floating objects on a plain background). The scene must "
    "match the exact moment/feeling in the provided title and description, "
    "not a different plausible scene you invent yourself — the copy has "
    "already committed to a specific angle and the image illustrates that "
    "same angle. Respond with a single JSON object: {\"scene_prompt\": "
    "\"...\"}. The scene_prompt must be a vivid, concrete image-generation "
    "prompt (not meta-commentary) for a photorealistic text-to-image model, "
    "vertical 2:3 format, no text or logos in the image. It must explicitly "
    "rule out plain/empty backgrounds and negative space — the scene should "
    "fill the entire frame."
)


def _static_fallback_prompt(content_type: str, copy: Dict[str, Any], subject: Dict[str, Any]) -> str:
    """Used only if the scene-reasoning LLM call fails — degrade, don't crash.

    Leads with the copy's own title (the specific pain-point/narrative angle
    it already committed to) rather than the generic product title, so even
    this degraded fallback illustrates the same story the copy tells instead
    of a generic, possibly-unrelated use-case.
    """
    keywords = ", ".join(copy.get("keywords", [])[:6])
    pain_point = copy.get("title", "")
    if content_type == "affiliate":
        product_title = subject.get("title", "")
        category = subject.get("category", CONFIG.primary_niche())
        features = ", ".join(subject.get("features", [])[:4])
        return (
            f"Candid, authentic-looking photo illustrating this exact scenario: "
            f"'{pain_point}'. The product shown, '{product_title}', must be the "
            f"large, sharply-focused main subject, actively being used in a real, "
            f"lived-in {category} setting — like a real person's phone photo, NOT "
            f"a staged advertisement or product catalog shot, NOT a white or "
            f"empty background, NOT professional studio lighting. Keep supporting "
            f"context minimal and purposeful (one or two relevant elements only) "
            f"so the product stays the unmistakable focal point — avoid busy or "
            f"cluttered scenes that compete for attention. Features to reflect: "
            f"{features}. Themes: {keywords}. Vertical 2:3 format, soft natural light."
        )
    niche = subject.get("niche") or CONFIG.primary_niche()
    return (
        f"Photorealistic, richly detailed lifestyle scene illustrating this exact "
        f"feeling/moment: '{pain_point}', in the '{niche}' niche — full of real, "
        f"authentic props and environment, NOT a plain or empty background. "
        f"Full-bleed scene filling the entire frame. Themes: {keywords}. "
        f"Vertical 2:3 format, soft natural light, no text, no logos."
    )


def _build_scene_prompt_llm(
    content_type: str, copy: Dict[str, Any], subject: Dict[str, Any], db: Optional[Database]
) -> Optional[str]:
    """Ask an LLM to reason about a concrete in-use scene. None on any failure.

    The copywriter runs BEFORE creative in the pipeline and has already
    invented a specific angle/pain-point/narrative (the title and
    description) — this call must illustrate THAT exact narrative, not
    independently invent a different one from the raw product facts. Passing
    only generic product data + a keyword list (the previous behavior) let
    the scene-writer pick its own unrelated scenario (e.g. copy about a
    "missing Lego piece" pain point, image showing folded clothes) since it
    never saw what story the copy actually told.

    This call never sees the actual product photo — it only has text data.
    When a real reference photo IS available (see _edit_reference_image), the
    scene description must stay silent on the product's own visual specifics
    (shape, exact label style, etc.): if it confidently asserts a
    plausible-but-wrong detail, that text can win out over the actual
    reference image during editing, producing a product that doesn't match
    what's really being sold.
    """
    client = OpenRouterClient(db=db)
    if content_type == "affiliate":
        system_prompt = AFFILIATE_SCENE_SYSTEM_PROMPT
        has_reference_photo = bool(subject.get("image_url"))
        user_prompt = (
            "The Pinterest copy for this pin has ALREADY been written. The "
            "image must depict the SAME specific moment, pain point, or "
            "scenario this copy describes — do not invent a different "
            "use-case or story than what's written below.\n"
            f"Title: {copy.get('title', '')}\n"
            f"Description: {copy.get('description', '')}\n\n"
            f"Product: {subject.get('title', '')}\n"
            f"Category/niche: {subject.get('category', CONFIG.primary_niche())}\n"
            f"Real product features: {subject.get('features', [])}\n\n"
        )
        if has_reference_photo:
            user_prompt += (
                "IMPORTANT: A real photo of the exact product will be attached "
                "separately when this scene is generated — you have NOT seen it. "
                "Do NOT invent, guess, or assert any specific visual detail of "
                "the product itself (its shape, size, color, label design, "
                "material, or exact appearance) — you will very likely guess "
                "wrong and contradict the real photo. Describe ONLY the "
                "surrounding context, setting, and action (e.g. where it is, "
                "what's happening, what else is nearby), and refer to the "
                "product generically (e.g. 'the labeled jar', 'the container') "
                "without describing what it looks like.\n\n"
            )
        user_prompt += (
            "Describe the specific real-world in-use scene now, matching the "
            "exact moment/scenario the title and description above convey."
        )
    else:
        system_prompt = ORGANIC_SCENE_SYSTEM_PROMPT
        user_prompt = (
            "The Pinterest copy for this pin has ALREADY been written. The "
            "image must depict the SAME specific feeling/moment this copy "
            "conveys — do not invent an unrelated scene.\n"
            f"Title: {copy.get('title', '')}\n"
            f"Description: {copy.get('description', '')}\n\n"
            f"Niche: {subject.get('niche', CONFIG.primary_niche())}\n"
            f"Topic: {subject.get('topic', '')}\n\n"
            "Describe the specific real-world scene now, matching the exact "
            "moment/feeling the title and description above convey."
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

    When a real product photo is available (``subject['image_url']``,
    affiliate only), priority order is:
      1. Edit it into a generated in-use scene via an image-editing model
         (OpenRouter, e.g. google/gemini-3-pro-image) that takes the real
         photo as a reference and is designed to preserve the subject while
         placing it in a new context.
      2. If that fails (missing key, budget cap, API error), fall back to
         using the real photo DIRECTLY, unedited — this guarantees exact
         product fidelity, which matters more than having a generated scene:
         showing a product that doesn't match what's actually sold at the
         link is a real misrepresentation risk, not just a style issue.
      3. Only when no real photo exists at all does this fall back to full
         text-to-image generation (Higgsfield), which cannot guarantee the
         product's appearance is accurate.

    When ``mock`` is True, NO image-generation API is called (neither
    Higgsfield nor the OpenRouter image-editing call) — for testing without
    any image-API key. If a real reference photo is available it's still
    downloaded and used directly, unedited (a plain HTTP GET, not generation,
    and the best available stand-in for testing); otherwise a locally drawn
    1000x1500 placeholder is produced. A mock image is intended only for
    ``--dry-run``.
    """
    from db import get_db

    db = db or get_db()

    reference_image_url = subject.get("image_url") if content_type == "affiliate" else None
    if reference_image_url:
        if not mock:
            edited = _edit_reference_image(content_type, copy, subject, reference_image_url, db)
            if edited is not None:
                path = _save_image(content_type, edited, output_dir, reference_image_url + "-edited")
                return path, edited

        image_bytes = _download_reference_image(reference_image_url)
        if image_bytes is not None:
            logger.info(
                "Using real product photo directly%s: %s",
                " (mock mode: no AI edit)" if mock else " (unedited)",
                reference_image_url,
            )
            path = _save_image(content_type, image_bytes, output_dir, reference_image_url)
            return path, image_bytes
        logger.warning("Reference photo unusable (download failed); falling back to generation")

    prompt = build_image_prompt(content_type, copy, subject, db=db)
    logger.info("Image prompt: %s", prompt)

    if mock:
        image_bytes = _mock_image(content_type, copy, subject)
        logger.warning("MOCK image generated (no image-generation API call) — dry-run only")
    else:
        client = HiggsfieldClient(db=db)
        image_bytes = client.generate_image(prompt, PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT)

    path = _save_image(content_type, image_bytes, output_dir, prompt)
    return path, image_bytes


def _edit_reference_image(
    content_type: str,
    copy: Dict[str, Any],
    subject: Dict[str, Any],
    reference_image_url: str,
    db: Optional[Database],
) -> Optional[bytes]:
    """Place the real reference photo into a generated in-use scene while
    preserving its exact appearance. Returns None on any failure — this is
    an enhancement, never a hard requirement (see generate_creative's
    fallback chain)."""
    scene_prompt = _build_scene_prompt_llm(content_type, copy, subject, db)
    if not scene_prompt:
        return None

    edit_prompt = (
        f"{scene_prompt}\n\n"
        "CRITICAL — READ THE REFERENCE IMAGE CAREFULLY BEFORE GENERATING: the "
        "attached reference image shows the exact real product being "
        "advertised. Before generating anything, look closely at its actual "
        "physical form in the reference image — its precise SHAPE (e.g. round "
        "vs. square vs. oval vs. rectangular — copy the exact shape shown, do "
        "not substitute a different, more common, or more 'typical' shape for "
        "this type of product), size, proportions, color(s), material, label/"
        "packaging design, and any visible text or logo. The product in the "
        "generated image must be visually IDENTICAL to the reference image in "
        "every one of these respects — not a redesigned, stylized, or "
        "'improved' reinterpretation of it, and not a generic/stereotypical "
        "version of this product category. If the reference shows round "
        "labels, the output must show round labels; if it shows square "
        "labels, the output must show square labels — copy exactly what is "
        "shown, do not default to what a typical product like this usually "
        "looks like. Only the surrounding scene, background, and context may "
        "be new; the product itself must be an exact, unaltered match to the "
        "reference image."
    )
    try:
        client = OpenRouterClient(db=db)
        return client.generate_image(
            model=CONFIG.openrouter_image_model,
            prompt=edit_prompt,
            aspect_ratio="2:3",
            reference_image_url=reference_image_url,
        )
    except (ApiError, BudgetError, ValueError) as exc:
        logger.warning(
            "Image-editing model failed to place reference photo in a scene, "
            "falling back to the unedited photo: %s", exc,
        )
        return None


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
