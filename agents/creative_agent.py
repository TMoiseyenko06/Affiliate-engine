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
from .clients import ApiError, HiggsfieldClient

logger = logging.getLogger("affiliate_engine.creative")


def build_image_prompt(content_type: str, copy: Dict[str, Any], subject: Dict[str, Any]) -> str:
    """Compose an image-generation prompt from copy + subject data."""
    keywords = ", ".join(copy.get("keywords", [])[:6])
    if content_type == "affiliate":
        title = subject.get("title", copy.get("title", ""))
        category = subject.get("category", CONFIG.primary_niche())
        return (
            f"High-quality lifestyle product photography for Pinterest. "
            f"Subject: {title}. Context: {category}. "
            f"Bright, clean, aspirational styling with tasteful composition and "
            f"negative space at the top for a text overlay. "
            f"Themes: {keywords}. Vertical 2:3 format, photorealistic, soft natural light."
        )
    # organic
    niche = subject.get("niche") or CONFIG.primary_niche()
    return (
        f"Aesthetic mood photography for Pinterest in the '{niche}' niche. "
        f"Calm, inspirational, beautifully styled scene with negative space at the "
        f"top for a text overlay. Themes: {keywords}. "
        f"Vertical 2:3 format, photorealistic, soft natural light, no text, no logos."
    )


def generate_creative(
    content_type: str,
    copy: Dict[str, Any],
    subject: Dict[str, Any],
    db: Optional[Database] = None,
    output_dir: Optional[str] = None,
) -> Tuple[str, bytes]:
    """Generate the base image; return (path, bytes). Raises ``ApiError``."""
    from db import get_db

    db = db or get_db()
    prompt = build_image_prompt(content_type, copy, subject)
    logger.info("Image prompt: %s", prompt)

    client = HiggsfieldClient(db=db)
    image_bytes = client.generate_image(prompt, PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT)

    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"creative_{content_type}_{os.getpid()}_{abs(hash(prompt)) % 10000}.png")
    try:
        with open(path, "wb") as fh:
            fh.write(image_bytes)
    except OSError as exc:
        raise ApiError(f"Failed to save generated image: {exc}") from exc
    return path, image_bytes
