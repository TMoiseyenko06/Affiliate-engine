"""Compositor — pure image processing (Pillow), NOT an LLM call.

Overlays the title text cleanly onto the generated image and validates the
final output: correct 2:3 aspect ratio and under Pinterest's file-size limit.

Returns the composited image path plus metadata used by the verifier.
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from config import (
    PINTEREST_MAX_IMAGE_BYTES,
    PIN_IMAGE_HEIGHT,
    PIN_IMAGE_WIDTH,
)

logger = logging.getLogger("affiliate_engine.compositor")

ASPECT_RATIO = PIN_IMAGE_WIDTH / PIN_IMAGE_HEIGHT  # 2/3
ASPECT_TOLERANCE = 0.01


class CompositorError(Exception):
    pass


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back to the default bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning("No TrueType font found; using default bitmap font")
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def compose(
    base_image_path: str,
    title: str,
    output_path: Optional[str] = None,
    max_bytes: int = PINTEREST_MAX_IMAGE_BYTES,
) -> Dict[str, Any]:
    """Overlay ``title`` on the base image and save the result.

    Returns metadata: {path, width, height, aspect_ratio, size_bytes,
    aspect_ok, size_ok}. Raises ``CompositorError`` on unrecoverable failure.
    """
    try:
        img = Image.open(base_image_path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise CompositorError(f"Cannot open base image: {exc}") from exc

    # Normalise to the target pin dimensions so aspect ratio is guaranteed.
    if (img.width, img.height) != (PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT):
        img = _cover_resize(img, PIN_IMAGE_WIDTH, PIN_IMAGE_HEIGHT)

    _draw_title(img, title)

    output_path = output_path or (os.path.splitext(base_image_path)[0] + "_final.jpg")
    quality = 90
    # Save as JPEG, stepping quality down until under the size cap.
    while True:
        img.save(output_path, format="JPEG", quality=quality, optimize=True)
        size_bytes = os.path.getsize(output_path)
        if size_bytes <= max_bytes or quality <= 40:
            break
        quality -= 10

    aspect = img.width / img.height
    meta = {
        "path": output_path,
        "width": img.width,
        "height": img.height,
        "aspect_ratio": aspect,
        "size_bytes": size_bytes,
        "aspect_ok": abs(aspect - ASPECT_RATIO) <= ASPECT_TOLERANCE,
        "size_ok": size_bytes <= max_bytes,
    }
    if not meta["aspect_ok"]:
        logger.error("Composited image has wrong aspect ratio: %s", aspect)
    if not meta["size_ok"]:
        logger.error("Composited image exceeds size cap: %s bytes", size_bytes)
    return meta


def _cover_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize + center-crop to exactly fill target dimensions (cover)."""
    src_ratio = img.width / img.height
    target_ratio = target_w / target_h
    if src_ratio > target_ratio:
        # Source is wider: match height, crop width.
        new_h = target_h
        new_w = int(round(target_h * src_ratio))
    else:
        new_w = target_w
        new_h = int(round(target_w / src_ratio))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _draw_title(img: Image.Image, title: str) -> None:
    """Draw a legible, wrapped title band near the top of the image."""
    if not title:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # Choose a font size relative to image width; wrap to fit.
    font_size = max(36, width // 14)
    font = _load_font(font_size)

    # Wrap text to roughly the image width.
    avg_char_w = max(_text_size(draw, "M", font)[0], 1)
    max_chars = max(8, int((width * 0.85) / avg_char_w))
    lines = textwrap.wrap(title, width=max_chars) or [title]

    line_heights = [_text_size(draw, ln, font)[1] for ln in lines]
    line_gap = int(font_size * 0.25)
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)

    pad = int(font_size * 0.4)
    band_top = int(height * 0.05)
    band_bottom = band_top + total_h + pad * 2

    # Semi-transparent band for legibility over any background.
    draw.rectangle([(0, band_top), (width, band_bottom)], fill=(0, 0, 0, 130))

    y = band_top + pad
    for ln, lh in zip(lines, line_heights):
        tw, _ = _text_size(draw, ln, font)
        x = (width - tw) // 2
        # Simple shadow + white fill for contrast.
        draw.text((x + 2, y + 2), ln, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
        y += lh + line_gap
