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
    content_type: str = "",
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

    _draw_title(img, title, content_type)

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


MAX_TITLE_LINES = 3
# Scrim (gradient shade) never covers more than this fraction of the image,
# so text can't swallow the whole pin even on very long titles.
MAX_SCRIM_FRACTION = 0.42
ACCENT_BAR_HEIGHT = 10
ACCENT_COLORS = {
    "affiliate": (232, 122, 65),   # warm terracotta
    "organic": (92, 138, 108),     # sage green
}
DEFAULT_ACCENT = (210, 175, 90)    # muted gold


def _fit_title(
    draw: ImageDraw.ImageDraw, title: str, width: int
) -> Tuple[ImageFont.FreeTypeFont, list]:
    """Pick the largest font size that wraps ``title`` into <= MAX_TITLE_LINES.

    Starts bold and large and shrinks only as far as needed, so most titles
    render big and confident rather than defaulting to a small safe size.
    """
    max_w = int(width * 0.86)
    for font_size in range(int(width * 0.11), 27, -4):
        font = _load_font(font_size)
        avg_char_w = max(_text_size(draw, "M", font)[0] * 0.62, 1)
        max_chars = max(6, int(max_w / avg_char_w))
        lines = textwrap.wrap(title, width=max_chars) or [title]
        if len(lines) <= MAX_TITLE_LINES:
            # Re-wrap tightly using measured widths so long words don't clip.
            lines = _wrap_to_pixels(draw, title, font, max_w)
            if len(lines) <= MAX_TITLE_LINES:
                return font, lines
    # Fallback: smallest size, hard-wrapped, truncate extra lines.
    font = _load_font(28)
    lines = _wrap_to_pixels(draw, title, font, max_w)[:MAX_TITLE_LINES]
    return font, lines


def _wrap_to_pixels(draw, text: str, font, max_w: int) -> list:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _text_size(draw, candidate, font)[0] <= max_w or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_gradient_scrim(img: Image.Image, top: int, bottom: int) -> None:
    """Paint a smooth dark-to-transparent gradient band for text legibility."""
    height = bottom - top
    if height <= 0:
        return
    overlay = Image.new("RGBA", (img.width, height), (0, 0, 0, 0))
    grad = ImageDraw.Draw(overlay)
    for row in range(height):
        # Strong near the anchor edge (top of the band), fading out.
        t = row / max(height - 1, 1)
        alpha = int(200 * (1 - t) ** 1.6)
        grad.line([(0, row), (img.width, row)], fill=(10, 10, 12, alpha))
    img.paste(overlay, (0, top), overlay)


def _draw_title(img: Image.Image, title: str, content_type: str = "") -> None:
    """Draw a bold, legible title over a gradient scrim near the top of the image."""
    if not title:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    font, lines = _fit_title(draw, title, width)
    line_heights = [_text_size(draw, ln, font)[1] for ln in lines]
    line_gap = int(font.size * 0.3)
    total_text_h = sum(line_heights) + line_gap * (len(lines) - 1)

    pad_top = int(font.size * 0.55)
    pad_bottom = int(font.size * 0.75)
    band_top = ACCENT_BAR_HEIGHT
    band_bottom = min(
        band_top + pad_top + total_text_h + pad_bottom,
        int(height * MAX_SCRIM_FRACTION),
    )

    # Accent bar: a small flourish of colour so the pin doesn't read as flat.
    accent = ACCENT_COLORS.get(content_type, DEFAULT_ACCENT)
    draw.rectangle([(0, 0), (width, ACCENT_BAR_HEIGHT)], fill=(*accent, 255))

    _draw_gradient_scrim(img, band_top, band_bottom)

    y = band_top + pad_top
    stroke_w = max(2, font.size // 18)
    for ln, lh in zip(lines, line_heights):
        tw, _ = _text_size(draw, ln, font)
        x = (width - tw) // 2
        draw.text(
            (x, y), ln, font=font, fill=(255, 255, 255, 255),
            stroke_width=stroke_w, stroke_fill=(0, 0, 0, 235),
        )
        y += lh + line_gap
