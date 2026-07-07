"""Compositor — pure image processing (Pillow), NOT an LLM call.

Overlays the title text cleanly onto the generated image and validates the
final output: correct 2:3 aspect ratio and under Pinterest's file-size limit.

Returns the composited image path plus metadata used by the verifier.
"""

from __future__ import annotations

import logging
import math
import os
import random
import textwrap
from typing import Any, Dict, List, Optional, Tuple

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


_FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")


def _font_path(filename: str) -> str:
    return os.path.join(_FONTS_DIR, filename)


_DEFAULT_FONT_PATH = _font_path("DejaVuSans-Bold.ttf")


def _load_font(size: int, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back to the default bitmap font.

    Bundled fonts (assets/fonts/) are tried first so rendering is identical on
    Windows/macOS/Linux — relying solely on OS-installed font paths silently
    produced PIL's ~10px placeholder bitmap font on any machine without one of
    the hardcoded Linux/Mac paths (e.g. Windows), which is why text once
    rendered far smaller than intended.
    """
    candidates = [
        font_path,
        _DEFAULT_FONT_PATH,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\seguisb.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning(
        "No TrueType font found (bundled fonts missing?); using default bitmap "
        "font — text will render far smaller than intended."
    )
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
# The decorative band never covers more than this fraction of the image, so
# text can't swallow the whole pin even on very long titles.
MAX_BAND_FRACTION = 0.42
# Shapes with a wavy/bumpy/torn top edge extend up to this many pixels above
# their nominal top — text always starts below this so it never collides
# with the decorative edge regardless of which shape gets picked.
EDGE_MARGIN = 40

# Warm, fun, inviting palettes — background colour paired with a text colour
# chosen for contrast against it (not always white-on-dark).
PALETTES: List[Dict[str, Any]] = [
    {"name": "coral", "bg": (255, 111, 89), "text": (255, 250, 240)},
    {"name": "honey", "bg": (232, 163, 61), "text": (61, 38, 10)},
    {"name": "terracotta", "bg": (193, 101, 47), "text": (255, 248, 235)},
    {"name": "blush", "bg": (217, 138, 138), "text": (255, 250, 245)},
    {"name": "sage", "bg": (143, 168, 136), "text": (255, 253, 240)},
    {"name": "sunshine", "bg": (244, 185, 66), "text": (66, 40, 10)},
    {"name": "peach", "bg": (247, 181, 137), "text": (77, 40, 20)},
]

# Warm/fun typeface variety. stroke_ratio=0 means "use a soft drop shadow
# instead of an outline" — a thick uniform outline reads badly on script
# letterforms. size_scale compensates for typefaces whose glyphs read
# smaller/larger than DejaVu at the same point size. max_title_len keeps the
# script font out of the running for long, keyword-heavy titles where a
# cursive face would hurt legibility.
FONT_STYLES: List[Dict[str, Any]] = [
    {"file": _font_path("DejaVuSans-Bold.ttf"), "size_scale": 1.0, "stroke_ratio": 1 / 16, "max_title_len": None},
    {"file": _font_path("Comfortaa-Bold.ttf"), "size_scale": 0.85, "stroke_ratio": 1 / 20, "max_title_len": None},
    {"file": _font_path("Quicksand-Bold.ttf"), "size_scale": 0.95, "stroke_ratio": 1 / 18, "max_title_len": None},
    {"file": _font_path("DancingScript-Bold.otf"), "size_scale": 1.35, "stroke_ratio": 0, "max_title_len": 45},
]


def _fit_title(
    draw: ImageDraw.ImageDraw, title: str, width: int, font_path: str, size_scale: float
) -> Tuple[ImageFont.FreeTypeFont, list]:
    """Pick the largest font size that wraps ``title`` into <= MAX_TITLE_LINES.

    Starts bold and large and shrinks only as far as needed, so most titles
    render big and confident rather than defaulting to a small safe size.
    """
    max_w = int(width * 0.86)
    start = max(int(width * 0.11 * size_scale), 31)
    floor = max(int(27 * size_scale), 24)
    for font_size in range(start, floor, -4):
        font = _load_font(font_size, font_path)
        avg_char_w = max(_text_size(draw, "M", font)[0] * 0.62, 1)
        max_chars = max(6, int(max_w / avg_char_w))
        lines = textwrap.wrap(title, width=max_chars) or [title]
        if len(lines) <= MAX_TITLE_LINES:
            # Re-wrap tightly using measured widths so long words don't clip.
            lines = _wrap_to_pixels(draw, title, font, max_w)
            if len(lines) <= MAX_TITLE_LINES:
                return font, lines
    # Fallback: smallest size, hard-wrapped, truncate extra lines.
    font = _load_font(floor, font_path)
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


def _shape_solid_block(img: Image.Image, content_top: int, palette: Dict[str, Any]) -> None:
    """A clean rounded card, flush with the bottom edge."""
    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        [0, content_top, width - 1, height + 40],
        radius=34, corners=(True, True, False, False), fill=(*palette["bg"], 244),
    )
    img.paste(overlay, (0, 0), overlay)


def _shape_gradient_fade(img: Image.Image, content_top: int, palette: Dict[str, Any]) -> None:
    """A soft gradient, opaque at the bottom edge, fading upward."""
    width, height = img.size
    band_h = height - content_top
    if band_h <= 0:
        return
    overlay = Image.new("RGBA", (width, band_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for row in range(band_h):
        t = row / max(band_h - 1, 1)  # 0 at top of band, 1 at bottom edge
        alpha = int(235 * t ** 1.3)
        draw.line([(0, row), (width, row)], fill=(*palette["bg"], alpha))
    img.paste(overlay, (0, content_top), overlay)


def _shape_wave_top(img: Image.Image, content_top: int, palette: Dict[str, Any]) -> None:
    """A block with a smooth sine-wave top edge."""
    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    amplitude = EDGE_MARGIN * 0.65
    wavelength = width / 1.6
    points = [(0, height)]
    steps = 40
    for i in range(steps + 1):
        x = width * i / steps
        y = content_top + EDGE_MARGIN + amplitude * math.sin(2 * math.pi * x / wavelength)
        points.append((x, y))
    points.append((width, height))
    draw.polygon(points, fill=(*palette["bg"], 244))
    img.paste(overlay, (0, 0), overlay)


def _shape_torn_paper(img: Image.Image, content_top: int, palette: Dict[str, Any]) -> None:
    """A block with an irregular, hand-torn-looking top edge."""
    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [(0, height)]
    steps = 24
    for i in range(steps + 1):
        x = width * i / steps
        jitter = random.uniform(-EDGE_MARGIN * 0.5, EDGE_MARGIN * 0.5)
        y = content_top + EDGE_MARGIN + jitter
        points.append((x, y))
    points.append((width, height))
    draw.polygon(points, fill=(*palette["bg"], 244))
    img.paste(overlay, (0, 0), overlay)


def _shape_scalloped(img: Image.Image, content_top: int, palette: Dict[str, Any]) -> None:
    """A block with a row of rounded scalloped bumps along the top edge."""
    width, height = img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bump_r = EDGE_MARGIN * 0.55
    bump_y = content_top + EDGE_MARGIN
    draw.rectangle([0, bump_y, width, height], fill=(*palette["bg"], 244))
    n = max(3, round(width / (bump_r * 2)))
    spacing = width / n
    for i in range(n + 1):
        cx = spacing * i
        draw.ellipse(
            [cx - bump_r, bump_y - bump_r, cx + bump_r, bump_y + bump_r],
            fill=(*palette["bg"], 244),
        )
    img.paste(overlay, (0, 0), overlay)


SHAPE_STYLES = [_shape_solid_block, _shape_gradient_fade, _shape_wave_top, _shape_torn_paper, _shape_scalloped]


def _draw_title(img: Image.Image, title: str, content_type: str = "") -> None:
    """Draw a fun, warm, randomly-styled title band at the bottom of the image.

    Each call independently randomizes the palette, font, and background
    shape (solid card / gradient / wave / torn-paper / scalloped) so pins
    don't all look identical — this is deliberate variety, not a bug if two
    consecutive pins look different from each other.
    """
    if not title:
        return
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    palette = random.choice(PALETTES)
    eligible_fonts = [f for f in FONT_STYLES if f["max_title_len"] is None or len(title) <= f["max_title_len"]]
    font_style = random.choice(eligible_fonts)
    shape_fn = random.choice(SHAPE_STYLES)

    font, lines = _fit_title(draw, title, width, font_style["file"], font_style["size_scale"])
    line_heights = [_text_size(draw, ln, font)[1] for ln in lines]
    line_gap = int(font.size * 0.3)
    total_text_h = sum(line_heights) + line_gap * (len(lines) - 1)

    pad_top = int(font.size * 0.5)
    pad_bottom = int(font.size * 0.6)
    band_h = min(EDGE_MARGIN + pad_top + total_text_h + pad_bottom, int(height * MAX_BAND_FRACTION))
    content_top = height - band_h

    shape_fn(img, content_top, palette)

    y = content_top + EDGE_MARGIN + pad_top
    stroke_ratio = font_style["stroke_ratio"]
    text_color = (*palette["text"], 255)
    for ln, lh in zip(lines, line_heights):
        tw, _ = _text_size(draw, ln, font)
        x = (width - tw) // 2
        if stroke_ratio > 0:
            stroke_w = max(2, int(font.size * stroke_ratio))
            draw.text(
                (x, y), ln, font=font, fill=text_color,
                stroke_width=stroke_w, stroke_fill=(0, 0, 0, 235),
            )
        else:
            # Script faces read poorly with a uniform outline — use a soft
            # offset shadow instead, which preserves the letterforms.
            offset = max(2, font.size // 30)
            draw.text((x + offset, y + offset), ln, font=font, fill=(0, 0, 0, 110))
            draw.text((x, y), ln, font=font, fill=text_color)
        y += lh + line_gap
