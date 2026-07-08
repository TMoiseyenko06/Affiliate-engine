"""Central configuration for the Pinterest affiliate pipeline.

All secrets are read from environment variables and are NEVER hardcoded.
Non-secret tunables (niche, ratios, budgets, lookback windows) have sane
defaults but can also be overridden via environment variables so the same
code can run in different deployments without edits.

Import ``CONFIG`` (a module-level singleton) everywhere else in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a ``.env`` file into ``os.environ``.

    Dependency-free and cross-platform (no need to ``source`` the file, which
    does not work on Windows PowerShell). Real environment variables always win:
    a value already set in the process environment is NOT overwritten, so you
    can still override the file with ``$env:FOO`` / ``export FOO``.

    Silently does nothing if the file is absent. Set ``DOTENV_PATH`` to point at
    a file elsewhere.
    """
    path = os.environ.get("DOTENV_PATH", path)
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                # Support an optional leading "export ".
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip matching surrounding quotes.
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # Never let config loading crash the process over a malformed file.
        pass


# Load .env before the CONFIG singleton reads the environment below.
_load_dotenv()


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Pinterest field limits (Pinterest API v5 / product constraints).
# Centralised so the copywriter prompt and the verifier agree on the numbers.
# ---------------------------------------------------------------------------
PINTEREST_TITLE_MAX = 100
PINTEREST_DESCRIPTION_MAX = 500
# Pinterest rejects images above 20 MB.
PINTEREST_MAX_IMAGE_BYTES = 20 * 1024 * 1024
PIN_IMAGE_WIDTH = 1000
PIN_IMAGE_HEIGHT = 1500  # 2:3 aspect ratio

# Content types that carry an Amazon Associates link/disclosure and share the
# same compliance rules — they differ only in whether the compositor draws a
# title band onto the image (see compositor.py / orchestrator.py).
AFFILIATE_CONTENT_TYPES = ("affiliate", "affiliate_image_only")
CONTENT_TYPES = ("affiliate", "affiliate_image_only", "organic")


@dataclass
class Config:
    # --- Niche / content strategy ---
    niches: List[str] = field(default_factory=lambda: _env_list("NICHES", ["home_organization"]))
    # Target mix across the three content types (affiliate with a title
    # overlay / affiliate image-only / organic). Normalized at use-time if
    # they don't sum to 1.0, so slightly-off values don't crash anything.
    target_affiliate_ratio: float = field(default_factory=lambda: _env_float("TARGET_AFFILIATE_RATIO", 1 / 3))
    target_affiliate_image_only_ratio: float = field(
        default_factory=lambda: _env_float("TARGET_AFFILIATE_IMAGE_ONLY_RATIO", 1 / 3)
    )
    target_organic_ratio: float = field(default_factory=lambda: _env_float("TARGET_ORGANIC_RATIO", 1 / 3))
    daily_post_count: int = field(default_factory=lambda: _env_int("DAILY_POST_COUNT", 4))
    # Recent-history window used when computing the actual affiliate/organic ratio.
    ratio_lookback_posts: int = field(default_factory=lambda: _env_int("RATIO_LOOKBACK_POSTS", 20))
    # "permanent": never re-post the same product once it's been posted
    # successfully (default — matches "don't post the same item twice").
    # "cooldown": the old behavior — a product becomes eligible again after
    # reuse_lookback_days. Only affects PRODUCT reuse (scout/verifier);
    # organic topic rotation is unaffected (its own small fixed topic list
    # would exhaust almost immediately under permanent exclusion).
    product_reuse_mode: str = field(default_factory=lambda: _env_str("PRODUCT_REUSE_MODE", "permanent"))
    # Cooldown window in days — used when product_reuse_mode="cooldown", and
    # always used for organic topic rotation regardless of product_reuse_mode.
    reuse_lookback_days: int = field(default_factory=lambda: _env_int("REUSE_LOOKBACK_DAYS", 14))
    # In permanent mode, if a niche's entire candidate pool has already been
    # posted, fall back to re-posting the least-recently-used one (and alert)
    # instead of stalling the pipeline entirely. Set false to hard-stop
    # instead (cycle is skipped/errors, same as the old behavior).
    product_reuse_fallback_enabled: bool = field(
        default_factory=lambda: _env_bool("PRODUCT_REUSE_FALLBACK_ENABLED", True)
    )

    # --- Amazon Associates ---
    amazon_associates_tag: str = field(default_factory=lambda: _env_str("AMAZON_ASSOCIATES_TAG", "example-20"))
    # Verbatim disclosure text required on every affiliate post.
    affiliate_disclosure_text: str = field(
        default_factory=lambda: _env_str(
            "AFFILIATE_DISCLOSURE_TEXT",
            "As an Amazon Associate I earn from qualifying purchases.",
        )
    )

    # --- OpenRouter (used by scout, copywriter, verifier) ---
    openrouter_api_key: Optional[str] = field(default_factory=lambda: _env_str("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(
        default_factory=lambda: _env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    # Cheap/fast model for scouting.
    scout_model: str = field(default_factory=lambda: _env_str("SCOUT_MODEL", "openai/gpt-4o-mini"))
    # Claude model for copywriting. NOTE: OpenRouter model slugs drift over time;
    # if a call 404s, check https://openrouter.ai/models for the current slug.
    copywriter_model: str = field(default_factory=lambda: _env_str("COPYWRITER_MODEL", "anthropic/claude-sonnet-5"))
    # DIFFERENT model for the verifier so the judgement is an independent check,
    # not the copywriter grading its own homework.
    verifier_model: str = field(default_factory=lambda: _env_str("VERIFIER_MODEL", "openai/gpt-4o-mini"))
    # Used by creative_agent to reason about a concrete in-use demonstration
    # scene (not a generic product-on-white prompt) before calling Higgsfield.
    creative_model: str = field(default_factory=lambda: _env_str("CREATIVE_MODEL", "anthropic/claude-sonnet-5"))
    # Image-editing model (via OpenRouter's Image API) used specifically when a
    # real product photo exists: places it into a generated in-use scene while
    # preserving the product's exact appearance, unlike Higgsfield's reference
    # mode which only guides style. Per https://openrouter.ai/docs/features/
    # multimodal/image-generation. Verified against OpenRouter's own docs.
    openrouter_image_model: str = field(
        default_factory=lambda: _env_str("OPENROUTER_IMAGE_MODEL", "google/gemini-3-pro-image")
    )

    # --- Higgsfield (image generation) ---
    # Per https://docs.higgsfield.ai (How to use API): async queue pattern —
    # POST {base}/{model_id} to submit, GET {base}/requests/{id}/status to poll.
    # Auth is a KEY + SECRET pair joined with a colon, NOT a bare bearer token:
    #   Authorization: Key {api_key}:{api_key_secret}
    higgsfield_api_key: Optional[str] = field(default_factory=lambda: _env_str("HIGGSFIELD_API_KEY"))
    higgsfield_api_secret: Optional[str] = field(default_factory=lambda: _env_str("HIGGSFIELD_API_SECRET"))
    higgsfield_base_url: str = field(
        default_factory=lambda: _env_str("HIGGSFIELD_BASE_URL", "https://platform.higgsfield.ai")
    )
    higgsfield_model_id: str = field(
        default_factory=lambda: _env_str("HIGGSFIELD_MODEL_ID", "higgsfield-ai/soul/standard")
    )
    # Model-quality tier; check the model's page in the Higgsfield dashboard for
    # the exact accepted values if this is rejected (varies per model_id).
    higgsfield_resolution: str = field(default_factory=lambda: _env_str("HIGGSFIELD_RESOLUTION", "1080p"))
    # Async job polling.
    higgsfield_poll_interval_seconds: float = field(
        default_factory=lambda: _env_float("HIGGSFIELD_POLL_INTERVAL_SECONDS", 3.0)
    )
    higgsfield_poll_timeout_seconds: int = field(
        default_factory=lambda: _env_int("HIGGSFIELD_POLL_TIMEOUT_SECONDS", 180)
    )
    # JSON body field name for a reference/input image in the submit payload.
    # UNVERIFIED against Higgsfield's own docs (their reference-image guide page
    # was unreachable while building this) — third-party mirrors of the Soul
    # model converge on "image". If real runs show this is wrong, override via
    # env; the client degrades gracefully (retries text-only) either way.
    higgsfield_image_param: str = field(default_factory=lambda: _env_str("HIGGSFIELD_IMAGE_PARAM", "image"))

    # --- Amazon product image sourcing (for affiliate creative reference) ---
    # CONFIRMED NON-VIABLE as of testing: plain HTTP scraping of Amazon listing
    # pages is blocked outright by Amazon's bot wall (opfcaptcha.amazon.com),
    # verified from two independent networks — every request returns a captcha
    # page, never the real listing. Disabled by default; only enable if you
    # have a working scraping strategy (e.g. a headless browser or a paid
    # scraping proxy service) — the long-term correct source is the official
    # Product Advertising API (PA-API 5.0), which requires an approved
    # Associates account and AWS-style request signing.
    scrape_product_images: bool = field(default_factory=lambda: _env_bool("SCRAPE_PRODUCT_IMAGES", False))
    # ScraperAPI (https://scraperapi.com) — third-party proxy/scraping service,
    # used because PA-API access requires an approved Associates account with
    # qualifying recent sales, which isn't available yet. Has a free tier
    # (1,000 credits). When set, this is tried BEFORE the direct-scrape
    # attempt above (which is confirmed blocked by Amazon's bot wall) via
    # ScraperAPI's dedicated structured Amazon product endpoint, which
    # returns real product JSON (images, title, features) rather than HTML
    # to parse. Swap this out for PA-API later with no caller changes needed.
    scraperapi_key: Optional[str] = field(default_factory=lambda: _env_str("SCRAPERAPI_KEY"))
    scraperapi_base_url: str = field(
        default_factory=lambda: _env_str("SCRAPERAPI_BASE_URL", "https://api.scraperapi.com")
    )
    scraperapi_country: str = field(default_factory=lambda: _env_str("SCRAPERAPI_COUNTRY", "us"))
    # Manual testing override: when set, this URL is used as the product
    # reference image whenever no scraped image is available. Lets you
    # exercise the compositor + Higgsfield reference-image flow with a real
    # photo without a working scraper. NOT meant for production — it would
    # apply the same static image to every affiliate product.
    test_product_image_url: Optional[str] = field(default_factory=lambda: _env_str("TEST_PRODUCT_IMAGE_URL"))
    # Manual testing override: when set to a real Amazon product URL, the
    # scout skips ranking/reuse-window logic entirely and always returns this
    # exact product, so you can repeatedly test one specific listing instead
    # of whichever candidate the ranking step happens to pick. NOT for
    # production — every affiliate cycle would promote the same product.
    test_force_product_url: Optional[str] = field(default_factory=lambda: _env_str("TEST_FORCE_PRODUCT_URL"))
    test_force_product_title: Optional[str] = field(
        default_factory=lambda: _env_str("TEST_FORCE_PRODUCT_TITLE")
    )

    # --- Pinterest API v5 ---
    pinterest_access_token: Optional[str] = field(default_factory=lambda: _env_str("PINTEREST_ACCESS_TOKEN"))
    pinterest_base_url: str = field(
        default_factory=lambda: _env_str("PINTEREST_BASE_URL", "https://api.pinterest.com/v5")
    )
    pinterest_board_id: Optional[str] = field(default_factory=lambda: _env_str("PINTEREST_BOARD_ID"))

    # --- Data source for Amazon product candidates (scout) ---
    product_source_url: Optional[str] = field(default_factory=lambda: _env_str("PRODUCT_SOURCE_URL"))

    # --- Storage ---
    # e.g. "sqlite:///affiliate_engine.db" or a postgres DSN.
    database_url: str = field(default_factory=lambda: _env_str("DATABASE_URL", "sqlite:///affiliate_engine.db"))

    # --- Budget / rate caps (per day) ---
    openrouter_daily_call_cap: int = field(default_factory=lambda: _env_int("OPENROUTER_DAILY_CALL_CAP", 200))
    higgsfield_daily_call_cap: int = field(default_factory=lambda: _env_int("HIGGSFIELD_DAILY_CALL_CAP", 20))

    # --- Alerting ---
    alert_file_path: str = field(default_factory=lambda: _env_str("ALERT_FILE_PATH", "alerts.log"))
    # Dead-man's-switch: alert if no successful post in this many hours.
    deadman_hours: int = field(default_factory=lambda: _env_int("DEADMAN_HOURS", 12))
    alert_webhook_url: Optional[str] = field(default_factory=lambda: _env_str("ALERT_WEBHOOK_URL"))

    # --- Networking ---
    http_timeout_seconds: int = field(default_factory=lambda: _env_int("HTTP_TIMEOUT_SECONDS", 60))

    def primary_niche(self) -> str:
        return self.niches[0] if self.niches else "general"

    def content_type_targets(self) -> Dict[str, float]:
        """Normalized target fraction for each of the three content types."""
        raw = {
            "affiliate": max(self.target_affiliate_ratio, 0.0),
            "affiliate_image_only": max(self.target_affiliate_image_only_ratio, 0.0),
            "organic": max(self.target_organic_ratio, 0.0),
        }
        total = sum(raw.values())
        if total <= 0:
            # Degenerate config (all zero/negative) — split evenly rather than divide by zero.
            return {k: 1 / 3 for k in raw}
        return {k: v / total for k, v in raw.items()}

    def validate_for_live_run(self) -> List[str]:
        """Return a list of missing config required for a real (non-dry-run) cycle.

        An empty list means the config is complete enough to post publicly.
        """
        missing: List[str] = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.higgsfield_api_key:
            missing.append("HIGGSFIELD_API_KEY")
        if not self.higgsfield_api_secret:
            missing.append("HIGGSFIELD_API_SECRET")
        if not self.pinterest_access_token:
            missing.append("PINTEREST_ACCESS_TOKEN")
        if not self.pinterest_board_id:
            missing.append("PINTEREST_BOARD_ID")
        if not self.amazon_associates_tag:
            missing.append("AMAZON_ASSOCIATES_TAG")
        return missing


# Module-level singleton used throughout the codebase.
CONFIG = Config()
