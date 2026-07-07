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
from typing import List, Optional


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


@dataclass
class Config:
    # --- Niche / content strategy ---
    niches: List[str] = field(default_factory=lambda: _env_list("NICHES", ["home_organization"]))
    # Fraction of posts that should be affiliate (rest are organic value posts).
    target_affiliate_ratio: float = field(default_factory=lambda: _env_float("TARGET_AFFILIATE_RATIO", 0.4))
    daily_post_count: int = field(default_factory=lambda: _env_int("DAILY_POST_COUNT", 4))
    # Recent-history window used when computing the actual affiliate/organic ratio.
    ratio_lookback_posts: int = field(default_factory=lambda: _env_int("RATIO_LOOKBACK_POSTS", 20))
    # Do not reuse the same product/topic within this many days.
    reuse_lookback_days: int = field(default_factory=lambda: _env_int("REUSE_LOOKBACK_DAYS", 14))

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

    # --- Higgsfield (image generation) ---
    # NOTE: the public API is asynchronous — submit a job, then poll for it.
    # These defaults follow Higgsfield's documented REST contract; override via
    # env if your account/docs differ.
    higgsfield_api_key: Optional[str] = field(default_factory=lambda: _env_str("HIGGSFIELD_API_KEY"))
    higgsfield_base_url: str = field(
        default_factory=lambda: _env_str("HIGGSFIELD_BASE_URL", "https://api.higgsfield.ai")
    )
    higgsfield_model: str = field(default_factory=lambda: _env_str("HIGGSFIELD_MODEL", "flux"))
    higgsfield_steps: int = field(default_factory=lambda: _env_int("HIGGSFIELD_STEPS", 40))
    # Async job polling.
    higgsfield_poll_interval_seconds: float = field(
        default_factory=lambda: _env_float("HIGGSFIELD_POLL_INTERVAL_SECONDS", 3.0)
    )
    higgsfield_poll_timeout_seconds: int = field(
        default_factory=lambda: _env_int("HIGGSFIELD_POLL_TIMEOUT_SECONDS", 180)
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

    def validate_for_live_run(self) -> List[str]:
        """Return a list of missing config required for a real (non-dry-run) cycle.

        An empty list means the config is complete enough to post publicly.
        """
        missing: List[str] = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.higgsfield_api_key:
            missing.append("HIGGSFIELD_API_KEY")
        if not self.pinterest_access_token:
            missing.append("PINTEREST_ACCESS_TOKEN")
        if not self.pinterest_board_id:
            missing.append("PINTEREST_BOARD_ID")
        if not self.amazon_associates_tag:
            missing.append("AMAZON_ASSOCIATES_TAG")
        return missing


# Module-level singleton used throughout the codebase.
CONFIG = Config()
