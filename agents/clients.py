"""Shared HTTP clients and JSON helpers for the pipeline.

Contains:
- ``extract_json`` : strip markdown fences and parse LLM output into a dict.
- ``BudgetError`` / ``check_and_consume_budget`` : per-day API call caps.
- ``OpenRouterClient`` : chat completions (scout / copywriter / verifier).
- ``HiggsfieldClient`` : image generation.
- ``ZernioClient`` : Pinterest posting + analytics via the Zernio scheduler
  (this pipeline does not call Pinterest's own API directly at all).

Every network call is wrapped in try/except and raises a typed ``ApiError`` so
callers can log and degrade gracefully rather than crashing the whole cycle.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from config import CONFIG

logger = logging.getLogger("affiliate_engine.clients")


_ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")


def resolve_asin(url: str) -> Optional[str]:
    """Pull the ASIN out of an Amazon product URL.

    Handles both direct listing URLs (/dp/ASIN, /gp/product/ASIN) and short
    links (a.co, amzn.to) by following the redirect chain — shorteners
    generally only resolve on GET, not HEAD. Never raises; returns None on
    any failure (unreachable URL, no ASIN found, non-Amazon URL).
    """
    if not url:
        return None
    match = _ASIN_RE.search(url)
    if match:
        return match.group(1)
    try:
        with requests.get(url, allow_redirects=True, timeout=10, stream=True) as resp:
            match = _ASIN_RE.search(resp.url)
            if match:
                return match.group(1)
    except requests.RequestException:
        pass
    return None


class ApiError(Exception):
    """Raised when an external API call fails."""


class BudgetError(Exception):
    """Raised when a per-day API budget cap has been reached."""


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Remove surrounding ```/```json markdown fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first fence line and any trailing fence.
        stripped = _FENCE_RE.sub("", stripped)
        # Remove a trailing fence that survived (fences on their own lines).
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
    return stripped.strip()


def extract_json(text: str) -> Dict[str, Any]:
    """Parse an LLM response into a dict, tolerating markdown fences and prose.

    Raises ``ValueError`` if no valid JSON object can be recovered.
    """
    if text is None:
        raise ValueError("Cannot parse JSON from None")
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from LLM output: {exc}") from exc
    raise ValueError("No JSON object found in LLM output")


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------
def check_and_consume_budget(db, provider: str, cap: int) -> int:
    """Increment today's usage for ``provider`` and enforce ``cap``.

    Raises ``BudgetError`` if the cap has already been reached (before this
    call would exceed it). Returns the new usage count on success.
    """
    current = db.api_usage_today(provider)
    if current >= cap:
        raise BudgetError(f"Daily API cap reached for {provider}: {current}/{cap}")
    return db.increment_api_usage(provider)


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
class OpenRouterClient:
    """Minimal OpenRouter chat-completions client.

    ``db`` is optional; when supplied, calls are counted against the daily cap.
    """

    PROVIDER = "openrouter"

    def __init__(self, db=None):
        self.db = db
        self.api_key = CONFIG.openrouter_api_key
        self.base_url = CONFIG.openrouter_base_url.rstrip("/")

    def chat_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """Call the model and return parsed JSON (dict).

        Raises ``ApiError`` on network/HTTP failure, ``BudgetError`` when the
        cap is hit, and ``ValueError`` when the response is not valid JSON.
        """
        if not self.api_key:
            raise ApiError("OPENROUTER_API_KEY is not set")
        if self.db is not None:
            check_and_consume_budget(self.db, self.PROVIDER, CONFIG.openrouter_daily_call_cap)

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Ask for JSON when the model supports it; harmless otherwise.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/affiliate-engine",
            "X-Title": "Affiliate Engine",
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            # Surface OpenRouter's own error body — a 404 here usually means the
            # model slug is unknown/deprecated, or your account's data-policy
            # settings expose no endpoint for it ("No endpoints found ...").
            body = ""
            if exc.response is not None:
                body = (exc.response.text or "")[:500]
            raise ApiError(
                f"OpenRouter HTTP error for model '{model}': {exc} :: {body}"
            ) from exc
        except requests.RequestException as exc:
            raise ApiError(f"OpenRouter request failed: {exc}") from exc
        except ValueError as exc:  # invalid JSON envelope
            raise ApiError(f"OpenRouter returned non-JSON envelope: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError(f"Unexpected OpenRouter response shape: {exc}") from exc
        return extract_json(content)

    def generate_image(
        self,
        model: str,
        prompt: str,
        aspect_ratio: Optional[str] = None,
        reference_image_url: Optional[str] = None,
    ) -> bytes:
        """Generate (or edit) an image via OpenRouter's dedicated Image API.

        Per https://openrouter.ai/docs/features/multimodal/image-generation:
            POST {base}/images
            body: {model, prompt, input_references?, aspect_ratio?}
            response: {data: [{b64_json: "..."}]}

        When ``reference_image_url`` is given, it's passed as an image-to-image
        reference (``input_references``) so an editing-capable model (e.g.
        google/gemini-3-pro-image) can preserve the subject while placing it in
        a new scene. Raises ``ApiError`` on failure, ``BudgetError`` on cap.
        """
        if not self.api_key:
            raise ApiError("OPENROUTER_API_KEY is not set")
        if self.db is not None:
            check_and_consume_budget(self.db, self.PROVIDER, CONFIG.openrouter_daily_call_cap)

        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if reference_image_url:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": reference_image_url}}
            ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/affiliate-engine",
            "X-Title": "Affiliate Engine",
        }
        try:
            resp = requests.post(
                f"{self.base_url}/images",
                headers=headers,
                json=payload,
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            body = (exc.response.text or "")[:500] if exc.response is not None else ""
            raise ApiError(f"OpenRouter image HTTP error for model '{model}': {exc} :: {body}") from exc
        except requests.RequestException as exc:
            raise ApiError(f"OpenRouter image request failed: {exc}") from exc
        except ValueError as exc:
            raise ApiError(f"OpenRouter image returned non-JSON envelope: {exc}") from exc

        try:
            b64 = data["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError(f"Unexpected OpenRouter image response shape: {exc} :: {list(data.keys())}") from exc
        try:
            return base64.b64decode(b64)
        except (ValueError, TypeError) as exc:
            raise ApiError(f"OpenRouter image base64 decode failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Higgsfield (image generation)
# ---------------------------------------------------------------------------
class HiggsfieldClient:
    """Client for the Higgsfield queue API.

    Contract (per https://docs.higgsfield.ai "How to use API"):
        POST   {base}/{model_id}                    -> submit, returns request_id
        GET    {base}/requests/{request_id}/status   -> poll for completion
        Auth:  Authorization: Key {api_key}:{api_key_secret}
    """

    PROVIDER = "higgsfield"

    def __init__(self, db=None):
        self.db = db
        self.api_key = CONFIG.higgsfield_api_key
        self.api_secret = CONFIG.higgsfield_api_secret
        self.base_url = CONFIG.higgsfield_base_url.rstrip("/")
        self.model_id = CONFIG.higgsfield_model_id.strip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Key {self.api_key}:{self.api_secret}",
            "Content-Type": "application/json",
        }

    def generate_image(
        self, prompt: str, width: int, height: int, reference_image_url: Optional[str] = None
    ) -> bytes:
        """Generate an image and return raw bytes.

        Submits to the queue, polls until ``completed``, then downloads the
        resulting image URL. Raises ``ApiError`` on failure (including
        ``failed``/``nsfw`` terminal statuses) and ``BudgetError`` when the
        daily cap is hit.

        If ``reference_image_url`` is given, it's sent as a reference/input
        image so generation is anchored on a real product photo. The exact
        request field for this is UNVERIFIED against Higgsfield's own docs
        (see HIGGSFIELD_IMAGE_PARAM) — if the submit call rejects it, this
        automatically retries once as a plain text-to-image request rather
        than failing the whole cycle over an optional enhancement.
        """
        if not self.api_key or not self.api_secret:
            raise ApiError(
                "HIGGSFIELD_API_KEY and HIGGSFIELD_API_SECRET must both be set "
                "(auth is a key:secret pair, not a single bearer token)"
            )
        if self.db is not None:
            check_and_consume_budget(self.db, self.PROVIDER, CONFIG.higgsfield_daily_call_cap)

        aspect_ratio = _aspect_ratio_string(width, height)
        try:
            request_id = self._submit_job(prompt, aspect_ratio, reference_image_url)
        except ApiError:
            if reference_image_url is None:
                raise
            logger.warning(
                "Higgsfield submit with reference image failed; retrying as text-only "
                "(check HIGGSFIELD_IMAGE_PARAM if this keeps happening)",
            )
            request_id = self._submit_job(prompt, aspect_ratio, None)

        image_url = self._poll_job(request_id)
        return self._download(image_url)

    def _submit_job(
        self, prompt: str, aspect_ratio: str, reference_image_url: Optional[str] = None
    ) -> str:
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": CONFIG.higgsfield_resolution,
        }
        if reference_image_url:
            payload[CONFIG.higgsfield_image_param] = reference_image_url
        try:
            resp = requests.post(
                f"{self.base_url}/{self.model_id}",
                headers=self._headers(),
                json=payload,
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = (exc.response.text or "")[:500] if exc.response is not None else ""
            raise ApiError(f"Higgsfield submit HTTP error: {exc} :: {body}") from exc
        except requests.RequestException as exc:
            raise ApiError(f"Higgsfield submit failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise ApiError(f"Higgsfield submit returned non-JSON: {exc}") from exc
        request_id = data.get("request_id")
        if not request_id:
            raise ApiError(f"Higgsfield submit response had no request_id: {list(data.keys())}")
        return str(request_id)

    def _poll_job(self, request_id: str) -> str:
        """Poll until the request completes; return the output image URL."""
        deadline = time.monotonic() + CONFIG.higgsfield_poll_timeout_seconds
        while True:
            try:
                resp = requests.get(
                    f"{self.base_url}/requests/{request_id}/status",
                    headers=self._headers(),
                    timeout=CONFIG.http_timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.HTTPError as exc:
                body = (exc.response.text or "")[:500] if exc.response is not None else ""
                raise ApiError(f"Higgsfield poll HTTP error: {exc} :: {body}") from exc
            except (requests.RequestException, ValueError) as exc:
                raise ApiError(f"Higgsfield poll failed: {exc}") from exc

            status = str(data.get("status", "")).lower()
            if status == "completed":
                url = self._extract_image_url(data)
                if not url:
                    raise ApiError(f"Higgsfield request completed but no image URL: {list(data.keys())}")
                return url
            if status == "nsfw":
                raise ApiError(
                    f"Higgsfield request {request_id} was blocked by content moderation (nsfw)"
                )
            if status == "failed":
                raise ApiError(f"Higgsfield request {request_id} failed: {data.get('error') or data}")
            # "queued" / "in_progress" -> keep polling.

            if time.monotonic() >= deadline:
                raise ApiError(
                    f"Higgsfield request {request_id} did not finish within "
                    f"{CONFIG.higgsfield_poll_timeout_seconds}s (last status: {status or 'unknown'})"
                )
            time.sleep(CONFIG.higgsfield_poll_interval_seconds)

    @staticmethod
    def _extract_image_url(data: Dict[str, Any]) -> Optional[str]:
        images = data.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict) and first.get("url"):
                return first["url"]
            if isinstance(first, str):
                return first
        return None

    def _download(self, url: str) -> bytes:
        try:
            resp = requests.get(url, timeout=CONFIG.http_timeout_seconds)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            raise ApiError(f"Failed to download generated image: {exc}") from exc


def _aspect_ratio_string(width: int, height: int) -> str:
    """Reduce a pixel width/height to a ``W:H`` ratio string, e.g. 1000x1500 -> '2:3'."""
    import math

    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}:{height // divisor}"


# ---------------------------------------------------------------------------
# Zernio (third-party scheduler — the only Pinterest integration this
# pipeline uses; it never calls Pinterest's own API directly)
# ---------------------------------------------------------------------------
# Statuses treated as a confirmed successful publish on the Pinterest platform
# entry. Zernio's docs don't fully detail the immediate-publish response
# shape, so this is a tolerant, non-exhaustive set — anything not in either
# set below is treated as "accepted, status unclear" (logged, not fatal).
_ZERNIO_SUCCESS_STATUSES = {"success", "posted", "published", "completed", "live"}
_ZERNIO_FAILURE_STATUSES = {"failed", "error", "rejected"}


class ZernioClient:
    """Client for Zernio (https://zernio.com), used to post to Pinterest.

    Contract per https://docs.zernio.com:
        POST /v1/media/presign          -> {uploadUrl, publicUrl, key, expiresIn}
        PUT  {uploadUrl}                -> upload raw bytes (no auth needed)
        POST /v1/posts                  -> create/publish the post

    Unlike Pinterest's own v5 API (base64 image bytes inline), Zernio requires
    a publicly reachable media URL — hence the presign+upload step before
    every post.
    """

    def __init__(self):
        self.api_key = CONFIG.zernio_api_key
        self.base_url = CONFIG.zernio_base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def upload_media(
        self, image_bytes: bytes, filename: str = "pin.jpg", content_type: str = "image/jpeg"
    ) -> str:
        """Upload image bytes via Zernio's presigned-URL flow; return the
        public URL to reference in a post's ``mediaItems``. Raises
        ``ApiError`` on failure."""
        if not self.api_key:
            raise ApiError("ZERNIO_API_KEY is not set")

        try:
            resp = requests.post(
                f"{self.base_url}/media/presign",
                headers=self._headers(),
                json={"filename": filename, "contentType": content_type},
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            body = (exc.response.text or "")[:500] if exc.response is not None else ""
            raise ApiError(f"Zernio presign HTTP error: {exc} :: {body}") from exc
        except (requests.RequestException, ValueError) as exc:
            raise ApiError(f"Zernio presign request failed: {exc}") from exc

        upload_url = data.get("uploadUrl")
        public_url = data.get("publicUrl")
        if not upload_url or not public_url:
            raise ApiError(f"Zernio presign response missing uploadUrl/publicUrl: {list(data.keys())}")

        try:
            put_resp = requests.put(
                upload_url,
                data=image_bytes,
                headers={"Content-Type": content_type},
                timeout=CONFIG.http_timeout_seconds,
            )
            put_resp.raise_for_status()
        except requests.HTTPError as exc:
            body = (exc.response.text or "")[:300] if exc.response is not None else ""
            raise ApiError(f"Zernio media upload HTTP error: {exc} :: {body}") from exc
        except requests.RequestException as exc:
            raise ApiError(f"Zernio media upload failed: {exc}") from exc

        return public_url

    def create_pinterest_post(
        self,
        title: str,
        description: str,
        image_public_url: str,
        board_id: str,
        link: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create (and immediately publish) a Pinterest pin via Zernio.

        Returns the parsed response dict. Raises ``ApiError`` on an HTTP
        failure OR on a 2xx response whose Pinterest platform entry reports
        an explicit failure status (Zernio's docs cite a 21.1% Pinterest
        failure rate on their platform, so a 2xx alone doesn't guarantee
        Pinterest actually accepted the pin).
        """
        if not self.api_key:
            raise ApiError("ZERNIO_API_KEY is not set")
        if not CONFIG.zernio_pinterest_account_id:
            raise ApiError("ZERNIO_PINTEREST_ACCOUNT_ID is not set")

        platform_specific_data: Dict[str, Any] = {"title": title, "boardId": board_id}
        if link:
            platform_specific_data["link"] = link

        payload = {
            "content": description,
            "mediaItems": [{"type": "image", "url": image_public_url}],
            "platforms": [
                {
                    "platform": "pinterest",
                    "accountId": CONFIG.zernio_pinterest_account_id,
                    "platformSpecificData": platform_specific_data,
                }
            ],
            "publishNow": True,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/posts",
                headers=self._headers(),
                json=payload,
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            body = (exc.response.text or "")[:500] if exc.response is not None else ""
            raise ApiError(f"Zernio create-post HTTP error: {exc} :: {body}") from exc
        except (requests.RequestException, ValueError) as exc:
            raise ApiError(f"Zernio create-post request failed: {exc}") from exc

        platform_entry = self._pinterest_platform_entry(data)
        status = str((platform_entry or {}).get("status", "")).lower()
        if status in _ZERNIO_FAILURE_STATUSES:
            reason = (platform_entry or {}).get("error") or (platform_entry or {}).get("failureReason") or status
            raise ApiError(f"Zernio reported the Pinterest post failed: {reason}")
        if status and status not in _ZERNIO_SUCCESS_STATUSES:
            logger.warning(
                "Zernio Pinterest post has an unrecognized status %r — treating as "
                "accepted since the API call itself succeeded (2xx). Response keys: %s",
                status, list(data.keys()),
            )

        return data

    @staticmethod
    def _pinterest_platform_entry(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        platforms = (data.get("post") or {}).get("platforms")
        if not isinstance(platforms, list):
            return None
        for entry in platforms:
            if isinstance(entry, dict) and entry.get("platform") == "pinterest":
                return entry
        return platforms[0] if platforms and isinstance(platforms[0], dict) else None

    def get_analytics(self, from_date: str, to_date: str, platform: str = "pinterest") -> Dict[str, Any]:
        """Fetch analytics for all posts on ``platform`` within a date range.

        A single call covers every post in the range — used instead of one
        request per pin. Per https://docs.zernio.com/analytics/get-analytics
        (``GET /v1/analytics?platform=&fromDate=&toDate=``), returns a dict
        with a ``posts`` list; exact per-post metric field names beyond
        "impressions/saves/clicks are available" aren't fully documented, so
        callers should extract metrics tolerantly (see
        ``analytics_pull.py::_extract_metric``). Raises ``ApiError`` on
        failure.
        """
        if not self.api_key:
            raise ApiError("ZERNIO_API_KEY is not set")
        try:
            resp = requests.get(
                f"{self.base_url}/analytics",
                headers=self._headers(),
                params={"platform": platform, "fromDate": from_date, "toDate": to_date},
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            body = (exc.response.text or "")[:500] if exc.response is not None else ""
            raise ApiError(f"Zernio analytics HTTP error: {exc} :: {body}") from exc
        except (requests.RequestException, ValueError) as exc:
            raise ApiError(f"Zernio analytics request failed: {exc}") from exc

    @staticmethod
    def extract_post_id(data: Dict[str, Any]) -> Optional[str]:
        post = data.get("post") or {}
        return post.get("_id") or data.get("_id")

    @staticmethod
    def extract_pin_url(data: Dict[str, Any]) -> Optional[str]:
        """Best-effort pin URL extraction. Zernio's docs mention immediate
        posts include ``platformPostUrl`` but don't fully specify where —
        this tries several plausible locations, tolerant of the ambiguity."""
        post = data.get("post") or {}
        for candidate in (
            post.get("platformPostUrl"),
            data.get("platformPostUrl"),
        ):
            if candidate:
                return candidate
        entry = ZernioClient._pinterest_platform_entry(data) or {}
        for key in ("postUrl", "url", "permalink", "platformPostUrl"):
            if entry.get(key):
                return entry[key]
        return None
