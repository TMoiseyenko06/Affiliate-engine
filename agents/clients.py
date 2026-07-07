"""Shared HTTP clients and JSON helpers for the pipeline.

Contains:
- ``extract_json`` : strip markdown fences and parse LLM output into a dict.
- ``BudgetError`` / ``check_and_consume_budget`` : per-day API call caps.
- ``OpenRouterClient`` : chat completions (scout / copywriter / verifier).
- ``HiggsfieldClient`` : image generation.
- ``PinterestClient`` : Pinterest API v5 pin creation + analytics.

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


# ---------------------------------------------------------------------------
# Higgsfield (image generation)
# ---------------------------------------------------------------------------
class HiggsfieldClient:
    PROVIDER = "higgsfield"

    def __init__(self, db=None):
        self.db = db
        self.api_key = CONFIG.higgsfield_api_key
        self.base_url = CONFIG.higgsfield_base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        """Generate an image and return raw bytes.

        The Higgsfield public API is asynchronous: submit a job to
        ``POST /v1/generations``, then poll ``GET /v1/generations/{id}`` until it
        completes, then download the resulting image URL. Raises ``ApiError`` on
        failure and ``BudgetError`` when the daily cap is hit.

        The exact response shape can vary; extraction below is deliberately
        tolerant of common field names.
        """
        if not self.api_key:
            raise ApiError("HIGGSFIELD_API_KEY is not set")
        if self.db is not None:
            check_and_consume_budget(self.db, self.PROVIDER, CONFIG.higgsfield_daily_call_cap)

        job_id = self._submit_job(prompt, width, height)
        image_url = self._poll_job(job_id)
        return self._download(image_url)

    def _submit_job(self, prompt: str, width: int, height: int) -> str:
        payload = {
            "task": "text-to-image",
            "model": CONFIG.higgsfield_model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": CONFIG.higgsfield_steps,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v1/generations",
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
        job_id = data.get("id") or data.get("generation_id") or data.get("job_id")
        if not job_id:
            raise ApiError(f"Higgsfield submit response had no job id: {list(data.keys())}")
        return str(job_id)

    def _poll_job(self, job_id: str) -> str:
        """Poll until the job completes; return the output image URL."""
        deadline = time.monotonic() + CONFIG.higgsfield_poll_timeout_seconds
        while True:
            try:
                resp = requests.get(
                    f"{self.base_url}/v1/generations/{job_id}",
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
            if status in ("completed", "succeeded", "success", "done"):
                url = self._extract_image_url(data)
                if not url:
                    raise ApiError(f"Higgsfield job completed but no image URL: {list(data.keys())}")
                return url
            if status in ("failed", "error", "canceled", "cancelled"):
                raise ApiError(f"Higgsfield job {job_id} failed: {data.get('error') or data}")

            if time.monotonic() >= deadline:
                raise ApiError(
                    f"Higgsfield job {job_id} did not finish within "
                    f"{CONFIG.higgsfield_poll_timeout_seconds}s (last status: {status or 'unknown'})"
                )
            time.sleep(CONFIG.higgsfield_poll_interval_seconds)

    @staticmethod
    def _extract_image_url(data: Dict[str, Any]) -> Optional[str]:
        # Tolerate a few plausible result shapes.
        for key in ("image_url", "url", "output_url"):
            if data.get(key):
                return data[key]
        output = data.get("output") or data.get("result") or data.get("results")
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("image_url", "url", "image"):
                if output.get(key):
                    return output[key]
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                for key in ("image_url", "url", "image"):
                    if first.get(key):
                        return first[key]
        return None

    def _download(self, url: str) -> bytes:
        try:
            resp = requests.get(url, timeout=CONFIG.http_timeout_seconds)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            raise ApiError(f"Failed to download generated image: {exc}") from exc


# ---------------------------------------------------------------------------
# Pinterest API v5
# ---------------------------------------------------------------------------
class PinterestClient:
    def __init__(self):
        self.access_token = CONFIG.pinterest_access_token
        self.base_url = CONFIG.pinterest_base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def create_pin(
        self,
        board_id: str,
        title: str,
        description: str,
        image_bytes: bytes,
        link: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a pin via the v5 create-pin endpoint (base64 image source).

        Returns the parsed response dict (includes ``id`` and, typically,
        a pin URL). Raises ``ApiError`` on failure.
        """
        if not self.access_token:
            raise ApiError("PINTEREST_ACCESS_TOKEN is not set")

        media_source = {
            "source_type": "image_base64",
            "content_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
        payload: Dict[str, Any] = {
            "board_id": board_id,
            "title": title,
            "description": description,
            "media_source": media_source,
        }
        if link:
            payload["link"] = link

        try:
            resp = requests.post(
                f"{self.base_url}/pins",
                headers=self._headers(),
                json=payload,
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            body = getattr(exc.response, "text", "") if hasattr(exc, "response") and exc.response else ""
            raise ApiError(f"Pinterest create_pin failed: {exc} :: {body}") from exc
        except ValueError as exc:
            raise ApiError(f"Pinterest returned non-JSON response: {exc}") from exc

    def get_pin_analytics(self, pin_id: str, metric_types: str = "SAVE,PIN_CLICK") -> Dict[str, Any]:
        """Fetch analytics for a single pin. Raises ``ApiError`` on failure."""
        if not self.access_token:
            raise ApiError("PINTEREST_ACCESS_TOKEN is not set")
        params = {
            "metric_types": metric_types,
        }
        try:
            resp = requests.get(
                f"{self.base_url}/pins/{pin_id}/analytics",
                headers=self._headers(),
                params=params,
                timeout=CONFIG.http_timeout_seconds,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise ApiError(f"Pinterest analytics failed for {pin_id}: {exc}") from exc
        except ValueError as exc:
            raise ApiError(f"Pinterest analytics non-JSON response: {exc}") from exc
