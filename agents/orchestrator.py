"""Orchestrator — drives one content cycle end to end.

Responsibilities:
- Read current state from the DB.
- Decide this cycle's content_type ("affiliate" vs "organic") from the target
  ratio vs. actual recent history.
- Affiliate: get a fresh product from the scout. Organic: pick a topic/angle.
- Run copywriter -> creative -> compositor -> verifier -> poster in sequence.
- Log every step's output and any failure to the DB.
- On verifier FAIL: retry the content-generation steps exactly once with the
  failure reasons appended to context; if it fails again, skip the cycle, log
  it, and raise an alert.

``run_cycle(dry_run=...)`` returns a summary dict describing what happened.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from config import CONFIG
from db import Database
from alerting import alert_skip
from .clients import ApiError, BudgetError
from . import copywriter_agent, creative_agent, scout_agent, verifier_agent, poster_agent
import compositor as compositor_mod

logger = logging.getLogger("affiliate_engine.orchestrator")


# A small rotating set of organic topics/angles per niche keeps organic posts
# varied without an external trend source.
ORGANIC_TOPICS = {
    "home_organization": [
        ("10-minute declutter routines", "quick-win inspiration"),
        ("small-space storage ideas", "aesthetic and practical"),
        ("minimalist home habits", "calm and intentional living"),
        ("seasonal reset checklist", "fresh-start motivation"),
    ],
    "general": [
        ("everyday inspiration", "uplifting and shareable"),
    ],
}


def decide_content_type(db: Database) -> str:
    """Choose affiliate vs organic to steer actual ratio toward the target."""
    actual = db.recent_affiliate_ratio(CONFIG.ratio_lookback_posts)
    target = CONFIG.target_affiliate_ratio
    # If we're under target on affiliate share, post affiliate; else organic.
    choice = "affiliate" if actual < target else "organic"
    logger.info(
        "Ratio decision: actual_affiliate=%.2f target=%.2f -> %s",
        actual, target, choice,
    )
    return choice


def _select_organic_topic(db: Database) -> Dict[str, Any]:
    """Pick an organic topic not used within the reuse window."""
    niche = CONFIG.primary_niche()
    topics = ORGANIC_TOPICS.get(niche, ORGANIC_TOPICS["general"])
    for topic, angle in topics:
        if not db.topic_used_within(topic, CONFIG.reuse_lookback_days):
            return {"niche": niche, "topic": topic, "angle": angle}
    # All recently used — fall back to the first (better to repeat than skip).
    topic, angle = topics[0]
    return {"niche": niche, "topic": topic, "angle": angle}


def _generate_and_verify(
    content_type: str,
    subject: Dict[str, Any],
    angle: str,
    db: Database,
    attempt_id: str,
    output_dir: Optional[str],
    extra_context: str = "",
    skip_llm_verify: bool = False,
) -> Dict[str, Any]:
    """Run copywriter -> creative -> compositor -> verifier once.

    Returns a dict with the produced artifacts and the verifier verdict.
    """
    # 1. Copywriter
    copy = copywriter_agent.write_copy(
        content_type, subject, angle, db=db, extra_context=extra_context
    )
    db.log_step(attempt_id, "copywriter", "ok", {"title": copy.get("title")})

    # 2. Creative
    base_path, _ = creative_agent.generate_creative(
        content_type, copy, subject, db=db, output_dir=output_dir
    )
    db.log_step(attempt_id, "creative", "ok", {"path": base_path})

    # 3. Compositor
    image_meta = compositor_mod.compose(base_path, copy.get("title", ""), )
    db.log_step(attempt_id, "compositor", "ok", image_meta)

    # 4. Verifier
    verdict = verifier_agent.verify(
        content_type, copy, image_meta, subject, db=db,
        post_attempt_id=attempt_id, skip_llm=skip_llm_verify,
    )
    db.log_step(
        attempt_id, "verifier",
        "pass" if verdict["pass"] else "fail",
        {"failures": verdict["failures"]},
    )

    return {"copy": copy, "image_meta": image_meta, "verdict": verdict}


def run_cycle(
    db: Optional[Database] = None,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    skip_llm_verify: bool = False,
) -> Dict[str, Any]:
    """Execute one full cycle. Returns a summary dict (never raises for
    expected pipeline failures — those are logged and reported in the summary).
    """
    from db import get_db

    db = db or get_db()
    attempt_id = uuid.uuid4().hex[:12]
    summary: Dict[str, Any] = {"attempt_id": attempt_id, "dry_run": dry_run}

    try:
        # --- decide ---
        content_type = decide_content_type(db)
        summary["content_type"] = content_type
        db.log_step(attempt_id, "orchestrator.decide", "ok", {"content_type": content_type})

        # --- select subject ---
        if content_type == "affiliate":
            subject = scout_agent.scout_product(db=db)
            angle = "product spotlight"
            db.log_step(attempt_id, "scout", "ok", subject)
        else:
            topic = _select_organic_topic(db)
            subject = topic
            angle = topic["angle"]
            db.log_step(attempt_id, "topic_select", "ok", topic)
        summary["subject"] = subject

        # --- generate + verify (attempt 1) ---
        result = _generate_and_verify(
            content_type, subject, angle, db, attempt_id, output_dir,
            skip_llm_verify=skip_llm_verify,
        )

        # --- retry once on failure ---
        if not result["verdict"]["pass"]:
            reasons = "; ".join(result["verdict"]["failures"])
            logger.warning("Verifier FAILED (attempt 1): %s — retrying once", reasons)
            db.log_step(attempt_id, "orchestrator.retry", "started", {"reasons": reasons})
            result = _generate_and_verify(
                content_type, subject, angle, db, attempt_id, output_dir,
                extra_context=f"Previous attempt failed verification for: {reasons}",
                skip_llm_verify=skip_llm_verify,
            )

        if not result["verdict"]["pass"]:
            reasons = "; ".join(result["verdict"]["failures"])
            logger.error("Verifier FAILED again — skipping cycle: %s", reasons)
            db.log_step(attempt_id, "orchestrator.skip", "skipped", {"reasons": reasons})
            # Record the skip as a post row for the deadman/audit trail.
            db.create_post(
                content_type=content_type,
                product_id_or_topic=subject.get("product_id") or subject.get("topic") or "unknown",
                board_id=CONFIG.pinterest_board_id,
                status="skipped",
            )
            alert_skip(attempt_id, reasons, db)
            summary.update({"status": "skipped", "failures": result["verdict"]["failures"]})
            return summary

        copy = result["copy"]
        image_meta = result["image_meta"]
        summary["copy"] = copy
        summary["image_meta"] = image_meta

        # --- dry run stops here ---
        if dry_run:
            db.log_step(attempt_id, "orchestrator.dry_run", "ok")
            summary["status"] = "dry_run"
            summary["would_post"] = {
                "content_type": content_type,
                "title": copy.get("title"),
                "description": copy.get("description"),
                "link": copy.get("link") or copy.get("link_or_null"),
                "image_path": image_meta.get("path"),
            }
            return summary

        # --- post ---
        with open(image_meta["path"], "rb") as fh:
            final_bytes = fh.read()
        post_result = poster_agent.post_pin(
            content_type, copy, final_bytes, subject, db=db,
        )
        db.log_step(
            attempt_id, "poster",
            "ok" if post_result.get("posted") else "fail",
            post_result,
        )
        if post_result.get("posted"):
            summary["status"] = "posted"
            summary["pin_id"] = post_result.get("pin_id")
            summary["pin_url"] = post_result.get("pin_url")
        else:
            summary["status"] = "post_failed"
            summary["error"] = post_result.get("error")
        return summary

    except BudgetError as exc:
        from alerting import alert_budget_capped
        logger.error("Budget cap hit: %s", exc)
        db.log_step(attempt_id, "orchestrator", "budget_capped", str(exc))
        provider = "openrouter" if "openrouter" in str(exc).lower() else "higgsfield"
        alert_budget_capped(provider, db)
        summary.update({"status": "budget_capped", "error": str(exc)})
        return summary
    except ApiError as exc:
        logger.error("API error aborted cycle: %s", exc)
        db.log_step(attempt_id, "orchestrator", "api_error", str(exc))
        summary.update({"status": "error", "error": str(exc)})
        return summary
    except Exception as exc:  # pragma: no cover - last-resort guard
        logger.exception("Unexpected error in cycle")
        db.log_step(attempt_id, "orchestrator", "error", str(exc))
        summary.update({"status": "error", "error": str(exc)})
        return summary
