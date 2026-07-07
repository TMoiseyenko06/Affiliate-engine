#!/usr/bin/env python3
"""Cron entrypoint for the Pinterest affiliate pipeline.

Runs one full content cycle. Invoked 3-5x/day by cron and is fully unattended.

Usage:
    python run_cycle.py                 # run a real cycle (posts publicly)
    python run_cycle.py --dry-run       # run through the verifier, do NOT post
    python run_cycle.py --pull-analytics  # pull Pinterest analytics for pins
    python run_cycle.py --check-deadman   # run only the dead-man's-switch check

Every step is logged; failures are written to the DB and the alerts file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from config import CONFIG
from db import get_db


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_run(args) -> int:
    from agents.orchestrator import run_cycle
    from alerting import check_deadman_switch

    db = get_db()

    if not args.dry_run:
        missing = CONFIG.validate_for_live_run()
        if missing:
            print(
                "ERROR: missing required config for a live run: "
                + ", ".join(missing)
                + "\nUse --dry-run to test without posting.",
                file=sys.stderr,
            )
            return 2

    if args.mock_images and not args.dry_run:
        print(
            "ERROR: --mock-images can only be used with --dry-run "
            "(a placeholder image must never be posted).",
            file=sys.stderr,
        )
        return 2

    summary = run_cycle(
        db=db,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        skip_llm_verify=args.skip_llm_verify,
        mock_images=args.mock_images,
    )

    # Dead-man's-switch check after each real cycle.
    if not args.dry_run:
        check_deadman_switch(db)

    print(json.dumps(summary, indent=2, default=str))
    status = summary.get("status")
    # Non-zero exit for hard failures so cron mail / monitoring can catch them.
    if status in ("error", "post_failed", "budget_capped"):
        return 1
    return 0


def cmd_pull_analytics(args) -> int:
    from analytics_pull import pull_all

    db = get_db()
    count = pull_all(db)
    print(f"Pulled analytics for {count} pins.")
    return 0


def cmd_check_deadman(args) -> int:
    from alerting import check_deadman_switch

    db = get_db()
    tripped = check_deadman_switch(db)
    print("Dead-man's-switch TRIPPED" if tripped else "OK: recent successful post found")
    return 1 if tripped else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinterest affiliate pipeline runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline through the verifier but stop before posting.",
    )
    parser.add_argument(
        "--pull-analytics",
        action="store_true",
        help="Pull Pinterest analytics for existing pins instead of running a cycle.",
    )
    parser.add_argument(
        "--check-deadman",
        action="store_true",
        help="Only run the dead-man's-switch check.",
    )
    parser.add_argument(
        "--skip-llm-verify",
        action="store_true",
        help="Skip the LLM judgement layer of the verifier (deterministic checks only).",
    )
    parser.add_argument(
        "--mock-images",
        action="store_true",
        help="Stub image generation with a local placeholder (no Higgsfield key "
             "needed). Only valid with --dry-run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated/composited images (default: temp dir).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.pull_analytics:
        return cmd_pull_analytics(args)
    if args.check_deadman:
        return cmd_check_deadman(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
