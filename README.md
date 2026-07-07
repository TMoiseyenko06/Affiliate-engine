# Affiliate Engine

A Python multi-agent system that automates a Pinterest content pipeline with
Amazon Associates affiliate links. It runs unattended on a schedule (3–5×/day
via cron) and applies a **hard verification gate** before anything is posted
publicly.

Each cycle either publishes an **affiliate** pin (product-anchored, disclosed,
Associates-tagged) or an **organic** value pin (no product, no CTA, no link),
chosen to steer the actual affiliate/organic ratio toward a configured target.

---

## Architecture

The pipeline is a sequence of small, independently testable modules under
`agents/`, driven by `run_cycle.py` (the cron entrypoint) via
`agents/orchestrator.py`.

```
run_cycle.py                    CLI entrypoint (cron)
└── agents/orchestrator.py      decides content_type, drives the sequence, retries, logs
    ├── agents/scout_agent.py       affiliate: pick a fresh product (OpenRouter, cheap model)
    ├── agents/copywriter_agent.py  copy (OpenRouter, Claude model) — mode-specific prompt
    ├── agents/creative_agent.py    image-gen prompt + Higgsfield (1000x1500, 2:3)
    ├── compositor.py               PURE CODE (Pillow): overlay title, validate aspect/size
    ├── agents/verifier_agent.py    HARD GATE: deterministic checks + independent LLM review
    └── agents/poster_agent.py      PURE CODE: Pinterest API v5 create-pin
db.py            SQLite persistence (posts, performance, products, verifier_log, alerts, …)
config.py        all tunables + secrets (from env vars)
alerting.py      dead-man's-switch, skip, and budget-cap alerts
analytics_pull.py  separate periodic pull from Pinterest Analytics -> performance table
```

### Flow of one cycle

1. **Decide** – Orchestrator compares the target affiliate ratio against the
   actual ratio over recent posts and picks `affiliate` or `organic`.
2. **Select subject** – affiliate → scout returns a product **not used within
   the lookback window**; organic → an unused topic/angle for the niche.
3. **Copywriter** – produces structured JSON copy. Affiliate mode injects the
   verbatim Associates disclosure and a correctly-tagged (non-cloaked) link;
   organic mode forbids product/CTA/link.
4. **Creative** – builds an image prompt and calls Higgsfield for a 2:3 image.
5. **Compositor** – overlays the title with Pillow and validates aspect ratio
   and file size (no LLM).
6. **Verifier (hard gate)** – deterministic checks first (disclosure, link/tag,
   dimensions, size, duplicates, char limits), then an **independent LLM review
   using a different model** than the copywriter/orchestrator. Nothing bypasses
   this.
7. **On FAIL** – the content-generation steps are retried **exactly once** with
   the failure reasons appended to context. If it fails again, the cycle is
   **skipped**, logged, and an **alert** is written.
8. **Poster** – on pass (and not `--dry-run`), formats a Pinterest API v5
   create-pin request, posts, and records the pin ID/URL.

Every LLM call returns structured JSON (markdown fences stripped, parsed with
error handling). Every external API call is wrapped in try/except and logged —
an unhandled exception can’t silently crash the whole cycle.

---

## Setup

Requires **Python 3.9+**.

```bash
git clone <this-repo>
cd Affiliate-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in the values
```

### Environment variables

All secrets are read from the environment — **never hardcoded**. See
`.env.example` for the full annotated list. The important ones:

| Variable | Required for live? | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | scout, copywriter, verifier LLM calls |
| `HIGGSFIELD_API_KEY` | yes | image generation |
| `PINTEREST_ACCESS_TOKEN` | yes | posting + analytics (Pinterest API v5) |
| `PINTEREST_BOARD_ID` | yes | board to post to |
| `AMAZON_ASSOCIATES_TAG` | yes | Associates tag appended to every affiliate link |
| `AFFILIATE_DISCLOSURE_TEXT` | no | verbatim disclosure (has a default) |
| `SCOUT_MODEL` / `COPYWRITER_MODEL` / `VERIFIER_MODEL` | no | model routing; **keep the verifier model different** from the copywriter so review is independent |
| `NICHES`, `TARGET_AFFILIATE_RATIO`, `DAILY_POST_COUNT` | no | content strategy |
| `REUSE_LOOKBACK_DAYS` | no | don’t reuse a product/topic within N days |
| `OPENROUTER_DAILY_CALL_CAP`, `HIGGSFIELD_DAILY_CALL_CAP` | no | per-day budget caps |
| `DATABASE_URL` | no | defaults to `sqlite:///affiliate_engine.db` |
| `ALERT_FILE_PATH`, `DEADMAN_HOURS`, `ALERT_WEBHOOK_URL` | no | alerting |

Load them before running:

```bash
set -a; source .env; set +a
```

---

## Running

```bash
# Full pipeline WITHOUT posting — runs through the verifier and prints what
# would have been posted. Great for testing prompts and config.
python run_cycle.py --dry-run

# Real cycle (posts publicly). Refuses to run if required config is missing.
python run_cycle.py

# Deterministic-only verification (skip the LLM judgement layer), e.g. offline.
python run_cycle.py --dry-run --skip-llm-verify

# Periodic analytics pull (run on its own schedule).
python run_cycle.py --pull-analytics

# Just check the dead-man's-switch.
python run_cycle.py --check-deadman
```

The runner prints a JSON summary of the cycle and exits non-zero on hard
failures (`error`, `post_failed`, `budget_capped`) so cron mail / monitoring
can catch them.

---

## Cron setup

Post 4×/day and pull analytics once daily. Wrap in a small script so the env is
loaded and output is logged:

```bash
# /opt/affiliate-engine/run.sh
#!/usr/bin/env bash
set -euo pipefail
cd /opt/affiliate-engine
set -a; source .env; set +a
exec .venv/bin/python run_cycle.py "$@"
```

```cron
# m  h            command
  15 8,12,17,21  /opt/affiliate-engine/run.sh            >> /var/log/affiliate/cycle.log 2>&1
  0  3           /opt/affiliate-engine/run.sh --pull-analytics >> /var/log/affiliate/analytics.log 2>&1
  0  */4         /opt/affiliate-engine/run.sh --check-deadman  >> /var/log/affiliate/deadman.log 2>&1
```

The pipeline is idempotent per invocation and safe to run unattended: budget
caps prevent runaway API spend, and the reuse window prevents duplicate posts.

---

## Reading the logs and alerts

- **Alerts file** (`ALERT_FILE_PATH`, default `alerts.log`) — one tab-separated
  line per alert: `TIMESTAMP<TAB>MESSAGE`. Alerts fire for:
  - no successful post within `DEADMAN_HOURS` (dead-man's-switch),
  - a cycle skipped after repeated verifier failure,
  - a daily API budget cap being hit.
  If `ALERT_WEBHOOK_URL` is set, alerts are also POSTed there (Slack-compatible).

- **Database** (`sqlite3 affiliate_engine.db`):
  ```sql
  -- what got posted / skipped / failed
  SELECT timestamp, content_type, product_id_or_topic, status, pin_url FROM posts ORDER BY id DESC LIMIT 20;
  -- why the verifier rejected something
  SELECT timestamp, post_attempt_id, pass, failures FROM verifier_log ORDER BY id DESC LIMIT 20;
  -- per-step audit trail for a cycle
  SELECT step, status, detail FROM cycle_log WHERE post_attempt_id = '<attempt_id>' ORDER BY id;
  -- alerts
  SELECT timestamp, message FROM alerts ORDER BY id DESC LIMIT 20;
  -- today's API usage vs caps
  SELECT day, provider, calls FROM api_usage ORDER BY day DESC;
  -- performance samples
  SELECT post_id, saves, clicks, pulled_at FROM performance ORDER BY id DESC LIMIT 20;
  ```

- **Console/cron logs** — the runner logs every step at INFO (use `-v` for
  DEBUG) and prints the cycle summary JSON.

### Database tables

| Table | Purpose |
|---|---|
| `posts` | one row per post attempt (`posted` / `skipped` / `failed`) |
| `performance` | analytics samples (saves, clicks) pulled periodically |
| `products` | product catalogue + `last_used_at` for reuse exclusion |
| `verifier_log` | every verifier verdict + failure reasons |
| `alerts` | dead-man's-switch, skip, and budget-cap notifications |
| `api_usage` | per-day per-provider call counters for budget caps |
| `cycle_log` | per-step audit trail keyed by `post_attempt_id` |

---

## Tests

Unit tests cover the deterministic verifier checks and the DB layer:

```bash
python -m unittest discover -s tests -v
```

The verifier tests assert that missing disclosures, wrong/missing/cloaked
Associates tags, bad dimensions, oversized images, over-limit text, duplicate
products, and organic CTA/link leakage are all caught. The DB tests cover
posts, ratio computation, product/topic reuse windows, budget counters, and
alerts.

---

## Safety & compliance notes

- **Disclosure is enforced twice**: the copywriter guarantees the verbatim
  Amazon Associates disclosure is present, and the verifier independently
  re-checks it before posting.
- **No link cloaking**: affiliate links point at the real Amazon domain with a
  `tag=` parameter; shorteners/redirectors are rejected by the verifier.
- **Independent review**: the verifier’s LLM check uses a *different* model from
  the copywriter, so content is never self-graded. If the verifier LLM is
  unavailable it **fails closed** (nothing posts).
- **Budget caps** on OpenRouter and Higgsfield prevent runaway spend; hitting a
  cap raises an alert and aborts the cycle cleanly.
