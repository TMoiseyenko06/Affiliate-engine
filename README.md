# Affiliate Engine

A Python multi-agent system that automates a Pinterest content pipeline with
Amazon Associates affiliate links. It runs unattended on a schedule (3–5×/day
via cron) and applies a **hard verification gate** before anything is posted
publicly.

Each cycle publishes one of three content types, chosen to steer the actual
mix toward a configured target (default: even thirds):
- **affiliate** — product-anchored, disclosed, Associates-tagged, with a
  pain-point/curiosity title overlaid on the image.
- **affiliate_image_only** — the identical product/copy/compliance pipeline,
  but with NO text drawn on the image at all — the product photo speaks for
  itself. The Pinterest title, description, disclosure, and tagged link are
  unaffected; only the image differs.
- **organic** — no product, no CTA, no link — pure niche value content.

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
    └── agents/poster_agent.py      PURE CODE: posts via Zernio (Pinterest scheduler)
db.py            SQLite persistence (posts, performance, products, verifier_log, alerts, …)
config.py        all tunables + secrets (from env vars)
alerting.py      dead-man's-switch, skip, and budget-cap alerts
analytics_pull.py  separate periodic pull from Pinterest Analytics -> performance table
                 (still uses the direct Pinterest API v5, not Zernio — see below)
```

### Flow of one cycle

1. **Decide** – Orchestrator computes each content type's actual share of
   recent posts vs. its target and picks whichever is furthest under target
   (`agents/orchestrator.py::decide_content_type`).
2. **Select subject** – `affiliate` / `affiliate_image_only` → scout returns a
   product **not used within the lookback window**; `organic` → an unused
   topic/angle for the niche.
3. **Copywriter** – produces structured JSON copy. Both affiliate variants use
   the identical path: verbatim Associates disclosure + a correctly-tagged
   (non-cloaked) link, pain-point/curiosity title; organic forbids
   product/CTA/link. The image-only distinction doesn't touch this step at
   all — the Pinterest title/description are the same either way.
4. **Creative** – builds an image scene prompt and generates/edits a 2:3
   image (see "Creative scene reasoning" below).
5. **Compositor** – overlays the title with Pillow and validates aspect ratio
   and file size (no LLM) — **unless** `content_type == affiliate_image_only`,
   in which case the title overlay is skipped entirely and the image is used
   as-is.
6. **Verifier (hard gate)** – deterministic checks first (disclosure, link/tag,
   dimensions, size, duplicates, char limits), then an **independent LLM review
   using a different model** than the copywriter/orchestrator. Nothing bypasses
   this.
7. **On FAIL** – the content-generation steps are retried **exactly once** with
   the failure reasons appended to context. If it fails again, the cycle is
   **skipped**, logged, and an **alert** is written.
8. **Poster** – on pass (and not `--dry-run`), posts via
   [Zernio](https://zernio.com) — a third-party scheduler, **not** Pinterest's
   own API v5 — and records the pin ID/URL. See "Posting via Zernio" below.

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
| `HIGGSFIELD_API_KEY` + `HIGGSFIELD_API_SECRET` | yes | image generation (both required — auth is a key:secret pair) |
| `ZERNIO_API_KEY` | yes | posting (via Zernio, see below) |
| `ZERNIO_PINTEREST_ACCOUNT_ID` | yes | the Zernio-side connected Pinterest account to post as |
| `PINTEREST_BOARD_ID` | yes | board to post to |
| `PINTEREST_ACCESS_TOKEN` | no | analytics only (`analytics_pull.py`) — **not** used for posting |
| `AMAZON_ASSOCIATES_TAG` | yes | Associates tag appended to every affiliate link |
| `AFFILIATE_DISCLOSURE_TEXT` | no | verbatim disclosure (has a default) |
| `SCOUT_MODEL` / `COPYWRITER_MODEL` / `CREATIVE_MODEL` / `VERIFIER_MODEL` | no | model routing; **keep the verifier model different** from the copywriter so review is independent |
| `NICHES`, `TARGET_AFFILIATE_RATIO`, `TARGET_AFFILIATE_IMAGE_ONLY_RATIO`, `TARGET_ORGANIC_RATIO`, `DAILY_POST_COUNT` | no | content strategy — 3-way target mix, defaults to even thirds |
| `PRODUCT_REUSE_MODE`, `REUSE_LOOKBACK_DAYS`, `PRODUCT_REUSE_FALLBACK_ENABLED` | no | product reuse policy (see below); default: never repost a product |
| `OPENROUTER_DAILY_CALL_CAP`, `HIGGSFIELD_DAILY_CALL_CAP` | no | per-day budget caps |
| `DATABASE_URL` | no | defaults to `sqlite:///affiliate_engine.db` |
| `ALERT_FILE_PATH`, `DEADMAN_HOURS`, `ALERT_WEBHOOK_URL` | no | alerting |
| `SCRAPE_PRODUCT_IMAGES` | no | fetch a real product photo to anchor affiliate creative (see below); default `false`, confirmed non-functional |

### Product reuse policy

By default (`PRODUCT_REUSE_MODE=permanent`), a product is **never posted
twice** — once `poster_agent.py` confirms a successful post, that product is
excluded from every future scout selection for the life of the database.
Set `PRODUCT_REUSE_MODE=cooldown` to restore the older behavior instead: a
product becomes eligible again after `REUSE_LOOKBACK_DAYS`. This only
governs product reuse — organic topic rotation always uses the cooldown
window (its own small fixed topic list would exhaust almost immediately
under permanent exclusion).

**This never costs extra API calls.** The exclusion check
(`db.all_used_product_ids()` in permanent mode, `db.products_used_within()`
in cooldown mode) is a single local SQL query, turned into an in-memory set
and used to filter the candidate list *before* any ranking or image-fetching
happens (`agents/scout_agent.py::scout_product`). The per-candidate ranking
LLM call and the ScraperAPI image-fetch call each fire **at most once per
cycle**, only for the single already-chosen, already-known-fresh winner —
never iterated across candidates to "find one that isn't a duplicate."

**When a niche's entire product catalog has already been posted**
(`PRODUCT_REUSE_FALLBACK_ENABLED=true`, the default), the scout falls back
to re-posting the single least-recently-used product rather than stalling
the pipeline, and writes an alert (`db.least_recently_used_product_id()`,
also a single local query) — a nudge to add more products to
`PRODUCT_SOURCE_URL` or the seed list. Set it to `false` to hard-stop
instead (the cycle fails/skips, matching the old exhausted-catalog
behavior). Note that the verifier's separate "was this posted very
recently" safety-net check (using `REUSE_LOOKBACK_DAYS`, independent of
`PRODUCT_REUSE_MODE`) still applies even to the fallback pick — with a very
small catalog under heavy posting cadence, this can cause a cycle to skip
rather than immediately re-post something you just posted minutes ago; that
is intentional caution, not a bug, and resolves itself once enough time
passes or you add more products.

### Posting via Zernio

Posting does **not** call Pinterest's API directly — it goes through
[Zernio](https://zernio.com), a third-party social scheduler. Set up the
Pinterest connection once via Zernio's own OAuth "Connecting Accounts" flow
(outside this pipeline), then copy the resulting account ID into
`ZERNIO_PINTEREST_ACCOUNT_ID`.

Per Zernio's docs, posting is a two-step flow (`agents/clients.py::ZernioClient`):
1. **Upload** the composited image via `POST /media/presign` (returns a
   presigned upload URL + the eventual public URL), then `PUT` the raw image
   bytes to that upload URL. Zernio requires a publicly reachable media URL —
   unlike Pinterest's own API, it does not accept inline base64 image bytes.
2. **Create the post** via `POST /posts`, referencing the public URL from
   step 1, with `platforms: [{ platform: "pinterest", accountId, ... }]` and
   `publishNow: true`.

Zernio's documentation doesn't fully specify the response schema for
immediate-publish posts, so `poster_agent.py` treats the per-platform
`status` field tolerantly: an explicit `failed`/`error`/`rejected` status
raises and the cycle records a failed post; a recognized success status
(`success`/`posted`/`published`/`completed`/`live`) is recorded as posted;
anything else is logged as an unrecognized-but-accepted status rather than
treated as fatal, since the HTTP call itself already succeeded (2xx).
Create-post is retried once on error (reusing the same uploaded image URL,
no need to re-upload); the media upload itself is not retried.

Note: `PINTEREST_ACCESS_TOKEN`/`PINTEREST_BASE_URL` (direct Pinterest API v5)
are unrelated to posting now — they're only used by `analytics_pull.py` to
pull pin performance stats, which Zernio does not currently replace.

### Creative scene reasoning

`creative_agent.py` doesn't hand the image model a generic "lifestyle product
photo" template — it first makes an OpenRouter call (`CREATIVE_MODEL`) asking
the model to reason concretely about how the product is actually used, and
describe a specific, full-bleed real-world scene (e.g. "spice jars on a rack,
a hand sprinkling seasoning onto food while cooking" rather than a product
floating on a white background). This scene description is used both by the
image-editing path (below, when a real product photo exists) and by
Higgsfield's text-to-image (when it doesn't). The prompt explicitly forbids
plain/empty backgrounds and reserved negative space — the compositor's title
band (see "Pin title overlay" below) is legible over a busy image on its own,
so there's no need to ask the image model to leave blank space for text. If
this call fails (missing key, budget cap, bad response), it falls back to a
simpler static prompt that still avoids plain backgrounds — this never blocks
a cycle, but produces a less specific scene.

**The scene must illustrate the copy's actual narrative, not a different one
it invents itself.** The copywriter runs before creative in the pipeline and
has already committed to a specific angle — e.g. a title like "Why I Stopped
Digging Through Random Boxes to Find One Missing Lego Piece" for a storage-bin
product. Earlier, the scene-writer only saw generic product facts (title,
category, features) and a keyword list, so it would independently invent its
own unrelated scenario (e.g. folded clothes instead of a toy hunt) — the copy
and image told two different stories on the same pin. The scene-reasoning
prompt now leads with the copywriter's actual `title` and `description` and
is explicitly instructed to depict that exact moment/pain-point, both in the
system prompt (a hard requirement alongside "product is the hero" and
"doesn't look like an ad") and the static fallback used when the LLM call
fails.

Both the scene prompt and the copywriter are also steered toward **native,
authentic-feeling content rather than ad-like content**: the image prompt
explicitly asks for a candid, slightly-imperfect "real person's phone photo"
look instead of a staged/glossy commercial shot, and the copywriter's system
prompt (`agents/copywriter_agent.py::AFFILIATE_SYSTEM_PROMPT`) asks the model
to write from a relatable, specific, matter-of-fact voice instead of ad-speak
superlatives ("amazing", "must-have", etc.) — the psychology being that
specificity and relatability earn clicks on Pinterest, while obvious ad
language gets scrolled past. This only changes *style*; it never touches
honesty or disclosure. The Amazon Associates disclosure is still forced into
every affiliate description verbatim by code regardless of what the LLM
writes (see `copywriter_agent._write_affiliate`), and the copywriter is still
hard-required to never invent features, reviews, or stats not present in the
real product data — the verifier re-checks both independently.

### Pin title overlay

For `affiliate` and `organic` posts, `compositor.py` draws the title in a
band anchored to the **bottom** of the image (not the top), and deliberately
does **not** look the same on every pin — each call to `compose()`
independently randomizes:

- **Background shape**: a clean rounded card, a soft gradient fade, a
  smooth sine-wave top edge, an irregular "torn paper" edge, or a row of
  scalloped bumps (`compositor.SHAPE_STYLES`).
- **Colour palette**: warm, inviting tones (coral, honey, terracotta,
  blush, sage, sunshine, peach), each paired with a text colour chosen for
  contrast against it — not always white-on-dark (`compositor.PALETTES`).
- **Typeface**: DejaVu Sans Bold, Comfortaa Bold, Quicksand Bold, or Dancing
  Script (a cursive/script face, bundled under `assets/fonts/`, all
  SIL Open Font License) — the script face is only used for shorter titles
  (`max_title_len`) since a cursive face hurts legibility on long,
  keyword-heavy text, and gets a soft drop shadow instead of a hard outline
  (a uniform stroke reads badly on script letterforms; the bolder sans
  faces still get the outline).

This gives real per-pin combinatorial variety (7 palettes × 4 fonts × 5
shapes) rather than a fixed look — that's intentional, not a bug, if two
pins in a row look different from each other. Text auto-sizes as large as
possible while wrapping to at most 3 lines (`compositor.MAX_TITLE_LINES`),
and the band never exceeds `MAX_BAND_FRACTION` (42%) of the image height so
even a long title can't swallow the whole pin. Wavy/bumpy/torn shapes are
capped by `EDGE_MARGIN` so decorative edges never collide with the text
itself. All fonts are bundled directly in the repo rather than relying on
whatever happens to be installed on the machine running the pipeline — an
earlier version depended on OS font paths and silently fell back to PIL's
~10px placeholder font on Windows, rendering text far smaller than intended.

For `affiliate_image_only` posts, `compose()` is called with
`draw_title_overlay=False` and skips all of the above entirely — no shape, no
palette, no font, no text at all is drawn on the image; the product photo (or
generated scene) is used exactly as produced. The verifier deterministically
checks that this actually happened (`image_meta["title_drawn"]` must be
`False` for image-only posts and `True` for the other two types) — a
mismatch fails the gate rather than silently posting the wrong treatment.

### Product imagery (affiliate posts)

When a real product photo is available, `creative_agent.generate_creative()`
follows a three-tier priority chain, falling back only as far as it needs to:

1. **Edit it into a generated in-use scene.** The scene-reasoning prompt (see
   above) is sent to an image-editing model via OpenRouter's Image API
   (`OPENROUTER_IMAGE_MODEL`, default `google/gemini-3-pro-image`) along with
   the real photo as a reference (`input_references`), explicitly instructed
   to preserve the product's exact shape, color, and label design and only
   change the surrounding scene. This is the best outcome: an authentic,
   in-use demonstration photo with the real, correct product in it.
2. **If editing fails** (missing key, budget cap, API error), fall back to
   using the real photo **directly, unedited** — no scene, but guaranteed
   exact fidelity. This was the earlier behavior and remains the fallback:
   showing a product that doesn't match what's actually sold at the link is
   a real misrepresentation risk, so fidelity always wins over having a
   generated scene.
3. **Only if no real photo exists at all** (scraping off, no test URL,
   download fails too) does it fall back to full text-to-image generation
   (Higgsfield) — a plausible but not guaranteed-accurate depiction.

Earlier this pipeline tried anchoring *generation* on a reference photo via
Higgsfield's own reference-image mode, but that mode (like most "image
reference" tools on generative platforms) guides style/mood, not exact
reproduction — testing showed it producing a plausible-looking but
*completely different* fictional product (wrong label design, wrong logo).
OpenRouter's dedicated image-editing models are built specifically for
"edit this image, preserve the subject," which is a different and more
appropriate capability for this use case.

Getting that real photo in the first place — `amazon_scraper.py` tries, in
order:

1. **ScraperAPI** (`SCRAPERAPI_KEY`) — a third-party scraping/proxy service,
   used because the official PA-API needs an approved Associates account
   with qualifying recent sales, which isn't available to every seller yet.
   Has a free tier (1,000 credits) to start. Uses ScraperAPI's dedicated
   structured Amazon product endpoint, which returns real product JSON
   (images, title, feature bullets) rather than HTML you have to parse
   yourself. The exact shape of its `images` field wasn't fully documented
   when this was built, so parsing tolerates a few plausible shapes and logs
   the raw response keys if none match — if it ever returns nothing, check
   the logs for `"no recognizable image field"` and the actual keys will be
   right there.
2. **Direct HTTP scrape** (`SCRAPE_PRODUCT_IMAGES=true`) — CONFIRMED
   NON-VIABLE: Amazon's bot wall (`opfcaptcha.amazon.com`) blocked every
   single request from two independent networks in testing. Off by default;
   kept only in case you have your own working workaround (e.g. a headless
   browser with anti-detection measures).
3. **Nothing configured** — falls through to `TEST_PRODUCT_IMAGE_URL` (see
   below) or, failing that, the AI scene-reasoning fallback described above.

The intended long-term source is still the official **Product Advertising
API (PA-API 5.0)**, which returns real image URLs directly once you have an
approved Associates account. Swap it in by adding a third strategy to
`fetch_product_image_url()` in `amazon_scraper.py` — no caller changes
needed, the rest of the pipeline just consumes whatever URL (or `None`)
comes back.

- **For testing without any of the above**, set `TEST_PRODUCT_IMAGE_URL` to
  any real image URL — used as the reference photo whenever no image was
  fetched some other way. It applies the same image to every affiliate
  product, so it's a testing aid only, not for production.
- When no real photo is available at all (nothing configured, or every
  source fails), affiliate creative falls back to the AI scene-reasoning
  path described above — a plausible but not guaranteed-accurate depiction.
  This is the accepted tradeoff until a real image source is wired up for
  every product, not a bug.
- **To test one specific product repeatedly** instead of whatever the
  ranking step picks, set `TEST_FORCE_PRODUCT_URL` to any Amazon product URL
  — direct listing links and short links (`a.co`, `amzn.to`) both work; short
  links are automatically resolved to the canonical `amazon.com/dp/ASIN`
  form before the Associates tag is appended (appending `?tag=` directly to
  a short link often silently fails to carry the tag through the redirect).
  If the URL's ASIN matches a product already in the built-in seed list, its
  real feature data is used automatically; otherwise set
  `TEST_FORCE_PRODUCT_TITLE` too so the copywriter has something to work
  with. This bypasses ranking and the reuse-window check — not for
  production.
- The exact Higgsfield request field for a reference image
  (`HIGGSFIELD_IMAGE_PARAM`, default `image`) is **unverified** against their
  own docs — if it's wrong, `HiggsfieldClient` automatically retries the same
  request as text-only rather than failing the cycle, so a wrong param name
  degrades gracefully instead of blocking posts. If you see repeated
  "retrying as text-only" warnings in the logs, correct the env var once you
  find the right field name in Higgsfield's dashboard/support.
- Whenever no reference image is available (scraping off, no test URL set,
  or scraping fails), the pipeline silently falls back to text-to-image —
  this never blocks a cycle.

**Loading them:** the app auto-loads a `.env` file from the working directory
at startup (dependency-free, cross-platform) — just create `.env` and run.
Values already set in the real environment take precedence over the file, so
you can still override per-run:

```powershell
# Windows PowerShell — override a single value for one run
$env:OPENROUTER_API_KEY = "sk-or-..."
python run_cycle.py --dry-run
```

```bash
# macOS/Linux — .env is loaded automatically; this only overrides it
export OPENROUTER_API_KEY="sk-or-..."
```

Point at a `.env` elsewhere with `DOTENV_PATH=/path/to/.env`. Note that
`source .env` does **not** work in PowerShell — rely on the auto-loader instead.

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
products, organic CTA/link leakage, and title-overlay/content-type mismatches
(the `affiliate_image_only` guarantee) are all caught. The DB tests cover
posts, ratio computation (including the 3-way content-type breakdown),
product/topic reuse windows, budget counters, and alerts.

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
