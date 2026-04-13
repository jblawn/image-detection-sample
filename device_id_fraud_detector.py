"""
device_id_fraud_detector.py
────────────────────────────
Baseline image-provenance fraud detector.

For each image in a GCS bucket/folder, Gemini is asked to:
  1. Visually fingerprint the image — describe its unique, searchable
     characteristics (device model, colour, background, composition,
     visible text, distinguishing marks, staging details).
  2. Use Google Search grounding to look for that image — or a near-identical
     copy — anywhere on the public web: stock-photo libraries, other
     marketplace listings, manufacturer press galleries, scam-report sites, etc.
  3. Return a structured verdict and the URLs where matches were found.

How grounding works here
────────────────────────────
Gemini's Google Search grounding issues real search queries derived from the
model's understanding of the image.  This is not pixel-level reverse-image
lookup (that is Google Vision API's Web Detection).  What it does instead:
  • Constructs targeted queries from visual characteristics the model observes
    (e.g. "iPhone 14 Pro Space Black cracked screen eBay listing photo")
  • Fetches and reads the live search results
  • Reports whether any result contains the same or a suspiciously similar image

This catches: recycled stock photos, manufacturer press images used as listing
photos, images reposted across multiple fraud listings, and images that appear
on known scam-report forums.

Dependencies
────────────────────────────
  pip install google-genai google-cloud-storage

Usage
────────────────────────────
  python device_id_fraud_detector.py \
      --project  my-gcp-project \
      --location us-central1 \
      --bucket   my-image-bucket \
      --folder   incoming/batch-001 \
      [--output-json results.json] \
      [--log-level DEBUG]
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from google.cloud import storage as gcs

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    stream=sys.stdout,
)
logger = logging.getLogger("image_fraud_detector")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID = "gemini-3.1-pro-preview"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

EXTENSION_TO_MIME = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WebMatch:
    url: str
    description: str      # what Gemini found at this URL that matches the image
    match_type: str       # EXACT | NEAR_IDENTICAL | SIMILAR | CONTEXTUAL


@dataclass
class ImageSearchReport:
    gcs_uri: str
    analysis_timestamp: str
    visual_fingerprint: str           # Gemini's description of unique image characteristics
    search_queries_used: list[str]    # queries Gemini issued via grounding
    verdict: str                      # ORIGINAL | LIKELY_REUSED | CONFIRMED_REUSED | UNKNOWN
    verdict_rationale: str            # 1–3 sentence explanation
    web_matches: list[WebMatch] = field(default_factory=list)
    grounding_chunks: list[str] = field(default_factory=list)  # raw URLs from grounding metadata

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────
def list_gcs_images(
    bucket_name: str,
    folder_prefix: str,
    gcs_client: gcs.Client,
) -> list[tuple[str, str]]:
    """
    List all supported image blobs under gs://bucket_name/folder_prefix.
    Returns list of (gcs_uri, mime_type).
    """
    prefix = folder_prefix.strip("/") + "/" if folder_prefix.strip("/") else ""
    logger.info(
        "Scanning gs://%s/%s",
        bucket_name,
        prefix if prefix else "(bucket root)",
    )

    blobs = list(gcs_client.list_blobs(bucket_name, prefix=prefix))
    logger.debug("Total blobs under prefix: %d", len(blobs))

    results: list[tuple[str, str]] = []
    for blob in blobs:
        ext = Path(blob.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping (unsupported type): %s", blob.name)
            continue

        mime = blob.content_type or EXTENSION_TO_MIME.get(ext, "")
        if not mime or mime == "application/octet-stream":
            mime = EXTENSION_TO_MIME.get(ext, "")
        if not mime:
            logger.warning("Cannot determine MIME for %s — skipping", blob.name)
            continue

        uri = f"gs://{bucket_name}/{blob.name}"
        logger.debug("Queued: %-60s  mime=%s", uri, mime)
        results.append((uri, mime))

    logger.info("%d image(s) found", len(results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Shared generation config  (built once, reused per call)
# ──────────────────────────────────────────────────────────────────────────────
_GENERATE_CONFIG = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=4096,
    response_mime_type="application/json",
    system_instruction=(
        "You are a multimodal reverse image search engine embedded in a Trust & Safety platform. "
        "You can both SEE images and search the web simultaneously. "
        "Your task is to use your visual understanding of an image as the search query itself — "
        "identify the most discriminating visual features, construct targeted search queries from them, "
        "and determine whether this exact image (or a near-duplicate) exists anywhere on the public internet. "
        "Think like Google Lens: you are not searching for the device model in general, "
        "you are hunting for THIS SPECIFIC PHOTO. "
        "Cite every URL. Return only valid JSON — no markdown, no prose."
    ),
    tools=[types.Tool(google_search=types.GoogleSearch())],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_ONLY_HIGH"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="BLOCK_ONLY_HIGH"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_ONLY_HIGH"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="BLOCK_ONLY_HIGH"),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────
def init_client(project: str, location: str) -> genai.Client:
    logger.info("Initializing google-genai client — project=%s  location=%s", project, location)
    client = genai.Client(vertexai=True, project=project, location=location)
    logger.info("Client ready  model=%s", MODEL_ID)
    return client


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
_SEARCH_PROMPT = """\
You are performing a multimodal reverse image search on the photo provided.
You can SEE the image in full detail AND search the web. Use both capabilities
together — your eyes are the query engine.

═══ PHASE 1 — VISUAL SIGNATURE EXTRACTION ════════════════════════════════════

Study the image with forensic precision. Extract a layered visual signature
across three tiers. The goal is to build search queries that would surface THIS
SPECIFIC PHOTO, not just photos of the same device model.

Tier A — Absolute identifiers  (things unique to this exact photo)
  • Every piece of visible text: UI strings on the device screen, label text,
    sticker text, watermarks, price tags, handwritten notes in frame
  • Exact pixel composition of any on-screen display (apps open, battery %, time)
  • Physical damage: precise location, shape, and extent of any cracks, scratches,
    dents, or discolouration — describe as if directing a forensic sketch
  • Accessories in frame: cable type/colour, case make/colour, box artwork detail

Tier B — Scene identifiers  (things unique to this photo session)
  • Background surface: material, colour, pattern, grain, visible edges
  • Lighting: direction, colour temperature, reflections, shadows, lens flare
  • Composition: exact angle (degrees from horizontal/vertical), crop, negative space
  • Any incidental objects: hands, furniture, floors, walls, reflections in the screen

Tier C — Subject identifiers  (device-level, shared with press/stock)
  • Exact device model, SKU colour name, storage/variant if determinable
  • Camera array layout, port positions, button configuration
  • Branding details: logo font, position, finish (matte/gloss)

═══ PHASE 2 — SEARCH STRATEGY ═══════════════════════════════════════════════

You are acting as Google Lens. Issue multiple targeted Google Search queries,
working from most-specific to least-specific. Use your Tier A and B findings
to construct queries that only this image would match.

Required search angles — execute ALL of them:

  1. EXACT TEXT SEARCH
     If any text is visible in the image (on-screen UI, labels, stickers):
     Search that exact text string + device model + "listing" or "for sale"
     Example: '"battery 73%" "iPhone 14 Pro" site:ebay.com OR site:facebook.com'

  2. DAMAGE / DEFECT SEARCH
     If physical damage is present, search the specific description:
     Example: '"cracked top-right corner" "Galaxy S23" used listing'

  3. STAGING / BACKGROUND SEARCH
     Search the distinctive background or staging detail + device:
     Example: '"marble surface" "AirPods Pro" listing photo'
     Example: '"red case" "iPhone 15" craigslist photo'

  4. STOCK & PRESS IMAGE SEARCH
     Search manufacturer and stock photo sites for the Tier C subject:
     Example: '"iPhone 14 Pro" "Deep Purple" official image site:apple.com'
     Example: '"Galaxy S24 Ultra" press photo site:samsung.com OR site:gsmarena.com'
     Example: '"Pixel 8 Pro" stock photo site:shutterstock.com OR site:gettyimages.com'

  5. CROSS-PLATFORM LISTING SEARCH
     Search for the image on marketplace and classifieds platforms:
     Example: '"iPhone 14 Pro" "Space Black" used 256GB site:swappa.com'
     Example: '"MacBook Pro 14" used listing photo site:ebay.com'

  6. IMAGE HOST & CDN SEARCH
     Search image hosting and CDN platforms where scraped images land:
     Example: '"iphone-14-pro-space-black" site:imgur.com OR site:i.sstatic.net'
     Example: '"{exact_screen_text}" filetype:jpg OR filetype:png'

  7. SCAM & FRAUD REPORT SEARCH
     Search fraud-report forums and consumer complaint sites:
     Example: '"iPhone 14 Pro" scam listing photo site:reddit.com'
     Example: '"[any visible serial or IMEI]" fraud stolen'

═══ PHASE 3 — MULTIMODAL MATCH EVALUATION ════════════════════════════════════

For each search result that might contain the image:
  • Use your visual reasoning to assess whether the image at that URL
    would match what you see in the provided photo.
  • A match is EXACT if composition, damage, staging, and on-screen content align.
  • A match is NEAR_IDENTICAL if the same photo session but possibly different crop/export.
  • A match is SIMILAR if same device + staging but likely a different photo.
  • A match is CONTEXTUAL if same device model appears but photo is clearly different.

═══ PHASE 4 — VERDICT ════════════════════════════════════════════════════════

  ORIGINAL         — exhaustive search found no matching images anywhere online;
                     photo shows unique characteristics consistent with a genuine
                     seller-taken photo
  LIKELY_REUSED    — strong visual overlap with images found online, but cannot
                     confirm pixel-level identity from search results alone
  CONFIRMED_REUSED — identical or near-identical image located at a specific URL;
                     this photo was not taken by the seller
  UNKNOWN          — image lacks searchable unique features OR search returned
                     no useful signal either way

═══ OUTPUT ═══════════════════════════════════════════════════════════════════

Return ONLY this JSON (no markdown fences, no extra keys):

{
  "visual_fingerprint": "<layered Tier A / B / C description>",
  "search_queries_used": ["<exact query string 1>", "<exact query string 2>", "..."],
  "verdict": "<ORIGINAL|LIKELY_REUSED|CONFIRMED_REUSED|UNKNOWN>",
  "verdict_rationale": "<1-3 sentences: what specifically led to this verdict>",
  "web_matches": [
    {
      "url": "<url>",
      "description": "<which visual elements matched and how confidently>",
      "match_type": "<EXACT|NEAR_IDENTICAL|SIMILAR|CONTEXTUAL>"
    }
  ]
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Core search call
# ──────────────────────────────────────────────────────────────────────────────
def search_image(
    gcs_uri: str,
    mime: str,
    client: genai.Client,
) -> ImageSearchReport:
    logger.info("━" * 70)
    logger.info("Searching: %s", gcs_uri)
    logger.info("━" * 70)

    report = ImageSearchReport(
        gcs_uri=gcs_uri,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        visual_fingerprint="",
        search_queries_used=[],
        verdict="UNKNOWN",
        verdict_rationale="",
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(file_data=types.FileData(file_uri=gcs_uri, mime_type=mime)),
                types.Part(text=_SEARCH_PROMPT),
            ],
        )
    ]

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=_GENERATE_CONFIG,
    )

    # ── Token & finish diagnostics ───────────────────────────────────────────
    candidate = response.candidates[0]
    logger.debug("Finish reason : %s", candidate.finish_reason)
    logger.debug(
        "Token usage   — prompt: %s  output: %s  total: %s",
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.candidates_token_count,
        response.usage_metadata.total_token_count,
    )

    # ── Grounding metadata ───────────────────────────────────────────────────
    grounding_urls: list[str] = []
    try:
        chunks = (
            getattr(
                getattr(candidate, "grounding_metadata", None),
                "grounding_chunks",
                [],
            ) or []
        )
        for chunk in chunks:
            uri = getattr(getattr(chunk, "web", None), "uri", None)
            if uri:
                grounding_urls.append(uri)
        logger.debug("Grounding chunks returned: %d", len(grounding_urls))
        if grounding_urls:
            for url in grounding_urls:
                logger.debug("  grounding ↳ %s", url)
    except Exception as exc:
        logger.warning("Could not extract grounding metadata: %s", exc)

    report.grounding_chunks = grounding_urls

    # ── Parse model JSON ─────────────────────────────────────────────────────
    raw = response.text
    logger.debug("Raw response (first 600 chars):\n%s", raw[:600])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s\nFull response:\n%s", exc, raw)
        report.verdict = "UNKNOWN"
        report.verdict_rationale = f"Model returned unparseable response: {exc}"
        return report

    report.visual_fingerprint  = data.get("visual_fingerprint", "")
    report.search_queries_used = data.get("search_queries_used", [])
    report.verdict             = data.get("verdict", "UNKNOWN").upper()
    report.verdict_rationale   = data.get("verdict_rationale", "")

    for m in data.get("web_matches", []):
        url = m.get("url", "").strip()
        if not url:
            continue
        report.web_matches.append(WebMatch(
            url=url,
            description=m.get("description", ""),
            match_type=m.get("match_type", "SIMILAR").upper(),
        ))

    logger.info(
        "Verdict: %-20s  matches=%d  grounding_urls=%d",
        report.verdict, len(report.web_matches), len(grounding_urls),
    )
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Terminal output
# ──────────────────────────────────────────────────────────────────────────────
_VERDICT_COLOR = {
    "ORIGINAL":         "\033[32m",    # green
    "LIKELY_REUSED":    "\033[33m",    # yellow
    "CONFIRMED_REUSED": "\033[31m",    # red
    "UNKNOWN":          "\033[90m",    # grey
}
_MATCH_COLOR = {
    "EXACT":         "\033[31m",
    "NEAR_IDENTICAL":"\033[91m",
    "SIMILAR":       "\033[33m",
    "CONTEXTUAL":    "\033[90m",
}
_RESET = "\033[0m"


def print_report(report: ImageSearchReport) -> None:
    color   = _VERDICT_COLOR.get(report.verdict, "")
    filename = report.gcs_uri.split("/")[-1]

    print("\n" + "─" * 70)
    print(f"  FILE      : {filename}")
    print(f"  GCS URI   : {report.gcs_uri}")
    print(f"  TIMESTAMP : {report.analysis_timestamp}")
    print(f"  VERDICT   : {color}{report.verdict}{_RESET}")
    print(f"  RATIONALE : {report.verdict_rationale}")

    if report.visual_fingerprint:
        print(f"\n  VISUAL FINGERPRINT:")
        # word-wrap at 66 chars
        words, line = report.visual_fingerprint.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 64:
                print(f"    {line}")
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            print(f"    {line}")

    if report.search_queries_used:
        print(f"\n  SEARCH QUERIES ISSUED ({len(report.search_queries_used)}):")
        for q in report.search_queries_used:
            print(f"    • {q}")

    if report.web_matches:
        print(f"\n  WEB MATCHES ({len(report.web_matches)}):")
        for m in report.web_matches:
            mc = _MATCH_COLOR.get(m.match_type, "")
            print(f"    [{mc}{m.match_type}{_RESET}]  {m.url}")
            print(f"             {m.description}")
    elif report.verdict != "UNKNOWN":
        print("\n  No web matches found.")

    if report.grounding_chunks:
        print(f"\n  GROUNDING METADATA ({len(report.grounding_chunks)} URL(s)):")
        for url in report.grounding_chunks:
            print(f"    ↳ {url}")

    print("─" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for device listing images on the web using Gemini grounding"
    )
    parser.add_argument("--project",  required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI region")
    parser.add_argument("--bucket",   required=True, help="GCS bucket (no gs://)")
    parser.add_argument("--folder",   default="",    help="Folder prefix inside the bucket (omit to scan the bucket root)")
    parser.add_argument("--output-json", default=None, help="Write reports to this JSON file")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    gcs_client = gcs.Client(project=args.project)
    client     = init_client(args.project, args.location)

    images = list_gcs_images(args.bucket, args.folder, gcs_client)
    if not images:
        logger.error(
            "No supported images found in gs://%s/%s",
            args.bucket,
            args.folder or "(bucket root)",
        )
        sys.exit(1)

    all_reports: list[ImageSearchReport] = []

    for gcs_uri, mime in images:
        try:
            report = search_image(gcs_uri, mime, client)
            all_reports.append(report)
            print_report(report)
        except Exception as exc:
            logger.exception("Failed processing %s: %s", gcs_uri, exc)

    # Batch summary
    print("\n" + "═" * 70)
    print(f"  BATCH SUMMARY — {len(all_reports)} image(s) scanned")
    print("═" * 70)
    for r in all_reports:
        color = _VERDICT_COLOR.get(r.verdict, "")
        matches = f"{len(r.web_matches)} match(es)" if r.web_matches else "no matches"
        print(
            f"  {color}{r.verdict:<20}{_RESET}  {matches:<16}  "
            f"{r.gcs_uri.split('/')[-1]}"
        )
    print("═" * 70)

    confirmed = sum(1 for r in all_reports if r.verdict == "CONFIRMED_REUSED")
    likely    = sum(1 for r in all_reports if r.verdict == "LIKELY_REUSED")
    print(f"  Confirmed reuse: {confirmed}   Likely reuse: {likely}")

    if args.output_json:
        out = Path(args.output_json)
        out.write_text(
            json.dumps([asdict(r) for r in all_reports], indent=2),
            encoding="utf-8",
        )
        logger.info("Reports written → %s", out)


if __name__ == "__main__":
    main()
