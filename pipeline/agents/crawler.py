"""
pipeline/agents/crawler.py

F-01: Crawler Agent — crawl4ai + BeautifulSoup

Crawls https://ollama.com/library for all models,
then visits each model's detail page for README + memory requirements.

Parsing logic ported from legacy/scraper.py (parse_pulls, parse_last_updated,
memory requirements regex, scrape_model_detail).

Rules:
- Uses httpx for HTTP (no requests library)
- Returns structured dicts — does NOT write to DB (separation of concerns)
- save_models_to_db() is the only function that touches the DB
- Respects REQUEST_DELAY between detail page fetches
- Skips already-crawled models unless force=True
"""

import asyncio
import logging
import re
from datetime import date, datetime, timedelta

import httpx
from bs4 import BeautifulSoup
from sqlmodel import Session, select

from pipeline.core.db import engine, init_db
from pipeline.core.models import Model
from pipeline.core.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://ollama.com"
LIBRARY_URL = f"{BASE_URL}/library"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Capability tags separated from size labels
CAPABILITY_TAGS = {"vision", "tools", "embedding", "code", "thinking", "cloud"}

# Uncensored keywords for fast detection at crawl time
UNCENSORED_KEYWORDS = {
    "uncensored", "abliterated", "ablation", "no-restrictions",
    "jailbreak", "dan", "dolphin", "instruct-no-filter",
}


# ── Pure Parsing Helpers (ported from legacy/scraper.py) ──────────────────────

def parse_pulls(pulls_text: str) -> int:
    """Convert '110.4M', '78.4K', '1.2B' style text to int."""
    pulls_text = pulls_text.strip().upper().replace(",", "")
    try:
        if "M" in pulls_text:
            return int(float(pulls_text.replace("M", "")) * 1_000_000)
        elif "K" in pulls_text:
            return int(float(pulls_text.replace("K", "")) * 1_000)
        elif "B" in pulls_text:
            return int(float(pulls_text.replace("B", "")) * 1_000_000_000)
        else:
            return int(pulls_text)
    except ValueError:
        return 0


def parse_last_updated(text: str) -> tuple[date, str]:
    """Convert '7 months ago', '1 year ago', '2 weeks ago' to (date, raw_str)."""
    today = datetime.now().date()
    text = text.strip().lower()

    match = re.search(r"(\d+)\s+(year|month|week|day)s?\s+ago", text)
    if not match:
        return today, text

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "year":
        approx = date(today.year - value, today.month, today.day)
    elif unit == "month":
        month = today.month - value
        year = today.year
        while month <= 0:
            month += 12
            year -= 1
        approx = date(year, month, today.day)
    elif unit == "week":
        approx = today - timedelta(weeks=value)
    else:
        approx = today - timedelta(days=value)

    return approx, text


def detect_uncensored(slug: str, name: str, description: str | None) -> bool:
    """Fast heuristic: check slug/name/description for uncensored signals."""
    combined = f"{slug} {name} {description or ''}".lower()
    return any(kw in combined for kw in UNCENSORED_KEYWORDS)


def parse_model_detail_html(html: str) -> dict:
    """
    Parse a model detail page HTML.
    Returns {readme, memory_requirements}.
    Ported from legacy/scraper.py scrape_model_detail().
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── README ────────────────────────────────────────────────────────────────
    readme_text = None
    readme_section = (
        soup.find(id="readme")
        or soup.find("section", string=re.compile("readme", re.I))
    )
    if not readme_section:
        all_text = soup.get_text(separator="\n")
        readme_match = re.search(
            r"Readme\n(.*?)(?:\nModels|\nTags|\Z)", all_text, re.DOTALL | re.IGNORECASE
        )
        if readme_match:
            readme_text = readme_match.group(1).strip()
    else:
        readme_text = readme_section.get_text(separator="\n").strip()

    # ── MEMORY REQUIREMENTS ───────────────────────────────────────────────────
    page_text = soup.get_text(separator="\n")

    size_pattern = re.compile(
        r"([\w.:_-]+)\s+([\d.]+\s*GB)\s*[·\-]?\s*([\d.]+[KM]?\s*(?:context|K|M)?)",
        re.IGNORECASE,
    )

    seen_sizes: set = set()
    seen_tags: set = set()
    raw_entries = []

    for match in size_pattern.finditer(page_text):
        tag_raw = match.group(1).strip()
        size_str = match.group(2).strip()
        context_str = match.group(3).strip()

        if "/" in tag_raw or "http" in tag_raw.lower():
            continue
        if tag_raw in seen_tags:
            continue

        size_gb_match = re.search(r"([\d.]+)", size_str)
        size_gb = float(size_gb_match.group(1)) if size_gb_match else None

        dedup_key = (size_gb, context_str)
        if dedup_key in seen_sizes:
            continue

        seen_tags.add(tag_raw)
        seen_sizes.add(dedup_key)

        quant_match = re.search(
            r"(q\d+[_k]?[ms]?\w*|fp16|bf16|int\d)", tag_raw, re.IGNORECASE
        )
        quantization = quant_match.group(1).lower() if quant_match else "q4_k_m"

        recommended_ram_gb = round(size_gb * 1.25, 1) if size_gb else None

        ctx_int = None
        ctx_match = re.search(r"([\d.]+)\s*([KMk])", context_str)
        if ctx_match:
            val = float(ctx_match.group(1))
            unit = ctx_match.group(2).upper()
            ctx_int = int(val * 1_000_000) if unit == "M" else int(val * 1_000)

        raw_entries.append({
            "tag": tag_raw,
            "size": size_str,
            "size_gb": size_gb,
            "recommended_ram_gb": recommended_ram_gb,
            "quantization": quantization,
            "context": context_str,
            "context_window": ctx_int,
        })

    memory_requirements = sorted(raw_entries, key=lambda x: x["size_gb"] or 999)

    # ── APPLICATIONS ──────────────────────────────────────────────────────────
    applications = []
    for h2 in soup.find_all("h2", string=lambda t: t and t.strip() == "Applications"):
        section = h2.find_parent(["section", "div"])
        if section:
            items = section.find_all("div", recursive=False) or section.find_all("li")
            # Fallback: grab all text pairs (name + command)
            texts = [t.strip() for t in section.stripped_strings if t.strip() and t.strip() != "Applications"]
            i = 0
            while i < len(texts):
                name = texts[i]
                launch_cmd = texts[i + 1] if i + 1 < len(texts) and texts[i + 1].startswith("ollama") else None
                if launch_cmd:
                    applications.append({"name": name, "launch_command": launch_cmd})
                    i += 2
                else:
                    i += 1

    return {
        "readme": readme_text[:10000] if readme_text else None,
        "memory_requirements": memory_requirements or None,
        "applications": applications or None,
    }


# ── HTTP Helpers ───────────────────────────────────────────────────────────────

async def fetch_html(client: httpx.AsyncClient, url: str) -> str | None:
    """Fetch a URL and return HTML string. Returns None on error."""
    try:
        resp = await client.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


# ── Library Page Crawler ───────────────────────────────────────────────────────

async def crawl_library(sort: str = "popular") -> list[dict]:
    """
    Crawl ollama.com/library and return list of raw model dicts.
    Does NOT visit detail pages — that's done in crawl_model_detail().
    """
    url = f"{LIBRARY_URL}?sort={sort}"
    print(f"\n{'─'*60}", flush=True)
    print(f"[CRAWLER] 🌐 Library sayfası çekiliyor: {url}", flush=True)
    logger.info(f"Crawling library: {url}")

    async with httpx.AsyncClient() as client:
        html = await fetch_html(client, url)

    if not html:
        print("[CRAWLER] ❌ Library sayfası alınamadı!", flush=True)
        logger.error("Library page fetch failed.")
        return []

    soup = BeautifulSoup(html, "html.parser")
    models = []
    seen: set = set()

    for link in soup.select("a[href^='/library/']"):
        href = link.get("href", "")
        if ":" in href or href == "/library":
            continue

        slug = href.replace("/library/", "").strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)

        name_el = link.select_one("h2")
        desc_el = link.select_one("p")
        model_name = name_el.get_text(strip=True) if name_el else slug
        description = desc_el.get_text(strip=True) if desc_el else None

        link_text = link.get_text()

        pulls_text = ""
        pulls_match = re.search(r"([\d.]+[MKB]?)\s+Pulls", link_text, re.IGNORECASE)
        if pulls_match:
            pulls_text = pulls_match.group(1)

        tags_count = 0
        tags_match = re.search(r"(\d+)\s+Tags", link_text, re.IGNORECASE)
        if tags_match:
            tags_count = int(tags_match.group(1))

        updated_text = ""
        updated_match = re.search(r"Updated\s+(.+?)(?:\n|$)", link_text)
        if updated_match:
            updated_text = updated_match.group(1).strip()

        label_els = link.select("span.px-2") or link.select("span[class*='px-']")
        all_tags = [
            el.get_text(strip=True).lower()
            for el in label_els
            if el.get_text(strip=True)
        ]

        capabilities = [t.capitalize() for t in all_tags if t in CAPABILITY_TAGS]
        labels = [t for t in all_tags if t not in CAPABILITY_TAGS]
        capability = capabilities[0] if capabilities else None

        # Fast uncensored detection from publicly visible data
        is_uncensored = detect_uncensored(slug, model_name, description)

        models.append({
            "slug": slug,
            "model_name": model_name,
            "description": description,
            "pulls_text": pulls_text,
            "tags_count": tags_count,
            "updated_text": updated_text,
            "labels": labels,
            "capabilities": capabilities,
            "capability": capability,
            "is_uncensored_hint": is_uncensored,  # LLM will confirm later
        })

    print(f"[CRAWLER] ✅ {len(models)} model bulundu", flush=True)
    logger.info(f"Found {len(models)} models on library page.")
    return models


# ── Model Detail Page Crawler ──────────────────────────────────────────────────

async def crawl_model_detail(slug: str) -> dict:
    """
    Crawl an individual model's detail page.
    Returns {readme, memory_requirements}.
    """
    url = f"{LIBRARY_URL}/{slug}"
    async with httpx.AsyncClient() as client:
        html = await fetch_html(client, url)

    if not html:
        return {"readme": None, "memory_requirements": None, "applications": None}

    return parse_model_detail_html(html)


# ── DB Upsert ─────────────────────────────────────────────────────────────────

def save_models_to_db(models: list[dict]) -> tuple[int, int]:
    """
    Upsert model list into DB.
    Commits every 20 models as checkpoints.
    Returns (inserted, updated) counts.
    """
    init_db()
    inserted = 0
    updated = 0
    now = datetime.now()
    total = len(models)

    with Session(engine) as session:
        for i, raw in enumerate(models, 1):
            slug = raw["slug"]
            detail = raw.get("_detail", {})

            pulls = parse_pulls(raw.get("pulls_text", ""))
            last_updated, last_updated_str = parse_last_updated(raw.get("updated_text", ""))

            mem = detail.get("memory_requirements") or []
            valid_sizes = [r["size_gb"] for r in mem if r.get("size_gb") and r["size_gb"] > 0]
            min_ram_gb = round(min(valid_sizes), 1) if valid_sizes else None

            if min_ram_gb is not None:
                if min_ram_gb <= 2.5:
                    speed_tier = "fast"
                elif min_ram_gb <= 6:
                    speed_tier = "medium"
                else:
                    speed_tier = "slow"
            else:
                speed_tier = None

            ctx_values = [r["context_window"] for r in mem if r.get("context_window")]
            context_window = max(ctx_values) if ctx_values else None

            existing = session.exec(
                select(Model).where(Model.model_identifier == slug)
            ).first()

            if existing:
                # Always refresh library-page stats
                existing.pulls = pulls
                existing.tags = raw["tags_count"]
                existing.last_updated = last_updated
                existing.last_updated_str = last_updated_str
                existing.description = raw["description"]
                existing.capability = raw["capability"]
                existing.capabilities = raw["capabilities"]
                if raw["labels"]:
                    existing.labels = raw["labels"]
                existing.timestamp = now
                # Only set uncensored hint if not already confirmed by LLM
                if existing.is_uncensored is None:
                    existing.is_uncensored = raw.get("is_uncensored_hint")
                # Only overwrite detail fields when a detail page was actually fetched
                # (_has_detail=False means this was a stats-only refresh run)
                if raw.get("_has_detail", False):
                    existing.readme = detail.get("readme")
                    existing.applications = detail.get("applications")
                    existing.memory_requirements = detail.get("memory_requirements")
                    existing.raw_html = raw.get("_html")
                    existing.min_ram_gb = min_ram_gb
                    existing.context_window = context_window
                    existing.speed_tier = speed_tier
                session.add(existing)
                updated += 1
            else:
                model = Model(
                    model_identifier=slug,
                    model_name=raw["model_name"],
                    model_type="official",
                    namespace=None,
                    url=f"{LIBRARY_URL}/{slug}",
                    description=raw["description"],
                    capability=raw["capability"],
                    capabilities=raw["capabilities"],
                    labels=raw["labels"] or [],
                    pulls=pulls,
                    tags=raw["tags_count"],
                    last_updated=last_updated,
                    last_updated_str=last_updated_str,
                    readme=detail.get("readme"),
                    applications=detail.get("applications"),
                    memory_requirements=detail.get("memory_requirements"),
                    raw_html=raw.get("_html"),
                    min_ram_gb=min_ram_gb,
                    context_window=context_window,
                    speed_tier=speed_tier,
                    is_uncensored=raw.get("is_uncensored_hint"),
                    timestamp=now,
                )
                session.add(model)
                inserted += 1

            if i % 20 == 0:
                session.commit()
                print(f"[CRAWLER] 💾 Checkpoint: {i}/{total} modeli kaydedildi", flush=True)
                logger.info(f"  → Checkpoint saved ({i}/{total})")

        session.commit()

    print(f"[CRAWLER] ✅ DB kaydı tamamlandı — Yeni: {inserted} | Güncellendi: {updated}", flush=True)
    logger.info(f"DB save complete. Inserted: {inserted}, Updated: {updated}")
    return inserted, updated


# ── Re-parse Utility ──────────────────────────────────────────────────────────

def reparse_from_html() -> tuple[int, int]:
    """
    Re-parse all stored raw_html values and update crawl fields (readme,
    memory_requirements, applications) WITHOUT hitting the network.

    Use this whenever parse_model_detail_html() logic is updated and
    you want to re-extract new fields from already-crawled HTML.

    Returns (updated, skipped) counts.
    """
    init_db()
    updated = 0
    skipped = 0

    with Session(engine) as session:
        models = list(session.exec(select(Model)).all())
        total = len(models)
        print(f"[REPARSE] {total} model için ham HTML'den yeniden parse ediliyor...", flush=True)

        for i, model in enumerate(models, 1):
            if not model.raw_html:
                skipped += 1
                continue

            detail = parse_model_detail_html(model.raw_html)
            model.readme = detail.get("readme")
            model.applications = detail.get("applications")
            model.memory_requirements = detail.get("memory_requirements")

            mem = detail.get("memory_requirements") or []
            valid_sizes = [r["size_gb"] for r in mem if r.get("size_gb") and r["size_gb"] > 0]
            if valid_sizes:
                model.min_ram_gb = round(min(valid_sizes), 1)
                model.speed_tier = "fast" if model.min_ram_gb <= 2.5 else "medium" if model.min_ram_gb <= 6 else "slow"
            ctx_values = [r["context_window"] for r in mem if r.get("context_window")]
            if ctx_values:
                model.context_window = max(ctx_values)

            session.add(model)
            updated += 1

            if i % 50 == 0:
                session.commit()
                print(f"[REPARSE] {i}/{total} tamamlandı", flush=True)

        session.commit()

    print(f"[REPARSE] Bitti — Güncellendi: {updated} | Ham HTML yok: {skipped}", flush=True)
    return updated, skipped


# ── Full Crawl Pipeline ────────────────────────────────────────────────────────

async def run_full_crawl(force: bool = False) -> list[dict]:
    """
    1. Crawl library page — refresh stats (pulls, last_updated, etc.) for ALL models.
    2. Detect new slugs not yet in DB.
    3. Fetch detail pages ONLY for new models (readme, memory requirements).
       If force=True, re-fetch detail pages for all models.
    4. Save to DB — stats update for existing, full insert/update for new.
    Returns the full raw_models list.
    """
    raw_models = await crawl_library()
    if not raw_models:
        logger.error("No models found on library page.")
        return []

    # Determine which models are new vs already in DB
    with Session(engine) as session:
        existing_slugs = set(session.exec(select(Model.model_identifier)).all())

    new_models = [m for m in raw_models if m["slug"] not in existing_slugs]
    existing_models = [m for m in raw_models if m["slug"] in existing_slugs]

    print(f"[CRAWLER] {'─'*60}", flush=True)
    print(
        f"[CRAWLER] 📊 {len(new_models)} yeni model | "
        f"{len(existing_models)} mevcut (stats yenilenecek)",
        flush=True,
    )
    logger.info(
        f"Library: {len(new_models)} new, {len(existing_models)} existing (stats refresh)"
    )

    # Detail pages: always for new models, existing only if force=True
    to_fetch_detail = raw_models if force else new_models
    stats_only = [] if force else existing_models

    # Mark stats-only models so save_models_to_db won't overwrite detail fields
    for m in stats_only:
        m["_detail"] = {}
        m["_html"] = None
        m["_has_detail"] = False

    print(f"[CRAWLER] 📃 {len(to_fetch_detail)} model için detail sayfası çekiliyor...", flush=True)
    logger.info(f"Fetching detail pages for {len(to_fetch_detail)} models...")

    async with httpx.AsyncClient() as client:
        for i, model in enumerate(to_fetch_detail, 1):
            slug = model["slug"]
            print(f"[CRAWLER] [{i:>3}/{len(to_fetch_detail)}] 🔍 Detail: {slug}", flush=True)
            logger.info(f"[{i}/{len(to_fetch_detail)}] Detail: {slug}")
            html = await fetch_html(client, f"{LIBRARY_URL}/{slug}")
            detail = parse_model_detail_html(html) if html else {}
            model["_detail"] = detail
            model["_html"] = html  # raw HTML preserved for future re-parsing
            model["_has_detail"] = True
            if detail.get("applications"):
                print(
                    f"[CRAWLER]         📱 {len(detail['applications'])} uygulama: "
                    f"{[a['name'] for a in detail['applications']]}",
                    flush=True,
                )
            await asyncio.sleep(settings.request_delay)

    # Save ALL models: existing get stats refresh, new get full insert
    save_models_to_db(raw_models)
    return raw_models
