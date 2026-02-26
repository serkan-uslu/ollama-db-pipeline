"""
Ollama Models Scraper
Ollama kütüphane sayfasından (https://ollama.com/library) model verilerini çekerek
SQLite veritabanına yazan bir araçtır.

Her modelin kendi sayfasına da gider: README ve memory requirements (GB) çeker.

Kullanım:
    poetry run python scraper.py
"""

import logging
import re
import sys
import time
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup
from sqlmodel import Session, select

from db import Model, SQLModel, engine

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} | {levelname} | {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BASE_URL = "https://ollama.com"
LIBRARY_URL = f"{BASE_URL}/library"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# Her istek arasında bekleme süresi (saniye) — sunucuya yük bindirmemek için
REQUEST_DELAY = 0.5

# Yetenek etiketleri — bunları labels'dan ayırıp capabilities'e koyuyoruz
CAPABILITY_TAGS = {"vision", "tools", "embedding", "code", "thinking", "cloud"}


def get_page(url: str) -> BeautifulSoup | None:
    """Verilen URL'yi çeker ve BeautifulSoup nesnesi döner."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def parse_pulls(pulls_text: str) -> int:
    """'110.4M', '78.4K', '1.2B' gibi metni int'e çevirir."""
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
    """'7 months ago', '1 year ago', '2 weeks ago' gibi metni date'e çevirir."""
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
    else:  # day
        approx = today - timedelta(days=value)

    return approx, text


def scrape_model_detail(slug: str) -> dict:
    """
    Bir modelin detay sayfasını çeker.
    Döner: {readme: str, memory_requirements: [{tag, size, size_gb, context}]}
    """
    url = f"{LIBRARY_URL}/{slug}"
    soup = get_page(url)
    if not soup:
        return {"readme": None, "memory_requirements": None}

    # ── README ──────────────────────────────────────────────────────────────
    readme_text = None
    # Readme bölümü genellikle id="readme" olan bir div ya da
    # "Readme" başlıklı bir section içinde bulunur
    readme_section = (
        soup.find(id="readme")
        or soup.find("section", string=re.compile("readme", re.I))
    )
    if not readme_section:
        # Fallback: tüm sayfa içinde "Readme" başlığından sonrasını al
        all_text = soup.get_text(separator="\n")
        readme_match = re.search(
            r"Readme\n(.*?)(?:\nModels|\nTags|\Z)", all_text, re.DOTALL | re.IGNORECASE
        )
        if readme_match:
            readme_text = readme_match.group(1).strip()
    else:
        readme_text = readme_section.get_text(separator="\n").strip()

    # ── MEMORY REQUIREMENTS (tag boyutları) ─────────────────────────────────
    # Sayfa metninden "tagname  X.XGB  YK context" satırlarını parse et
    page_text = soup.get_text(separator="\n")

    size_pattern = re.compile(
        r"([\w.:_-]+)\s+([\d.]+\s*GB)\s*[·\-]?\s*([\d.]+[KM]?\s*(?:context|K|M)?)",
        re.IGNORECASE,
    )

    seen_sizes = set()   # (size_gb, context) → same size = same model, skip dupe
    seen_tags = set()
    raw_entries = []

    for match in size_pattern.finditer(page_text):
        tag_raw = match.group(1).strip()
        size_str = match.group(2).strip()
        context_str = match.group(3).strip()

        # Geçersiz token'ları atla
        if "/" in tag_raw or "http" in tag_raw.lower():
            continue
        if tag_raw in seen_tags:
            continue

        # GB float'a çevir
        size_gb_match = re.search(r"([\d.]+)", size_str)
        size_gb = float(size_gb_match.group(1)) if size_gb_match else None

        # Aynı boyut+context kombinasyonu → duplicate (ör. "latest" + "qwen3:latest")
        dedup_key = (size_gb, context_str)
        if dedup_key in seen_sizes:
            continue

        seen_tags.add(tag_raw)
        seen_sizes.add(dedup_key)

        # Quantization bilgisini tag adından çıkar (q4_0, q8_0, fp16, ...)
        quant_match = re.search(r"(q\d+[_k]?[ms]?\w*|fp16|bf16|int\d)", tag_raw, re.IGNORECASE)
        if quant_match:
            quantization = quant_match.group(1).lower()
        else:
            # Ollama default quantization for unspecified tags is Q4_K_M
            quantization = "q4_k_m"

        # CPU'da çalıştırmak için önerilen RAM ≈ size_gb × 1.25
        recommended_ram_gb = round(size_gb * 1.25, 1) if size_gb else None

        # context_window: parse "128K context" / "256K" / "2M context" → integer tokens
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
            "context_window": ctx_int,   # integer token count, e.g. 128000
        })

    # Boyuta göre küçükten büyüğe sırala
    memory_requirements = sorted(
        raw_entries, key=lambda x: x["size_gb"] or 999
    )

    return {
        "readme": readme_text[:10000] if readme_text else None,
        "memory_requirements": memory_requirements or None,
    }


def scrape_library_page(sort: str = "popular") -> list[dict]:
    """Ollama /library sayfasındaki tüm modelleri çeker."""
    logger.info(f"Fetching: {LIBRARY_URL}?sort={sort}")
    soup = get_page(f"{LIBRARY_URL}?sort={sort}")
    if not soup:
        return []

    models = []
    model_links = soup.select("a[href^='/library/']")

    seen = set()
    for link in model_links:
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

        pulls_text = ""
        pulls_match = re.search(r"([\d.]+[MKB]?)\s+Pulls", link.get_text(), re.IGNORECASE)
        if pulls_match:
            pulls_text = pulls_match.group(1)

        tags_count = 0
        tags_match = re.search(r"(\d+)\s+Tags", link.get_text(), re.IGNORECASE)
        if tags_match:
            tags_count = int(tags_match.group(1))

        updated_text = ""
        updated_match = re.search(r"Updated\s+(.+?)(?:\n|$)", link.get_text())
        if updated_match:
            updated_text = updated_match.group(1).strip()

        label_els = link.select("span.px-2") or link.select("span[class*='px-']")
        all_tags = [el.get_text(strip=True).lower() for el in label_els if el.get_text(strip=True)]

        # Capability etiketleri (vision, tools, vb.) ile boyut etiketlerini (8b, 70b) ayır
        capabilities = [tag.capitalize() for tag in all_tags if tag in CAPABILITY_TAGS]
        labels = [tag for tag in all_tags if tag not in CAPABILITY_TAGS]

        # Geriye dönük uyumluluk için capability (tek string) de tutalım
        capability = capabilities[0] if capabilities else None

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
        })

    logger.info(f"Found {len(models)} models on library page")
    return models


def insert_models(raw_models: list[dict]):
    """Model verilerini veritabanına ekler/günceller."""
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        inserted = 0
        updated = 0
        now = datetime.now()
        total = len(raw_models)

        for i, raw in enumerate(raw_models, 1):
            slug = raw["slug"]
            logger.info(f"[{i}/{total}] Processing: {slug}")

            # Her model için detay sayfasını çek
            time.sleep(REQUEST_DELAY)
            detail = scrape_model_detail(slug)

            pulls = parse_pulls(raw["pulls_text"])
            last_updated, last_updated_str = parse_last_updated(raw["updated_text"])

            # En küçük modelin RAM ihtiyacı (hardware uyumluluk filtresi)
            mem = detail.get("memory_requirements") or []
            valid_sizes = [r["size_gb"] for r in mem if r.get("size_gb") and r["size_gb"] > 0]
            min_ram_gb = round(min(valid_sizes), 1) if valid_sizes else None

            # speed_tier: deterministik — min model boyutuna göre
            if min_ram_gb is not None:
                if min_ram_gb <= 2.5:
                    speed_tier = "fast"    # 1-3B models
                elif min_ram_gb <= 6:
                    speed_tier = "medium"  # 7-8B models
                else:
                    speed_tier = "slow"    # 13B+
            else:
                speed_tier = None

            # context_window: memory_requirements içindeki en büyük context_window değeri
            ctx_values = [r["context_window"] for r in mem if r.get("context_window")]
            context_window = max(ctx_values) if ctx_values else None

            existing = session.exec(
                select(Model).where(Model.model_identifier == slug)
            ).first()

            if existing:
                existing.pulls = pulls
                existing.tags = raw["tags_count"]
                existing.last_updated = last_updated
                existing.last_updated_str = last_updated_str
                existing.description = raw["description"]
                existing.capability = raw["capability"]
                existing.capabilities = raw["capabilities"]
                if raw["labels"]:
                    existing.labels = raw["labels"]
                existing.readme = detail["readme"]
                existing.memory_requirements = detail["memory_requirements"]
                existing.min_ram_gb = min_ram_gb
                existing.context_window = context_window
                existing.speed_tier = speed_tier
                existing.timestamp = now
                session.add(existing)
                updated += 1
            else:
                model = Model(
                    model_identifier=slug,
                    namespace=None,
                    model_name=raw["model_name"],
                    model_type="official",
                    description=raw["description"],
                    capability=raw["capability"],
                    capabilities=raw["capabilities"],
                    labels=raw["labels"] or [],
                    pulls=pulls,
                    tags=raw["tags_count"],
                    last_updated=last_updated,
                    last_updated_str=last_updated_str,
                    url=f"https://ollama.com/library/{slug}",
                    readme=detail["readme"],
                    memory_requirements=detail["memory_requirements"],
                    min_ram_gb=min_ram_gb,
                    context_window=context_window,
                    speed_tier=speed_tier,
                    timestamp=now,
                )
                session.add(model)
                inserted += 1

            # Her 20 modelde bir commit yap
            if i % 20 == 0:
                session.commit()
                logger.info(f"  → Checkpoint saved ({i}/{total})")

        session.commit()
        logger.info(f"Done! Inserted: {inserted}, Updated: {updated}")
        return inserted, updated


def main():
    logger.info("=== Ollama Scraper Starting ===")
    raw_models = scrape_library_page(sort="popular")

    if not raw_models:
        logger.error("No models found! Check the scraper selectors.")
        return

    logger.info(f"Total models to process: {len(raw_models)}")
    logger.info(f"Estimated time: ~{len(raw_models) * REQUEST_DELAY / 60:.1f} min")
    insert_models(raw_models)
    logger.info("=== Scraper Finished ===")


if __name__ == "__main__":
    main()
