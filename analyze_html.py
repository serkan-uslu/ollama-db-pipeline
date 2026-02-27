"""
analyze_html.py

HTML Intelligence Discovery Tool
─────────────────────────────────
Reads raw_html from DB, extracts meaningful text using BeautifulSoup,
then asks the local Ollama model an open-ended question:
  "What structured metadata can you find in this model page?"

Results are saved to output/html_analysis.json.
A field-frequency summary is printed at the end so we can decide
which new fields to add to the enricher schema.

Usage:
    python analyze_html.py [--sample N] [--model MODEL] [--all]

Options:
    --sample N    Number of models to analyse (default: 25)
    --model MODEL Ollama model to use (default: from .env / qwen2.5:3b)
    --all         Analyse all 214 models (slow, ~35 min)
    --show-text   Print extracted text for first model (debug)
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "ollama.db"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "html_analysis.json"

# Load .env manually (no dotenv dependency needed)
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

load_env()

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# ── HTML → Clean Text Extractor ─────────────────────────────────────────────────

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing beautifulsoup4...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "beautifulsoup4"], check=True)
    from bs4 import BeautifulSoup


def extract_text_from_html(html: str, max_chars: int = 4000) -> dict:
    """
    Parse raw HTML and extract structured text sections.
    Returns a dict with named sections, not one big blob.
    """
    soup = BeautifulSoup(html, "html.parser")

    sections = {}

    # ── Page title / model name ───────────────────────────────────────────────
    title = soup.find("h1")
    if title:
        sections["page_title"] = title.get_text(strip=True)

    # ── Description (often in meta or first paragraph) ───────────────────────
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        sections["meta_description"] = meta_desc["content"].strip()

    # ── Tags / capability labels ──────────────────────────────────────────────
    # Look for tag-like elements (small spans, badges)
    tag_elements = []
    for el in soup.find_all(["span", "a"], class_=re.compile(r"tag|label|badge|chip|pill", re.I)):
        t = el.get_text(strip=True)
        if t and len(t) < 50:
            tag_elements.append(t)
    if tag_elements:
        sections["tags_labels"] = list(dict.fromkeys(tag_elements))[:20]  # dedupe, limit 20

    # ── All H2 section headings (reveals what sections exist) ─────────────────
    headings = [h.get_text(strip=True) for h in soup.find_all("h2")]
    if headings:
        sections["section_headings"] = headings

    # ── Key-value stat rows (parameters, context window, etc.) ───────────────
    # Many model pages have definition lists or table-like rows
    kv_pairs = {}
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            key = dt.get_text(strip=True)
            val = dd.get_text(strip=True)
            if key and val:
                kv_pairs[key] = val
    # Also look for table rows with 2 cells
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) == 2:
            k = cells[0].get_text(strip=True)
            v = cells[1].get_text(strip=True)
            if k and v and len(k) < 60:
                kv_pairs[k] = v
    if kv_pairs:
        sections["key_value_stats"] = kv_pairs

    # ── README / main content text ────────────────────────────────────────────
    # Find the largest text block (likely the readme)
    readme_el = (
        soup.find(id=re.compile(r"readme", re.I))
        or soup.find(class_=re.compile(r"readme|markdown|content", re.I))
        or soup.find("article")
        or soup.find("main")
    )
    if readme_el:
        raw = readme_el.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r" {2,}", " ", raw)
        sections["readme_text"] = raw[:2000]  # cap at 2K chars
    else:
        # Fallback: extract all paragraph text
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 30]
        if paragraphs:
            sections["paragraphs"] = "\n".join(paragraphs[:10])[:1500]

    # ── Applications section ──────────────────────────────────────────────────
    apps = []
    for h in soup.find_all(["h2", "h3"]):
        if "application" in h.get_text(strip=True).lower():
            # Collect the content after this heading until next heading
            sibling = h.find_next_sibling()
            while sibling and sibling.name not in ("h2", "h3"):
                t = sibling.get_text(strip=True)
                if t:
                    apps.append(t)
                sibling = sibling.find_next_sibling()
    if apps:
        sections["applications_raw"] = apps[:10]

    # ── Numbers extraction (context window, params, etc.) ────────────────────
    full_text = soup.get_text(" ", strip=True)
    # Find patterns like "128K context", "7B parameters", "32768 tokens"
    patterns = {
        "context_window_mentions": re.findall(r"\b(\d+[KMB]?\s*(?:context|token|window)s?)\b", full_text, re.I),
        "parameter_mentions":      re.findall(r"\b(\d+(?:\.\d+)?[BM]\s*(?:param|B model)s?)\b", full_text, re.I),
        "license_mentions":        re.findall(r"\b(MIT|Apache[\s\-]2\.0|Llama[\s\w]*License|Gemma[\s\w]*(?:Terms|License)|DeepSeek License|Qwen License|CC BY[\s\d\.]+|GPL[\s\-]?\d?|AGPL|Proprietary|Commercial)\b", full_text, re.I),
        "benchmark_scores":        re.findall(r"\b([A-Z][A-Za-z\-]+(?:\d+)?\s*[:=]\s*\d+(?:\.\d+)?%?)\b", full_text)[:10],
        "huggingface_urls":        re.findall(r"https?://huggingface\.co/[^\s\"'>]+", full_text),
    }
    for k, v in patterns.items():
        if v:
            # Deduplicate and normalise
            sections[k] = list(dict.fromkeys(v[:8]))

    # ── Trim total to max_chars ───────────────────────────────────────────────
    # Already capped per section, but do a final size check
    total = sum(len(str(v)) for v in sections.values())
    if total > max_chars:
        # Drop paragraphs first (least structured)
        sections.pop("paragraphs", None)

    return sections


# ── Ollama Prompt & Call ────────────────────────────────────────────────────────

DISCOVERY_SYSTEM = """You are a metadata extraction expert. 
Given extracted content from an AI model page, identify and extract ALL structured metadata fields you can.
Be exhaustive — find every field that could be useful in a model database.
Think about: model identity, capabilities, technical specs, content policies, training info, performance, use cases, etc.
Return ONLY a valid JSON object. No explanation, no markdown, just raw JSON."""

DISCOVERY_USER_TMPL = """Analyse this AI model page content and extract every structured metadata field you can find.

=== PAGE CONTENT ===
{content}
===================

Return a JSON object with ALL fields you can extract. Include fields like (but not limited to):
- model_name, model_family, base_model, parameter_sizes
- context_window, max_tokens
- languages_supported, domains
- license, release_date, creator_org
- use_cases, capabilities, strengths, limitations  
- is_uncensored, is_fine_tuned, is_multimodal
- recommended_hardware, min_vram_gb
- benchmark_scores (any you see: MMLU, HumanEval, GSM8K, SWE-bench, MATH, etc.)
- applications (apps that use this model)
- huggingface_url (any https://huggingface.co/... link you find)
- any other fields you notice

Be specific and accurate. Use null for fields you cannot determine."""


def _repair_truncated_json(raw: str) -> dict | None:
    """
    Try to fix a truncated JSON string by closing open braces/brackets.
    Works for the common case where the LLM ran out of tokens mid-output.
    """
    # Find the last valid key-value boundary before truncation
    # Strategy: find last complete key:value pair and close from there
    s = raw

    # Remove trailing incomplete key or value (find last clean comma or opening brace)
    # Walk backwards to find a safe cut point
    for cutoff in range(len(s), max(0, len(s) - 200), -1):
        candidate = s[:cutoff]
        # Count open braces/brackets to determine what needs closing
        depth_brace = candidate.count("{") - candidate.count("}")
        depth_bracket = candidate.count("[") - candidate.count("]")
        # If we have matching depth, try to close cleanly
        if depth_brace >= 0 and depth_bracket >= 0:
            # Strip trailing partial token (comma, colon, partial string)
            clean = candidate.rstrip().rstrip(",").rstrip(":")
            # Close any open string literal
            quote_count = clean.count('"') - clean.count('\\"')
            if quote_count % 2 == 1:
                clean += '"'
            # Close open arrays then objects
            clean += "]" * depth_bracket + "}" * depth_brace
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue
    return None


def call_ollama(prompt_content: str, model: str, base_url: str) -> dict | None:
    """Send discovery prompt to Ollama, parse JSON response."""
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
        from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")

    user_msg = DISCOVERY_USER_TMPL.format(content=json.dumps(prompt_content, ensure_ascii=False, indent=2))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt to recover truncated JSON by closing open structures
        recovered = _repair_truncated_json(raw)
        if recovered:
            return recovered

        print(f"    ⚠ JSON unrecoverable | raw[:150]={raw[:150]!r}", flush=True)
        return None

    except Exception as e:
        print(f"    ⚠ Ollama error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"    ⚠ Ollama error: {e}", flush=True)
        return None


# ── DB Helpers ─────────────────────────────────────────────────────────────────

def fetch_models(sample: int | None) -> list[dict]:
    """Fetch models from DB with raw_html, optionally limited & diversified."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if sample is None:
        rows = conn.execute(
            "SELECT model_identifier, model_name, description, raw_html FROM model WHERE raw_html IS NOT NULL"
        ).fetchall()
    else:
        # Diverse sampling: order by rowid to spread across alphabet/insertion order
        # Also ensure we grab some large and small HTML sizes
        rows = conn.execute(
            """
            SELECT model_identifier, model_name, description, raw_html
            FROM model
            WHERE raw_html IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (sample,),
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


# ── Analysis & Summarisation ───────────────────────────────────────────────────

def summarise_results(results: list[dict]) -> dict:
    """
    From all model results, compute:
    - field_frequency: how many models had each field (non-null, non-empty)
    - field_value_samples: up to 3 example values per field
    - new_field_candidates: fields not in current enricher schema
    """
    EXISTING_SCHEMA_FIELDS = {
        "use_cases", "domain", "languages", "complexity",
        "model_family", "base_model", "is_fine_tuned", "is_uncensored",
        "best_for", "license", "strengths", "limitations", "target_audience",
        # New v2 fields
        "creator_org", "is_multimodal", "benchmark_scores", "huggingface_url",
        # DB fields (already captured by crawler)
        "model_name", "description", "capabilities", "labels",
        "min_ram_gb", "memory_requirements", "context_window",
        "speed_tier", "applications", "readme",
    }

    field_counter = Counter()
    field_samples = defaultdict(list)

    for entry in results:
        extracted = entry.get("extracted_fields", {})
        if not extracted:
            continue
        for field, value in extracted.items():
            field_lower = field.lower().replace(" ", "_").replace("-", "_")
            if value is not None and value != "" and value != [] and value != {}:
                field_counter[field_lower] += 1
                if len(field_samples[field_lower]) < 3:
                    field_samples[field_lower].append({
                        "model": entry["model_identifier"],
                        "value": value,
                    })

    total_models = len([r for r in results if r.get("extracted_fields")])
    total_models = max(total_models, 1)

    # Sort by frequency
    sorted_fields = sorted(field_counter.items(), key=lambda x: -x[1])

    # Identify genuinely new candidates (>= 20% of models have them, not already in schema)
    new_candidates = [
        {
            "field": field,
            "frequency": count,
            "pct": round(100 * count / total_models, 1),
            "samples": field_samples[field][:3],
        }
        for field, count in sorted_fields
        if field not in EXISTING_SCHEMA_FIELDS and count >= max(2, total_models * 0.15)
    ]

    all_fields = [
        {
            "field": field,
            "frequency": count,
            "pct": round(100 * count / total_models, 1),
            "in_existing_schema": field in EXISTING_SCHEMA_FIELDS,
            "samples": field_samples[field][:2],
        }
        for field, count in sorted_fields
    ]

    return {
        "total_models_analysed": total_models,
        "total_unique_fields_found": len(field_counter),
        "new_field_candidates": new_candidates,
        "all_fields_ranked": all_fields,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse raw_html from DB to discover enrichable fields")
    parser.add_argument("--sample", type=int, default=25, help="Number of models to analyse (default: 25)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--all", dest="all_models", action="store_true", help="Analyse all models")
    parser.add_argument("--show-text", action="store_true", help="Print extracted text for first model")
    args = parser.parse_args()

    sample_n = None if args.all_models else args.sample
    label = "ALL" if args.all_models else str(sample_n)

    print(f"\n{'─'*60}", flush=True)
    print(f"  HTML Intelligence Discovery", flush=True)
    print(f"  Models: {label} | Ollama: {args.model}", flush=True)
    print(f"{'─'*60}\n", flush=True)

    # Fetch models
    models = fetch_models(sample_n)
    print(f"[DB] Fetched {len(models)} models with raw_html\n", flush=True)

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = []
    failed = 0

    for i, m in enumerate(models, 1):
        model_id = m["model_identifier"]
        print(f"[{i:>3}/{len(models)}] {model_id}", end=" ", flush=True)

        # Extract text sections from HTML
        sections = extract_text_from_html(m["raw_html"])

        if args.show_text and i == 1:
            print("\n\n=== EXTRACTED TEXT (first model) ===")
            print(json.dumps(sections, indent=2, ensure_ascii=False))
            print("=== END ===\n")

        # Call Ollama
        extracted = call_ollama(sections, model=args.model, base_url=OLLAMA_BASE_URL)

        if extracted:
            print(f"→ {len(extracted)} fields", flush=True)
        else:
            print("→ FAILED", flush=True)
            failed += 1

        results.append({
            "model_identifier": model_id,
            "model_name": m.get("model_name"),
            "description": m.get("description", "")[:100],
            "extracted_sections": {k: v for k, v in sections.items() if k != "readme_text"},  # skip large text
            "extracted_fields": extracted,
            "analysed_at": datetime.utcnow().isoformat(),
        })

    print(f"\n{'─'*60}", flush=True)
    print(f"  Done: {len(results) - failed}/{len(results)} succeeded, {failed} failed", flush=True)

    # Summarise
    summary = summarise_results(results)

    # Build final output
    output = {
        "meta": {
            "run_at": datetime.utcnow().isoformat(),
            "ollama_model": args.model,
            "models_analysed": len(results),
            "models_succeeded": len(results) - failed,
        },
        "summary": summary,
        "per_model_results": results,
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Output saved → {OUTPUT_FILE}", flush=True)

    # ── Print Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}", flush=True)
    print(f"  FIELD DISCOVERY SUMMARY", flush=True)
    print(f"  Models analysed: {summary['total_models_analysed']}", flush=True)
    print(f"  Unique fields found: {summary['total_unique_fields_found']}", flush=True)
    print(f"{'═'*60}", flush=True)

    print(f"\n🆕 NEW FIELD CANDIDATES (not in current schema, ≥15% of models):\n", flush=True)
    if summary["new_field_candidates"]:
        for fc in summary["new_field_candidates"]:
            print(f"  {fc['pct']:>5.1f}%  {fc['field']}", flush=True)
            for s in fc["samples"][:1]:
                val_str = str(s["value"])[:80]
                print(f"           └ [{s['model']}] {val_str}", flush=True)
    else:
        print("  (none found above threshold)", flush=True)

    print(f"\n📊 ALL FIELDS (top 30 by frequency):\n", flush=True)
    for fc in summary["all_fields_ranked"][:30]:
        marker = "✓" if fc["in_existing_schema"] else "★"
        print(f"  {marker} {fc['pct']:>5.1f}%  {fc['field']}", flush=True)

    print(f"\n{'═'*60}", flush=True)
    print(f"  Full results → {OUTPUT_FILE}", flush=True)
    print(f"{'═'*60}\n", flush=True)


if __name__ == "__main__":
    main()
