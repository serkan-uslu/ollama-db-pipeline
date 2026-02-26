"""
enrich.py — AI Enrichment Script

Uses local Ollama (qwen2.5:3b) to analyze each model and writes structured
metadata back to the database.

HOW VERSIONING WORKS
--------------------
ENRICH_VERSION is an integer stored per-model in the `enrich_version` column.
When you add new fields, bump ENRICH_VERSION. On next --only-missing run,
all models with enrich_version < ENRICH_VERSION will be re-processed.

Usage:
    poetry run python enrich.py                  # Enrich all models
    poetry run python enrich.py --only-missing   # Only unenriched / outdated
    poetry run python enrich.py --model llama3.1 # Single model test
"""

import argparse
import json
import logging
import re
import sys
import time

import requests
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

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"   # Fast local model, great at JSON
REQUEST_DELAY = 0.3

# Bump this number when adding NEW fields to the prompt/schema.
# All models with enrich_version < ENRICH_VERSION will be re-processed.
ENRICH_VERSION = 1

# ── Allowed values (kept narrow for consistent filtering) ─────────────────────
ALLOWED_USE_CASES = [
    "Chat Assistant", "Code Generation", "Code Review",
    "Text Summarization", "Question Answering", "RAG / Retrieval",
    "Text Embedding", "Image Understanding", "Reasoning", "Translation",
    "Math", "Creative Writing", "Data Analysis", "Function Calling", "Role Play",
]
ALLOWED_DOMAINS = [
    "General", "Code", "Vision", "Embedding",
    "Reasoning", "Math", "Medical", "Science", "Language",
]
ALLOWED_LANGUAGES = [
    "English", "Multilingual", "Chinese", "Arabic",
    "Japanese", "Korean", "French", "German", "Spanish",
]
ALLOWED_COMPLEXITY = ["beginner", "intermediate", "advanced"]
ALLOWED_SPEED = ["fast", "medium", "slow"]
ALLOWED_AUDIENCE = ["Developers", "Beginners", "Researchers", "Data Scientists", "DevOps", "Students"]


def clean_readme(raw: str | None, max_chars: int = 700) -> str:
    if not raw:
        return "N/A"
    return re.sub(r"\s+", " ", raw).strip()[:max_chars]


def build_prompt(m: Model) -> str:
    readme_snippet = clean_readme(m.readme)
    return f"""Analyze this Ollama AI model listing. Return ONLY valid JSON — no markdown, no explanation.

Model      : {m.model_identifier}
Description: {m.description or 'N/A'}
Tags/caps  : {json.dumps(m.capabilities or [])}
Sizes      : {json.dumps(m.labels or [])}
README     : {readme_snippet}

Return exactly this JSON structure (use null for genuinely unknown fields):
{{
  "use_cases": [],
  "domain": "",
  "languages": [],
  "complexity": "",
  "best_for": "",
  "license": null,
  "base_model": null,
  "is_fine_tuned": null,
  "strengths": [],
  "limitations": [],
  "target_audience": []
}}

Field rules:
- use_cases     : 2-5 items from: {json.dumps(ALLOWED_USE_CASES)}
- domain        : one from: {json.dumps(ALLOWED_DOMAINS)}
- languages     : 1-4 items from: {json.dumps(ALLOWED_LANGUAGES)}
- complexity    : one from: {json.dumps(ALLOWED_COMPLEXITY)} (beginner=<8GB RAM, intermediate=8-32GB, advanced=>32GB)
- best_for      : single concise English sentence about ideal user/task
- license       : SCAN THE README CAREFULLY for license info. Common values: "MIT", "Apache 2.0",
                  "Llama 3 Community License", "Llama 2 Community License", "Gemma Terms of Use",
                  "DeepSeek License", "Qwen License", "CC BY 4.0", "Proprietary".
                  Return null ONLY if there is truly no license mention anywhere.
- base_model    : if this is a fine-tune, return the base model identifier (e.g. "llama3.1").
                  Return null if this IS a base/foundation model.
- is_fine_tuned : true if fine-tuned or distilled from another model, false if it is a base model
- strengths     : 2-3 short strings highlighting what this model does best
- limitations   : 1-3 short strings about known weaknesses or constraints
- target_audience: 1-3 items from: {json.dumps(ALLOWED_AUDIENCE)}"""


def call_ollama(prompt: str, retries: int = 2) -> dict | None:
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",   # Guaranteed JSON output
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Strip optional markdown fences
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error (attempt {attempt}): {e}")
        except requests.RequestException as e:
            logger.warning(f"  Request error (attempt {attempt}): {e}")
        time.sleep(2)
    return None


def validate_and_clean(data: dict) -> dict:
    """Keep only allowed values. Uses None for empty/unknown."""

    def pick_list(key, allowed, max_items=5):
        return [v for v in (data.get(key) or []) if v in allowed][:max_items] or None

    def pick_one(key, allowed):
        v = data.get(key)
        return v if v in allowed else None

    def pick_int(key):
        v = data.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def pick_bool(key):
        v = data.get(key)
        if isinstance(v, bool):
            return v
        return None

    def pick_str(key, max_len=300):
        v = data.get(key)
        return str(v)[:max_len] if v else None

    def pick_free_list(key, max_items=3, max_item_len=80):
        items = data.get(key) or []
        return [str(i)[:max_item_len] for i in items if i][:max_items] or None

    return {
        "use_cases":       pick_list("use_cases", ALLOWED_USE_CASES),
        "domain":          pick_one("domain", ALLOWED_DOMAINS) or "General",
        "ai_languages":    pick_list("languages", ALLOWED_LANGUAGES),
        "complexity":      pick_one("complexity", ALLOWED_COMPLEXITY),
        "best_for":        pick_str("best_for"),
        "license":         pick_str("license", 120),
        "base_model":      pick_str("base_model", 80),
        "is_fine_tuned":   pick_bool("is_fine_tuned"),
        "strengths":       pick_free_list("strengths"),
        "limitations":     pick_free_list("limitations"),
        "target_audience": pick_list("target_audience", ALLOWED_AUDIENCE),
        # Note: context_window and speed_tier are computed in scraper.py, not here
    }


def enrich_models(only_missing: bool = False, single_slug: str | None = None):
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        query = select(Model)

        if single_slug:
            query = query.where(Model.model_identifier == single_slug)
        elif only_missing:
            # Re-process anything not yet enriched OR enriched with an older version
            query = query.where(
                (Model.enrich_version == None) |  # noqa: E711
                (Model.enrich_version < ENRICH_VERSION)
            )

        models = session.exec(query).all()
        total = len(models)
        logger.info(f"=== AI Enrichment starting — {total} models to process ===")
        logger.info(f"    Model          : {OLLAMA_MODEL}")
        logger.info(f"    Enrich version : {ENRICH_VERSION}")
        logger.info(f"    Est. time      : ~{total * (REQUEST_DELAY + 3) / 60:.1f} min")

        ok = 0
        failed = 0

        for i, m in enumerate(models, 1):
            logger.info(f"[{i}/{total}] {m.model_identifier}")

            result = call_ollama(build_prompt(m))

            if result is None:
                logger.warning(f"  → FAILED: {m.model_identifier}")
                failed += 1
                continue

            cleaned = validate_and_clean(result)
            logger.info(
                f"  → domain={cleaned['domain']} | "
                f"complexity={cleaned['complexity']} | "
                f"license={cleaned['license']} | "
                f"use_cases={cleaned['use_cases']}"
            )

            # Write all fields
            m.use_cases       = cleaned["use_cases"]
            m.domain          = cleaned["domain"]
            m.ai_languages    = cleaned["ai_languages"]
            m.complexity      = cleaned["complexity"]
            m.best_for        = cleaned["best_for"]
            m.license         = cleaned["license"]
            # context_window + speed_tier are set by scraper.py
            m.base_model      = cleaned["base_model"]
            m.is_fine_tuned   = cleaned["is_fine_tuned"]
            m.strengths       = cleaned["strengths"]
            m.limitations     = cleaned["limitations"]
            m.target_audience = cleaned["target_audience"]
            m.enrich_version  = ENRICH_VERSION

            session.add(m)
            session.commit()   # Commit per model — crash safe
            ok += 1

            if i % 10 == 0:
                logger.info(f"  ✓ Progress: {i}/{total}")

            time.sleep(REQUEST_DELAY)

        logger.info(f"=== Done! Success: {ok} | Failed: {failed} ===")
        if failed:
            logger.info("  Re-run with --only-missing to retry failed models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Enrichment for Ollama models")
    parser.add_argument("--only-missing", action="store_true",
                        help="Process models where enrich_version < ENRICH_VERSION")
    parser.add_argument("--model", type=str, default=None,
                        help="Enrich a single model by identifier")
    args = parser.parse_args()

    enrich_models(only_missing=args.only_missing, single_slug=args.model)
