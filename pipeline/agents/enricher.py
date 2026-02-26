"""
pipeline/agents/enricher.py

F-02: Enricher Agent — instructor + Groq API

Reads unenriched models from DB, sends them to Groq LLM with a structured
prompt, and writes the enrichment results back to the DB.

Uses `instructor` to guarantee valid structured output — no manual JSON parsing.
instructor handles retries automatically (max_retries=3).

Rules:
- Never hardcode API keys — uses settings.groq_api_key
- build_prompt() logic ported from legacy/enrich.py
- Only touches DB via session — agents return data, flow.py coordinates writes
"""

import logging
from datetime import datetime

import instructor
from groq import Groq
from sqlmodel import Session, select

from pipeline.core.db import engine, init_db
from pipeline.core.models import Model
from pipeline.core.settings import settings
from pipeline.schemas.enrichment import EnrichmentOutput

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI model metadata expert specialising in Ollama open-source models.
Your job is to analyse model listings and extract accurate, structured metadata.

Rules:
- Be precise and concise — avoid filler phrases.
- Base your answers on the model name, description, capabilities, labels, and README.
- For is_uncensored: check for keywords like 'uncensored', 'abliterated', 'no restrictions', 'DAN'.
- For model_family: infer from the model name (e.g. llama3.1 → Llama, mistral → Mistral).
- For languages: include 'Multilingual' if the model supports many languages.
- For license: scan the README carefully — return null only if truly absent.
- Always respond with valid structured output matching the schema exactly.
"""


# ── Prompt Builder (ported from legacy/enrich.py) ─────────────────────────────

def clean_readme(raw: str | None, max_chars: int = 700) -> str:
    """Trim and clean README snippet for prompt inclusion."""
    import re
    if not raw:
        return "N/A"
    return re.sub(r"\s+", " ", raw).strip()[:max_chars]


def build_prompt(model: Model) -> str:
    """Build the user-facing prompt for LLM enrichment."""
    import json
    readme_snippet = clean_readme(model.readme)

    return f"""Analyse this Ollama AI model listing and return structured metadata.

Model      : {model.model_identifier}
Name       : {model.model_name}
Description: {model.description or 'N/A'}
Capabilities: {json.dumps(model.capabilities or [])}
Size labels: {json.dumps(model.labels or [])}
Min RAM    : {model.min_ram_gb} GB
README     : {readme_snippet}

Fill every field accurately based on the information above.
"""


# ── Groq Client (lazy init) ────────────────────────────────────────────────────

_client: instructor.Instructor | None = None


def get_client() -> instructor.Instructor:
    """Lazily initialise the instructor+Groq client."""
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and restart."
            )
        groq = Groq(api_key=settings.groq_api_key)
        _client = instructor.from_groq(groq, mode=instructor.Mode.JSON)
    return _client


# ── Core Enrichment Function ───────────────────────────────────────────────────

def enrich_model(model: Model) -> EnrichmentOutput | None:
    """
    Send a model to Groq LLM and return structured EnrichmentOutput.
    Returns None on failure (after instructor's internal retries).
    """
    client = get_client()
    try:
        result: EnrichmentOutput = client.chat.completions.create(
            model=settings.llm_model,
            response_model=EnrichmentOutput,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(model)},
            ],
            max_retries=3,  # instructor handles retries automatically
        )
        return result
    except Exception as e:
        logger.error(f"Enrichment failed for {model.model_identifier}: {e}")
        return None


# ── DB Query Helpers ───────────────────────────────────────────────────────────

def get_unenriched_models(
    session: Session,
    single_slug: str | None = None,
    force: bool = False,
) -> list[Model]:
    """
    Return models that need enrichment:
    - enrich_version IS NULL (never enriched)
    - enrich_version < settings.enrich_version (schema bumped)
    - If force=True: all models
    - If single_slug: only that model
    """
    query = select(Model)

    if single_slug:
        query = query.where(Model.model_identifier == single_slug)
    elif not force:
        query = query.where(
            (Model.enrich_version.is_(None))  # type: ignore[union-attr]
            | (Model.enrich_version < settings.enrich_version)
        )

    return list(session.exec(query).all())


# ── Enrichment Runner ──────────────────────────────────────────────────────────

def run_enricher(
    force: bool = False,
    single_slug: str | None = None,
) -> dict:
    """
    Enrich all unenriched (or outdated) models.
    Writes results directly to DB — commits per model for crash safety.

    Returns summary stats: {total, ok, failed, skipped}
    """
    init_db()
    ok = 0
    failed = 0

    with Session(engine) as session:
        models = get_unenriched_models(session, single_slug=single_slug, force=force)
        total = len(models)
        logger.info(
            f"=== Enricher starting — {total} models | "
            f"LLM: {settings.llm_model} | "
            f"enrich_version: {settings.enrich_version} ==="
        )

        for i, model in enumerate(models, 1):
            logger.info(f"[{i}/{total}] Enriching: {model.model_identifier}")

            result = enrich_model(model)

            if result is None:
                logger.warning(f"  → FAILED: {model.model_identifier}")
                failed += 1
                continue

            # Write all enrichment fields to model
            model.use_cases = list(result.use_cases)
            model.domain = result.domain
            model.ai_languages = list(result.languages)
            model.complexity = result.complexity
            model.best_for = result.best_for
            model.license = result.license
            model.base_model = result.base_model
            model.model_family = result.model_family
            model.is_fine_tuned = result.is_fine_tuned
            # LLM confirmation overrides crawler heuristic for is_uncensored
            model.is_uncensored = result.is_uncensored
            model.strengths = list(result.strengths)
            model.limitations = list(result.limitations)
            model.target_audience = list(result.target_audience)
            model.enrich_version = settings.enrich_version
            # Reset validation so validator re-checks
            model.validated = None
            model.validation_failed = None

            session.add(model)
            session.commit()

            logger.info(
                f"  → OK | family={result.model_family} | "
                f"domain={result.domain} | "
                f"uncensored={result.is_uncensored} | "
                f"langs={result.languages}"
            )
            ok += 1

    logger.info(f"=== Enricher done. OK: {ok} | Failed: {failed} ===")
    if failed:
        logger.info("  Re-run with --force-enrich to retry failed models.")

    return {"total": total, "ok": ok, "failed": failed}
