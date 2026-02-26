"""
pipeline/agents/enricher.py

F-02: Enricher Agent — instructor + Groq API OR local Ollama

Reads unenriched models from DB, sends them to LLM with a structured
prompt, and writes the enrichment results back to the DB.

Providers:
  - "groq"   → Groq cloud API (GROQ_API_KEY required)
  - "ollama" → Local Ollama via OpenAI-compatible API (no key needed)

Parallelism:
  - ThreadPoolExecutor with ENRICH_WORKERS workers
  - Workers do only LLM I/O (thread-safe, no DB access)
  - Main thread collects results and writes to DB sequentially

instructor handles LLM retries automatically (max_retries=3).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import instructor
from openai import OpenAI
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


# ── Client Factory ─────────────────────────────────────────────────────────────

def _make_client() -> instructor.Instructor:
    """
    Create a fresh instructor client based on LLM_PROVIDER setting.
    Called once per worker thread — each thread gets its own client.
    """
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from groq import Groq
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return instructor.from_groq(Groq(api_key=settings.groq_api_key), mode=instructor.Mode.JSON)

    elif provider == "ollama":
        # Ollama exposes an OpenAI-compatible API at localhost:11434/v1
        openai_client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # Ollama ignores the key but OpenAI client requires it
        )
        return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'groq' or 'ollama'.")


# ── Core Enrichment Function (thread-safe — no DB access) ─────────────────────

def enrich_model(model: Model) -> EnrichmentOutput | None:
    """
    Send a model to LLM and return structured EnrichmentOutput.
    Thread-safe: creates its own client, no shared state.
    Returns None on failure.
    """
    # Each call creates a fresh client — safe for ThreadPoolExecutor
    client = _make_client()
    try:
        result: EnrichmentOutput = client.chat.completions.create(
            model=settings.llm_model,
            response_model=EnrichmentOutput,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(model)},
            ],
            max_retries=3,
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
    - force=True: all models
    - single_slug: only that one model
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


# ── Parallel Enrichment Runner ────────────────────────────────────────────────

def run_enricher(
    force: bool = False,
    single_slug: str | None = None,
) -> dict:
    """
    Enrich all unenriched (or outdated) models in parallel.

    Design:
    - ThreadPoolExecutor workers do LLM I/O only (thread-safe)
    - Main thread writes results to DB sequentially (SQLite-safe)
    - Commits per model for crash safety

    Returns summary stats: {total, ok, failed, provider, workers}
    """
    init_db()
    ok = 0
    failed = 0

    workers = settings.enrich_workers if not single_slug else 1
    provider = settings.llm_provider

    with Session(engine) as session:
        models = get_unenriched_models(session, single_slug=single_slug, force=force)
        total = len(models)

        print(f"\n[ENRICHER] {'─'*60}", flush=True)
        print(f"[ENRICHER] 🤖 Başlıyor", flush=True)
        print(f"[ENRICHER]    Modeller  : {total}", flush=True)
        print(f"[ENRICHER]    Provider  : {provider.upper()}", flush=True)
        print(f"[ENRICHER]    LLM Model : {settings.llm_model}", flush=True)
        print(f"[ENRICHER]    Workers   : {workers} paralel", flush=True)
        print(f"[ENRICHER]    Versiyon  : {settings.enrich_version}", flush=True)
        logger.info(
            f"=== Enricher starting === {total} models | {provider} | {settings.llm_model} | {workers} workers"
        )

        if total == 0:
            print("[ENRICHER] ✅ Tüm modeller zaten enriched, yapacak iş yok", flush=True)
            logger.info("=== Nothing to enrich ===")
            return {"total": 0, "ok": 0, "failed": 0, "provider": provider, "workers": workers}

        # ── Submit all LLM jobs in parallel ───────────────────────────────────
        # future → model  mapping so we can write results in order
        future_to_model: dict = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for model in models:
                future = executor.submit(enrich_model, model)
                future_to_model[future] = model

            completed = 0
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                completed += 1

                try:
                    result: EnrichmentOutput | None = future.result()
                except Exception as e:
                    print(f"[ENRICHER] [{completed:>3}/{total}] ❌ HATA: {model.model_identifier} → {e}", flush=True)
                    logger.error(f"[{completed}/{total}] Worker exception for {model.model_identifier}: {e}")
                    failed += 1
                    continue

                if result is None:
                    print(f"[ENRICHER] [{completed:>3}/{total}] ⚠️  BAŞARISIZ: {model.model_identifier}", flush=True)
                    logger.warning(f"[{completed}/{total}] FAILED: {model.model_identifier}")
                    failed += 1
                    continue

                # ── Write to DB (main thread, sequential) ─────────────────────
                # Re-fetch from session to ensure we have the latest state
                db_model = session.get(Model, model.id)
                if db_model is None:
                    logger.warning(f"  Model {model.model_identifier} not found in DB, skipping write")
                    failed += 1
                    continue

                db_model.use_cases = list(result.use_cases)
                db_model.domain = result.domain
                db_model.ai_languages = list(result.languages)
                db_model.complexity = result.complexity
                db_model.best_for = result.best_for
                db_model.license = result.license
                db_model.base_model = result.base_model
                db_model.model_family = result.model_family
                db_model.is_fine_tuned = result.is_fine_tuned
                db_model.is_uncensored = result.is_uncensored
                db_model.strengths = list(result.strengths)
                db_model.limitations = list(result.limitations)
                db_model.target_audience = list(result.target_audience)
                db_model.enrich_version = settings.enrich_version
                db_model.validated = None
                db_model.validation_failed = None

                session.add(db_model)
                session.commit()

                uncensored_flag = " 🔞" if result.is_uncensored else ""
                print(
                    f"[ENRICHER] [{completed:>3}/{total}] ✅ {model.model_identifier}"
                    f" │ {result.model_family or '?'} │ {result.domain}"
                    f" │ {result.complexity} │ langs={result.languages}{uncensored_flag}",
                    flush=True,
                )
                logger.info(
                    f"[{completed}/{total}] ✓ {model.model_identifier}"
                    f" | family={result.model_family}"
                    f" | domain={result.domain}"
                    f" | uncensored={result.is_uncensored}"
                )
                ok += 1

    print(f"[ENRICHER] {'─'*60}", flush=True)
    print(f"[ENRICHER] 🏁 Tamamlandı — ✅ OK: {ok} | ❌ Başarısız: {failed} | Provider: {provider.upper()}", flush=True)
    logger.info(f"=== Enricher done — OK: {ok} | Failed: {failed} | Provider: {provider.upper()} ===")
    return {"total": total, "ok": ok, "failed": failed, "provider": provider, "workers": workers}
