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

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import instructor
from bs4 import BeautifulSoup
from openai import OpenAI
from sqlmodel import Session, select

from pipeline.core.db import engine, init_db
from pipeline.core.models import Model
from pipeline.core.settings import settings
from pipeline.schemas.enrichment import (
    DomainFamilyOutput,
    UseCasesOutput,
    BasicsOutput,
    SummaryOutput,
    QualityOutput,
    EnrichmentOutput,
    MetadataOutput,
)

logger = logging.getLogger(__name__)

# ── HTML Section Extractor ────────────────────────────────────────────────────

def _extract_html_sections(html: str) -> dict:
    """
    Extract structured text sections from raw HTML using BeautifulSoup.
    Returns a dict of named sections that get injected into the enrichment prompt.
    """
    soup = BeautifulSoup(html, "html.parser")
    sections: dict = {}

    # README / main content
    readme_el = (
        soup.find(id=re.compile(r"readme", re.I))
        or soup.find(class_=re.compile(r"readme|markdown|content", re.I))
        or soup.find("article")
        or soup.find("main")
    )
    if readme_el:
        raw = readme_el.get_text(separator=" ", strip=True)
        raw = re.sub(r"\s+", " ", raw).strip()
        sections["readme"] = raw[:2500]

    # Full page text for regex extraction
    full_text = soup.get_text(" ", strip=True)

    # HuggingFace URLs
    hf_links = list(dict.fromkeys(
        re.findall(r"https?://huggingface\.co/[^\s\"'>]+", full_text)
    ))[:5]
    if hf_links:
        sections["hf_links"] = hf_links

    # Context window / token mentions
    ctx = list(dict.fromkeys(
        re.findall(r"\b(\d+[KkMm]?\s*(?:context|token|window)s?)", full_text, re.I)
    ))[:5]
    if ctx:
        sections["context_mentions"] = ctx

    # License
    lic = list(dict.fromkeys(
        re.findall(
            r"\b(MIT|Apache[\s\-]2\.0|Llama[\s\w]*License|Gemma[\s\w]*(?:Terms|License)"
            r"|DeepSeek License|Qwen License|CC BY[\s\d\.]+|GPL[\s\-]?\d?|AGPL"
            r"|Proprietary|Commercial|Apache License)",
            full_text, re.I
        )
    ))[:3]
    if lic:
        sections["license_mentions"] = lic

    # Benchmark scores with numbers
    benchmarks = re.findall(
        r"\b((?:MMLU|HumanEval|GSM8K|SWE-bench|MATH|HellaSwag|ARC|MBPP|BigBench|TruthfulQA|WinoGrande|PIQA|GPQA|AIME|AMC)[^\n]{0,60}?\d+\.?\d*\s*%?)",
        full_text, re.I
    )[:8]
    if benchmarks:
        sections["benchmark_mentions"] = list(dict.fromkeys(benchmarks))

    # Applications section
    apps: list[str] = []
    for h in soup.find_all(["h2", "h3"]):
        if "application" in h.get_text(strip=True).lower():
            sib = h.find_next_sibling()
            while sib and sib.name not in ("h2", "h3"):
                t = sib.get_text(strip=True)
                if t:
                    apps.append(t)
                sib = sib.find_next_sibling()
    if apps:
        sections["applications_raw"] = apps[:8]

    return sections


def clean_readme(raw: str | None, max_chars: int = 1200) -> str:
    """Trim and clean README snippet for prompt inclusion."""
    if not raw:
        return "N/A"
    return re.sub(r"\s+", " ", raw).strip()[:max_chars]


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


# ── Fuzzy mappers (small models ignore enum constraints) ──────────────────────

_DOMAIN_MAP = {
    "natural language processing": "Language", "nlp": "Language", "language": "Language",
    "general purpose": "General", "general": "General", "text": "General",
    "code": "Code", "coding": "Code", "programming": "Code",
    "vision": "Vision", "image": "Vision",
    "multimodal": "Multimodal", "multi-modal": "Multimodal",
    "embedding": "Embedding", "embeddings": "Embedding",
    "reasoning": "Reasoning", "logic": "Reasoning",
    "math": "Math", "mathematics": "Math", "mathematical": "Math",
    "medical": "Medical", "medicine": "Medical", "health": "Medical",
    "science": "Science", "scientific": "Science",
    "audio": "Audio", "speech": "Audio",
    "finance": "Finance", "financial": "Finance",
    "legal": "Legal", "law": "Legal",
    "education": "Education", "educational": "Education",
}
_VALID_DOMAINS = {"General","Code","Vision","Multimodal","Embedding","Reasoning","Math","Medical","Science","Language","Audio","Finance","Legal","Education"}

_USE_CASE_MAP = {
    "mathematics": "Math", "mathematical reasoning": "Math",
    "text analysis": "Data Analysis", "analysis": "Data Analysis",
    "information retrieval": "RAG / Retrieval", "retrieval": "RAG / Retrieval", "rag": "RAG / Retrieval",
    "content generation": "Creative Writing", "content creation": "Creative Writing",
    "document classification": "Document Processing", "classification": "Document Processing",
    "general purpose language models": "Chat Assistant", "general purpose": "Chat Assistant",
    "instruction following": "Chat Assistant",
    "embeddings": "Text Embedding", "text embeddings": "Text Embedding",
    "image understanding": "Image Understanding", "vision": "Image Understanding",
    "audio processing": "Audio Processing", "speech": "Audio Processing",
    "automation": "Agent / Automation", "agent": "Agent / Automation",
    "function calling": "Function Calling", "tool use": "Function Calling",
}
_VALID_USE_CASES = {"Chat Assistant","Role Play","Creative Writing","Code Generation","Code Review","Code Explanation","Question Answering","Reasoning","Math","Text Summarization","Translation","RAG / Retrieval","Text Embedding","Image Understanding","Video Understanding","Audio Processing","Function Calling","Agent / Automation","Data Analysis","Document Processing"}

_AUDIENCE_MAP = {
    "researcher": "Researchers", "researchers": "Researchers",
    "data scientist": "Data Scientists", "data scientists": "Data Scientists",
    "developer": "Developers", "developers": "Developers",
    "student": "Students", "students": "Students",
    "beginner": "Beginners", "beginners": "Beginners",
    "devops": "DevOps",
    "content creator": "Content Creators", "content creators": "Content Creators",
    "educator": "Educators", "educators": "Educators",
    "medical professional": "Medical Professionals", "medical professionals": "Medical Professionals",
    "legal professional": "Legal Professionals", "legal professionals": "Legal Professionals",
    "business analyst": "Business Analysts", "business analysts": "Business Analysts",
}
_VALID_AUDIENCES = {"Developers","Beginners","Researchers","Data Scientists","DevOps","Students","Content Creators","Educators","Medical Professionals","Legal Professionals","Business Analysts"}

_LANGUAGE_MAP = {
    "multilingual": "Multilingual", "multi-lingual": "Multilingual",
    "chinese": "Chinese", "mandarin": "Chinese",
    "japanese": "Japanese", "korean": "Korean",
    "french": "French", "german": "German", "spanish": "Spanish",
    "portuguese": "Portuguese", "italian": "Italian", "russian": "Russian",
    "arabic": "Arabic", "hindi": "Hindi", "turkish": "Turkish",
    "polish": "Polish", "dutch": "Dutch", "swedish": "Swedish",
    "romanian": "Romanian", "czech": "Czech", "ukrainian": "Ukrainian",
    "vietnamese": "Vietnamese", "thai": "Thai", "indonesian": "Indonesian",
    "persian": "Persian", "urdu": "Urdu",
}
_VALID_LANGUAGES = {"English","Multilingual","Chinese","Japanese","Korean","Vietnamese","Thai","Indonesian","Hindi","Arabic","Persian","Turkish","Urdu","French","German","Spanish","Portuguese","Italian","Russian","Polish","Dutch","Swedish","Romanian","Czech","Ukrainian"}

_FAMILY_MAP = {
    "llama": "Llama", "llama3": "Llama", "llama2": "Llama",
    "mistral": "Mistral", "mixtral": "Mistral",
    "gemma": "Gemma", "phi": "Phi", "qwen": "Qwen",
    "deepseek": "DeepSeek", "yi": "Yi", "command": "Command", "falcon": "Falcon",
    "granite": "Granite", "orca": "Orca", "dolphin": "Dolphin",
    "bloom": "BLOOM", "bert": "BERT", "nomic": "Nomic", "mxbai": "mxbai",
    "llava": "LLaVA", "llava:": "LLaVA", "moondream": "Moondream", "bakllava": "BakLLaVA",
    "vicuna": "Vicuna", "wizardlm": "WizardLM", "codellama": "CodeLlama",
    "starcoder": "StarCoder", "sqlcoder": "SQLCoder", "openhermes": "OpenHermes",
}
_VALID_FAMILIES = {"Llama","Mistral","Gemma","Phi","Qwen","DeepSeek","Yi","Command","Falcon","Granite","Orca","Dolphin","BLOOM","BERT","Nomic","mxbai","LLaVA","Moondream","BakLLaVA","Vicuna","WizardLM","CodeLlama","StarCoder","SQLCoder","OpenHermes","Other"}

_COMPLEXITY_MAP = {"beginner": "beginner", "easy": "beginner", "simple": "beginner",
                   "intermediate": "intermediate", "medium": "intermediate",
                   "advanced": "advanced", "expert": "advanced", "hard": "advanced"}


def _map_domain(val: str) -> str:
    if val in _VALID_DOMAINS:
        return val
    return _DOMAIN_MAP.get(val.lower().strip(), "General")


def _map_use_cases(vals: list[str]) -> list[str]:
    result = []
    for v in vals:
        if v in _VALID_USE_CASES:
            result.append(v)
        else:
            mapped = _USE_CASE_MAP.get(v.lower().strip())
            if mapped:
                result.append(mapped)
    return list(dict.fromkeys(result)) or ["Chat Assistant"]


def _map_languages(vals: list[str]) -> list[str]:
    result = []
    for v in vals:
        if v in _VALID_LANGUAGES:
            result.append(v)
        else:
            mapped = _LANGUAGE_MAP.get(v.lower().strip())
            if mapped:
                result.append(mapped)
    result = list(dict.fromkeys(result))[:8]
    return result or ["English"]


def _map_family(val: str) -> str:
    if val in _VALID_FAMILIES:
        return val
    key = val.lower().strip()
    for prefix, fam in _FAMILY_MAP.items():
        if key.startswith(prefix) or prefix in key:
            return fam
    return "Other"


def _map_audience(vals: list[str]) -> list[str]:
    result = []
    for v in vals:
        if v in _VALID_AUDIENCES:
            result.append(v)
        else:
            mapped = _AUDIENCE_MAP.get(v.lower().strip())
            if mapped:
                result.append(mapped)
    result = list(dict.fromkeys(result))[:4]
    return result or ["Developers"]


def _map_complexity(val: str) -> str:
    return _COMPLEXITY_MAP.get(val.lower().strip(), "intermediate")


# ── 6-Call Prompts (2-4 fields each) ─────────────────────────────────────────

_SYS = "You are an AI model metadata expert. Respond ONLY with valid JSON. Use EXACT values listed."

def _p1_domain_family(model: Model) -> str:
    return f"""Model: {model.model_identifier} — {model.description or ''}

Return JSON with exactly 2 fields:
- "domain": pick ONE from: "General" | "Code" | "Vision" | "Multimodal" | "Embedding" | "Reasoning" | "Math" | "Medical" | "Science" | "Language" | "Audio" | "Finance" | "Legal" | "Education"
- "model_family": pick ONE from: "Llama" | "Mistral" | "Gemma" | "Phi" | "Qwen" | "DeepSeek" | "Yi" | "Command" | "Falcon" | "Granite" | "Orca" | "Dolphin" | "BLOOM" | "BERT" | "Nomic" | "mxbai" | "LLaVA" | "Moondream" | "BakLLaVA" | "Vicuna" | "WizardLM" | "CodeLlama" | "StarCoder" | "SQLCoder" | "OpenHermes" | "Other"

Hints: embedding models → domain="Embedding". Infer family from name."""


def _p2_use_cases(model: Model) -> str:
    return f"""Model: {model.model_identifier} — {model.description or ''}

Return JSON with exactly 1 field:
- "use_cases": list of 1-4 items, pick ONLY from: "Chat Assistant" | "Role Play" | "Creative Writing" | "Code Generation" | "Code Review" | "Code Explanation" | "Question Answering" | "Reasoning" | "Math" | "Text Summarization" | "Translation" | "RAG / Retrieval" | "Text Embedding" | "Image Understanding" | "Video Understanding" | "Audio Processing" | "Function Calling" | "Agent / Automation" | "Data Analysis" | "Document Processing"

Use ONLY the exact strings above."""


def _p3_basics(model: Model) -> str:
    return f"""Model: {model.model_identifier} — {model.description or ''} — sizes: {json.dumps(model.labels or [])}

Return JSON with exactly 4 fields:
- "languages": list of 1-6 items, pick from: "English" | "Multilingual" | "Chinese" | "Japanese" | "Korean" | "French" | "German" | "Spanish" | "Portuguese" | "Arabic" | "Hindi" | "Turkish" | "Russian" | "Italian" | "Vietnamese" | "Thai" | "Indonesian" | "Polish" | "Dutch" | "Swedish" | "Romanian" | "Czech" | "Ukrainian" | "Persian" | "Urdu"
- "complexity": "beginner" (runs on <8GB RAM) | "intermediate" (8-32GB) | "advanced" (>32GB)
- "is_fine_tuned": true if instruct/chat/fine-tuned variant, false if base model
- "is_uncensored": true ONLY if name contains 'uncensored' or 'abliterated', otherwise false

If language unknown, use ["English"]. Use ONLY exact strings."""


def _p4_summary(model: Model, readme: str) -> str:
    return f"""Model: {model.model_identifier} — {model.description or ''}
README: {readme}

Return JSON with exactly 2 fields:
- "best_for": one sentence (15-150 chars) describing the ideal use case
- "license": license name like "MIT", "Apache 2.0", "Llama 3 Community License", "Gemma Terms of Use", "CC BY 4.0" — or null if not mentioned"""


def _p5_quality(model: Model) -> str:
    return f"""Model: {model.model_identifier} — {model.description or ''}

Return JSON with exactly 3 fields:
- "strengths": list of 2-3 short strings about what this model excels at
- "limitations": list of 0-2 short strings about weaknesses (use [] if none known)
- "target_audience": list of 1-3 items, pick ONLY from: "Developers" | "Beginners" | "Researchers" | "Data Scientists" | "DevOps" | "Students" | "Content Creators" | "Educators" | "Medical Professionals" | "Legal Professionals" | "Business Analysts"

Use ONLY the exact target_audience strings above."""


def _p6_metadata(model: Model, html_sections: dict) -> str:
    hf = html_sections.get('hf_links', [])
    bench = html_sections.get('benchmark_mentions', [])
    sizes = json.dumps(model.labels or [])
    return f"""Model: {model.model_identifier} — {model.description or ''}
Size labels: {sizes}
HuggingFace links: {hf or 'none'}
Benchmark mentions: {bench or 'none'}

Return JSON with exactly 6 fields:
- "creator_org": org name like "Meta", "Mistral AI", "Google DeepMind", "Microsoft", "Alibaba Cloud", "DeepSeek", "IBM", "Nomic AI" — infer from model name, or null
- "is_multimodal": true only if model processes images/audio, false for text-only
- "base_model": base model name if this is a fine-tune (e.g. "llama3.1"), null if this IS the base
- "huggingface_url": exact URL from HuggingFace links above, or null
- "benchmark_scores": list of {{"name":"MMLU","score":85.2,"unit":"%"}} from benchmark mentions, or null
- "parameter_sizes": list like ["7B","70B"] from size labels, null if single size"""


# ── Small helper ──────────────────────────────────────────────────────────────

def _call(client, response_model, prompt: str, label: str, model_id: str):
    """Make a single instructor call; return None and log on failure."""
    try:
        return client.chat.completions.create(
            model=settings.llm_model,
            response_model=response_model,
            messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}],
            max_retries=2,
        )
    except Exception as e:
        logger.error(f"[{label}] failed for {model_id}: {e}")
        return None


# ── Core Enrichment Function (thread-safe — no DB access) ─────────────────────

def enrich_model(model: Model) -> EnrichmentOutput | None:
    """6 small focused LLM calls → merge into EnrichmentOutput.

    Calls 1-5 are required (core metadata). Call 6 (html metadata) is best-effort:
    if it fails, falls back to None values rather than dropping the whole model.
    """
    client = _make_client()
    html_sections = _extract_html_sections(model.raw_html) if model.raw_html else {}
    # Use html-extracted readme when available, fall back to stored readme
    readme = html_sections.get("readme") or clean_readme(model.readme, max_chars=500)
    mid = model.model_identifier

    c1 = _call(client, DomainFamilyOutput, _p1_domain_family(model), "Call1/DomainFamily", mid)
    if not c1: return None

    c2 = _call(client, UseCasesOutput, _p2_use_cases(model), "Call2/UseCases", mid)
    if not c2: return None

    c3 = _call(client, BasicsOutput, _p3_basics(model), "Call3/Basics", mid)
    if not c3: return None

    c4 = _call(client, SummaryOutput, _p4_summary(model, readme), "Call4/Summary", mid)
    if not c4: return None

    c5 = _call(client, QualityOutput, _p5_quality(model), "Call5/Quality", mid)
    if not c5: return None

    # Call 6 is best-effort — html metadata enriches but isn't required
    c6 = _call(client, MetadataOutput, _p6_metadata(model, html_sections), "Call6/Metadata", mid)

    return EnrichmentOutput(
        domain=_map_domain(c1.domain),
        model_family=_map_family(c1.model_family),
        use_cases=_map_use_cases(c2.use_cases),
        languages=_map_languages(c3.languages),
        complexity=_map_complexity(c3.complexity),
        is_fine_tuned=c3.is_fine_tuned,
        is_uncensored=c3.is_uncensored,
        best_for=c4.best_for,
        license=c4.license,
        strengths=c5.strengths,
        limitations=c5.limitations or [],
        target_audience=_map_audience(c5.target_audience),
        creator_org=c6.creator_org if c6 and c6.creator_org and c6.creator_org.lower() not in ("unknown", "n/a") else None,
        is_multimodal=c6.is_multimodal if c6 else False,
        base_model=c6.base_model if c6 else None,
        huggingface_url=c6.huggingface_url if c6 else None,
        benchmark_scores=c6.benchmark_scores if c6 else None,
        parameter_sizes=c6.parameter_sizes if c6 else None,
    )


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
                db_model.creator_org = result.creator_org
                db_model.is_multimodal = result.is_multimodal
                db_model.huggingface_url = result.huggingface_url
                db_model.benchmark_scores = result.benchmark_scores
                db_model.parameter_sizes = result.parameter_sizes
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
