# ollama-pipeline — Build Instructions for AI Agent

## Context

You are building `ollama-pipeline`: a Python-based, AI-powered data pipeline that crawls
`ollama.com/library`, enriches model metadata using an LLM, validates the data, and opens
a Pull Request to `serkan-uslu/ollama-explorer` with the updated `models.json`.

Read `FEATURES.md` first to understand all features, data schemas, and enum values before
writing any code.

---

## Tech Stack

| Purpose | Library | Version |
|---|---|---|
| Web crawling | `httpx` + `beautifulsoup4` | latest |
| LLM structured output | `instructor` | latest |
| Data validation schema | `pydantic` | v2 |
| Database ORM | `sqlmodel` | latest |
| Database | SQLite | built-in |
| LLM — local | Ollama (OpenAI-compat. API) | v0.6+ |
| LLM — cloud | Groq (`groq`) | latest |
| Orchestration | `prefect` | latest |
| HTTP requests | `httpx` | latest |
| Environment vars | `pydantic-settings` | latest |
| Package manager | `poetry` | latest |

---

## Project Structure

Build exactly this structure:

```
ollama-pipeline/
├── .github/
│   └── workflows/
│       └── update-models.yml        # GitHub Actions workflow
├── pipeline/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── crawler.py               # F-01: httpx + BeautifulSoup4 scraper
│   │   ├── enricher.py              # F-02: Ollama/Groq LLM enrichment (6 focused calls)
│   │   ├── validator.py             # F-03: Data quality validation
│   │   ├── exporter.py              # F-04: JSON export
│   │   └── pr_creator.py            # F-05: GitHub PR creation
│   ├── core/
│   │   ├── __init__.py
│   │   ├── db.py                    # SQLite engine + session
│   │   ├── models.py                # SQLModel table definition
│   │   └── settings.py              # Pydantic Settings (env vars)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── enrichment.py            # Pydantic schemas for LLM output (6 schemas)
│   └── flow.py                      # Prefect flow orchestration
├── output/
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_crawler.py
│   ├── test_enricher.py
│   └── test_validator.py
├── .env.example
├── .gitignore
├── FEATURES.md
├── INSTRUCTIONS.md
├── README.md
├── pyproject.toml
└── main.py                          # CLI entrypoint
```

---

## Step-by-Step Build Order

Follow this exact order. Complete each step fully before moving to the next.

---

### STEP 1 — Project Setup

1.1 Initialize Poetry project:
```bash
poetry new ollama-pipeline --name pipeline
cd ollama-pipeline
```

1.2 Add all dependencies to `pyproject.toml`:
```bash
poetry add instructor groq openai pydantic pydantic-settings sqlmodel httpx beautifulsoup4 prefect
poetry add --group dev pytest pytest-asyncio ruff
```

1.3 Create `.env.example`:
```env
# LLM Provider: "ollama" (local, default) or "groq" (cloud)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma3:27b
ENRICH_WORKERS=3

# Only required when LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here

# GitHub PR (required for PR creation step)
CROSS_REPO_TOKEN=your_github_pat_here
GITHUB_TARGET_REPO=serkan-uslu/ollama-explorer
GITHUB_TARGET_BRANCH=data/models-update

OLLAMA_LIBRARY_URL=https://ollama.com/library
REQUEST_DELAY=0.5
ENRICH_VERSION=3
```

1.4 Create `.gitignore` — include: `.env`, `*.db`, `output/*.json`, `__pycache__`, `.prefect`

---

### STEP 2 — Core: Settings, DB, Models

2.1 **`pipeline/core/settings.py`** — Use `pydantic-settings` BaseSettings:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM provider: "ollama" (local, default) or "groq" (cloud)
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma3:27b"
    enrich_workers: int = 3

    # Only required when llm_provider="groq"
    groq_api_key: str | None = None

    # GitHub PR
    cross_repo_token: str | None = None
    github_target_repo: str = "serkan-uslu/ollama-explorer"
    github_target_branch: str = "data/models-update"

    # Crawl
    ollama_library_url: str = "https://ollama.com/library"
    request_delay: float = 0.5
    enrich_version: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
```

2.2 **`pipeline/core/models.py`** — SQLModel table exactly matching schema in FEATURES.md.
- Use `Field(default=None)` for all nullable fields
- JSON fields (lists, dicts) use `JSON` type from `sqlalchemy`
- `id` is `uuid.UUID` with `default_factory=uuid.uuid4`

2.3 **`pipeline/core/db.py`**:
```python
from sqlmodel import create_engine, Session, SQLModel

DATABASE_URL = "sqlite:///ollama.db"
engine = create_engine(DATABASE_URL, echo=False)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    SQLModel.metadata.create_all(engine)
```

---

### STEP 3 — Pydantic Schemas for LLM Output

3.1 **`pipeline/schemas/enrichment.py`**:

Use **6 focused schemas** — one per LLM call. Smaller schemas = better LLM accuracy.

```python
from pydantic import BaseModel, Field
from typing import Literal

UseCase = Literal["Chat Assistant", "Code Generation", ...]  # all from FEATURES.md
Domain = Literal["General", "Code", "Vision", ...]
ModelFamily = Literal["Llama", "Mistral", "Qwen", "Gemma", "Phi", "Other"]
Language = Literal["English", "Multilingual", ...]
Complexity = Literal["beginner", "intermediate", "advanced"]
Audience = Literal["Developers", "Beginners", ...]

class DomainFamilyOutput(BaseModel):
    domain: Domain
    model_family: ModelFamily

class UseCasesOutput(BaseModel):
    use_cases: list[UseCase] = Field(min_length=1, max_length=5)
    target_audience: list[Audience] = Field(min_length=1, max_length=3)

class BasicsOutput(BaseModel):
    complexity: Complexity
    best_for: str = Field(min_length=10, max_length=300)
    is_fine_tuned: bool
    is_uncensored: bool
    is_multimodal: bool

class SummaryOutput(BaseModel):
    strengths: list[str] = Field(min_length=1, max_length=3)
    limitations: list[str] = Field(max_length=3)  # empty list allowed

class QualityOutput(BaseModel):
    benchmark_scores: list[dict] | None = None  # [{name, score, unit}]
    parameter_sizes: list[str] | None = None

class MetadataOutput(BaseModel):
    license: str | None = None
    base_model: str | None = None
    creator_org: str | None = None
    huggingface_url: str | None = None
    ai_languages: list[Language] = Field(min_length=1, max_length=4)

# Aggregate wrapper (for exporter / type hints)
class EnrichmentOutput(BaseModel):
    domain: Domain
    model_family: ModelFamily | None = None
    use_cases: list[UseCase] = Field(min_length=1)
    target_audience: list[Audience] = Field(min_length=1)
    complexity: Complexity
    best_for: str
    is_fine_tuned: bool
    is_uncensored: bool
    is_multimodal: bool
    strengths: list[str]
    limitations: list[str]
    benchmark_scores: list[dict] | None = None
    parameter_sizes: list[str] | None = None
    license: str | None = None
    base_model: str | None = None
    creator_org: str | None = None
    huggingface_url: str | None = None
    ai_languages: list[Language]
```

Each focused schema is passed to `instructor` separately — it guarantees valid structured output per call.

---

### STEP 4 — Crawler Agent

4.1 **`pipeline/agents/crawler.py`**

Use `httpx` + `BeautifulSoup4` (no crawl4ai):

```python
import httpx
from bs4 import BeautifulSoup

async def crawl_library() -> list[dict]:
    """GET ollama.com/library, parse model cards. Returns list of model dicts."""
    ...

async def crawl_model_detail(slug: str) -> dict:
    """GET individual model page, return readme + raw_html + memory_requirements."""
    ...

async def run_full_crawl(force: bool = False) -> list[dict]:
    """
    - Fetch stats for ALL models (pulls, last_updated, description, capabilities).
    - Fetch detail page ONLY for new models (not yet in DB) or all if force=True.
    - Mark each model dict with _has_detail=True/False.
    """
    ...
```

- Parse HTML with BeautifulSoup
- Add `REQUEST_DELAY` between detail page requests
- Return structured dicts, do NOT write to DB here

4.2 Create `save_models_to_db(models: list[dict])` function:
- Upsert logic: if `model_identifier` exists → update, else → insert
- Stats fields (`pulls`, `last_updated`, `description`, `capabilities`) **always** updated
- Detail fields (`readme`, `raw_html`, `memory_requirements`, `min_ram_gb`, `context_window`, `speed_tier`) **only** updated when `raw["_has_detail"] is True`
- Commit every 20 models (checkpoint)

---

### STEP 5 — Enricher Agent

5.1 **`pipeline/agents/enricher.py`**

```python
import instructor
from openai import OpenAI
from groq import Groq
from pipeline.schemas.enrichment import (
    DomainFamilyOutput, UseCasesOutput, BasicsOutput,
    SummaryOutput, QualityOutput, MetadataOutput,
)

# Build client based on provider setting
if settings.llm_provider == "groq":
    client = instructor.from_groq(Groq(api_key=settings.groq_api_key))
else:  # ollama
    client = instructor.from_openai(
        OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    )

def enrich_model(model: Model) -> dict:
    """Run 6 focused LLM calls and return merged enrichment dict."""
    p1 = _p1_domain_family(model)   # DomainFamilyOutput
    p2 = _p2_use_cases(model)       # UseCasesOutput
    p3 = _p3_basics(model)          # BasicsOutput
    p4 = _p4_summary(model)         # SummaryOutput
    p5 = _p5_quality(model)         # QualityOutput
    p6 = _p6_metadata(model)        # MetadataOutput
    return {**p1.model_dump(), **p2.model_dump(), ...}
```

- Each `_pN_*` function calls `client.chat.completions.create(response_model=<Schema>, max_retries=3)`
- `instructor` handles LLM retries automatically — do NOT write manual retry logic
- Parallel enrichment via `ThreadPoolExecutor(max_workers=settings.enrich_workers)`
- After merging results, write all fields to DB model and set `enrich_version`

5.2 `get_unenriched_models()` — query DB for models where `enrich_version IS NULL` or `enrich_version < settings.enrich_version`

---

### STEP 6 — Validator Agent

6.1 **`pipeline/agents/validator.py`**

```python
from pipeline.core.models import Model

VALIDATION_RULES = {
    "domain": lambda v: v is not None,
    "use_cases": lambda v: v and len(v) >= 1,
    "ai_languages": lambda v: v and len(v) >= 1,
    "complexity": lambda v: v is not None,
    "best_for": lambda v: v and len(v) >= 10,
    "strengths": lambda v: v and len(v) >= 1,
    "limitations": lambda v: v is not None,  # None fails; empty list [] passes
}

def validate_model(model: Model) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_failed_rules)."""
    ...

def validate_all(session: Session) -> dict:
    """Validate all enriched models. Returns summary stats."""
    ...
```

- If validation fails: set `validated=False`, increment `model.validation_retries` in DB (persistent across runs)
- If `model.validation_retries < 3`: reset `enrich_version` to None (triggers re-enrichment on next run or Step 3b)
- If `model.validation_retries >= 3`: set `validation_failed=True`, log warning
- On success: set `validated=True`, reset `validation_retries = 0`

---

### STEP 7 — Exporter

7.1 **`pipeline/agents/exporter.py`**

```python
def export_to_json(output_path: str = "output/models.json") -> dict:
    """Export all validated models to JSON. Returns export stats."""
    ...
```

- Query only models where `validated=True` OR `validation_failed IS NULL` (include all enriched)
- Use `model_to_dict()` — reuse from `legacy/export_json.py`
- Sort models by `pulls` descending
- Write to `output/models.json`
- Return stats: `{total, enriched, validated, validation_failed, exported}`

---

### STEP 8 — PR Creator

8.1 **`pipeline/agents/pr_creator.py`**

```python
import httpx

def create_pull_request(models_json_path: str) -> str:
    """
    1. Get current models.json from target repo via GitHub API
    2. Compare with new models.json
    3. Create/update branch with new file content
    4. Open PR if changes detected
    Returns PR URL.
    """
    ...
```

Use GitHub REST API directly with `httpx` — no need for PyGitHub:
- `GET /repos/{owner}/{repo}/contents/{path}` → get current file + sha
- `PUT /repos/{owner}/{repo}/contents/{path}` → update file on branch
- `POST /repos/{owner}/{repo}/pulls` → create PR

PR body should include:
```markdown
## 🦙 Ollama Models Update — {date}

**Changes:**
- Total models: {N}
- New models: {new_count}
- Updated models: {updated_count}

**Pipeline stats:**
- Enriched: {enriched}/{total}
- Validated: {validated}/{total}
- Failed validation: {failed}

*Auto-generated by [ollama-pipeline](https://github.com/serkan-uslu/ollama-pipeline)*
```

---

### STEP 9 — Prefect Flow

9.1 **`pipeline/flow.py`**

```python
from prefect import flow, task

@task(name="crawler", retries=3, retry_delay_seconds=30)
def run_crawler_task(force: bool = False): ...

@task(name="enricher", retries=2, retry_delay_seconds=10)
def run_enricher_task(force: bool = False, single_slug: str | None = None): ...

@task(name="validator", retries=2)
def run_validator_task(): ...

@task(name="exporter")
def run_exporter_task(): ...

@task(name="pr-creator")
def run_pr_creator_task(export_stats: dict | None = None): ...

@flow(name="ollama-pipeline", log_prints=True)
def ollama_pipeline(
    skip_crawl: bool = False,
    skip_enrich: bool = False,
    force_crawl: bool = False,
    force_enrich: bool = False,
    model: str | None = None,
    dry_run: bool = False,
):
    init_db()
    if not skip_crawl:
        run_crawler_task(force=force_crawl)
    if not skip_enrich:
        run_enricher_task(force=force_enrich, single_slug=model)
    if not dry_run:
        val_stats = run_validator_task()
        # Step 3b: re-enrich + re-validate if any models were re-queued
        if val_stats.get("re_queued", 0) > 0:
            run_enricher_task(force=False, single_slug=None)
            val_stats = run_validator_task()
        export_stats = run_exporter_task()
        run_pr_creator_task(export_stats=export_stats)
```

---

### STEP 10 — CLI Entrypoint

10.1 **`main.py`**:

```python
import argparse
import asyncio
from pipeline.core.db import init_db
from pipeline.flow import ollama_pipeline

def main():
    parser = argparse.ArgumentParser(description="Ollama Pipeline")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--force-crawl", action="store_true")
    parser.add_argument("--force-enrich", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    init_db()
    asyncio.run(ollama_pipeline(
        skip_crawl=args.skip_crawl,
        skip_enrich=args.skip_enrich,
        force_crawl=args.force_crawl,
        force_enrich=args.force_enrich,
        model=args.model,
        dry_run=args.dry_run,
    ))

if __name__ == "__main__":
    main()
```

---

### STEP 11 — GitHub Actions Workflow

11.1 **`.github/workflows/update-models.yml`**:

```yaml
name: Update Ollama Models

on:
  schedule:
    - cron: '0 3 * * 1'  # Every Monday at 03:00 UTC
  workflow_dispatch:
    inputs:
      skip_crawl:
        description: 'Skip crawling (use existing DB)'
        type: boolean
        default: false
      force_enrich:
        description: 'Force re-enrich all models'
        type: boolean
        default: false

jobs:
  update-models:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Checkout pipeline repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Poetry
        run: pip install poetry

      - name: Install dependencies
        run: poetry install --no-root

      - name: Run pipeline
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          CROSS_REPO_TOKEN: ${{ secrets.CROSS_REPO_TOKEN }}
        run: |
          poetry run python main.py \
            ${{ inputs.skip_crawl && '--skip-crawl' || '' }} \
            ${{ inputs.force_enrich && '--force-enrich' || '' }}

      - name: Upload DB artifact (for debugging)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: ollama-db
          path: ollama.db
          retention-days: 7
```

---

### STEP 12 — README

12.1 Write `README.md` with:
- Project description + architecture diagram (ASCII)
- Pipeline flow diagram: `Crawler → Enricher → Validator → Exporter → PR Creator`
- Setup instructions
- CLI usage examples
- Environment variables table
- GitHub Actions setup (how to add secrets)
- Tech stack badges (shields.io)
- Link to `ollama-explorer` (the frontend that consumes this data)

---

## Important Rules for the AI Agent

1. **Read FEATURES.md before writing any code** — all enum values, schemas, and field definitions are there.
2. **Do not use `requests` library** — use `httpx` for all HTTP calls (async-compatible).
3. **Never hardcode API keys** — always use `settings` from `pipeline/core/settings.py`.
4. **Separation of concerns** — agents return data; `flow.py` and `save_models_to_db` coordinate DB writes.
5. **Type hints everywhere** — use Python 3.12+ syntax (`str | None`, `list[str]`).
6. **Error handling** — every agent function must have try/except with proper logging.
7. **`_has_detail` flag is critical** — always set it on crawler dicts; `save_models_to_db` uses it to avoid wiping existing README/HTML with empty data.
8. **`instructor` handles LLM retries** — do NOT write manual retry logic for LLM calls; instructor does it automatically with `max_retries=3`.
9. **`validation_retries` is persistent** — use `model.validation_retries` (DB field), NOT an in-memory dict. It survives between pipeline runs.
10. **6 focused LLM calls per model** — do NOT use a single `build_prompt` / `EnrichmentOutput`. Use separate `_p1_*` through `_p6_*` functions with their own small schemas.
11. **Test each agent independently** — write at least one test per agent in `/tests`.
12. **Commit after each step** — each step should result in a working, committable state.
