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
| Web crawling | `crawl4ai` | latest |
| LLM structured output | `instructor` | latest |
| Data validation schema | `pydantic` | v2 |
| Database ORM | `sqlmodel` | latest |
| Database | SQLite | built-in |
| LLM API | Groq (`groq`) | latest |
| Orchestration | `prefect` | latest |
| HTTP requests | `httpx` | latest |
| Environment vars | `python-dotenv` | latest |
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
│   │   ├── crawler.py               # F-01: Crawl4AI scraper
│   │   ├── enricher.py              # F-02: Groq LLM enrichment
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
│   │   └── enrichment.py            # Pydantic schemas for LLM output
│   └── flow.py                      # Prefect flow orchestration
├── output/
│   └── .gitkeep
├── legacy/
│   ├── scraper.py                   # Old scraper (reference only)
│   ├── enrich.py                    # Old enricher (reference only)
│   └── export_json.py               # Old exporter (reference only)
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
poetry add crawl4ai instructor groq pydantic sqlmodel httpx python-dotenv prefect
poetry add --group dev pytest pytest-asyncio ruff
```

1.3 Create `.env.example`:
```env
GROQ_API_KEY=your_groq_api_key_here
CROSS_REPO_TOKEN=your_github_pat_here
GITHUB_TARGET_REPO=serkan-uslu/ollama-explorer
GITHUB_TARGET_BRANCH=data/models-update
OLLAMA_LIBRARY_URL=https://ollama.com/library
REQUEST_DELAY=0.5
ENRICH_VERSION=1
LLM_MODEL=llama-3.3-70b-versatile
```

1.4 Create `.gitignore` — include: `.env`, `*.db`, `output/*.json`, `__pycache__`, `.prefect`

---

### STEP 2 — Core: Settings, DB, Models

2.1 **`pipeline/core/settings.py`** — Use `pydantic-settings` BaseSettings:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    cross_repo_token: str
    github_target_repo: str = "serkan-uslu/ollama-explorer"
    github_target_branch: str = "data/models-update"
    ollama_library_url: str = "https://ollama.com/library"
    request_delay: float = 0.5
    enrich_version: int = 1
    llm_model: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"

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

```python
from pydantic import BaseModel, Field
from typing import Literal

UseCase = Literal["Chat Assistant", "Code Generation", ...]  # all from FEATURES.md
Domain = Literal["General", "Code", "Vision", ...]
Language = Literal["English", "Multilingual", ...]
Complexity = Literal["beginner", "intermediate", "advanced"]
Audience = Literal["Developers", "Beginners", ...]

class EnrichmentOutput(BaseModel):
    use_cases: list[UseCase] = Field(min_length=1, max_length=5)
    domain: Domain
    languages: list[Language] = Field(min_length=1, max_length=4)
    complexity: Complexity
    best_for: str = Field(min_length=10, max_length=300)
    license: str | None = None
    base_model: str | None = None
    is_fine_tuned: bool
    strengths: list[str] = Field(min_length=1, max_length=3)
    limitations: list[str] = Field(min_length=1, max_length=3)
    target_audience: list[Audience] = Field(min_length=1, max_length=3)
```

This schema is passed to `instructor` — it guarantees the LLM always returns valid data.

---

### STEP 4 — Crawler Agent

4.1 **`pipeline/agents/crawler.py`**

Use `AsyncWebCrawler` from `crawl4ai`:

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async def crawl_library() -> list[dict]:
    """Crawl ollama.com/library and return list of model dicts."""
    ...

async def crawl_model_detail(slug: str) -> dict:
    """Crawl individual model page, return readme + memory_requirements."""
    ...
```

- Parse HTML with BeautifulSoup after crawl4ai fetches it
- Reuse parsing logic from `legacy/scraper.py` (`parse_pulls`, `parse_last_updated`, memory requirements regex)
- Add `REQUEST_DELAY` between detail page requests
- Return structured dicts, do NOT write to DB here (separation of concerns)

4.2 Create `save_models_to_db(models: list[dict])` function:
- Upsert logic: if `model_identifier` exists → update, else → insert
- Commit every 20 models (checkpoint)

---

### STEP 5 — Enricher Agent

5.1 **`pipeline/agents/enricher.py`**

```python
import instructor
from groq import Groq
from pipeline.schemas.enrichment import EnrichmentOutput

client = instructor.from_groq(Groq(api_key=settings.groq_api_key))

def enrich_model(model: Model) -> EnrichmentOutput:
    """Send model data to Groq LLM and return structured enrichment."""
    return client.chat.completions.create(
        model=settings.llm_model,
        response_model=EnrichmentOutput,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(model)},
        ],
        max_retries=3,  # instructor handles retries automatically
    )
```

- `SYSTEM_PROMPT`: "You are an AI model metadata expert. Analyze Ollama model listings and extract structured metadata. Always respond with accurate, concise information."
- `build_prompt(model)`: Include model_identifier, description, capabilities, labels, readme snippet (max 700 chars). Reuse logic from `legacy/enrich.py`.
- After getting `EnrichmentOutput`, write all fields to DB model, set `enrich_version`

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
    "limitations": lambda v: v and len(v) >= 1,
}

def validate_model(model: Model) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_failed_rules)."""
    ...

def validate_all(session: Session) -> dict:
    """Validate all enriched models. Returns summary stats."""
    ...
```

- If validation fails: set `validated=False`, increment retry counter
- If retry count < 3: reset `enrich_version` to None (triggers re-enrichment)
- If retry count >= 3: set `validation_failed=True`, log warning

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

@task(retries=3, retry_delay_seconds=30)
async def run_crawler(force: bool = False): ...

@task(retries=2, retry_delay_seconds=10)
def run_enricher(force: bool = False): ...

@task(retries=2)
def run_validator(): ...

@task
def run_exporter(): ...

@task
def run_pr_creator(): ...

@flow(name="ollama-pipeline")
async def ollama_pipeline(
    skip_crawl: bool = False,
    skip_enrich: bool = False,
    force_crawl: bool = False,
    force_enrich: bool = False,
    model: str | None = None,
    dry_run: bool = False,
):
    if not skip_crawl:
        await run_crawler(force=force_crawl)
    if not skip_enrich:
        run_enricher(force=force_enrich)
    run_validator()
    if not dry_run:
        run_exporter()
        run_pr_creator()
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

      - name: Install Playwright (for Crawl4AI)
        run: poetry run playwright install chromium

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
4. **Separation of concerns** — agents do NOT write to DB directly, they return data. Only flow.py coordinates DB writes.
5. **Type hints everywhere** — use Python 3.11+ syntax (`str | None`, `list[str]`).
6. **Error handling** — every agent function must have try/except with proper logging.
7. **Reuse legacy code** — `parse_pulls()`, `parse_last_updated()`, memory requirements regex, `model_to_dict()` are in `/legacy` folder. Import and reuse them.
8. **Instructor handles LLM retries** — do NOT write manual retry logic for LLM calls, instructor does it automatically with `max_retries=3`.
9. **Test each agent independently** — write at least one test per agent in `/tests`.
10. **Commit after each step** — each step should result in a working, committable state.
