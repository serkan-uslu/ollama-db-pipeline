# ollama-pipeline — Feature Specification

## Overview

An automated, AI-powered data pipeline that:
1. Crawls `ollama.com/library` for all available models
2. Enriches each model with structured metadata using an LLM (Groq API)
3. Validates the enriched data and retries if quality is insufficient
4. Exports a `models.json` file
5. Opens a Pull Request to the `serkan-uslu/ollama-explorer` repository via GitHub Actions

---

## Core Features

### F-01 · Crawler Agent
- Crawl `https://ollama.com/library` using **Crawl4AI**
- Extract all model slugs, names, descriptions, pull counts, tag counts, capabilities, labels, last updated
- Visit each model's detail page and extract: README content, memory requirements table (tag, size GB, context window, quantization)
- Store raw data in **SQLite** via **SQLModel**
- Respect rate limiting: 0.5s delay between requests
- Skip already-crawled models unless `--force` flag is passed

### F-02 · Enricher Agent
- Read unenriched models from DB (where `enrich_version IS NULL` or outdated)
- Build a structured prompt with model data
- Call **Groq API** (model: `llama-3.3-70b-versatile` or `qwen-qwq-32b`) with JSON mode
- Use **Instructor** + **Pydantic** schema to guarantee valid structured output
- Extract fields: `use_cases`, `domain`, `ai_languages`, `complexity`, `best_for`, `license`, `base_model`, `is_fine_tuned`, `strengths`, `limitations`, `target_audience`
- Save enriched data back to DB with `enrich_version` stamp

### F-03 · Validator Agent
- After enrichment, validate each model's data quality
- Check: required fields are not null, values are within allowed enums, lists are not empty
- If validation fails: re-send to Enricher Agent (max 3 retries)
- Mark models as `validated=True` or `validation_failed=True` in DB
- Log all failures with reason

### F-04 · Exporter
- Read all validated models from DB
- Serialize to `models.json` with consistent field ordering
- Output statistics: total models, enriched count, missing fields summary
- Write file to `/output/models.json`

### F-05 · GitHub PR Creator
- Read generated `models.json`
- Clone `serkan-uslu/ollama-explorer` repo
- Copy `models.json` to `public/data/models.json` in that repo
- Commit with message: `chore: update models data — {date} ({N} models)`
- Open a Pull Request with:
  - Title: `chore: update ollama models — {date}`
  - Body: summary of changes (new models, updated models, stats)
- Use `CROSS_REPO_TOKEN` secret for authentication

### F-06 · Orchestration (Prefect)
- Define a Prefect Flow that runs all agents in sequence:
  `Crawler → Enricher → Validator → Exporter → PR Creator`
- Each agent is a Prefect Task with retry logic
- Flow can be triggered:
  - Manually via CLI: `python main.py`
  - On schedule via GitHub Actions (weekly, every Monday 03:00 UTC)
  - Partially: `python main.py --skip-crawl` to only re-enrich

### F-07 · GitHub Actions Workflow
- File: `.github/workflows/update-models.yml`
- Triggers: `schedule` (weekly) + `workflow_dispatch` (manual)
- Steps: checkout → install deps → run pipeline → open PR
- Secrets required: `GROQ_API_KEY`, `CROSS_REPO_TOKEN`

---

## Data Schema

### Model (SQLite table)

```python
class Model(SQLModel, table=True):
    # Identity
    id: uuid.UUID
    model_identifier: str          # e.g. "deepseek-r1"
    model_name: str
    model_type: str                # "official"
    namespace: str | None
    url: str

    # Raw scraped
    description: str | None
    readme: str | None
    capability: str | None         # legacy single string
    capabilities: list[str]        # ["Tools", "Vision", "Thinking", "Cloud"]
    labels: list[str]              # ["8b", "70b", "405b"]
    pulls: int
    tags: int
    last_updated: date | None
    last_updated_str: str | None

    # Hardware
    memory_requirements: list[dict]  # [{tag, size, size_gb, recommended_ram_gb, quantization, context, context_window}]
    min_ram_gb: float | None
    context_window: int | None
    speed_tier: str | None         # "fast" | "medium" | "slow"

    # AI Enriched
    use_cases: list[str] | None
    domain: str | None
    ai_languages: list[str] | None
    complexity: str | None         # "beginner" | "intermediate" | "advanced"
    best_for: str | None
    license: str | None
    base_model: str | None
    is_fine_tuned: bool | None
    strengths: list[str] | None
    limitations: list[str] | None
    target_audience: list[str] | None

    # Pipeline metadata
    enrich_version: int | None
    validated: bool | None
    validation_failed: bool | None
    timestamp: datetime
```

---

## Allowed Enum Values

```python
ALLOWED_USE_CASES = [
    "Chat Assistant", "Code Generation", "Code Review",
    "Text Summarization", "Question Answering", "RAG / Retrieval",
    "Text Embedding", "Image Understanding", "Reasoning", "Translation",
    "Math", "Creative Writing", "Function Calling", "Role Play",
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

ALLOWED_AUDIENCE = [
    "Developers", "Beginners", "Researchers",
    "Data Scientists", "DevOps", "Students",
]
```

---

## Quality Rules (Validator)

| Field | Rule |
|---|---|
| `domain` | Must not be null |
| `use_cases` | At least 1 item |
| `ai_languages` | At least 1 item |
| `complexity` | Must not be null |
| `best_for` | Must not be null, min 10 chars |
| `strengths` | At least 1 item |
| `limitations` | At least 1 item |

If any rule fails → re-enrich (max 3 attempts) → mark `validation_failed=True`

---

## CLI Interface

```bash
# Run full pipeline
python main.py

# Skip crawling (use existing DB data)
python main.py --skip-crawl

# Skip enrichment (use existing enriched data)
python main.py --skip-enrich

# Force re-crawl all models
python main.py --force-crawl

# Force re-enrich all models
python main.py --force-enrich

# Process single model
python main.py --model deepseek-r1

# Dry run (no DB writes, no PR)
python main.py --dry-run
```
