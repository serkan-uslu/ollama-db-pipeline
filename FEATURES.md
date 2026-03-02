# ollama-pipeline — Feature Specification

## Overview

An automated, AI-powered data pipeline that:
1. Crawls `ollama.com/library` for all available models
2. Enriches each model with structured metadata using an LLM (Ollama local or Groq cloud)
3. Validates the enriched data and retries if quality is insufficient
4. Exports a `models_<llm_model>.json` file
5. Opens a Pull Request to the `serkan-uslu/ollama-explorer` repository via GitHub Actions

---

## Core Features

### F-01 · Crawler Agent
- Crawl `https://ollama.com/library` using **httpx + BeautifulSoup4**
- Extract all model slugs, names, descriptions, pull counts, tag counts, capabilities, labels, last updated
- Visit each **new** model's detail page and extract: README content, raw HTML, memory requirements table (tag, size GB, context window, quantization), applications
- For **existing** models: refresh stats-only fields (pulls, last_updated, description, capabilities) without re-fetching detail pages
- Use `_has_detail` flag to guard detail field overwrites — never wipes existing README/html with empty data
- Store raw data in **SQLite** via **SQLModel**
- Respect rate limiting: 0.5s delay between requests
- Force full re-crawl (including detail pages) for all models when `--force-crawl` flag is passed

### F-02 · Enricher Agent
- Read unenriched models from DB (where `enrich_version IS NULL` or outdated)
- Support two LLM providers via `LLM_PROVIDER` setting:
  - **Ollama** (default, local): OpenAI-compatible API at `OLLAMA_BASE_URL`; no key required
  - **Groq** (cloud): `GROQ_API_KEY` required; model: `llama-3.3-70b-versatile`
- Use 6 focused LLM calls per model (not a single prompt) for higher accuracy:
  1. `_p1_domain_family` — domain + model family classification
  2. `_p2_use_cases` — use cases + target audience
  3. `_p3_basics` — complexity, best_for, is_fine_tuned, is_uncensored, is_multimodal
  4. `_p4_summary` — strengths + limitations
  5. `_p5_quality` — benchmark scores + parameter sizes
  6. `_p6_metadata` — license, base_model, creator_org, huggingface_url, ai_languages
- Use **Instructor** + **Pydantic** schemas to guarantee valid structured output from each call
- Parallel enrichment via `ThreadPoolExecutor` with `ENRICH_WORKERS` workers
- Save enriched data back to DB with `enrich_version` stamp
- Default model: `gemma3:27b` (Ollama)

### F-03 · Validator Agent
- After enrichment, validate each model's data quality
- Check: required fields are not null, values are within allowed enums, lists have minimum items
- If validation fails: reset `enrich_version` to null to trigger re-enrichment; increment `validation_retries` in DB (persistent across runs)
- If `validation_retries >= 3`: mark `validation_failed=True`, stop retrying
- Mark models as `validated=True` or `validation_failed=True` in DB
- Log all failures with reason
- Returns `re_queued` count — if > 0, flow immediately re-runs enricher + validator in the same pipeline run (Step 3b)

### F-04 · Exporter
- Read all validated models from DB
- Serialize to `models_<llm_model>.json` with consistent field ordering (filename derived from `settings.llm_model`)
- Output statistics: total models, enriched count, validated count, missing fields summary
- Write file to `output/models_<llm_model>.json`

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
  `Crawler → Enricher → Validator [→ Re-Enricher → Re-Validator] → Exporter → PR Creator`
- Each agent is a Prefect Task with retry logic
- **Step 3b**: If validator re-queues any models (`re_queued > 0`), the enricher and validator are immediately re-run within the same pipeline execution before proceeding to export
- Flow can be triggered:
  - Manually via CLI: `python main.py`
  - On schedule via GitHub Actions (weekly, every Monday 03:00 UTC)
  - Partially: `python main.py --skip-crawl` to only re-enrich

### F-07 · GitHub Actions Workflow
- File: `.github/workflows/update-models.yml`
- Triggers: `schedule` (weekly) + `workflow_dispatch` (manual)
- Steps: checkout → install deps → run pipeline → open PR
- Secrets required: `CROSS_REPO_TOKEN` (always); `GROQ_API_KEY` (only if `LLM_PROVIDER=groq`)

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
    raw_html: str | None           # full detail page HTML for re-parsing
    capability: str | None         # legacy single string
    capabilities: list[str] | None # ["Tools", "Vision", "Thinking", "Cloud"]
    labels: list[str]              # ["8b", "70b", "405b"]
    applications: list[dict] | None  # [{name, launch_command}]

    # Stats
    pulls: int
    tags: int
    last_updated: date | None
    last_updated_str: str | None

    # Hardware
    memory_requirements: list[dict] | None  # [{tag, size, size_gb, recommended_ram_gb, quantization, context, context_window}]
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
    model_family: str | None       # "Llama" | "Mistral" | "Qwen" | ...
    is_fine_tuned: bool | None
    is_uncensored: bool | None
    is_multimodal: bool | None
    creator_org: str | None        # e.g. "Meta", "Mistral AI", "Google DeepMind"
    huggingface_url: str | None
    benchmark_scores: list[dict] | None  # [{name, score, unit}]
    parameter_sizes: list[str] | None    # ["1.5B", "7B", "13B", "70B"]
    strengths: list[str] | None
    limitations: list[str] | None
    target_audience: list[str] | None

    # Pipeline metadata
    enrich_version: int | None
    validated: bool | None
    validation_failed: bool | None
    validation_retries: int        # persistent retry counter (survives between runs)
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
| `limitations` | Must not be null (empty list `[]` is allowed) |

If any rule fails → re-enrich (max 3 attempts, tracked in `validation_retries` DB field) → mark `validation_failed=True`

---

## CLI Interface

```bash
# Run full pipeline (also initialises the DB on first run)
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
