# 🦙 ollama-pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-cyan.svg)](https://python-poetry.org/)
[![Prefect](https://img.shields.io/badge/orchestration-prefect-blue.svg)](https://www.prefect.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An automated, AI-powered data pipeline that crawls **ollama.com/library**, enriches each model with structured metadata using a Groq LLM, validates the results, and opens a Pull Request to [ollama-explorer](https://github.com/serkan-uslu/ollama-explorer) with the updated `models.json`.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        ollama-pipeline                            │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Crawler  │───▶│ Enricher │───▶│Validator │───▶│ Exporter │   │
│  │ (crawl4ai│    │  (Groq + │    │(pydantic │    │  (JSON)  │   │
│  │ + httpx) │    │instructor│    │ rules)   │    │          │   │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘   │
│       │                │               │               │         │
│       ▼                ▼               ▼               ▼         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    SQLite (ollama.db)                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                        │         │
│                                               ┌────────▼──────┐  │
│                                               │  PR Creator   │  │
│                                               │  (GitHub API) │  │
│                                               └───────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                         ┌──────────────────────────┐
                                         │  serkan-uslu/            │
                                         │  ollama-explorer         │
                                         │  (PR: public/models.json)│
                                         └──────────────────────────┘
```

## Pipeline Flow

```
Crawler → Enricher → Validator → Exporter → PR Creator
```

| Step | Agent | Description |
|------|-------|-------------|
| 1 | **Crawler** | Crawls `ollama.com/library` with Crawl4AI + httpx, extracts model slugs, descriptions, capabilities, hardware requirements |
| 2 | **Enricher** | Sends model data to Groq LLM (via `instructor`) and extracts structured metadata: use cases, domain, languages, complexity, etc. |
| 3 | **Validator** | Validates enriched data against quality rules. Re-queues failures (max 3 retries) |
| 4 | **Exporter** | Serializes all models to `output/models.json` sorted by pull count |
| 5 | **PR Creator** | Pushes `models.json` to `ollama-explorer` via GitHub REST API and opens a Pull Request |

---

## Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Groq API key — [get one free](https://console.groq.com)
- GitHub Personal Access Token with `repo` scope (for PR creation)

### Installation

```bash
# Clone the repo
git clone https://github.com/serkan-uslu/ollama-pipeline.git
cd ollama-pipeline

# Install dependencies
poetry install --no-root

# Install Playwright (used by Crawl4AI for JS-rendered pages)
poetry run playwright install chromium
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | **Yes** (enrich step) | — | Groq API key for LLM enrichment |
| `CROSS_REPO_TOKEN` | **Yes** (PR step) | — | GitHub PAT with `repo` scope |
| `GITHUB_TARGET_REPO` | No | `serkan-uslu/ollama-explorer` | Target repo for PR |
| `GITHUB_TARGET_BRANCH` | No | `data/models-update` | Branch name for PR |
| `OLLAMA_LIBRARY_URL` | No | `https://ollama.com/library` | Ollama library URL |
| `REQUEST_DELAY` | No | `0.5` | Seconds to wait between detail page fetches |
| `ENRICH_VERSION` | No | `1` | Bump this to force re-enrichment of all models |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use for enrichment |

---

## CLI Usage

```bash
# Run the full pipeline (crawl → enrich → validate → export → PR)
poetry run python main.py

# Skip crawling — use existing DB data
poetry run python main.py --skip-crawl

# Skip enrichment — use already-enriched data
poetry run python main.py --skip-enrich

# Force re-crawl all models (ignore already-crawled)
poetry run python main.py --force-crawl

# Force re-enrich all models (ignore enrich_version)
poetry run python main.py --force-enrich

# Process a single model only
poetry run python main.py --model deepseek-r1

# Dry run — crawl + enrich only, no export, no PR
poetry run python main.py --dry-run

# Combine flags
poetry run python main.py --skip-crawl --force-enrich --dry-run
```

---

## GitHub Actions Setup

The pipeline runs automatically every **Monday at 03:00 UTC** and can also be triggered manually.

### 1. Add Secrets

In your GitHub repo → **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | Your Groq API key |
| `CROSS_REPO_TOKEN` | GitHub PAT with `repo` scope (must have access to `ollama-explorer`) |

### 2. Manual Trigger

Go to **Actions → Update Ollama Models → Run workflow**.

Available inputs:
- **Skip crawling** — use existing DB (faster runs)
- **Force re-enrich** — regenerate all enrichment even if up to date
- **Dry run** — no export, no PR (for testing)

### 3. Artifacts

Every run uploads two artifacts (kept for 7–30 days):
- `ollama-db-{run_id}` — the SQLite database (for debugging)
- `models-json-{run_id}` — the exported `models.json`

---

## Tech Stack

| Purpose | Library |
|---------|---------|
| Web crawling | [crawl4ai](https://github.com/unclecode/crawl4ai) |
| LLM structured output | [instructor](https://github.com/jxnl/instructor) |
| Data validation | [pydantic v2](https://docs.pydantic.dev) |
| Database ORM | [sqlmodel](https://sqlmodel.tiangolo.com/) |
| Database | SQLite (built-in) |
| LLM API | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) |
| Orchestration | [Prefect](https://www.prefect.io/) |
| HTTP | [httpx](https://www.python-httpx.org/) |
| Environment vars | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Package manager | [Poetry](https://python-poetry.org/) |

---

## Project Structure

```
ollama-pipeline/
├── .github/
│   └── workflows/
│       └── update-models.yml    # GitHub Actions (weekly schedule + manual)
├── pipeline/
│   ├── agents/
│   │   ├── crawler.py           # F-01: Crawl4AI scraper
│   │   ├── enricher.py          # F-02: Groq LLM enrichment
│   │   ├── validator.py         # F-03: Data quality validation
│   │   ├── exporter.py          # F-04: JSON export
│   │   └── pr_creator.py        # F-05: GitHub PR creation
│   ├── core/
│   │   ├── db.py                # SQLite engine + session
│   │   ├── models.py            # SQLModel table definition
│   │   └── settings.py          # Pydantic Settings (env vars)
│   ├── schemas/
│   │   └── enrichment.py        # Pydantic schemas for LLM output
│   └── flow.py                  # Prefect flow orchestration
├── legacy/                      # Original scripts (reference only)
├── tests/                       # Unit tests for each agent
├── output/
│   └── models.json              # Generated output (gitignored)
├── .env.example
├── pyproject.toml
└── main.py                      # CLI entrypoint
```

---

## Data Schema

Each model in `models.json` contains:

```json
{
  "id": "uuid",
  "model_identifier": "deepseek-r1",
  "model_name": "DeepSeek-R1",
  "model_type": "official",
  "url": "https://ollama.com/library/deepseek-r1",
  "description": "...",
  "capabilities": ["Thinking"],
  "labels": ["1.5b", "7b", "14b", "32b", "70b", "671b"],
  "pulls": 12000000,
  "tags": 58,
  "last_updated": "2025-01-20",
  "min_ram_gb": 1.1,
  "context_window": 131072,
  "speed_tier": "slow",
  "domain": "Reasoning",
  "use_cases": ["Reasoning", "Math", "Code Generation"],
  "complexity": "advanced",
  "best_for": "Complex reasoning tasks requiring step-by-step thinking",
  "ai_languages": ["English", "Multilingual"],
  "model_family": "DeepSeek",
  "is_fine_tuned": false,
  "is_uncensored": false,
  "strengths": ["Strong reasoning", "Chain-of-thought", "Multilingual"],
  "limitations": ["Very large", "Slow for simple queries"],
  "target_audience": ["Researchers", "Developers"],
  "license": "MIT",
  "enrich_version": 1,
  "validated": true
}
```

---

## Related

- **[ollama-explorer](https://github.com/serkan-uslu/ollama-explorer)** — The frontend that consumes `models.json` to provide a searchable, filterable UI for all Ollama models.

---

## Running Tests

```bash
poetry run pytest tests/ -v
```
