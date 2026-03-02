# ⛏️ Ollama Miner

<table>
<tr>
<td width="210">
  <img src="assets/olli-miner.jpg" alt="Olli the Data Miner — ollama-pipeline mascot" width="195" style="border-radius: 14px;" />
</td>
<td>

### The data engine behind [Ollama Explorer](https://ollama-explorer.vercel.app)

> **Ollama has 200+ open-source AI models. Finding the right one is painful.**
>
> The official library shows a wall of model cards with almost no filtering. You open dozens of tabs, read raw descriptions, and still can't tell if your machine can even run it.
>
> **Ollama Miner fixes the data layer.** Every week it crawls `ollama.com/library`, pushes each model through 6 focused LLM enrichment calls, validates every field, and opens a Pull Request to [Ollama Explorer](https://github.com/serkan-uslu/ollama-explorer) — giving the UI fresh, structured metadata to search and filter by.

</td>
</tr>
</table>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-cyan.svg)](https://python-poetry.org/)
[![Prefect](https://img.shields.io/badge/orchestration-prefect-blue.svg)](https://www.prefect.io/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🌐 Part of the Ollama Ecosystem

This repo is **Part 1 of 2** — the automated data backend. The two repos form a complete product:

| | Repo | Role |
|:-:|------|------|
| ⛏️ **You are here** | [`Ollama Miner`](https://github.com/serkan-uslu/ollama-db-pipeline) | Python pipeline — crawls `ollama.com/library`, enriches with LLM, validates, exports `models.json` |
| 🌐 **Frontend** | [`Ollama Explorer`](https://github.com/serkan-uslu/ollama-explorer) | Next.js 16 app — fast, searchable, filterable directory of every Ollama model |

```
  ollama.com/library
        ▼  (crawl every Monday 03:00 UTC)
┌─────────────────────┐      PR: models.json      ┌──────────────────────────┐
│     Ollama Miner       │ ────────────────────────▶ │    Ollama Explorer      │
│  crawl → enrich     │   public/data/models.json  │  ollama-explorer.vercel  │
│  validate → export  │                            │  .app  (🌐 live)         │
└─────────────────────┘                            └──────────────────────────┘
```

🌐 **Live app** → [ollama-explorer.vercel.app](https://ollama-explorer.vercel.app)  
📦 **Frontend source** → [github.com/serkan-uslu/ollama-explorer](https://github.com/serkan-uslu/ollama-explorer)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       Ollama Miner                               │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Crawler  │───▶│ Enricher │───▶│Validator │───▶│ Exporter │   │
│  │ (httpx + │    │(Ollama / │    │(pydantic │    │  (JSON)  │   │
│  │Beautiful │    │Groq +inst│    │ rules +  │    │          │   │
│  │ Soup)    │    │ ructor)  │    │re-enrich)│    │          │   │
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
Crawler → Enricher → Validator [→ Re-Enricher → Re-Validator] → Exporter → PR Creator
```

| Step | Agent | Description |
|------|-------|-------------|
| 1 | **Crawler** | Crawls `ollama.com/library` with httpx + BeautifulSoup. Refreshes stats (pulls, last\_updated) for **all** models; fetches detail pages (readme, memory requirements) **only** for new ones |
| 2 | **Enricher** | Sends model data to LLM (Ollama local or Groq cloud, via `instructor`) — 6 focused calls per model — and writes structured metadata to DB |
| 3 | **Validator** | Validates enriched data against quality rules. Re-queues failures (max 3 retries, persisted across runs). If any re-queued, **immediately re-runs Enricher + Validator** in the same pipeline run |
| 4 | **Exporter** | Serializes all enriched models to `output/models_<llm_model>.json` sorted by pull count |
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
git clone https://github.com/serkan-uslu/ollama-db-pipeline.git
cd ollama-db-pipeline

# Install dependencies
poetry install --no-root
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Only if `LLM_PROVIDER=groq` | — | Groq API key for LLM enrichment |
| `CROSS_REPO_TOKEN` | **Yes** (PR step) | — | GitHub PAT with `repo` scope |
| `GITHUB_TARGET_REPO` | No | `serkan-uslu/ollama-explorer` | Target repo for PR |
| `GITHUB_TARGET_BRANCH` | No | `data/models-update` | Branch name for PR |
| `LLM_PROVIDER` | No | `ollama` | `ollama` (local) or `groq` (cloud) |
| `LLM_MODEL` | No | `llama3.3:70b` | Model name for the chosen provider |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama OpenAI-compatible API URL |
| `ENRICH_WORKERS` | No | `3` | Parallel enrichment workers (2–4 for Ollama) |
| `ENRICH_VERSION` | No | `1` | Bump this to force re-enrichment of all models |
| `REQUEST_DELAY` | No | `0.5` | Seconds to wait between detail page fetches |

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

# Re-parse stored raw_html for all models (no network) — updates readme, memory_requirements
poetry run python main.py --reparse

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
| Web crawling | [httpx](https://www.python-httpx.org/) + [BeautifulSoup4](https://beautiful-soup-4.readthedocs.io/) |
| LLM structured output | [instructor](https://github.com/jxnl/instructor) |
| Data validation | [pydantic v2](https://docs.pydantic.dev) |
| Database ORM | [sqlmodel](https://sqlmodel.tiangolo.com/) |
| Database | SQLite (built-in) |
| LLM API (local) | [Ollama](https://ollama.com/) (OpenAI-compatible, default) |
| LLM API (cloud) | [Groq](https://groq.com/) (optional) |
| Orchestration | [Prefect](https://www.prefect.io/) |
| HTTP | [httpx](https://www.python-httpx.org/) |
| Environment vars | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
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
│   │   ├── crawler.py           # F-01: httpx + BeautifulSoup scraper
│   │   ├── enricher.py          # F-02: Ollama/Groq LLM enrichment (6 calls/model)
│   │   ├── validator.py         # F-03: Data quality validation + persistent retries
│   │   ├── exporter.py          # F-04: JSON export
│   │   └── pr_creator.py        # F-05: GitHub PR creation
│   ├── core/
│   │   ├── db.py                # SQLite engine + session
│   │   ├── models.py            # SQLModel table definition
│   │   └── settings.py          # Pydantic Settings (env vars)
│   ├── schemas/
│   │   └── enrichment.py        # Pydantic schemas for LLM output
│   └── flow.py                  # Prefect flow orchestration
├── tests/                       # Unit tests for each agent
├── output/
│   └── models_*.json            # Generated output (gitignored)
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

## 🌐 Related

<table>
<tr>
<td width="110" align="center">
  <img src="https://raw.githubusercontent.com/serkan-uslu/ollama-explorer/main/public/olli.jpg" alt="Olli" width="100" style="border-radius: 10px;" />
</td>
<td>

**[Ollama Miner](https://github.com/serkan-uslu/ollama-db-pipeline)** — The data pipeline that produces `models.json` consumed by this app.  
Crawls `ollama.com/library`, enriches 214+ models via LLM, validates every field, and opens a PR automatically every week.  
⛏️ **[github.com/serkan-uslu/ollama-db-pipeline](https://github.com/serkan-uslu/ollama-db-pipeline)**

</td>
</tr>
</table>

---

## Running Tests

```bash
poetry run pytest tests/ -v
```
