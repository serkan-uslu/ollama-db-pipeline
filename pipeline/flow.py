"""
pipeline/flow.py

F-06: Prefect Flow Orchestration

Runs all agents in sequence:
    Crawler → Enricher → Validator → Exporter → PR Creator

Each agent is wrapped as a Prefect @task with retry logic.
The flow supports partial runs via skip/force flags.
"""

import logging

from prefect import flow, task

from pipeline.agents.crawler import run_full_crawl
from pipeline.agents.enricher import run_enricher
from pipeline.agents.exporter import export_to_json
from pipeline.agents.pr_creator import create_pull_request
from pipeline.agents.validator import run_validator
from pipeline.core.db import init_db

logger = logging.getLogger(__name__)


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="crawler", retries=3, retry_delay_seconds=30)
async def run_crawler_task(force: bool = False):
    """Crawl ollama.com/library + detail pages, upsert to DB."""
    logger.info("=== Crawler starting ===")
    models = await run_full_crawl(force=force)
    logger.info(f"=== Crawler done — {len(models)} models processed ===")
    return len(models)


@task(name="enricher", retries=2, retry_delay_seconds=10)
def run_enricher_task(force: bool = False, single_slug: str | None = None):
    """Enrich unenriched/outdated models via Groq LLM."""
    logger.info("=== Enricher starting ===")
    stats = run_enricher(force=force, single_slug=single_slug)
    logger.info(f"=== Enricher done — {stats} ===")
    return stats


@task(name="validator", retries=2)
def run_validator_task():
    """Validate enriched models, re-queue failures."""
    logger.info("=== Validator starting ===")
    stats = run_validator()
    logger.info(f"=== Validator done — {stats} ===")
    return stats


@task(name="exporter")
def run_exporter_task() -> dict:
    """Export all enriched models to output/models.json."""
    logger.info("=== Exporter starting ===")
    stats = export_to_json()
    logger.info(f"=== Exporter done — {stats} ===")
    return stats


@task(name="pr-creator")
def run_pr_creator_task(export_stats: dict | None = None) -> str | None:
    """Push models.json to GitHub and open a Pull Request."""
    logger.info("=== PR Creator starting ===")
    pr_url = create_pull_request(export_stats=export_stats)
    if pr_url:
        logger.info(f"=== PR Creator done — {pr_url} ===")
    else:
        logger.info("=== PR Creator done — no changes / skipped ===")
    return pr_url


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="ollama-pipeline", log_prints=True)
async def ollama_pipeline(
    skip_crawl: bool = False,
    skip_enrich: bool = False,
    force_crawl: bool = False,
    force_enrich: bool = False,
    model: str | None = None,    # single model slug for targeted runs
    dry_run: bool = False,        # skip export + PR (DB writes only)
):
    """
    Full Ollama pipeline:
        Crawler → Enricher → Validator → Exporter → PR Creator

    Flags:
        --skip-crawl    : use existing DB data, skip crawling
        --skip-enrich   : skip LLM enrichment step
        --force-crawl   : re-crawl all models (ignore already-crawled)
        --force-enrich  : re-enrich all models (ignore enrich_version)
        --model         : process a single model slug only
        --dry-run       : stop before export/PR (useful for testing)
    """
    init_db()

    # Step 1 — Crawl
    if not skip_crawl:
        await run_crawler_task(force=force_crawl)
    else:
        logger.info("Skipping crawler (--skip-crawl).")

    # Step 2 — Enrich
    if not skip_enrich:
        enrich_stats = run_enricher_task(force=force_enrich, single_slug=model)
    else:
        logger.info("Skipping enricher (--skip-enrich).")
        enrich_stats = {}

    # Step 3 — Validate (always runs unless dry-run)
    if not dry_run:
        val_stats = run_validator_task()
    else:
        logger.info("Dry run — skipping validator, exporter, PR creator.")
        return

    # Step 4 — Export
    export_stats = run_exporter_task()

    # Step 5 — PR
    pr_url = run_pr_creator_task(export_stats=export_stats)

    # Summary
    logger.info("=" * 50)
    logger.info("Pipeline complete!")
    logger.info(f"  Export: {export_stats}")
    logger.info(f"  Validation: {val_stats}")
    logger.info(f"  PR: {pr_url or 'No PR (no changes or dry run)'}")
    logger.info("=" * 50)

    return {
        "export": export_stats,
        "validation": val_stats,
        "pr_url": pr_url,
    }
