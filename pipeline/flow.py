"""
pipeline/flow.py

F-06: Prefect Flow Orchestration

Runs all agents in sequence:
    Crawler → Enricher → Validator → Exporter → PR Creator

Each agent is wrapped as a Prefect @task with retry logic.
The flow supports partial runs via skip/force flags.
"""

import asyncio
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
def run_crawler_task(force: bool = False):
    """Crawl ollama.com/library + detail pages, upsert to DB."""
    print(f"\n{'═'*60}", flush=True)
    print(f"[FLOW] ▶ ADIM 1 / 5 — CRAWLER", flush=True)
    print(f"{'═'*60}", flush=True)
    # run_full_crawl is async — run it in its own event loop inside this sync task
    models = asyncio.run(run_full_crawl(force=force))
    print(f"[FLOW] ✅ Crawler bitti — {len(models)} model işlendi", flush=True)
    return len(models)


@task(name="enricher", retries=2, retry_delay_seconds=10)
def run_enricher_task(force: bool = False, single_slug: str | None = None):
    """Enrich unenriched/outdated models via Groq LLM."""
    print(f"\n{'═'*60}", flush=True)
    print(f"[FLOW] ▶ ADIM 2 / 5 — ENRICHER", flush=True)
    print(f"{'═'*60}", flush=True)
    stats = run_enricher(force=force, single_slug=single_slug)
    print(f"[FLOW] ✅ Enricher bitti — ok={stats.get('ok', 0)} | failed={stats.get('failed', 0)}", flush=True)
    return stats


@task(name="validator", retries=2)
def run_validator_task():
    """Validate enriched models, re-queue failures."""
    print(f"\n{'═'*60}", flush=True)
    print(f"[FLOW] ▶ ADIM 3 / 5 — VALIDATOR", flush=True)
    print(f"{'═'*60}", flush=True)
    stats = run_validator()
    print(f"[FLOW] ✅ Validator bitti — valid={stats.get('valid', 0)} | re_queued={stats.get('re_queued', 0)}", flush=True)
    return stats


@task(name="exporter")
def run_exporter_task() -> dict:
    """Export all enriched models to output/models.json."""
    print(f"\n{'═'*60}", flush=True)
    print(f"[FLOW] ▶ ADIM 4 / 5 — EXPORTER", flush=True)
    print(f"{'═'*60}", flush=True)
    stats = export_to_json()
    print(
        f"[FLOW] ✅ Export bitti — {stats.get('total', 0)} model → {stats.get('output_path', '')}",
        flush=True,
    )
    return stats


@task(name="pr-creator")
def run_pr_creator_task(export_stats: dict | None = None) -> str | None:
    """Push models.json to GitHub and open a Pull Request."""
    print(f"\n{'═'*60}", flush=True)
    print(f"[FLOW] ▶ ADIM 5 / 5 — PR CREATOR", flush=True)
    print(f"{'═'*60}", flush=True)
    pr_url = create_pull_request(export_stats=export_stats)
    if pr_url:
        print(f"[FLOW] ✅ PR oluşturuldu — {pr_url}", flush=True)
    else:
        print(f"[FLOW] ⏭  PR atlandı (değişiklik yok veya dry-run)", flush=True)
    return pr_url


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="ollama-pipeline", log_prints=True)
def ollama_pipeline(
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

    print(f"\n{'█'*60}", flush=True)
    print(f"[PIPELINE] 🚀 OLLAMA PIPELINE BAŞLIYOR", flush=True)
    print(f"[PIPELINE]    skip_crawl={skip_crawl} | skip_enrich={skip_enrich}", flush=True)
    print(f"[PIPELINE]    force_crawl={force_crawl} | force_enrich={force_enrich}", flush=True)
    print(f"[PIPELINE]    model={model!r} | dry_run={dry_run}", flush=True)
    print(f"{'█'*60}", flush=True)

    # Step 1 — Crawl
    if not skip_crawl:
        run_crawler_task(force=force_crawl)
    else:
        print("[FLOW] ⏭  ADIM 1 atlandı (--skip-crawl)", flush=True)

    # Step 2 — Enrich
    if not skip_enrich:
        enrich_stats = run_enricher_task(force=force_enrich, single_slug=model)
    else:
        print("[FLOW] ⏭  ADIM 2 atlandı (--skip-enrich)", flush=True)
        enrich_stats = {}

    # Step 3 — Validate (always runs unless dry-run)
    if not dry_run:
        val_stats = run_validator_task()
    else:
        print("[FLOW] ⏭  Dry-run — validator, exporter, PR creator atlandı", flush=True)
        return

    # Step 4 — Export
    export_stats = run_exporter_task()

    # Step 5 — PR
    pr_url = run_pr_creator_task(export_stats=export_stats)

    # Summary
    print(f"\n{'█'*60}", flush=True)
    print(f"[PIPELINE] 🏁 PIPELINE TAMAMLANDI", flush=True)
    print(f"[PIPELINE]    Export  : {export_stats.get('total', '?')} model → {export_stats.get('output_path', '?')}", flush=True)
    print(f"[PIPELINE]    Validate: valid={val_stats.get('valid', '?')} | failed={val_stats.get('failed', '?')}", flush=True)
    print(f"[PIPELINE]    PR      : {pr_url or '(PR oluşturulmadı)'}", flush=True)
    print(f"{'█'*60}", flush=True)

    return {
        "export": export_stats,
        "validation": val_stats,
        "pr_url": pr_url,
    }
