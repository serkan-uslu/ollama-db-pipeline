"""
main.py — CLI Entrypoint for ollama-pipeline

Usage:
    poetry run python main.py                    # Full pipeline
    poetry run python main.py --skip-crawl       # Skip crawling
    poetry run python main.py --skip-enrich      # Skip enrichment
    poetry run python main.py --force-crawl      # Re-crawl all models
    poetry run python main.py --force-enrich     # Re-enrich all models
    poetry run python main.py --model deepseek-r1  # Single model
    poetry run python main.py --dry-run          # No export / PR
"""

import argparse
import asyncio
import logging
import sys

from pipeline.core.db import init_db
from pipeline.flow import ollama_pipeline

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ollama-pipeline")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ollama-pipeline",
        description="AI-powered Ollama model metadata pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Full pipeline (crawl → enrich → validate → export → PR)
  python main.py --skip-crawl             Use existing DB data, skip crawling
  python main.py --skip-enrich           Skip enrichment (use already-enriched data)
  python main.py --force-crawl           Re-crawl all models (ignore already-crawled)
  python main.py --force-enrich          Re-enrich all models (ignore enrich_version)
  python main.py --model deepseek-r1     Process a single model only
  python main.py --dry-run               Stop before export + PR (DB writes only)
        """,
    )

    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip the crawling step and use existing DB data.",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip the enrichment step.",
    )
    parser.add_argument(
        "--force-crawl",
        action="store_true",
        help="Force re-crawl all models, even if already in DB.",
    )
    parser.add_argument(
        "--force-enrich",
        action="store_true",
        help="Force re-enrich all models, regardless of enrich_version.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="SLUG",
        help="Process a single model by its identifier (e.g. deepseek-r1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawl + enrich only. Skip export and PR creation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 55)
    logger.info("🦙  Ollama Pipeline  Starting")
    logger.info(f"   skip_crawl   = {args.skip_crawl}")
    logger.info(f"   skip_enrich  = {args.skip_enrich}")
    logger.info(f"   force_crawl  = {args.force_crawl}")
    logger.info(f"   force_enrich = {args.force_enrich}")
    logger.info(f"   model        = {args.model or 'all'}")
    logger.info(f"   dry_run      = {args.dry_run}")
    logger.info("=" * 55)

    init_db()

    result = asyncio.run(
        ollama_pipeline(
            skip_crawl=args.skip_crawl,
            skip_enrich=args.skip_enrich,
            force_crawl=args.force_crawl,
            force_enrich=args.force_enrich,
            model=args.model,
            dry_run=args.dry_run,
        )
    )

    if result:
        logger.info("Pipeline finished successfully.")
    else:
        logger.info("Pipeline finished (dry run or no result).")


if __name__ == "__main__":
    main()
