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
import logging
import os
import sys

# ── Prefect: tüm logları API'ye değil direkt terminale yaz ────────────────────
os.environ["PREFECT_LOGGING_TO_API_ENABLED"] = "False"
os.environ["PREFECT_LOGGING_LEVEL"] = "WARNING"   # Prefect kendi iç loglarını sustur
os.environ["PREFECT_API_URL"] = ""                 # ephemeral server kullan

from pipeline.core.db import init_db
from pipeline.flow import ollama_pipeline
from pipeline.agents.crawler import reparse_from_html

# ── Logging: pipeline logları daima stdout'a, flush ile anlık görünür ─────────
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
))
handler.stream = open(sys.stdout.fileno(), mode='w', buffering=1, closefd=False)

root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers.clear()
root.addHandler(handler)

# pipeline.* loggerlarını da yakala
for name in ("pipeline", "pipeline.agents.crawler", "pipeline.agents.enricher",
             "pipeline.agents.validator", "pipeline.agents.exporter",
             "pipeline.agents.pr_creator"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = True

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
  python main.py --reparse               Re-parse stored raw_html (no network) and update crawl fields
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
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Re-parse stored raw_html for all models (no network). Updates readme, applications, memory_requirements.",
    )

    return parser.parse_args()


def main():
    import datetime, traceback

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

    # ── --reparse: no pipeline, just re-extract fields from stored HTML ───────
    if args.reparse:
        print("\n[REPARSE] Ham HTML'den yeniden parse ediliyor...", flush=True)
        updated, skipped = reparse_from_html()
        print(f"[REPARSE] Tamamlandi: {updated} guncellendi, {skipped} html yok", flush=True)
        return

    start = datetime.datetime.now()
    try:
        result = ollama_pipeline(
            skip_crawl=args.skip_crawl,
            skip_enrich=args.skip_enrich,
            force_crawl=args.force_crawl,
            force_enrich=args.force_enrich,
            model=args.model,
            dry_run=args.dry_run,
        )
        elapsed = datetime.datetime.now() - start
        print(f"\n{'█'*55}", flush=True)
        print(f"✅  PIPELINE BAŞARIYLA TAMAMLANDI  ({elapsed})", flush=True)
        print(f"{'█'*55}\n", flush=True)
        logger.info(f"Pipeline finished OK in {elapsed}.")

    except KeyboardInterrupt:
        elapsed = datetime.datetime.now() - start
        print(f"\n{'█'*55}", flush=True)
        print(f"⏹️   PIPELINE DURDURULDU (Ctrl+C)  ({elapsed})", flush=True)
        print(f"{'█'*55}\n", flush=True)
        sys.exit(0)

    except Exception as e:
        elapsed = datetime.datetime.now() - start
        print(f"\n{'█'*55}", flush=True)
        print(f"❌  PIPELINE ÇÖKTÜ  ({elapsed})", flush=True)
        print(f"    HATA: {type(e).__name__}: {e}", flush=True)
        print(f"{'█'*55}\n", flush=True)
        logger.error(f"Pipeline CRASHED after {elapsed}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
