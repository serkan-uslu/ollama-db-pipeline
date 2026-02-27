"""
pipeline/agents/exporter.py

F-04: Exporter Agent

Reads all validated (or enriched) models from DB and serializes them
to output/models.json for consumption by ollama-explorer frontend.

model_to_dict() ported from legacy/export_json.py — extended with new fields
(model_family, is_uncensored, context_window, speed_tier, strengths, limitations,
target_audience, base_model, is_fine_tuned, license).
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

from sqlmodel import Session, select

from pipeline.core.db import engine, init_db
from pipeline.core.models import Model

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "models.json"


# ── Serializer (ported + extended from legacy/export_json.py) ─────────────────

def model_to_dict(m: Model) -> dict:
    """Convert a Model SQLModel instance to a JSON-serializable dict."""

    def _date(v) -> str | None:
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        return v

    return {
        # ── Identity ────────────────────────────────────────────────────────
        "id": str(m.id),
        "model_identifier": m.model_identifier,
        "model_name": m.model_name,
        "model_type": m.model_type,
        "namespace": m.namespace,
        "url": m.url,

        # ── Description & Content ────────────────────────────────────────────
        "description": m.description,
        "readme": m.readme,

        # ── Capabilities & Labels ────────────────────────────────────────────
        "capabilities": m.capabilities or [],       # ["Tools", "Vision", ...]
        "capability": m.capability,                  # legacy single string
        "labels": m.labels or [],                    # ["8b", "70b", "405b"]

        # ── Hardware ─────────────────────────────────────────────────────────
        "applications": m.applications or [],
        "memory_requirements": m.memory_requirements or [],
        "min_ram_gb": m.min_ram_gb,
        "context_window": m.context_window,
        "speed_tier": m.speed_tier,                  # "fast" | "medium" | "slow"

        # ── AI Enrichment ─────────────────────────────────────────────────────
        "use_cases": m.use_cases or [],
        "domain": m.domain,
        "ai_languages": m.ai_languages or [],
        "complexity": m.complexity,
        "best_for": m.best_for,
        "model_family": m.model_family,              # "Llama" | "Mistral" | ...
        "base_model": m.base_model,
        "is_fine_tuned": m.is_fine_tuned,
        "is_uncensored": m.is_uncensored,
        "license": m.license,
        "strengths": m.strengths or [],
        "limitations": m.limitations or [],
        "target_audience": m.target_audience or [],
        "creator_org": m.creator_org,
        "is_multimodal": m.is_multimodal,
        "huggingface_url": m.huggingface_url,
        "benchmark_scores": m.benchmark_scores or [],
        "parameter_sizes": m.parameter_sizes or [],

        # ── Stats ─────────────────────────────────────────────────────────────
        "pulls": m.pulls,
        "tags": m.tags,

        # ── Dates ─────────────────────────────────────────────────────────────
        "last_updated": _date(m.last_updated),
        "last_updated_str": m.last_updated_str,
        "timestamp": _date(m.timestamp),

        # ── Pipeline Metadata ─────────────────────────────────────────────────
        "enrich_version": m.enrich_version,
        "validated": m.validated,
        "validation_failed": m.validation_failed,
    }


# ── Export Function ────────────────────────────────────────────────────────────

def export_to_json(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    """
    Export all enriched models to JSON.
    Includes validated=True models AND enriched-but-not-yet-validated models.
    Excludes models with no enrichment at all.
    Sorts by pulls descending.

    Returns export stats dict.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    init_db()
    with Session(engine) as session:
        # Include all enriched models (validated OR pending validation OR failed)
        # Exclude completely unenriched models (enrich_version IS NULL)
        models = list(
            session.exec(
                select(Model).where(Model.enrich_version.is_not(None))  # type: ignore[union-attr]
            ).all()
        )

    # Sort by pulls descending (most popular first)
    models_sorted = sorted(models, key=lambda m: m.pulls, reverse=True)
    data = [model_to_dict(m) for m in models_sorted]

    # Write JSON
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Compute stats
    total = len(data)
    stats = {
        "total": total,
        "enriched": sum(1 for m in data if m["enrich_version"] is not None),
        "validated": sum(1 for m in data if m["validated"] is True),
        "validation_failed": sum(1 for m in data if m["validation_failed"] is True),
        "exported": total,
        "with_readme": sum(1 for m in data if m["readme"]),
        "with_memory": sum(1 for m in data if m["memory_requirements"]),
        "uncensored": sum(1 for m in data if m["is_uncensored"] is True),
        "output_path": str(output_path),
    }

    print(
        f"[EXPORTER] ✅ {total} model dışa aktarıldı → {output_path}\n"
        f"[EXPORTER]    validated={stats['validated']} | "
        f"failed={stats['validation_failed']} | "
        f"uncensored={stats['uncensored']}",
        flush=True,
    )
    logger.info(
        f"Exported {total} models → {output_path} | "
        f"validated={stats['validated']} | "
        f"failed={stats['validation_failed']}"
    )
    return stats
