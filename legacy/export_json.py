"""
export_json.py

Exports all models from SQLite to models.json for Supabase seeding.

Usage:
    poetry run python export_json.py
    # → models.json
"""

import json
import uuid
from datetime import date, datetime
from pathlib import Path

from sqlmodel import Session, select

from db import Model, engine


def model_to_dict(m: Model) -> dict:
    return {
        # ── Identity ───────────────────────────────────────────
        "id": str(m.id),
        "model_identifier": m.model_identifier,
        "model_name": m.model_name,
        "model_type": m.model_type,
        "namespace": m.namespace,
        "url": m.url,

        # ── Description & Content ──────────────────────────────
        "description": m.description,
        "readme": m.readme,

        # ── HTML-parsed capabilities ───────────────────────────
        # e.g. ["Tools", "Vision", "Thinking", "Cloud"]
        "capabilities": m.capabilities or [],
        "capability": m.capability,   # legacy single string

        # ── Size labels (parameters only) ──────────────────────
        # e.g. ["8b", "70b", "405b"]
        "labels": m.labels or [],

        # ── Hardware compatibility ──────────────────────────────
        # [{tag, size, size_gb, recommended_ram_gb, quantization, context}]
        "memory_requirements": m.memory_requirements or [],
        # Smallest variant size in GB — used for GPU/RAM filter
        "min_ram_gb": m.min_ram_gb,

        # ── AI Enrichment (enrich.py / glm-5:cloud) ───────────
        # e.g. ["Code Generation", "RAG / Retrieval"]
        "use_cases": m.use_cases or [],
        # "Code" | "General" | "Vision" | "Embedding" | "Reasoning" | ...
        "domain": m.domain,
        # ["English", "Multilingual", "Chinese", ...]
        "ai_languages": m.ai_languages or [],
        # "beginner" | "intermediate" | "advanced"
        "complexity": m.complexity,
        # One-liner ideal use-case description
        "best_for": m.best_for,

        # ── Stats ──────────────────────────────────────────────
        "pulls": m.pulls,
        "tags": m.tags,

        # ── Dates ──────────────────────────────────────────────
        "last_updated": m.last_updated.isoformat() if m.last_updated else None,
        "last_updated_str": m.last_updated_str,
        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
    }


def main():
    output_path = Path("models.json")

    with Session(engine) as session:
        models = session.exec(select(Model)).all()
        data = [model_to_dict(m) for m in models]

    with_ram = sum(1 for m in data if m["min_ram_gb"] is not None)
    with_caps = sum(1 for m in data if m["capabilities"])
    with_readme = sum(1 for m in data if m["readme"])
    with_ai = sum(1 for m in data if m["use_cases"])

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Exported {len(data)} models -> {output_path}")
    print(f"  min_ram_gb   : {with_ram}/{len(data)}")
    print(f"  capabilities : {with_caps}/{len(data)}")
    print(f"  readme       : {with_readme}/{len(data)}")
    print(f"  AI enriched  : {with_ai}/{len(data)} (run enrich.py --only-missing for rest)")
    print()

    print("Sample (codellama):")
    sample = next((m for m in data if m["model_identifier"] == "codellama"), data[0])
    preview = {k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
               for k, v in sample.items() if k != "readme"}
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
