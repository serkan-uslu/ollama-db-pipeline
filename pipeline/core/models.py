"""
pipeline/core/models.py

SQLModel table definition — single source of truth for the DB schema.
Matches the schema defined in FEATURES.md exactly.

JSON fields use SQLAlchemy JSON column type.
All nullable fields use Field(default=None).
"""

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Text
from sqlmodel import Field, SQLModel


class Model(SQLModel, table=True):
    # ── Identity ──────────────────────────────────────────────────────────────
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
    )
    model_identifier: str = Field(index=True, unique=True)  # e.g. "deepseek-r1"
    model_name: str
    model_type: str = "official"
    namespace: Optional[str] = Field(default=None)
    url: str

    # ── Raw Scraped ────────────────────────────────────────────────────────────
    description: Optional[str] = Field(default=None)
    readme: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    raw_html: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))  # full detail page HTML for re-parsing

    # Capability tags: ["Tools", "Vision", "Thinking", "Cloud"]
    capability: Optional[str] = Field(default=None)  # legacy single string
    capabilities: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    # Size labels: ["8b", "70b", "405b"]
    labels: list = Field(sa_column=Column(JSON, nullable=False))

    # ── Stats ──────────────────────────────────────────────────────────────────
    pulls: int = Field(default=0)
    tags: int = Field(default=0)
    last_updated: Optional[date] = Field(default=None)
    last_updated_str: Optional[str] = Field(default=None)

    # ── Applications ───────────────────────────────────────────────────────────
    # [{name: "Claude Code", launch_command: "ollama launch claude --model lfm2"}]
    applications: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    # ── Hardware ───────────────────────────────────────────────────────────────
    # [{tag, size, size_gb, recommended_ram_gb, quantization, context, context_window}]
    memory_requirements: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    min_ram_gb: Optional[float] = Field(default=None)     # smallest variant GB
    context_window: Optional[int] = Field(default=None)   # max tokens
    speed_tier: Optional[str] = Field(default=None)       # "fast" | "medium" | "slow"

    # ── AI Enriched ───────────────────────────────────────────────────────────
    use_cases: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    domain: Optional[str] = Field(default=None)
    ai_languages: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    complexity: Optional[str] = Field(default=None)      # "beginner"|"intermediate"|"advanced"
    best_for: Optional[str] = Field(default=None)

    license: Optional[str] = Field(default=None)
    base_model: Optional[str] = Field(default=None)
    model_family: Optional[str] = Field(default=None)    # "Llama" | "Mistral" | "Qwen" | ...
    is_fine_tuned: Optional[bool] = Field(default=None)
    is_uncensored: Optional[bool] = Field(default=None)  # True if uncensored/abliterated

    strengths: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    limitations: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    target_audience: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    # ── Pipeline Metadata ─────────────────────────────────────────────────────
    enrich_version: Optional[int] = Field(default=None)
    validated: Optional[bool] = Field(default=None)
    validation_failed: Optional[bool] = Field(default=None)
    timestamp: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    class Config:
        protected_namespaces = ()
