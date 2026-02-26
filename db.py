"""
db.py — Self-contained database setup for ollama-db-pipeline

Replaces the app/ folder. Contains:
  - SQLite engine setup
  - Model table definition (all fields)
"""

import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from sqlmodel import JSON, Column, DateTime, Field, SQLModel, Text, create_engine, func

# ── Engine ────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "ollama_models.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


# ── Model Table ───────────────────────────────────────────────────────────────
class Model(SQLModel, table=True):
    # Identity
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    model_identifier: str = Field(index=True, unique=True)
    model_name: str
    model_type: str
    namespace: Optional[str] = None
    url: str

    # Content
    description: Optional[str] = None
    readme: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    # HTML-parsed capability tags (Vision, Tools, Thinking, Embedding, Cloud)
    capabilities: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
    capability: Optional[str] = None   # legacy single-string, backward compat

    # Size labels (parameters only, e.g. ["8b", "70b"])
    labels: list = Field(sa_column=Column(JSON, nullable=False))

    # Hardware compatibility
    # [{tag, size_gb, recommended_ram_gb, quantization, context, context_window}]
    memory_requirements: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
    min_ram_gb: Optional[float] = None          # smallest variant GB (GPU filter)
    context_window: Optional[int] = None        # max tokens (parsed from context string)
    speed_tier: Optional[str] = None            # fast | medium | slow

    # Stats
    pulls: int = 0
    tags: int = 0
    last_updated: Optional[date] = None
    last_updated_str: Optional[str] = None
    timestamp: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True, server_default=func.now()),
    )

    # ── AI Enrichment (enrich.py / qwen2.5:3b) ───────────────────────────────
    # Bump ENRICH_VERSION in enrich.py when adding new fields.
    enrich_version: Optional[int] = None

    use_cases: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
    domain: Optional[str] = None                # General|Code|Vision|Embedding|Reasoning|...
    ai_languages: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
    complexity: Optional[str] = None            # beginner|intermediate|advanced
    best_for: Optional[str] = None              # one-liner ideal use case

    license: Optional[str] = None               # MIT|Apache 2.0|Llama 3 Community|...
    base_model: Optional[str] = None            # e.g. "llama3.1" for fine-tunes
    is_fine_tuned: Optional[bool] = None
    strengths: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
    limitations: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
    target_audience: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Validators
    @field_validator("labels", mode="before")
    def parse_labels(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("capabilities", mode="before")
    def parse_capabilities(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        protected_namespaces = ()
