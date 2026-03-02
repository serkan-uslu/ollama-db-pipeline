"""
tests/test_enricher.py

Unit tests for pipeline/agents/enricher.py

Covers:
- clean_readme()       — README trimming/cleaning
- build_prompt()       — prompt generation from model data
- get_unenriched_models() — DB query logic (in-memory SQLite)

LLM calls (enrich_model + run_enricher) are NOT tested here
since they require a live Groq API key.
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from pipeline.agents.enricher import (
    _p1_domain_family,
    _p2_use_cases,
    _p3_basics,
    clean_readme,
    get_unenriched_models,
)
from pipeline.core.models import Model


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def in_memory_session():
    """Provide a fresh in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def make_model(
    identifier: str = "test-model",
    enrich_version: int | None = None,
    **kwargs,
) -> Model:
    return Model(
        id=uuid.uuid4(),
        model_identifier=identifier,
        model_name=identifier.replace("-", " ").title(),
        model_type="official",
        url=f"https://ollama.com/library/{identifier}",
        pulls=100_000,
        tags=10,
        timestamp=datetime.now(),
        enrich_version=enrich_version,
        **{"labels": ["7b", "13b"], **kwargs},
    )


# ── clean_readme ───────────────────────────────────────────────────────────────

class TestCleanReadme:
    def test_none_returns_na(self):
        assert clean_readme(None) == "N/A"

    def test_empty_string_returns_na(self):
        assert clean_readme("") == "N/A"

    def test_collapses_whitespace(self):
        result = clean_readme("hello   world\n\nfoo")
        assert "  " not in result
        assert "\n\n" not in result

    def test_truncates_to_max_chars(self):
        long = "x" * 2000
        result = clean_readme(long, max_chars=500)
        assert len(result) == 500

    def test_default_max_is_1200(self):
        long = "a" * 2000
        result = clean_readme(long)
        assert len(result) == 1200

    def test_short_text_preserved(self):
        text = "Short model description."
        result = clean_readme(text)
        assert result == text


# ── Prompt helpers ────────────────────────────────────────────────────────────
# NOTE: enricher uses 6 focused LLM calls (_p1–_p6) instead of a single
# build_prompt(). Tests here verify each helper includes the expected content.

class TestPromptHelpers:
    """Tests for the individual prompt-builder functions (_p1–_p3)."""

    # ── _p1_domain_family ──────────────────────────────────────────────────────

    def test_p1_contains_model_identifier(self):
        model = make_model("deepseek-r1", description="Fast reasoning model")
        prompt = _p1_domain_family(model)
        assert "deepseek-r1" in prompt

    def test_p1_contains_description(self):
        model = make_model("gemma2", description="Google's language model")
        prompt = _p1_domain_family(model)
        assert "Google's language model" in prompt

    def test_p1_contains_domain_options(self):
        model = make_model("llama3")
        prompt = _p1_domain_family(model)
        assert "General" in prompt
        assert "domain" in prompt.lower()

    def test_p1_contains_family_options(self):
        model = make_model("mistral")
        prompt = _p1_domain_family(model)
        assert "model_family" in prompt
        assert "Llama" in prompt

    # ── _p2_use_cases ──────────────────────────────────────────────────────────

    def test_p2_contains_model_identifier(self):
        model = make_model("llama3", description="General assistant")
        prompt = _p2_use_cases(model)
        assert "llama3" in prompt

    def test_p2_contains_use_case_options(self):
        model = make_model("codellama")
        prompt = _p2_use_cases(model)
        assert "Code Generation" in prompt
        assert "use_cases" in prompt

    # ── _p3_basics ─────────────────────────────────────────────────────────────

    def test_p3_contains_labels(self):
        model = make_model("llama3", labels=["8b", "70b"])
        prompt = _p3_basics(model)
        assert "8b" in prompt

    def test_p3_contains_complexity_options(self):
        model = make_model("phi3")
        prompt = _p3_basics(model)
        assert "complexity" in prompt
        assert "beginner" in prompt


# ── get_unenriched_models ──────────────────────────────────────────────────────

class TestGetUnenrichedModels:
    def test_returns_unenriched_models(self, in_memory_session):
        m1 = make_model("llama3", enrich_version=None)
        m2 = make_model("mistral", enrich_version=1)
        in_memory_session.add(m1)
        in_memory_session.add(m2)
        in_memory_session.commit()

        # Mock enrich_version=1 so that mistral (v=1) is NOT re-enriched
        with patch("pipeline.agents.enricher.settings") as mock_settings:
            mock_settings.enrich_version = 1
            result = get_unenriched_models(in_memory_session)
        ids = [m.model_identifier for m in result]
        assert "llama3" in ids
        assert "mistral" not in ids

    def test_force_returns_all(self, in_memory_session):
        m1 = make_model("llama3", enrich_version=None)
        m2 = make_model("mistral", enrich_version=1)
        in_memory_session.add(m1)
        in_memory_session.add(m2)
        in_memory_session.commit()

        result = get_unenriched_models(in_memory_session, force=True)
        assert len(result) == 2

    def test_single_slug_filter(self, in_memory_session):
        m1 = make_model("llama3", enrich_version=None)
        m2 = make_model("mistral", enrich_version=None)
        in_memory_session.add(m1)
        in_memory_session.add(m2)
        in_memory_session.commit()

        result = get_unenriched_models(in_memory_session, single_slug="mistral")
        assert len(result) == 1
        assert result[0].model_identifier == "mistral"

    def test_empty_db_returns_empty(self, in_memory_session):
        result = get_unenriched_models(in_memory_session)
        assert result == []

    def test_outdated_version_included(self, in_memory_session):
        """Model with enrich_version=0 should be re-enriched when settings.enrich_version=1."""
        m = make_model("phi3", enrich_version=0)
        in_memory_session.add(m)
        in_memory_session.commit()

        with patch("pipeline.agents.enricher.settings") as mock_settings:
            mock_settings.enrich_version = 1
            result = get_unenriched_models(in_memory_session)

        ids = [m.model_identifier for m in result]
        assert "phi3" in ids
