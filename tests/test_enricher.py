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

from pipeline.agents.enricher import build_prompt, clean_readme, get_unenriched_models
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
        result = clean_readme(long, max_chars=700)
        assert len(result) == 700

    def test_default_max_is_700(self):
        long = "a" * 1000
        result = clean_readme(long)
        assert len(result) == 700

    def test_short_text_preserved(self):
        text = "Short model description."
        result = clean_readme(text)
        assert result == text


# ── build_prompt ───────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_model_identifier(self):
        model = make_model("deepseek-r1", description="Fast reasoning model")
        prompt = build_prompt(model)
        assert "deepseek-r1" in prompt

    def test_contains_model_name(self):
        model = make_model("llama3")
        prompt = build_prompt(model)
        assert "Llama3" in prompt or "llama3" in prompt

    def test_contains_description(self):
        model = make_model("gemma2", description="Google's language model")
        prompt = build_prompt(model)
        assert "Google's language model" in prompt

    def test_none_description_shows_na(self):
        model = make_model("phi3", description=None)
        prompt = build_prompt(model)
        assert "N/A" in prompt

    def test_contains_readme_snippet(self):
        model = make_model("mistral", readme="A powerful base model from Mistral AI.")
        prompt = build_prompt(model)
        assert "powerful base model" in prompt

    def test_readme_truncated_in_prompt(self):
        model = make_model("qwen", readme="x" * 2000)
        prompt = build_prompt(model)
        # Prompt should contain the truncated readme, not the full 2000 chars
        assert prompt.count("x") <= 700

    def test_capabilities_included(self):
        model = make_model("llava", capabilities=["Vision", "Tools"])
        prompt = build_prompt(model)
        assert "Vision" in prompt

    def test_labels_included(self):
        model = make_model("llama3", labels=["8b", "70b"])
        prompt = build_prompt(model)
        assert "8b" in prompt


# ── get_unenriched_models ──────────────────────────────────────────────────────

class TestGetUnenrichedModels:
    def test_returns_unenriched_models(self, in_memory_session):
        m1 = make_model("llama3", enrich_version=None)
        m2 = make_model("mistral", enrich_version=1)
        in_memory_session.add(m1)
        in_memory_session.add(m2)
        in_memory_session.commit()

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
