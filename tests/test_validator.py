"""
tests/test_validator.py

Unit tests for pipeline/agents/validator.py

Covers:
- validate_model()   — single model validation logic
- validate_all()     — batch validation with re-queue and failure marking
- VALIDATION_RULES   — each individual rule checked independently
"""

import uuid
from datetime import datetime

import pytest
from sqlmodel import Session, SQLModel, create_engine

from pipeline.agents.validator import MAX_RETRIES, VALIDATION_RULES, validate_all, validate_model
from pipeline.core.models import Model


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def in_memory_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def make_enriched_model(
    identifier: str = "test-model",
    enrich_version: int = 1,
    **overrides,
) -> Model:
    defaults = dict(
        id=uuid.uuid4(),
        model_identifier=identifier,
        model_name=identifier.replace("-", " ").title(),
        model_type="official",
        url=f"https://ollama.com/library/{identifier}",
        labels=["7b"],
        pulls=50_000,
        tags=5,
        timestamp=datetime.now(),
        enrich_version=enrich_version,
        domain="General",
        use_cases=["Chat Assistant"],
        ai_languages=["English"],
        complexity="intermediate",
        best_for="General purpose question answering and chat",
        strengths=["Versatile", "Fast"],
        limitations=["Small context window"],
        model_family="Llama",
        target_audience=["Developers"],
    )
    defaults.update(overrides)
    return Model(**defaults)


# ── validate_model ─────────────────────────────────────────────────────────────

class TestValidateModel:
    def test_valid_model_passes(self):
        model = make_enriched_model()
        is_valid, failures = validate_model(model)
        assert is_valid is True
        assert failures == []

    def test_missing_domain_fails(self):
        model = make_enriched_model(domain=None)
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("domain" in f for f in failures)

    def test_empty_use_cases_fails(self):
        model = make_enriched_model(use_cases=[])
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("use_cases" in f for f in failures)

    def test_none_use_cases_fails(self):
        model = make_enriched_model(use_cases=None)
        is_valid, failures = validate_model(model)
        assert is_valid is False

    def test_empty_ai_languages_fails(self):
        model = make_enriched_model(ai_languages=[])
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("ai_languages" in f for f in failures)

    def test_missing_complexity_fails(self):
        model = make_enriched_model(complexity=None)
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("complexity" in f for f in failures)

    def test_short_best_for_fails(self):
        model = make_enriched_model(best_for="Short")
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("best_for" in f for f in failures)

    def test_best_for_exactly_10_chars_passes(self):
        model = make_enriched_model(best_for="1234567890")
        is_valid, failures = validate_model(model)
        # check only the best_for rule
        best_for_failures = [f for f in failures if "best_for" in f]
        assert best_for_failures == []

    def test_empty_strengths_fails(self):
        model = make_enriched_model(strengths=[])
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("strengths" in f for f in failures)

    def test_empty_limitations_passes(self):
        """Empty limitations list is valid (schema allows min_length=0)."""
        model = make_enriched_model(limitations=[])
        is_valid, failures = validate_model(model)
        assert is_valid is True

    def test_null_limitations_fails(self):
        """None limitations fails the 'must not be null' rule."""
        model = make_enriched_model(limitations=None)
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("limitations" in f for f in failures)

    def test_multiple_failures_reported(self):
        model = make_enriched_model(domain=None, use_cases=[], complexity=None)
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert len(failures) >= 3

    def test_missing_model_family_fails(self):
        model = make_enriched_model(model_family=None)
        is_valid, failures = validate_model(model)
        assert is_valid is False
        assert any("model_family" in f for f in failures)


# ── validate_all ───────────────────────────────────────────────────────────────

class TestValidateAll:
    def test_valid_models_marked_validated(self, in_memory_session):
        m = make_enriched_model("llama3")
        in_memory_session.add(m)
        in_memory_session.commit()

        stats = validate_all(in_memory_session)

        assert stats["valid"] == 1
        assert stats["invalid"] == 0

        in_memory_session.refresh(m)
        assert m.validated is True

    def test_invalid_model_requeued(self, in_memory_session):
        m = make_enriched_model("bad-model", domain=None)
        in_memory_session.add(m)
        in_memory_session.commit()

        stats = validate_all(in_memory_session)

        assert stats["re_queued"] == 1
        in_memory_session.refresh(m)
        # enrich_version reset to None to trigger re-enrichment
        assert m.enrich_version is None

    def test_already_validated_skipped(self, in_memory_session):
        m = make_enriched_model("validated-model", validated=True)
        in_memory_session.add(m)
        in_memory_session.commit()

        stats = validate_all(in_memory_session)

        # Already validated — counted as valid, not re-processed
        assert stats["valid"] == 1
        assert stats["re_queued"] == 0

    def test_already_failed_skipped(self, in_memory_session):
        m = make_enriched_model(
            "failed-model",
            validation_failed=True,
            validated=False,
        )
        in_memory_session.add(m)
        in_memory_session.commit()

        stats = validate_all(in_memory_session)

        assert stats["failed"] == 1
        assert stats["re_queued"] == 0

    def test_unenriched_model_not_validated(self, in_memory_session):
        m = make_enriched_model("unenriched", enrich_version=None)
        in_memory_session.add(m)
        in_memory_session.commit()

        stats = validate_all(in_memory_session)

        # Model has no enrich_version → should not be processed
        assert stats["total"] == 0

    def test_stats_keys_present(self, in_memory_session):
        stats = validate_all(in_memory_session)
        assert "total" in stats
        assert "valid" in stats
        assert "invalid" in stats
        assert "re_queued" in stats
        assert "failed" in stats

    def test_mixed_batch(self, in_memory_session):
        good = make_enriched_model("good-model")
        bad = make_enriched_model("bad-model", domain=None)
        in_memory_session.add(good)
        in_memory_session.add(bad)
        in_memory_session.commit()

        stats = validate_all(in_memory_session)

        assert stats["valid"] == 1
        assert stats["re_queued"] == 1


# ── VALIDATION_RULES completeness ─────────────────────────────────────────────

class TestValidationRulesStructure:
    """Ensure all required fields from FEATURES.md are covered."""

    REQUIRED_FIELDS = {
        "domain",
        "use_cases",
        "ai_languages",
        "complexity",
        "best_for",
        "strengths",
        "limitations",
    }

    def test_all_required_fields_have_rules(self):
        for field in self.REQUIRED_FIELDS:
            assert field in VALIDATION_RULES, f"Missing validation rule for: {field}"

    def test_rules_are_callable(self):
        for field, (rule_fn, message) in VALIDATION_RULES.items():
            assert callable(rule_fn), f"Rule for '{field}' is not callable"
            assert isinstance(message, str), f"Message for '{field}' is not a string"
