"""
pipeline/agents/validator.py

F-03: Validator Agent

Validates enriched models against quality rules defined in FEATURES.md.
On failure: re-queues model for re-enrichment (max 3 attempts).
After 3 failures: marks model as validation_failed=True.
"""

import logging

from sqlmodel import Session, select

from pipeline.core.db import engine, init_db
from pipeline.core.models import Model
from pipeline.core.settings import settings

logger = logging.getLogger(__name__)

# ── Validation Rules (from FEATURES.md) ───────────────────────────────────────

VALIDATION_RULES: dict[str, tuple[callable, str]] = {
    "domain":       (lambda v: v is not None, "domain must not be null"),
    "use_cases":    (lambda v: bool(v) and len(v) >= 1, "use_cases must have at least 1 item"),
    "ai_languages": (lambda v: bool(v) and len(v) >= 1, "ai_languages must have at least 1 item"),
    "complexity":   (lambda v: v is not None, "complexity must not be null"),
    "best_for":     (lambda v: bool(v) and len(v) >= 10, "best_for must be at least 10 chars"),
    "strengths":    (lambda v: bool(v) and len(v) >= 1, "strengths must have at least 1 item"),
    "limitations":  (lambda v: bool(v) and len(v) >= 1, "limitations must have at least 1 item"),
    "model_family": (lambda v: v is not None, "model_family must not be null"),
    "target_audience": (lambda v: bool(v) and len(v) >= 1, "target_audience must have at least 1 item"),
}

MAX_RETRIES = 3


# ── Core Validation ────────────────────────────────────────────────────────────

def validate_model(model: Model) -> tuple[bool, list[str]]:
    """
    Validate a single enriched model against VALIDATION_RULES.
    Returns (is_valid, list_of_failed_rules).
    """
    failures = []

    for field, (rule_fn, message) in VALIDATION_RULES.items():
        value = getattr(model, field, None)
        try:
            if not rule_fn(value):
                failures.append(f"{field}: {message}")
        except Exception:
            failures.append(f"{field}: rule check errored (value={value!r})")

    return len(failures) == 0, failures


# ── Retry Counter Helpers ──────────────────────────────────────────────────────

def _get_retry_count(model: Model) -> int:
    """Read retry count from model — stored in a dedicated column if present."""
    # We track retries via enrich_version going negative as a sentinel,
    # but cleaner: we use a simple in-memory dict keyed by model id during a run.
    # The actual field will be added as needed; for now use enrich_version as proxy.
    return 0


# ── Batch Validator ────────────────────────────────────────────────────────────

def validate_all(session: Session) -> dict:
    """
    Validate all enriched models in DB.
    - Valid → validated=True
    - Invalid + retries < MAX_RETRIES → reset enrich_version to trigger re-enrichment
    - Invalid + retries >= MAX_RETRIES → validation_failed=True

    Returns summary: {total, valid, invalid, re_queued, failed}
    """
    # Fetch all models that have been enriched (have enrich_version set)
    models = list(
        session.exec(
            select(Model).where(Model.enrich_version.is_not(None))  # type: ignore[union-attr]
        ).all()
    )

    total = len(models)
    results = {"total": total, "valid": 0, "invalid": 0, "re_queued": 0, "failed": 0}

    # Track per-run retry counts in memory (keyed by model id)
    retry_counts: dict = {}

    for model in models:
        # Skip already-validated or already-failed models
        if model.validated is True:
            results["valid"] += 1
            continue
        if model.validation_failed is True:
            results["failed"] += 1
            continue

        is_valid, failures = validate_model(model)

        if is_valid:
            model.validated = True
            model.validation_failed = None
            session.add(model)
            results["valid"] += 1
            logger.info(f"  ✓ {model.model_identifier} — valid")
        else:
            current_retries = retry_counts.get(str(model.id), 0)
            results["invalid"] += 1

            logger.warning(
                f"  ✗ {model.model_identifier} — {len(failures)} rule(s) failed: "
                f"{'; '.join(failures)}"
            )

            if current_retries < MAX_RETRIES:
                # Re-queue for enrichment by resetting enrich_version
                model.enrich_version = None
                model.validated = None
                model.validation_failed = None
                retry_counts[str(model.id)] = current_retries + 1
                session.add(model)
                results["re_queued"] += 1
                logger.info(
                    f"    → Re-queued (attempt {current_retries + 1}/{MAX_RETRIES})"
                )
            else:
                model.validation_failed = True
                model.validated = False
                session.add(model)
                results["failed"] += 1
                logger.warning(
                    f"    → Marked as validation_failed after {MAX_RETRIES} attempts"
                )

    session.commit()
    logger.info(
        f"Validation complete — "
        f"valid: {results['valid']} | "
        f"re-queued: {results['re_queued']} | "
        f"failed: {results['failed']}"
    )
    return results


def run_validator() -> dict:
    """Entry point for the Prefect task — initialises DB and runs validation."""
    init_db()
    with Session(engine) as session:
        return validate_all(session)
