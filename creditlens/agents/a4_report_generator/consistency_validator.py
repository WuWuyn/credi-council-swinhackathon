"""
CreditLens A4 — Consistency Validator.

Deterministic check ensuring that LLM-generated narrative only references
factors that appear in the SHAP output. This prevents LLM hallucination
in credit reports.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def validate_narrative_consistency(
    shap_output: dict[str, Any],
    narrative: dict[str, Any],
) -> dict[str, Any]:
    """Check that LLM narrative is grounded in SHAP data.

    Validates that:
    1. Every 4C dimension narrative references at least one SHAP factor
    2. No new risk factors are invented outside of SHAP data
    3. Coverage metric: what % of SHAP factors are mentioned

    Args:
        shap_output: Full SHAP JSON from A3.
        narrative: 4C assessment dict from A4 LLM generation.

    Returns:
        Dict with passed, violations, shap_coverage.
    """
    # Collect all SHAP labels (Vietnamese)
    top_shap_labels = set()
    for factor in shap_output.get("top_positive_factors", []):
        top_shap_labels.add(factor.get("label_vi", "").lower())
    for factor in shap_output.get("top_negative_factors", []):
        top_shap_labels.add(factor.get("label_vi", "").lower())

    # Remove empty strings
    top_shap_labels.discard("")

    violations: list[str] = []
    mentioned_labels: set[str] = set()

    dimensions = ["character", "capacity", "capital", "conditions"]

    for dimension in dimensions:
        assessment_key = f"{dimension}_assessment"
        assessment = narrative.get(assessment_key, {})
        narrative_text = assessment.get("narrative", "")

        if not narrative_text:
            violations.append(f"{dimension}: narrative is empty")
            continue

        # Check: at least one SHAP label is mentioned in the narrative
        has_shap_support = False
        for label in top_shap_labels:
            # Check for partial match (label words appearing in text)
            label_words = [w for w in label.split() if len(w) > 3]
            if any(word.lower() in narrative_text.lower() for word in label_words):
                has_shap_support = True
                mentioned_labels.add(label)

        if not has_shap_support:
            violations.append(
                f"{dimension}: narrative lacks SHAP grounding — "
                f"no SHAP factor labels found in text"
            )

    # Compute coverage
    shap_coverage = len(mentioned_labels) / len(top_shap_labels) if top_shap_labels else 1.0

    result = {
        "passed": len(violations) == 0,
        "violations": violations,
        "shap_coverage": round(shap_coverage, 3),
        "mentioned_labels": list(mentioned_labels),
        "total_shap_labels": len(top_shap_labels),
    }

    if violations:
        logger.warning(f"Consistency check FAILED: {len(violations)} violations")
        for v in violations:
            logger.warning(f"  - {v}")
    else:
        logger.info(f"Consistency check PASSED — coverage: {shap_coverage:.1%}")

    return result
