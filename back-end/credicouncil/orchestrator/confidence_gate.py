"""
CREDICOUNCIL Orchestrator — LangGraph Confidence Gate.

Conditional router that determines pipeline routing based on
critical field confidence and overall data quality.
"""

from __future__ import annotations

import logging
from typing import Any

from credicouncil.config.feature_config import (
    FIELD_DEFINITIONS,
    CONFIDENCE_THRESHOLDS,
    TIER_WEIGHTS,
    FeatureTier,
    OVERALL_CONFIDENCE_AUTO_PROCEED,
    OVERALL_CONFIDENCE_PROCEED_WITH_WARNINGS,
)
from credicouncil.state.credit_state import CreditState, RoutingDecision

logger = logging.getLogger(__name__)


def confidence_gate(state: CreditState) -> dict[str, Any]:
    """Confidence gate — LangGraph conditional router node.

    Logic:
        1. Check ALL CRITICAL fields have confidence ≥ 0.85
           - If ANY critical field fails → HALT pipeline
        2. Calculate overall_confidence
           - ≥ 0.85 → AUTO_PROCEED
           - 0.70-0.85 → PROCEED_WITH_WARNINGS
           - < 0.70 → ESCALATE_TO_HUMAN

    Args:
        state: Current pipeline state with confidence_map.

    Returns:
        State update with routing decision.
    """
    confidence_map = state.get("confidence_map", {})
    missing_fields = state.get("missing_fields", [])

    logger.info("Confidence gate evaluation started")

    # ── Step 1: Check CRITICAL fields ──
    critical_failures = []
    for field_name, field_def in FIELD_DEFINITIONS.items():
        if field_def.tier != FeatureTier.CRITICAL:
            continue

        conf = confidence_map.get(field_name, 0.0)
        threshold = CONFIDENCE_THRESHOLDS[FeatureTier.CRITICAL]

        if conf < threshold:
            critical_failures.append({
                "field": field_name,
                "confidence": conf,
                "threshold": threshold,
                "gap": round(threshold - conf, 3),
            })

    if critical_failures:
        logger.warning(
            f"HALT: {len(critical_failures)} critical field(s) below threshold"
        )
        for failure in critical_failures:
            logger.warning(
                f"  CRITICAL: {failure['field']} = {failure['confidence']:.2f} "
                f"< {failure['threshold']:.2f}"
            )

        return {
            "routing": RoutingDecision.HALT.value,
            "warnings": state.get("warnings", []) + [
                f"⛔ Pipeline dừng: trường critical '{f['field']}' "
                f"có confidence {f['confidence']:.0%} < {f['threshold']:.0%}. "
                f"Yêu cầu bổ sung tài liệu."
                for f in critical_failures
            ],
        }

    # ── Step 2: Calculate overall confidence ──
    weighted_sum = 0.0
    weight_total = 0.0

    for field_name, field_def in FIELD_DEFINITIONS.items():
        conf = confidence_map.get(field_name, 0.0)
        weight = TIER_WEIGHTS[field_def.tier]
        weighted_sum += weight * conf
        weight_total += weight

    overall_confidence = weighted_sum / weight_total if weight_total > 0 else 0.0

    # ── Step 3: Route based on overall confidence ──
    if overall_confidence >= OVERALL_CONFIDENCE_AUTO_PROCEED:
        routing = RoutingDecision.PROCEED.value
        logger.info(f"AUTO_PROCEED: overall_confidence = {overall_confidence:.3f}")
    elif overall_confidence >= OVERALL_CONFIDENCE_PROCEED_WITH_WARNINGS:
        routing = RoutingDecision.PROCEED_WITH_WARNINGS.value
        logger.info(f"PROCEED_WITH_WARNINGS: overall_confidence = {overall_confidence:.3f}")
    else:
        routing = RoutingDecision.ESCALATE_TO_HUMAN.value
        logger.warning(f"ESCALATE_TO_HUMAN: overall_confidence = {overall_confidence:.3f}")

    # Identify fields needing imputation
    imputation_needed = []
    for field_name, field_def in FIELD_DEFINITIONS.items():
        if field_def.tier == FeatureTier.IMPORTANT:
            conf = confidence_map.get(field_name, 0.0)
            if conf < CONFIDENCE_THRESHOLDS[FeatureTier.IMPORTANT]:
                imputation_needed.append(field_name)

    warnings = state.get("warnings", [])
    if imputation_needed:
        warnings.append(
            f"Các trường cần ước tính (imputation): {', '.join(imputation_needed)}"
        )

    return {
        "routing": routing,
        "overall_confidence": round(overall_confidence, 3),
        "missing_fields": missing_fields + imputation_needed,
        "warnings": warnings,
    }
