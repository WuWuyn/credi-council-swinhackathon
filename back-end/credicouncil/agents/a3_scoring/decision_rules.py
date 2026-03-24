"""
CREDICOUNCIL A3 — Hard Override Decision Rules.

Policy-based rules that override ML scoring decisions.
These are deterministic, non-negotiable business rules.
"""

from __future__ import annotations

import logging
from typing import Any

from credicouncil.state.credit_state import RoutingDecision

logger = logging.getLogger(__name__)


def apply_hard_overrides(
    credit_score: int,
    risk_band: str,
    auto_decision: str,
    structured_feats: dict[str, Any],
    overall_confidence: float,
) -> dict[str, Any]:
    """Apply hard override rules on top of ML scoring decision.

    These policy rules ALWAYS take precedence over the ML model's
    decision. They are based on regulatory requirements and bank policy.

    Rules (from design document Section 6.4):
        1. CIC Nhóm 4-5 → REJECT regardless of ML score
        2. Loan amount > 10 tỷ VND → ESCALATE to head office
        3. overall_confidence < 0.65 → HUMAN REVIEW before any decision
        4. thin_file_flag=True + score < 560 → increase collateral requirement

    Args:
        credit_score: ML-generated credit score (300-850).
        risk_band: Risk band classification.
        auto_decision: ML-recommended decision.
        structured_feats: Structured features from ingestion.
        overall_confidence: Overall data confidence score.

    Returns:
        Dict with final_decision, override_applied, override_reason, conditions.
    """
    overrides: list[str] = []
    conditions: list[str] = []
    final_decision = auto_decision

    # ── Rule 1: CIC Group 4-5 → Mandatory REJECT ──
    debt_group = structured_feats.get("debt_group") or structured_feats.get("debt_group_proxy", 1)
    if isinstance(debt_group, (int, float)) and debt_group >= 4:
        final_decision = RoutingDecision.REJECT.value
        overrides.append(
            f"CIC Nhóm {int(debt_group)} → REJECT bắt buộc theo quy định "
            f"(bất kể ML score {credit_score})"
        )
        logger.warning(f"OVERRIDE: CIC Group {debt_group} → REJECT (score was {credit_score})")

    # ── Rule 2: High loan amount → ESCALATE ──
    loan_amount = structured_feats.get("loan_amount_vnd", 0)
    if isinstance(loan_amount, (int, float)) and loan_amount > 10_000_000_000:
        final_decision = RoutingDecision.ESCALATE.value
        overrides.append(
            f"Khoản vay {loan_amount:,.0f} VND > 10 tỷ → ESCALATE phê duyệt cấp cao"
        )
        logger.warning(f"OVERRIDE: Loan {loan_amount:,.0f} VND > 10B → ESCALATE")

    # ── Rule 3: Low confidence → HUMAN REVIEW ──
    if overall_confidence < 0.65:
        if final_decision not in (RoutingDecision.REJECT.value, RoutingDecision.ESCALATE.value):
            final_decision = RoutingDecision.REVIEW.value
        overrides.append(
            f"Confidence {overall_confidence:.1%} < 65% → Yêu cầu xem xét thủ công"
        )
        logger.warning(f"OVERRIDE: Low confidence {overall_confidence:.1%} → HUMAN REVIEW")

    # ── Rule 4: Thin-file + low score → Increase collateral ──
    thin_file = structured_feats.get("thin_file_flag", False)
    if thin_file and credit_score < 560:
        conditions.append(
            "Khách hàng thin-file với score < 560 → Yêu cầu tài sản bảo đảm bổ sung"
        )
        if final_decision not in (RoutingDecision.REJECT.value, RoutingDecision.ESCALATE.value):
            final_decision = "CONDITIONAL"
        logger.info(f"CONDITION: Thin-file + score {credit_score} < 560 → increase TSBĐ")

    result = {
        "final_decision": final_decision,
        "override_applied": len(overrides) > 0,
        "override_reasons": overrides,
        "additional_conditions": conditions,
        "original_auto_decision": auto_decision,
        "credit_score": credit_score,
        "risk_band": risk_band,
    }

    if overrides:
        logger.info(f"Hard overrides applied: {len(overrides)} rules triggered")
    else:
        logger.debug(f"No hard overrides — using ML decision: {auto_decision}")

    return result
