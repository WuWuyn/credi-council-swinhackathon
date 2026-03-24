"""
CREDICOUNCIL A4 — Deterministic Decision Engine.

Rule-based credit decision routing following document_new.md §3.3:
  - Credit Score → Risk Band → Base Decision
  - Hard Override Rules (CIC, LTV, thin-file)
  - DTI/DSCR guard rails

Design: Same input → Same output. No LLM involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Decision result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CreditDecision:
    """Deterministic credit decision result."""
    recommendation: str      # APPROVE | APPROVE_REVIEW | REVIEW | CONDITIONAL | REJECT | ESCALATE
    risk_band: str           # AAA, AA, A, BBB, BB, B, CCC, CC, C
    credit_score: int
    reasons: list[str] = field(default_factory=list)
    overrides_applied: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    @property
    def recommendation_vi(self) -> str:
        """Vietnamese label for recommendation."""
        return {
            "APPROVE": "PHÊ DUYỆT",
            "APPROVE_REVIEW": "DUYỆT XEM XÉT",
            "REVIEW": "XEM XÉT",
            "CONDITIONAL": "CÓ ĐIỀU KIỆN",
            "REJECT": "TỪ CHỐI",
            "ESCALATE": "LEO THANG",
        }.get(self.recommendation, self.recommendation)


# ─────────────────────────────────────────────────────────────────────────────
# Decision rules (from document_new.md §3.3)
# ─────────────────────────────────────────────────────────────────────────────

BAND_DECISION = [
    # (min_score, max_score, band, base_decision, extra_condition_desc)
    (720, 850, "AAA", "APPROVE",         "CIC Nhóm 1 + DTI < 40%"),
    (640, 719, "AA",  "APPROVE_REVIEW",  "Chuyên viên xem nhanh báo cáo"),
    (560, 639, "A",   "REVIEW",          "Đánh giá đầy đủ 5C + điều kiện"),
    (460, 559, "BBB", "CONDITIONAL",     "Cần bổ sung TSBĐ hoặc guarantor"),
    (300, 459, "CC",  "REJECT",          "Trừ đặc cách có thẩm quyền cao"),
]


def _base_decision(credit_score: int) -> tuple[str, str, str]:
    """Map credit score to (risk_band, base_recommendation, description)."""
    for min_s, max_s, band, decision, desc in BAND_DECISION:
        if min_s <= credit_score <= max_s:
            return band, decision, desc
    # Below 300 — extreme risk
    return "C", "REJECT", "Điểm tín dụng dưới ngưỡng tối thiểu"


# ─────────────────────────────────────────────────────────────────────────────
# Hard override rules
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overrides(
    base_decision: str,
    credit_score: int,
    app_row: dict,
    financial_ratios: dict,
    five_c_scores: dict,
    llm_feats: dict,
) -> tuple[str, list[str], list[str]]:
    """
    Apply hard override rules on top of base decision.

    Returns: (final_decision, overrides_applied, conditions)
    """
    fr = financial_ratios or {}
    app = app_row or {}
    feats = llm_feats or {}
    decision = base_decision
    overrides = []
    conditions = []

    # ── Override 1: DTI guard for APPROVE ──
    # AAA auto-approve requires DTI < 40%
    dti = fr.get("dti")
    if decision == "APPROVE" and dti is not None and dti >= 0.40:
        decision = "APPROVE_REVIEW"
        overrides.append(f"DTI {dti*100:.1f}% >= 40% → chuyển từ APPROVE sang APPROVE_REVIEW")
        conditions.append("Yêu cầu chứng minh thu nhập bổ sung do DTI cao")

    # ── Override 2: LTV > 80% adds conditions ──
    ltv = fr.get("ltv")
    if ltv is not None and ltv > 0.80:
        conditions.append(f"LTV {ltv*100:.0f}% vượt ngưỡng 80% — yêu cầu bổ sung TSBĐ hoặc giảm hạn mức")
        if decision in ("APPROVE", "APPROVE_REVIEW"):
            decision = "REVIEW"
            overrides.append(f"LTV {ltv*100:.0f}% > 80% → chuyển sang REVIEW")

    # ── Override 3: Thin-file + Score < 560 ──
    thin_file = feats.get("thin_file_flag", False)
    if thin_file and credit_score < 560:
        conditions.append("Thin-file + Score < 560 — tăng yêu cầu tài sản thế chấp")
        if decision not in ("REJECT",):
            decision = "CONDITIONAL"
            overrides.append("Thin-file + low score → CONDITIONAL")

    # ── Override 4: 5C total too low for the base decision ──
    total_5c = sum(s.get("score", 0) if isinstance(s, dict) else getattr(s, "score", 0)
                   for s in (five_c_scores or {}).values())
    if total_5c < 40 and decision not in ("REJECT",):
        decision = "REJECT"
        overrides.append(f"5C total {total_5c}/120 < 40 → REJECT")
    elif total_5c < 60 and decision in ("APPROVE", "APPROVE_REVIEW"):
        decision = "REVIEW"
        overrides.append(f"5C total {total_5c}/120 < 60 → chuyển sang REVIEW")

    # ── Override 5: DSCR < 1.0 → cannot approve ──
    dscr = fr.get("dscr")
    if dscr is not None and dscr < 1.0 and decision in ("APPROVE", "APPROVE_REVIEW"):
        decision = "REVIEW"
        overrides.append(f"DSCR {dscr:.2f} < 1.0 → chuyển sang REVIEW")
        conditions.append("DSCR dưới 1.0 — khả năng trả nợ không đảm bảo")

    # ── Override 6: Very low income warning ──
    income_monthly = fr.get("income_monthly_vnd")
    if income_monthly is not None and income_monthly < 100_000:
        conditions.append(f"Thu nhập {income_monthly:,.0f} VND/tháng — cực kỳ thấp, cần xác minh")

    return decision, overrides, conditions


# ─────────────────────────────────────────────────────────────────────────────
# Main decision engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_decision(
    credit_score: int,
    app_row: dict,
    financial_ratios: dict,
    five_c_scores: dict,
    llm_feats: dict,
) -> CreditDecision:
    """
    Compute deterministic credit decision.

    Flow:
      1. Credit Score → Risk Band → Base Decision
      2. Apply hard override rules
      3. Return final decision with reasons
    """
    # Step 1: Base decision from credit score
    risk_band, base_decision, base_desc = _base_decision(credit_score)
    reasons = [f"Credit Score {credit_score} → Band {risk_band} → {base_decision} ({base_desc})"]

    # Step 2: Apply overrides
    final_decision, overrides, conditions = _apply_overrides(
        base_decision=base_decision,
        credit_score=credit_score,
        app_row=app_row,
        financial_ratios=financial_ratios,
        five_c_scores=five_c_scores,
        llm_feats=llm_feats,
    )

    if overrides:
        reasons.extend(overrides)

    result = CreditDecision(
        recommendation=final_decision,
        risk_band=risk_band,
        credit_score=credit_score,
        reasons=reasons,
        overrides_applied=overrides,
        conditions=conditions,
    )

    logger.info(f"  Decision Engine: {credit_score} → {risk_band} → {final_decision}")
    if overrides:
        for o in overrides:
            logger.info(f"    Override: {o}")

    return result
