"""
CREDICOUNCIL State Schema — LangGraph CreditState definition.

Defines the shared state that flows through the 4-agent pipeline.
Every agent reads and writes to this state via LangGraph.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict


# ─── Enums ────────────────────────────────────────────────────────────────────


class RoutingDecision(str, Enum):
    """Pipeline routing labels assigned by confidence gate and decision router."""

    AUTO_APPROVE = "AUTO_APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    HALT = "HALT"
    PROCEED = "PROCEED"
    PROCEED_WITH_WARNINGS = "PROCEED_WITH_WARNINGS"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class RiskBand(str, Enum):
    """Credit risk classification bands."""

    AAA = "AAA"  # 720-850, PD < 2%
    AA = "AA"  # 640-719, PD 2-8%
    A = "A"  # 560-639, PD 8-18%
    BBB = "BBB"  # 460-559, PD 18-35%
    CC = "CC"  # 300-459, PD > 35%


class DebtServiceBehavior(str, Enum):
    """Payment behavior classification from transaction analysis."""

    ON_TIME = "ON_TIME"
    LATE_1_30 = "LATE_1_30"
    LATE_31_60 = "LATE_31_60"
    MISSING = "MISSING"


class LoanPurposeCategory(str, Enum):
    """Loan purpose categories extracted by A2 semantic analysis."""

    PRODUCTION = "PRODUCTION"
    CONSUMPTION = "CONSUMPTION"
    INVESTMENT = "INVESTMENT"
    REFINANCING = "REFINANCING"
    UNCLEAR = "UNCLEAR"


class IdentityConsistencyFlag(str, Enum):
    """Cross-document identity verification result."""

    OK = "OK"
    MISMATCH = "MISMATCH"
    MISSING = "MISSING"


# ─── State Schema ─────────────────────────────────────────────────────────────


class CreditState(TypedDict, total=False):
    """
    LangGraph shared state for the CREDICOUNCIL pipeline.

    This TypedDict flows through all 4 agents. Each agent reads its required
    inputs and writes its outputs into the corresponding sections.
    """

    # ── Core identifiers ──
    application_id: str  # SHA-256(applicant_id + timestamp)
    customer_type: Literal["INDIVIDUAL", "SME"]

    # ── A1 outputs — Data Ingestion ──
    raw_ocr_text: dict[str, str]  # {doc_type: extracted_text}
    structured_feats: dict[str, Any]  # {feature_name: value}
    confidence_map: dict[str, float]  # {feature_name: confidence_0_to_1}
    missing_fields: list[str]  # critical/important fields below threshold

    # ── A2 outputs — LLM Feature Engineer ──
    llm_feats: dict[str, Any]  # semantic features + imputed values
    imputation_log: list[dict]  # [{field, method, confidence, source}]
    warnings: list[str]  # human-readable warning messages
    overall_confidence: float  # weighted mean across all fields

    # ── A3 outputs — ML Scoring Engine ──
    credit_score: int  # 300-850
    pd_pct: float  # probability of default %
    risk_band: str  # AAA/AA/A/BBB/CC
    shap_values: dict  # full SHAP JSON (see SHAP output schema)

    # ── A4 outputs — Report Generator ──
    five_c_scores: dict[str, float]  # {character, capacity, capital, conditions, collateral}
    narrative: dict[str, str]  # LLM text per 5C dimension
    consistency_check: dict  # narrative vs SHAP validation result
    final_report: dict  # complete structured report (6 sections)

    # ── Routing & audit ──
    routing: str  # AUTO_APPROVE|REVIEW|REJECT|ESCALATE|HALT
    audit_trail: list[dict]  # immutable append-only log


# ─── Audit Entry Schema ──────────────────────────────────────────────────────


class AuditEntry(TypedDict):
    """Single entry in the immutable audit trail."""

    timestamp: str  # ISO-8601 UTC
    agent: str  # A1/A2/A3/A4/GATE/ROUTER
    action: str  # Description of action taken
    input_summary: dict[str, Any]  # Summarized input (not full data)
    output_summary: dict[str, Any]  # Summarized output
    model_version: str | None  # For ML/LLM model identification
    confidence: float | None  # Overall confidence for this step


# ─── SHAP Output Schema ──────────────────────────────────────────────────────


class ShapFactor(TypedDict):
    """Individual SHAP feature factor."""

    feature: str  # feature name
    shap: float  # SHAP value (positive = reduces default risk)
    value: Any  # actual feature value
    label_vi: str  # Vietnamese human-readable label


class FiveCAllocation(TypedDict):
    """SHAP contribution allocation to a 5C dimension."""

    shap_sum: float  # Sum of SHAP values for features in this dimension
    pct: int  # Percentage contribution (0-100)


# Backward compat alias
FourCAllocation = FiveCAllocation


class ShapOutput(TypedDict):
    """Complete SHAP output from A3 — bridge to A4."""

    credit_score: int
    pd_pct: float
    risk_band: str
    model_version: str
    inference_timestamp: str
    top_positive_factors: list[ShapFactor]
    top_negative_factors: list[ShapFactor]
    five_c_shap_allocation: dict[str, FiveCAllocation]
    all_features_shap: dict[str, float]


# ─── Report Schema ────────────────────────────────────────────────────────────


class FiveCAssessment(TypedDict):
    """Assessment for a single 5C dimension.

    5C: Character(30), Capacity(40), Capital(20), Conditions(10), Collateral(20) = 120 total.
    """

    score: float  # Max varies by dimension
    status: str  # DAT | XEM_XET | KHONG_DAT
    shap_pct: int  # SHAP contribution percentage
    indicators_met: list[str]  # Positive indicators
    indicators_review: list[str]  # Indicators needing review + action
    narrative: str  # 100-150 word Vietnamese text, SHAP-grounded


# Backward compat alias
FourCAssessment = FiveCAssessment


class CreditReport(TypedDict, total=False):
    """Complete credit report output from A4 (6 sections)."""

    # Section I — Thông tin khách hàng
    customer_info: dict[str, str]

    # Section II — Tóm tắt đánh giá (Executive Summary)
    credit_score: int
    risk_band: str
    recommendation: str  # APPROVE|CONDITIONAL|REVIEW|REJECT
    pd_pct: float
    five_c_total: int
    five_c_shap_allocation: dict[str, FiveCAllocation]

    # Section III — 5C Scorecard
    character_assessment: FiveCAssessment
    capacity_assessment: FiveCAssessment
    capital_assessment: FiveCAssessment
    conditions_assessment: FiveCAssessment
    collateral_assessment: FiveCAssessment

    # Section IV — Tình hình tài chính
    financial_summary: dict[str, Any]

    # Section V — Tài sản bảo đảm
    collateral_detail: dict[str, Any]

    # Section VI — Khuyến nghị & Caveats
    suggested_terms: dict[str, Any]  # {max_amount_vnd, max_term_months}
    caveats: list[str]  # imputation warnings, data quality notes

    # Audit Reference
    application_id: str
    model_version: str
    inference_timestamp: str
    shap_json_hash: str
    rag_chunks_used: list[str]
