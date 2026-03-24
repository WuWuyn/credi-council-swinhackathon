"""
CREDICOUNCIL A2 — Semantic Feature Extractor.

Uses LLM + Pydantic structured output to extract semantic features
from loan application text. These features are used by the ML model.

Upgraded: Uses LLMService.generate_structured() with Pydantic schema
for type-safe, validated output.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from credicouncil.services.llm_service import LLMService

logger = logging.getLogger(__name__)


# ── Pydantic Schema ──────────────────────────────────────────────────────────────

class SemanticFeatures(BaseModel):
    """Structured output for semantic loan analysis."""
    loan_purpose_category: Optional[str] = Field(
        None,
        description="Loan purpose: PRODUCTION | CONSUMPTION | INVESTMENT | REFINANCING | UNCLEAR",
    )
    repayment_plan_quality: Optional[str] = Field(
        None,
        description="Repayment plan detail: DETAILED | GENERAL | VAGUE | NONE",
    )
    stated_income_consistency: Optional[bool] = Field(
        None,
        description="True if stated income is consistent with employment docs",
    )
    risk_flags: Optional[list[str]] = Field(
        default_factory=list,
        description="List of risk indicators found (e.g., address mismatch, unstable employment, high DTI)",
    )
    positive_signals: Optional[list[str]] = Field(
        default_factory=list,
        description="List of positive indicators (e.g., stable salary, property owner, low debt)",
    )
    extraction_confidence: Optional[float] = Field(
        None,
        description="Confidence score 0.0-1.0 of this extraction",
    )


# ── Mappings ─────────────────────────────────────────────────────────────────────

LOAN_PURPOSE_MAP = {
    "PRODUCTION": 0, "CONSUMPTION": 1, "INVESTMENT": 2,
    "REFINANCING": 3, "UNCLEAR": 4,
}
REPAYMENT_QUALITY_MAP = {
    "DETAILED": 3, "GENERAL": 2, "VAGUE": 1, "NONE": 0,
}

# ── Prompts ──────────────────────────────────────────────────────────────────────

SEMANTIC_SYSTEM = """You are a Vietnamese credit analyst AI. Analyze loan application documents
and extract structured features.

Rules:
1. loan_purpose_category: Categorize as PRODUCTION / CONSUMPTION / INVESTMENT / REFINANCING / UNCLEAR
2. repayment_plan_quality: DETAILED (specific amounts+timeline) / GENERAL (plan but vague) / VAGUE (mentioned briefly) / NONE (not mentioned)
3. stated_income_consistency: true if income matches employment evidence, false if contradictory
4. risk_flags: List specific risk indicators found (address mismatch, unstable employment, high DTI, etc.)
5. positive_signals: List specific positive indicators (stable salary, property ownership, low debt, etc.)
6. extraction_confidence: Your confidence in this analysis (0.0-1.0)

Respond with JSON only, no explanation."""

SEMANTIC_USER = """Analyze this Vietnamese loan application and extract semantic features:

{ocr_text}

Return JSON per the schema."""


class SemanticExtractor:
    """Extract semantic features from loan application text using LLM + Pydantic.

    Uses LLMService.generate_structured() for type-safe JSON output.

    Output features:
    - loan_purpose_category (categorical → encoded)
    - repayment_plan_quality (ordinal → encoded)
    - stated_income_consistency (binary)
    - risk_flag_count + risk_flags_list
    - positive_signals
    """

    def __init__(self):
        self.llm = LLMService()

    def extract_loan_features(self, ocr_text: str) -> dict[str, Any]:
        """Extract semantic features from loan application text."""
        prompt = SEMANTIC_USER.format(ocr_text=ocr_text[:6000])

        result = self.llm.generate_structured(
            system_prompt=SEMANTIC_SYSTEM,
            user_prompt=prompt,
            schema_class=SemanticFeatures,
            max_tokens=2048,
            temperature=0.2,
        )

        # Encode categorical features
        result["loan_purpose_category_encoded"] = LOAN_PURPOSE_MAP.get(
            result.get("loan_purpose_category", "UNCLEAR"), 4
        )
        result["repayment_plan_quality_encoded"] = REPAYMENT_QUALITY_MAP.get(
            result.get("repayment_plan_quality", "NONE"), 0
        )
        result["risk_flag_count"] = len(result.get("risk_flags") or [])

        return result
