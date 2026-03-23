"""
CreditLens A2 — Semantic Feature Extractor (Variant A).

# LOCAL_SUB: Uses Gemini API instead of Bedrock Claude.
# See LOCAL_SUBSTITUTIONS.md for migration guide.

Uses LLM to extract semantic features from unstructured
loan application text. Always runs for every application with OCR text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from creditlens.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Expected output keys
REQUIRED_EXTRACTION_KEYS = {
    "loan_purpose_category",
    "repayment_plan_quality",
    "stated_income_consistency",
    "risk_flags",
    "positive_signals",
    "extraction_confidence",
}

LOAN_PURPOSE_MAP = {
    "PRODUCTION": 0, "CONSUMPTION": 1, "INVESTMENT": 2,
    "REFINANCING": 3, "UNCLEAR": 4,
}
REPAYMENT_QUALITY_MAP = {
    "DETAILED": 3, "GENERAL": 2, "VAGUE": 1, "NONE": 0,
}

# ── Prompts ──
SEMANTIC_SYSTEM = """You are a Vietnamese credit analyst AI. Analyze loan application documents
and extract structured features. Respond ONLY with valid JSON, no other text.

Output JSON schema:
{
    "loan_purpose_category": "PRODUCTION|CONSUMPTION|INVESTMENT|REFINANCING|UNCLEAR",
    "repayment_plan_quality": "DETAILED|GENERAL|VAGUE|NONE",
    "stated_income_consistency": true/false (does stated income match evidence?),
    "risk_flags": ["list of risk indicators found"],
    "positive_signals": ["list of positive indicators found"],
    "extraction_confidence": 0.0-1.0
}"""

SEMANTIC_USER = """Analyze this Vietnamese loan application text and extract features:

{ocr_text}

Focus on:
1. What is the loan purpose? Categorize as PRODUCTION/CONSUMPTION/INVESTMENT/REFINANCING/UNCLEAR
2. How detailed is the repayment plan? DETAILED/GENERAL/VAGUE/NONE
3. Is the stated income consistent with employment docs? true/false
4. List any risk flags (e.g., address mismatch, unstable employment, high DTI)
5. List any positive signals (e.g., stable salary, owns property, low debt)

Respond with JSON only."""


class SemanticExtractor:
    """Variant A — Extract semantic features from loan application text.

    Uses LLM to analyze OCR text and extract:
    - loan_purpose_category (categorical)
    - repayment_plan_quality (ordinal)
    - stated_income_consistency (binary)
    - risk_flag_count + risk_flags_list
    - positive_signals
    """

    def __init__(self, use_mock: bool = False):
        self.llm = LLMService(use_mock=use_mock)
        self.use_mock = use_mock

    def extract_loan_features(self, ocr_text: str) -> dict[str, Any]:
        """Extract semantic features from loan application text."""
        if self.use_mock:
            return self._mock_loan_extraction(ocr_text)

        prompt = SEMANTIC_USER.format(ocr_text=ocr_text[:4000])
        result = self.llm.generate_json(
            SEMANTIC_SYSTEM, prompt, REQUIRED_EXTRACTION_KEYS
        )

        # Encode categorical features
        result["loan_purpose_category_encoded"] = LOAN_PURPOSE_MAP.get(
            result.get("loan_purpose_category", "UNCLEAR"), 4
        )
        result["repayment_plan_quality_encoded"] = REPAYMENT_QUALITY_MAP.get(
            result.get("repayment_plan_quality", "NONE"), 0
        )
        result["risk_flag_count"] = len(result.get("risk_flags", []) or [])

        return result

    def _mock_loan_extraction(self, ocr_text: str) -> dict[str, Any]:
        """Mock semantic extraction based on actual text content."""
        # Simple heuristic-based extraction from OCR text
        text_lower = ocr_text.lower()

        # Detect purpose
        if any(w in text_lower for w in ["xe", "oto", "o to", "car"]):
            purpose = "CONSUMPTION"
        elif any(w in text_lower for w in ["san xuat", "kinh doanh", "business"]):
            purpose = "PRODUCTION"
        elif any(w in text_lower for w in ["bat dong san", "nha dat", "dau tu"]):
            purpose = "INVESTMENT"
        else:
            purpose = "CONSUMPTION"

        # Detect positive signals
        positive = []
        if "khong xac dinh thoi han" in text_lower:
            positive.append("Permanent employment contract")
        if any(w in text_lower for w in ["vinhomes", "chung cu", "can ho"]):
            positive.append("Owns apartment/property")
        if any(w in text_lower for w in ["luong", "thu nhap"]):
            positive.append("Verified income document")

        # Detect risk flags
        risks = []
        if "khong co" not in text_lower and "tranh chap" in text_lower:
            risks.append("Potential property dispute")

        return {
            "loan_purpose_category": purpose,
            "loan_purpose_category_encoded": LOAN_PURPOSE_MAP.get(purpose, 4),
            "repayment_plan_quality": "GENERAL",
            "repayment_plan_quality_encoded": 2,
            "stated_income_consistency": True,
            "risk_flags": risks,
            "positive_signals": positive,
            "extraction_confidence": 0.85,
            "risk_flag_count": len(risks),
        }
