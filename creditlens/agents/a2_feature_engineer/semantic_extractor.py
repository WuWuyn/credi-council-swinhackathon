"""
CreditLens A2 — Semantic Feature Extractor (Variant A).

Uses Claude (via Bedrock) to extract semantic features from unstructured
loan application text. Always runs for every application with OCR text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from creditlens.config.prompts import (
    A2_SEMANTIC_EXTRACTION_SYSTEM,
    A2_SEMANTIC_EXTRACTION_USER,
    A2_TRANSACTION_PURPOSE_SYSTEM,
    A2_TRANSACTION_PURPOSE_USER,
    A2_BUSINESS_LEGITIMACY_SYSTEM,
    A2_BUSINESS_LEGITIMACY_USER,
)

logger = logging.getLogger(__name__)

# Expected output keys for validation
REQUIRED_EXTRACTION_KEYS = {
    "loan_purpose_category",
    "repayment_plan_quality",
    "stated_income_consistency",
    "risk_flags",
    "positive_signals",
    "extraction_confidence",
}

# Mapping from string categories to numeric encoding
LOAN_PURPOSE_MAP = {
    "PRODUCTION": 0,
    "CONSUMPTION": 1,
    "INVESTMENT": 2,
    "REFINANCING": 3,
    "UNCLEAR": 4,
}

REPAYMENT_QUALITY_MAP = {
    "DETAILED": 3,
    "GENERAL": 2,
    "VAGUE": 1,
    "NONE": 0,
}


class SemanticExtractor:
    """Variant A — Extract semantic features from loan application text.

    Input:
        - OCR text from loan application documents
        - Transaction descriptions
        - Business registration info (SME)

    Output:
        - loan_purpose_category (categorical → one-hot)
        - repayment_plan_quality (ordinal → integer)
        - stated_income_consistency (binary)
        - transaction_purpose_distribution (dict of floats)
        - business_legitimacy_score (float, SME only)
        - risk_flag_count + risk_flags_list
    """

    def __init__(self, bedrock_client=None, use_mock: bool = True):
        self.bedrock_client = bedrock_client
        self.use_mock = use_mock

    def extract_loan_features(self, ocr_text: str) -> dict[str, Any]:
        """Extract semantic features from loan application text.

        Args:
            ocr_text: Full OCR text from loan application documents.

        Returns:
            Dict with extracted features and metadata.
        """
        if self.use_mock:
            return self._mock_loan_extraction(ocr_text)

        prompt = A2_SEMANTIC_EXTRACTION_USER.format(ocr_text=ocr_text[:3000])

        response = self._call_llm(A2_SEMANTIC_EXTRACTION_SYSTEM, prompt)
        result = self._parse_and_validate(response, REQUIRED_EXTRACTION_KEYS)

        # Encode categorical features to numeric
        result["loan_purpose_category_encoded"] = LOAN_PURPOSE_MAP.get(
            result.get("loan_purpose_category", "UNCLEAR"), 4
        )
        result["repayment_plan_quality_encoded"] = REPAYMENT_QUALITY_MAP.get(
            result.get("repayment_plan_quality", "NONE"), 0
        )
        result["risk_flag_count"] = len(result.get("risk_flags", []))

        return result

    def extract_transaction_purposes(self, transactions_text: str) -> dict[str, Any]:
        """Classify transaction purpose distribution.

        Args:
            transactions_text: Last 50 transaction descriptions.

        Returns:
            Dict with transaction_purpose_distribution.
        """
        if self.use_mock:
            return self._mock_transaction_extraction()

        prompt = A2_TRANSACTION_PURPOSE_USER.format(transactions_text=transactions_text)
        response = self._call_llm(A2_TRANSACTION_PURPOSE_SYSTEM, prompt)
        return self._parse_and_validate(response, {"transaction_purpose_distribution"})

    def extract_business_legitimacy(
        self,
        gpkd_text: str,
        web_info: str = "",
        industry: str = "",
        reg_age_months: int = 0,
    ) -> dict[str, Any]:
        """Evaluate SME business legitimacy.

        Args:
            gpkd_text: Business registration document text.
            web_info: Web presence information.
            industry: Industry classification.
            reg_age_months: Business registration age in months.

        Returns:
            Dict with business_legitimacy_score.
        """
        if self.use_mock:
            return self._mock_business_extraction()

        prompt = A2_BUSINESS_LEGITIMACY_USER.format(
            gpkd_text=gpkd_text[:2000],
            web_info=web_info[:500],
            industry=industry,
            reg_age_months=reg_age_months,
        )
        response = self._call_llm(A2_BUSINESS_LEGITIMACY_SYSTEM, prompt)
        return self._parse_and_validate(response, {"business_legitimacy_score"})

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude via Bedrock."""
        if self.bedrock_client is None:
            raise RuntimeError("Bedrock client not initialized")

        response = self.bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }),
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def _parse_and_validate(self, response_text: str, required_keys: set) -> dict[str, Any]:
        """Parse JSON response and validate schema."""
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.error(f"Failed to parse LLM response as JSON: {response_text[:200]}")
                return {key: None for key in required_keys}

        missing_keys = required_keys - set(result.keys())
        if missing_keys:
            logger.warning(f"Missing keys in LLM response: {missing_keys}")
            for key in missing_keys:
                result[key] = None

        return result

    def _mock_loan_extraction(self, ocr_text: str) -> dict[str, Any]:
        """Mock semantic extraction for development."""
        return {
            "loan_purpose_category": "CONSUMPTION",
            "loan_purpose_category_encoded": 1,
            "repayment_plan_quality": "GENERAL",
            "repayment_plan_quality_encoded": 2,
            "stated_income_consistency": True,
            "risk_flags": ["Overdraft 2 times in 6 months"],
            "positive_signals": ["Stable salary pattern", "Regular bill payments"],
            "extraction_confidence": 0.85,
            "risk_flag_count": 1,
        }

    def _mock_transaction_extraction(self) -> dict[str, Any]:
        return {
            "transaction_purpose_distribution": {
                "salary": 0.35,
                "rent": 0.10,
                "business": 0.05,
                "retail": 0.30,
                "transfer": 0.20,
            },
            "classification_confidence": 0.82,
        }

    def _mock_business_extraction(self) -> dict[str, Any]:
        return {
            "business_legitimacy_score": 0.75,
            "factors": {
                "registration_valid": True,
                "reg_age_score": 0.8,
                "web_presence_score": 0.5,
                "industry_risk_level": "MEDIUM",
                "description_quality_score": 0.7,
            },
            "assessment_confidence": 0.78,
        }
