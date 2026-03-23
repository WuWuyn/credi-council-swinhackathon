"""
CreditLens A2 — Intelligent Imputer (Variant B).

# LOCAL_SUB: Uses Gemini API instead of Bedrock Claude.

Uses LLM to impute IMPORTANT fields when confidence < 0.70.
Instead of mean/median imputation, uses LLM to reason from available context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from creditlens.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# ── Prompts ──
IMPUTATION_SYSTEM = """You are a Vietnamese credit analyst AI.
Given context data about a loan applicant, estimate the value of a missing field.

Respond ONLY with valid JSON:
{
    "estimated_value": <the estimated value>,
    "confidence": 0.0-1.0,
    "reasoning": "1-sentence explanation",
    "source": "data sources used for estimation"
}"""

IMPUTATION_USER = """Estimate the missing field for a Vietnamese loan application:

Field to estimate: {field_name}
Field description: {field_description}

Available context data:
{context_data}

Provide your best estimate with confidence level. Respond with JSON only."""


class IntelligentImputer:
    """Variant B — LLM-based intelligent imputation.

    Uses available context to estimate missing field values.
    """

    def __init__(self):
        self.llm = LLMService()

    def impute_field(
        self,
        field_name: str,
        field_description: str,
        context_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Impute a single missing field."""


        prompt = IMPUTATION_USER.format(
            field_name=field_name,
            field_description=field_description,
            context_data=json.dumps(context_data, default=str, indent=2),
        )

        return self.llm.generate_json(
            IMPUTATION_SYSTEM, prompt,
            {"estimated_value", "confidence", "reasoning", "source"}
        )

    def impute_missing_fields(
        self,
        missing_fields: list[str],
        application_row: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict]]:
        """Impute all missing fields using LLM reasoning.

        Args:
            missing_fields: List of field names with None values.
            application_row: Current application row for context.

        Returns:
            Tuple of (imputed_values, imputation_log)
        """
        imputed_values: dict[str, Any] = {}
        imputation_log: list[dict] = []

        # Build compact context from non-null fields
        context = {k: v for k, v in application_row.items()
                   if v is not None and k not in missing_fields}

        for field_name in missing_fields:
            result = self.impute_field(
                field_name=field_name,
                field_description=f"Dataset column: {field_name}",
                context_data=context,
            )

            if result.get("confidence", 0) >= 0.60:
                imputed_values[field_name] = result["estimated_value"]
                log_entry = {
                    "field": field_name,
                    "method": "llm_imputation",
                    "confidence": result["confidence"],
                    "source": result.get("source", "llm"),
                    "reasoning": result.get("reasoning", ""),
                    "imputation_flag": True,
                }
            else:
                log_entry = {
                    "field": field_name,
                    "method": "llm_imputation_rejected",
                    "confidence": result.get("confidence", 0),
                    "source": result.get("source", ""),
                    "reasoning": f"Confidence {result.get('confidence', 0):.2f} < 0.60",
                    "imputation_flag": False,
                }

            imputation_log.append(log_entry)

        logger.info(f"Imputation: {len(imputed_values)}/{len(missing_fields)} fields imputed")
        return imputed_values, imputation_log


