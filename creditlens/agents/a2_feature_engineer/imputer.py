"""
CreditLens A2 — Intelligent Imputer (Variant B).

Uses Claude to impute IMPORTANT fields when confidence < 0.70.
Instead of mean/median imputation, uses LLM to reason from available context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from creditlens.config.prompts import A2_IMPUTATION_SYSTEM, A2_IMPUTATION_USER
from creditlens.config.feature_config import FIELD_DEFINITIONS

logger = logging.getLogger(__name__)


class IntelligentImputer:
    """Variant B — LLM-based intelligent imputation.

    Uses available context data to estimate missing IMPORTANT field values.
    Every imputation produces:
        - estimated_value: the imputed value
        - confidence: 0-1 reliability of the estimate
        - reasoning: 1-sentence explanation
        - source: data sources used

    All imputations are logged with imputation_flag=True and appear
    in the final report's "Data Warnings" section.
    """

    def __init__(self, bedrock_client=None, use_mock: bool = True):
        self.bedrock_client = bedrock_client
        self.use_mock = use_mock

    def impute_field(
        self,
        field_name: str,
        context_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Impute a single missing field using LLM reasoning.

        Args:
            field_name: Name of the missing field to impute.
            context_data: Available data from other sources for context.

        Returns:
            Dict with estimated_value, confidence, reasoning, source.
        """
        field_def = FIELD_DEFINITIONS.get(field_name)
        field_description = field_def.description if field_def else field_name

        if self.use_mock:
            return self._mock_impute(field_name, context_data)

        prompt = A2_IMPUTATION_USER.format(
            field_name=field_name,
            field_description=field_description,
            context_data=json.dumps(context_data, default=str, indent=2),
        )

        response_text = self._call_llm(A2_IMPUTATION_SYSTEM, prompt)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse imputation response for {field_name}")
            return {
                "estimated_value": None,
                "confidence": 0.0,
                "reasoning": "Failed to generate imputation",
                "source": "error",
            }

        logger.info(
            f"Imputed {field_name}: value={result.get('estimated_value')}, "
            f"confidence={result.get('confidence')}"
        )
        return result

    def impute_missing_fields(
        self,
        missing_fields: list[str],
        structured_feats: dict[str, Any],
        confidence_map: dict[str, float],
    ) -> tuple[dict[str, Any], list[dict]]:
        """Impute all missing IMPORTANT fields.

        Args:
            missing_fields: List of field names to impute.
            structured_feats: Current structured features for context.
            confidence_map: Current confidence scores.

        Returns:
            Tuple of (imputed_values dict, imputation_log list).
        """
        imputed_values: dict[str, Any] = {}
        imputation_log: list[dict] = []

        for field_name in missing_fields:
            field_def = FIELD_DEFINITIONS.get(field_name)
            if field_def is None:
                continue

            # Only impute IMPORTANT tier fields
            if field_def.tier.value != "IMPORTANT":
                continue

            # Build context from available data
            context = self._build_context(field_name, structured_feats)

            result = self.impute_field(field_name, context)

            if result["confidence"] >= 0.60:
                imputed_values[field_name] = result["estimated_value"]
                log_entry = {
                    "field": field_name,
                    "method": "llm_imputation",
                    "confidence": result["confidence"],
                    "source": result["source"],
                    "reasoning": result["reasoning"],
                    "imputation_flag": True,
                }
            else:
                log_entry = {
                    "field": field_name,
                    "method": "llm_imputation_rejected",
                    "confidence": result["confidence"],
                    "source": result["source"],
                    "reasoning": f"Confidence {result['confidence']:.2f} < 0.60 threshold",
                    "imputation_flag": False,
                }
                logger.warning(
                    f"Imputation for {field_name} rejected: confidence "
                    f"{result['confidence']:.2f} < 0.60"
                )

            imputation_log.append(log_entry)

        logger.info(
            f"Imputation complete: {len(imputed_values)} fields imputed, "
            f"{len(imputation_log)} total attempts"
        )
        return imputed_values, imputation_log

    def _build_context(self, field_name: str, structured_feats: dict[str, Any]) -> dict[str, Any]:
        """Build context dict for LLM imputation based on field type."""
        context_keys = {
            "employment_duration": [
                "employment_type", "salary_pattern_detected",
                "avg_monthly_inflow_vnd", "income_stability_index",
            ],
            "collateral_value": [
                "loan_amount_vnd", "flag_own_car", "flag_own_realty",
            ],
            "income_stability_index": [
                "avg_monthly_inflow_vnd", "salary_pattern_detected",
                "employment_type",
            ],
            "debt_service_behavior": [
                "regular_bill_payment_ratio", "overdraft_count_6m",
                "inflow_outflow_ratio",
            ],
        }

        relevant_keys = context_keys.get(field_name, list(structured_feats.keys())[:10])
        return {k: structured_feats[k] for k in relevant_keys if k in structured_feats}

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude via Bedrock."""
        response = self.bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }),
        )
        return json.loads(response["body"].read())["content"][0]["text"]

    def _mock_impute(self, field_name: str, context_data: dict[str, Any]) -> dict[str, Any]:
        """Mock imputation for development."""
        mock_values = {
            "employment_duration": {
                "estimated_value": 24,
                "confidence": 0.72,
                "reasoning": "Inferred from 6-month salary history and contract date",
                "source": "salary_history_6mo + contract_date",
            },
            "collateral_value": {
                "estimated_value": 450_000_000,
                "confidence": 0.65,
                "reasoning": "Estimated from vehicle type and market rates",
                "source": "document_description + market_data",
            },
            "income_stability_index": {
                "estimated_value": 0.75,
                "confidence": 0.78,
                "reasoning": "Computed from 6-month inflow variance",
                "source": "bank_statement_6mo",
            },
            "debt_service_behavior": {
                "estimated_value": "ON_TIME",
                "confidence": 0.68,
                "reasoning": "No late payments detected in transaction history",
                "source": "transaction_analysis",
            },
        }
        return mock_values.get(field_name, {
            "estimated_value": None,
            "confidence": 0.50,
            "reasoning": "Default estimate — insufficient context",
            "source": "default",
        })
