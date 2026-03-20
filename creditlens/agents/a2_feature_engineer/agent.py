"""
CreditLens A2 — Main LLM Feature Engineer Agent.

Orchestrates Variant A (semantic extraction) and Variant B (imputation).
This is the LangGraph node for Agent A2.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from creditlens.agents.a2_feature_engineer.semantic_extractor import SemanticExtractor
from creditlens.agents.a2_feature_engineer.imputer import IntelligentImputer
from creditlens.agents.a2_feature_engineer.thin_file_handler import activate_thin_file_path
from creditlens.config.feature_config import (
    FIELD_DEFINITIONS,
    CONFIDENCE_THRESHOLDS,
    TIER_WEIGHTS,
    FeatureTier,
    OVERALL_CONFIDENCE_AUTO_PROCEED,
    OVERALL_CONFIDENCE_PROCEED_WITH_WARNINGS,
)
from creditlens.state.credit_state import CreditState

logger = logging.getLogger(__name__)


class FeatureEngineerAgent:
    """Agent A2 — LLM Feature Engineer.

    Two operating modes:
        Variant A: Semantic extraction — ALWAYS runs on OCR text
        Variant B: Intelligent imputation — ONLY for missing IMPORTANT fields

    Also handles thin-file path activation when needed.
    """

    def __init__(self, use_mock: bool = True, bedrock_client=None):
        self.semantic_extractor = SemanticExtractor(
            bedrock_client=bedrock_client,
            use_mock=use_mock,
        )
        self.imputer = IntelligentImputer(
            bedrock_client=bedrock_client,
            use_mock=use_mock,
        )

    def process(self, state: CreditState) -> dict[str, Any]:
        """Run A2 feature engineering — LangGraph node function.

        Args:
            state: Current pipeline state with A1 outputs.

        Returns:
            State update dict with A2 outputs.
        """
        logger.info(f"A2 Feature Engineer — App {state.get('application_id', 'unknown')}")

        llm_feats: dict[str, Any] = {}
        imputation_log: list[dict] = []
        warnings: list[str] = list(state.get("warnings", []))
        structured_feats = state.get("structured_feats", {})
        confidence_map = state.get("confidence_map", {})

        # ── Variant A: Semantic extraction (always runs) ──
        raw_ocr = state.get("raw_ocr_text", {})
        if raw_ocr:
            ocr_combined = " ".join(raw_ocr.values())
            semantic_feats = self.semantic_extractor.extract_loan_features(ocr_combined)
            llm_feats.update(semantic_feats)
            logger.info(f"Variant A: extracted {len(semantic_feats)} semantic features")

        # ── Variant B: Imputation (only for missing IMPORTANT fields) ──
        missing_fields = state.get("missing_fields", [])
        if missing_fields:
            imputed, imp_log = self.imputer.impute_missing_fields(
                missing_fields=missing_fields,
                structured_feats=structured_feats,
                confidence_map=confidence_map,
            )
            llm_feats.update(imputed)
            imputation_log.extend(imp_log)

            # Add imputation warnings
            for entry in imp_log:
                if entry.get("imputation_flag"):
                    warnings.append(
                        f"Trường '{entry['field']}' được ước tính "
                        f"(confidence: {entry['confidence']:.0%}). "
                        f"Nguồn: {entry['source']}"
                    )

        # ── Thin-file path ──
        if structured_feats.get("thin_file_flag"):
            thin_file_result = activate_thin_file_path(structured_feats, confidence_map)
            llm_feats["thin_file_info"] = thin_file_result
            warnings.extend(thin_file_result.get("warnings", []))

        # ── Compute overall confidence ──
        overall_confidence = self._compute_overall_confidence(confidence_map)

        # ── Imputation metadata ──
        n_imputed = sum(1 for e in imputation_log if e.get("imputation_flag"))
        if n_imputed > 0:
            llm_feats["income_imputed_flag"] = 1
            llm_feats["imputation_confidence"] = sum(
                e["confidence"] for e in imputation_log if e.get("imputation_flag")
            ) / n_imputed
        else:
            llm_feats["income_imputed_flag"] = 0
            llm_feats["imputation_confidence"] = 1.0

        # ── Audit entry ──
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A2",
            "action": "llm_feature_engineering",
            "input_summary": {
                "has_ocr_text": bool(raw_ocr),
                "n_missing_fields": len(missing_fields),
                "thin_file": structured_feats.get("thin_file_flag", False),
            },
            "output_summary": {
                "n_semantic_features": len(semantic_feats) if raw_ocr else 0,
                "n_imputed_fields": n_imputed,
                "overall_confidence": overall_confidence,
            },
            "model_version": "claude-3.5-sonnet",
            "confidence": overall_confidence,
        }

        return {
            "llm_feats": llm_feats,
            "imputation_log": imputation_log,
            "warnings": warnings,
            "overall_confidence": overall_confidence,
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
        }

    def _compute_overall_confidence(self, confidence_map: dict[str, float]) -> float:
        """Compute weighted overall confidence score.

        Formula: Σ(weight_i × confidence_i) / Σ(weight_i)
        Weights: CRITICAL=3, IMPORTANT=2, OPTIONAL=1
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for field_name, field_def in FIELD_DEFINITIONS.items():
            conf = confidence_map.get(field_name, 0.0)
            weight = TIER_WEIGHTS[field_def.tier]
            weighted_sum += weight * conf
            weight_total += weight

        if weight_total == 0:
            return 0.0

        return round(weighted_sum / weight_total, 3)
