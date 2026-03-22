"""
CreditLens A2 — LLM Feature Engineer Agent (Local Version).

# LOCAL_SUB: Uses Gemini API instead of Bedrock Claude.

Orchestrates the feature engineering pipeline:
1. Semantic extraction from OCR text (LLM-based)
2. Intelligent imputation of missing fields (LLM-based)
3. Feature engineering: 218 raw columns → 753 ML features
   (reusing existing training/feature_engineering.py logic)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from creditlens.agents.a2_feature_engineer.semantic_extractor import SemanticExtractor
from creditlens.agents.a2_feature_engineer.imputer import IntelligentImputer

logger = logging.getLogger(__name__)

# Add project root to path for training module import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FeatureEngineerAgent:
    """Agent A2 — LLM Feature Engineer.

    Takes A1 output (application_row + DataFrames) and produces:
    1. LLM semantic features (loan purpose, risk flags, etc.)
    2. Imputed missing fields
    3. Full 753-feature vector for ML model (via feature_engineering.py)

    This is the bridge between raw data and ML scoring.
    """

    def __init__(self, use_mock: bool = True):
        self.semantic_extractor = SemanticExtractor(use_mock=use_mock)
        self.imputer = IntelligentImputer(use_mock=use_mock)

    def process(self, a1_output: dict[str, Any]) -> dict[str, Any]:
        """Run A2 feature engineering pipeline.

        Args:
            a1_output: Output from A1 IngestionAgent.ingest()

        Returns:
            Dict with:
                - feature_vector: pd.Series/dict with 753 features
                - llm_feats: semantic features from LLM
                - imputation_log: list of imputed fields
                - warnings: list of warning messages
                - audit_trail: audit entries
        """
        logger.info("="*60)
        logger.info("  A2 Feature Engineer — Processing")
        logger.info("="*60)

        application_row = a1_output["application_row"]
        warnings: list[str] = []
        imputation_log: list[dict] = []
        llm_feats: dict[str, Any] = {}

        # ── Step 1: Semantic extraction from OCR text ──
        raw_texts = a1_output.get("raw_texts", {})
        if raw_texts:
            ocr_combined = " ".join(str(v) for v in raw_texts.values())
            semantic = self.semantic_extractor.extract_loan_features(ocr_combined)
            llm_feats.update(semantic)
            logger.info(f"  Step 1: {len(semantic)} semantic features extracted")
            logger.info(f"    Purpose: {semantic.get('loan_purpose_category')}")
            logger.info(f"    Positive: {semantic.get('positive_signals')}")
            logger.info(f"    Risks: {semantic.get('risk_flags')}")

        # ── Step 2: Imputation of missing fields ──
        missing_fields = [k for k, v in application_row.items() if v is None]
        if missing_fields:
            imputed, imp_log = self.imputer.impute_missing_fields(
                missing_fields, application_row
            )
            application_row.update(imputed)
            imputation_log.extend(imp_log)

            n_imputed = sum(1 for e in imp_log if e.get("imputation_flag"))
            logger.info(f"  Step 2: {n_imputed}/{len(missing_fields)} fields imputed")

            for entry in imp_log:
                if entry.get("imputation_flag"):
                    warnings.append(
                        f"Field '{entry['field']}' was imputed "
                        f"(confidence: {entry['confidence']:.0%})"
                    )

        # ── Step 3: Feature engineering (218 raw → 753 features) ──
        logger.info("  Step 3: Running feature engineering pipeline...")
        feature_vector = self._run_feature_engineering(a1_output)

        if feature_vector is not None:
            logger.info(f"  Step 3: {len(feature_vector)} features generated")
        else:
            logger.warning("  Step 3: Feature engineering failed — using raw features only")
            warnings.append("Feature engineering failed — model will use raw features")

        # ── Imputation metadata ──
        n_imputed = sum(1 for e in imputation_log if e.get("imputation_flag"))
        llm_feats["income_imputed_flag"] = 1 if n_imputed > 0 else 0
        llm_feats["imputation_confidence"] = (
            sum(e["confidence"] for e in imputation_log if e.get("imputation_flag")) / n_imputed
            if n_imputed > 0 else 1.0
        )

        # ── Thin file flag propagation ──
        thin_file = a1_output.get("thin_file_flag", False)
        if thin_file:
            llm_feats["thin_file_flag"] = True
            warnings.append("Thin-file customer — limited credit history available")

        # ── Audit ──
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A2",
            "action": "feature_engineering",
            "input_summary": {
                "has_ocr": bool(raw_texts),
                "n_missing_before": len(missing_fields) if missing_fields else 0,
                "thin_file": thin_file,
            },
            "output_summary": {
                "n_semantic_features": len(llm_feats),
                "n_imputed": n_imputed,
                "n_ml_features": len(feature_vector) if feature_vector is not None else 0,
            },
            "model_version": "gemini-2.5-flash-lite",
        }

        return {
            "feature_vector": feature_vector,
            "application_row": application_row,
            "llm_feats": llm_feats,
            "imputation_log": imputation_log,
            "warnings": warnings,
            "audit_trail": a1_output.get("audit_trail", []) + [audit_entry],
        }

    def _run_feature_engineering(self, a1_output: dict[str, Any]) -> pd.Series | None:
        """Run full feature engineering pipeline on A1 output.

        Uses SingleCustomerFE which applies the same feature engineering
        as training/feature_engineering.py but for a single customer.
        """
        try:
            from creditlens.agents.a2_feature_engineer.single_customer_fe import SingleCustomerFE

            fe = SingleCustomerFE("models/fe_stats.pkl")
            return fe.build_features(a1_output)

        except FileNotFoundError as e:
            logger.warning(f"  FE stats not found: {e}")
            logger.warning("  Run: python training/precompute_fe_stats.py --data-dir home-credit-default-risk/")
            return None

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return None
