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

    def __init__(self):
        self.semantic_extractor = SemanticExtractor()
        self.imputer = IntelligentImputer()

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

        # ── Step 1: Semantic extraction ──
        # Source: OCR text (from PDFs) or structured summary (from application_row)
        raw_texts = a1_output.get("raw_texts", {})
        if raw_texts:
            # USE_OCR=true: combine raw OCR text from all PDFs
            input_text = " ".join(str(v) for v in raw_texts.values())
            logger.info("  Step 1: Semantic extraction from OCR text")
        else:
            # USE_OCR=false: build structured summary from application_row
            # This is actually MORE accurate than noisy OCR text
            input_text = self._build_text_from_application_row(application_row)
            logger.info("  Step 1: Semantic extraction from application_row summary")

        if input_text.strip():
            semantic = self.semantic_extractor.extract_loan_features(input_text)
            llm_feats.update(semantic)
            logger.info(f"  Step 1: {len(semantic)} semantic features extracted")
            logger.info(f"    Purpose: {semantic.get('loan_purpose_category')}")
            logger.info(f"    Positive: {semantic.get('positive_signals')}")
            logger.info(f"    Risks: {semantic.get('risk_flags')}")

        # ── Fallback: derive loan_purpose from application_row if UNCLEAR ──
        if not llm_feats.get("loan_purpose_category") or \
                str(llm_feats.get("loan_purpose_category", "")).upper() in ("UNCLEAR", "NONE", "NULL", ""):
            contract_type = application_row.get("NAME_CONTRACT_TYPE", "")
            goods_price = application_row.get("AMT_GOODS_PRICE")
            income_type = application_row.get("NAME_INCOME_TYPE", "")

            # Map Home Credit contract types to 5 loan purpose categories
            CONTRACT_PURPOSE_MAP = {
                "Cash loans":      "CONSUMPTION",      # personal cash loan
                "Revolving loans": "CONSUMPTION",      # credit-card style
            }
            INCOME_PURPOSE_MAP = {
                "Businessman":    "PRODUCTION",        # business owner → production
                "Commercial associate": "INVESTMENT",  # commercial → investment
            }

            purpose = CONTRACT_PURPOSE_MAP.get(str(contract_type), None)
            if not purpose:
                purpose = INCOME_PURPOSE_MAP.get(str(income_type), None)
            if not purpose:
                # If goods_price is close to credit amount, likely consumer goods
                credit = application_row.get("AMT_CREDIT")
                if goods_price and credit and abs(goods_price - credit) / max(credit, 1) < 0.05:
                    purpose = "CONSUMPTION"
                else:
                    purpose = "UNCLEAR"

            llm_feats["loan_purpose_category"] = purpose
            logger.info(f"  Loan purpose fallback: {contract_type} → {purpose}")


        # ── Step 2: Log missing fields (no LLM imputation) ──
        missing_fields = [k for k, v in application_row.items() if v is None]
        if missing_fields:
            logger.info(f"  Step 2: {len(missing_fields)} fields missing (skipped LLM imputation)")

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

    @staticmethod
    def _build_text_from_application_row(app: dict) -> str:
        """Convert application_row dict into structured text for LLM semantic extraction.

        This produces a text summary that the SemanticExtractor can analyze,
        enabling semantic features even when OCR text is not available (USE_OCR=false).
        """
        lines = []

        # Personal info
        gender = app.get("CODE_GENDER", "N/A")
        education = app.get("NAME_EDUCATION_TYPE", "N/A")
        family = app.get("NAME_FAMILY_STATUS", "N/A")
        housing = app.get("NAME_HOUSING_TYPE", "N/A")
        income_type = app.get("NAME_INCOME_TYPE", "N/A")
        occupation = app.get("OCCUPATION_TYPE", "N/A")
        org_type = app.get("ORGANIZATION_TYPE", "N/A")

        lines.append(f"Applicant: {gender}, {education}, {family}")
        lines.append(f"Housing: {housing}")
        lines.append(f"Income type: {income_type}, Occupation: {occupation}")
        lines.append(f"Organization: {org_type}")

        # Age and employment
        days_birth = app.get("DAYS_BIRTH")
        if days_birth:
            age_years = abs(int(days_birth)) // 365
            lines.append(f"Age: {age_years} years")

        days_employed = app.get("DAYS_EMPLOYED")
        if days_employed and int(days_employed) < 0:
            emp_years = abs(int(days_employed)) / 365
            lines.append(f"Employment duration: {emp_years:.1f} years")

        # Income and loan
        income = app.get("AMT_INCOME_TOTAL")
        credit = app.get("AMT_CREDIT")
        annuity = app.get("AMT_ANNUITY")
        goods = app.get("AMT_GOODS_PRICE")
        contract = app.get("NAME_CONTRACT_TYPE", "N/A")

        if income:
            lines.append(f"Annual income: {income:,.0f}")
        if credit:
            lines.append(f"Loan amount: {credit:,.0f}")
        if annuity:
            lines.append(f"Annuity: {annuity:,.0f}")
        if goods:
            lines.append(f"Goods price: {goods:,.0f}")
        lines.append(f"Contract type: {contract}")

        # DTI ratio
        if income and annuity:
            dti = (annuity / 12) / (income / 12) * 100
            lines.append(f"DTI ratio: {dti:.1f}%")

        # Assets
        has_car = app.get("FLAG_OWN_CAR", "N")
        has_realty = app.get("FLAG_OWN_REALTY", "N")
        lines.append(f"Own car: {has_car}, Own property: {has_realty}")

        car_age = app.get("OWN_CAR_AGE")
        if car_age is not None:
            lines.append(f"Car age: {car_age} years")

        # External scores (CIC)
        for i in [1, 2, 3]:
            ext = app.get(f"EXT_SOURCE_{i}")
            if ext is not None:
                lines.append(f"External credit score {i}: {ext:.4f}")

        # Children and family
        children = app.get("CNT_CHILDREN", 0)
        family_members = app.get("CNT_FAM_MEMBERS")
        if children:
            lines.append(f"Children: {children}")
        if family_members:
            lines.append(f"Family members: {int(family_members)}")

        # Contact info
        flags = []
        if app.get("FLAG_MOBIL") == 1: flags.append("mobile")
        if app.get("FLAG_EMP_PHONE") == 1: flags.append("employer phone")
        if app.get("FLAG_WORK_PHONE") == 1: flags.append("work phone")
        if app.get("FLAG_PHONE") == 1: flags.append("landline")
        if app.get("FLAG_EMAIL") == 1: flags.append("email")
        if flags:
            lines.append(f"Contact methods: {', '.join(flags)}")

        # Region
        region_rating = app.get("REGION_RATING_CLIENT")
        if region_rating:
            lines.append(f"Region rating: {region_rating}")

        return "\n".join(lines)
