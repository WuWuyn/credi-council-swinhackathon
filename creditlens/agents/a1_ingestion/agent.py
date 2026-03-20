"""
CreditLens A1 — Main Data Ingestion Agent.

Orchestrates the 3-channel data ingestion pipeline:
    1. PDF/Scan → Textract → structured identity/employment fields
    2. CIC API → credit bureau data
    3. Bank Statement CSV → 8 alternative data features

Supports both monolithic ingestion (for testing) and split methods
(for 9-node LangGraph graph with parallel CIC + transaction branches).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from creditlens.agents.a1_ingestion.textract_service import TextractService
from creditlens.agents.a1_ingestion.cic_service import CICService
from creditlens.agents.a1_ingestion.bank_statement_parser import parse_bank_statement
from creditlens.state.credit_state import CreditState

logger = logging.getLogger(__name__)


class IngestionAgent:
    """Agent A1 — Data Ingestion & Feature Pipeline.

    Receives raw application data from 3 channels and produces
    structured_feats + confidence_map + missing_fields.

    Supports 3 split methods for 9-node LangGraph graph:
        - ingest_documents(): Channel 1 — Textract OCR
        - check_cic(): Channel 2 — CIC API
        - analyze_transactions(): Channel 3 — Bank Statement
    """

    def __init__(self, use_mock: bool = True):
        self.textract = TextractService(use_mock=use_mock)
        self.cic = CICService(use_mock=use_mock)

    # ── Split methods for 9-node graph ────────────────────────────────────────

    def ingest_documents(
        self,
        applicant_id: str,
        customer_type: str = "INDIVIDUAL",
        documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Node 1: ingest_documents — Textract OCR processing.

        Processes PDF/Scan documents via AWS Textract:
        - CCCD/CMND → identity fields
        - Hợp đồng lao động → employment fields
        - GPKD → business fields (SME)
        - TSBĐ → collateral fields

        Includes cross-document identity validation
        (identity_consistency_flag: OK / MISMATCH / MISSING).

        Args:
            applicant_id: Unique applicant identifier.
            customer_type: INDIVIDUAL or SME.
            documents: List of {type, bytes} for PDF documents.

        Returns:
            State update dict with OCR results.
        """
        application_id = hashlib.sha256(
            f"{applicant_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        logger.info(f"A1 ingest_documents — App {application_id}, type: {customer_type}")

        structured_feats: dict[str, Any] = {}
        confidence_map: dict[str, float] = {}
        raw_ocr_text: dict[str, str] = {}

        # Cross-document identity validation names
        identity_names: list[str] = []

        if documents:
            for doc in documents:
                doc_type = doc.get("type", "auto")
                doc_bytes = doc.get("bytes", b"")
                result = self.textract.extract_document(doc_bytes, doc_type)
                raw_ocr_text[doc_type] = str(result.get("fields", {}))
                structured_feats.update(result.get("fields", {}))
                confidence_map.update(result.get("confidence", {}))

                if "identity_verified" in result:
                    structured_feats["identity_verified"] = result["identity_verified"]
                    confidence_map["identity_verified"] = min(
                        result.get("confidence", {}).values(), default=0.5
                    )

                # Collect names for cross-validation
                name = result.get("fields", {}).get("full_name", "")
                if name:
                    identity_names.append(name.strip().upper())

        # ── Cross-document identity validation ──
        if len(identity_names) >= 2:
            # Check if all names match
            if len(set(identity_names)) == 1:
                structured_feats["identity_consistency_flag"] = "OK"
            else:
                structured_feats["identity_consistency_flag"] = "MISMATCH"
                logger.warning(
                    f"Identity MISMATCH detected: {identity_names}"
                )
        elif len(identity_names) == 1:
            structured_feats["identity_consistency_flag"] = "OK"
        else:
            structured_feats["identity_consistency_flag"] = "MISSING"

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A1",
            "action": "ingest_documents",
            "input_summary": {
                "n_documents": len(documents) if documents else 0,
            },
            "output_summary": {
                "n_fields_extracted": len(structured_feats),
                "identity_consistency": structured_feats.get("identity_consistency_flag"),
            },
            "model_version": None,
            "confidence": None,
        }

        return {
            "application_id": application_id,
            "customer_type": customer_type,
            "raw_ocr_text": raw_ocr_text,
            "structured_feats": structured_feats,
            "confidence_map": confidence_map,
            "audit_trail": [audit_entry],
        }

    def check_cic(
        self,
        applicant_id: str,
    ) -> dict[str, Any]:
        """Node 2: check_cic — Credit Information Center API query.

        Queries CIC for credit history. If no record found,
        sets thin_file_flag = True to activate alternative scoring path.

        Runs in PARALLEL with analyze_transactions.

        Args:
            applicant_id: Unique applicant identifier.

        Returns:
            State update dict with CIC features and confidence.
        """
        logger.info(f"A1 check_cic — Querying CIC for {applicant_id}")

        cic_result = self.cic.query(applicant_id)

        structured_feats: dict[str, Any] = {
            "cic_score": cic_result.get("cic_score"),
            "debt_group": cic_result.get("debt_group"),
            "num_active_loans": cic_result.get("num_active_loans", 0),
            "total_outstanding": cic_result.get("total_outstanding", 0),
            "worst_ever_group": cic_result.get("worst_ever_group"),
            "thin_file_flag": cic_result.get("thin_file_flag", False),
        }

        confidence_map: dict[str, float] = {}
        if cic_result.get("thin_file_flag"):
            confidence_map["thin_file_flag"] = 1.0
            confidence_map["debt_group"] = 0.0  # no CIC data
        else:
            confidence_map["cic_score"] = 0.95
            confidence_map["debt_group"] = 0.95
            confidence_map["thin_file_flag"] = 1.0

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A1",
            "action": "check_cic",
            "input_summary": {"applicant_id": applicant_id},
            "output_summary": {
                "thin_file": structured_feats["thin_file_flag"],
                "cic_score": structured_feats["cic_score"],
                "debt_group": structured_feats["debt_group"],
            },
            "model_version": None,
            "confidence": None,
        }

        return {
            "structured_feats": structured_feats,
            "confidence_map": confidence_map,
            "audit_trail": [audit_entry],
        }

    def analyze_transactions(
        self,
        bank_statement_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Node 3: analyze_transactions — Bank statement CSV parser.

        Parses 6-month bank statement and extracts 8 alternative data features.
        This is the core innovation enabling thin-file credit assessment.

        Runs in PARALLEL with check_cic.

        Args:
            bank_statement_path: Path to bank statement CSV.

        Returns:
            State update dict with transaction features and confidence.
        """
        logger.info(f"A1 analyze_transactions — Parsing bank statement")

        structured_feats: dict[str, Any] = {}
        confidence_map: dict[str, float] = {}
        missing_fields: list[str] = []

        if bank_statement_path:
            try:
                bank_result = parse_bank_statement(bank_statement_path)
                bank_features = bank_result["features"]
                structured_feats.update(bank_features)

                # Bank statement confidence based on data quality
                n_months = bank_result["metadata"]["n_months"]
                base_conf = min(0.95, 0.70 + 0.05 * n_months)
                for feat_name in bank_features:
                    confidence_map[feat_name] = base_conf

                # Monthly income proxy
                structured_feats["monthly_income_or_inflow"] = bank_features["avg_monthly_inflow_vnd"]
                confidence_map["monthly_income_or_inflow"] = base_conf

            except ValueError as e:
                logger.warning(f"Bank statement parsing error: {e}")
                confidence_map["monthly_income_or_inflow"] = 0.0
        else:
            logger.warning("No bank statement provided")
            confidence_map["monthly_income_or_inflow"] = 0.0

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A1",
            "action": "analyze_transactions",
            "input_summary": {
                "has_bank_statement": bank_statement_path is not None,
            },
            "output_summary": {
                "n_features_extracted": len(structured_feats),
            },
            "model_version": None,
            "confidence": None,
        }

        return {
            "structured_feats": structured_feats,
            "confidence_map": confidence_map,
            "missing_fields": missing_fields,
            "audit_trail": [audit_entry],
        }

    # ── Monolithic ingestion (for testing / backward compat) ──────────────────

    def ingest(
        self,
        applicant_id: str,
        customer_type: str = "INDIVIDUAL",
        documents: list[dict[str, Any]] | None = None,
        bank_statement_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run full ingestion pipeline as a single node (backward compatible).

        For the 9-node graph, use the split methods instead:
        ingest_documents(), check_cic(), analyze_transactions().
        """
        # Run all 3 channels
        doc_result = self.ingest_documents(applicant_id, customer_type, documents)
        cic_result = self.check_cic(applicant_id)
        tx_result = self.analyze_transactions(bank_statement_path)

        # Merge results
        structured_feats = {
            **doc_result.get("structured_feats", {}),
            **cic_result.get("structured_feats", {}),
            **tx_result.get("structured_feats", {}),
        }
        confidence_map = {
            **doc_result.get("confidence_map", {}),
            **cic_result.get("confidence_map", {}),
            **tx_result.get("confidence_map", {}),
        }

        # Identify missing fields
        from creditlens.config.feature_config import FIELD_DEFINITIONS, CONFIDENCE_THRESHOLDS

        missing_fields = []
        for field_name, field_def in FIELD_DEFINITIONS.items():
            threshold = CONFIDENCE_THRESHOLDS[field_def.tier]
            conf = confidence_map.get(field_name, 0.0)
            if conf < threshold:
                missing_fields.append(field_name)

        # Merge audit trails
        audit_trail = (
            doc_result.get("audit_trail", [])
            + cic_result.get("audit_trail", [])
            + tx_result.get("audit_trail", [])
        )

        return {
            "application_id": doc_result.get("application_id"),
            "customer_type": customer_type,
            "raw_ocr_text": doc_result.get("raw_ocr_text", {}),
            "structured_feats": structured_feats,
            "confidence_map": confidence_map,
            "missing_fields": missing_fields,
            "audit_trail": audit_trail,
        }
