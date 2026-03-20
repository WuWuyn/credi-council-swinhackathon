"""
CreditLens A1 — AWS Textract Service.

Wraps AWS Textract Analyze Lending API for PDF/Scan document processing.
Extracts identity fields, employment info, and collateral data.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TextractService:
    """AWS Textract document processing service.

    Uses Textract's Analyze Lending API for:
    - CCCD/CMND: identity extraction
    - Hợp đồng lao động: employment fields
    - GPKD: business registration (SME)
    - TSBĐ: collateral documents
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.client = None

        if not use_mock:
            import boto3
            from creditlens.config.settings import get_settings
            settings = get_settings()
            self.client = boto3.client("textract", region_name=settings.aws_region)

    def extract_document(self, document_bytes: bytes, doc_type: str = "auto") -> dict[str, Any]:
        """Extract fields from a document.

        Args:
            document_bytes: Raw document bytes (PDF or image).
            doc_type: Document type hint: "cccd", "employment", "gpkd", "collateral", "auto".

        Returns:
            Dict with extracted fields and confidence scores.
        """
        if self.use_mock:
            return self._mock_extract(doc_type)

        # Real Textract API call
        response = self.client.analyze_document(
            Document={"Bytes": document_bytes},
            FeatureTypes=["FORMS", "TABLES"],
        )

        return self._parse_textract_response(response, doc_type)

    def _mock_extract(self, doc_type: str) -> dict[str, Any]:
        """Generate mock extraction results for development."""

        if doc_type in ("cccd", "auto"):
            return {
                "doc_type": "CCCD",
                "fields": {
                    "full_name": "NGUYỄN VĂN A",
                    "id_number": "001099012345",
                    "date_of_birth": "1998-05-15",
                    "gender": "Nam",
                    "permanent_address": "Hà Nội",
                },
                "confidence": {
                    "full_name": 0.97,
                    "id_number": 0.95,
                    "date_of_birth": 0.93,
                    "gender": 0.99,
                    "permanent_address": 0.88,
                },
                "identity_verified": True,
                "identity_consistency_flag": "OK",
            }

        if doc_type == "employment":
            return {
                "doc_type": "HOP_DONG_LAO_DONG",
                "fields": {
                    "employer": "FPT Software",
                    "position": "Software Engineer",
                    "monthly_salary": 15_000_000,
                    "start_date": "2025-09-01",
                    "contract_type": "UNLIMITED",
                },
                "confidence": {
                    "employer": 0.92,
                    "position": 0.88,
                    "monthly_salary": 0.85,
                    "start_date": 0.90,
                    "contract_type": 0.87,
                },
            }

        if doc_type == "gpkd":
            return {
                "doc_type": "GIAY_PHEP_KINH_DOANH",
                "fields": {
                    "business_name": "Công ty TNHH ABC",
                    "registration_number": "0123456789",
                    "registration_date": "2020-03-15",
                    "business_type": "Thương mại dịch vụ",
                    "owner": "NGUYỄN VĂN A",
                },
                "confidence": {
                    "business_name": 0.94,
                    "registration_number": 0.96,
                    "registration_date": 0.91,
                    "business_type": 0.85,
                    "owner": 0.93,
                },
            }

        # Default
        return {"doc_type": "UNKNOWN", "fields": {}, "confidence": {}}

    def _parse_textract_response(self, response: dict, doc_type: str) -> dict[str, Any]:
        """Parse Textract API response into structured fields."""
        fields = {}
        confidence = {}

        for block in response.get("Blocks", []):
            if block["BlockType"] == "KEY_VALUE_SET" and "KEY" in block.get("EntityTypes", []):
                key_text = self._get_text_from_block(block, response["Blocks"])
                value_block = self._find_value_block(block, response["Blocks"])
                if value_block:
                    value_text = self._get_text_from_block(value_block, response["Blocks"])
                    fields[key_text] = value_text
                    confidence[key_text] = block.get("Confidence", 0) / 100

        return {"doc_type": doc_type, "fields": fields, "confidence": confidence}

    @staticmethod
    def _get_text_from_block(block: dict, all_blocks: list) -> str:
        """Extract text content from a Textract block."""
        text_parts = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    child = next((b for b in all_blocks if b["Id"] == child_id), None)
                    if child and child["BlockType"] == "WORD":
                        text_parts.append(child.get("Text", ""))
        return " ".join(text_parts)

    @staticmethod
    def _find_value_block(key_block: dict, all_blocks: list) -> dict | None:
        """Find the value block associated with a key block."""
        for rel in key_block.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for val_id in rel["Ids"]:
                    return next((b for b in all_blocks if b["Id"] == val_id), None)
        return None
