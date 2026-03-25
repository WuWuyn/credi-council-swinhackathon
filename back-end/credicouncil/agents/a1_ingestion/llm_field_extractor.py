"""
CREDICOUNCIL — LLM-based Field Extractor.

Uses Gemini structured output with Pydantic schemas to extract
typed fields from OCR text. Replaces regex-based field matching
with LLM semantic understanding.

Pipeline:
    OCR text (from Docling/PyMuPDF) → LLMService → Pydantic validation → typed dict
"""

from __future__ import annotations

import logging
from typing import Any

from credicouncil.services.llm_service import LLMService
from credicouncil.schemas.document_schemas import (
    DOC_SCHEMAS,
    DOC_TYPE_VI,
)

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM_PROMPT = """Bạn là chuyên gia trích xuất dữ liệu từ tài liệu tài chính ngân hàng Việt Nam.

NHIỆM VỤ: Đọc văn bản OCR bên dưới và trích xuất thông tin theo đúng JSON schema cho sẵn.

QUY TẮC BẮT BUỘC:
1. CHỈ trích xuất thông tin CÓ TRONG văn bản. KHÔNG suy đoán, KHÔNG bịa thêm.
2. Nếu một trường không tìm thấy trong văn bản → trả null.
3. Ngày tháng → format YYYY-MM-DD (ví dụ: "15/03/1965" → "1965-03-15").
4. Số tiền VND → số nguyên, KHÔNG có dấu chấm/phẩy phân cách.
   Ví dụ: "35.000.000 VND/tháng" → 35000000; "900,000" → 900000.
5. Boolean → true/false (KHÔNG phải "Có"/"Không"/"Co"/"Khong").
   Ví dụ: "Có" → true; "Không"/"Khong" → false; "Co" → true.
6. Giá trị normalized (0-1) → giữ nguyên dạng float.
7. Giá trị "N/A", "Không có", "-", "null", trống → trả null.
8. Giá trị tình trạng hôn nhân phải là 1 trong: "Married", "Single / not married", "Separated", "Widow", "Civil marriage".
   Map: "Đã kết hôn"/"Kết hôn"/"Ket hon"/"Da ket hon" → "Married"; "Độc thân"/"Doc than" → "Single / not married"; "Ly hôn"/"Ly hon" → "Separated"; "Góa"/"Goa" → "Widow".
9. Giới tính (gender):
   - "Nữ"/"Nu"/"Nữ giới"/"Female"/"F" → "Nữ"
   - "Nam"/"Male"/"M"/"Nam giới" → "Nam"
   - Nếu tên có "Thị" (ví dụ: "Nguyễn Thị ...") thì giới tính = "Nữ"
   - Nếu tên có "Văn" (ví dụ: "Nguyễn Văn ...") thì giới tính = "Nam"
   - Nếu không rõ → trả null.
10. Tình trạng khẩn cấp (emergency_state):
    - "Bình thường"/"Binh thuong"/"Không khẩn cấp"/"Khong khan cap"/"Normal"/"No" → "No"
    - "Khẩn cấp"/"Khan cap"/"Emergency"/"Yes" → "Yes"
    - Nếu cụm từ chứa "binh thuong" hoặc "khong khan cap" → "No"
11. Chỉ trả về JSON, không giải thích, không markdown."""

EXTRACT_USER_TEMPLATE = """Trích xuất thông tin từ tài liệu "{doc_type_vi}" sau đây.

=== VĂN BẢN OCR ===
{ocr_text}
=== HẾT VĂN BẢN ===

Trả về JSON theo đúng schema. Trường nào không tìm thấy → null."""


class LLMFieldExtractor:
    """Extract structured fields from OCR text using Gemini + Pydantic.

    Uses LLMService.generate_structured() for unified Gemini access.

    Usage:
        extractor = LLMFieldExtractor()
        fields, confidence = extractor.extract("housing", ocr_text)
    """

    def __init__(self):
        self.llm = LLMService()

    def extract(
        self,
        doc_type: str,
        ocr_text: str,
        max_text_chars: int = 8000,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Extract fields from OCR text for a specific document type.

        Args:
            doc_type: One of "cccd", "employment", "household", "housing", "loan_application".
            ocr_text: OCR-extracted text (plain or markdown).
            max_text_chars: Maximum text length to send to LLM.

        Returns:
            Tuple of (fields_dict, confidence_dict).
        """
        schema_class = DOC_SCHEMAS.get(doc_type)
        if schema_class is None:
            logger.warning(f"No schema for doc_type={doc_type}, skipping LLM extraction")
            return {}, {}

        doc_type_vi = DOC_TYPE_VI.get(doc_type, doc_type)

        # Truncate text if too long
        text = ocr_text[:max_text_chars]

        user_prompt = EXTRACT_USER_TEMPLATE.format(
            doc_type_vi=doc_type_vi,
            ocr_text=text,
        )

        fields = self.llm.generate_structured(
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_class=schema_class,
            max_tokens=8192,
            temperature=0.1,
        )
        confidence = self._estimate_confidence(fields, ocr_text)

        n_filled = sum(1 for v in fields.values() if v is not None)
        logger.info(
            f"LLM extracted {n_filled}/{len(fields)} fields for {doc_type}"
        )

        return fields, confidence

    async def extract_async(
        self,
        doc_type: str,
        ocr_text: str,
        max_text_chars: int = 8000,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Async version of extract() for parallel LLM calls.

        Uses LLMService.generate_structured_async() to enable
        concurrent extraction of multiple PDFs.
        """
        schema_class = DOC_SCHEMAS.get(doc_type)
        if schema_class is None:
            logger.warning(f"No schema for doc_type={doc_type}, skipping LLM extraction")
            return {}, {}

        doc_type_vi = DOC_TYPE_VI.get(doc_type, doc_type)
        text = ocr_text[:max_text_chars]

        user_prompt = EXTRACT_USER_TEMPLATE.format(
            doc_type_vi=doc_type_vi,
            ocr_text=text,
        )

        fields = await self.llm.generate_structured_async(
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_class=schema_class,
            max_tokens=8192,
            temperature=0.1,
        )
        confidence = self._estimate_confidence(fields, ocr_text)

        n_filled = sum(1 for v in fields.values() if v is not None)
        logger.info(
            f"LLM extracted {n_filled}/{len(fields)} fields for {doc_type}"
        )

        return fields, confidence

    def _estimate_confidence(
        self,
        fields: dict[str, Any],
        ocr_text: str,
    ) -> dict[str, float]:
        """Estimate confidence for each extracted field.

        Heuristic: LLM extraction has base confidence 0.90.
        Higher if the value appears verbatim in OCR text.
        """
        confidence = {}
        for key, val in fields.items():
            if val is None:
                confidence[key] = 0.0
            elif isinstance(val, (str,)) and val in ocr_text:
                confidence[key] = 0.95  # Exact match in source
            elif isinstance(val, bool):
                confidence[key] = 0.90
            elif isinstance(val, (int, float)):
                # Check if numeric value appears in text
                if str(val) in ocr_text or str(int(val)) in ocr_text:
                    confidence[key] = 0.93
                else:
                    confidence[key] = 0.88
            else:
                confidence[key] = 0.90
        return confidence
