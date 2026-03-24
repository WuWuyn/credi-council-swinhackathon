"""
CREDICOUNCIL — Docling OCR Service.

Smart 3-tier extraction:
1. PyMuPDF (instant) → nếu có text-layer → xong
2. Docling FULL (EasyOCR + layout) → nếu scanned PDF → xong
3. Empty fallback

Supports Vietnamese via EasyOCR lang=['vi', 'en'].
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized Docling converter (only loaded when needed)
_full_converter = None


def _get_full_converter():
    """Full Docling converter — only initialized when scanned PDFs are detected."""
    global _full_converter
    if _full_converter is not None:
        return _full_converter

    import os
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
        AcceleratorDevice,
    )

    # Read device config: cpu | cuda | mps
    device = os.getenv("DOCLING_DEVICE", "cpu").strip().lower()
    use_gpu = device in ("cuda", "gpu")

    # Map to Docling AcceleratorDevice enum
    if device == "cuda" or device == "gpu":
        accel = AcceleratorDevice.CUDA
    elif device == "mps":
        accel = AcceleratorDevice.MPS
    else:
        accel = AcceleratorDevice.CPU

    logger.info(f"Initializing Docling FULL converter (EasyOCR, device={device})...")

    ocr_options = EasyOcrOptions(
        lang=["vi", "en"],
        use_gpu=use_gpu,  # EasyOCR GPU flag
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.accelerator_device = accel

    _full_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    logger.info(f"Docling FULL converter ready (device={device}).")
    return _full_converter


class DoclingOCRService:
    """Smart OCR service: PyMuPDF (fast) → Docling+EasyOCR (full).

    Tier 1: PyMuPDF text extraction (~0.01s per PDF)
        - Digital PDFs with text layer → instant extraction
    Tier 2: Docling + EasyOCR (~5-10s per PDF)
        - Scanned PDFs, image-only PDFs → full OCR pipeline
        - Only loaded/initialized when actually needed

    Usage:
        service = DoclingOCRService()
        result = service.extract(pdf_path)
        markdown = result["markdown"]
    """

    MIN_TEXT_THRESHOLD = 50  # Minimum chars for successful text extraction

    def extract(self, pdf_path: str | Path) -> dict[str, Any]:
        """Extract text from PDF with smart tier selection."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return self._empty_result()

        t0 = time.time()

        # ── Tier 1: PyMuPDF (instant) ────────────────────────────────
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            pages_text = [page.get_text() for page in doc]
            text = "\n".join(pages_text)
            doc.close()

            if len(text.strip()) >= self.MIN_TEXT_THRESHOLD:
                dt = time.time() - t0
                logger.info(
                    f"PyMuPDF: {pdf_path.name} → {len(text)} chars in {dt:.2f}s"
                )
                return {
                    "markdown": text,
                    "raw_text": text,
                    "tables": [],
                    "n_pages": len(pages_text),
                    "mode": "pymupdf",
                    "time_s": round(dt, 2),
                }

            logger.info(
                f"PyMuPDF: {pdf_path.name} → only {len(text.strip())} chars, "
                f"switching to Docling OCR..."
            )
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path.name}: {e}")

        # ── Tier 2: Docling + EasyOCR (for scanned PDFs) ────────────
        try:
            converter = _get_full_converter()
            result = converter.convert(str(pdf_path))
            doc = result.document

            markdown = doc.export_to_markdown()

            tables = []
            if hasattr(doc, 'tables') and doc.tables:
                for table in doc.tables:
                    try:
                        df = table.export_to_dataframe()
                        tables.append(df)
                    except Exception:
                        pass

            n_pages = len(doc.pages) if hasattr(doc, 'pages') else 0
            dt = time.time() - t0

            logger.info(
                f"Docling OCR: {pdf_path.name} → {len(markdown)} chars in {dt:.1f}s"
            )
            return {
                "markdown": markdown,
                "raw_text": markdown,
                "tables": tables,
                "n_pages": n_pages,
                "mode": "docling_ocr",
                "time_s": round(dt, 1),
            }
        except Exception as e:
            logger.error(f"Docling OCR failed for {pdf_path.name}: {e}")

        # ── Empty fallback ───────────────────────────────────────────
        return self._empty_result()

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "markdown": "",
            "raw_text": "",
            "tables": [],
            "n_pages": 0,
            "mode": "none",
            "time_s": 0,
        }
