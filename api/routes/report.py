"""
CreditLens API — PDF Report Endpoint.

GET  /report/{customer_id}/pdf     → Stream PDF for browser preview
GET  /report/{customer_id}/pdf?download=1 → Force download with filename
POST /report/generate-pdf          → Generate from raw report + shap JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Default mock data directory
MOCK_DIR = Path("data/mock")


def _load_customer_data(customer_id: str) -> tuple[dict, dict]:
    """Load credit_report.json + shap_values from customer folder."""
    # Support formats: "001", "1", "customer_001"
    folder_id = customer_id.zfill(3) if customer_id.isdigit() else customer_id
    if not folder_id.startswith("customer_"):
        folder_id = f"customer_{folder_id}"

    folder = MOCK_DIR / folder_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Customer folder not found: {folder}")

    report_path = folder / "credit_report.json"
    shap_path   = folder / "shap_values.json"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"credit_report.json not found for {customer_id}")

    with open(report_path, encoding="utf-8") as f:
        report_data = json.load(f)

    shap_data = {}
    if shap_path.exists():
        with open(shap_path, encoding="utf-8") as f:
            shap_data = json.load(f)

    return report_data, shap_data


@router.get("/report/{customer_id}/pdf")
async def get_report_pdf(
    customer_id: str,
    download: bool = Query(False, description="Set true to force download instead of preview"),
):
    """
    Stream a professional PDF credit report for a given customer.

    - **customer_id**: e.g. "001", "1", "customer_001"
    - **download**: if true, browser downloads the file; otherwise inline preview
    """
    from creditlens.agents.a4_report_generator.pdf_generator import generate_credit_pdf

    report_data, shap_data = _load_customer_data(customer_id)

    # Extract customer name from report
    customer_info = report_data.get("customer_info", {})
    customer_name = customer_info.get("name") or f"KH #{customer_id}"

    logger.info(f"Generating PDF for customer {customer_id}")
    try:
        pdf_bytes = generate_credit_pdf(
            report_data=report_data,
            shap_data=shap_data,
            customer_name=customer_name,
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

    # Disposition: inline for preview, attachment for download
    filename = f"credit_report_{customer_id}.pdf"
    disposition = "attachment" if download else "inline"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


class PDFFromJSONRequest:
    """Request body for POST /report/generate-pdf."""
    report_data: dict[str, Any]
    shap_data: dict[str, Any] = {}
    customer_name: str = "Khách hàng"


@router.post("/report/generate-pdf")
async def generate_pdf_from_json(
    report_data: dict[str, Any],
    shap_data: dict[str, Any] = {},
    customer_name: str = Query("Khách hàng"),
    download: bool = Query(False),
):
    """Generate PDF directly from report JSON (no customer folder needed)."""
    from creditlens.agents.a4_report_generator.pdf_generator import generate_credit_pdf

    try:
        pdf_bytes = generate_credit_pdf(
            report_data=report_data,
            shap_data=shap_data,
            customer_name=customer_name,
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    disposition = "attachment" if download else "inline"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'{disposition}; filename="credit_report.pdf"'},
    )
