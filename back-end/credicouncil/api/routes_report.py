"""
CREDICOUNCIL API — Report Routes.

GET  /v1/report/{id}/json       → Get JSON report data
GET  /v1/report/{id}/pdf        → Stream PDF credit report
POST /v1/report/generate-pdf    → Generate PDF from raw JSON
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from credicouncil.api.config import OUTPUT_DIR, settings
from credicouncil.api.data_access import load_customer_report, normalize_folder_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.API_V1_STR, tags=["Report"])


@router.get("/report/{customer_id}/json")
async def get_report_json(customer_id: str):
    """
    Get the raw JSON report data for a given customer.

    Reads from data/output/ first, falls back to data/mock/.
    """
    try:
        report_data, shap_data = load_customer_report(customer_id)
        if not report_data:
            raise HTTPException(
                status_code=404,
                detail=f"No report found for {customer_id}. Run pipeline first.",
            )
        return {"report_data": report_data, "shap_data": shap_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load JSON report for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{customer_id}/pdf")
async def get_report_pdf(
    customer_id: str,
    download: bool = Query(False, description="Set true to force file download"),
):
    """
    Stream a PDF credit report for a given customer.

    Reads from data/output/ first — if PDF exists there, serves directly.
    Otherwise generates from JSON and caches to output/.
    """
    folder_id = normalize_folder_id(customer_id)
    output_folder = OUTPUT_DIR / folder_id
    pdf_path = output_folder / "credit_report.pdf"
    disposition = "attachment" if download else "inline"
    filename = f"credit_report_{customer_id}.pdf"

    # ── Serve existing PDF from output/ ──
    if pdf_path.exists():
        logger.info(f"Serving PDF from output/ for customer {customer_id}")
        pdf_bytes = pdf_path.read_bytes()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'{disposition}; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    # ── Generate from JSON (output/ → mock/ fallback) ──
    try:
        report_data, shap_data = load_customer_report(customer_id)
        if not report_data:
            raise HTTPException(
                status_code=404,
                detail=f"No report data found for {customer_id}.",
            )
    except HTTPException:
        raise

    customer_info = report_data.get("customer_info", {})
    customer_name = customer_info.get("name") or f"KH #{customer_id}"

    logger.info(f"Generating PDF for customer {customer_id}")
    try:
        from credicouncil.agents.a4_report_generator.pdf_generator import generate_credit_pdf

        pdf_bytes = generate_credit_pdf(
            report_data=report_data,
            shap_data=shap_data,
            customer_name=customer_name,
        )
        # Cache PDF to output/
        output_folder.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(pdf_bytes)
        logger.info(f"PDF cached: {pdf_path}")
    except Exception as e:
        logger.error(f"PDF generation failed for {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@router.post("/report/generate-pdf")
async def generate_pdf_from_json(
    report_data: dict[str, Any],
    shap_data: dict[str, Any] = {},
    customer_name: str = Query("Khách hàng"),
    download: bool = Query(False),
):
    """Generate PDF directly from raw report JSON (no customer folder needed)."""
    from credicouncil.agents.a4_report_generator.pdf_generator import generate_credit_pdf

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
