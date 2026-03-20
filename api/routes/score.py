"""
CreditLens API — Score Endpoint.

POST /v1/score — Submit loan application for credit scoring.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ScoreResponse(BaseModel):
    """Credit scoring response."""

    application_id: str
    credit_score: int
    pd_pct: float
    risk_band: str
    recommendation: str
    overall_confidence: float
    four_c_scores: dict[str, float]
    warnings: list[str]
    report: dict[str, Any]


@router.post("/score", response_model=ScoreResponse)
async def score_application(
    bank_statement: UploadFile = File(None, description="Bank statement CSV (6 months)"),
    documents: list[UploadFile] = File(None, description="PDF documents (CCCD, contracts, etc.)"),
    applicant_id: str = Form("anonymous"),
    customer_type: str = Form("INDIVIDUAL"),
):
    """Submit a loan application for credit scoring.

    Process flow:
        1. Receive uploaded files
        2. Run full CreditLens pipeline (A1 → Gate → A2 → A3 → A4)
        3. Return credit report

    Returns:
        ScoreResponse with credit score, risk band, 4C assessment, and full report.
    """
    from creditlens.orchestrator.graph import run_pipeline

    # Save bank statement to temp file if provided
    bank_statement_path = None
    if bank_statement:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            content = await bank_statement.read()
            tmp.write(content)
            bank_statement_path = tmp.name

    # Read document bytes
    doc_list = None
    if documents:
        doc_list = []
        for doc in documents:
            content = await doc.read()
            doc_list.append({
                "type": doc.filename.split(".")[-1] if doc.filename else "auto",
                "bytes": content,
            })

    # Run pipeline
    result = run_pipeline(
        applicant_id=applicant_id,
        customer_type=customer_type,
        documents=doc_list,
        bank_statement_path=bank_statement_path,
        use_mock=True,  # Use mock in dev mode
    )

    return ScoreResponse(
        application_id=result.get("application_id", "unknown"),
        credit_score=result.get("credit_score", 0),
        pd_pct=result.get("pd_pct", 0.0),
        risk_band=result.get("risk_band", "CC"),
        recommendation=result.get("routing", "REVIEW"),
        overall_confidence=result.get("overall_confidence", 0.0),
        four_c_scores=result.get("four_c_scores", {}),
        warnings=result.get("warnings", []),
        report=result.get("final_report", {}),
    )
