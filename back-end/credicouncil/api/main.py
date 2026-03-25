"""
CREDICOUNCIL FastAPI Service — scoring endpoint for credit applications.

Usage:
    uvicorn CREDICOUNCIL.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env — check PROJECT_ROOT first, then parent (repo root)
try:
    from dotenv import load_dotenv
    env_local = PROJECT_ROOT / ".env"
    env_parent = PROJECT_ROOT.parent / ".env"
    if env_local.exists():
        load_dotenv(env_local)
    elif env_parent.exists():
        load_dotenv(env_parent)
    else:
        load_dotenv()  # fallback: search cwd upward
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="credicouncil AI",
    description="AI-powered credit scoring pipeline: A1→A2→A3→A4",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files — check PROJECT_ROOT and parent (repo root)
_FRONTEND_DIR = PROJECT_ROOT / "front-end" / "app"
if not _FRONTEND_DIR.exists():
    _FRONTEND_DIR = PROJECT_ROOT.parent / "front-end" / "app"
if _FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
    logger.info(f"Frontend served at /app from {_FRONTEND_DIR}")

# ─── Lazy-loaded global agents ────────────────────────────────────────────────

_agents: dict[str, Any] = {}


def get_agents():
    """Lazy-load all agents on first request."""
    if _agents:
        return _agents

    from credicouncil.agents.a1_ingestion.agent import IngestionAgent
    from credicouncil.agents.a2_feature_engineer.agent import FeatureEngineerAgent
    from credicouncil.agents.a3_scoring.agent import ScoringAgent
    from credicouncil.agents.a4_report_generator.agent import ReportGeneratorAgent

    model_path = os.environ.get("MODEL_PATH", "models/lgbm_ref_v1.pkl")

    # Resolve relative paths from project root
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)

    _agents["a1"] = IngestionAgent()
    _agents["a2"] = FeatureEngineerAgent()
    _agents["a3"] = ScoringAgent(model_path=model_path)
    _agents["a4"] = ReportGeneratorAgent()

    logger.info(f"Agents loaded (model={model_path})")
    return _agents


# ─── Pydantic schemas ────────────────────────────────────────────────────────

class ScoringResult(BaseModel):
    credit_score: int
    pd_probability: float
    risk_band: str
    decision: str
    shap_top_positive: list[dict[str, Any]]
    shap_top_negative: list[dict[str, Any]]
    five_c_scores: dict[str, int]
    five_c_total: int
    recommendation: str
    consistency_check: bool
    audit_trail: list[dict[str, Any]]
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = Path(os.environ.get("MODEL_PATH", "models/lgbm_ref_v1.pkl")).exists()
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
    )


@app.post("/score/customer-folder", response_model=ScoringResult)
async def score_customer_folder(customer_folder: str = Form(...)):
    """Score a customer from a pre-existing mock data folder.

    Args:
        customer_folder: Path to customer data folder (e.g., data/mock/customer_001)
    """
    folder = Path(customer_folder)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Customer folder not found: {customer_folder}")

    try:
        agents = get_agents()
        result = _run_pipeline(agents, str(folder))
        return ScoringResult(**result)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/upload", response_model=ScoringResult)
async def score_upload(
    files: list[UploadFile] = File(...),
    sk_id_curr: int = Form(default=100002),
):
    """Score a customer by uploading files.

    Upload PDFs, CSV bank statements, and JSON CIC data.
    Files are identified by extension and naming convention:
      - *.pdf → Document PDFs
      - *.csv → Bank statement
      - *cic*.json → CIC API data
      - *internal*.json → Internal DB data
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            file_path = Path(tmpdir) / f.filename
            content = await f.read()
            file_path.write_bytes(content)

        try:
            agents = get_agents()
            result = _run_pipeline(agents, tmpdir, sk_id_curr=sk_id_curr)
            return ScoringResult(**result)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/json")
async def score_json(data: dict[str, Any]):
    """Score a customer from raw JSON data.

    Accepts pre-formatted application data directly.
    Expects: {"application_row": {...}, "bureau_records": [...], ...}
    """
    try:
        agents = get_agents()

        # Skip A1, go directly to A2
        a1_output = data
        if "application_row" not in a1_output:
            raise HTTPException(status_code=400, detail="Missing 'application_row' in request body")

        logger.info("[2/4] A2: Feature Engineering...")
        a2_output = agents["a2"].process(a1_output)

        logger.info("[3/4] A3: Scoring...")
        a3_output = agents["a3"].score(a2_output)

        logger.info("[4/4] A4: Report...")
        a4_output = agents["a4"].generate(a3_output, a2_output, a1_output)

        return _format_result(a3_output, a4_output)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Mock customer demo endpoint ──────────────────────────────────────────────

_MOCK_FOLDERS = {
    "001": "customer_001",
    "002": "customer_002",
    "003": "customer_003",
    "004": "customer_004",
}


@app.post("/score/mock", response_model=ScoringResult)
async def score_mock_customer(customer_id: str = Form(...)):
    """Score one of the 4 pre-built demo customers by ID (001–004).

    Used by the demo dashboard frontend.
    """
    folder_name = _MOCK_FOLDERS.get(customer_id)
    if not folder_name:
        raise HTTPException(status_code=400, detail=f"Unknown customer_id: {customer_id}. Use 001-004.")

    customer_folder = PROJECT_ROOT / "data" / "mock" / folder_name
    if not customer_folder.exists():
        raise HTTPException(status_code=404, detail=f"Mock folder not found: {customer_folder}")

    try:
        agents = get_agents()
        result = _run_pipeline(agents, str(customer_folder))
        return ScoringResult(**result)
    except Exception as e:
        logger.error(f"Pipeline error for customer {customer_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── PDF Report endpoint ──────────────────────────────────────────────────────

@app.get("/v1/report/{customer_id}/pdf")
async def get_report_pdf(customer_id: str, download: int = 0):
    """Generate or serve cached PDF credit report for a customer.

    Used by the frontend PDF viewer.
    First checks for cached PDF; if not found, generates from cached JSON report.
    """
    folder_name = _MOCK_FOLDERS.get(customer_id)
    if not folder_name:
        raise HTTPException(status_code=400, detail=f"Unknown customer_id: {customer_id}")

    customer_folder = PROJECT_ROOT / "data" / "mock" / folder_name

    # Check for cached PDF
    pdf_path = customer_folder / "credit_report.pdf"
    if pdf_path.exists():
        content_disp = "attachment" if download else "inline"
        return StreamingResponse(
            open(pdf_path, "rb"),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'{content_disp}; filename="credit_report_{customer_id}.pdf"'
            },
        )

    # No cached PDF — try to generate from cached JSON report
    report_path = customer_folder / "credit_report.json"
    shap_path = customer_folder / "shap_values.json"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No report found for customer {customer_id}. Run pipeline first."
        )

    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        shap_data = json.loads(shap_path.read_text(encoding="utf-8")) if shap_path.exists() else {}

        from credicouncil.agents.a4_report_generator.pdf_generator import generate_credit_pdf
        pdf_bytes = generate_credit_pdf(
            report_data=report_data,
            shap_data=shap_data,
            customer_name=f"Customer_{customer_id}",
        )

        # Cache the PDF for next time
        pdf_path.write_bytes(pdf_bytes)

        content_disp = "attachment" if download else "inline"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'{content_disp}; filename="credit_report_{customer_id}.pdf"'
            },
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


# ─── Pipeline runner ──────────────────────────────────────────────────────────

def _run_pipeline(agents: dict, customer_folder: str, sk_id_curr: int = None) -> dict:
    """Run the full A1→A2→A3→A4 pipeline."""

    logger.info(f"[1/4] A1: Ingesting from {customer_folder}...")
    a1_output = agents["a1"].ingest(customer_folder)

    logger.info("[2/4] A2: Feature Engineering...")
    a2_output = agents["a2"].process(a1_output)

    logger.info("[3/4] A3: ML Scoring...")
    a3_output = agents["a3"].score(a2_output)

    logger.info("[4/4] A4: Report Generation...")
    a4_output = agents["a4"].generate(a3_output, a2_output, a1_output)

    # Cache report + SHAP for PDF generation
    try:
        folder = Path(customer_folder)
        report = a4_output.get("final_report", {})
        shap_data = a3_output.get("shap_values", {})

        with open(folder / "credit_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        with open(folder / "shap_values.json", "w", encoding="utf-8") as f:
            json.dump(shap_data, f, ensure_ascii=False, indent=2, default=str)

        # Delete stale PDF cache so next /v1/report/{id}/pdf regenerates from new JSON
        pdf_cache = folder / "credit_report.pdf"
        if pdf_cache.exists():
            pdf_cache.unlink()

        # Also generate and cache PDF
        from credicouncil.agents.a4_report_generator.pdf_generator import generate_credit_pdf
        pdf_bytes = generate_credit_pdf(report_data=report, shap_data=shap_data)
        pdf_cache.write_bytes(pdf_bytes)
        logger.info(f"  PDF cached: {pdf_cache}")
    except Exception as e:
        logger.warning(f"Failed to cache reports: {e}")

    return _format_result(a3_output, a4_output)


def _format_result(a3_output: dict, a4_output: dict) -> dict:
    """Format pipeline outputs into API response."""
    report = a4_output.get("final_report", {})
    executive = report.get("executive_summary", {})
    shap = a3_output.get("shap_values", {})

    return {
        "credit_score": a3_output.get("credit_score", 0),
        "pd_probability": a3_output.get("pd_pct", 0),
        "risk_band": a3_output.get("risk_band", "N/A"),
        "decision": a3_output.get("routing", "REVIEW"),
        "shap_top_positive": shap.get("top_positive_factors", []),
        "shap_top_negative": shap.get("top_negative_factors", []),
        "five_c_scores": a4_output.get("five_c_scores", {}),
        "five_c_total": executive.get("five_c_total", 0),
        "recommendation": executive.get("recommendation", "REVIEW"),
        "consistency_check": a4_output.get("consistency_check", {}).get("passed", False),
        "audit_trail": a4_output.get("audit_trail", []),
        "warnings": a4_output.get("warnings", []),
    }
