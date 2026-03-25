"""
CREDICOUNCIL API — Scoring Routes.

POST /v1/score          → Run pipeline for a customer (used by frontend)
POST /score/mock        → Legacy: score mock customer by ID
POST /score/json        → Legacy: score from raw JSON
POST /score/upload      → Legacy: score from uploaded files
POST /score/customer-folder → Legacy: score from folder path
"""

from __future__ import annotations

import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from credicouncil.api.config import MOCK_DIR, settings
from credicouncil.api.data_access import normalize_folder_id
from credicouncil.api.pipeline import (
    format_legacy_result,
    format_score_response,
    get_agents,
    run_pipeline_for_customer,
)
from credicouncil.api.schemas import ScoreResponse, ScoringResult

logger = logging.getLogger(__name__)

# ─── Batch config ────────────────────────────────────────────────────────────
BATCH_STAGGER_DELAY = 2.0   # seconds between each pipeline launch
BATCH_MAX_WORKERS = 5       # max concurrent pipelines

# ─── v1 router (used by frontend) ───────────────────────────────────────────
router_v1 = APIRouter(prefix=settings.API_V1_STR, tags=["Scoring"])

# ─── Legacy router (backward compat) ────────────────────────────────────────
router_legacy = APIRouter(tags=["Legacy Scoring"])


# ═══════════════════════════════════════════════════════════════════════════════
# /v1/score — Single customer scoring
# ═══════════════════════════════════════════════════════════════════════════════


@router_v1.post("/score", response_model=ScoreResponse)
async def score_application(
    bank_statement: UploadFile = File(None, description="Bank statement CSV"),
    documents: list[UploadFile] = File(None, description="PDF documents"),
    applicant_id: str = Form("anonymous"),
    customer_type: str = Form("INDIVIDUAL"),
):
    """
    Submit a customer for credit scoring.

    ALWAYS runs the full pipeline A1→A2→A3→A4.
    Results saved to data/output/. Falls back to data/mock/ on error.
    """
    try:
        result = run_pipeline_for_customer(applicant_id)
        formatted = format_score_response(result, applicant_id)
        return ScoreResponse(**formatted)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline error for {applicant_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# /v1/score/batch — Parallel batch scoring (like test_batch_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════


class BatchRequest(BaseModel):
    customer_ids: list[str]
    stagger_delay: float = BATCH_STAGGER_DELAY


@router_v1.post("/score/batch")
async def score_batch(req: BatchRequest):
    """
    Run pipelines for multiple customers in PARALLEL with staggered starts.

    Similar to test_batch_pipeline.py but triggered via HTTP.
    All pipelines run concurrently using ThreadPoolExecutor.

    Returns results for ALL customers (always succeeds per-customer via fallback).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    customer_ids = req.customer_ids
    stagger = req.stagger_delay
    total = len(customer_ids)

    if total == 0:
        return {"results": {}, "total": 0, "success_count": 0, "duration_s": 0}

    logger.info(f"[Batch] START — {total} customers, stagger={stagger}s")
    batch_start = time.time()

    results: dict[str, dict] = {}
    futures = {}

    def _run_one(cid: str, idx: int) -> tuple[str, dict]:
        """Run pipeline for one customer (in thread)."""
        try:
            result = run_pipeline_for_customer(cid)
            formatted = format_score_response(result, cid)
            return cid, {**formatted, "status": "SUCCESS"}
        except Exception as e:
            logger.error(f"[Batch] Pipeline failed for {cid}: {e}")
            return cid, {
                "application_id": cid,
                "credit_score": 0,
                "pd_pct": 0.0,
                "risk_band": "ERR",
                "recommendation": "ERROR",
                "overall_confidence": 0.0,
                "four_c_scores": {},
                "warnings": [str(e)],
                "report": {},
                "status": "FAILED",
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=BATCH_MAX_WORKERS) as executor:
        for idx, cid in enumerate(customer_ids):
            future = executor.submit(_run_one, cid, idx)
            futures[future] = cid
            logger.info(f"[Batch] Launched {idx + 1}/{total}: customer {cid}")

            # Stagger: wait before launching next (except last)
            if idx < total - 1 and stagger > 0:
                time.sleep(stagger)

        # Collect results as they complete
        for future in as_completed(futures):
            cid, result = future.result()
            results[cid] = result
            status = result.get("status", "UNKNOWN")
            score = result.get("credit_score", "—")
            logger.info(f"[Batch] Completed: {cid} → {status} (score={score})")

    batch_duration = time.time() - batch_start
    success_count = sum(1 for r in results.values() if r.get("status") == "SUCCESS")

    logger.info(
        f"[Batch] DONE — {success_count}/{total} succeeded in {batch_duration:.1f}s"
    )

    return {
        "results": results,
        "total": total,
        "success_count": success_count,
        "failed_count": total - success_count,
        "duration_s": round(batch_duration, 1),
    }



# ═══════════════════════════════════════════════════════════════════════════════
# Legacy /score/* endpoints — kept for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════


@router_legacy.post("/score/mock", response_model=ScoringResult)
async def score_mock_customer(customer_id: str = Form(...)):
    """Score a pre-built demo customer by ID (e.g. '001', '002')."""
    try:
        result = run_pipeline_for_customer(customer_id)
        formatted = format_legacy_result(result)
        return ScoringResult(**formatted)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline error for customer {customer_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router_legacy.post("/score/customer-folder", response_model=ScoringResult)
async def score_customer_folder(customer_folder: str = Form(...)):
    """Score a customer from a pre-existing data folder path."""
    folder = Path(customer_folder)
    if not folder.exists():
        raise HTTPException(
            status_code=404, detail=f"Customer folder not found: {customer_folder}"
        )
    try:
        # Extract customer_id from folder name
        folder_name = folder.name
        customer_id = folder_name.replace("customer_", "")
        result = run_pipeline_for_customer(customer_id)
        formatted = format_legacy_result(result)
        return ScoringResult(**formatted)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router_legacy.post("/score/upload", response_model=ScoringResult)
async def score_upload(
    files: list[UploadFile] = File(...),
    sk_id_curr: int = Form(default=100002),
):
    """Score a customer by uploading files (PDF, CSV, JSON)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            file_path = Path(tmpdir) / f.filename
            content = await f.read()
            file_path.write_bytes(content)
        try:
            agents = get_agents()
            a1_output = agents["a1"].ingest(tmpdir)
            a2_output = agents["a2"].process(a1_output)
            a3_output = agents["a3"].score(a2_output)
            a4_output = agents["a4"].generate(a3_output, a2_output, a1_output)

            result = {
                "a3_output": a3_output,
                "a4_output": a4_output,
            }
            formatted = format_legacy_result(result)
            return ScoringResult(**formatted)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@router_legacy.post("/score/json")
async def score_json(data: dict[str, Any]):
    """Score from raw pre-formatted JSON (skips A1 ingestion)."""
    if "application_row" not in data:
        raise HTTPException(
            status_code=400, detail="Missing 'application_row' in request body"
        )
    try:
        agents = get_agents()
        a2_output = agents["a2"].process(data)
        a3_output = agents["a3"].score(a2_output)
        a4_output = agents["a4"].generate(a3_output, a2_output, data)

        result = {"a3_output": a3_output, "a4_output": a4_output}
        return format_legacy_result(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
