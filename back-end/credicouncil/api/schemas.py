"""
CREDICOUNCIL API — Pydantic Response/Request Schemas.

Defines all request and response models used by the API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool


class ScoreResponse(BaseModel):
    """Unified credit scoring response (used by /v1/score)."""

    application_id: str
    credit_score: int
    pd_pct: float
    risk_band: str
    recommendation: str
    overall_confidence: float
    four_c_scores: dict[str, float]
    warnings: list[str]
    report: dict[str, Any]


class ScoringResult(BaseModel):
    """Full pipeline result (used by legacy /score/* endpoints)."""

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


# ── Human-in-the-Loop (2-Phase Pipeline) ─────────────────────────────────────


class ExtractedFieldInfo(BaseModel):
    """A single extracted field with its value, confidence, and metadata."""

    value: Any = None
    confidence: float = 0.0
    source_document: str = ""
    label_vi: str = ""
    field_type: str = "text"  # text, number, date, boolean, enum


class IngestionResponse(BaseModel):
    """Response from Phase 1 (A1 Ingestion only).

    Returns extracted features + confidence for human review.
    """

    application_id: str
    customer_id: str
    application_row: dict[str, Any]
    confidence_map: dict[str, float]
    identity_consistency_flag: str
    thin_file_flag: bool
    raw_texts: dict[str, str]
    field_metadata: list[dict[str, Any]]  # grouped field info for UI
    warnings: list[str]


class ProcessRequest(BaseModel):
    """Request for Phase 2 — submit approved/edited data.

    After human review, the frontend sends back the (possibly edited)
    application_row to continue the pipeline (A2→A3→A4).
    """

    customer_id: str
    application_row: dict[str, Any]
    # Original A1 output fields needed by A2
    raw_texts: dict[str, str] = {}
    thin_file_flag: bool = False
    identity_consistency_flag: str = "OK"
