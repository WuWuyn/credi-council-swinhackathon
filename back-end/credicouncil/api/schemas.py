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
