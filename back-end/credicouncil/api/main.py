"""
CREDICOUNCIL FastAPI Application — Main Entry Point.

Assembles routers, middleware, and serves the application.

Usage:
    uvicorn credicouncil.api.main:app --host 0.0.0.0 --port 8000 --reload

Module structure:
    api/
    ├── main.py              ← This file (app assembly)
    ├── config.py            ← Settings, paths, logging
    ├── schemas.py           ← Pydantic request/response models
    ├── data_access.py       ← File-system data layer (mock/ & output/)
    ├── pipeline.py          ← Pipeline execution & agent management
    ├── routes_customers.py  ← GET /v1/customers
    ├── routes_report.py     ← GET /v1/report/{id}/json, /pdf
    └── routes_scoring.py    ← POST /v1/score, /score/mock, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ─── Config (must import first — bootstraps paths, logging, .env) ─────────
from credicouncil.api.config import PROJECT_ROOT, settings
from credicouncil.api.schemas import HealthResponse

# ─── Routers ──────────────────────────────────────────────────────────────
from credicouncil.api.routes_customers import router as customers_router
from credicouncil.api.routes_report import router as report_router
from credicouncil.api.routes_scoring import router_legacy as scoring_legacy_router
from credicouncil.api.routes_scoring import router_v1 as scoring_v1_router
from credicouncil.api.routes_ws import router_ws as ws_router

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include routers ─────────────────────────────────────────────────────────
app.include_router(customers_router)
app.include_router(report_router)
app.include_router(scoring_v1_router)
app.include_router(scoring_legacy_router)
app.include_router(ws_router)

# ─── Serve frontend static files (production build) ──────────────────────────
_FRONTEND_CANDIDATES = [
    PROJECT_ROOT / "front-end" / "app",
    PROJECT_ROOT.parent / "front-end" / "app",
    PROJECT_ROOT / "front-end" / "dist",
    PROJECT_ROOT.parent / "front-end" / "dist",
]
for _fe_dir in _FRONTEND_CANDIDATES:
    if _fe_dir.exists():
        app.mount("/app", StaticFiles(directory=str(_fe_dir), html=True), name="frontend")
        logger.info(f"Frontend served at /app from {_fe_dir}")
        break


# ─── Health check ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check — returns service status and whether the ML model file exists."""
    model_path = settings.MODEL_PATH
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)
    model_loaded = Path(model_path).exists()
    return HealthResponse(
        status="healthy",
        service="credicouncil-api",
        version=settings.VERSION,
        model_loaded=model_loaded,
    )
