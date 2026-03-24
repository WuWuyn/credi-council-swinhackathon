"""
CREDICOUNCIL FastAPI Application — Main Entry Point.

REST API for the CREDICOUNCIL credit scoring pipeline.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(
    title="credicouncil AI API",
    description="Credit Scoring & Creditworthiness Assessment for Underbanked & Micro SMEs",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "credicouncil-api", "version": "0.1.0"}


# Import routes
from api.routes.score import router as score_router
from api.routes.report import router as report_router
app.include_router(score_router, prefix="/v1")
app.include_router(report_router, prefix="/v1")
