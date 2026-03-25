"""
CREDICOUNCIL API — Configuration & Settings.

Centralized paths, environment loading, and application settings.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

# ─── Path bootstrap ───────────────────────────────────────────────────────────
# back-end/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── .env loading ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    _env_candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
    ]
    for _p in _env_candidates:
        if _p.exists():
            load_dotenv(_p)
            break
    else:
        load_dotenv()
except ImportError:
    pass

# ─── Settings ─────────────────────────────────────────────────────────────────
try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        API_V1_STR: str = "/v1"
        PROJECT_NAME: str = "CrediCouncil AI API"
        PROJECT_DESCRIPTION: str = (
            "Credit Scoring & Creditworthiness Assessment for Underbanked & Micro SMEs"
        )
        VERSION: str = "1.0.0"
        APP_ENV: str = "development"
        LOG_LEVEL: str = "INFO"
        MODEL_PATH: str = "models/lgbm_ref_v1.pkl"
        BACKEND_CORS_ORIGINS: List[str] = ["*"]

        class Config:
            env_file = ".env"
            case_sensitive = True
            extra = "ignore"

    settings = Settings()
except Exception:

    class _FallbackSettings:
        API_V1_STR = "/v1"
        PROJECT_NAME = "CrediCouncil AI API"
        PROJECT_DESCRIPTION = "Credit Scoring Pipeline"
        VERSION = "1.0.0"
        APP_ENV = os.getenv("APP_ENV", "development")
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_ref_v1.pkl")
        BACKEND_CORS_ORIGINS = ["*"]

    settings = _FallbackSettings()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Data directories ────────────────────────────────────────────────────────
MOCK_DIR = PROJECT_ROOT / "data" / "mock"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
