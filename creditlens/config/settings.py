"""
CreditLens Settings — Environment-based configuration.

Uses pydantic-settings to load from .env file or environment variables.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "home-credit-default-risk"
MODELS_DIR = PROJECT_ROOT / "models"
POLICY_DOCS_DIR = PROJECT_ROOT / "policy_docs"


# ─── Settings class ──────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS General ──
    aws_region: str = "ap-southeast-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # ── Amazon Bedrock (Claude) ──
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_max_tokens: int = 4096

    # ── Amazon Textract ──
    textract_bucket: str = "creditlens-documents"

    # ── SageMaker ──
    sagemaker_endpoint_name: str = "creditlens-lgbm-endpoint"

    # ── DynamoDB ──
    dynamodb_table_state: str = "creditlens-state"
    dynamodb_table_audit: str = "creditlens-audit"

    # ── OpenSearch (RAG) ──
    opensearch_endpoint: str = ""
    opensearch_index: str = "policy-docs"

    # ── S3 ──
    s3_bucket_documents: str = "creditlens-documents"
    s3_bucket_models: str = "creditlens-models"

    # ── Application ──
    app_env: str = "development"
    log_level: str = "INFO"
    model_path: str = "models/lgbm_ref_v1.pkl"
    data_dir: str = "home-credit-default-risk"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def model_full_path(self) -> Path:
        return PROJECT_ROOT / self.model_path

    @property
    def data_full_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
