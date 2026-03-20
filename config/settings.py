"""
Configuration module for MASCA system.
Loads settings from environment variables / .env file.
All LLM model settings are configurable - no hardcoded model names.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class LLMConfig:
    """Configuration for the LLM Provider."""

    # Provider: "google", "openrouter", or "cliproxy"
    provider: str = os.getenv("LLM_PROVIDER", "cliproxy").lower()

    # Google Gemini Config
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # OpenRouter Config
    openrouter_model_name: str = os.getenv("OPENROUTER_MODEL_NAME", "stepfun/step-3.5-flash:free")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # CLIProxy Local Config
    # CLIProxy exposes an OpenAI-compatible API at localhost:8317 by default.
    # Set the api-key to match one entry in the proxy's 'api-keys' list (config.yaml).
    cliproxy_model_name: str = os.getenv("CLIPROXY_MODEL_NAME", "gemini-2.5-flash")
    cliproxy_api_key: str = os.getenv("CLIPROXY_API_KEY", "")
    cliproxy_base_url: str = os.getenv("CLIPROXY_BASE_URL", "http://localhost:8765/v1")

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.95"))
    top_k: int = int(os.getenv("LLM_TOP_K", "40"))

    def validate(self) -> None:
        """Validate that required configuration is present."""
        if self.provider == "google" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER is google.")
        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER is openrouter.")
        if self.provider == "cliproxy" and not self.cliproxy_api_key:
            raise ValueError("CLIPROXY_API_KEY is required when LLM_PROVIDER is cliproxy.")


@dataclass
class AppConfig:
    """Application-level configuration."""

    dataset_path: str = os.getenv("DATASET_PATH", str(_PROJECT_ROOT / "home-credit-default-risk" / "application_train.csv"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_concurrent_agents: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "4"))


def get_llm_config() -> LLMConfig:
    """Factory function to create a validated LLMConfig."""
    config = LLMConfig()
    config.validate()
    return config


def get_app_config() -> AppConfig:
    """Factory function to create AppConfig."""
    return AppConfig()
