"""
CreditLens — LLM Service (Gemini API).

# LOCAL_SUB: Replaces Amazon Bedrock (Claude 3.5 Sonnet).
# Production: Replace with bedrock.invoke_model() calls.
# See LOCAL_SUBSTITUTIONS.md for migration guide.

Unified LLM interface used by A2 (Feature Engineer) and A4 (Report Generator).
Provides structured JSON extraction with retry and fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized client (google.genai SDK)
_client = None
_MODEL_ID = "gemini-2.5-flash-lite"


def _get_client():
    """Lazy-initialize google.genai Client."""
    global _client
    if _client is None:
        from google import genai  # new SDK: pip install google-genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it in .env or export GEMINI_API_KEY=your_key"
            )
        _client = genai.Client(api_key=api_key)
        logger.info(f"Gemini client initialized (google.genai), model={_MODEL_ID}")
    return _client


class LLMService:
    """Unified LLM service using Google Gemini API.

    # LOCAL_SUB: Replace with BedrockLLMService for production.

    Usage:
        llm = LLMService()
        result = llm.generate_json(system_prompt, user_prompt, required_keys)
        text = llm.generate_text(system_prompt, user_prompt)
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: set[str] | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Generate structured JSON output from LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            required_keys: Expected keys in the JSON response.
            max_tokens: Max response tokens.

        Returns:
            Parsed JSON dict, with missing keys filled as None.
        """
        if self.use_mock:
            return {key: None for key in (required_keys or set())}

        response_text = self.generate_text(system_prompt, user_prompt, max_tokens)
        return self._parse_json(response_text, required_keys or set())

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Generate free-form text from LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            max_tokens: Max response tokens.

        Returns:
            Generated text string.
        """
        if self.use_mock:
            return "[Mock LLM response]"

        from google.genai import types
        client = _get_client()

        # New SDK: combine system + user into one prompt
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        try:
            response = client.models.generate_content(
                model=_MODEL_ID,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                ),
            )
            text = response.text
            logger.debug(f"LLM response ({len(text)} chars)")
            return text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    def _parse_json(self, text: str, required_keys: set[str]) -> dict[str, Any]:
        """Parse JSON from LLM response, with fallback regex extraction."""
        # Try direct parse
        try:
            result = json.loads(text)
            return self._fill_missing_keys(result, required_keys)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return self._fill_missing_keys(result, required_keys)
            except json.JSONDecodeError:
                pass

        # Try extracting bare JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return self._fill_missing_keys(result, required_keys)
            except json.JSONDecodeError:
                pass

        logger.error(f"Failed to parse JSON from LLM response: {text[:200]}")
        return {key: None for key in required_keys}

    @staticmethod
    def _fill_missing_keys(result: dict, required_keys: set[str]) -> dict[str, Any]:
        """Fill missing required keys with None."""
        for key in required_keys:
            if key not in result:
                logger.warning(f"Missing key in LLM JSON: {key}")
                result[key] = None
        return result
