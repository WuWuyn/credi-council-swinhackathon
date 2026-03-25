"""
CREDICOUNCIL — LLM Service (Gemini API).

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
import time
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized client (google.genai SDK)
_client = None
_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _get_client():
    """Lazy-initialize google.genai Client."""
    global _client
    if _client is None:
        # Ensure .env is loaded
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            search = Path(__file__).resolve().parent
            for _ in range(6):
                candidate = search / ".env"
                if candidate.exists():
                    load_dotenv(candidate, override=False)
                    break
                search = search.parent
        except ImportError:
            pass

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

    # Retry config
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds (exponential: 1s, 2s, 4s)

    def __init__(self):
        pass

    @staticmethod
    def _call_with_retry(fn, max_retries=3, base_delay=1.0):
        """Call fn() with exponential backoff on rate-limit / transient errors.

        Retries on:
          - 429 Resource Exhausted (rate limit)
          - 503 Service Unavailable
          - Connection/timeout errors
        """
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(kw in err_str for kw in [
                    "429", "resource_exhausted", "rate",
                    "503", "unavailable", "deadline",
                    "timeout", "connection",
                ])
                if not is_retryable or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Gemini API error (attempt {attempt+1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                last_exc = e
        raise last_exc  # should not reach here

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: set[str] | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Generate structured JSON output from LLM.

        Uses response_mime_type='application/json' to force the model
        to return clean JSON without markdown fences.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            required_keys: Expected keys in the JSON response.
            max_tokens: Max response tokens.

        Returns:
            Parsed JSON dict, with missing keys filled as None.
        """
        from google.genai import types
        client = _get_client()

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        try:
            def _call():
                return client.models.generate_content(
                    model=_MODEL_ID,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
            response = self._call_with_retry(_call, self.MAX_RETRIES, self.RETRY_BASE_DELAY)
            response_text = response.text
            logger.debug(f"LLM JSON response ({len(response_text)} chars)")
        except Exception as e:
            logger.error(f"Gemini API error (JSON mode): {e}")
            # Fallback to plain text mode
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


        from google.genai import types
        client = _get_client()

        # New SDK: combine system + user into one prompt
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        try:
            def _call():
                return client.models.generate_content(
                    model=_MODEL_ID,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.2,
                    ),
                )
            response = self._call_with_retry(_call, self.MAX_RETRIES, self.RETRY_BASE_DELAY)
            text = response.text
            logger.debug(f"LLM response ({len(text)} chars)")
            return text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_class: type,
        max_tokens: int = 8192,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Generate structured output validated by a Pydantic schema.

        Uses Gemini response_schema for type-safe JSON output.

        Args:
            system_prompt: System instructions.
            user_prompt: User prompt with context.
            schema_class: Pydantic BaseModel class for response validation.
            max_tokens: Max response tokens.
            temperature: Sampling temperature.

        Returns:
            Validated dict from Pydantic model_dump().
        """
        from google.genai import types
        client = _get_client()

        try:
            def _call():
                return client.models.generate_content(
                    model=_MODEL_ID,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema_class,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
            response = self._call_with_retry(_call, self.MAX_RETRIES, self.RETRY_BASE_DELAY)
            result = schema_class.model_validate_json(response.text)
            return result.model_dump()

        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}, trying fallback...")
            # Fallback: plain JSON without schema constraint
            try:
                response = client.models.generate_content(
                    model=_MODEL_ID,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                raw = json.loads(response.text)
                result = schema_class.model_validate(raw)
                return result.model_dump()
            except Exception as e2:
                logger.error(f"Fallback structured extraction also failed: {e2}")
                return schema_class().model_dump()

    async def generate_structured_async(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_class: type,
        max_tokens: int = 8192,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Async wrapper for generate_structured().

        Uses asyncio.to_thread to run the blocking Gemini API call
        in a thread pool, allowing concurrent LLM calls.
        """
        import asyncio
        return await asyncio.to_thread(
            self.generate_structured,
            system_prompt, user_prompt, schema_class, max_tokens, temperature,
        )

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

        # Try extracting bare JSON object (greedy — largest match)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return self._fill_missing_keys(result, required_keys)
            except json.JSONDecodeError:
                pass

        # Try fixing truncated JSON — close any open braces/brackets
        json_match = re.search(r"\{.*", text, re.DOTALL)
        if json_match:
            truncated = json_match.group()
            fixed = self._try_fix_truncated_json(truncated)
            if fixed:
                return self._fill_missing_keys(fixed, required_keys)

        logger.error(f"Failed to parse JSON from LLM response: {text[:200]}")
        return {key: None for key in required_keys}

    @staticmethod
    def _try_fix_truncated_json(text: str) -> dict | None:
        """Attempt to fix truncated JSON by closing open braces/brackets."""
        # Count unclosed braces and brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')

        # Remove trailing comma if present
        fixed = text.rstrip().rstrip(',')

        # Close open structures
        fixed += ']' * max(0, open_brackets)
        fixed += '}' * max(0, open_braces)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fill_missing_keys(result: dict, required_keys: set[str]) -> dict[str, Any]:
        """Fill missing required keys with None."""
        for key in required_keys:
            if key not in result:
                logger.warning(f"Missing key in LLM JSON: {key}")
                result[key] = None
        return result
