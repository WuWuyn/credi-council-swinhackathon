import json
import logging
import re
from abc import ABC, abstractmethod

from config.settings import LLMConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all MASCA agents."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        
        if self.provider == "google":
            from google import genai
            self.client = genai.Client(api_key=config.gemini_api_key)
        elif self.provider == "openrouter":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        elif self.provider == "cliproxy":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.cliproxy_api_key,
                base_url=config.cliproxy_base_url
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'google', 'openrouter', or 'cliproxy'.")

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent display name."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's role and responsibilities."""
        ...

    def invoke(self, user_input: str) -> dict:
        """
        Send input to LLM and parse the JSON response.

        Args:
            user_input: The formatted input data for this agent.

        Returns:
            Parsed JSON dict from the agent's response with token usage metadata.
        """
        logger.info(f"[{self.name}] Invoking agent (Provider: {self.provider})...")

        try:
            token_usage = {}
            raw_text = ""

            if self.provider == "google":
                from google.genai import types
                
                # Configure generation parameters including required JSON format
                generation_config = types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                )

                # Invoke the Gemini model
                response = self.client.models.generate_content(
                    model=self.config.gemini_model_name,
                    contents=user_input,
                    config=generation_config,
                )

                raw_text = response.text or ""
                
                # Extract token usage from response metadata
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    token_usage = {
                        "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                        "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                        "total_tokens": getattr(usage, 'total_token_count', 0),
                    }
            
            elif self.provider in ("openrouter", "cliproxy"):
                # Both OpenRouter and CLIProxy use the OpenAI SDK compatible API
                model_name = (
                    self.config.openrouter_model_name
                    if self.provider == "openrouter"
                    else self.config.cliproxy_model_name
                )
                # Invoke the model using OpenAI-compatible SDK
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt + "\n\nYOU MUST RETURN A VALID JSON OBJECT."},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    top_p=self.config.top_p,
                    response_format={"type": "json_object"}
                )

                raw_text = response.choices[0].message.content or ""
                
                # Extract token usage from response metadata
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    token_usage = {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(usage, 'completion_tokens', 0),
                        "total_tokens": getattr(usage, 'total_tokens', 0),
                    }

            logger.debug(f"[{self.name}] Raw response: {raw_text[:500]}...")

            result = self._parse_json(raw_text)
            if not isinstance(result, dict):
                result = {"raw_response": str(result), "error": "Parsed result was not a dictionary"}

            # Add metadata to result
            result["_metadata"] = {
                "agent": self.name,
                "token_usage": token_usage
            }

            logger.info(f"[{self.name}] Successfully parsed response. Tokens: {token_usage.get('total_tokens', 0)}")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return {"error": str(e), "agent": self.name, "_metadata": {"agent": self.name, "token_usage": {}}}

    def _parse_json(self, text: str) -> dict:
        """
        Parse JSON from the agent's response text.
        Handles cases where the response may be wrapped in markdown code blocks
        or contains preamble/postamble strings.
        """
        if not text or text.strip().lower() == "null":
            return {"raw_response": str(text), "error": "Empty or null response from LLM"}

        # 1. Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2. Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 3. Aggressively extract anything between the first { and last }
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Return raw text wrapped in a dict
        logger.warning(f"[{self.name}] Could not parse JSON, returning raw text.")
        return {"raw_response": text, "agent": self.name}
