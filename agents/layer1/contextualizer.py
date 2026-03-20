"""Contextualizer Agent - Layer 1: Data Ingestion & Contextualization."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class ContextualizerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Contextualizer"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["contextualizer"]
