"""Data Analyst Agent - Layer 1: Data Ingestion & Contextualization."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class DataAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Data Analyst"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["data_analyst"]
