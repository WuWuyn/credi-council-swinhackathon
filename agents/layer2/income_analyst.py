"""Income & Stability Analyst Agent - Layer 2: Multidimensional Assessment."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class IncomeAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Income & Stability Analyst"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["income_analyst"]
