"""Debt Analyst Agent - Layer 2: Multidimensional Assessment."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class DebtAnalystAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Debt Analyst"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["debt_analyst"]
