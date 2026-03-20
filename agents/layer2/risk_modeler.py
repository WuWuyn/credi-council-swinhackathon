"""Risk Modeler Agent - Layer 2: Multidimensional Assessment."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class RiskModelerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Risk Modeler"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["risk_modeler"]
