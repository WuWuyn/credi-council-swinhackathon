"""Decision Orchestrator Agent - Layer 3: Strategic Optimization."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class DecisionOrchestratorAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Decision Orchestrator"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["decision_orchestrator"]
