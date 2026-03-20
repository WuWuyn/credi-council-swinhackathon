"""Risk-Reward Optimizer Agent - Layer 3: Strategic Optimization."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class RiskRewardOptimizerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Risk-Reward Optimizer"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["risk_reward_optimizer"]
