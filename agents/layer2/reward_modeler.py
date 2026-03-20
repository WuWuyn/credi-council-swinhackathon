"""Reward Modeler Agent - Layer 2: Multidimensional Assessment."""

from agents.base import BaseAgent
from prompts.templates import PROMPTS


class RewardModelerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "Reward Modeler"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["reward_modeler"]
