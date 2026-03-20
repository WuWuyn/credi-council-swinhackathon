"""
MASCA Pipeline Orchestrator.
Runs the 3-layer multi-agent credit assessment pipeline:
  Layer 1: Data Ingestion & Contextualization (parallel)
  Layer 2: Multidimensional Assessment (parallel)
  Layer 3: Strategic Optimization (sequential)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config.settings import LLMConfig
from agents.layer1 import DataAnalystAgent, ContextualizerAgent, FeatureEngineerAgent
from agents.layer2 import (
    RiskModelerAgent,
    IncomeAnalystAgent,
    DebtAnalystAgent,
    RewardModelerAgent,
    MLModelerAgent,
)
from agents.layer3 import RiskRewardOptimizerAgent, DecisionOrchestratorAgent

logger = logging.getLogger(__name__)


class MASCAOrchestrator:
    """
    Orchestrates the full MASCA credit assessment pipeline.

    Execution flow:
    1. Layer 1 agents run in PARALLEL on raw input data.
    2. Layer 2 agents run in PARALLEL on Layer 1 aggregated output.
    3. Layer 3 agents run SEQUENTIALLY:
       a. Risk-Reward Optimizer processes Layer 2 outputs.
       b. Decision Orchestrator makes the final decision.
    """

    def __init__(self, config: LLMConfig, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers

        # Layer 1 agents
        self.data_analyst = DataAnalystAgent(config)
        self.contextualizer = ContextualizerAgent(config)
        self.feature_engineer = FeatureEngineerAgent(config)

        # Layer 2 agents
        self.risk_modeler = RiskModelerAgent(config)
        self.income_analyst = IncomeAnalystAgent(config)
        self.debt_analyst = DebtAnalystAgent(config)
        self.reward_modeler = RewardModelerAgent(config)
        self.ml_modeler = MLModelerAgent(config)  # deterministic ML scoring

        # Layer 3 agents
        self.risk_reward_optimizer = RiskRewardOptimizerAgent(config)
        self.decision_orchestrator = DecisionOrchestratorAgent(config)

    def _run_parallel(
        self, agents_with_inputs: list[tuple], layer_name: str
    ) -> dict[str, dict]:
        """Run multiple agents in parallel and collect results."""
        results = {}
        total_start = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"  {layer_name}")
        logger.info(f"{'='*60}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {}
            for agent, input_text in agents_with_inputs:
                future = executor.submit(agent.invoke, input_text)
                future_to_agent[future] = agent.name

            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                start = time.time()
                try:
                    result = future.result()
                    elapsed = time.time() - start
                    results[agent_name] = result
                    
                    # Log token usage if available
                    tokens = result.get("_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
                    logger.info(f"  ✓ {agent_name} completed ({elapsed:.1f}s, {tokens} tokens)")
                except Exception as e:
                    logger.error(f"  ✗ {agent_name} failed: {e}")
                    results[agent_name] = {"error": str(e), "_metadata": {"agent": agent_name, "token_usage": {}}}

        total_elapsed = time.time() - total_start
        logger.info(f"  Layer completed in {total_elapsed:.1f}s total")
        return results

    def run(self, raw_input: str) -> dict[str, Any]:
        """
        Execute the full MASCA pipeline.

        Args:
            raw_input: Formatted applicant data string.

        Returns:
            Dict with results from all layers, final decision, and token usage.
        """
        pipeline_start = time.time()
        all_results: dict[str, Any] = {}
        
        # Initialize token tracking
        total_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # ──────────────────────────────────────────────
        # Layer 1: Data Ingestion & Contextualization
        # ──────────────────────────────────────────────
        layer1_agents = [
            (self.data_analyst, raw_input),
            (self.contextualizer, raw_input),
            (self.feature_engineer, raw_input),
        ]
        layer1_results = self._run_parallel(
            layer1_agents, "LAYER 1: Data Ingestion & Contextualization"
        )
        all_results["layer1"] = layer1_results
        
        # Aggregate Layer 1 token usage
        for agent_result in layer1_results.values():
            if not isinstance(agent_result, dict):
                agent_result = {}
            usage = agent_result.get("_metadata", {}).get("token_usage", {})
            total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
            total_tokens["total_tokens"] += usage.get("total_tokens", 0)

        # Aggregate Layer 1 outputs for Layer 2
        feature_engineer_output = layer1_results.get('Feature Engineer', {})
        missing_notes = feature_engineer_output.get('_missing_data_notes', '')
        credit_history_context = (
            f"\n--- Credit History Context ---\n{missing_notes}\n"
            if missing_notes
            else "\n--- Credit History Context ---\nNo missing data notes. Applicant has records across all data sources.\n"
        )

        layer1_summary = (
            f"=== Aggregated Layer 1 Results ===\n\n"
            f"{credit_history_context}\n"
            f"--- Data Analyst Output ---\n"
            f"{json.dumps(layer1_results.get('Data Analyst', {}), indent=2)}\n\n"
            f"--- Contextualizer Output ---\n"
            f"{json.dumps(layer1_results.get('Contextualizer', {}), indent=2)}\n\n"
            f"--- Feature Engineer Output ---\n"
            f"{json.dumps(feature_engineer_output, indent=2)}\n\n"
            f"--- Original Application Data ---\n"
            f"{raw_input}"
        )

        # ──────────────────────────────────────────────
        # Layer 2: Multidimensional Assessment
        # ──────────────────────────────────────────────
        # Extract features dict from Feature Engineer for ML Modeler
        feature_engineer_result = layer1_results.get('Feature Engineer', {})
        ml_features = feature_engineer_result.get('features', {})
        if not ml_features:
            # Fallback: pass empty dict; MLModelerAgent will handle gracefully
            ml_features = {}

        layer2_agents = [
            (self.risk_modeler, layer1_summary),
            (self.income_analyst, layer1_summary),
            (self.debt_analyst, layer1_summary),
            (self.reward_modeler, layer1_summary),
        ]
        layer2_results = self._run_parallel(
            layer2_agents, "LAYER 2: Multidimensional Assessment"
        )
        # Run ML Modeler separately (it takes features dict, not text)
        logger.info("  Running ML Modeler (LightGBM inference)...")
        ml_result = self.ml_modeler.invoke(ml_features)
        layer2_results["ML Modeler"] = ml_result
        logger.info(
            f"  ✓ ML Modeler completed — "
            f"P(default)={ml_result.get('default_probability', 'N/A')}, "
            f"score={ml_result.get('ml_credit_score', 'N/A')}"
        )
        all_results["layer2"] = layer2_results
        
        # Aggregate Layer 2 token usage
        for agent_result in layer2_results.values():
            if not isinstance(agent_result, dict):
                agent_result = {}
            usage = agent_result.get("_metadata", {}).get("token_usage", {})
            total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
            total_tokens["total_tokens"] += usage.get("total_tokens", 0)

        # ──────────────────────────────────────────────
        # Layer 3: Strategic Optimization (Sequential)
        # ──────────────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"  LAYER 3: Strategic Optimization")
        logger.info(f"{'='*60}")

        # 3a. Risk-Reward Optimizer
        layer2_summary = (
            f"=== Layer 2 Assessment Results ===\n\n"
            f"--- Risk Modeler ---\n"
            f"{json.dumps(layer2_results.get('Risk Modeler', {}), indent=2)}\n\n"
            f"--- Income & Stability Analyst ---\n"
            f"{json.dumps(layer2_results.get('Income & Stability Analyst', {}), indent=2)}\n\n"
            f"--- Debt Analyst ---\n"
            f"{json.dumps(layer2_results.get('Debt Analyst', {}), indent=2)}\n\n"
            f"--- Reward Modeler ---\n"
            f"{json.dumps(layer2_results.get('Reward Modeler', {}), indent=2)}\n\n"
            f"--- ML Modeler (LightGBM Credit Score) ---\n"
            f"{json.dumps(layer2_results.get('ML Modeler', {}), indent=2)}"
        )

        start = time.time()
        optimizer_result = self.risk_reward_optimizer.invoke(layer2_summary)
        logger.info(
            f"  ✓ Risk-Reward Optimizer completed ({time.time() - start:.1f}s)"
        )
        all_results["risk_reward_optimization"] = optimizer_result
        
        # Track optimizer tokens
        if not isinstance(optimizer_result, dict):
            optimizer_result = {}
        usage = optimizer_result.get("_metadata", {}).get("token_usage", {})
        total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
        total_tokens["total_tokens"] += usage.get("total_tokens", 0)

        # 3b. Decision Orchestrator (final decision)
        final_input = (
            f"=== Complete Assessment Summary ===\n\n"
            f"--- Original Application ---\n{raw_input}\n\n"
            f"--- Layer 1 Results ---\n{json.dumps(layer1_results, indent=2)}\n\n"
            f"--- Layer 2 Results ---\n{json.dumps(layer2_results, indent=2)}\n\n"
            f"--- Risk-Reward Optimization ---\n{json.dumps(optimizer_result, indent=2)}"
        )

        start = time.time()
        decision_result = self.decision_orchestrator.invoke(final_input)
        logger.info(
            f"  ✓ Decision Orchestrator completed ({time.time() - start:.1f}s)"
        )
        all_results["final_decision"] = decision_result
        
        # Track decision orchestrator tokens
        if not isinstance(decision_result, dict):
            decision_result = {}
        usage = decision_result.get("_metadata", {}).get("token_usage", {})
        total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
        total_tokens["total_tokens"] += usage.get("total_tokens", 0)

        total_elapsed = time.time() - pipeline_start
        logger.info(f"\n{'='*60}")
        logger.info(f"  PIPELINE COMPLETE ({total_elapsed:.1f}s total)")
        logger.info(f"  Total Tokens: {total_tokens['total_tokens']} (prompt: {total_tokens['prompt_tokens']}, completion: {total_tokens['completion_tokens']})")
        logger.info(f"{'='*60}")

        all_results["pipeline_time_seconds"] = round(total_elapsed, 2)
        all_results["token_usage"] = total_tokens
        return all_results
