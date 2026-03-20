"""
CreditLens A3 — Main Scoring Agent.

Orchestrates the ML scoring pipeline: model prediction → score mapping →
SHAP explanation → decision rules. This is the LangGraph node for A3.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from creditlens.agents.a3_scoring.model import CreditLensModel
from creditlens.agents.a3_scoring.score_mapper import map_prediction
from creditlens.agents.a3_scoring.shap_explainer import SHAPExplainer
from creditlens.agents.a3_scoring.decision_rules import apply_hard_overrides
from creditlens.state.credit_state import CreditState

logger = logging.getLogger(__name__)


class ScoringAgent:
    """Agent A3 — ML Scoring Engine.

    Deterministic agent: same input ALWAYS gives same output.
    No LLM, no randomness — pure mathematics.

    Pipeline:
        1. Receive unified feature vector (from A1 + A2)
        2. LightGBM predict_proba → PD percentage
        3. Map PD → credit score (300-850) + risk band
        4. SHAP TreeExplainer → feature attribution JSON
        5. Apply hard override decision rules
        6. Output: credit_score, pd_pct, risk_band, shap_values, routing
    """

    def __init__(self, model: CreditLensModel):
        """Initialize scoring agent.

        Args:
            model: Trained CreditLensModel instance.
        """
        self.model = model
        self.shap_explainer = SHAPExplainer(
            model=model.model,
            feature_names=model.feature_names,
        )

    def score(self, state: CreditState) -> dict[str, Any]:
        """Score an applicant — LangGraph node function.

        Args:
            state: Current pipeline state with structured_feats and llm_feats.

        Returns:
            State update dict with A3 outputs.
        """
        logger.info(f"A3 Scoring — Application {state.get('application_id', 'unknown')}")

        # 1. Build feature vector from state
        features = self._build_feature_vector(state)

        # 2. Model prediction
        pd_prob = float(self.model.predict_proba(features)[0])

        # 3. Score mapping
        score_result = map_prediction(pd_prob)
        credit_score = score_result["credit_score"]
        risk_band = score_result["risk_band"]

        # 4. SHAP explanation
        shap_output = self.shap_explainer.explain(
            X=features,
            credit_score=credit_score,
            pd_pct=score_result["pd_pct"],
            risk_band=risk_band,
            model_version=self.model.model_version,
        )

        # 5. Decision rules
        decision = apply_hard_overrides(
            credit_score=credit_score,
            risk_band=risk_band,
            auto_decision=score_result["auto_decision"],
            structured_feats=state.get("structured_feats", {}),
            overall_confidence=state.get("overall_confidence", 0.5),
        )

        # 6. Audit entry
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A3",
            "action": "ml_scoring",
            "input_summary": {
                "n_features": features.shape[1],
                "feature_names": list(features.columns),
            },
            "output_summary": {
                "credit_score": credit_score,
                "pd_pct": score_result["pd_pct"],
                "risk_band": risk_band,
                "decision": decision["final_decision"],
            },
            "model_version": self.model.model_version,
            "confidence": state.get("overall_confidence"),
        }

        return {
            "credit_score": credit_score,
            "pd_pct": score_result["pd_pct"],
            "risk_band": risk_band,
            "shap_values": shap_output,
            "routing": decision["final_decision"],
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
        }

    def _build_feature_vector(self, state: CreditState) -> pd.DataFrame:
        """Build feature DataFrame from pipeline state.

        Merges structured_feats (A1) and llm_feats (A2) into a single
        row DataFrame matching the model's expected feature names.
        """
        # Combine A1 + A2 features
        all_feats = {}
        all_feats.update(state.get("structured_feats", {}))
        all_feats.update(state.get("llm_feats", {}))

        # Map to model feature names
        feature_row = {}
        for feat_name in self.model.feature_names:
            if feat_name in all_feats:
                feature_row[feat_name] = all_feats[feat_name]
            else:
                feature_row[feat_name] = 0  # default for missing features

        return pd.DataFrame([feature_row])
