"""
CreditLens A3 — ML Scoring Agent (Local Version).

Deterministic agent: same input ALWAYS gives same output.
No LLM, no randomness — pure mathematics.

Pipeline:
    1. Receive feature vector from A2
    2. LightGBM predict_proba → PD probability
    3. Map PD → credit score (300-850) + risk band
    4. SHAP TreeExplainer → top feature attributions
    5. Apply hard override decision rules
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from creditlens.agents.a3_scoring.model import CreditLensModel
from creditlens.agents.a3_scoring.score_mapper import map_prediction
from creditlens.agents.a3_scoring.decision_rules import apply_hard_overrides

logger = logging.getLogger(__name__)


def _safe_value(val):
    """Convert numpy/pandas value to JSON-serializable Python type."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


class ScoringAgent:
    """Agent A3 — ML Scoring Engine.

    Deterministic agent: produces credit score, PD%, SHAP explanations.
    """

    def __init__(self, model_path: str | Path = "models/lgbm_ref_v1.pkl"):
        """Initialize with trained model.

        Args:
            model_path: Path to saved model pickle.
        """
        self.model = CreditLensModel()
        self.model.load(str(model_path))
        logger.info(f"A3 Model loaded: {model_path}")
        logger.info(f"  Feature names: {len(self.model.feature_names)} features")
        self._shap_explainer = None

    def score(self, a2_output: dict[str, Any]) -> dict[str, Any]:
        """Score an applicant.

        Args:
            a2_output: Output from A2 FeatureEngineerAgent.process()

        Returns:
            Dict with credit_score, pd_pct, risk_band, shap_top, decision
        """
        logger.info("="*60)
        logger.info("  A3 ML Scoring Engine")
        logger.info("="*60)

        feature_vector = a2_output.get("feature_vector")
        application_row = a2_output.get("application_row", {})

        # Build feature DataFrame
        features_df = self._build_feature_df(feature_vector, application_row)
        logger.info(f"  Feature vector: {features_df.shape[1]} features")

        # 1. Predict
        pd_prob = float(self.model.predict_proba(features_df)[0])
        logger.info(f"  PD probability: {pd_prob:.4f}")

        # 2. Score mapping
        score_result = map_prediction(pd_prob)
        credit_score = score_result["credit_score"]
        risk_band = score_result["risk_band"]
        logger.info(f"  Credit score: {credit_score}")
        logger.info(f"  Risk band: {risk_band}")

        # 3. SHAP explanation
        shap_output = self._compute_shap(features_df, credit_score, pd_prob, risk_band)

        # 4. Decision rules
        decision = apply_hard_overrides(
            credit_score=credit_score,
            risk_band=risk_band,
            auto_decision=score_result.get("auto_decision", "REVIEW"),
            structured_feats=application_row,
            overall_confidence=a2_output.get("llm_feats", {}).get("imputation_confidence", 1.0),
        )
        logger.info(f"  Decision: {decision.get('final_decision', 'REVIEW')}")

        # 5. Audit
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A3",
            "action": "ml_scoring",
            "input_summary": {"n_features": features_df.shape[1]},
            "output_summary": {
                "credit_score": credit_score,
                "pd_pct": score_result["pd_pct"],
                "risk_band": risk_band,
                "decision": decision.get("final_decision"),
            },
            "model_version": self.model.model_version,
        }

        return {
            "credit_score": credit_score,
            "pd_pct": score_result["pd_pct"],
            "pd_prob": pd_prob,
            "risk_band": risk_band,
            "shap_values": shap_output,
            "routing": decision.get("final_decision", "REVIEW"),
            "decision_details": decision,
            "features_df": features_df,
            "audit_trail": a2_output.get("audit_trail", []) + [audit_entry],
        }

    def _build_feature_df(
        self,
        feature_vector: pd.Series | None,
        application_row: dict,
    ) -> pd.DataFrame:
        """Build feature DataFrame matching model's expected features.

        Uses feature_vector from A2 if available, else builds from application_row.
        """
        if feature_vector is not None and isinstance(feature_vector, pd.Series):
            # Use A2's feature-engineered vector
            row = {}
            for feat in self.model.feature_names:
                if feat in feature_vector.index:
                    row[feat] = feature_vector[feat]
                else:
                    row[feat] = 0.0  # default for missing
            return pd.DataFrame([row])

        # Fallback: build from raw application_row
        row = {}
        for feat in self.model.feature_names:
            if feat in application_row:
                val = application_row[feat]
                # Convert strings to numeric where possible
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.0
                row[feat] = val if val is not None else 0.0
            else:
                row[feat] = 0.0

        return pd.DataFrame([row])

    def _compute_shap(
        self,
        features_df: pd.DataFrame,
        credit_score: int,
        pd_prob: float,
        risk_band: str,
    ) -> dict[str, Any]:
        """Compute SHAP values for the prediction.

        Returns top positive and negative factors.
        """
        try:
            import shap
            from creditlens.config.feature_config import get_label_vi, get_5c_dimension

            if self._shap_explainer is None:
                # Extract inner LightGBM model from BaggingClassifier
                model_obj = self.model.model
                if isinstance(model_obj, list):
                    # model is list of BaggingClassifiers, get first inner estimator
                    bag = model_obj[0]
                    if hasattr(bag, 'estimators_') and len(bag.estimators_) > 0:
                        model_obj = bag.estimators_[0]
                    else:
                        model_obj = bag
                self._shap_explainer = shap.TreeExplainer(model_obj)

            shap_values = self._shap_explainer.shap_values(features_df)

            # For binary classification, use class 1 (positive = default)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            # Build attribution dict with label_vi and value
            feature_names = features_df.columns.tolist()
            feature_values = features_df.iloc[0]
            attributions = sorted(
                zip(feature_names, sv),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            top_positive = [
                {
                    "feature": name,
                    "shap_value": round(float(val), 4),
                    "value": _safe_value(feature_values.get(name)),
                    "label_vi": get_label_vi(name),
                    "dimension_5c": get_5c_dimension(name),
                    "direction": "positive_for_default",
                }
                for name, val in attributions if val > 0
            ][:10]

            top_negative = [
                {
                    "feature": name,
                    "shap_value": round(float(val), 4),
                    "value": _safe_value(feature_values.get(name)),
                    "label_vi": get_label_vi(name),
                    "dimension_5c": get_5c_dimension(name),
                    "direction": "negative_for_default",
                }
                for name, val in attributions if val < 0
            ][:10]

            # 5C SHAP allocation
            five_c_shap = {"character": 0.0, "capacity": 0.0, "capital": 0.0,
                           "conditions": 0.0, "collateral": 0.0}
            for name, val in zip(feature_names, sv):
                dim = get_5c_dimension(name)
                five_c_shap[dim] += abs(float(val))
            total_shap = sum(five_c_shap.values()) or 1.0
            five_c_allocation = {
                dim: {"shap_sum": round(v, 4), "pct": round(v / total_shap * 100)}
                for dim, v in five_c_shap.items()
            }

            logger.info(f"  SHAP computed: {len(top_positive)} pos, {len(top_negative)} neg factors")

            return {
                "credit_score": credit_score,
                "pd_prob": pd_prob,
                "risk_band": risk_band,
                "top_positive_factors": top_positive,
                "top_negative_factors": top_negative,
                "five_c_shap_allocation": five_c_allocation,
                "model_version": self.model.model_version,
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except ImportError:
            logger.warning("SHAP not available — skipping explanation")
            return self._mock_shap(credit_score, pd_prob, risk_band)

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return self._mock_shap(credit_score, pd_prob, risk_band)

    def _mock_shap(self, credit_score: int, pd_prob: float, risk_band: str) -> dict[str, Any]:
        """Fallback SHAP output when SHAP is unavailable."""
        from creditlens.config.feature_config import get_label_vi, get_5c_dimension

        return {
            "credit_score": credit_score,
            "pd_prob": pd_prob,
            "risk_band": risk_band,
            "top_positive_factors": [
                {
                    "feature": "EXT_SOURCE_2", "shap_value": 0.15,
                    "value": 0.89, "label_vi": get_label_vi("EXT_SOURCE_2"),
                    "dimension_5c": get_5c_dimension("EXT_SOURCE_2"),
                    "direction": "negative_for_default",
                },
                {
                    "feature": "EXT_SOURCE_3", "shap_value": 0.12,
                    "value": 0.75, "label_vi": get_label_vi("EXT_SOURCE_3"),
                    "dimension_5c": get_5c_dimension("EXT_SOURCE_3"),
                    "direction": "negative_for_default",
                },
            ],
            "top_negative_factors": [
                {
                    "feature": "DAYS_BIRTH", "shap_value": -0.08,
                    "value": -12000, "label_vi": get_label_vi("DAYS_BIRTH"),
                    "dimension_5c": get_5c_dimension("DAYS_BIRTH"),
                    "direction": "positive_for_default",
                },
            ],
            "five_c_shap_allocation": {
                "character": {"shap_sum": 0.27, "pct": 45},
                "capacity": {"shap_sum": 0.18, "pct": 30},
                "capital": {"shap_sum": 0.06, "pct": 10},
                "conditions": {"shap_sum": 0.06, "pct": 10},
                "collateral": {"shap_sum": 0.03, "pct": 5},
            },
            "model_version": self.model.model_version,
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "SHAP unavailable — using mock values",
        }
