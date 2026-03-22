"""
CreditLens A3 — SHAP Explainer.

Generates SHAP TreeExplainer output in the standardized JSON schema
defined in the design document (Section 6.5). This is the mathematical
foundation for all explainability in CreditLens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import shap

from creditlens.config.feature_config import FEATURE_TO_4C_MAPPING, SHAP_LABEL_VI, get_5c_dimension

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for LightGBM credit scoring model.

    Produces the standardized SHAP output JSON that serves as the
    bridge between A3 (ML Scoring) and A4 (Report Generator).

    The SHAP output is the ground truth for all explainability —
    A4 must only reference factors that appear in this output.
    """

    def __init__(self, model, feature_names: list[str]):
        """Initialize SHAP explainer.

        Args:
            model: Trained LightGBM model (sklearn-compatible).
            feature_names: List of feature names matching model input.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        logger.info(f"SHAP TreeExplainer initialized with {len(feature_names)} features")

    def explain(
        self,
        X: pd.DataFrame,
        credit_score: int,
        pd_pct: float,
        risk_band: str,
        model_version: str = "lgbm_v1.0_homecredit",
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Generate full SHAP explanation for a single applicant.

        Args:
            X: Single-row DataFrame with features.
            credit_score: Mapped credit score (300-850).
            pd_pct: Probability of default percentage.
            risk_band: Risk band classification.
            model_version: Model version string.
            top_n: Number of top positive/negative factors to include.

        Returns:
            Complete SHAP output dict matching the design spec schema.
        """
        if X.shape[0] != 1:
            raise ValueError(f"Expected single-row DataFrame, got {X.shape[0]} rows")

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 (default) SHAP values
        else:
            sv = shap_values[0]

        # Build feature → SHAP mapping
        feature_shap = dict(zip(self.feature_names, sv))

        # Separate positive and negative factors
        positive_factors = []
        negative_factors = []

        for feat_name, shap_val in sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True):
            # Note: for credit scoring, positive SHAP = increases default risk
            # We invert for credit score: positive = GOOD for borrower
            inverted_shap = -shap_val  # Invert: negative SHAP (reduces default) → positive (good for credit)

            feat_value = X[feat_name].iloc[0] if feat_name in X.columns else None
            label_vi = SHAP_LABEL_VI.get(feat_name, feat_name)

            # Add value info to label
            if isinstance(feat_value, (bool, np.bool_)):
                label_detail = f"{label_vi}"
            elif isinstance(feat_value, (int, np.integer)):
                label_detail = f"{label_vi} ({feat_value})"
            elif isinstance(feat_value, (float, np.floating)):
                label_detail = f"{label_vi} ({feat_value:.2f})"
            else:
                label_detail = label_vi

            factor = {
                "feature": feat_name,
                "shap": round(float(inverted_shap), 4),
                "value": _serialize_value(feat_value),
                "label_vi": label_detail,
                "dimension_5c": get_5c_dimension(feat_name),
            }

            if inverted_shap > 0:
                positive_factors.append(factor)
            elif inverted_shap < 0:
                negative_factors.append(factor)

        # Sort by magnitude
        positive_factors.sort(key=lambda x: x["shap"], reverse=True)
        negative_factors.sort(key=lambda x: x["shap"])

        # Compute 5C SHAP allocation
        five_c_allocation = self._compute_5c_allocation(feature_shap)

        # Build output
        output = {
            "credit_score": credit_score,
            "pd_pct": round(pd_pct, 2),
            "risk_band": risk_band,
            "model_version": model_version,
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "top_positive_factors": positive_factors[:top_n],
            "top_negative_factors": negative_factors[:top_n],
            "five_c_shap_allocation": five_c_allocation,
            "all_features_shap": {k: round(float(v), 6) for k, v in feature_shap.items()},
        }

        logger.info(
            f"SHAP explanation: score={credit_score}, {len(positive_factors)} positive, "
            f"{len(negative_factors)} negative factors"
        )
        return output

    def explain_batch(
        self,
        X: pd.DataFrame,
        credit_scores: list[int],
        pd_pcts: list[float],
        risk_bands: list[str],
        model_version: str = "lgbm_v1.0_homecredit",
    ) -> list[dict[str, Any]]:
        """Generate SHAP explanations for a batch of applicants.

        Args:
            X: Multi-row DataFrame with features.
            credit_scores: List of credit scores.
            pd_pcts: List of PD percentages.
            risk_bands: List of risk bands.
            model_version: Model version string.

        Returns:
            List of SHAP output dicts.
        """
        results = []
        for i in range(X.shape[0]):
            row = X.iloc[[i]]
            result = self.explain(
                row,
                credit_scores[i],
                pd_pcts[i],
                risk_bands[i],
                model_version,
            )
            results.append(result)
        return results

    def _compute_5c_allocation(self, feature_shap: dict[str, float]) -> dict[str, dict[str, Any]]:
        """Compute SHAP contribution allocation to 5C dimensions.

        Maps each feature's SHAP value to its 5C dimension and computes
        the total SHAP contribution and percentage for each dimension.
        """
        dimension_shap: dict[str, float] = {
            "character": 0.0,
            "capacity": 0.0,
            "capital": 0.0,
            "conditions": 0.0,
            "collateral": 0.0,
        }

        for feat_name, shap_val in feature_shap.items():
            dimension = get_5c_dimension(feat_name)
            if dimension in dimension_shap:
                dimension_shap[dimension] += abs(shap_val)

        total_shap = sum(dimension_shap.values()) or 1.0

        return {
            dim: {
                "shap_sum": round(val, 4),
                "pct": round(val / total_shap * 100),
            }
            for dim, val in dimension_shap.items()
        }


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON output."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if pd.isna(value):
        return None
    return value
