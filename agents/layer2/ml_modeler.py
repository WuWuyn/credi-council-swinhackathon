"""
ML Modeler Agent - Layer 2: Multidimensional Assessment.

This agent loads the pre-trained LightGBM models (lgb1.pkl, lgb2.pkl, lgb3.pkl)
and runs inference on the feature vector produced by the Layer 1 FeatureEngineerAgent.
The final ML credit risk score is the average of the three model predictions.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path to the model files relative to this file
_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'models')
)


class MLModelerAgent:
    """
    Layer 2 agent that uses pre-trained LightGBM models to produce a credit risk score.
    
    Unlike other agents it does NOT call an LLM — it runs deterministic ML inference.
    It exposes the same dict interface as other agents so the orchestrator can treat
    it uniformly.
    """

    def __init__(self, config=None, models_dir: str = None):
        self.name = "ML Modeler"
        self.models_dir = models_dir or _MODELS_DIR
        self._models = None
        self._feature_names = None

    def _load_models(self):
        """Lazy-load the three LightGBM booster text models."""
        import lightgbm as lgb
        if self._models is not None:
            return
        
        self._models = []
        for fname in ['lgb1_booster.txt', 'lgb2_booster.txt', 'lgb3_booster.txt']:
            fpath = os.path.join(self.models_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Model file not found: {fpath}\n"
                    f"Please place lgb1_booster.txt, lgb2_booster.txt, lgb3_booster.txt in: {self.models_dir}"
                )
            booster = lgb.Booster(model_file=fpath)
            self._models.append(booster)
            logger.info(f"[ML Modeler] Loaded model: {fname}")
        
        # Get expected feature names from the first model
        self._feature_names = self._models[0].feature_name()

    def invoke(self, features: dict) -> dict:
        """
        Run ML inference on the feature dict.

        Args:
            features: A flat dict of feature_name -> value (output of aggregator).

        Returns:
            Dict with 'ml_credit_score', 'default_probability', individual model scores, and metadata.
        """
        try:
            self._load_models()
            
            # Build a single-row DataFrame aligned to model's expected feature order
            if self._feature_names:
                # Align to the exact features the model expects
                row = {f: features.get(f, np.nan) for f in self._feature_names}
                X = pd.DataFrame([row], columns=self._feature_names)
            else:
                # Fallback: use whatever numeric features we have
                numeric_features = {
                    k: v for k, v in features.items()
                    if isinstance(v, (int, float)) and not k.startswith('_') and k != 'SK_ID_CURR'
                }
                X = pd.DataFrame([numeric_features])
            
            # Run predict_proba on all three models; average the default probability (class 1)
            probs = []
            individual_scores = {}
            for i, booster in enumerate(self._models, 1):
                try:
                    fnames = booster.feature_name()
                    if fnames:
                        row_vals = [features.get(f, np.nan) for f in fnames]
                    else:
                        row_vals = list(X.iloc[0].values)
                    
                    X_arr = np.array([row_vals], dtype=np.float64)
                    raw_pred = booster.predict(X_arr)
                    prob = float(raw_pred[0])
                except Exception as inner_e:
                    logger.warning(f"[ML Modeler] Model {i} inference error: {inner_e}. Using 0.5 fallback.")
                    prob = 0.5
                
                probs.append(prob)
                individual_scores[f'lgb{i}_default_prob'] = round(prob, 4)
            
            avg_prob = float(np.mean(probs))
            # Credit score: higher is better (100 = perfect, 0 = certain default)
            credit_score = round((1 - avg_prob) * 100)
            
            result = {
                "ml_credit_score": credit_score,
                "default_probability": round(avg_prob, 4),
                "risk_level": _prob_to_risk_level(avg_prob),
                **individual_scores,
                "features_used": len([v for v in X.iloc[0] if not pd.isna(v)]),
                "total_features": len(self._feature_names) if self._feature_names else len(X.columns),
                "_metadata": {"agent": self.name, "token_usage": {}},
            }
            logger.info(
                f"[ML Modeler] SK_ID_CURR={features.get('SK_ID_CURR')} | "
                f"P(default)={avg_prob:.4f} | credit_score={credit_score}"
            )
            return result

        except Exception as e:
            logger.error(f"[ML Modeler] Inference failed: {e}")
            return {
                "error": str(e),
                "agent": self.name,
                "_metadata": {"agent": self.name, "token_usage": {}},
            }


def _prob_to_risk_level(prob: float) -> str:
    if prob < 0.05:
        return "Very Low"
    elif prob < 0.15:
        return "Low"
    elif prob < 0.30:
        return "Medium"
    elif prob < 0.50:
        return "High"
    else:
        return "Very High"
