"""Feature Engineer Agent - Layer 1: Data Ingestion & Contextualization.

This agent uses the generate_comprehensive_features tool to dynamically
extract features from the Home Credit CSV dataset for the given applicant.
"""

import json
import logging
import os

from agents.base import BaseAgent
from prompts.templates import PROMPTS

logger = logging.getLogger(__name__)

# Path to the data directory
_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'home-credit-default-risk')
)


class FeatureEngineerAgent(BaseAgent):
    """
    Layer 1 agent that performs dynamic feature engineering via tool calling.
    
    Given raw applicant context (including SK_ID_CURR), this agent calls the
    `generate_comprehensive_features` tool to extract bureau, previous applications,
    installments, POS, credit card, and application-level features from the dataset.
    """

    @property
    def name(self) -> str:
        return "Feature Engineer"

    @property
    def system_prompt(self) -> str:
        return PROMPTS["feature_engineer"]

    def invoke(self, user_input: str) -> dict:
        """
        Invokes the feature engineering pipeline.
        
        Extracts SK_ID_CURR from the input and runs the aggregator to get
        comprehensive features for Layer 2 consumption.
        """
        logger.info(f"[{self.name}] Invoking feature engineering...")

        # Try to extract SK_ID_CURR from the user_input string
        sk_id_curr = self._extract_sk_id(user_input)
        
        if sk_id_curr is None:
            logger.warning(f"[{self.name}] Could not extract SK_ID_CURR from input. Using LLM fallback.")
            return super().invoke(user_input)

        # Lazy import to avoid circular import at module load time
        from pipeline.feature_extractors.aggregator import generate_comprehensive_features

        # Call the feature aggregator directly (tool call simulation)
        try:
            logger.info(f"[{self.name}] Calling generate_comprehensive_features for SK_ID_CURR={sk_id_curr}")
            data_dir = os.environ.get('HOME_CREDIT_DATA_DIR', _DATA_DIR)
            features = generate_comprehensive_features(sk_id_curr, data_dir)
            
            n_features = len([v for v in features.values() if isinstance(v, (int, float))])
            
            result = {
                "sk_id_curr": sk_id_curr,
                "feature_count": n_features,
                "features": features,
                "engineering_method": "generate_comprehensive_features",
                "sources": ["application_train", "bureau", "previous_application",
                           "installments_payments", "POS_CASH_balance", "credit_card_balance"],
                "_metadata": {"agent": self.name, "token_usage": {}},
            }
            
            errors = features.get("_extraction_errors")
            if errors:
                result["extraction_warnings"] = errors
                
            logger.info(f"[{self.name}] Feature engineering complete. {n_features} features extracted.")
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Feature extraction failed: {e}")
            return {
                "error": str(e),
                "sk_id_curr": sk_id_curr,
                "agent": self.name,
                "_metadata": {"agent": self.name, "token_usage": {}},
            }

    def _extract_sk_id(self, text: str):
        """
        Try to extract SK_ID_CURR from the applicant text.
        Looking for patterns like:
        - Application ID (SK_ID_CURR): 100002
        - SK_ID_CURR: 100001
        """
        import re
        patterns = [
            r'SK_ID_CURR[\)\s:=]+(\d+)',
            r'"SK_ID_CURR"\s*:\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
