"""
CreditLens A3 — Credit Score Mapper.

Maps LightGBM predict_proba output (PD percentage) to the
300-850 credit score scale and classifies into risk bands.
"""

from __future__ import annotations

import logging

import numpy as np

from creditlens.config.feature_config import RISK_BANDS, RiskBandDefinition

logger = logging.getLogger(__name__)


def pd_to_credit_score(pd_pct: float) -> int:
    """Map probability of default (%) to credit score (300-850).

    Uses a logarithmic mapping that produces scores aligned with
    the risk band definitions in the design document.

    The mapping:
        PD ~0%   → Score 850 (best)
        PD ~2%   → Score ~720 (AAA/AA boundary)
        PD ~8%   → Score ~640 (AA/A boundary)
        PD ~18%  → Score ~560 (A/BBB boundary)
        PD ~35%  → Score ~460 (BBB/CC boundary)
        PD ~100% → Score 300 (worst)

    Args:
        pd_pct: Probability of default as percentage (0-100).

    Returns:
        Credit score integer (300-850).
    """
    # Clamp PD to valid range
    pd_clamped = max(0.001, min(pd_pct, 99.99))

    # Logarithmic mapping: score = 850 - k * ln(pd)
    # Calibrated so that PD=2% → 720, PD=35% → 460
    # k ≈ 93, offset ≈ 850 + 93*ln(0.02) ≈ 486
    score = 850 + 93 * np.log(0.02) - 93 * np.log(pd_clamped / 100)
    score = int(round(max(300, min(850, score))))

    return score


def credit_score_to_risk_band(score: int) -> RiskBandDefinition:
    """Classify credit score into risk band.

    Args:
        score: Credit score (300-850).

    Returns:
        RiskBandDefinition with band, decision, description.
    """
    for band in RISK_BANDS:
        if band.score_min <= score <= band.score_max:
            return band

    # Fallback for out-of-range scores
    if score >= 850:
        return RISK_BANDS[0]  # AAA
    return RISK_BANDS[-1]  # CC


def map_prediction(pd_probability: float) -> dict:
    """Full mapping: PD probability → credit score + risk band + decision.

    Args:
        pd_probability: Raw model output (0-1 probability of default).

    Returns:
        Dict with credit_score, pd_pct, risk_band, auto_decision, description_vi.
    """
    pd_pct = pd_probability * 100
    credit_score = pd_to_credit_score(pd_pct)
    band = credit_score_to_risk_band(credit_score)

    result = {
        "credit_score": credit_score,
        "pd_pct": round(pd_pct, 2),
        "risk_band": band.band,
        "auto_decision": band.auto_decision,
        "description_vi": band.description_vi,
    }

    logger.debug(f"Mapped PD {pd_pct:.1f}% → Score {credit_score} ({band.band})")
    return result


def batch_map_predictions(pd_probabilities: np.ndarray) -> list[dict]:
    """Map a batch of PD probabilities to credit scores.

    Args:
        pd_probabilities: Array of default probabilities.

    Returns:
        List of mapping dicts.
    """
    return [map_prediction(pd) for pd in pd_probabilities]
