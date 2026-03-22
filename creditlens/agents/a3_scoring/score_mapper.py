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

    Uses piecewise linear interpolation in log-PD space, anchored
    exactly to the RISK_BANDS boundaries:

        PD ≤ 0.5%  → Score 850 (best)
        PD = 2%    → Score 720 (AAA/AA boundary)
        PD = 8%    → Score 640 (AA/A boundary)
        PD = 18%   → Score 560 (A/BBB boundary)
        PD = 35%   → Score 460 (BBB/CC boundary)
        PD ≥ 100%  → Score 300 (worst)

    Args:
        pd_pct: Probability of default as percentage (0-100).

    Returns:
        Credit score integer (300-850).
    """
    # Clamp PD to valid range
    pd_clamped = max(0.01, min(pd_pct, 99.99))

    # Anchor points: (PD%, Score) — exactly matching RISK_BANDS boundaries
    # Piecewise linear interpolation in log(PD) space ensures smooth,
    # monotonic mapping while hitting every boundary precisely.
    anchors = [
        (0.5, 850),   # Best possible
        (2.0, 720),   # AAA / AA boundary
        (8.0, 640),   # AA / A boundary
        (18.0, 560),  # A / BBB boundary
        (35.0, 460),  # BBB / CC boundary
        (100.0, 300), # Worst possible
    ]

    ln_pd = np.log(pd_clamped)

    # If below lowest anchor, cap at max score
    if pd_clamped <= anchors[0][0]:
        return 850

    # If above highest anchor, cap at min score
    if pd_clamped >= anchors[-1][0]:
        return 300

    # Find the segment and interpolate linearly in log-PD space
    for i in range(len(anchors) - 1):
        pd_lo, score_lo = anchors[i]
        pd_hi, score_hi = anchors[i + 1]
        if pd_clamped <= pd_hi:
            ln_lo = np.log(pd_lo)
            ln_hi = np.log(pd_hi)
            # Linear interpolation in log space
            t = (ln_pd - ln_lo) / (ln_hi - ln_lo)
            score = score_lo + t * (score_hi - score_lo)
            return int(round(max(300, min(850, score))))

    return 300


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
