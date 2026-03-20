"""
CreditLens A2 — Thin-file Alternative Scoring Path.

When thin_file_flag=True (no CIC record), this handler activates
alternative feature weights that rely on bank transaction data
instead of traditional credit bureau information.
"""

from __future__ import annotations

import logging
from typing import Any

from creditlens.config.feature_config import (
    THIN_FILE_FEATURE_WEIGHTS,
    THIN_FILE_MIN_MONTHS,
)

logger = logging.getLogger(__name__)


def activate_thin_file_path(
    structured_feats: dict[str, Any],
    confidence_map: dict[str, float],
) -> dict[str, Any]:
    """Activate the thin-file alternative scoring path.

    When a customer has no CIC history, CreditLens does NOT reject them.
    Instead, it reweights features to rely on alternative data signals:
        - income_stability: 30%
        - salary_pattern: 25%
        - debt_service: 25%
        - bill_payment: 15%
        - inflow_outflow: 5%

    Requirements:
        - Minimum 3 months of continuous bank statement data
        - If less than 3 months → ESCALATE (insufficient data)

    Args:
        structured_feats: Current feature dictionary.
        confidence_map: Current confidence scores.

    Returns:
        Dict with thin-file adjustments and routing.
    """
    logger.info("Activating thin-file alternative scoring path")

    # Check minimum bank data requirement
    n_months = structured_feats.get("n_months_bank_data", 6)
    if n_months < THIN_FILE_MIN_MONTHS:
        logger.warning(
            f"Insufficient bank data for thin-file path: {n_months} months "
            f"< {THIN_FILE_MIN_MONTHS} minimum"
        )
        return {
            "thin_file_activated": True,
            "thin_file_eligible": False,
            "routing": "ESCALATE",
            "warnings": [
                f"Thin-file customer với chỉ {n_months} tháng sao kê. "
                f"Yêu cầu tối thiểu {THIN_FILE_MIN_MONTHS} tháng."
            ],
        }

    # Compute alternative data quality score
    alt_data_features = {}
    alt_data_quality = 0.0

    for feat_name, weight in THIN_FILE_FEATURE_WEIGHTS.items():
        if feat_name in structured_feats and confidence_map.get(feat_name, 0) >= 0.50:
            alt_data_features[feat_name] = {
                "value": structured_feats[feat_name],
                "weight": weight,
                "confidence": confidence_map.get(feat_name, 0.5),
            }
            alt_data_quality += weight * confidence_map.get(feat_name, 0.5)

    warnings = [
        "Khách hàng được đánh giá theo hướng thin-file. "
        "Kết quả dựa trên dữ liệu giao dịch thay thế, "
        "không có lịch sử tín dụng từ CIC."
    ]

    if alt_data_quality < 0.50:
        warnings.append(
            f"Chất lượng dữ liệu thay thế thấp ({alt_data_quality:.1%}). "
            f"Khuyến nghị bổ sung tài liệu."
        )

    result = {
        "thin_file_activated": True,
        "thin_file_eligible": True,
        "thin_file_feature_weights": THIN_FILE_FEATURE_WEIGHTS,
        "alt_data_features": alt_data_features,
        "alt_data_quality": round(alt_data_quality, 3),
        "n_months_bank_data": n_months,
        "warnings": warnings,
    }

    logger.info(
        f"Thin-file path activated: {len(alt_data_features)} alt features, "
        f"quality={alt_data_quality:.1%}"
    )
    return result
