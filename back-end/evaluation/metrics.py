"""
CreditLens Evaluation — Metrics Module.

Tính toán các metric chuẩn để đánh giá ML core (A3):
    - AUC-ROC, Gini, KS Statistic
    - Per risk-band breakdown (AAA / AA / A / BBB / CC)
    - Precision-Recall, F1 tại các threshold
    - Thin-file sub-AUC
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# Risk band thresholds mapping (credit score → band)
RISK_BANDS = {
    "AAA": (720, 851),
    "AA":  (640, 720),
    "A":   (560, 640),
    "BBB": (460, 560),
    "CC":  (300, 460),
}

# Score mapping: PD → credit score (300-850)
# Dựa theo design document Section 6.3
def pd_to_credit_score(pd_value: float) -> int:
    """Map PD probability to credit score 300-850."""
    # Logistic-like mapping: PD=0 → 850, PD=1 → 300
    return max(300, min(850, int(850 - (pd_value ** 0.5) * 550)))


def credit_score_to_band(score: int) -> str:
    """Map credit score to risk band."""
    for band, (low, high) in RISK_BANDS.items():
        if low <= score < high:
            return band
    return "CC"


def compute_core_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, Any]:
    """Tính toán metrics cốt lõi: AUC, Gini, KS.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred_proba: Predicted probabilities of default.

    Returns:
        Dict chứa AUC, Gini, KS và các thông tin bổ sung.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1

    # KS Statistic (Kolmogorov-Smirnov)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = float(np.max(tpr - fpr))
    ks_threshold = float(thresholds[np.argmax(tpr - fpr)])

    # Average Precision (PR-AUC)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    # Best threshold by F1
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores[:-1])  # drop last (precision=1, recall=0)
    best_threshold = float(pr_thresholds[best_f1_idx])
    best_f1 = float(f1_scores[best_f1_idx])

    # Default rate info
    n_total = len(y_true)
    n_default = int(y_true.sum())
    default_rate = n_default / n_total

    logger.info(f"AUC-ROC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks:.4f}")

    return {
        "auc_roc": round(auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks, 4),
        "ks_threshold": round(ks_threshold, 4),
        "pr_auc": round(pr_auc, 4),
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_threshold, 4),
        "n_total": n_total,
        "n_default": n_default,
        "default_rate_pct": round(default_rate * 100, 2),
    }


def compute_riskband_breakdown(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    credit_scores: np.ndarray,
) -> pd.DataFrame:
    """Phân tích metrics theo từng risk band.

    Args:
        y_true: Labels thực tế.
        y_pred_proba: Predicted PD probabilities.
        credit_scores: Credit score đã map (300-850).

    Returns:
        DataFrame với per-band metrics.
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
        "credit_score": credit_scores,
        "risk_band": [credit_score_to_band(s) for s in credit_scores],
    })

    rows = []
    for band in ["AAA", "AA", "A", "BBB", "CC"]:
        subset = df[df["risk_band"] == band]
        if len(subset) < 10:
            continue

        n = len(subset)
        n_default = int(subset["y_true"].sum())
        default_rate = n_default / n

        # AUC per band nếu có đủ cả 2 class
        band_auc = None
        if subset["y_true"].nunique() == 2:
            band_auc = round(roc_auc_score(subset["y_true"], subset["y_pred_proba"]), 4)

        rows.append({
            "risk_band": band,
            "n_samples": n,
            "n_default": n_default,
            "default_rate_pct": round(default_rate * 100, 2),
            "auc": band_auc,
            "avg_pd_pct": round(subset["y_pred_proba"].mean() * 100, 2),
            "score_range": f"{int(subset['credit_score'].min())}–{int(subset['credit_score'].max())}",
        })

    result = pd.DataFrame(rows)
    logger.info(f"Risk band breakdown:\n{result.to_string(index=False)}")
    return result


def compute_thinfile_subauc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thin_file_mask: np.ndarray,
) -> dict[str, Any]:
    """Tính sub-AUC riêng cho thin-file customers.

    Quan trọng vì đây là core innovation: khả năng score customer
    không có lịch sử tín dụng.

    Args:
        y_true: Labels thực tế.
        y_pred_proba: Predicted PD probabilities.
        thin_file_mask: Boolean mask, True = thin-file customer.

    Returns:
        Dict với AUC cho thin-file và non-thin-file.
    """
    result: dict[str, Any] = {}

    # Thin-file sub-AUC
    thin_mask = thin_file_mask.astype(bool)
    if thin_mask.sum() >= 10 and len(np.unique(y_true[thin_mask])) == 2:
        result["thin_file_auc"] = round(
            roc_auc_score(y_true[thin_mask], y_pred_proba[thin_mask]), 4
        )
        result["thin_file_n"] = int(thin_mask.sum())
        result["thin_file_default_rate_pct"] = round(
            y_true[thin_mask].mean() * 100, 2
        )

    # Non-thin-file sub-AUC
    normal_mask = ~thin_mask
    if normal_mask.sum() >= 10 and len(np.unique(y_true[normal_mask])) == 2:
        result["normal_file_auc"] = round(
            roc_auc_score(y_true[normal_mask], y_pred_proba[normal_mask]), 4
        )
        result["normal_file_n"] = int(normal_mask.sum())

    logger.info(f"Thin-file sub-AUC: {result}")
    return result


def compute_classification_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Classification report tại threshold cụ thể.

    Args:
        y_true: Labels thực tế.
        y_pred_proba: Predicted probabilities.
        threshold: Classification cutoff.

    Returns:
        Dict với precision, recall, F1, confusion matrix.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "approval_rate_pct": round((tn + fn) / len(y_true) * 100, 2),  # y_pred=0 = approve
        "rejection_rate_pct": round((tp + fp) / len(y_true) * 100, 2),
    }
