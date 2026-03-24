"""
CreditLens Evaluation — Plots Module.

Tạo các visualization cho evaluation report:
    - ROC Curve
    - Precision-Recall Curve
    - Score Distribution by Risk Band
    - Calibration Plot (predicted vs actual PD)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

BAND_COLORS = {
    "AAA": "#15803D",   # green
    "AA":  "#2563EB",   # blue
    "A":   "#CA8A04",   # yellow
    "BBB": "#EA580C",   # orange
    "CC":  "#DC2626",   # red
}


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    auc: float,
    output_dir: Path,
) -> None:
    """Vẽ ROC Curve với AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random")

    # KS point
    ks_idx = np.argmax(tpr - fpr)
    ax.plot(fpr[ks_idx], tpr[ks_idx], "ro", markersize=8,
            label=f"KS = {tpr[ks_idx] - fpr[ks_idx]:.4f}")
    ax.axvline(fpr[ks_idx], color="#DC2626", lw=1, linestyle=":")
    ax.axhline(tpr[ks_idx], color="#DC2626", lw=1, linestyle=":")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — CreditLens A3", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    logger.info(f"ROC curve saved → {output_dir / 'roc_curve.png'}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    pr_auc: float,
    output_dir: Path,
) -> None:
    """Vẽ Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#16A34A", lw=2, label=f"PR Curve (AP = {pr_auc:.4f})")
    ax.axhline(baseline, color="#9CA3AF", lw=1.5, linestyle="--",
               label=f"Baseline (default rate = {baseline:.1%})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — CreditLens A3", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close(fig)
    logger.info(f"PR curve saved → {output_dir / 'pr_curve.png'}")


def plot_score_distribution(
    credit_scores: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
) -> None:
    """Phân phối credit score, tô màu theo default/non-default."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Distribution by class
    ax = axes[0]
    default_scores = credit_scores[y_true == 1]
    nondefault_scores = credit_scores[y_true == 0]

    bins = range(300, 861, 20)
    ax.hist(nondefault_scores, bins=bins, alpha=0.65, color="#2563EB",
            label="Non-default", density=True)
    ax.hist(default_scores, bins=bins, alpha=0.65, color="#DC2626",
            label="Default", density=True)

    # Band boundaries
    for score in [460, 560, 640, 720]:
        ax.axvline(score, color="#6B7280", lw=1, linestyle="--", alpha=0.7)

    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution by Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Right: Risk band pie
    ax2 = axes[1]
    band_names = ["AAA", "AA", "A", "BBB", "CC"]
    band_thresholds = [(720, 851), (640, 720), (560, 640), (460, 560), (300, 460)]
    band_counts = []
    for low, high in band_thresholds:
        count = ((credit_scores >= low) & (credit_scores < high)).sum()
        band_counts.append(count)

    colors = [BAND_COLORS[b] for b in band_names]
    wedges, texts, autotexts = ax2.pie(
        band_counts,
        labels=[f"{b}\n({c:,})" for b, c in zip(band_names, band_counts)],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax2.set_title("Distribution by Risk Band", fontsize=13, fontweight="bold")

    plt.suptitle("CreditLens A3 — Score Distribution Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Score distribution saved → {output_dir / 'score_distribution.png'}")


def plot_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: Path,
    n_bins: int = 10,
) -> None:
    """Calibration plot: predicted PD vs actual default rate.

    Đánh giá độ tin cậy của predicted probability.
    Đường thẳng 45° = perfectly calibrated.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#2563EB", lw=2, markersize=8,
            label="CreditLens A3")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--",
            label="Perfect calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Fraction of Defaults", fontsize=12)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(output_dir / "calibration_plot.png", dpi=150)
    plt.close(fig)
    logger.info(f"Calibration plot saved → {output_dir / 'calibration_plot.png'}")
