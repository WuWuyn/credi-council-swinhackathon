"""
CreditLens Evaluation — SHAP Analysis Module.

Phân tích SHAP values trên tập test để đánh giá explainability:
    - Feature importance tổng thể (mean |SHAP|)
    - SHAP beeswarm / summary plot
    - 5C dimension SHAP allocation
    - SHAP coverage validation (bao nhiêu % narrative factors được SHAP support)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_global_shap_importance(
    model,
    X_test: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> pd.DataFrame:
    """Tính global feature importance dựa trên mean |SHAP|.

    Args:
        model: Trained LightGBM model.
        X_test: Test set features.
        output_dir: Thư mục lưu plots.
        top_n: Số feature hiển thị trong chart.

    Returns:
        DataFrame feature_name × mean_abs_shap, sắp xếp giảm dần.
    """
    logger.info(f"Computing SHAP values for {len(X_test):,} samples...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary classification: lấy class 1 (default)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)

    # Save to CSV
    importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    logger.info(f"SHAP importance saved → {output_dir / 'shap_feature_importance.csv'}")

    # Plot: bar chart top N features
    fig, ax = plt.subplots(figsize=(10, 7))
    top = importance_df.head(top_n)
    bars = ax.barh(
        top["feature"][::-1],
        top["mean_abs_shap"][::-1],
        color="#2563EB",
        alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (SHAP)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "shap_feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info(f"SHAP bar chart saved → {output_dir / 'shap_feature_importance.png'}")

    return importance_df


def compute_shap_summary_plot(
    model,
    X_test: pd.DataFrame,
    output_dir: Path,
    max_display: int = 20,
) -> None:
    """Generate SHAP beeswarm summary plot.

    Args:
        model: Trained LightGBM model.
        X_test: Test set features (sample nếu quá lớn).
        output_dir: Thư mục lưu plot.
        max_display: Số feature hiển thị.
    """
    # Sample nếu quá lớn (SHAP plot chậm với nhiều samples)
    if len(X_test) > 2000:
        X_sample = X_test.sample(2000, random_state=42)
        logger.info("Sampling 2000 rows for SHAP beeswarm plot (performance)")
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv, X_sample,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP beeswarm saved → {output_dir / 'shap_beeswarm.png'}")


def compute_5c_shap_allocation(
    model,
    X_test: pd.DataFrame,
    feature_to_4c_mapping: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Tính SHAP allocation theo 5 chiều: Character/Capacity/Capital/Conditions/Collateral.

    Args:
        model: Trained model.
        X_test: Test features.
        feature_to_4c_mapping: Dict {feature_name: dimension}. Also uses get_5c_dimension for prefix matching.
        output_dir: Output directory.

    Returns:
        DataFrame với tổng SHAP và % cho mỗi dimension.
    """
    from creditlens.config.feature_config import get_5c_dimension

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_shap = dict(zip(X_test.columns, mean_abs_shap))

    # Aggregate per dimension (5C)
    dim_shap = {"character": 0.0, "capacity": 0.0, "capital": 0.0, "conditions": 0.0, "collateral": 0.0}
    dim_features: dict[str, list[str]] = {d: [] for d in dim_shap}

    for feat, shap_val in feature_shap.items():
        dim = get_5c_dimension(feat)
        if dim in dim_shap:
            dim_shap[dim] += shap_val
            dim_features[dim].append(feat)

    total = sum(dim_shap.values()) or 1.0
    rows = []
    for dim, val in dim_shap.items():
        rows.append({
            "dimension": dim.upper(),
            "shap_sum": round(val, 4),
            "pct": round(val / total * 100, 1),
            "n_features": len(dim_features[dim]),
            "features": ", ".join(dim_features[dim]),
        })

    result = pd.DataFrame(rows).sort_values("shap_sum", ascending=False)
    result.to_csv(output_dir / "shap_5c_allocation.csv", index=False)

    # Pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]
    wedges, texts, autotexts = ax.pie(
        result["shap_sum"],
        labels=[f"{r['dimension']}\n({r['pct']}%)" for _, r in result.iterrows()],
        autopct="%1.1f%%",
        colors=colors[:len(result)],
        startangle=90,
    )
    ax.set_title("SHAP Contribution — 5C Dimensions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "shap_5c_allocation.png", dpi=150)
    plt.close(fig)

    logger.info(f"5C SHAP allocation:\n{result[['dimension','shap_sum','pct']].to_string(index=False)}")
    return result


# Backward compat alias
compute_4c_shap_allocation = compute_5c_shap_allocation


def compute_shap_coverage(
    model,
    X_test: pd.DataFrame,
    top_n: int = 5,
) -> float:
    """Tính SHAP coverage: % variance được giải thích bởi top N features.

    Đây là metric đánh giá explainability: bao nhiêu % tổng SHAP
    được cover bởi top 5 factors (dùng trong A4 narrative).

    Args:
        model: Trained model.
        X_test: Test features.
        top_n: Số features để tính coverage.

    Returns:
        Coverage ratio (0–1).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    total = mean_abs.sum()
    top_sum = np.sort(mean_abs)[::-1][:top_n].sum()

    coverage = top_sum / total if total > 0 else 0.0
    logger.info(f"SHAP coverage (top {top_n}): {coverage:.1%}")
    return round(float(coverage), 4)
