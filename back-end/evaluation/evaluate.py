"""
CREDICOUNCIL Evaluation — Main Runner.

Đánh giá ML Core (A3 — LightGBM) trên tập Home Credit dataset.
Sử dụng pipeline mới mirroring lgb1.ipynb (reference code).

Usage:
    # Evaluate với model đã train:
    python evaluation/evaluate.py --data-dir home-credit-default-risk/ --model-path models/lgbm_ref_v1.pkl --no-shap

    # Train mới + evaluate:
    python evaluation/evaluate.py --data-dir home-credit-default-risk/ --train --no-shap

Output (evaluation/results/):
    ├── metrics_summary.json       ← tất cả metrics dạng JSON
    ├── metrics_summary.csv        ← summary dạng bảng
    ├── riskband_breakdown.csv     ← per-band metrics
    ├── classification_report.json ← precision/recall/F1
    ├── roc_curve.png
    ├── pr_curve.png
    ├── score_distribution.png
    ├── calibration_plot.png
    ├── shap_feature_importance.csv
    ├── shap_feature_importance.png
    └── shap_beeswarm.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from training.feature_engineering import build_all_features
from credicouncil.agents.a3_scoring.model import CrediCouncilModel
from sklearn.model_selection import train_test_split

# 4C mapping for SHAP analysis (optional)
try:
    from credicouncil.config.feature_config import FEATURE_TO_4C_MAPPING
except ImportError:
    FEATURE_TO_4C_MAPPING = {}

from evaluation.metrics import (
    compute_core_metrics,
    compute_riskband_breakdown,
    compute_thinfile_subauc,
    compute_classification_report,
    pd_to_credit_score,
    credit_score_to_band,
)
from evaluation.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_score_distribution,
    plot_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def run_evaluation(
    data_dir: Path,
    model_path: Path | None = None,
    train_first: bool = False,
    no_shap: bool = False,
    output_dir: Path = RESULTS_DIR,
    sample: int | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Chạy toàn bộ evaluation pipeline.

    Args:
        data_dir:     Thư mục chứa Home Credit CSV.
        model_path:   Path tới model .pkl đã train. None = train mới.
        train_first:  Train model mới trước khi evaluate.
        no_shap:      Bỏ qua SHAP analysis.
        output_dir:   Thư mục lưu kết quả.
        sample:       Giới hạn số sample test set.
        test_size:    Fraction dành cho test (default 0.20).
        random_state: Random seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    logger.info("=" * 65)
    logger.info("  CREDICOUNCIL A3 — Evaluation Pipeline (lgb1.ipynb reference)")
    logger.info(f"  Dataset : {data_dir}")
    logger.info(f"  Model   : {model_path or '(train new)'}")
    logger.info(f"  Output  : {output_dir}")
    logger.info("=" * 65)

    # ── Step 1: Build feature matrix from all tables ─────────────────
    logger.info("\n[1/6] Building feature matrix (all 7 tables, ~5-10 min)...")
    X, y, meanenc_feats, cat_feats = build_all_features(data_dir)
    logger.info(f"      Feature matrix: {X.shape[0]:,} × {X.shape[1]}  |  default rate: {y.mean():.1%}")

    # ── Step 2: Stratified 80/20 split ───────────────────────────────
    logger.info(f"\n[2/6] Stratified split (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional: sample test set to speed up evaluation
    if sample and sample < len(y_test):
        logger.info(f"      Sampling test set → {sample:,} rows")
        idx = np.random.default_rng(random_state).choice(len(y_test), sample, replace=False)
        X_test = X_test.iloc[idx]
        y_test = y_test.iloc[idx]

    logger.info(f"      Train: {len(y_train):,} ({y_train.mean():.1%}) | Test: {len(y_test):,} ({y_test.mean():.1%})")

    # ── Step 3: Train hoặc Load model ────────────────────────────────
    model = CrediCouncilModel()

    if train_first or model_path is None:
        logger.info("\n[3/6] Training new LightGBM model (lgb1-reference pipeline)...")
        from training.train_pipeline import run_training
        output_models = PROJECT_ROOT / "models"
        run_training(data_dir=data_dir, output_dir=output_models)
        # Load the freshly trained model
        model_path = output_models / "lgbm_ref_v1.pkl"
        model.load(model_path)
        logger.info(f"      Model saved and loaded from {model_path}")
    else:
        logger.info(f"\n[3/6] Loading model from {model_path}...")
        model.load(model_path)

    # ── Step 4: Inference on test set ────────────────────────────────
    logger.info("\n[4/6] Running inference on test set...")

    # Apply mean encoding to test set (using full X_train as reference)
    # This mirrors exactly what train_pipeline does for test evaluation
    if hasattr(model, "meanenc_feats") and model.meanenc_feats:
        logger.info("      Applying mean encoding to test set...")
        from training.train_pipeline import mean_encode, EXCLUDED_FEATS_SUFFIXES
        train_for_enc = X_train.copy()
        train_for_enc["TARGET"] = y_train.values
        test_for_enc = X_test.copy()
        test_for_enc["TARGET"] = 0  # dummy
        _, X_test_enc = mean_encode(train_for_enc, test_for_enc, model.meanenc_feats, "TARGET", drop=True)
        del train_for_enc, test_for_enc

        # Select only trained features in correct order
        excluded = set(["SK_ID_CURR", "TARGET"] + EXCLUDED_FEATS_SUFFIXES)
        available_features = [f for f in model.feature_names if f in X_test_enc.columns and f not in excluded]
        X_test_pred = X_test_enc[available_features]
    else:
        # Old CREDICOUNCIL model format — no mean encoding
        if model.feature_names:
            available = [f for f in model.feature_names if f in X_test.columns]
            X_test_pred = X_test[available]
        else:
            X_test_pred = X_test

    y_pred_proba = model.predict_proba(X_test_pred)


    # Map PD → credit score → risk band
    credit_scores = np.array([pd_to_credit_score(p) for p in y_pred_proba])
    risk_bands = [credit_score_to_band(s) for s in credit_scores]

    # ── Step 5: Compute metrics ───────────────────────────────────────
    logger.info("\n[5/6] Computing metrics...")

    # Core metrics
    core = compute_core_metrics(y_test.values, y_pred_proba)

    # Risk band breakdown
    riskband_df = compute_riskband_breakdown(
        y_test.values, y_pred_proba, credit_scores
    )
    riskband_df.to_csv(output_dir / "riskband_breakdown.csv", index=False)

    # Classification report at best threshold
    clf_report = compute_classification_report(
        y_test.values, y_pred_proba, threshold=core["best_threshold"]
    )

    # Thin-file sub-AUC (proxy: low credit score = potential thin-file)
    thinfile_metrics = compute_thinfile_subauc(
        y_test.values, y_pred_proba,
        thin_file_mask=(credit_scores < 450)
    )

    # ── Step 5b: Plots ───────────────────────────────────────────────
    plot_roc_curve(y_test.values, y_pred_proba, core["auc_roc"], output_dir)
    plot_pr_curve(y_test.values, y_pred_proba, core["pr_auc"], output_dir)
    plot_score_distribution(credit_scores, y_test.values, output_dir)
    plot_calibration(y_test.values, y_pred_proba, output_dir)

    # ── Step 6: SHAP Analysis ────────────────────────────────────────
    shap_metrics: dict = {}
    if not no_shap:
        from evaluation.shap_analysis import (
            compute_global_shap_importance,
            compute_shap_summary_plot,
            compute_4c_shap_allocation,
            compute_shap_coverage,
        )

        logger.info("\n[6/6] Running SHAP analysis (may take a few minutes)...")

        # Sample test set if large (SHAP is slow)
        shap_sample = X_test if len(X_test) <= 5000 else X_test.sample(5000, random_state=42)

        importance_df = compute_global_shap_importance(
            model.model, shap_sample, output_dir
        )
        compute_shap_summary_plot(model.model, shap_sample, output_dir)
        allocation_df = compute_4c_shap_allocation(
            model.model, shap_sample, FEATURE_TO_4C_MAPPING, output_dir
        )
        shap_coverage = compute_shap_coverage(model.model, shap_sample, top_n=5)

        shap_metrics = {
            "shap_coverage_top5": shap_coverage,
            "top1_feature": importance_df.iloc[0]["feature"] if len(importance_df) > 0 else None,
            "top1_mean_abs_shap": float(importance_df.iloc[0]["mean_abs_shap"]) if len(importance_df) > 0 else None,
        }
    else:
        logger.info("\n[6/6] SHAP analysis skipped (--no-shap)")

    # ── Build summary ─────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)

    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "n_features": X.shape[1],
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
        "elapsed_seconds": elapsed,
        "core_metrics": core,
        "classification_at_best_threshold": clf_report,
        "thinfile_subauc": thinfile_metrics,
        "shap_metrics": shap_metrics,
    }

    # Export JSON summary
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Human-readable CSV summary
    flat_rows = [
        ("AUC-ROC",        f"{core['auc_roc']:.4f}",        ">= 0.77 (target)"),
        ("Gini",           f"{core['gini']:.4f}",            ">= 0.54 (target)"),
        ("KS Statistic",   f"{core['ks_statistic']:.4f}",    ">= 0.35 (good)"),
        ("PR-AUC",         f"{core['pr_auc']:.4f}",          "-"),
        ("Best F1",        f"{core['best_f1']:.4f}",         "-"),
        ("Best Threshold", f"{core['best_threshold']:.4f}",  "-"),
        ("Default Rate",   f"{core['default_rate_pct']}%",   "~8% (dataset)"),
        ("Test Samples",   str(core["n_total"]),              "-"),
        ("Approval Rate",  f"{clf_report['approval_rate_pct']}%", "-"),
        ("N Features",     str(X.shape[1]),                   "-"),
    ]
    summary_csv = pd.DataFrame(flat_rows, columns=["Metric", "Value", "Target"])
    summary_csv.to_csv(output_dir / "metrics_summary.csv", index=False)

    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(clf_report, f, indent=2)

    # ── Print final report ────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  AUC-ROC:       {core['auc_roc']:.4f}   (target >= 0.77)")
    logger.info(f"  Gini:          {core['gini']:.4f}   (target >= 0.54)")
    logger.info(f"  KS Statistic:  {core['ks_statistic']:.4f}")
    logger.info(f"  PR-AUC:        {core['pr_auc']:.4f}")
    if shap_metrics:
        logger.info(f"  SHAP Coverage: {shap_metrics.get('shap_coverage_top5', 'N/A'):.1%}  (target >= 0.85)")
    logger.info(f"  Elapsed:       {elapsed}s")
    logger.info(f"  Results:       {output_dir}/")
    logger.info("=" * 65)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="credicouncil A3 — ML Core Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate pre-trained model (no SHAP, fast):
  python evaluation/evaluate.py --data-dir home-credit-default-risk/ --model-path models/lgbm_ref_v1.pkl --no-shap

  # Evaluate with SHAP:
  python evaluation/evaluate.py --data-dir home-credit-default-risk/ --model-path models/lgbm_ref_v1.pkl

  # Train new model then evaluate:
  python evaluation/evaluate.py --data-dir home-credit-default-risk/ --train --no-shap

  # Sample test set for faster evaluation:
  python evaluation/evaluate.py --data-dir home-credit-default-risk/ --model-path models/lgbm_ref_v1.pkl --no-shap --sample 10000
        """
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Thư mục chứa Home Credit CSV files"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path tới model .pkl đã train. Bỏ trống + --train → train mới"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train model mới trước khi evaluate"
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Bỏ qua SHAP analysis (nhanh hơn)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Thư mục lưu kết quả (default: evaluation/results)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Giới hạn số sample trong test set (vd: --sample 10000)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction dành cho test set (default: 0.20)"
    )

    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    model_path = Path(args.model_path) if args.model_path else None
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if model_path and not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    if not args.train and model_path is None:
        logger.error("Provide either --train or --model-path")
        sys.exit(1)

    run_evaluation(
        data_dir=data_dir,
        model_path=model_path,
        train_first=args.train,
        no_shap=args.no_shap,
        output_dir=output_dir,
        sample=args.sample,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
