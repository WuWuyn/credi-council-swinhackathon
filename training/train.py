"""
CreditLens Training Pipeline.

Full training script: data loading → feature engineering → ADASYN →
LightGBM training → SHAP validation → model export.

Usage:
    python training/train.py --data-dir home-credit-default-risk/
    python training/train.py --data-dir home-credit-default-risk/ --tune --n-trials 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from creditlens.data.loader import load_all_tables
from creditlens.data.feature_engineering import build_feature_matrix, select_features
from creditlens.data.preprocessing import clean_feature_matrix, split_data, apply_adasyn
from creditlens.agents.a3_scoring.model import CreditLensModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CreditLens Training Pipeline")
    parser.add_argument(
        "--data-dir", type=str, default="home-credit-default-risk",
        help="Path to Home Credit data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--feature-set", type=str, default="production",
        choices=["production", "pilot"],
        help="Feature set to use for training"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--no-adasyn", action="store_true",
        help="Skip ADASYN oversampling"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── Step 1: Load all tables ──
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Home Credit tables")
    logger.info("=" * 60)
    tables = load_all_tables(data_dir)

    # ── Step 2: Feature engineering ──
    logger.info("=" * 60)
    logger.info("STEP 2: Feature engineering")
    logger.info("=" * 60)
    feature_matrix = build_feature_matrix(tables)

    # Free memory
    del tables

    # ── Step 3: Select features ──
    logger.info("=" * 60)
    logger.info(f"STEP 3: Selecting {args.feature_set} features")
    logger.info("=" * 60)
    X, y = select_features(feature_matrix, feature_set=args.feature_set)
    X = clean_feature_matrix(X)

    logger.info(f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"Default rate: {y.mean():.1%}")

    # ── Step 4: Train/Val/Test split ──
    logger.info("=" * 60)
    logger.info("STEP 4: Train/Validation/Test split (70/15/15)")
    logger.info("=" * 60)
    splits = split_data(X, y)

    # ── Step 5: ADASYN oversampling ──
    if not args.no_adasyn:
        logger.info("=" * 60)
        logger.info("STEP 5: ADASYN oversampling")
        logger.info("=" * 60)
        X_train, y_train = apply_adasyn(splits["X_train"], splits["y_train"])
    else:
        X_train, y_train = splits["X_train"], splits["y_train"]
        logger.info("STEP 5: Skipping ADASYN (--no-adasyn flag)")

    # ── Step 6: Model training ──
    logger.info("=" * 60)
    logger.info("STEP 6: Training LightGBM model")
    logger.info("=" * 60)

    model = CreditLensModel()

    # Optional: Optuna tuning
    if args.tune:
        logger.info(f"Running Optuna tuning with {args.n_trials} trials...")
        tune_result = model.tune_with_optuna(
            X_train, y_train,
            splits["X_val"], splits["y_val"],
            n_trials=args.n_trials,
        )
        logger.info(f"Best AUC from tuning: {tune_result['best_auc']:.4f}")

    # Train final model
    train_result = model.train(
        X_train, y_train,
        X_val=splits["X_val"],
        y_val=splits["y_val"],
    )

    # ── Step 7: Evaluation on test set ──
    logger.info("=" * 60)
    logger.info("STEP 7: Evaluation on held-out test set")
    logger.info("=" * 60)

    from sklearn.metrics import roc_auc_score, classification_report

    test_pred = model.predict_proba(splits["X_test"])
    test_auc = roc_auc_score(splits["y_test"], test_pred)
    gini = 2 * test_auc - 1

    logger.info(f"Test AUC-ROC: {test_auc:.4f}")
    logger.info(f"Gini Coefficient: {gini:.4f}")

    # Binary predictions at 0.5 threshold
    test_binary = (test_pred >= 0.5).astype(int)
    logger.info(f"\nClassification Report:\n{classification_report(splits['y_test'], test_binary)}")

    # ── Step 8: Save model ──
    logger.info("=" * 60)
    logger.info("STEP 8: Saving model")
    logger.info("=" * 60)

    output_path = Path(args.output_dir) / "lgbm_v1.pkl"
    model.save(output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Model: {output_path}")
    logger.info(f"  Features: {args.feature_set} ({X.shape[1]})")
    logger.info(f"  Training samples: {X_train.shape[0]:,} (with ADASYN)")
    logger.info(f"  Validation AUC: {train_result.get('val_auc', 'N/A')}")
    logger.info(f"  Test AUC: {test_auc:.4f}")
    logger.info(f"  Gini: {gini:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
