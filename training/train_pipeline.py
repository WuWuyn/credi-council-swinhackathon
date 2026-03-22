"""
Training Pipeline — Mirror of lgb1.ipynb (Home Credit Kaggle Reference).

Differences from reference:
  - Reference: train on 100% data, predict on test set (no TARGET) for submission
  - CreditLens: Stratified 80/20 split → keep 20% test set for evaluation

Key faithfulness to reference:
  - bagging_classifier: within each outer fold, majority class is split into 3 sub-folds
    and results are averaged (downsampling to handle class imbalance)
  - mean_encode: applied PER OUTER FOLD after fold split, before training
  - LightGBM params from lgb1 reference
  - excluded_feats: SK_ID_CURR, TARGET, prev_sum_CODE_REJECT_REASON_CLIENT, bureau_sum_CREDIT_ACTIVE_Active

Usage:
    python training/train_pipeline.py --data-dir home-credit-default-risk/ --output-dir models/
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.feature_engineering import build_all_features, mean_encode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Features excluded from training (not predictors) ─────────────────────────

EXCLUDED_FEATS_SUFFIXES = [
    "prev_sum_CODE_REJECT_REASON_CLIENT",
    "bureau_sum_CREDIT_ACTIVE_Active",
]


# ─── LightGBM params (identical to lgb1.ipynb cell 23) ──────────────────────

LGB_PARAMS = dict(
    n_estimators        = 5000,
    learning_rate       = 0.03,
    num_leaves          = 26,
    metric              = "auc",
    colsample_bytree    = 0.3,
    subsample           = 0.9320,
    max_depth           = 4,
    reg_alpha           = 4.8299,
    reg_lambda          = 3.6335,
    min_split_gain      = 0.0068,
    min_child_weight    = 9.8138,
    silent              = True,
    verbose             = -1,
    n_jobs              = -1,
    class_weight        = {0: 1, 1: 1.0122},
)


# ─── Bagging classifier (mirrors lgb1.ipynb bagging_classifier class) ────────

class BaggingClassifier:
    """
    Down-sampling bagging that mimics lgb1.ipynb.

    For each outer fold:
        minority (TARGET==1) is kept whole.
        majority (TARGET==0) is split into n_estimators sub-folds;
        we train once per sub-fold on minority + 1/n_estimators of majority.
    Predictions are averaged across the n_estimators.
    """

    def __init__(self, base_params: dict, n_estimators: int = 3):
        self.base_params_    = base_params
        self.n_estimators_   = n_estimators
        self.estimators_     = []
        self.feature_importances_gain_  = None
        self.feature_importances_split_ = None
        self.n_classes_ = 2

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set=None, eval_metric="auc",
            verbose=200, early_stopping_rounds=100,
            categorical_feature=None):
        self.estimators_ = []
        n_feats = X.shape[1]
        self.feature_importances_gain_  = np.zeros(n_feats)
        self.feature_importances_split_ = np.zeros(n_feats)

        if self.n_estimators_ == 1:
            logger.info("n_estimators=1, no downsampling")
            estimator = LGBMClassifier(**self.base_params_)
            fit_kwargs = dict(
                eval_set=[(X, y)] + (eval_set or []),
                eval_metric=eval_metric,
                callbacks=[
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(verbose),
                ],
            )
            if categorical_feature:
                fit_kwargs["categorical_feature"] = categorical_feature
            estimator.fit(X, y, **fit_kwargs)
            self.estimators_.append(estimator)
            self.feature_importances_gain_  += estimator.booster_.feature_importance(importance_type="gain")
            self.feature_importances_split_ += estimator.booster_.feature_importance(importance_type="split")
            return

        # --- downsampling ---
        minority_cls = y.value_counts().sort_values().index[0]
        majority_cls = y.value_counts().sort_values().index[1]
        logger.info(f"majority class: {majority_cls}  minority class: {minority_cls}")

        X_min = X.loc[y == minority_cls]
        y_min = y.loc[y == minority_cls]
        X_maj = X.loc[y == majority_cls]
        y_maj = y.loc[y == majority_cls]

        kf = KFold(n_splits=self.n_estimators_, shuffle=True, random_state=42)
        for _rest, this in kf.split(y_maj):
            logger.info("  training on a subset")
            X_sub = pd.concat([X_min, X_maj.iloc[this]])
            y_sub = pd.concat([y_min, y_maj.iloc[this]])

            estimator = LGBMClassifier(**deepcopy(self.base_params_))
            fit_kwargs = dict(
                eval_set=[(X_sub, y_sub)] + (eval_set or []),
                eval_metric=eval_metric,
                callbacks=[
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(verbose),
                ],
            )
            if categorical_feature:
                fit_kwargs["categorical_feature"] = categorical_feature
            estimator.fit(X_sub, y_sub, **fit_kwargs)
            self.estimators_.append(estimator)
            self.feature_importances_gain_  += estimator.booster_.feature_importance(importance_type="gain") / self.n_estimators_
            self.feature_importances_split_ += estimator.booster_.feature_importance(importance_type="split") / self.n_estimators_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        for estimator in self.estimators_:
            proba += estimator.predict_proba(
                X, num_iteration=estimator.best_iteration_
            ) / len(self.estimators_)
        return proba


# ─── Outer KFold training loop (mirrors lgb1.ipynb cell 23) ──────────────────

def train_with_kfold(
    data: pd.DataFrame,          # full feature matrix INCLUDING TARGET
    y:    pd.Series,
    meanenc_feats: list[str],
    cat_feats:     list[str],
    n_splits: int    = 5,
    n_bag:    int    = 3,
) -> tuple[list[BaggingClassifier], np.ndarray, list[str]]:
    """
    5-fold outer loop matching lgb1.ipynb cell 23 exactly:
      - mean_encode applied per fold (train+val+test combined as val_test)
      - BaggingClassifier with 3 sub-estimators per fold
      - random_state = n_fold * 619  (per reference)

    Returns:
        (list_of_classifiers, oof_preds_on_train, feature_names)
    """
    # Features to exclude from model input
    excluded_feats = ["SK_ID_CURR", "TARGET"] + EXCLUDED_FEATS_SUFFIXES
    # make sure we only exclude cols that actually exist
    excluded_feats = [f for f in excluded_feats if f in data.columns]

    oof_preds = np.zeros(data.shape[0])
    classifiers = []
    scores = []

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=90210)

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data, data["TARGET"])):
        trn = data.iloc[trn_idx]
        val = data.iloc[val_idx]

        logger.info(f"\nFold {n_fold + 1}/{n_splits}  doing mean_encoding...")
        trn_enc, val_enc = mean_encode(trn, val, meanenc_feats, "TARGET", drop=True)

        # Feature list (same for both trn and val after mean_encode)
        features = [f for f in trn_enc.columns if f not in excluded_feats]

        trn_x = trn_enc[features].copy()
        val_x = val_enc[features].copy()
        trn_y = trn_enc["TARGET"]
        val_y = val_enc["TARGET"]

        # Encode any remaining object columns (e.g. _mode aggregations from aux tables)
        for col in trn_x.select_dtypes(include="object").columns:
            trn_x[col], indexer = pd.factorize(trn_x[col])
            val_x[col] = indexer.get_indexer(val_x[col])

        params = deepcopy(LGB_PARAMS)
        params["random_state"] = n_fold * 619

        # Only keep cat_feats that still exist in features after mean_encode drop
        # and ensure they are numeric dtype (LightGBM requirement)
        features_set = set(features)
        active_cat_feats = [f for f in cat_feats if f in features_set
                            and trn_x[f].dtype != object]

        clf = BaggingClassifier(base_params=params, n_estimators=n_bag)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(val_x, val_y)],
            eval_metric="auc",
            verbose=200,
            early_stopping_rounds=100,
            categorical_feature=active_cat_feats if active_cat_feats else "auto",
        )
        classifiers.append(clf)

        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
        fold_score = roc_auc_score(val_y, oof_preds[val_idx])
        scores.append(fold_score)
        logger.info(f"Fold {n_fold + 1} AUC: {fold_score:.6f}")

        del trn, val, trn_enc, val_enc, trn_x, val_x
        gc.collect()

    oof_auc = roc_auc_score(y, oof_preds)
    logger.info(f"\nFull AUC score {oof_auc:.6f} +- {np.std(scores):.4f}")

    # Feature names = features from last fold (stable across folds)
    excluded_feats_set = set(excluded_feats)
    # Recompute for return
    trn_tmp, val_tmp = data.iloc[:10], data.iloc[:5]
    trn_tmp2, _ = mean_encode(trn_tmp, val_tmp, meanenc_feats, "TARGET", drop=True)
    feature_names = [f for f in trn_tmp2.columns if f not in excluded_feats_set]
    del trn_tmp, val_tmp, trn_tmp2

    return classifiers, oof_preds, feature_names


# ─── Main training entry point ────────────────────────────────────────────────

def run_training(
    data_dir: Path,
    output_dir: Path,
    test_size: float = 0.20,
    n_splits: int    = 5,
    n_bag:    int    = 3,
    random_state: int = 42,
) -> None:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build feature matrix ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1/4: Feature engineering")
    logger.info("=" * 60)
    X, y, meanenc_feats, cat_feats = build_all_features(data_dir)
    logger.info(f"Feature matrix ready: {X.shape} | default rate: {y.mean():.1%}")
    logger.info(f"meanenc_feats: {len(meanenc_feats)}  cat_feats: {len(cat_feats)}")

    # ── Stratified train/test split ─────────────────────────────────────────
    logger.info("\nStep 2/4: Stratified split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"  Train: {X_train.shape} ({y_train.mean():.1%} default)")
    logger.info(f"  Test:  {X_test.shape}  ({y_test.mean():.1%} default)")

    # Build combined DataFrames that include TARGET (needed for mean_encode)
    train_data = X_train.copy()
    train_data["TARGET"] = y_train.values
    test_x = X_test.copy()

    del X, X_train; gc.collect()

    # ── StratifiedKFold training ─────────────────────────────────────────────
    logger.info("\nStep 3/4: StratifiedKFold training (with mean encoding + bagging)")
    logger.info("=" * 60)
    classifiers, oof_preds, feature_names = train_with_kfold(
        train_data, y_train,
        meanenc_feats=meanenc_feats,
        cat_feats=cat_feats,
        n_splits=n_splits,
        n_bag=n_bag,
    )

    # ── Test evaluation ──────────────────────────────────────────────────────
    logger.info("\nStep 4/4: Test set evaluation")
    # For test, we need to mean-encode with full train data as reference
    # Use entire train_data as "train" and test_x as "val"
    logger.info("  Applying mean encoding to test set (using full train as reference)...")
    train_data_copy = train_data.copy()
    test_x_copy = test_x.copy()
    test_x_copy["TARGET"] = 0  # dummy for mean_encode interface
    _, test_enc = mean_encode(train_data_copy, test_x_copy, meanenc_feats, "TARGET", drop=True)
    excluded_feats = set(["SK_ID_CURR", "TARGET"] + EXCLUDED_FEATS_SUFFIXES)
    features = [f for f in test_enc.columns if f not in excluded_feats]
    del train_data_copy, test_x_copy

    # Average predictions from all fold classifiers
    test_preds = np.zeros(len(test_x))
    for clf in classifiers:
        test_preds += clf.predict_proba(test_enc[features])[:, 1] / len(classifiers)

    test_auc = roc_auc_score(y_test, test_preds)
    logger.info(f"  Test AUC: {test_auc:.6f}")

    # ── Save artifacts ───────────────────────────────────────────────────────
    # Save all classifiers (one per fold)
    model_path = output_dir / "lgbm_ref_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "classifiers": classifiers,
            "feature_names": feature_names,
            "meanenc_feats": meanenc_feats,
            "cat_feats": cat_feats,
        }, f)
    logger.info(f"\nModel saved → {model_path}")

    # OOF predictions
    oof_df = pd.DataFrame({"oof_pred": oof_preds, "target": y_train.values})
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # Test predictions
    test_df = pd.DataFrame({"test_pred": test_preds, "target": y_test.values})
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Feature names
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Feature importance (averaged across all fold classifiers and sub-estimators)
    importance_gain  = np.zeros(len(feature_names))
    importance_split = np.zeros(len(feature_names))
    for clf in classifiers:
        if clf.feature_importances_gain_ is not None:
            importance_gain  += clf.feature_importances_gain_  / len(classifiers)
            importance_split += clf.feature_importances_split_ / len(classifiers)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain":  importance_gain,
        "importance_split": importance_split,
    }).sort_values("importance_gain", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done in {elapsed / 60:.1f} min")
    logger.info(f"  OOF AUC:  {roc_auc_score(y_train, oof_preds):.6f}")
    logger.info(f"  Test AUC: {test_auc:.6f}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on Home Credit (lgb1.ipynb mirror)"
    )
    parser.add_argument("--data-dir",   required=True, help="Path to Home Credit CSV directory")
    parser.add_argument("--output-dir", default="models/", help="Directory to save model artifacts")
    parser.add_argument("--test-size",  type=float, default=0.20)
    parser.add_argument("--n-splits",   type=int,   default=5)
    parser.add_argument("--n-bag",      type=int,   default=3,
                        help="Number of bagging sub-estimators per fold (reference=3)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    run_training(
        data_dir     = Path(args.data_dir),
        output_dir   = Path(args.output_dir),
        test_size    = args.test_size,
        n_splits     = args.n_splits,
        n_bag        = args.n_bag,
        random_state = args.seed,
    )


if __name__ == "__main__":
    main()
