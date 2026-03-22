"""
Pre-compute training statistics needed for single-customer feature engineering.

Run ONCE on the full training dataset to generate:
  models/fe_stats.pkl — medians, factorize maps, feature column order

Usage:
    python training/precompute_fe_stats.py --data-dir home-credit-default-risk/
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.feature_engineering import (
    REJECTED_APP_FEATURES,
    build_all_features,
    sanitize_column_names,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def precompute_stats(data_dir: str, output_path: str = "models/fe_stats.pkl"):
    """Pre-compute all statistics needed by single-customer FE.

    Saves a dict with:
        - feature_names: ordered list of 753 feature names
        - inc_by_org: median income by ORGANIZATION_TYPE
        - factorize_maps: {col_name: {value: int}} for label encoding
        - group_medians: {group_col: {group_val: median_income}}
        - global_scores_std_mean: mean of NEW_SCORES_STD for fillna
        - global_target_mean: overall TARGET mean for mean encoding
        - mean_encode_maps: {feature: {value: mean_target}} for val encoding
    """
    data_path = Path(data_dir)

    logger.info("Building full feature matrix to extract statistics...")
    full_df, y, meanenc_feats, cat_feats = build_all_features(data_path)

    feature_names = list(full_df.columns)
    logger.info(f"Feature matrix shape: {full_df.shape}")
    logger.info(f"Feature names: {len(feature_names)}")

    # --- Recompute application-level stats from raw data ---
    logger.info("Computing application-level statistics...")
    app = pd.read_csv(data_path / "application_train.csv")

    # Income by organization type
    inc_by_org = app.groupby("ORGANIZATION_TYPE")["AMT_INCOME_TOTAL"].median().to_dict()

    # Group medians for mean-income features
    group_medians = {}
    for grp_col, col_name in [
        ("CODE_GENDER",          "gender_mean_income"),
        ("FLAG_OWN_CAR",         "own_car_mean_income"),
        ("FLAG_OWN_REALTY",      "own_realty_mean_income"),
        ("NAME_FAMILY_STATUS",   "family_status_mean_income"),
    ]:
        grp_map = app.groupby(grp_col)["AMT_INCOME_TOTAL"].median().to_dict()
        group_medians[col_name] = grp_map
        logger.info(f"  {col_name}: {len(grp_map)} groups")

    # Factorize maps for categorical columns
    factorize_maps = {}
    for col in app.select_dtypes(include="object").columns:
        unique_vals = app[col].dropna().unique()
        factorize_maps[col] = {v: i for i, v in enumerate(unique_vals)}
        logger.info(f"  factorize {col}: {len(unique_vals)} values")

    # Scores STD mean
    ext_scores = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    scores_std_mean = float(ext_scores.mean())

    # Global target mean
    global_target_mean = float(app["TARGET"].mean())

    # Mean encode maps from training (for the val encoding path)
    mean_encode_maps = {}
    for feat in meanenc_feats:
        if feat in app.columns:
            encode_map = app.groupby(feat)["TARGET"].mean().to_dict()
            mean_encode_maps[feat] = encode_map

    # --- Save ---
    stats = {
        "feature_names": feature_names,
        "inc_by_org": inc_by_org,
        "factorize_maps": factorize_maps,
        "group_medians": group_medians,
        "global_scores_std_mean": scores_std_mean,
        "global_target_mean": global_target_mean,
        "mean_encode_maps": mean_encode_maps,
        "meanenc_feats": meanenc_feats,
        "cat_feats": cat_feats,
        "rejected_app_features": REJECTED_APP_FEATURES,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Statistics saved to {output}")
    logger.info(f"  Feature names: {len(feature_names)}")
    logger.info(f"  Organization types: {len(inc_by_org)}")
    logger.info(f"  Factorize maps: {len(factorize_maps)}")
    logger.info(f"  Mean encode maps: {len(mean_encode_maps)}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="home-credit-default-risk/")
    parser.add_argument("--output", default="models/fe_stats.pkl")
    args = parser.parse_args()
    precompute_stats(args.data_dir, args.output)
