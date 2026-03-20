"""
CreditLens Preprocessing — Data cleaning, splitting, and balancing.

Handles:
- Missing value imputation
- Train/Validation/Test split (70/15/15 stratified)
- ADASYN oversampling for class imbalance
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def clean_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Clean feature matrix: handle NaN, infinities, and type issues.

    Args:
        X: Raw feature DataFrame.

    Returns:
        Cleaned DataFrame with no NaN or infinity values.
    """
    logger.info(f"Cleaning feature matrix: {X.shape}")

    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # Count NaN per column
    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.info(f"Columns with NaN: {dict(nan_cols)}")

    # Fill numeric NaN with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    # Fill categorical NaN with mode
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    for col in cat_cols:
        if X[col].isna().any():
            mode_val = X[col].mode()
            X[col] = X[col].fillna(mode_val[0] if len(mode_val) > 0 else "UNKNOWN")

    logger.info(f"Cleaned: {X.isna().sum().sum()} remaining NaN values")
    return X


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split data into train/validation/test sets with stratification.

    Follows the exact split ratios from the design document:
    - Train: 70% (215,258 records) — ADASYN applied
    - Validation: 15% (46,127 records) — real distribution
    - Test: 15% (46,126 records) — locked, no peeking

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion for test set.
        val_size: Proportion for validation set (from remaining after test).
        random_state: Random seed for reproducibility.

    Returns:
        Dict with X_train, X_val, X_test, y_train, y_val, y_test.
    """
    logger.info(f"Splitting data: {X.shape[0]:,} total samples")

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: separate validation from train
    # val_size is proportion of REMAINING data after test split
    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_proportion,
        random_state=random_state,
        stratify=y_temp,
    )

    logger.info(
        f"Split results — Train: {X_train.shape[0]:,} ({y_train.mean():.1%} default) | "
        f"Val: {X_val.shape[0]:,} ({y_val.mean():.1%} default) | "
        f"Test: {X_test.shape[0]:,} ({y_test.mean():.1%} default)"
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def apply_adasyn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply ADASYN oversampling to handle class imbalance.

    The Home Credit dataset has ~8% default rate, making it imbalanced.
    ADASYN (Adaptive Synthetic Sampling) generates synthetic samples
    for the minority class, targeting a 5:1 ratio (sampling_strategy=0.2).

    Args:
        X_train: Training features.
        y_train: Training targets.
        sampling_strategy: Target ratio of minority to majority class.
        random_state: Random seed.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    from imblearn.over_sampling import ADASYN

    logger.info(
        f"Applying ADASYN — Before: {X_train.shape[0]:,} samples, "
        f"default rate: {y_train.mean():.1%}"
    )

    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    logger.info(
        f"After ADASYN: {X_res.shape[0]:,} samples, "
        f"default rate: {y_res.mean():.1%} "
        f"(+{X_res.shape[0] - X_train.shape[0]:,} synthetic samples)"
    )

    return X_res, y_res
