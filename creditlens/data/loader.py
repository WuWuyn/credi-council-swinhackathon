"""
CreditLens Data Loader — Home Credit Default Risk dataset.

Handles loading and merging all 8 tables from the Home Credit dataset.
Optimized for memory with proper dtypes and chunked reading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from creditlens.config.settings import DATA_DIR

logger = logging.getLogger(__name__)


# ─── Column dtypes for memory optimization ────────────────────────────────────

APPLICATION_DTYPES: dict[str, str] = {
    "SK_ID_CURR": "int32",
    "TARGET": "int8",
    "CODE_GENDER": "category",
    "FLAG_OWN_CAR": "category",
    "FLAG_OWN_REALTY": "category",
    "CNT_CHILDREN": "int8",
    "AMT_INCOME_TOTAL": "float32",
    "AMT_CREDIT": "float32",
    "AMT_ANNUITY": "float32",
    "AMT_GOODS_PRICE": "float32",
    "NAME_CONTRACT_TYPE": "category",
    "NAME_INCOME_TYPE": "category",
    "NAME_EDUCATION_TYPE": "category",
    "NAME_FAMILY_STATUS": "category",
    "NAME_HOUSING_TYPE": "category",
    "DAYS_BIRTH": "int32",
    "DAYS_EMPLOYED": "int32",
    "DAYS_REGISTRATION": "float32",
    "DAYS_ID_PUBLISH": "int32",
    "OCCUPATION_TYPE": "category",
    "ORGANIZATION_TYPE": "category",
    "EXT_SOURCE_1": "float32",
    "EXT_SOURCE_2": "float32",
    "EXT_SOURCE_3": "float32",
}


def load_application_train(data_dir: Path | None = None) -> pd.DataFrame:
    """Load application_train.csv — main table with TARGET column.

    Args:
        data_dir: Path to the Home Credit data directory.

    Returns:
        DataFrame with 307,511 rows, dtypes optimized.
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / "application_train.csv"
    logger.info(f"Loading application_train from {path}")

    df = pd.read_csv(path, dtype={k: v for k, v in APPLICATION_DTYPES.items()})
    logger.info(f"Loaded application_train: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_application_test(data_dir: Path | None = None) -> pd.DataFrame:
    """Load application_test.csv — test set without TARGET."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / "application_test.csv"
    logger.info(f"Loading application_test from {path}")
    return pd.read_csv(path)


def load_bureau(data_dir: Path | None = None) -> pd.DataFrame:
    """Load bureau.csv — credit bureau history.

    1.7M rows with credit history from external credit bureaus.
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / "bureau.csv"
    logger.info(f"Loading bureau from {path}")

    df = pd.read_csv(path, dtype={"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32"})
    logger.info(f"Loaded bureau: {df.shape[0]:,} rows")
    return df


def load_bureau_balance(data_dir: Path | None = None) -> pd.DataFrame:
    """Load bureau_balance.csv — monthly balance of bureau credits."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / "bureau_balance.csv"
    logger.info(f"Loading bureau_balance from {path}")

    df = pd.read_csv(path, dtype={"SK_ID_BUREAU": "int32"})
    logger.info(f"Loaded bureau_balance: {df.shape[0]:,} rows")
    return df


def load_installments_payments(data_dir: Path | None = None) -> pd.DataFrame:
    """Load installments_payments.csv — installment payment history.

    13.6M rows — the largest table. Used for payment consistency (income_stability proxy).
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / "installments_payments.csv"
    logger.info(f"Loading installments_payments from {path}")

    df = pd.read_csv(
        path,
        dtype={
            "SK_ID_CURR": "int32",
            "SK_ID_PREV": "int32",
            "NUM_INSTALMENT_VERSION": "float32",
            "NUM_INSTALMENT_NUMBER": "int16",
        },
    )
    logger.info(f"Loaded installments_payments: {df.shape[0]:,} rows")
    return df


def load_credit_card_balance(data_dir: Path | None = None) -> pd.DataFrame:
    """Load credit_card_balance.csv — credit card monthly balances.

    3.8M rows. Used for credit utilization and spending patterns.
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / "credit_card_balance.csv"
    logger.info(f"Loading credit_card_balance from {path}")

    df = pd.read_csv(path, dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"})
    logger.info(f"Loaded credit_card_balance: {df.shape[0]:,} rows")
    return df


def load_pos_cash_balance(data_dir: Path | None = None) -> pd.DataFrame:
    """Load POS_CASH_balance.csv — POS and cash loan monthly balances.

    10M rows. Used for DPD (Days Past Due) → overdraft and debt service proxies.
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / "POS_CASH_balance.csv"
    logger.info(f"Loading POS_CASH_balance from {path}")

    df = pd.read_csv(path, dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"})
    logger.info(f"Loaded POS_CASH_balance: {df.shape[0]:,} rows")
    return df


def load_previous_application(data_dir: Path | None = None) -> pd.DataFrame:
    """Load previous_application.csv — previous loan applications.

    1.67M rows. Used for loan purpose history.
    """
    data_dir = data_dir or DATA_DIR
    path = data_dir / "previous_application.csv"
    logger.info(f"Loading previous_application from {path}")

    df = pd.read_csv(path, dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"})
    logger.info(f"Loaded previous_application: {df.shape[0]:,} rows")
    return df


def load_all_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all 8 tables and return as a dictionary.

    Returns:
        Dict mapping table name to DataFrame.
    """
    data_dir = data_dir or DATA_DIR
    logger.info(f"Loading all Home Credit tables from {data_dir}")

    tables = {
        "application_train": load_application_train(data_dir),
        "bureau": load_bureau(data_dir),
        "bureau_balance": load_bureau_balance(data_dir),
        "installments_payments": load_installments_payments(data_dir),
        "credit_card_balance": load_credit_card_balance(data_dir),
        "pos_cash_balance": load_pos_cash_balance(data_dir),
        "previous_application": load_previous_application(data_dir),
    }

    total_rows = sum(df.shape[0] for df in tables.values())
    logger.info(f"All tables loaded: {len(tables)} tables, {total_rows:,} total rows")
    return tables
