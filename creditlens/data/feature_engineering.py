"""
CreditLens Feature Engineering — Home Credit → Production features.

This module transforms raw Home Credit tables into the 25-feature production
feature vector as defined in the technical design document (Section 6.2).

Each production feature is mapped from specific Home Credit columns with
clearly documented engineering logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Bureau Features ─────────────────────────────────────────────────────────


def engineer_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau.csv per SK_ID_CURR → CIC proxy features.

    Produces:
        - cic_score_proxy: weighted mean of external sources (added later from app)
        - debt_group_proxy: worst credit status
        - num_active_loans: count of active credits
        - total_outstanding: sum of credit amounts with overdue
    """
    logger.info("Engineering bureau features...")

    agg = bureau.groupby("SK_ID_CURR").agg(
        num_active_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        num_closed_loans=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        max_overdue=("AMT_CREDIT_SUM_OVERDUE", "max"),
        avg_days_credit=("DAYS_CREDIT", "mean"),
        num_bureau_records=("SK_ID_BUREAU", "count"),
    ).reset_index()

    # debt_group_proxy: 1=current (no overdue), 2=watchlist, 3-5=bad
    agg["debt_group_proxy"] = np.where(
        agg["max_overdue"] == 0, 1,
        np.where(agg["max_overdue"] < 50000, 2, 3)
    )

    logger.info(f"Bureau features: {agg.shape[0]:,} applicants")
    return agg


# ─── Installment Payment Features ────────────────────────────────────────────


def engineer_installment_features(installments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments.csv per SK_ID_CURR.

    Produces:
        - income_stability_index: 1 - CV of payment amounts (proxy)
        - payment_consistency: stability of installment payments
        - avg_payment_delay: average days between due date and payment
    """
    logger.info("Engineering installment payment features...")

    # Payment timing: negative = paid early, positive = paid late
    installments["payment_delay"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )

    # Payment ratio: actual payment / expected installment
    installments["payment_ratio"] = (
        installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"].replace(0, np.nan)
    )

    agg = installments.groupby("SK_ID_CURR").agg(
        max_instalment_number=("NUM_INSTALMENT_NUMBER", "max"),
        avg_days_instalment=("DAYS_INSTALMENT", "mean"),
        amt_instalment_mean=("AMT_INSTALMENT", "mean"),
        amt_instalment_std=("AMT_INSTALMENT", "std"),
        total_paid=("AMT_PAYMENT", "sum"),
        avg_payment_delay=("payment_delay", "mean"),
        max_payment_delay=("payment_delay", "max"),
        late_payment_count=("payment_delay", lambda x: (x > 0).sum()),
        total_payments_count=("payment_delay", "count"),
    ).reset_index()

    # income_stability_index proxy: 1 - (std / mean) of installment amounts
    agg["income_stability_index"] = (
        1 - (agg["amt_instalment_std"] / agg["amt_instalment_mean"].replace(0, np.nan))
    ).clip(0, 1).fillna(0.5)

    # payment_consistency: % of on-time payments
    agg["payment_consistency"] = (
        1 - agg["late_payment_count"] / agg["total_payments_count"].replace(0, 1)
    ).clip(0, 1)

    logger.info(f"Installment features: {agg.shape[0]:,} applicants")
    return agg


# ─── POS Cash Balance Features ───────────────────────────────────────────────


def engineer_pos_cash_features(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv per SK_ID_CURR.

    Produces:
        - debt_service_behavior: based on DPD (Days Past Due)
        - overdraft_count_6m: count of months with DPD > 0 in last 6 entries
    """
    logger.info("Engineering POS cash balance features...")

    agg = pos_cash.groupby("SK_ID_CURR").agg(
        max_dpd=("SK_DPD", "max"),
        max_dpd_def=("SK_DPD_DEF", "max"),
        mean_dpd=("SK_DPD", "mean"),
        pos_count=("SK_DPD", "count"),
    ).reset_index()

    # debt_service_behavior proxy: ON_TIME / LATE_1_30 / LATE_31_60
    agg["debt_service_behavior"] = np.where(
        agg["max_dpd"] == 0, "ON_TIME",
        np.where(agg["max_dpd"] <= 30, "LATE_1_30", "LATE_31_60")
    )

    # overdraft_count_6m proxy: last 6 entries with DPD > 0
    last_6 = pos_cash.sort_values("MONTHS_BALANCE").groupby("SK_ID_CURR").tail(6)
    overdraft_counts = last_6[last_6["SK_DPD"] > 0].groupby("SK_ID_CURR").size().reset_index(name="overdraft_count_6m")

    agg = agg.merge(overdraft_counts, on="SK_ID_CURR", how="left")
    agg["overdraft_count_6m"] = agg["overdraft_count_6m"].fillna(0).astype(int)

    logger.info(f"POS cash features: {agg.shape[0]:,} applicants")
    return agg


# ─── Credit Card Balance Features ────────────────────────────────────────────


def engineer_credit_card_features(cc_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv per SK_ID_CURR.

    Produces:
        - credit_utilization: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
        - avg_drawings: average monthly drawings (spending proxy)
    """
    logger.info("Engineering credit card balance features...")

    agg = cc_balance.groupby("SK_ID_CURR").agg(
        avg_balance=("AMT_BALANCE", "mean"),
        avg_credit_limit=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        avg_drawings=("AMT_DRAWINGS_CURRENT", "mean"),
        max_balance=("AMT_BALANCE", "max"),
    ).reset_index()

    # Credit utilization
    agg["credit_utilization"] = (
        agg["avg_balance"] / agg["avg_credit_limit"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    logger.info(f"Credit card features: {agg.shape[0]:,} applicants")
    return agg


# ─── Application Features ────────────────────────────────────────────────────


def engineer_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from application_train.csv.

    Produces:
        - age: from DAYS_BIRTH
        - employment_duration: from DAYS_EMPLOYED
        - dti_ratio: AMT_ANNUITY / (AMT_INCOME_TOTAL / 12)
        - cic_score_proxy: weighted EXT_SOURCE combination
        - income features
    """
    logger.info("Engineering application features...")

    df = app[["SK_ID_CURR", "TARGET"]].copy()

    # Age
    df["age"] = (-app["DAYS_BIRTH"] / 365.25).astype(int)

    # Gender (binary encoding)
    df["gender"] = (app["CODE_GENDER"] == "M").astype(int)

    # ID verified proxy (always True for Home Credit dataset)
    df["id_verified"] = 1

    # Employment duration in months
    df["employment_duration_months"] = np.where(
        app["DAYS_EMPLOYED"] > 0,  # 365243 means unemployed
        0,
        (-app["DAYS_EMPLOYED"] / 30).astype(int)
    )

    # Flag: owns car (proxy for collateral)
    df["flag_own_car"] = (app["FLAG_OWN_CAR"] == "Y").astype(int)

    # Financial amounts
    df["loan_amount_vnd"] = app["AMT_CREDIT"].fillna(0)
    df["income_total"] = app["AMT_INCOME_TOTAL"].fillna(0)
    df["annuity"] = app["AMT_ANNUITY"].fillna(0)

    # DTI ratio: monthly debt / monthly income
    monthly_income = df["income_total"] / 12
    df["dti_ratio"] = (df["annuity"] / monthly_income.replace(0, np.nan)).clip(0, 2).fillna(0)

    # Term months proxy
    df["term_months"] = np.where(
        df["annuity"] > 0,
        (df["loan_amount_vnd"] / df["annuity"]).clip(0, 360),
        0
    ).astype(int)

    # CIC score proxy from EXT_SOURCE variables
    # Weighted: 0.5 × EXT_SOURCE_2 + 0.3 × EXT_SOURCE_3 + 0.2 × EXT_SOURCE_1
    ext_1 = app["EXT_SOURCE_1"].fillna(0.5)
    ext_2 = app["EXT_SOURCE_2"].fillna(0.5)
    ext_3 = app["EXT_SOURCE_3"].fillna(0.5)
    cic_raw = 0.5 * ext_2 + 0.3 * ext_3 + 0.2 * ext_1
    df["cic_score_proxy"] = (cic_raw * 600 + 150).clip(150, 750).astype(int)

    # Thin file flag: if all EXT_SOURCE are NaN → thin file
    df["thin_file_flag"] = (
        app["EXT_SOURCE_1"].isna() & app["EXT_SOURCE_2"].isna() & app["EXT_SOURCE_3"].isna()
    ).astype(int)

    # Salary pattern detected proxy
    # Per document.md Section 6.2: DAYS_EMPLOYED > 0 AND income_stability > 0.7
    # Note: DAYS_EMPLOYED < 0 means currently employed (Home Credit encoding)
    df["salary_pattern_detected"] = 0  # default
    # Will be updated after income_stability_index is computed during merge

    logger.info(f"Application features: {df.shape[0]:,} applicants, {df.shape[1]} features")
    return df


# ─── Merge All Features ──────────────────────────────────────────────────────


def build_feature_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the complete unified feature matrix from all Home Credit tables.

    This is the central function that produces the 25-feature production vector
    by merging features engineered from each table.

    Args:
        tables: Dict of {table_name: DataFrame} from loader.load_all_tables()

    Returns:
        DataFrame with SK_ID_CURR as index, TARGET column, and all engineered features.
    """
    logger.info("Building unified feature matrix...")

    # 1. Application features (base)
    app_feats = engineer_application_features(tables["application_train"])

    # 2. Bureau features
    bureau_feats = engineer_bureau_features(tables["bureau"])

    # 3. Installment features
    install_feats = engineer_installment_features(tables["installments_payments"])

    # 4. POS cash features
    pos_feats = engineer_pos_cash_features(tables["pos_cash_balance"])

    # 5. Credit card features
    cc_feats = engineer_credit_card_features(tables["credit_card_balance"])

    # Merge all on SK_ID_CURR
    result = app_feats
    for feats in [bureau_feats, install_feats, pos_feats, cc_feats]:
        result = result.merge(feats, on="SK_ID_CURR", how="left")

    # Fix salary_pattern_detected AFTER income_stability_index is available
    # Per document.md Section 6.2: DAYS_EMPLOYED > 0 AND income_stability > 0.7
    employed = tables["application_train"]["DAYS_EMPLOYED"] < 0  # negative = employed
    has_stability = result["income_stability_index"].fillna(0) > 0.7
    result["salary_pattern_detected"] = (employed.values & has_stability.values).astype(int)

    # Compute inflow_outflow_ratio proxy
    # (AMT_INCOME_TOTAL / 12) / (AMT_ANNUITY + avg_drawings)
    monthly_income = result["income_total"] / 12
    monthly_outflow = result["annuity"] + result.get("avg_drawings", pd.Series(0, index=result.index)).fillna(0)
    result["inflow_outflow_ratio"] = (
        monthly_income / monthly_outflow.replace(0, np.nan)
    ).clip(0, 5).fillna(1.0)

    # Bill payment ratio proxy (set to mean for HC dataset — no direct proxy)
    result["regular_bill_payment_ratio"] = np.where(
        result["payment_consistency"].notna(),
        result["payment_consistency"],
        0.5
    )

    # Max single outflow ratio proxy
    result["max_single_outflow_ratio"] = (
        result.get("max_balance", pd.Series(0, index=result.index)).fillna(0) /
        monthly_income.replace(0, np.nan)
    ).clip(0, 2).fillna(0)

    # Imputation flags (for HC dataset, assume all are real data)
    result["income_imputed_flag"] = 0
    result["imputation_confidence"] = 1.0

    # Avg monthly inflow proxy
    result["avg_monthly_inflow_vnd"] = monthly_income

    logger.info(f"Feature matrix built: {result.shape[0]:,} rows, {result.shape[1]} columns")
    return result


# ─── Select Production Features ──────────────────────────────────────────────

PRODUCTION_FEATURES = [
    # Identity & KYC
    "age", "gender", "id_verified",
    # Credit Bureau
    "cic_score_proxy", "debt_group_proxy", "num_active_loans", "thin_file_flag",
    # Transaction Behavioral
    "avg_monthly_inflow_vnd", "income_stability_index", "salary_pattern_detected",
    "regular_bill_payment_ratio", "debt_service_behavior", "overdraft_count_6m",
    "inflow_outflow_ratio", "max_single_outflow_ratio",
    # LLM Semantic (A2-A) — proxied for HC dataset
    # These will be dummy features during training, real in production
    # Loan Terms
    "loan_amount_vnd", "term_months", "dti_ratio",
    # Imputed fields
    "income_imputed_flag", "imputation_confidence",
]

PILOT_FEATURES = [
    "age", "gender", "id_verified",
    "cic_score_proxy", "debt_group_proxy", "num_active_loans", "thin_file_flag",
    "income_stability_index", "salary_pattern_detected",
    "dti_ratio",
]


def select_features(
    feature_matrix: pd.DataFrame,
    feature_set: str = "production",
) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns from the full feature matrix.

    Args:
        feature_matrix: Full engineered feature matrix.
        feature_set: "production" (25 features) or "pilot" (10 core features).

    Returns:
        Tuple of (X, y) DataFrames.
    """
    features = PRODUCTION_FEATURES if feature_set == "production" else PILOT_FEATURES

    # Filter to features that exist in the matrix
    available = [f for f in features if f in feature_matrix.columns]
    missing = [f for f in features if f not in feature_matrix.columns]
    if missing:
        logger.warning(f"Missing features (will be filled with defaults): {missing}")

    X = feature_matrix[available].copy()

    # Encode categorical debt_service_behavior
    if "debt_service_behavior" in X.columns:
        dsb_map = {"ON_TIME": 0, "LATE_1_30": 1, "LATE_31_60": 2, "MISSING": 3}
        X["debt_service_behavior"] = X["debt_service_behavior"].map(dsb_map).fillna(3).astype(int)

    y = feature_matrix["TARGET"]

    logger.info(f"Selected {len(available)} features ({feature_set} set)")
    return X, y
