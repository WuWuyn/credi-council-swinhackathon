"""
CreditLens A1 — Bank Statement Parser.

Parses 6-month bank statement CSVs to extract 8 alternative data features.
This is the CORE INNOVATION of CreditLens — enabling credit assessment
for thin-file customers who lack traditional CIC history.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Regex Patterns ───────────────────────────────────────────────────────────

SALARY_PATTERN = re.compile(r"LUONG|SALARY|THU NHAP|TIEN LUONG|CHUYEN LUONG", re.IGNORECASE)
BILL_PATTERN = re.compile(r"DIEN|NUOC|VTC|FPT|VNPT|INTERNET|EVN|ELECTRIC|WATER", re.IGNORECASE)
LOAN_PATTERN = re.compile(r"TRA NO|TRA GOP|KHOAN VAY|LOAN|INSTALMENT|THANH TOAN NO", re.IGNORECASE)


def parse_bank_statement(csv_path: str | Path) -> dict[str, Any]:
    """Parse a 6-month bank statement CSV and extract 8 alternative data features.

    Expected CSV format:
        date, description, amount, running_balance
        (amount > 0 = credit/inflow, amount < 0 = debit/outflow)

    Args:
        csv_path: Path to the bank statement CSV file.

    Returns:
        Dict with 8 features + metadata.

    Raises:
        ValueError: If CSV has fewer than 3 months of data.
    """
    logger.info(f"Parsing bank statement: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Validate minimum data requirement
    df["month"] = df["date"].dt.to_period("M")
    n_months = df["month"].nunique()

    if n_months < 3:
        raise ValueError(
            f"Bank statement has only {n_months} months of data. "
            f"Minimum 3 months required for reliable scoring."
        )

    # ── Feature 1: Average monthly inflow ──
    monthly_inflow = df[df["amount"] > 0].groupby("month")["amount"].sum()
    avg_monthly_inflow = monthly_inflow.mean() if len(monthly_inflow) > 0 else 0

    # ── Feature 2: Income stability index ──
    if len(monthly_inflow) >= 2 and monthly_inflow.mean() > 0:
        income_stability = 1 - (monthly_inflow.std() / monthly_inflow.mean())
        income_stability = max(0, min(1, income_stability))  # clamp to [0, 1]
    else:
        income_stability = 0.5  # default if insufficient data

    # ── Feature 3: Salary pattern detection ──
    # Per design doc: credit ≈ same_amount (±5%), ngày 1-5, regex(LUONG|SALARY|THU NHAP)
    salary_txns = df[
        (df["amount"] > 0) &
        (df["description"].str.contains(SALARY_PATTERN, na=False)) &
        (df["date"].dt.day <= 5)  # salary typically arrives on day 1-5
    ]
    salary_months = salary_txns["month"].nunique()
    salary_detected = salary_months >= 3  # at least 3 months of salary

    # Additional: check if amounts are consistent (±5%)
    if salary_detected and len(salary_txns) >= 3:
        salary_amounts = salary_txns.groupby("month")["amount"].sum()
        salary_cv = salary_amounts.std() / salary_amounts.mean() if salary_amounts.mean() > 0 else 1
        salary_detected = salary_detected and (salary_cv < 0.05)

    # Fallback: also check without day restriction if strict check fails
    if not salary_detected:
        salary_txns_any = df[
            (df["amount"] > 0) &
            (df["description"].str.contains(SALARY_PATTERN, na=False))
        ]
        salary_months_any = salary_txns_any["month"].nunique()
        if salary_months_any >= 3:
            salary_amounts_any = salary_txns_any.groupby("month")["amount"].sum()
            salary_cv_any = salary_amounts_any.std() / salary_amounts_any.mean() if salary_amounts_any.mean() > 0 else 1
            salary_detected = salary_cv_any < 0.05

    # ── Feature 4: Regular bill payment ratio ──
    # Per design doc: % months with bill payments on time (not late > 5 days)
    bill_txns = df[
        (df["amount"] < 0) &
        (df["description"].str.contains(BILL_PATTERN, na=False))
    ]

    if len(bill_txns) > 0:
        # Check for on-time bill payments (within first 5 days of typical due date)
        bill_by_month = bill_txns.groupby("month").agg(
            min_day=("date", lambda x: x.dt.day.min()),
        )
        # Assume bills due by 15th; paid before 20th = on time (< 5 days late)
        on_time_months = (bill_by_month["min_day"] <= 20).sum()
        bill_payment_ratio = on_time_months / n_months if n_months > 0 else 0
    else:
        bill_months = 0
        bill_payment_ratio = 0.0

    # ── Feature 5: Debt service behavior ──
    loan_txns = df[
        (df["amount"] < 0) &
        (df["description"].str.contains(LOAN_PATTERN, na=False))
    ]
    if len(loan_txns) > 0:
        # Check if loan payments are regular and on time
        loan_txns_per_month = loan_txns.groupby("month").size()
        if loan_txns_per_month.min() >= 1:
            debt_service = "ON_TIME"
        else:
            missing_months = n_months - len(loan_txns_per_month)
            if missing_months <= 1:
                debt_service = "LATE_1_30"
            else:
                debt_service = "LATE_31_60"
    else:
        debt_service = "MISSING"

    # ── Feature 6: Overdraft count (6 months) ──
    if "running_balance" in df.columns:
        daily_balance = df.groupby("date")["running_balance"].last()
        overdraft_count = int((daily_balance < 500_000).sum())
    else:
        overdraft_count = 0

    # ── Feature 7: Inflow/Outflow ratio ──
    monthly_outflow = df[df["amount"] < 0].groupby("month")["amount"].sum().abs()
    if len(monthly_outflow) > 0 and monthly_outflow.mean() > 0:
        io_ratio = monthly_inflow.mean() / monthly_outflow.mean()
    else:
        io_ratio = 1.0  # neutral if no outflow data

    # ── Feature 8: Max single outflow ratio ──
    max_outflow = df[df["amount"] < 0]["amount"].abs().max() if len(df[df["amount"] < 0]) > 0 else 0
    max_single_outflow_ratio = max_outflow / avg_monthly_inflow if avg_monthly_inflow > 0 else 0

    features = {
        "avg_monthly_inflow_vnd": round(avg_monthly_inflow),
        "income_stability_index": round(income_stability, 3),
        "salary_pattern_detected": bool(salary_detected),
        "regular_bill_payment_ratio": round(bill_payment_ratio, 3),
        "debt_service_behavior": debt_service,
        "overdraft_count_6m": overdraft_count,
        "inflow_outflow_ratio": round(io_ratio, 3),
        "max_single_outflow_ratio": round(max_single_outflow_ratio, 3),
    }

    # Metadata
    metadata = {
        "n_months": n_months,
        "n_transactions": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "total_inflow": round(df[df["amount"] > 0]["amount"].sum()),
        "total_outflow": round(df[df["amount"] < 0]["amount"].sum()),
    }

    logger.info(f"Bank statement features extracted: {n_months} months, {len(df)} transactions")
    return {"features": features, "metadata": metadata}
