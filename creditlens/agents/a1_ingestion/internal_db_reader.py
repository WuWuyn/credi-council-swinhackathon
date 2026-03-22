"""
CreditLens A1 — Internal Database Reader.

Reads mock internal loan system data (previous applications, POS cash,
installments, credit card balances) from JSON files.

In production, this would query the internal loan management database.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class InternalDBReader:
    """Reads internal loan system data.

    # LOCAL_SUB: Mock reads from JSON. Production: query internal SQL database.

    Returns DataFrames matching Home Credit dataset structure:
    - previous_application (37 columns)
    - POS_CASH_balance (8 columns)
    - installments_payments (8 columns)
    - credit_card_balance (23 columns)
    """

    def read(self, internal_db_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
        """Read internal DB data and return as DataFrames.

        Args:
            internal_db_path: Path to mock JSON file.

        Returns:
            Dict of DataFrames: {table_name: DataFrame}
        """
        if internal_db_path is None:
            logger.warning("No internal DB path provided — returning empty DataFrames")
            return self._empty_result()

        path = Path(internal_db_path)
        if not path.exists():
            logger.warning(f"Internal DB file not found: {path}")
            return self._empty_result()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sk_id_curr = data.get("SK_ID_CURR", 0)
        logger.info(f"Internal DB loaded for SK_ID_CURR={sk_id_curr}")

        result = {}

        # Previous Applications
        prev_apps = data.get("previous_applications", [])
        if prev_apps:
            result["previous_application"] = pd.DataFrame(prev_apps)
            logger.info(f"  previous_application: {len(prev_apps)} records")
        else:
            result["previous_application"] = pd.DataFrame()

        # POS Cash Balance
        pos_cash = data.get("pos_cash_balance", [])
        if pos_cash:
            result["POS_CASH_balance"] = pd.DataFrame(pos_cash)
            logger.info(f"  POS_CASH_balance: {len(pos_cash)} records")
        else:
            result["POS_CASH_balance"] = pd.DataFrame()

        # Installments Payments
        installments = data.get("installments_payments", [])
        if installments:
            result["installments_payments"] = pd.DataFrame(installments)
            logger.info(f"  installments_payments: {len(installments)} records")
        else:
            result["installments_payments"] = pd.DataFrame()

        # Credit Card Balance
        credit_card = data.get("credit_card_balance", [])
        if credit_card:
            result["credit_card_balance"] = pd.DataFrame(credit_card)
            logger.info(f"  credit_card_balance: {len(credit_card)} records")
        else:
            result["credit_card_balance"] = pd.DataFrame()

        return result

    def _empty_result(self) -> dict[str, pd.DataFrame]:
        """Return empty DataFrames for all tables."""
        return {
            "previous_application": pd.DataFrame(),
            "POS_CASH_balance": pd.DataFrame(),
            "installments_payments": pd.DataFrame(),
            "credit_card_balance": pd.DataFrame(),
        }
