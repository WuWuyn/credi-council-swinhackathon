"""
CREDICOUNCIL API — Data Access Layer.

Centralized helpers for reading customer data from the file system.

Architecture:
    data/mock/customer_XXX/   → INPUT data (raw documents, application_row)
    data/output/customer_XXX/ → OUTPUT from pipeline runs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from credicouncil.api.config import MOCK_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ─── ID normalisation ────────────────────────────────────────────────────────

def normalize_folder_id(customer_id: str) -> str:
    """Normalize customer_id to folder name: '1' → 'customer_001'."""
    folder_id = customer_id.zfill(3) if customer_id.isdigit() else customer_id
    if not folder_id.startswith("customer_"):
        folder_id = f"customer_{folder_id}"
    return folder_id


# ─── Customer map ────────────────────────────────────────────────────────────

def load_customer_map() -> dict:
    """Load customer_map.json (SK_ID → folder metadata)."""
    map_path = MOCK_DIR / "customer_map.json"
    if map_path.exists():
        with open(map_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── Customer summary ────────────────────────────────────────────────────────

def get_customer_summary(folder: Path) -> dict:
    """Extract brief summary from a customer's credit_report.json + application_row.json."""
    summary: dict[str, Any] = {}

    report_path = folder / "credit_report.json"
    app_path = folder / "application_row.json"

    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        cinfo = report.get("customer_info", {})
        exec_summary = report.get("executive_summary", {})
        summary.update(
            {
                "gender": cinfo.get("gender", "N/A"),
                "age": cinfo.get("age", 0),
                "income_type": cinfo.get("income_type", "N/A"),
                "education": cinfo.get("education", "N/A"),
                "housing": cinfo.get("housing", "N/A"),
                "loan_purpose": cinfo.get("loan_purpose", "N/A"),
                "own_realty": cinfo.get("own_realty", "N"),
                "own_car": cinfo.get("own_car", "N"),
                "family_status": cinfo.get("family_status", "N/A"),
                "credit_score": exec_summary.get("credit_score", 0),
                "risk_band": exec_summary.get("risk_band", "N/A"),
                "pd_pct": exec_summary.get("pd_pct", 0),
                "recommendation": exec_summary.get("recommendation", "N/A"),
                "five_c_total": exec_summary.get("five_c_total", 0),
                "five_c_scores": exec_summary.get("five_c_scores", {}),
                "has_report": True,
            }
        )

    if app_path.exists():
        with open(app_path, encoding="utf-8") as f:
            app_row = json.load(f)
        summary.update(
            {
                "sk_id_curr": app_row.get("SK_ID_CURR"),
                "amt_income_total": app_row.get("AMT_INCOME_TOTAL", 0),
                "amt_credit": app_row.get("AMT_CREDIT", 0),
                "amt_annuity": app_row.get("AMT_ANNUITY", 0),
                "amt_goods_price": app_row.get("AMT_GOODS_PRICE", 0),
                "contract_type": app_row.get("NAME_CONTRACT_TYPE", "N/A"),
            }
        )

    return summary


def check_output_exists(folder_id: str) -> bool:
    """Check whether pipeline output exists for a given customer."""
    output_folder = OUTPUT_DIR / folder_id
    return (output_folder / "credit_report.json").exists()


# ─── Load customer report data ───────────────────────────────────────────────

def load_customer_report(customer_id: str) -> tuple[dict, dict]:
    """
    Load credit_report.json + shap_values.json for display.

    Reads from data/output/ only — no mock fallback.
    """
    folder_id = normalize_folder_id(customer_id)
    output_folder = OUTPUT_DIR / folder_id

    report_data: dict = {}
    shap_data: dict = {}

    report_path = output_folder / "credit_report.json"
    if not report_path.exists():
        return {}, {}

    with open(report_path, encoding="utf-8") as f:
        report_data = json.load(f)

    shap_path = output_folder / "shap_values.json"
    if shap_path.exists():
        with open(shap_path, encoding="utf-8") as f:
            shap_data = json.load(f)

    return report_data, shap_data


