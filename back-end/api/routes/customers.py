"""
CREDICOUNCIL API — Customers Endpoint.

GET /v1/customers — List all available customer profiles from mock data.
GET /v1/customers/{customer_id} — Get customer summary info.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

MOCK_DIR = Path("data/mock")
CUSTOMER_MAP_PATH = MOCK_DIR / "customer_map.json"


def _load_customer_map() -> dict:
    """Load the customer_map.json that maps SK_ID → folder info."""
    if CUSTOMER_MAP_PATH.exists():
        with open(CUSTOMER_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_customer_summary(folder: Path) -> dict:
    """Extract a brief summary from a customer's credit_report.json."""
    report_path = folder / "credit_report.json"
    app_path = folder / "application_row.json"

    summary = {}

    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        cinfo = report.get("customer_info", {})
        exec_summary = report.get("executive_summary", {})
        summary.update({
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
        })
    
    if app_path.exists():
        with open(app_path, encoding="utf-8") as f:
            app_row = json.load(f)
        summary.update({
            "sk_id_curr": app_row.get("SK_ID_CURR"),
            "amt_income_total": app_row.get("AMT_INCOME_TOTAL", 0),
            "amt_credit": app_row.get("AMT_CREDIT", 0),
            "amt_annuity": app_row.get("AMT_ANNUITY", 0),
            "amt_goods_price": app_row.get("AMT_GOODS_PRICE", 0),
            "contract_type": app_row.get("NAME_CONTRACT_TYPE", "N/A"),
        })

    return summary


@router.get("/customers")
async def list_customers():
    """
    List all available customer profiles.

    Returns a list of customers with basic info extracted from their data folders.
    """
    customer_map = _load_customer_map()
    customers = []

    # Build from customer_map.json (authoritative source)
    for sk_id, meta in customer_map.items():
        folder_name = meta.get("dir", "")
        folder_path = MOCK_DIR / folder_name
        
        if not folder_path.exists():
            continue

        # Extract numeric ID from folder name (e.g., "customer_001" -> "001")
        folder_num = folder_name.replace("customer_", "")

        summary = _get_customer_summary(folder_path)
        
        customers.append({
            "id": folder_num,
            "sk_id_curr": sk_id,
            "folder_id": folder_name,
            "label": f"Customer #{folder_num}",
            "target": meta.get("target", 0),
            "target_label": meta.get("label", ""),
            **summary,
        })

    # Sort by folder number
    customers.sort(key=lambda c: c["id"])

    return {
        "customers": customers,
        "total": len(customers),
    }


@router.get("/customers/{customer_id}")
async def get_customer(customer_id: str):
    """
    Get detailed summary for a specific customer.
    
    - **customer_id**: e.g. "001", "1", "customer_001"
    """
    # Normalize customer_id
    folder_id = customer_id.zfill(3) if customer_id.isdigit() else customer_id
    if not folder_id.startswith("customer_"):
        folder_id = f"customer_{folder_id}"

    folder_path = MOCK_DIR / folder_id
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"Customer not found: {customer_id}")

    folder_num = folder_id.replace("customer_", "")
    summary = _get_customer_summary(folder_path)

    return {
        "id": folder_num,
        "folder_id": folder_id,
        "label": f"Customer #{folder_num}",
        **summary,
    }
