"""
CREDICOUNCIL API — Customer Routes.

GET    /v1/customers       → List all customer profiles
GET    /v1/customers/{id}  → Get single customer detail
DELETE /v1/output           → Clear all pipeline output data (for demo reset)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from credicouncil.api.config import MOCK_DIR, OUTPUT_DIR, settings
from credicouncil.api.data_access import (
    check_output_exists,
    get_customer_summary,
    load_customer_map,
    normalize_folder_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.API_V1_STR, tags=["Customers"])


@router.get("/customers")
async def list_customers():
    """
    List all available customer profiles.

    - Customer list comes from data/mock/ (customer_map.json).
    - Basic info (gender, age, income, etc.) comes from data/mock/.
    - Score fields (credit_score, risk_band, pd_pct, etc.) come ONLY from data/output/.
    - has_output flag indicates whether pipeline output exists in data/output/.
    """
    customer_map = load_customer_map()
    customers = []

    # Score fields that should ONLY come from output, never from mock
    SCORE_FIELDS = {
        "credit_score", "risk_band", "pd_pct", "recommendation",
        "five_c_total", "five_c_scores", "has_report",
    }

    for sk_id, meta in customer_map.items():
        folder_name = meta.get("dir", "")
        mock_folder = MOCK_DIR / folder_name

        if not mock_folder.exists():
            continue

        folder_num = folder_name.replace("customer_", "")

        # Basic info from mock (strip score fields — they must come from output only)
        summary = get_customer_summary(mock_folder)
        for key in SCORE_FIELDS:
            summary.pop(key, None)

        # Check if pipeline output exists
        has_output = check_output_exists(folder_name)

        # Score data ONLY from output/
        if has_output:
            output_folder = OUTPUT_DIR / folder_name
            output_summary = get_customer_summary(output_folder)
            for key in SCORE_FIELDS:
                if key in output_summary:
                    summary[key] = output_summary[key]

        customers.append(
            {
                "id": folder_num,
                "sk_id_curr": sk_id,
                "folder_id": folder_name,
                "label": f"Customer #{folder_num}",
                "target": meta.get("target", 0),
                "target_label": meta.get("label", ""),
                "has_output": has_output,
                **summary,
            }
        )

    customers.sort(key=lambda c: c["id"])
    return {"customers": customers, "total": len(customers)}


@router.get("/customers/{customer_id}")
async def get_customer(customer_id: str):
    """Get detailed summary for a specific customer."""
    folder_id = normalize_folder_id(customer_id)
    mock_folder = MOCK_DIR / folder_id

    if not mock_folder.exists():
        raise HTTPException(status_code=404, detail=f"Customer not found: {customer_id}")

    folder_num = folder_id.replace("customer_", "")
    summary = get_customer_summary(mock_folder)
    has_output = check_output_exists(folder_id)

    # Overlay output score data if available
    if has_output:
        output_folder = OUTPUT_DIR / folder_id
        output_summary = get_customer_summary(output_folder)
        for key in (
            "credit_score", "risk_band", "pd_pct", "recommendation",
            "five_c_total", "five_c_scores", "has_report",
        ):
            if key in output_summary:
                summary[key] = output_summary[key]

    return {
        "id": folder_num,
        "folder_id": folder_id,
        "label": f"Customer #{folder_num}",
        "has_output": has_output,
        **summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /v1/output — Clear all pipeline output (demo reset)
# ═══════════════════════════════════════════════════════════════════════════════


@router.delete("/output")
async def clear_output():
    """
    Delete all files in data/output/ to reset for a fresh demo.

    Only removes customer subdirectories (customer_*), preserves the output/ dir itself.
    """
    import shutil

    if not OUTPUT_DIR.exists():
        return {"cleared": 0, "message": "Output directory does not exist"}

    cleared = 0
    errors = []

    for item in OUTPUT_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
                cleared += 1
            elif item.is_file() and item.name != ".gitkeep":
                item.unlink()
                cleared += 1
        except Exception as e:
            errors.append(f"{item.name}: {e}")
            logger.warning(f"Failed to remove {item}: {e}")

    logger.info(f"[Clear Output] Removed {cleared} items from output/")

    return {
        "cleared": cleared,
        "message": f"Cleared {cleared} items from output/",
        "errors": errors if errors else None,
    }
