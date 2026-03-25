"""
CREDICOUNCIL API — Pipeline Service.

Runs the full A1→A2→A3→A4 credit scoring pipeline.
Writes results to data/output/.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
import traceback
from pathlib import Path
from typing import Any

from credicouncil.api.config import MOCK_DIR, OUTPUT_DIR, PROJECT_ROOT, settings
from credicouncil.api.data_access import normalize_folder_id
from credicouncil.services.llm_service import reset_token_counter, get_token_counts

logger = logging.getLogger(__name__)

# ─── Lazy-loaded agents ──────────────────────────────────────────────────────
_agents: dict[str, Any] = {}


def get_agents() -> dict:
    """Lazy-load all pipeline agents on first request."""
    if _agents:
        return _agents

    from credicouncil.agents.a1_ingestion.agent import IngestionAgent
    from credicouncil.agents.a2_feature_engineer.agent import FeatureEngineerAgent
    from credicouncil.agents.a3_scoring.agent import ScoringAgent
    from credicouncil.agents.a4_report_generator.agent import ReportGeneratorAgent

    model_path = settings.MODEL_PATH
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)

    _agents["a1"] = IngestionAgent()
    _agents["a2"] = FeatureEngineerAgent()
    _agents["a3"] = ScoringAgent(model_path=model_path)
    _agents["a4"] = ReportGeneratorAgent()

    logger.info(f"Agents loaded — model: {model_path}")
    return _agents


# ─── Pipeline execution ─────────────────────────────────────────────────────

def run_pipeline_for_customer(customer_id: str) -> dict:
    """
    Run the full pipeline for a customer and save results to data/output/.

    Args:
        customer_id: e.g. "001", "1", "customer_001"

    Returns:
        Formatted scoring result dict.

    Raises:
        FileNotFoundError: If customer mock folder not found.
        Exception: If pipeline execution fails.
    """
    folder_id = normalize_folder_id(customer_id)
    mock_folder = MOCK_DIR / folder_id
    output_folder = OUTPUT_DIR / folder_id

    if not mock_folder.exists():
        raise FileNotFoundError(f"Customer mock folder not found: {mock_folder}")

    # ── Run real pipeline ──
    logger.info(f"[Pipeline] START real pipeline for {folder_id}")
    result = _execute_pipeline(str(mock_folder), folder_id)

    # ── Save results to data/output/ ──
    _save_results_to_output(output_folder, result)

    logger.info(f"[Pipeline] ✅ SUCCESS for {folder_id}")
    return result


# ─── 2-Phase Pipeline (Human-in-the-Loop) ────────────────────────────────────


def execute_ingestion_only(customer_id: str) -> dict:
    """Phase 1: Run A1 ingestion only and return extracted data for human review.

    Returns the raw A1 output including application_row, confidence_map,
    and structured field metadata for the review UI.
    """
    folder_id = normalize_folder_id(customer_id)
    mock_folder = MOCK_DIR / folder_id

    if not mock_folder.exists():
        raise FileNotFoundError(f"Customer mock folder not found: {mock_folder}")

    agents = get_agents()
    logger.info(f"[Phase 1] A1 Ingestion for {folder_id}...")

    # Reset token counter — A1 is the first step, so we start counting from here.
    # Phase 2 (A2→A4) will accumulate on top of this.
    reset_token_counter()

    a1_output = agents["a1"].ingest(str(mock_folder))

    # Build field metadata for review UI
    field_metadata = build_field_metadata(a1_output)

    return {
        "application_id": a1_output.get("application_id", ""),
        "customer_id": customer_id,
        "application_row": a1_output.get("application_row", {}),
        "confidence_map": a1_output.get("confidence_map", {}),
        "identity_consistency_flag": a1_output.get("identity_consistency_flag", "MISSING"),
        "thin_file_flag": a1_output.get("thin_file_flag", True),
        "raw_texts": a1_output.get("raw_texts", {}),
        "field_metadata": field_metadata,
        "warnings": [],
    }


def execute_processing(
    customer_id: str,
    application_row: dict,
    raw_texts: dict | None = None,
    thin_file_flag: bool = False,
    identity_consistency_flag: str = "OK",
) -> dict:
    """Phase 2: Run A2→A3→A4 with approved/edited data.

    Takes the human-reviewed application_row and runs the rest of the pipeline.
    """
    folder_id = normalize_folder_id(customer_id)
    output_folder = OUTPUT_DIR / folder_id

    agents = get_agents()

    import pandas as pd

    # Reconstruct A1-like output from the approved data
    a1_output = {
        "application_id": customer_id,
        "application_row": application_row,
        "raw_texts": raw_texts or {},
        "thin_file_flag": thin_file_flag,
        "identity_consistency_flag": identity_consistency_flag,
        "confidence_map": {},
        "audit_trail": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "HUMAN_REVIEW",
            "action": "approved_extracted_data",
            "input_summary": {"fields_count": len(application_row)},
            "output_summary": {"status": "APPROVED"},
        }],
        # Empty DataFrames — will be rebuilt from approved data
        "bureau_df": pd.DataFrame(),
        "bureau_balance_df": pd.DataFrame(),
        "previous_application_df": pd.DataFrame(),
        "pos_cash_df": pd.DataFrame(),
        "installments_df": pd.DataFrame(),
        "credit_card_df": pd.DataFrame(),
    }

    # Also re-read CIC and internal DB to get DataFrames
    mock_folder = MOCK_DIR / folder_id
    if mock_folder.exists():
        from credicouncil.agents.a1_ingestion.cic_service import CICService
        from credicouncil.agents.a1_ingestion.internal_db_reader import InternalDBReader

        cic = CICService()
        cic_path = mock_folder / "07_cic_api_response.json"
        cic_result = cic.query(cic_path if cic_path.exists() else None)

        internal_db = InternalDBReader()
        internal_path = mock_folder / "08_internal_db.json"
        internal_dfs = internal_db.read(internal_path if internal_path.exists() else None)

        # Rebuild bureau DataFrames from CIC
        from credicouncil.agents.a1_ingestion.agent import IngestionAgent
        temp_agent = IngestionAgent()
        bureau_df, bureau_balance_df = temp_agent._build_bureau_dfs(cic_result)

        a1_output["bureau_df"] = bureau_df
        a1_output["bureau_balance_df"] = bureau_balance_df
        a1_output["previous_application_df"] = internal_dfs.get("previous_application", pd.DataFrame())
        a1_output["pos_cash_df"] = internal_dfs.get("POS_CASH_balance", pd.DataFrame())
        a1_output["installments_df"] = internal_dfs.get("installments_payments", pd.DataFrame())
        a1_output["credit_card_df"] = internal_dfs.get("credit_card_balance", pd.DataFrame())

    logger.info(f"[Phase 2] A2→A3→A4 for {customer_id}...")

    a2_output = agents["a2"].process(a1_output)
    a3_output = agents["a3"].score(a2_output)
    a4_output = agents["a4"].generate(a3_output, a2_output, a1_output)

    result = {
        "a1_output": a1_output,
        "a2_output": a2_output,
        "a3_output": a3_output,
        "a4_output": a4_output,
    }

    # Save results
    _save_results_to_output(output_folder, result)

    logger.info(f"[Phase 2] ✅ Complete for {customer_id}")
    return result


def execute_processing_with_events(
    customer_id: str,
    application_row: dict,
    raw_texts: dict | None = None,
    thin_file_flag: bool = False,
    identity_consistency_flag: str = "OK",
    event_callback=None,
) -> dict:
    """Phase 2 with realtime event broadcasting via callback.

    Tracks execution time and LLM token usage across all agents.

    Same as execute_processing() but calls event_callback(event_dict) after
    each pipeline step so the WebSocket can relay progress to the frontend.
    """
    folder_id = normalize_folder_id(customer_id)
    output_folder = OUTPUT_DIR / folder_id

    # ── Start tracking timing (tokens already counting from Phase 1) ──
    start_time = time.time()

    agents = get_agents()

    import pandas as pd

    # Reconstruct A1-like output from the approved data
    a1_output = {
        "application_id": customer_id,
        "application_row": application_row,
        "raw_texts": raw_texts or {},
        "thin_file_flag": thin_file_flag,
        "identity_consistency_flag": identity_consistency_flag,
        "confidence_map": {},
        "audit_trail": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "HUMAN_REVIEW",
            "action": "approved_extracted_data",
            "input_summary": {"fields_count": len(application_row)},
            "output_summary": {"status": "APPROVED"},
        }],
        "bureau_df": pd.DataFrame(),
        "bureau_balance_df": pd.DataFrame(),
        "previous_application_df": pd.DataFrame(),
        "pos_cash_df": pd.DataFrame(),
        "installments_df": pd.DataFrame(),
        "credit_card_df": pd.DataFrame(),
    }

    # Re-read CIC and internal DB to get DataFrames
    mock_folder = MOCK_DIR / folder_id
    if mock_folder.exists():
        from credicouncil.agents.a1_ingestion.cic_service import CICService
        from credicouncil.agents.a1_ingestion.internal_db_reader import InternalDBReader

        cic = CICService()
        cic_path = mock_folder / "07_cic_api_response.json"
        cic_result = cic.query(cic_path if cic_path.exists() else None)

        internal_db = InternalDBReader()
        internal_path = mock_folder / "08_internal_db.json"
        internal_dfs = internal_db.read(internal_path if internal_path.exists() else None)

        from credicouncil.agents.a1_ingestion.agent import IngestionAgent
        temp_agent = IngestionAgent()
        bureau_df, bureau_balance_df = temp_agent._build_bureau_dfs(cic_result)

        a1_output["bureau_df"] = bureau_df
        a1_output["bureau_balance_df"] = bureau_balance_df
        a1_output["previous_application_df"] = internal_dfs.get("previous_application", pd.DataFrame())
        a1_output["pos_cash_df"] = internal_dfs.get("POS_CASH_balance", pd.DataFrame())
        a1_output["installments_df"] = internal_dfs.get("installments_payments", pd.DataFrame())
        a1_output["credit_card_df"] = internal_dfs.get("credit_card_balance", pd.DataFrame())

    def _emit(event):
        if event_callback:
            try:
                event_callback(event)
            except Exception as e:
                logger.warning(f"[WS] Event callback error: {e}")

    # ── A2: Feature Engineering ──
    _emit({"event": "started", "step": "A2"})
    logger.info(f"[Phase 2 WS] A2: Feature Engineering for {customer_id}...")
    a2_output = agents["a2"].process(a1_output)

    # Extract real feature count from a2_output
    feature_vector = a2_output.get("feature_vector")
    if feature_vector is not None:
        if isinstance(feature_vector, pd.DataFrame):
            features_count = feature_vector.shape[1]
        elif isinstance(feature_vector, pd.Series):
            features_count = len(feature_vector)
        elif isinstance(feature_vector, dict):
            features_count = len(feature_vector)
        else:
            features_count = 0
    else:
        features_count = len(a2_output.get("feature_row", a2_output.get("application_row", {})))

    _emit({"event": "completed", "step": "A2", "data": {"features_count": features_count}})

    # ── A3: ML Scoring ──
    _emit({"event": "started", "step": "A3"})
    logger.info(f"[Phase 2 WS] A3: ML Scoring for {customer_id}...")
    a3_output = agents["a3"].score(a2_output)
    _emit({"event": "completed", "step": "A3", "data": {
        "credit_score": a3_output.get("credit_score", 0),
        "pd_pct": a3_output.get("pd_pct", 0.0),
        "risk_band": a3_output.get("risk_band", "N/A"),
    }})

    # ── A4: Report Generation ──
    _emit({"event": "started", "step": "A4"})
    logger.info(f"[Phase 2 WS] A4: Report Generation for {customer_id}...")
    a4_output = agents["a4"].generate(a3_output, a2_output, a1_output)

    five_c_scores = a4_output.get("five_c_scores", {})
    report = a4_output.get("final_report", {})
    executive = report.get("executive_summary", {})
    if not five_c_scores:
        five_c_scores = executive.get("five_c_scores", {})

    five_c_total = sum(five_c_scores.values()) if five_c_scores else 0
    recommendation = executive.get("recommendation", a3_output.get("routing", "REVIEW"))

    _emit({"event": "completed", "step": "A4", "data": {
        "five_c_total": five_c_total,
        "recommendation": recommendation,
    }})

    result = {
        "a1_output": a1_output,
        "a2_output": a2_output,
        "a3_output": a3_output,
        "a4_output": a4_output,
    }

    # Save results
    _save_results_to_output(output_folder, result)

    # ── Collect and save pipeline metrics ──
    elapsed_seconds = round(time.time() - start_time, 2)
    token_counts = get_token_counts()
    metrics = {
        "runtime_seconds": elapsed_seconds,
        "prompt_tokens": token_counts["prompt_tokens"],
        "candidates_tokens": token_counts["candidates_tokens"],
        "total_tokens": token_counts["total_tokens"],
        "customer_id": customer_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save metrics to output folder
    try:
        with open(output_folder / "pipeline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"[Phase 2 WS] Metrics saved: {elapsed_seconds}s, {token_counts['total_tokens']} tokens")
    except Exception as e:
        logger.warning(f"[Phase 2 WS] Could not save metrics: {e}")

    logger.info(f"[Phase 2 WS] ✅ Complete for {customer_id}")
    # Attach metrics to result for the caller (WebSocket route)
    result["__metrics"] = metrics
    return result


# ── Field metadata builder ───────────────────────────────────────────────────

# Field grouping and Vietnamese labels for the review UI
FIELD_GROUPS = [
    {
        "group_id": "identity",
        "group_label": "Thông tin nhân thân (CCCD)",
        "icon": "badge",
        "source": "01_cccd.pdf",
        "fields": {
            "SK_ID_CURR": {"label": "Mã hồ sơ", "type": "number"},
            "CODE_GENDER": {"label": "Giới tính", "type": "enum", "options": ["M", "F", "XNA"]},
            "DAYS_BIRTH": {"label": "Ngày sinh (DAYS_BIRTH)", "type": "number"},
            "DAYS_ID_PUBLISH": {"label": "Ngày cấp CCCD (DAYS)", "type": "number"},
            "DAYS_REGISTRATION": {"label": "Ngày đăng ký (DAYS)", "type": "number"},
        },
    },
    {
        "group_id": "employment",
        "group_label": "Việc làm & Thu nhập (HĐLĐ)",
        "icon": "work",
        "source": "02_hop_dong_lao_dong.pdf",
        "fields": {
            "AMT_INCOME_TOTAL": {"label": "Thu nhập năm (VND)", "type": "number"},
            "NAME_INCOME_TYPE": {"label": "Loại thu nhập", "type": "enum", "options": [
                "Working", "Commercial associate", "Pensioner", "State servant",
                "Unemployed", "Student", "Businessman", "Maternity leave",
            ]},
            "DAYS_EMPLOYED": {"label": "Ngày bắt đầu làm việc (DAYS)", "type": "number"},
            "ORGANIZATION_TYPE": {"label": "Loại tổ chức", "type": "text"},
            "OCCUPATION_TYPE": {"label": "Nghề nghiệp", "type": "text"},
        },
    },
    {
        "group_id": "family",
        "group_label": "Gia đình (Sổ Hộ khẩu)",
        "icon": "family_restroom",
        "source": "03_so_ho_khau.pdf",
        "fields": {
            "CNT_CHILDREN": {"label": "Số con", "type": "number"},
            "CNT_FAM_MEMBERS": {"label": "Số thành viên gia đình", "type": "number"},
            "NAME_FAMILY_STATUS": {"label": "Tình trạng hôn nhân", "type": "enum", "options": [
                "Married", "Single / not married", "Separated", "Widow", "Civil marriage",
            ]},
        },
    },
    {
        "group_id": "loan",
        "group_label": "Thông tin khoản vay (Đơn vay)",
        "icon": "account_balance",
        "source": "05_don_vay.pdf",
        "fields": {
            "NAME_CONTRACT_TYPE": {"label": "Loại hợp đồng", "type": "enum", "options": ["Cash loans", "Revolving loans"]},
            "AMT_CREDIT": {"label": "Số tiền vay (VND)", "type": "number"},
            "AMT_ANNUITY": {"label": "Trả hàng năm (VND)", "type": "number"},
            "AMT_GOODS_PRICE": {"label": "Giá trị hàng hóa (VND)", "type": "number"},
            "NAME_TYPE_SUITE": {"label": "Người đồng hành", "type": "text"},
            "NAME_EDUCATION_TYPE": {"label": "Trình độ học vấn", "type": "enum", "options": [
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree",
            ]},
        },
    },
    {
        "group_id": "assets",
        "group_label": "Tài sản",
        "icon": "directions_car",
        "source": "05_don_vay.pdf",
        "fields": {
            "FLAG_OWN_CAR": {"label": "Sở hữu ô tô", "type": "enum", "options": ["Y", "N"]},
            "FLAG_OWN_REALTY": {"label": "Sở hữu bất động sản", "type": "enum", "options": ["Y", "N"]},
            "OWN_CAR_AGE": {"label": "Tuổi xe (năm)", "type": "number"},
        },
    },
    {
        "group_id": "housing_basic",
        "group_label": "Nhà ở & Thẩm định",
        "icon": "home",
        "source": "04_tham_dinh_nha_o.pdf",
        "fields": {
            "NAME_HOUSING_TYPE": {"label": "Loại nhà ở", "type": "text"},
            "REGION_POPULATION_RELATIVE": {"label": "Mật độ dân số (tương đối)", "type": "number"},
            "REGION_RATING_CLIENT": {"label": "Xếp hạng khu vực", "type": "number"},
            "REGION_RATING_CLIENT_W_CITY": {"label": "Xếp hạng khu vực (TP)", "type": "number"},
            "FONDKAPREMONT_MODE": {"label": "Quỹ bảo trì", "type": "text"},
            "HOUSETYPE_MODE": {"label": "Loại hình nhà ở", "type": "text"},
            "TOTALAREA_MODE": {"label": "Tổng diện tích (norm)", "type": "number"},
            "WALLSMATERIAL_MODE": {"label": "Vật liệu tường", "type": "text"},
            "EMERGENCYSTATE_MODE": {"label": "Tình trạng khẩn cấp", "type": "enum", "options": ["No", "Yes"]},
        },
    },
    {
        "group_id": "housing_norm",
        "group_label": "Đặc điểm nhà ở (Normalized)",
        "icon": "apartment",
        "source": "04_tham_dinh_nha_o.pdf",
        "fields": {
            # 14 base fields × 3 suffixes = 42 columns
            "APARTMENTS_AVG": {"label": "Số căn hộ (AVG)", "type": "number"},
            "APARTMENTS_MODE": {"label": "Số căn hộ (MODE)", "type": "number"},
            "APARTMENTS_MEDI": {"label": "Số căn hộ (MEDI)", "type": "number"},
            "BASEMENTAREA_AVG": {"label": "Diện tích tầng hầm (AVG)", "type": "number"},
            "BASEMENTAREA_MODE": {"label": "Diện tích tầng hầm (MODE)", "type": "number"},
            "BASEMENTAREA_MEDI": {"label": "Diện tích tầng hầm (MEDI)", "type": "number"},
            "YEARS_BEGINEXPLUATATION_AVG": {"label": "Năm khai thác (AVG)", "type": "number"},
            "YEARS_BEGINEXPLUATATION_MODE": {"label": "Năm khai thác (MODE)", "type": "number"},
            "YEARS_BEGINEXPLUATATION_MEDI": {"label": "Năm khai thác (MEDI)", "type": "number"},
            "YEARS_BUILD_AVG": {"label": "Năm xây dựng (AVG)", "type": "number"},
            "YEARS_BUILD_MODE": {"label": "Năm xây dựng (MODE)", "type": "number"},
            "YEARS_BUILD_MEDI": {"label": "Năm xây dựng (MEDI)", "type": "number"},
            "COMMONAREA_AVG": {"label": "Diện tích chung (AVG)", "type": "number"},
            "COMMONAREA_MODE": {"label": "Diện tích chung (MODE)", "type": "number"},
            "COMMONAREA_MEDI": {"label": "Diện tích chung (MEDI)", "type": "number"},
            "ELEVATORS_AVG": {"label": "Thang máy (AVG)", "type": "number"},
            "ELEVATORS_MODE": {"label": "Thang máy (MODE)", "type": "number"},
            "ELEVATORS_MEDI": {"label": "Thang máy (MEDI)", "type": "number"},
            "ENTRANCES_AVG": {"label": "Số lối vào (AVG)", "type": "number"},
            "ENTRANCES_MODE": {"label": "Số lối vào (MODE)", "type": "number"},
            "ENTRANCES_MEDI": {"label": "Số lối vào (MEDI)", "type": "number"},
            "FLOORSMAX_AVG": {"label": "Số tầng tối đa (AVG)", "type": "number"},
            "FLOORSMAX_MODE": {"label": "Số tầng tối đa (MODE)", "type": "number"},
            "FLOORSMAX_MEDI": {"label": "Số tầng tối đa (MEDI)", "type": "number"},
            "FLOORSMIN_AVG": {"label": "Số tầng tối thiểu (AVG)", "type": "number"},
            "FLOORSMIN_MODE": {"label": "Số tầng tối thiểu (MODE)", "type": "number"},
            "FLOORSMIN_MEDI": {"label": "Số tầng tối thiểu (MEDI)", "type": "number"},
            "LANDAREA_AVG": {"label": "Diện tích đất (AVG)", "type": "number"},
            "LANDAREA_MODE": {"label": "Diện tích đất (MODE)", "type": "number"},
            "LANDAREA_MEDI": {"label": "Diện tích đất (MEDI)", "type": "number"},
            "LIVINGAPARTMENTS_AVG": {"label": "Căn hộ ở (AVG)", "type": "number"},
            "LIVINGAPARTMENTS_MODE": {"label": "Căn hộ ở (MODE)", "type": "number"},
            "LIVINGAPARTMENTS_MEDI": {"label": "Căn hộ ở (MEDI)", "type": "number"},
            "LIVINGAREA_AVG": {"label": "Diện tích sống (AVG)", "type": "number"},
            "LIVINGAREA_MODE": {"label": "Diện tích sống (MODE)", "type": "number"},
            "LIVINGAREA_MEDI": {"label": "Diện tích sống (MEDI)", "type": "number"},
            "NONLIVINGAPARTMENTS_AVG": {"label": "Căn hộ không ở (AVG)", "type": "number"},
            "NONLIVINGAPARTMENTS_MODE": {"label": "Căn hộ không ở (MODE)", "type": "number"},
            "NONLIVINGAPARTMENTS_MEDI": {"label": "Căn hộ không ở (MEDI)", "type": "number"},
            "NONLIVINGAREA_AVG": {"label": "DT không ở (AVG)", "type": "number"},
            "NONLIVINGAREA_MODE": {"label": "DT không ở (MODE)", "type": "number"},
            "NONLIVINGAREA_MEDI": {"label": "DT không ở (MEDI)", "type": "number"},
        },
    },
    {
        "group_id": "address",
        "group_label": "Đối chiếu địa chỉ",
        "icon": "location_on",
        "source": "04_tham_dinh_nha_o.pdf",
        "fields": {
            "REG_REGION_NOT_LIVE_REGION": {"label": "ĐK ≠ Nơi sống (vùng)", "type": "number"},
            "REG_REGION_NOT_WORK_REGION": {"label": "ĐK ≠ Nơi làm (vùng)", "type": "number"},
            "LIVE_REGION_NOT_WORK_REGION": {"label": "Sống ≠ Làm (vùng)", "type": "number"},
            "REG_CITY_NOT_LIVE_CITY": {"label": "ĐK ≠ Nơi sống (TP)", "type": "number"},
            "REG_CITY_NOT_WORK_CITY": {"label": "ĐK ≠ Nơi làm (TP)", "type": "number"},
            "LIVE_CITY_NOT_WORK_CITY": {"label": "Sống ≠ Làm (TP)", "type": "number"},
        },
    },
    {
        "group_id": "cic",
        "group_label": "Điểm tín dụng (CIC)",
        "icon": "credit_score",
        "source": "07_cic_api_response.json",
        "fields": {
            "EXT_SOURCE_1": {"label": "External Score 1", "type": "number"},
            "EXT_SOURCE_2": {"label": "External Score 2", "type": "number"},
            "EXT_SOURCE_3": {"label": "External Score 3", "type": "number"},
        },
    },
    {
        "group_id": "cic_inquiry",
        "group_label": "Lịch sử tra cứu CIC",
        "icon": "query_stats",
        "source": "07_cic_api_response.json",
        "fields": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": {"label": "Tra cứu (giờ qua)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_DAY": {"label": "Tra cứu (ngày qua)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_WEEK": {"label": "Tra cứu (tuần qua)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_MON": {"label": "Tra cứu (tháng qua)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_QRT": {"label": "Tra cứu (quý qua)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_YEAR": {"label": "Tra cứu (năm qua)", "type": "number"},
        },
    },
    {
        "group_id": "social",
        "group_label": "Vòng tròn xã hội (CIC)",
        "icon": "groups",
        "source": "07_cic_api_response.json",
        "fields": {
            "OBS_30_CNT_SOCIAL_CIRCLE": {"label": "Quan sát 30 ngày", "type": "number"},
            "DEF_30_CNT_SOCIAL_CIRCLE": {"label": "Nợ xấu 30 ngày", "type": "number"},
            "OBS_60_CNT_SOCIAL_CIRCLE": {"label": "Quan sát 60 ngày", "type": "number"},
            "DEF_60_CNT_SOCIAL_CIRCLE": {"label": "Nợ xấu 60 ngày", "type": "number"},
        },
    },
    {
        "group_id": "contact",
        "group_label": "Thông tin liên lạc",
        "icon": "phone",
        "source": "05_don_vay.pdf",
        "fields": {
            "FLAG_MOBIL": {"label": "Có SĐT di động", "type": "number"},
            "FLAG_EMP_PHONE": {"label": "Có SĐT nơi làm", "type": "number"},
            "FLAG_WORK_PHONE": {"label": "Có SĐT bàn công ty", "type": "number"},
            "FLAG_CONT_MOBILE": {"label": "SĐT liên lạc được", "type": "number"},
            "FLAG_PHONE": {"label": "Có SĐT bàn", "type": "number"},
            "FLAG_EMAIL": {"label": "Có email", "type": "number"},
            "DAYS_LAST_PHONE_CHANGE": {"label": "Ngày đổi SĐT gần nhất (DAYS)", "type": "number"},
        },
    },
    {
        "group_id": "process",
        "group_label": "Thông tin xử lý hồ sơ",
        "icon": "schedule",
        "source": "05_don_vay.pdf",
        "fields": {
            "WEEKDAY_APPR_PROCESS_START": {"label": "Ngày trong tuần nộp hồ sơ", "type": "text"},
            "HOUR_APPR_PROCESS_START": {"label": "Giờ nộp hồ sơ", "type": "number"},
        },
    },
    {
        "group_id": "documents",
        "group_label": "Cờ hồ sơ nộp (FLAG_DOCUMENT)",
        "icon": "description",
        "source": "05_don_vay.pdf",
        "fields": {
            **{f"FLAG_DOCUMENT_{i}": {"label": f"Tài liệu #{i}", "type": "number"}
               for i in range(2, 22)},
        },
    },
]


def build_field_metadata(a1_output: dict) -> list[dict]:
    """Build structured field metadata for the review UI.

    Groups fields by document source, attaches Vietnamese labels,
    current values, confidence scores, and field types.
    """
    application_row = a1_output.get("application_row", {})
    confidence_map = a1_output.get("confidence_map", {})

    result = []
    for group in FIELD_GROUPS:
        group_fields = []
        for field_name, meta in group["fields"].items():
            value = application_row.get(field_name)

            # Find matching confidence — try multiple key patterns
            conf = 0.0
            for conf_key, conf_val in confidence_map.items():
                if conf_key.endswith(f".{field_name}") or conf_key == field_name:
                    conf = conf_val
                    break
            # If no specific confidence found, use default based on source
            if conf == 0.0 and value is not None:
                if group["source"].endswith(".json"):
                    conf = 0.99  # JSON sources are deterministic
                else:
                    conf = 0.90  # Default LLM confidence

            field_info = {
                "field_name": field_name,
                "label_vi": meta["label"],
                "value": value,
                "confidence": round(conf, 3),
                "field_type": meta["type"],
            }
            if "options" in meta:
                field_info["options"] = meta["options"]
            group_fields.append(field_info)

        result.append({
            "group_id": group["group_id"],
            "group_label": group["group_label"],
            "icon": group["icon"],
            "source_document": group["source"],
            "fields": group_fields,
        })

    return result


def _execute_pipeline(customer_dir: str, folder_id: str) -> dict:
    """Execute the full A1→A2→A3→A4 pipeline and return raw results."""
    agents = get_agents()

    logger.info(f"  [1/4] A1: Ingesting from {customer_dir}...")
    a1_output = agents["a1"].ingest(customer_dir)

    logger.info("  [2/4] A2: Feature Engineering...")
    a2_output = agents["a2"].process(a1_output)

    logger.info("  [3/4] A3: ML Scoring...")
    a3_output = agents["a3"].score(a2_output)

    logger.info("  [4/4] A4: Report Generation...")
    a4_output = agents["a4"].generate(a3_output, a2_output, a1_output)

    return {
        "a1_output": a1_output,
        "a2_output": a2_output,
        "a3_output": a3_output,
        "a4_output": a4_output,
    }


def _save_results_to_output(output_folder: Path, pipeline_result: dict) -> None:
    """Save pipeline outputs (JSON report + SHAP + PDF) to data/output/."""
    output_folder.mkdir(parents=True, exist_ok=True)

    a3_output = pipeline_result["a3_output"]
    a4_output = pipeline_result["a4_output"]
    report = a4_output.get("final_report", {})
    shap_data = a3_output.get("shap_values", {})

    # Write credit_report.json
    with open(output_folder / "credit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # Write shap_values.json
    with open(output_folder / "shap_values.json", "w", encoding="utf-8") as f:
        json.dump(shap_data, f, ensure_ascii=False, indent=2, default=str)

    # Generate and write PDF
    try:
        from credicouncil.agents.a4_report_generator.pdf_generator import generate_credit_pdf

        pdf_bytes = generate_credit_pdf(report_data=report, shap_data=shap_data)
        (output_folder / "credit_report.pdf").write_bytes(pdf_bytes)
        logger.info(f"  Output saved: {output_folder}")
    except Exception as e:
        logger.warning(f"  PDF generation failed (non-critical): {e}")


def _load_result_from_output(output_folder: Path) -> dict:
    """Load pre-existing result from output folder and format as pipeline result."""
    report_path = output_folder / "credit_report.json"
    shap_path = output_folder / "shap_values.json"

    report = {}
    shap_data = {}

    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    if shap_path.exists():
        with open(shap_path, encoding="utf-8") as f:
            shap_data = json.load(f)

    executive = report.get("executive_summary", {})

    return {
        "a3_output": {
            "credit_score": executive.get("credit_score", 0),
            "pd_pct": executive.get("pd_pct", 0),
            "risk_band": executive.get("risk_band", "N/A"),
            "shap_values": shap_data,
            "routing": executive.get("recommendation", "REVIEW"),
        },
        "a4_output": {
            "final_report": report,
            "five_c_scores": executive.get("five_c_scores", {}),
        },
    }


# ─── Result formatting ──────────────────────────────────────────────────────

def format_score_response(pipeline_result: dict, customer_id: str) -> dict:
    """Format raw pipeline result into ScoreResponse-compatible dict."""
    a3 = pipeline_result["a3_output"]
    a4 = pipeline_result["a4_output"]
    report = a4.get("final_report", {})
    executive = report.get("executive_summary", {})

    return {
        "application_id": customer_id,
        "credit_score": a3.get("credit_score", 0),
        "pd_pct": a3.get("pd_pct", 0.0),
        "risk_band": a3.get("risk_band", "N/A"),
        "recommendation": executive.get("recommendation", a3.get("routing", "REVIEW")),
        "overall_confidence": a3.get("overall_confidence", 0.0),
        "four_c_scores": a4.get("five_c_scores", executive.get("five_c_scores", {})),
        "warnings": a4.get("warnings", []),
        "report": report,
    }


def format_legacy_result(pipeline_result: dict) -> dict:
    """Format pipeline outputs into ScoringResult-compatible dict."""
    a3 = pipeline_result["a3_output"]
    a4 = pipeline_result["a4_output"]
    report = a4.get("final_report", {})
    executive = report.get("executive_summary", {})
    shap = a3.get("shap_values", {})

    return {
        "credit_score": a3.get("credit_score", 0),
        "pd_probability": a3.get("pd_pct", 0),
        "risk_band": a3.get("risk_band", "N/A"),
        "decision": a3.get("routing", "REVIEW"),
        "shap_top_positive": shap.get("top_positive_factors", []),
        "shap_top_negative": shap.get("top_negative_factors", []),
        "five_c_scores": a4.get("five_c_scores", {}),
        "five_c_total": executive.get("five_c_total", 0),
        "recommendation": executive.get("recommendation", "REVIEW"),
        "consistency_check": a4.get("consistency_check", {}).get("passed", False),
        "audit_trail": a4.get("audit_trail", []),
        "warnings": a4.get("warnings", []),
    }
