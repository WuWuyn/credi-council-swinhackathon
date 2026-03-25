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


# ─── Batch Pipeline Functions ────────────────────────────────────────────────


def execute_batch_ingestion(
    customer_ids: list[str],
    event_callback=None,
) -> list[dict]:
    """Phase 1 Batch: Run A1 ingestion in parallel for all customers.

    Uses ThreadPoolExecutor with staggered launches to avoid LLM rate-limit bursts.
    Calls event_callback after each customer's A1 completes.
    Returns a list of ingestion results (one per customer).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from credicouncil.config.batch_config import A1_MAX_WORKERS, A1_STAGGER_DELAY

    total = len(customer_ids)

    def _emit(event):
        if event_callback:
            try:
                event_callback(event)
            except Exception as e:
                logger.warning(f"[Batch A1] Event callback error: {e}")

    _emit({"event": "batch_started", "total": total, "phase": "ingestion"})

    def _ingest_one(cid: str, idx: int) -> dict:
        """Run A1 for a single customer in a thread."""
        _emit({"event": "a1_started", "customer_id": cid, "index": idx})
        try:
            ingestion_result = execute_ingestion_only(cid)
            _emit({
                "event": "a1_completed",
                "customer_id": cid,
                "index": idx,
                "data": ingestion_result,
            })
            return {
                "customer_id": cid,
                "status": "OK",
                "data": ingestion_result,
            }
        except Exception as e:
            logger.error(f"[Batch A1] Failed for {cid}: {e}")
            traceback.print_exc()
            _emit({
                "event": "a1_error",
                "customer_id": cid,
                "index": idx,
                "error": str(e),
            })
            return {
                "customer_id": cid,
                "status": "ERROR",
                "error": str(e),
                "data": None,
            }

    # Launch all A1 pipelines with staggered starts
    results_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=A1_MAX_WORKERS) as executor:
        futures = {}
        for idx, cid in enumerate(customer_ids):
            future = executor.submit(_ingest_one, cid, idx)
            futures[future] = cid
            logger.info(f"[Batch A1] Launched {idx+1}/{total}: {cid}")

            # Stagger launches to avoid LLM rate-limit bursts
            if idx < total - 1 and A1_STAGGER_DELAY > 0:
                time.sleep(A1_STAGGER_DELAY)

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            cid = result["customer_id"]
            results_map[cid] = result
            completed += 1
            _emit({
                "event": "a1_progress",
                "completed": completed,
                "total": total,
            })

    # Preserve original order
    results = [results_map[cid] for cid in customer_ids if cid in results_map]

    _emit({"event": "phase_ingestion_done", "results_count": len(results)})
    return results


def execute_batch_processing_parallel(
    approved_customers: list[dict],
    event_callback=None,
) -> dict:
    """Phase 3 Batch: Run A2→A3→A4 in parallel for all approved customers.

    Args:
        approved_customers: List of dicts, each with:
            - customer_id: str
            - application_row: dict (edited/approved data)
            - raw_texts: dict
            - thin_file_flag: bool
            - identity_consistency_flag: str
        event_callback: Called with events for progress tracking.

    Returns:
        Batch summary dict with per-customer results and aggregated metrics.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from credicouncil.config.batch_config import (
        PROCESSING_MAX_WORKERS,
        PROCESSING_STAGGER_DELAY,
    )

    total = len(approved_customers)

    def _emit(event):
        if event_callback:
            try:
                event_callback(event)
            except Exception as e:
                logger.warning(f"[Batch Processing] Event callback error: {e}")

    _emit({"event": "processing_started", "total": total, "phase": "processing"})

    batch_start = time.time()
    customer_results = {}

    def _process_one(customer_data: dict) -> tuple[str, dict]:
        """Run A2→A4 for one customer in a thread."""
        cid = customer_data["customer_id"]

        # Reset token counter for this customer's pipeline
        reset_token_counter()
        start = time.time()

        try:
            def _per_customer_emit(event):
                """Prefix events with customer_id for batch tracking."""
                event["customer_id"] = cid
                event["event"] = "customer_step"
                _emit(event)

            result = execute_processing_with_events(
                customer_id=cid,
                application_row=customer_data["application_row"],
                raw_texts=customer_data.get("raw_texts", {}),
                thin_file_flag=customer_data.get("thin_file_flag", False),
                identity_consistency_flag=customer_data.get(
                    "identity_consistency_flag", "OK"
                ),
                event_callback=_per_customer_emit,
            )

            elapsed = round(time.time() - start, 2)
            metrics = result.pop("__metrics", {})
            formatted = format_score_response(result, cid)

            _emit({
                "event": "customer_done",
                "customer_id": cid,
                "result": formatted,
                "metrics": metrics,
            })

            return cid, {
                **formatted,
                "status": "SUCCESS",
                "metrics": metrics,
            }

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"[Batch Processing] Failed for {cid}: {e}")
            traceback.print_exc()

            _emit({
                "event": "customer_error",
                "customer_id": cid,
                "error": str(e),
            })

            return cid, {
                "application_id": cid,
                "credit_score": 0,
                "pd_pct": 0.0,
                "risk_band": "ERR",
                "recommendation": "ERROR",
                "status": "FAILED",
                "error": str(e),
                "metrics": {
                    "runtime_seconds": elapsed,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "candidates_tokens": 0,
                },
            }

    # Launch all pipelines in parallel with staggered starts
    futures = {}
    with ThreadPoolExecutor(max_workers=PROCESSING_MAX_WORKERS) as executor:
        for idx, cust_data in enumerate(approved_customers):
            future = executor.submit(_process_one, cust_data)
            futures[future] = cust_data["customer_id"]
            logger.info(
                f"[Batch Processing] Launched {idx+1}/{total}: "
                f"customer {cust_data['customer_id']}"
            )

            # Stagger launches
            if idx < total - 1 and PROCESSING_STAGGER_DELAY > 0:
                time.sleep(PROCESSING_STAGGER_DELAY)

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            cid, result = future.result()
            customer_results[cid] = result
            completed += 1
            _emit({
                "event": "batch_progress",
                "completed": completed,
                "total": total,
            })

    batch_duration = round(time.time() - batch_start, 2)

    # Aggregate batch summary
    total_tokens = sum(
        r.get("metrics", {}).get("total_tokens", 0)
        for r in customer_results.values()
    )
    success_count = sum(
        1 for r in customer_results.values() if r.get("status") == "SUCCESS"
    )

    summary = {
        "total_time_seconds": batch_duration,
        "total_tokens": total_tokens,
        "total_customers": total,
        "success_count": success_count,
        "failed_count": total - success_count,
        "customers": [
            {
                "customer_id": cid,
                "status": r.get("status", "UNKNOWN"),
                "credit_score": r.get("credit_score", 0),
                "risk_band": r.get("risk_band", "N/A"),
                "pd_pct": r.get("pd_pct", 0.0),
                "recommendation": r.get("recommendation", "N/A"),
                "time_seconds": r.get("metrics", {}).get("runtime_seconds", 0),
                "tokens": r.get("metrics", {}).get("total_tokens", 0),
            }
            for cid, r in customer_results.items()
        ],
    }

    _emit({"event": "batch_done", "summary": summary})

    logger.info(
        f"[Batch Processing] ✅ DONE — {success_count}/{total} succeeded "
        f"in {batch_duration:.1f}s, {total_tokens} tokens"
    )

    return {
        "results": customer_results,
        "summary": summary,
    }


# ── Field metadata builder ───────────────────────────────────────────────────

# Field grouping and labels for the review UI
FIELD_GROUPS = [
    {
        "group_id": "identity",
        "group_label": "Personal Identity (CCCD)",
        "icon": "badge",
        "source": "01_cccd.pdf",
        "fields": {
            "SK_ID_CURR": {"label": "Application ID", "type": "number"},
            "CODE_GENDER": {"label": "Gender", "type": "enum", "options": ["M", "F", "XNA"]},
            "DAYS_BIRTH": {"label": "Date of Birth (DAYS_BIRTH)", "type": "number"},
            "DAYS_ID_PUBLISH": {"label": "ID Issue Date (DAYS)", "type": "number"},
            "DAYS_REGISTRATION": {"label": "Registration Date (DAYS)", "type": "number"},
        },
    },
    {
        "group_id": "employment",
        "group_label": "Employment & Income",
        "icon": "work",
        "source": "02_hop_dong_lao_dong.pdf",
        "fields": {
            "AMT_INCOME_TOTAL": {"label": "Annual Income (VND)", "type": "number"},
            "NAME_INCOME_TYPE": {"label": "Income Type", "type": "enum", "options": [
                "Working", "Commercial associate", "Pensioner", "State servant",
                "Unemployed", "Student", "Businessman", "Maternity leave",
            ]},
            "DAYS_EMPLOYED": {"label": "Employment Start Date (DAYS)", "type": "number"},
            "ORGANIZATION_TYPE": {"label": "Organization Type", "type": "text"},
            "OCCUPATION_TYPE": {"label": "Occupation", "type": "text"},
        },
    },
    {
        "group_id": "family",
        "group_label": "Family (Household Registration)",
        "icon": "family_restroom",
        "source": "03_so_ho_khau.pdf",
        "fields": {
            "CNT_CHILDREN": {"label": "Number of Children", "type": "number"},
            "CNT_FAM_MEMBERS": {"label": "Family Members", "type": "number"},
            "NAME_FAMILY_STATUS": {"label": "Marital Status", "type": "enum", "options": [
                "Married", "Single / not married", "Separated", "Widow", "Civil marriage",
            ]},
        },
    },
    {
        "group_id": "loan",
        "group_label": "Loan Information",
        "icon": "account_balance",
        "source": "05_don_vay.pdf",
        "fields": {
            "NAME_CONTRACT_TYPE": {"label": "Contract Type", "type": "enum", "options": ["Cash loans", "Revolving loans"]},
            "AMT_CREDIT": {"label": "Loan Amount (VND)", "type": "number"},
            "AMT_ANNUITY": {"label": "Annual Payment (VND)", "type": "number"},
            "AMT_GOODS_PRICE": {"label": "Goods Price (VND)", "type": "number"},
            "NAME_TYPE_SUITE": {"label": "Accompanying Person", "type": "text"},
            "NAME_EDUCATION_TYPE": {"label": "Education Level", "type": "enum", "options": [
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree",
            ]},
        },
    },
    {
        "group_id": "assets",
        "group_label": "Assets",
        "icon": "directions_car",
        "source": "05_don_vay.pdf",
        "fields": {
            "FLAG_OWN_CAR": {"label": "Owns Car", "type": "enum", "options": ["Y", "N"]},
            "FLAG_OWN_REALTY": {"label": "Owns Real Estate", "type": "enum", "options": ["Y", "N"]},
            "OWN_CAR_AGE": {"label": "Car Age (years)", "type": "number"},
        },
    },
    {
        "group_id": "housing_basic",
        "group_label": "Housing & Appraisal",
        "icon": "home",
        "source": "04_tham_dinh_nha_o.pdf",
        "fields": {
            "NAME_HOUSING_TYPE": {"label": "Housing Type", "type": "text"},
            "REGION_POPULATION_RELATIVE": {"label": "Population Density (relative)", "type": "number"},
            "REGION_RATING_CLIENT": {"label": "Region Rating", "type": "number"},
            "REGION_RATING_CLIENT_W_CITY": {"label": "Region Rating (w/ City)", "type": "number"},
            "FONDKAPREMONT_MODE": {"label": "Maintenance Fund", "type": "text"},
            "HOUSETYPE_MODE": {"label": "House Type", "type": "text"},
            "TOTALAREA_MODE": {"label": "Total Area (norm)", "type": "number"},
            "WALLSMATERIAL_MODE": {"label": "Wall Material", "type": "text"},
            "EMERGENCYSTATE_MODE": {"label": "Emergency State", "type": "enum", "options": ["No", "Yes"]},
        },
    },
    {
        "group_id": "housing_norm",
        "group_label": "Housing Features (Normalized)",
        "icon": "apartment",
        "source": "04_tham_dinh_nha_o.pdf",
        "fields": {
            # 14 base fields × 3 suffixes = 42 columns
            "APARTMENTS_AVG": {"label": "Apartments (AVG)", "type": "number"},
            "APARTMENTS_MODE": {"label": "Apartments (MODE)", "type": "number"},
            "APARTMENTS_MEDI": {"label": "Apartments (MEDI)", "type": "number"},
            "BASEMENTAREA_AVG": {"label": "Basement Area (AVG)", "type": "number"},
            "BASEMENTAREA_MODE": {"label": "Basement Area (MODE)", "type": "number"},
            "BASEMENTAREA_MEDI": {"label": "Basement Area (MEDI)", "type": "number"},
            "YEARS_BEGINEXPLUATATION_AVG": {"label": "Year of Exploitation (AVG)", "type": "number"},
            "YEARS_BEGINEXPLUATATION_MODE": {"label": "Year of Exploitation (MODE)", "type": "number"},
            "YEARS_BEGINEXPLUATATION_MEDI": {"label": "Year of Exploitation (MEDI)", "type": "number"},
            "YEARS_BUILD_AVG": {"label": "Year Built (AVG)", "type": "number"},
            "YEARS_BUILD_MODE": {"label": "Year Built (MODE)", "type": "number"},
            "YEARS_BUILD_MEDI": {"label": "Year Built (MEDI)", "type": "number"},
            "COMMONAREA_AVG": {"label": "Common Area (AVG)", "type": "number"},
            "COMMONAREA_MODE": {"label": "Common Area (MODE)", "type": "number"},
            "COMMONAREA_MEDI": {"label": "Common Area (MEDI)", "type": "number"},
            "ELEVATORS_AVG": {"label": "Elevators (AVG)", "type": "number"},
            "ELEVATORS_MODE": {"label": "Elevators (MODE)", "type": "number"},
            "ELEVATORS_MEDI": {"label": "Elevators (MEDI)", "type": "number"},
            "ENTRANCES_AVG": {"label": "Entrances (AVG)", "type": "number"},
            "ENTRANCES_MODE": {"label": "Entrances (MODE)", "type": "number"},
            "ENTRANCES_MEDI": {"label": "Entrances (MEDI)", "type": "number"},
            "FLOORSMAX_AVG": {"label": "Max Floors (AVG)", "type": "number"},
            "FLOORSMAX_MODE": {"label": "Max Floors (MODE)", "type": "number"},
            "FLOORSMAX_MEDI": {"label": "Max Floors (MEDI)", "type": "number"},
            "FLOORSMIN_AVG": {"label": "Min Floors (AVG)", "type": "number"},
            "FLOORSMIN_MODE": {"label": "Min Floors (MODE)", "type": "number"},
            "FLOORSMIN_MEDI": {"label": "Min Floors (MEDI)", "type": "number"},
            "LANDAREA_AVG": {"label": "Land Area (AVG)", "type": "number"},
            "LANDAREA_MODE": {"label": "Land Area (MODE)", "type": "number"},
            "LANDAREA_MEDI": {"label": "Land Area (MEDI)", "type": "number"},
            "LIVINGAPARTMENTS_AVG": {"label": "Living Apartments (AVG)", "type": "number"},
            "LIVINGAPARTMENTS_MODE": {"label": "Living Apartments (MODE)", "type": "number"},
            "LIVINGAPARTMENTS_MEDI": {"label": "Living Apartments (MEDI)", "type": "number"},
            "LIVINGAREA_AVG": {"label": "Living Area (AVG)", "type": "number"},
            "LIVINGAREA_MODE": {"label": "Living Area (MODE)", "type": "number"},
            "LIVINGAREA_MEDI": {"label": "Living Area (MEDI)", "type": "number"},
            "NONLIVINGAPARTMENTS_AVG": {"label": "Non-Living Apartments (AVG)", "type": "number"},
            "NONLIVINGAPARTMENTS_MODE": {"label": "Non-Living Apartments (MODE)", "type": "number"},
            "NONLIVINGAPARTMENTS_MEDI": {"label": "Non-Living Apartments (MEDI)", "type": "number"},
            "NONLIVINGAREA_AVG": {"label": "Non-Living Area (AVG)", "type": "number"},
            "NONLIVINGAREA_MODE": {"label": "Non-Living Area (MODE)", "type": "number"},
            "NONLIVINGAREA_MEDI": {"label": "Non-Living Area (MEDI)", "type": "number"},
        },
    },
    {
        "group_id": "address",
        "group_label": "Address Cross-Check",
        "icon": "location_on",
        "source": "04_tham_dinh_nha_o.pdf",
        "fields": {
            "REG_REGION_NOT_LIVE_REGION": {"label": "Reg ≠ Living (region)", "type": "number"},
            "REG_REGION_NOT_WORK_REGION": {"label": "Reg ≠ Work (region)", "type": "number"},
            "LIVE_REGION_NOT_WORK_REGION": {"label": "Living ≠ Work (region)", "type": "number"},
            "REG_CITY_NOT_LIVE_CITY": {"label": "Reg ≠ Living (city)", "type": "number"},
            "REG_CITY_NOT_WORK_CITY": {"label": "Reg ≠ Work (city)", "type": "number"},
            "LIVE_CITY_NOT_WORK_CITY": {"label": "Living ≠ Work (city)", "type": "number"},
        },
    },
    {
        "group_id": "cic",
        "group_label": "Credit Score (CIC)",
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
        "group_label": "CIC Inquiry History",
        "icon": "query_stats",
        "source": "07_cic_api_response.json",
        "fields": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": {"label": "Inquiries (past hour)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_DAY": {"label": "Inquiries (past day)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_WEEK": {"label": "Inquiries (past week)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_MON": {"label": "Inquiries (past month)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_QRT": {"label": "Inquiries (past quarter)", "type": "number"},
            "AMT_REQ_CREDIT_BUREAU_YEAR": {"label": "Inquiries (past year)", "type": "number"},
        },
    },
    {
        "group_id": "social",
        "group_label": "Social Circle (CIC)",
        "icon": "groups",
        "source": "07_cic_api_response.json",
        "fields": {
            "OBS_30_CNT_SOCIAL_CIRCLE": {"label": "Observations (30 days)", "type": "number"},
            "DEF_30_CNT_SOCIAL_CIRCLE": {"label": "Defaults (30 days)", "type": "number"},
            "OBS_60_CNT_SOCIAL_CIRCLE": {"label": "Observations (60 days)", "type": "number"},
            "DEF_60_CNT_SOCIAL_CIRCLE": {"label": "Defaults (60 days)", "type": "number"},
        },
    },
    {
        "group_id": "contact",
        "group_label": "Contact Information",
        "icon": "phone",
        "source": "05_don_vay.pdf",
        "fields": {
            "FLAG_MOBIL": {"label": "Has Mobile Phone", "type": "number"},
            "FLAG_EMP_PHONE": {"label": "Has Employer Phone", "type": "number"},
            "FLAG_WORK_PHONE": {"label": "Has Work Phone", "type": "number"},
            "FLAG_CONT_MOBILE": {"label": "Mobile Reachable", "type": "number"},
            "FLAG_PHONE": {"label": "Has Home Phone", "type": "number"},
            "FLAG_EMAIL": {"label": "Has Email", "type": "number"},
            "DAYS_LAST_PHONE_CHANGE": {"label": "Last Phone Change (DAYS)", "type": "number"},
        },
    },
    {
        "group_id": "process",
        "group_label": "Application Processing",
        "icon": "schedule",
        "source": "05_don_vay.pdf",
        "fields": {
            "WEEKDAY_APPR_PROCESS_START": {"label": "Submission Day of Week", "type": "text"},
            "HOUR_APPR_PROCESS_START": {"label": "Submission Hour", "type": "number"},
        },
    },
    {
        "group_id": "documents",
        "group_label": "Submitted Documents (FLAG_DOCUMENT)",
        "icon": "description",
        "source": "05_don_vay.pdf",
        "fields": {
            **{f"FLAG_DOCUMENT_{i}": {"label": f"Document #{i}", "type": "number"}
               for i in range(2, 22)},
        },
    },
]


def build_field_metadata(a1_output: dict) -> list[dict]:
    """Build structured field metadata for the review UI.

    Groups fields by document source, attaches labels,
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
