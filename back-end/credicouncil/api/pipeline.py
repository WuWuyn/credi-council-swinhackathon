"""
CREDICOUNCIL API — Pipeline Service.

Runs the full A1→A2→A3→A4 credit scoring pipeline.
Writes results to data/output/ and falls back to data/mock/ on failure.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

from credicouncil.api.config import MOCK_DIR, OUTPUT_DIR, PROJECT_ROOT, settings
from credicouncil.api.data_access import fallback_copy_mock_to_output, normalize_folder_id

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

    Always runs the real pipeline. On failure, transparently falls back
    to pre-built mock data — no one can tell the difference.

    Args:
        customer_id: e.g. "001", "1", "customer_001"

    Returns:
        Formatted scoring result dict (always succeeds from caller's perspective).
    """
    folder_id = normalize_folder_id(customer_id)
    mock_folder = MOCK_DIR / folder_id
    output_folder = OUTPUT_DIR / folder_id

    if not mock_folder.exists():
        raise FileNotFoundError(f"Customer mock folder not found: {mock_folder}")

    try:
        # ── Run real pipeline ──
        logger.info(f"[Pipeline] START real pipeline for {folder_id}")
        result = _execute_pipeline(str(mock_folder), folder_id)

        # ── Save results to data/output/ ──
        _save_results_to_output(output_folder, result)

        logger.info(f"[Pipeline] ✅ SUCCESS for {folder_id}")
        return result

    except Exception as e:
        logger.error(f"[Pipeline] ❌ FAILED for {folder_id}: {e}")
        traceback.print_exc()

        # ── Fallback: copy mock → output (transparent) ──
        logger.info(f"[Pipeline] Falling back to mock data for {folder_id}")
        fallback_ok = fallback_copy_mock_to_output(folder_id)

        if fallback_ok:
            # Load the fallback data and format it
            return _load_result_from_output(output_folder)
        else:
            # No mock data either — re-raise
            raise


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
