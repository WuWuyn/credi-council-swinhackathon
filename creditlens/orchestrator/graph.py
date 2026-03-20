"""
CreditLens Orchestrator — LangGraph StateGraph Definition.

Assembles all 4 agents + routing nodes into a single LangGraph
StateGraph that processes credit applications end-to-end.

Graph topology (9 nodes per design document Section 3.1):
    START → ingest_documents → parallel(check_cic, analyze_transactions) →
    confidence_gate → llm_feature_engineer → ml_score →
    report_generator → consistency_validator → decision_router → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import START, END, StateGraph

from creditlens.state.credit_state import CreditState, RoutingDecision
from creditlens.orchestrator.confidence_gate import confidence_gate

logger = logging.getLogger(__name__)


def build_graph(
    ingestion_agent=None,
    feature_engineer_agent=None,
    scoring_agent=None,
    report_generator_agent=None,
    use_mock: bool = True,
) -> StateGraph:
    """Build the CreditLens LangGraph StateGraph.

    9-node graph with parallel CIC + transaction branches:
        ingest_documents → [check_cic ∥ analyze_transactions] →
        confidence_gate → llm_feature_engineer → ml_score →
        report_generator → consistency_validator → decision_router → END

    Args:
        ingestion_agent: A1 IngestionAgent instance.
        feature_engineer_agent: A2 FeatureEngineerAgent instance.
        scoring_agent: A3 ScoringAgent instance.
        report_generator_agent: A4 ReportGeneratorAgent instance.
        use_mock: Use mock implementations if agents not provided.

    Returns:
        Compiled LangGraph StateGraph.
    """
    logger.info("Building CreditLens LangGraph StateGraph (9 nodes)...")

    # ── Initialize agents if not provided ──
    if ingestion_agent is None:
        from creditlens.agents.a1_ingestion.agent import IngestionAgent
        ingestion_agent = IngestionAgent(use_mock=use_mock)

    if feature_engineer_agent is None:
        from creditlens.agents.a2_feature_engineer.agent import FeatureEngineerAgent
        feature_engineer_agent = FeatureEngineerAgent(use_mock=use_mock)

    if report_generator_agent is None:
        from creditlens.agents.a4_report_generator.agent import ReportGeneratorAgent
        report_generator_agent = ReportGeneratorAgent(use_mock=use_mock)

    # ── Define node functions ──

    def ingest_documents_node(state: CreditState) -> dict[str, Any]:
        """Node 1: ingest_documents — Textract OCR extraction from PDF/scan documents.

        This node handles Channel 1 only (PDF/Scan). CIC and bank statement
        analysis are performed in parallel by separate nodes.
        """
        return ingestion_agent.ingest_documents(
            applicant_id=state.get("application_id", "unknown"),
            customer_type=state.get("customer_type", "INDIVIDUAL"),
            documents=state.get("_input_documents"),
        )

    def check_cic_node(state: CreditState) -> dict[str, Any]:
        """Node 2: check_cic — API call to Credit Information Center.

        Runs in PARALLEL with analyze_transactions after ingest_documents.
        Queries CIC for credit history; sets thin_file_flag if no record found.
        """
        return ingestion_agent.check_cic(
            applicant_id=state.get("application_id", "unknown"),
        )

    def analyze_transactions_node(state: CreditState) -> dict[str, Any]:
        """Node 3: analyze_transactions — Parse bank statement CSV.

        Runs in PARALLEL with check_cic after ingest_documents.
        Extracts 8 alternative data features from 6-month bank statement.
        """
        return ingestion_agent.analyze_transactions(
            bank_statement_path=state.get("_input_bank_statement"),
        )

    def confidence_gate_node(state: CreditState) -> dict[str, Any]:
        """Node 4: confidence_gate — Conditional router.

        Joins results from check_cic + analyze_transactions.
        Routes: HALT (critical missing) or PROCEED.
        """
        return confidence_gate(state)

    def feature_engineer_node(state: CreditState) -> dict[str, Any]:
        """Node 5: llm_feature_engineer — Claude LLM processing.

        Variant A: Semantic extraction (always runs).
        Variant B: Intelligent imputation (when IMPORTANT fields missing).
        """
        return feature_engineer_agent.process(state)

    def ml_score_node(state: CreditState) -> dict[str, Any]:
        """Node 6: ml_score — LightGBM + SHAP.

        Deterministic scoring: predict_proba → PD → credit score → SHAP.
        """
        if scoring_agent is None:
            logger.warning("A3 Scoring Agent not initialized — returning mock score")
            return {
                "credit_score": 672,
                "pd_pct": 5.8,
                "risk_band": "AA",
                "shap_values": {
                    "credit_score": 672,
                    "pd_pct": 5.8,
                    "risk_band": "AA",
                    "model_version": "lgbm_v1_mock",
                    "top_positive_factors": [
                        {"feature": "salary_pattern_detected", "shap": 0.089,
                         "value": True, "label_vi": "Phát hiện giao dịch lương đều đặn"},
                        {"feature": "income_stability_index", "shap": 0.072,
                         "value": 0.81, "label_vi": "Thu nhập ổn định 6 tháng"},
                    ],
                    "top_negative_factors": [
                        {"feature": "dti_ratio", "shap": -0.063,
                         "value": 0.48, "label_vi": "Tỷ lệ nợ/thu nhập ở mức cao (48%)"},
                    ],
                    "4c_shap_allocation": {
                        "character": {"shap_sum": 0.118, "pct": 28},
                        "capacity": {"shap_sum": 0.172, "pct": 41},
                        "capital": {"shap_sum": 0.080, "pct": 19},
                        "conditions": {"shap_sum": 0.050, "pct": 12},
                    },
                },
            }
        return scoring_agent.score(state)

    def report_generator_node(state: CreditState) -> dict[str, Any]:
        """Node 7: report_generator — Claude LLM report generation.

        Generates 4C credit assessment narrative in Vietnamese.
        Does NOT perform consistency validation (separate node).
        """
        return report_generator_agent.generate_report_only(state)

    def consistency_validator_node(state: CreditState) -> dict[str, Any]:
        """Node 8: consistency_validator — Deterministic check.

        Validates that LLM narrative references only SHAP factors.
        If fails, re-prompts report_generator (max 2 retries).
        """
        from creditlens.agents.a4_report_generator.consistency_validator import (
            validate_narrative_consistency,
        )

        shap_output = state.get("shap_values", {})
        narrative = state.get("narrative", {})

        result = validate_narrative_consistency(shap_output, narrative)

        retry_count = state.get("_consistency_retry_count", 0)

        if result["passed"]:
            return {
                "consistency_check": result,
                "routing": state.get("routing", "REVIEW"),
            }
        elif retry_count < 2:
            # Re-generate report with violation feedback
            logger.warning(
                f"Consistency check failed (attempt {retry_count + 1}/2): "
                f"{result['violations']}"
            )
            regenerated = report_generator_agent.generate_report_only(
                state,
                violation_feedback=result["violations"],
            )
            regenerated["_consistency_retry_count"] = retry_count + 1
            # Re-validate
            new_narrative = regenerated.get("narrative", {})
            recheck = validate_narrative_consistency(shap_output, new_narrative)
            regenerated["consistency_check"] = recheck
            if not recheck["passed"]:
                regenerated["warnings"] = state.get("warnings", []) + [
                    "Consistency check failed after retries — flagged for human review"
                ]
            return regenerated
        else:
            # Max retries reached → flag for human review
            return {
                "consistency_check": result,
                "warnings": state.get("warnings", []) + [
                    "Consistency check failed after max retries — requires human review"
                ],
            }

    def decision_router_node(state: CreditState) -> dict[str, Any]:
        """Node 9: decision_router — Policy-based hard override rules.

        Applies deterministic policy rules that override ML decisions:
        - CIC Group 4-5 → REJECT
        - Loan > 10B VND → ESCALATE
        - Confidence < 0.65 → HUMAN REVIEW
        - Thin-file + score < 560 → increase collateral requirement
        """
        from creditlens.agents.a3_scoring.decision_rules import apply_hard_overrides

        result = apply_hard_overrides(
            credit_score=state.get("credit_score", 0),
            risk_band=state.get("risk_band", "CC"),
            auto_decision=state.get("routing", "REVIEW"),
            structured_feats=state.get("structured_feats", {}),
            overall_confidence=state.get("overall_confidence", 0.5),
        )

        return {
            "routing": result["final_decision"],
            "final_report": {
                **state.get("final_report", {}),
                "decision": result["final_decision"],
                "override_applied": result.get("override_applied", False),
                "override_reason": result.get("override_reason", ""),
                "additional_conditions": result.get("additional_conditions", []),
            },
        }

    # ── Route condition functions ──

    def should_continue_after_gate(state: CreditState) -> str:
        """Determine routing after confidence gate."""
        routing = state.get("routing", "")
        if routing == RoutingDecision.HALT.value:
            return "halt"
        return "proceed"

    # ── Build graph (9 nodes) ──
    graph = StateGraph(CreditState)

    # Add all 9 nodes
    graph.add_node("ingest_documents", ingest_documents_node)
    graph.add_node("check_cic", check_cic_node)
    graph.add_node("analyze_transactions", analyze_transactions_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("llm_feature_engineer", feature_engineer_node)
    graph.add_node("ml_score", ml_score_node)
    graph.add_node("report_generator", report_generator_node)
    graph.add_node("consistency_validator", consistency_validator_node)
    graph.add_node("decision_router", decision_router_node)

    # ── Add edges ──

    # START → ingest_documents
    graph.add_edge(START, "ingest_documents")

    # ingest_documents → parallel(check_cic, analyze_transactions)
    graph.add_edge("ingest_documents", "check_cic")
    graph.add_edge("ingest_documents", "analyze_transactions")

    # Both parallel branches → confidence_gate (join)
    graph.add_edge("check_cic", "confidence_gate")
    graph.add_edge("analyze_transactions", "confidence_gate")

    # Conditional: confidence gate → proceed or halt
    graph.add_conditional_edges(
        "confidence_gate",
        should_continue_after_gate,
        {
            "proceed": "llm_feature_engineer",
            "halt": END,
        },
    )

    # Linear pipeline after gate
    graph.add_edge("llm_feature_engineer", "ml_score")
    graph.add_edge("ml_score", "report_generator")
    graph.add_edge("report_generator", "consistency_validator")
    graph.add_edge("consistency_validator", "decision_router")
    graph.add_edge("decision_router", END)

    logger.info("LangGraph StateGraph built successfully (9 nodes)")
    return graph.compile()


def run_pipeline(
    applicant_id: str,
    customer_type: str = "INDIVIDUAL",
    documents: list[dict] | None = None,
    bank_statement_path: str | None = None,
    use_mock: bool = True,
    scoring_agent=None,
) -> CreditState:
    """Run the full CreditLens pipeline.

    Convenience function that builds the graph and runs it end-to-end.

    Args:
        applicant_id: Unique applicant identifier.
        customer_type: INDIVIDUAL or SME.
        documents: List of {type, bytes} for PDF documents.
        bank_statement_path: Path to bank statement CSV.
        use_mock: Use mock AWS services.
        scoring_agent: Pre-built A3 ScoringAgent (with trained model).

    Returns:
        Final CreditState with all agent outputs.
    """
    graph = build_graph(use_mock=use_mock, scoring_agent=scoring_agent)

    initial_state: CreditState = {
        "application_id": applicant_id,
        "customer_type": customer_type,
        "_input_documents": documents,
        "_input_bank_statement": bank_statement_path,
        "audit_trail": [],
        "warnings": [],
        "_consistency_retry_count": 0,
    }

    logger.info(f"Running CreditLens pipeline for {applicant_id}")
    result = graph.invoke(initial_state)

    logger.info(
        f"Pipeline complete — Score: {result.get('credit_score')}, "
        f"Band: {result.get('risk_band')}, "
        f"Decision: {result.get('routing')}"
    )

    return result
