"""
CreditLens A4 — Main Report Generator Agent.

Generates 4C credit assessments grounded in SHAP values, with RAG-based
policy citation. This is the LangGraph node for Agent A4.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from creditlens.agents.a4_report_generator.consistency_validator import validate_narrative_consistency
from creditlens.config.prompts import A4_REPORT_GENERATION_SYSTEM, A4_REPORT_GENERATION_USER
from creditlens.state.credit_state import CreditState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class ReportGeneratorAgent:
    """Agent A4 — Report Generator & Explainability Stack.

    Converts SHAP mathematical output into human-readable Vietnamese
    credit reports with 3 layers of explainability:
        1. SHAP Feature Attribution (mathematical)
        2. Grounded LLM Narrative (human-readable, constrained)
        3. Immutable Audit Trail (regulatory)

    Hard constraint: narrative MUST only reference SHAP factors.
    """

    def __init__(self, use_mock: bool = True, bedrock_client=None):
        self.use_mock = use_mock
        self.bedrock_client = bedrock_client

    def generate_report_only(
        self,
        state: CreditState,
        violation_feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate 4C report WITHOUT consistency validation.

        Used by the 9-node graph where consistency_validator is a separate node.

        Args:
            state: Current pipeline state with A3 SHAP output.
            violation_feedback: Previous consistency violations for retry.

        Returns:
            State update dict with narrative and report (no consistency check).
        """
        logger.info(f"A4 generate_report_only — App {state.get('application_id', 'unknown')}")

        shap_values = state.get("shap_values", {})
        warnings = state.get("warnings", [])

        narrative = self._generate_narrative(
            shap_values=shap_values,
            warnings=warnings,
            customer_type=state.get("customer_type", "INDIVIDUAL"),
            thin_file=state.get("structured_feats", {}).get("thin_file_flag", False),
            previous_violations=violation_feedback or [],
        )

        # Build 4C scores
        four_c_scores = {}
        for dim in ["character", "capacity", "capital", "conditions"]:
            assessment = narrative.get(f"{dim}_assessment", {})
            four_c_scores[dim] = assessment.get("score", 0)

        # Build final report
        final_report = {
            "executive_summary": {
                "credit_score": state.get("credit_score"),
                "risk_band": state.get("risk_band"),
                "pd_pct": state.get("pd_pct"),
                "recommendation": narrative.get("recommendation", "REVIEW"),
                "overall_confidence": state.get("overall_confidence"),
            },
            "four_c_scorecard": narrative,
            "suggested_terms": narrative.get("suggested_terms", {}),
            "caveats": narrative.get("caveats", []) + warnings,
            "audit_reference": {
                "application_id": state.get("application_id"),
                "model_version": shap_values.get("model_version"),
                "inference_timestamp": shap_values.get("inference_timestamp"),
            },
        }

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A4",
            "action": "report_generation",
            "input_summary": {
                "shap_factors": len(shap_values.get("top_positive_factors", []))
                + len(shap_values.get("top_negative_factors", [])),
            },
            "output_summary": {
                "recommendation": narrative.get("recommendation"),
            },
            "model_version": "claude-3.5-sonnet",
            "confidence": None,
        }

        return {
            "four_c_scores": four_c_scores,
            "narrative": narrative,
            "final_report": final_report,
            "warnings": warnings,
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
        }

    def generate(self, state: CreditState) -> dict[str, Any]:
        """Generate credit report — LangGraph node function (legacy single-node).

        Includes retry loop: if consistency validation fails,
        re-generates with violation feedback (max 2 retries).

        Args:
            state: Current pipeline state with A3 SHAP output.

        Returns:
            State update dict with A4 outputs.
        """
        logger.info(f"A4 Report Generator — App {state.get('application_id', 'unknown')}")

        shap_values = state.get("shap_values", {})
        warnings = state.get("warnings", [])

        narrative = None
        consistency_result = None

        for attempt in range(1 + MAX_RETRIES):
            # Generate report
            narrative = self._generate_narrative(
                shap_values=shap_values,
                warnings=warnings,
                customer_type=state.get("customer_type", "INDIVIDUAL"),
                thin_file=state.get("structured_feats", {}).get("thin_file_flag", False),
                previous_violations=consistency_result.get("violations", []) if consistency_result else [],
            )

            # Validate consistency
            consistency_result = validate_narrative_consistency(shap_values, narrative)

            if consistency_result["passed"]:
                logger.info(f"Report generated (attempt {attempt + 1}): consistency PASSED")
                break
            else:
                logger.warning(
                    f"Report attempt {attempt + 1}: consistency FAILED "
                    f"({len(consistency_result['violations'])} violations)"
                )

        # If still failed after retries, flag for human review
        if not consistency_result["passed"]:
            warnings.append(
                "⚠️ Báo cáo tự động không đạt consistency check sau 3 lần thử. "
                "Cần xem xét thủ công."
            )

        # Build 4C scores
        four_c_scores = {}
        for dim in ["character", "capacity", "capital", "conditions"]:
            assessment = narrative.get(f"{dim}_assessment", {})
            four_c_scores[dim] = assessment.get("score", 0)

        # Build final report
        final_report = {
            "executive_summary": {
                "credit_score": state.get("credit_score"),
                "risk_band": state.get("risk_band"),
                "pd_pct": state.get("pd_pct"),
                "recommendation": narrative.get("recommendation", "REVIEW"),
                "overall_confidence": state.get("overall_confidence"),
            },
            "four_c_scorecard": narrative,
            "suggested_terms": narrative.get("suggested_terms", {}),
            "caveats": narrative.get("caveats", []) + warnings,
            "audit_reference": {
                "application_id": state.get("application_id"),
                "model_version": shap_values.get("model_version"),
                "inference_timestamp": shap_values.get("inference_timestamp"),
            },
        }

        # Audit entry
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A4",
            "action": "report_generation",
            "input_summary": {
                "shap_factors": len(shap_values.get("top_positive_factors", []))
                + len(shap_values.get("top_negative_factors", [])),
            },
            "output_summary": {
                "recommendation": narrative.get("recommendation"),
                "consistency_passed": consistency_result["passed"],
                "shap_coverage": consistency_result["shap_coverage"],
                "attempts": min(3, (consistency_result.get("violations", []) and 3) or 1),
            },
            "model_version": "claude-3.5-sonnet",
            "confidence": consistency_result["shap_coverage"],
        }

        return {
            "four_c_scores": four_c_scores,
            "narrative": narrative,
            "consistency_check": consistency_result,
            "final_report": final_report,
            "warnings": warnings,
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
        }

    def _generate_narrative(
        self,
        shap_values: dict,
        warnings: list[str],
        customer_type: str,
        thin_file: bool,
        previous_violations: list[str],
    ) -> dict[str, Any]:
        """Generate 4C narrative from SHAP values.

        Args:
            shap_values: SHAP JSON from A3.
            warnings: Current warning list.
            customer_type: INDIVIDUAL or SME.
            thin_file: Whether this is a thin-file customer.
            previous_violations: Violations from previous attempt (for retry).

        Returns:
            4C assessment narrative dict.
        """
        if self.use_mock:
            return self._mock_narrative(shap_values)

        # Build prompt
        system = A4_REPORT_GENERATION_SYSTEM
        if previous_violations:
            system += (
                "\n\nPREVIOUS ATTEMPT FAILED. Fix these violations:\n"
                + "\n".join(f"- {v}" for v in previous_violations)
            )

        user = A4_REPORT_GENERATION_USER.format(
            shap_json=json.dumps(shap_values, indent=2),
            rag_context="[Policy context will be retrieved from RAG]",
            warnings_json=json.dumps(warnings),
            customer_type=customer_type,
            thin_file_flag=thin_file,
        )

        response = self.bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }),
        )
        response_body = json.loads(response["body"].read())
        return json.loads(response_body["content"][0]["text"])

    def _mock_narrative(self, shap_values: dict) -> dict[str, Any]:
        """Generate mock 4C narrative for development."""
        score = shap_values.get("credit_score", 672)
        risk_band = shap_values.get("risk_band", "AA")

        return {
            "character_assessment": {
                "score": 28,
                "status": "DAT",
                "indicators_met": [
                    "Phát hiện giao dịch lương đều đặn — lịch sử 6 tháng nhất quán",
                    "Thanh toán hóa đơn đúng hạn 90% — tín hiệu uy tín tốt",
                ],
                "indicators_review": [],
                "narrative": (
                    "Khách hàng thể hiện uy tín tín dụng tốt qua các chỉ tiêu hành vi. "
                    "Phát hiện giao dịch lương đều đặn trong 6 tháng liên tục, "
                    "thanh toán hóa đơn đúng hạn đạt 90%. "
                    "Không ghi nhận nợ xấu trong lịch sử tín dụng."
                ),
            },
            "capacity_assessment": {
                "score": 31,
                "status": "XEM_XET",
                "indicators_met": [
                    "Thu nhập ổn định 6 tháng (index 0.81)",
                    "Thu nhập khai báo khớp sao kê ngân hàng",
                ],
                "indicators_review": [
                    "Tỷ lệ nợ/thu nhập ở mức cao (48%) > ngưỡng tốt 40% → "
                    "Đề xuất: giảm hạn mức 20% hoặc yêu cầu chứng minh thu nhập bổ sung",
                    "Số dư về âm 2 lần trong 6 tháng cần giải trình",
                ],
                "narrative": (
                    "Năng lực trả nợ của khách hàng ở mức khá nhưng cần lưu ý. "
                    "Thu nhập ổn định 6 tháng với index 0.81, thu nhập khai báo khớp sao kê. "
                    "Tuy nhiên, tỷ lệ nợ/thu nhập ở mức cao 48%, vượt ngưỡng tốt 40%. "
                    "Ghi nhận số dư về âm 2 lần trong 6 tháng."
                ),
            },
            "capital_assessment": {
                "score": 16,
                "status": "DAT",
                "indicators_met": [
                    "Tài sản bảo đảm đáp ứng yêu cầu",
                ],
                "indicators_review": [],
                "narrative": (
                    "Vốn và tài sản bảo đảm của khách hàng đáp ứng yêu cầu. "
                    "Tài sản bảo đảm đủ buffer cho khoản vay hiện tại."
                ),
            },
            "conditions_assessment": {
                "score": 9,
                "status": "DAT",
                "indicators_met": [
                    "Mục đích vay rõ ràng",
                    "Kế hoạch trả nợ chi tiết",
                ],
                "indicators_review": [],
                "narrative": (
                    "Điều kiện khoản vay phù hợp. Mục đích vay rõ ràng — tiêu dùng. "
                    "Kế hoạch trả nợ chi tiết, phù hợp với năng lực tài chính."
                ),
            },
            "recommendation": "APPROVE" if score >= 640 else "REVIEW",
            "suggested_terms": {
                "max_amount_vnd": 80_000_000,
                "max_term_months": 24,
            },
            "caveats": [],
        }
