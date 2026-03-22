"""
CreditLens A4 — Report Generator Agent (Local Version).

# LOCAL_SUB: Uses Gemini API instead of Bedrock Claude.
# Production: Replace LLMService with BedrockLLMService.

Generates Vietnamese credit assessment reports:
1. Thông tin khách hàng (Customer Info)
2. Tóm tắt đánh giá (Executive Summary) + Scorecard
3. 5C Scorecard (Character, Capacity, Capital, Conditions, Collateral)
4. Tình hình tài chính
5. Tài sản bảo đảm
6. Khuyến nghị & Caveats + Audit trail
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from creditlens.services.llm_service import LLMService
from creditlens.agents.a4_report_generator.consistency_validator import validate_narrative_consistency
from creditlens.config.feature_config import get_label_vi

logger = logging.getLogger(__name__)

# ── Report Generation Prompts (5C + 6 Sections) ──
REPORT_SYSTEM = """Bạn là chuyên gia phân tích tín dụng tại ngân hàng Việt Nam.
Tạo báo cáo đánh giá tín dụng 5C bằng tiếng Việt theo chuẩn TT39/2016.

5C: Character (Uy tín), Capacity (Năng lực trả nợ), Capital (Vốn tự có),
    Conditions (Điều kiện), Collateral (Tài sản bảo đảm).

QUAN TRỌNG:
- Chỉ trích dẫn các yếu tố từ SHAP values được cung cấp
- Không bịa đặt thông tin hoặc yếu tố không có trong dữ liệu
- Sử dụng số liệu cụ thể khi có thể
- Viết bằng văn phong ngân hàng trang trọng

Trả lời CHỈ bằng JSON hợp lệ theo schema sau:
{
    "customer_info": {
        "summary": "Tóm tắt hồ sơ khách hàng (1-2 câu)"
    },
    "character_assessment": {
        "score": 0-30,
        "status": "DAT|XEM_XET|KHONG_DAT",
        "shap_pct": "% SHAP contribution",
        "indicators_met": ["..."],
        "indicators_review": ["..."],
        "narrative": "100-150 chữ"
    },
    "capacity_assessment": {
        "score": 0-40,
        "status": "DAT|XEM_XET|KHONG_DAT",
        "shap_pct": "% SHAP contribution",
        "indicators_met": ["..."],
        "indicators_review": ["..."],
        "narrative": "100-150 chữ"
    },
    "capital_assessment": {
        "score": 0-20,
        "status": "DAT|XEM_XET|KHONG_DAT",
        "shap_pct": "% SHAP contribution",
        "indicators_met": ["..."],
        "indicators_review": ["..."],
        "narrative": "100-150 chữ"
    },
    "conditions_assessment": {
        "score": 0-10,
        "status": "DAT|XEM_XET|KHONG_DAT",
        "shap_pct": "% SHAP contribution",
        "indicators_met": ["..."],
        "indicators_review": ["..."],
        "narrative": "100-150 chữ"
    },
    "collateral_assessment": {
        "score": 0-20,
        "status": "DAT|XEM_XET|KHONG_DAT",
        "indicators_met": ["..."],
        "indicators_review": ["..."],
        "narrative": "50-100 chữ"
    },
    "financial_summary": {
        "income_analysis": "Phân tích thu nhập và dòng tiền",
        "debt_analysis": "Phân tích nợ và khả năng trả nợ",
        "key_ratios": {"dti": "...", "dscr": "...", "ltv": "..."}
    },
    "recommendation": "APPROVE|REVIEW|REJECT",
    "suggested_terms": {
        "max_amount_vnd": number,
        "max_term_months": number,
        "interest_rate_suggestion": "theo biểu phí hiện hành",
        "conditions": ["Điều kiện tiên quyết"]
    },
    "caveats": ["..."]
}"""

REPORT_USER = """Tạo báo cáo 5C + 6 phần cho đơn vay với thông tin sau:

=== SHAP Feature Attribution ===
{shap_json}

=== Phân bổ SHAP theo 5C ===
{five_c_allocation}

=== Cảnh báo ===
{warnings_json}

=== Thông tin bổ sung ===
- Loại khách hàng: {customer_type}
- Thin-file: {thin_file_flag}
- Tổng số điểm tín dụng: {credit_score}
- Risk band: {risk_band}
- PD%: {pd_pct}

=== Thông tin tài chính ===
{financial_info}

=== Thông tin tài sản bảo đảm ===
{collateral_info}

Tạo đánh giá chi tiết cho từng C, chỉ TRÍCH DẪN các yếu tố từ SHAP values.

CHÚ Ý THANG ĐIỂM (KHÔNG ĐƯỢC VƯỢT):
- Character: 0-30 điểm
- Capacity: 0-40 điểm
- Capital: 0-20 điểm
- Conditions: 0-10 điểm
- Collateral: 0-20 điểm
Tổng tối đa: 120 điểm."""


class ReportGeneratorAgent:
    """Agent A4 — Report Generator + Explainability.

    # LOCAL_SUB: Uses Gemini API. Production: use Bedrock Claude.

    Converts SHAP output into human-readable Vietnamese 5C reports (6 sections).
    """

    def __init__(self, use_mock: bool = True):
        self.llm = LLMService(use_mock=use_mock)
        self.use_mock = use_mock

    def generate(self, a3_output: dict[str, Any], a2_output: dict[str, Any]) -> dict[str, Any]:
        """Generate credit report from A3 scoring output.

        Args:
            a3_output: Output from A3 ScoringAgent.score()
            a2_output: Output from A2 FeatureEngineerAgent.process()

        Returns:
            Dict with final_report, narrative, consistency_check (5C + 6 sections)
        """
        logger.info("=" * 60)
        logger.info("  A4 Report Generator (5C + 6 Sections)")
        logger.info("=" * 60)

        credit_score = a3_output.get("credit_score", 0)
        pd_pct = a3_output.get("pd_pct", 0)
        risk_band = a3_output.get("risk_band", "CCC")
        shap_values = a3_output.get("shap_values", {})
        warnings = a2_output.get("warnings", [])
        llm_feats = a2_output.get("llm_feats", {})

        # Generate narrative (5C)
        narrative = self._generate_narrative(
            shap_values=shap_values,
            warnings=warnings,
            credit_score=credit_score,
            risk_band=risk_band,
            pd_pct=pd_pct,
            customer_type="INDIVIDUAL",
            thin_file=llm_feats.get("thin_file_flag", False),
            llm_feats=llm_feats,
        )

        # Validate consistency (narrative must only cite SHAP factors)
        consistency = validate_narrative_consistency(shap_values, narrative)
        logger.info(f"  Consistency check: {'PASSED' if consistency['passed'] else 'FAILED'}")

        # Build 5C scores (with clamping to valid ranges)
        five_c_max = {
            "character": 30, "capacity": 40, "capital": 20,
            "conditions": 10, "collateral": 20,
        }
        five_c_scores = {}
        for dim, max_score in five_c_max.items():
            assessment = narrative.get(f"{dim}_assessment") or {}
            raw_score = assessment.get("score", 0) if isinstance(assessment, dict) else 0
            clamped = max(0, min(int(raw_score), max_score))
            five_c_scores[dim] = clamped
            # Write clamped score back to narrative for consistency
            if isinstance(assessment, dict) and assessment.get("score") != clamped:
                assessment["score"] = clamped

        total_5c = sum(five_c_scores.values())
        logger.info(f"  5C Total: {total_5c}/120")
        for dim, score in five_c_scores.items():
            logger.info(f"    {dim}: {score}/{five_c_max[dim]}")

        # Build final report (6 sections)
        final_report = {
            # Section I: Thông tin khách hàng
            "customer_info": narrative.get("customer_info", {}),
            # Section II: Tóm tắt đánh giá
            "executive_summary": {
                "credit_score": credit_score,
                "risk_band": risk_band,
                "pd_pct": pd_pct,
                "recommendation": narrative.get("recommendation", "REVIEW"),
                "five_c_total": total_5c,
                "five_c_scores": five_c_scores,
                "five_c_shap_allocation": shap_values.get("five_c_shap_allocation", {}),
            },
            # Section III: 5C Scorecard
            "five_c_scorecard": {
                "character_assessment": narrative.get("character_assessment", {}),
                "capacity_assessment": narrative.get("capacity_assessment", {}),
                "capital_assessment": narrative.get("capital_assessment", {}),
                "conditions_assessment": narrative.get("conditions_assessment", {}),
                "collateral_assessment": narrative.get("collateral_assessment", {}),
            },
            # Section IV: Tình hình tài chính
            "financial_summary": narrative.get("financial_summary", {}),
            # Section V: Tài sản bảo đảm (detail from collateral assessment)
            "collateral_detail": narrative.get("collateral_assessment", {}),
            # Section VI: Khuyến nghị & Caveats
            "suggested_terms": narrative.get("suggested_terms", {}),
            "llm_insights": {
                "loan_purpose": llm_feats.get("loan_purpose_category"),
                "positive_signals": llm_feats.get("positive_signals", []),
                "risk_flags": llm_feats.get("risk_flags", []),
            },
            "caveats": narrative.get("caveats", []) + warnings,
            "audit_reference": {
                "model_version": shap_values.get("model_version"),
                "inference_timestamp": shap_values.get("inference_timestamp"),
            },
        }

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A4",
            "action": "report_generation_5c",
            "output_summary": {
                "recommendation": narrative.get("recommendation"),
                "five_c_total": total_5c,
                "consistency_passed": consistency.get("passed"),
            },
            "model_version": "gemini-2.5-flash-lite",
        }

        logger.info(f"  Recommendation: {narrative.get('recommendation', 'REVIEW')}")

        return {
            "credit_score": credit_score,
            "pd_pct": pd_pct,
            "risk_band": risk_band,
            "five_c_scores": five_c_scores,
            "narrative": narrative,
            "consistency_check": consistency,
            "final_report": final_report,
            "warnings": warnings,
            "audit_trail": a3_output.get("audit_trail", []) + [audit_entry],
        }

    def _generate_narrative(
        self,
        shap_values: dict,
        warnings: list[str],
        credit_score: int,
        risk_band: str,
        pd_pct: float,
        customer_type: str,
        thin_file: bool,
        llm_feats: dict | None = None,
    ) -> dict[str, Any]:
        """Generate 5C narrative using LLM or mock."""
        if self.use_mock:
            return self._mock_narrative(shap_values, credit_score, llm_feats or {})

        five_c_alloc = shap_values.get("five_c_shap_allocation", {})
        financial_info = self._build_financial_context(llm_feats or {})
        collateral_info = self._build_collateral_context(llm_feats or {})

        prompt = REPORT_USER.format(
            shap_json=json.dumps(shap_values, indent=2, default=str),
            five_c_allocation=json.dumps(five_c_alloc, indent=2, default=str),
            warnings_json=json.dumps(warnings, default=str),
            customer_type=customer_type,
            thin_file_flag=thin_file,
            credit_score=credit_score,
            risk_band=risk_band,
            pd_pct=pd_pct,
            financial_info=financial_info,
            collateral_info=collateral_info,
        )

        return self.llm.generate_json(
            REPORT_SYSTEM, prompt,
            {"character_assessment", "capacity_assessment",
             "capital_assessment", "conditions_assessment",
             "collateral_assessment", "recommendation"},
            max_tokens=8192,
        )

    def _build_financial_context(self, llm_feats: dict) -> str:
        """Build financial context string from available data."""
        lines = []
        if llm_feats.get("avg_monthly_inflow_vnd"):
            lines.append(f"Dòng tiền vào TB: {llm_feats['avg_monthly_inflow_vnd']:,.0f} VND/tháng")
        if llm_feats.get("income_stability_index"):
            lines.append(f"Chỉ số ổn định thu nhập: {llm_feats['income_stability_index']:.2f}")
        if llm_feats.get("inflow_outflow_ratio"):
            lines.append(f"Tỷ lệ thu/chi: {llm_feats['inflow_outflow_ratio']:.2f}")
        return "\n".join(lines) if lines else "Không có dữ liệu tài chính bổ sung"

    def _build_collateral_context(self, llm_feats: dict) -> str:
        """Build collateral context string from available data."""
        lines = []
        if llm_feats.get("collateral_type"):
            lines.append(f"Loại TSBĐ: {llm_feats['collateral_type']}")
        if llm_feats.get("collateral_value_vnd"):
            lines.append(f"Giá trị thẩm định: {llm_feats['collateral_value_vnd']:,.0f} VND")
        return "\n".join(lines) if lines else "Chưa có thông tin TSBĐ chi tiết"

    def _mock_narrative(
        self, shap_values: dict, credit_score: int, llm_feats: dict
    ) -> dict[str, Any]:
        """Generate deterministic mock 5C narrative (6 sections)."""
        # Derive scores based on credit score
        if credit_score >= 700:
            char_s, cap_s, capital_s, cond_s, coll_s = 28, 35, 18, 9, 16
            recommendation = "APPROVE"
        elif credit_score >= 600:
            char_s, cap_s, capital_s, cond_s, coll_s = 25, 28, 15, 8, 14
            recommendation = "REVIEW"
        elif credit_score >= 460:
            char_s, cap_s, capital_s, cond_s, coll_s = 20, 22, 12, 6, 10
            recommendation = "CONDITIONAL"
        else:
            char_s, cap_s, capital_s, cond_s, coll_s = 14, 15, 8, 4, 7
            recommendation = "REJECT"

        # Extract SHAP factor labels for narrative
        top_pos = shap_values.get("top_positive_factors", [])
        top_neg = shap_values.get("top_negative_factors", [])
        pos_labels = [f.get("label_vi", f.get("feature", "")) for f in top_neg[:3]]
        neg_labels = [f.get("label_vi", f.get("feature", "")) for f in top_pos[:3]]

        # 5C SHAP allocation (from A3 output)
        five_c_alloc = shap_values.get("five_c_shap_allocation", {})

        return {
            # Section I
            "customer_info": {
                "summary": (
                    f"Khách hàng {'cá nhân' if not llm_feats.get('is_sme') else 'doanh nghiệp'}. "
                    f"{'Thin-file — không có lịch sử tín dụng CIC. ' if llm_feats.get('thin_file_flag') else ''}"
                    f"Điểm tín dụng: {credit_score}/850."
                ),
            },
            # Section III — 5C assessments
            "character_assessment": {
                "score": char_s,
                "status": "DAT" if char_s >= 22 else "XEM_XET" if char_s >= 15 else "KHONG_DAT",
                "shap_pct": five_c_alloc.get("character", {}).get("pct", 0),
                "indicators_met": [
                    "Lịch sử tín dụng tích cực từ CIC",
                    "Thanh toán đúng hạn theo sao kê ngân hàng",
                ],
                "indicators_review": (
                    ["Thin-file — chưa đủ lịch sử tín dụng"] if llm_feats.get("thin_file_flag") else []
                ),
                "narrative": (
                    f"Khách hàng thể hiện uy tín tín dụng {'tốt' if char_s >= 22 else 'cần xem xét'}. "
                    f"Điểm CIC ở mức {'khá' if char_s >= 22 else 'chưa đủ đánh giá'}, "
                    f"không ghi nhận nợ xấu. "
                    f"Các yếu tố tích cực: {', '.join(pos_labels) if pos_labels else 'không có dữ liệu đầy đủ'}."
                ),
            },
            "capacity_assessment": {
                "score": cap_s,
                "status": "DAT" if cap_s >= 28 else "XEM_XET" if cap_s >= 18 else "KHONG_DAT",
                "shap_pct": five_c_alloc.get("capacity", {}).get("pct", 0),
                "indicators_met": [
                    "Thu nhập ổn định, có xác minh qua HĐLĐ",
                    "Tỷ lệ nợ/thu nhập trong ngưỡng chấp nhận được",
                ],
                "indicators_review": (
                    [f"Cần lưu ý: {', '.join(neg_labels)}"] if neg_labels else []
                ),
                "narrative": (
                    f"Năng lực trả nợ {'đáp ứng yêu cầu' if cap_s >= 28 else 'cần xem xét thêm'}. "
                    f"Thu nhập được xác minh qua hợp đồng lao động và sao kê ngân hàng. "
                    f"SHAP contribution: {five_c_alloc.get('capacity', {}).get('pct', 0)}%."
                ),
            },
            "capital_assessment": {
                "score": capital_s,
                "status": "DAT" if capital_s >= 14 else "XEM_XET" if capital_s >= 10 else "KHONG_DAT",
                "shap_pct": five_c_alloc.get("capital", {}).get("pct", 0),
                "indicators_met": ["Vốn tự có và tài sản ròng ở mức chấp nhận được"],
                "indicators_review": [],
                "narrative": (
                    f"Vốn tự có {'đáp ứng' if capital_s >= 14 else 'cần bổ sung thêm cho'} "
                    f"yêu cầu khoản vay."
                ),
            },
            "conditions_assessment": {
                "score": cond_s,
                "status": "DAT" if cond_s >= 7 else "XEM_XET" if cond_s >= 5 else "KHONG_DAT",
                "shap_pct": five_c_alloc.get("conditions", {}).get("pct", 0),
                "indicators_met": [
                    "Mục đích vay rõ ràng",
                    "Kế hoạch trả nợ phù hợp",
                ],
                "indicators_review": [],
                "narrative": (
                    "Điều kiện khoản vay phù hợp, mục đích vay rõ ràng. "
                    "Ngành nghề ổn định, điều kiện thị trường thuận lợi."
                ),
            },
            "collateral_assessment": {
                "score": coll_s,
                "status": "DAT" if coll_s >= 14 else "XEM_XET" if coll_s >= 8 else "KHONG_DAT",
                "indicators_met": ["Có tài sản bảo đảm phù hợp"] if coll_s >= 14 else [],
                "indicators_review": (
                    ["Cần bổ sung hoặc tái định giá TSBĐ"] if coll_s < 14 else []
                ),
                "narrative": (
                    f"Tài sản bảo đảm {'đáp ứng yêu cầu' if coll_s >= 14 else 'cần đánh giá thêm'} "
                    f"cho khoản vay. "
                    f"{'Đề xuất tái định giá sau 12 tháng.' if coll_s < 18 else ''}"
                ),
            },
            # Section IV
            "financial_summary": {
                "income_analysis": (
                    f"Thu nhập ổn định qua xác minh sao kê ngân hàng. "
                    f"{'Income stability index ở mức tốt.' if credit_score >= 600 else 'Cần xem xét thêm.'}"
                ),
                "debt_analysis": (
                    f"Tỷ lệ nợ/thu nhập {'trong ngưỡng an toàn' if credit_score >= 640 else 'ở mức cao'}."
                ),
                "key_ratios": {
                    "dti": f"{'< 40%' if credit_score >= 700 else '40-50%' if credit_score >= 600 else '> 50%'}",
                    "dscr": f"{'> 1.2' if credit_score >= 700 else '1.0-1.2' if credit_score >= 600 else '< 1.0'}",
                    "ltv": "N/A (chưa có thông tin TSBĐ chi tiết)",
                },
            },
            # Section VI
            "recommendation": recommendation,
            "suggested_terms": {
                "max_amount_vnd": 300_000_000 if credit_score >= 650 else 100_000_000,
                "max_term_months": 36 if credit_score >= 650 else 12,
                "interest_rate_suggestion": "Theo biểu phí hiện hành",
                "conditions": (
                    ["Chứng minh thu nhập bổ sung (slip lương 3 tháng)"]
                    if credit_score < 700 else []
                ),
            },
            "caveats": [],
        }
