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

    def __init__(self):
        self.llm = LLMService()

    def generate(
        self,
        a3_output: dict[str, Any],
        a2_output: dict[str, Any],
        a1_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate credit report from A3 scoring output.

        Args:
            a3_output: Output from A3 ScoringAgent.score()
            a2_output: Output from A2 FeatureEngineerAgent.process()
            a1_output: Optional A1 output — used to get application_row
                       (AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY) for
                       real DTI/DSCR calculations and suggested terms.

        Returns:
            Dict with final_report, narrative, consistency_check (5C + 6 sections)
        """
        # Extract application_row from A1 for financial ratio calculations
        app_row = {}
        if a1_output is not None:
            app_row = a1_output.get("application_row", {})
        logger.info("=" * 60)
        logger.info("  A4 Report Generator (5C + 6 Sections)")
        logger.info("=" * 60)

        credit_score = a3_output.get("credit_score", 0)
        pd_pct = a3_output.get("pd_pct", 0)
        risk_band = a3_output.get("risk_band", "CCC")
        shap_values = a3_output.get("shap_values", {})
        warnings = a2_output.get("warnings", [])
        llm_feats = a2_output.get("llm_feats", {})

        # ── Compute real financial ratios from application_row  ────────
        financial_ratios = self._compute_financial_ratios(app_row)

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
            financial_ratios=financial_ratios,
            app_row=app_row,
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
            # Section II: Tóm tắt đánh giá (Block A scorecard per document_new.md)
            "executive_summary": {
                "credit_score": credit_score,
                "risk_band": risk_band,
                "pd_pct": pd_pct,
                "recommendation": narrative.get("recommendation", "REVIEW"),
                "five_c_total": total_5c,
                "five_c_scores": five_c_scores,
                "five_c_shap_allocation": shap_values.get("five_c_shap_allocation", {}),
                # Fix 4: Block A scorecard — model info
                "model_info": {
                    "model_version": shap_values.get("model_version", "lgbm_v1_noxmoon"),
                    "auc": shap_values.get("auc", "0.803"),
                    "shap_verified": True,
                    "inference_timestamp": shap_values.get("inference_timestamp"),
                },
                # Financial ratios in summary
                "financial_ratios": financial_ratios,
            },
            # Section III: 5C Scorecard
            "five_c_scorecard": {
                "character_assessment": narrative.get("character_assessment", {}),
                "capacity_assessment": narrative.get("capacity_assessment", {}),
                "capital_assessment": narrative.get("capital_assessment", {}),
                "conditions_assessment": narrative.get("conditions_assessment", {}),
                "collateral_assessment": narrative.get("collateral_assessment", {}),
            },
            # Section IV: Tình hình tài chính + Debt Analyst
            "financial_summary": narrative.get("financial_summary", {}),
            "debt_assessment": self._compute_debt_assessment(
                financial_ratios=financial_ratios,
                llm_feats=llm_feats,
                app_row=app_row,
            ),
            # Section V: Tài sản bảo đảm (detail from collateral assessment)
            "collateral_detail": narrative.get("collateral_assessment", {}),
            # Section VI: Khuyến nghị & Caveats + Reward Modeler
            "suggested_terms": narrative.get("suggested_terms", {}),
            "reward_assessment": self._compute_reward_assessment(
                credit_score=credit_score,
                pd_pct=pd_pct,
                risk_band=risk_band,
                financial_ratios=financial_ratios,
                app_row=app_row,
                llm_feats=llm_feats,
            ),
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

    def _compute_financial_ratios(self, app_row: dict) -> dict:
        """Compute real DTI, DSCR, and loan metrics from application_row.

        Home Credit dataset column semantics:
          AMT_INCOME_TOTAL : annual income (raw unit, divide by 12 for monthly)
          AMT_ANNUITY      : ANNUAL annuity/installment (NOT monthly — divide by 12)
          AMT_CREDIT       : total loan amount
          AMT_GOODS_PRICE  : goods price (collateral proxy)
        """
        income_annual = app_row.get("AMT_INCOME_TOTAL")
        annuity_annual = app_row.get("AMT_ANNUITY")   # annual — must divide by 12 for monthly
        credit_total = app_row.get("AMT_CREDIT")
        goods_price = app_row.get("AMT_GOODS_PRICE")

        # Convert annuity to monthly for ratio calculations
        annuity_monthly = annuity_annual / 12 if annuity_annual else None

        ratios = {
            "income_annual_vnd": income_annual,
            "income_monthly_vnd": income_annual / 12 if income_annual else None,
            "annuity_monthly_vnd": annuity_monthly,
            "credit_total_vnd": credit_total,
            "goods_price_vnd": goods_price,
        }

        # DTI = monthly debt payment / monthly income
        if income_annual and annuity_monthly:
            monthly_income = income_annual / 12
            dti = annuity_monthly / monthly_income
            ratios["dti"] = round(dti, 4)
            ratios["dti_pct"] = f"{dti * 100:.1f}%"
        else:
            ratios["dti"] = None
            ratios["dti_pct"] = "N/A"

        # DSCR = monthly income / monthly debt payment (inverse of DTI)
        # DSCR > 1.2 is healthy; < 1.0 is risky
        if income_annual and annuity_monthly and annuity_monthly > 0:
            monthly_income = income_annual / 12
            dscr = monthly_income / annuity_monthly
            ratios["dscr"] = round(dscr, 2)
        else:
            ratios["dscr"] = None

        # LTV only meaningful if we have goods_price (collateral proxy)
        if credit_total and goods_price and goods_price > 0:
            ltv = credit_total / goods_price
            ratios["ltv"] = round(ltv, 4)
            ratios["ltv_pct"] = f"{ltv * 100:.1f}%"
        else:
            ratios["ltv"] = None
            ratios["ltv_pct"] = "N/A"

        return ratios

    def _compute_debt_assessment(self, financial_ratios: dict, llm_feats: dict, app_row: dict) -> dict:
        """Debt Analyst — score DTI, DSCR, LTV and loan purpose quality.

        Deterministic scoring. No LLM needed — uses ratios already computed.
        Each metric contributes to an overall debt health score (0-100).
        """
        fr = financial_ratios or {}
        feats = llm_feats or {}
        score = 0
        max_score = 100
        metrics = []

        # DTI scoring (40 pts)
        dti = fr.get("dti")
        if dti is not None:
            if dti < 0.30:
                score += 40; dti_status = "Tốt"; dti_flag = "OK"
            elif dti < 0.40:
                score += 30; dti_status = "Chấp nhận được"; dti_flag = "OK"
            elif dti < 0.50:
                score += 15; dti_status = "Cần theo dõi"; dti_flag = "!!"
            else:
                score += 0;  dti_status = "Rủi ro cao"; dti_flag = "!!"
            metrics.append({
                "name": "DTI (Nợ/Thu nhập)",
                "value": fr.get("dti_pct", f"{dti*100:.1f}%"),
                "threshold": "< 40%",
                "status": dti_status,
                "flag": dti_flag,
            })
        else:
            metrics.append({"name": "DTI", "value": "N/A", "threshold": "< 40%", "status": "Không có dữ liệu", "flag": "—"})

        # DSCR scoring (35 pts)
        dscr = fr.get("dscr")
        if dscr is not None:
            if dscr >= 1.5:
                score += 35; dscr_status = "Tốt"; dscr_flag = "OK"
            elif dscr >= 1.2:
                score += 25; dscr_status = "Đạt ngưỡng"; dscr_flag = "OK"
            elif dscr >= 1.0:
                score += 10; dscr_status = "Sát ngưỡng"; dscr_flag = "!!"
            else:
                score += 0;  dscr_status = "Dưới ngưỡng"; dscr_flag = "!!"
            metrics.append({
                "name": "DSCR (Dòng tiền/Nợ)",
                "value": f"{dscr:.2f}",
                "threshold": "> 1.20",
                "status": dscr_status,
                "flag": dscr_flag,
            })
        else:
            metrics.append({"name": "DSCR", "value": "N/A", "threshold": "> 1.20", "status": "Không có dữ liệu", "flag": "—"})

        # LTV scoring (15 pts)
        ltv = fr.get("ltv")
        if ltv is not None:
            if ltv < 0.70:
                score += 15; ltv_status = "Tốt"; ltv_flag = "OK"
            elif ltv < 0.80:
                score += 8;  ltv_status = "Chấp nhận"; ltv_flag = "OK"
            else:
                score += 0;  ltv_status = "Rủi ro"; ltv_flag = "!!"
            metrics.append({
                "name": "LTV (Vay/TSBĐ)",
                "value": fr.get("ltv_pct", f"{ltv*100:.1f}%"),
                "threshold": "< 70%",
                "status": ltv_status,
                "flag": ltv_flag,
            })
        else:
            metrics.append({"name": "LTV", "value": "N/A", "threshold": "< 70%", "status": "Không có dữ liệu", "flag": "—"})

        # Loan purpose (10 pts)
        loan_purpose = feats.get("loan_purpose_category", "UNCLEAR")
        purpose_score_map = {"PRODUCTION": 10, "INVESTMENT": 8, "CONSUMPTION": 6,
                             "REFINANCING": 4, "UNCLEAR": 0}
        purpose_pts = purpose_score_map.get((loan_purpose or "").upper(), 0)
        score += purpose_pts
        purpose_label_map = {"PRODUCTION": "Sản xuất kinh doanh", "INVESTMENT": "Đầu tư",
                             "CONSUMPTION": "Tiêu dùng", "REFINANCING": "Tái cơ cấu nợ", "UNCLEAR": "Chưa rõ"}
        metrics.append({
            "name": "Mục đích vay",
            "value": purpose_label_map.get((loan_purpose or "").upper(), loan_purpose or "N/A"),
            "threshold": "PRODUCTION / INVESTMENT",
            "status": "Rõ ràng" if purpose_pts >= 6 else "Cần làm rõ",
            "flag": "OK" if purpose_pts >= 6 else "!!",
        })

        # Overall status
        pct = score / max_score
        if pct >= 0.70:
            overall = "ĐẠT"; overall_color = "green"
        elif pct >= 0.45:
            overall = "XEM_XET"; overall_color = "orange"
        else:
            overall = "KHONG_DAT"; overall_color = "red"

        return {
            "score": score,
            "max_score": max_score,
            "score_pct": f"{pct*100:.0f}%",
            "overall_status": overall,
            "overall_color": overall_color,
            "metrics": metrics,
            "summary": (
                f"Phân tích nợ tổng hợp: {score}/{max_score} điểm ({pct*100:.0f}%). "
                f"Mục đích vay: {purpose_label_map.get((loan_purpose or '').upper(), 'N/A')}. "
                + (f"DTI {fr.get('dti_pct', 'N/A')} — {'trong ngưỡng an toàn' if dti and dti < 0.40 else 'cần theo dõi'}. " if dti else "")
                + (f"DSCR {dscr:.2f} — {'đạt ngưỡng' if dscr and dscr >= 1.2 else 'sát/dưới ngưỡng tối thiểu 1.2'}." if dscr else "")
            ),
        }

    def _compute_reward_assessment(self, credit_score: int, pd_pct: float, risk_band: str,
                                   financial_ratios: dict, app_row: dict, llm_feats: dict) -> dict:
        """Reward Modeler — estimate risk-adjusted profitability.

        Deterministic. Estimates expected yield, risk-adjusted return, and
        customer lifetime value tier. Based on MASCA Reward Modeler concept.
        """
        fr = financial_ratios or {}
        feats = llm_feats or {}
        app = app_row or {}

        # Interest rate proxy by risk band (Vietnamese market reference rates)
        rate_by_band = {
            "AAA": 0.085, "AA": 0.095, "A": 0.110,
            "BBB": 0.130, "BB": 0.155, "B": 0.180, "CCC": 0.200, "CC": 0.220, "C": 0.240,
        }
        interest_rate = rate_by_band.get((risk_band or "").upper(), 0.120)

        # Loan amount and term
        loan_amount = fr.get("credit_total_vnd") or app.get("AMT_CREDIT") or 0
        term_months = 36
        if app.get("AMT_CREDIT") and app.get("AMT_ANNUITY") and app["AMT_ANNUITY"] > 0:
            term_months = max(6, min(360, round(app["AMT_CREDIT"] / app["AMT_ANNUITY"])))

        # Gross interest income over loan life
        pd_decimal = (pd_pct or 0) / 100
        gross_income = loan_amount * interest_rate * (term_months / 12) if loan_amount else 0

        # Expected loss = PD × LGD (assume LGD = 45% industry standard)
        lgd = 0.45
        expected_loss = loan_amount * pd_decimal * lgd if loan_amount else 0

        # Risk-adjusted return (RAROC proxy)
        risk_adj_income = gross_income - expected_loss
        raroc = (risk_adj_income / loan_amount) if loan_amount > 0 else 0

        # Customer segment & LTV potential
        if credit_score >= 720:
            segment = "Premium"; upsell = ["Bảo hiểm nhân thọ", "Thẻ tín dụng hạng vàng", "Quỹ tiết kiệm"]
        elif credit_score >= 640:
            segment = "Mid-tier"; upsell = ["Bảo hiểm tài sản", "Thẻ tín dụng cơ bản"]
        elif credit_score >= 560:
            segment = "Mass"; upsell = ["Bảo hiểm khoản vay"]
        else:
            segment = "Sub-prime"; upsell = []

        # Profitability verdict
        if raroc >= 0.08:
            verdict = "Tốt"; verdict_flag = "OK"; verdict_color = "green"
        elif raroc >= 0.04:
            verdict = "Chấp nhận được"; verdict_flag = "OK"; verdict_color = "orange"
        elif raroc > 0:
            verdict = "Thấp"; verdict_flag = "!!"; verdict_color = "orange"
        else:
            verdict = "Không khả thi"; verdict_flag = "!!"; verdict_color = "red"

        def _fmt_vnd(v):
            if not v: return "N/A"
            if v >= 1_000_000_000: return f"{v/1_000_000_000:.1f} tỷ VND"
            if v >= 1_000_000:     return f"{v/1_000_000:.0f} triệu VND"
            return f"{v:,.0f} VND"

        return {
            "interest_rate_pct": f"{interest_rate*100:.1f}%",
            "loan_amount_fmt": _fmt_vnd(loan_amount),
            "term_months": term_months,
            "gross_income_fmt": _fmt_vnd(gross_income),
            "expected_loss_fmt": _fmt_vnd(expected_loss),
            "risk_adj_income_fmt": _fmt_vnd(risk_adj_income),
            "raroc_pct": f"{raroc*100:.1f}%",
            "verdict": verdict,
            "verdict_flag": verdict_flag,
            "verdict_color": verdict_color,
            "customer_segment": segment,
            "upsell_opportunities": upsell,
            "summary": (
                f"Lợi nhuận điều chỉnh rủi ro (RAROC): {raroc*100:.1f}% — {verdict}. "
                f"Phân khúc: {segment}. "
                f"Thu nhập lãi ước tính: {_fmt_vnd(gross_income)}, "
                f"Tổn thất kỳ vọng: {_fmt_vnd(expected_loss)}."
            ),
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
        financial_ratios: dict | None = None,
        app_row: dict | None = None,
    ) -> dict[str, Any]:
        """Generate 5C narrative using LLM or mock."""


        five_c_alloc = shap_values.get("five_c_shap_allocation", {})
        financial_info = self._build_financial_context(llm_feats or {}, financial_ratios or {})
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

    def _build_financial_context(self, llm_feats: dict, financial_ratios: dict | None = None) -> str:
        """Build financial context string from available data."""
        lines = []
        fr = financial_ratios or {}

        # Real values from application_row (priority)
        if fr.get("income_monthly_vnd"):
            lines.append(f"Thu nhập tháng: {fr['income_monthly_vnd']:,.0f} VND")
        if fr.get("annuity_monthly_vnd"):
            lines.append(f"Trả nợ tháng: {fr['annuity_monthly_vnd']:,.0f} VND")
        if fr.get("dti_pct") and fr["dti_pct"] != "N/A":
            lines.append(f"DTI: {fr['dti_pct']}")
        if fr.get("dscr"):
            lines.append(f"DSCR: {fr['dscr']:.2f}")
        if fr.get("ltv_pct") and fr["ltv_pct"] != "N/A":
            lines.append(f"LTV: {fr['ltv_pct']}")

        # LLM-derived supplemental
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
        self,
        shap_values: dict,
        credit_score: int,
        llm_feats: dict,
        financial_ratios: dict | None = None,
        app_row: dict | None = None,
    ) -> dict[str, Any]:
        """Generate deterministic mock 5C narrative (6 sections)."""
        fr = financial_ratios or {}
        app = app_row or {}

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

        # Fix 1: correct pos/neg label assignment
        # top_positive_factors = factors that REDUCE default risk (SHAP > 0 toward good)
        # top_negative_factors = factors that INCREASE default risk (SHAP < 0)
        top_pos = shap_values.get("top_positive_factors", [])  # reduces default risk
        top_neg = shap_values.get("top_negative_factors", [])  # increases default risk
        pos_labels = [f.get("label_vi", f.get("feature", "")) for f in top_pos[:3]]  # strengths
        neg_labels = [f.get("label_vi", f.get("feature", "")) for f in top_neg[:3]]  # concerns

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
            # Section IV — Fix 2: real DTI/DSCR from application_row
            "financial_summary": {
                "income_analysis": (
                    "Thu nhập ổn định qua xác minh sao kê ngân hàng. "
                    + (f"Thu nhập tháng: {fr['income_monthly_vnd']:,.0f} VND. " if fr.get("income_monthly_vnd") else "")
                    + ("Chỉ số ổn định ở mức tốt." if credit_score >= 600 else "Cần xem xét thêm.")
                ),
                "debt_analysis": (
                    f"Tỷ lệ nợ/thu nhập (DTI): {fr.get('dti_pct', 'N/A')}. "
                    + (f"DSCR: {fr['dscr']:.2f} {'(đạt ngưỡng)' if fr['dscr'] >= 1.2 else '(dưới ngưỡng 1.2 — cần lưu ý)'}. " if fr.get("dscr") else "")
                    + (f"Trả nợ hàng tháng: {fr['annuity_monthly_vnd']:,.0f} VND." if fr.get("annuity_monthly_vnd") else "")
                ),
                "key_ratios": {
                    "dti": fr.get("dti_pct", "N/A"),
                    "dscr": str(fr.get("dscr", "N/A")),
                    "ltv": fr.get("ltv_pct", "N/A (chưa có thông tin TSBĐ chi tiết)"),
                },
            },
            # Section VI — Fix 3: terms from real AMT_CREDIT
            "recommendation": recommendation,
            "suggested_terms": {
                "requested_amount_vnd": app.get("AMT_CREDIT"),
                "max_amount_vnd": (
                    app.get("AMT_CREDIT")
                    or (300_000_000 if credit_score >= 650 else 100_000_000)
                ),
                "requested_term_months": (
                    round(app["AMT_CREDIT"] / app["AMT_ANNUITY"])
                    if app.get("AMT_CREDIT") and app.get("AMT_ANNUITY")
                    else (36 if credit_score >= 650 else 12)
                ),
                "interest_rate_suggestion": "Theo biểu phí hiện hành",
                "conditions": (
                    ["Chứng minh thu nhập bổ sung (slip lương 3 tháng)"]
                    if credit_score < 700 else []
                ),
                "dti_at_approval": fr.get("dti_pct", "N/A"),
            },
            "caveats": [],
        }
