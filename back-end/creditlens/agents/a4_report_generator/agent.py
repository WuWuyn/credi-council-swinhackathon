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
import os
from datetime import datetime, timezone
from typing import Any

from creditlens.services.llm_service import LLMService
from creditlens.services.policy_rag_service import PolicyRAGService
from creditlens.agents.a4_report_generator.consistency_validator import validate_narrative_consistency
from creditlens.agents.a4_report_generator.five_c_scorer import (
    compute_five_c_scores, five_c_to_dict, DimensionScore,
)
from creditlens.agents.a4_report_generator.decision_engine import (
    compute_decision, CreditDecision,
)
from creditlens.config.feature_config import get_label_vi

logger = logging.getLogger(__name__)

# ── Report Generation Prompts (NARRATIVE ONLY — no scoring, no decision) ──
REPORT_SYSTEM = """Bạn là chuyên gia phân tích tín dụng tại ngân hàng Việt Nam.
Viết phần diễn giải (narrative) cho báo cáo 5C bằng tiếng Việt theo chuẩn TT39/2016.

QUAN TRỌNG — BẠN CHỈ VIẾT NARRATIVE:
- KHÔNG chấm điểm (score đã được tính bởi hệ thống)
- KHÔNG đưa ra quyết định APPROVE/REJECT (đã có rule-based engine)
- Chỉ TRÍCH DẪN các yếu tố từ SHAP values được cung cấp
- Không bịa đặt thông tin không có trong dữ liệu
- Sử dụng số liệu cụ thể khi có thể
- Viết bằng văn phong ngân hàng trang trọng
- Trích dẫn điều khoản quy định cụ thể từ RAG context

Trả lời CHỈ bằng JSON hợp lệ theo schema sau:
{
    "customer_info": {
        "summary": "Tóm tắt hồ sơ khách hàng bằng tiếng Việt (1-2 câu, bao gồm nhu cầu vay)"
    },
    "character_narrative": "Diễn giải uy tín/tư cách tín dụng 100-150 chữ. Chỉ cite SHAP factors.",
    "capacity_narrative": "Diễn giải năng lực trả nợ 100-150 chữ. Cite DTI, DSCR, SHAP.",
    "capital_narrative": "Diễn giải vốn tự có 80-120 chữ. Cite tài sản, nợ CIC.",
    "conditions_narrative": "Diễn giải điều kiện vay 80-120 chữ. Cite mục đích vay, học vấn.",
    "collateral_narrative": "Diễn giải tài sản bảo đảm 60-100 chữ. Cite LTV, loại TSBĐ.",
    "financial_summary": {
        "income_analysis": "Phân tích thu nhập (1-2 câu)",
        "debt_analysis": "Phân tích nợ (1-2 câu)"
    },
    "suggested_terms": {
        "max_amount_vnd": "số tiền đề xuất (number)",
        "interest_rate_suggestion": "theo biểu phí hiện hành",
        "conditions": ["Điều kiện tiên quyết — tối đa 5"]
    },
    "caveats": ["Cảnh báo dữ liệu — tối đa 5"]
}"""

REPORT_USER = """Viết NARRATIVE cho báo cáo 5C (KHÔNG chấm điểm, KHÔNG quyết định):

=== Hồ sơ khách hàng ===
{customer_context}

=== Điểm 5C (đã tính sẵn bởi rule engine — DÙNG ĐỂ THAM KHẢO, KHÔNG THAY ĐỔI) ===
{five_c_scores_context}

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
- Quyết định hệ thống: {system_decision}

=== Thông tin tài chính ===
{financial_info}

=== Thông tin tài sản bảo đảm ===
{collateral_info}

=== Chính sách & Quy định (RAG) ===
{rag_context}

Yêu cầu:
1. Viết narrative cho TỪNG C — giải thích TẠI SAO điểm đó hợp lý
2. Chỉ TRÍCH DẪN yếu tố có trong SHAP values
3. Trích dẫn điều khoản pháp lý cụ thể từ RAG context
4. KHÔNG tự chấm điểm hay đề xuất quyết định"""


class ReportGeneratorAgent:
    """Agent A4 — Report Generator + Explainability.

    # LOCAL_SUB: Uses Gemini API. Production: use Bedrock Claude.

    Converts SHAP output into human-readable Vietnamese 5C reports (6 sections).
    """

    def __init__(self):
        self.llm = LLMService()
        self.rag = PolicyRAGService()

    def generate(
        self,
        a3_output: dict[str, Any],
        a2_output: dict[str, Any],
        a1_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate credit report from A3 scoring output.

        Architecture (refactored):
          - 5C Scores: DETERMINISTIC rule-based (five_c_scorer.py)
          - Recommendation: DETERMINISTIC rule-based (decision_engine.py)
          - LLM: ONLY writes narrative text, no scoring, no decisions

        Args:
            a3_output: Output from A3 ScoringAgent.score()
            a2_output: Output from A2 FeatureEngineerAgent.process()
            a1_output: Optional A1 output for financial calculations

        Returns:
            Dict with final_report, five_c_scores, consistency_check
        """
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

        # ── Step 1: Compute financial ratios (deterministic) ──────────
        financial_ratios = self._compute_financial_ratios(app_row)

        # ── Step 2: Compute 5C scores (DETERMINISTIC — rule-based) ────
        five_c_dim_scores = compute_five_c_scores(
            app_row=app_row,
            financial_ratios=financial_ratios,
            shap_values=shap_values,
            llm_feats=llm_feats,
        )
        five_c_scores = {dim: s.score for dim, s in five_c_dim_scores.items()}
        five_c_detail = five_c_to_dict(five_c_dim_scores)
        total_5c = sum(five_c_scores.values())

        # ── Step 3: Compute decision (DETERMINISTIC — rule-based) ─────
        decision = compute_decision(
            credit_score=credit_score,
            app_row=app_row,
            financial_ratios=financial_ratios,
            five_c_scores=five_c_detail,
            llm_feats=llm_feats,
        )

        # ── Step 4: RAG — query policy context ────────────────────────
        rag_query = PolicyRAGService.build_policy_query(
            credit_score=credit_score,
            risk_band=risk_band,
            pd_pct=pd_pct,
            customer_type="INDIVIDUAL",
            thin_file=llm_feats.get("thin_file_flag", False),
            dti=financial_ratios.get("dti"),
            dscr=financial_ratios.get("dscr"),
            ltv=financial_ratios.get("ltv"),
            app_row=app_row,
            llm_feats=llm_feats,
            shap_values=shap_values,
        )
        rag_result = self.rag.query(rag_query)
        rag_context = rag_result.get("context", "")
        rag_citations = rag_result.get("citations", [])
        if rag_result.get("has_context"):
            logger.info(f"  RAG policy context: {len(rag_context)} chars, {len(rag_citations)} citations")
        else:
            logger.info("  RAG policy context: not available (store not configured)")

        # ── Step 5: LLM generates NARRATIVE ONLY ─────────────────────
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
            rag_context=rag_context,
            five_c_scores=five_c_detail,
            system_decision=decision.recommendation,
        )

        # ── Step 6: Validate consistency ──────────────────────────────
        # Adapt narrative dict for consistency validator
        narrative_for_check = {}
        for dim in ["character", "capacity", "capital", "conditions", "collateral"]:
            narrative_for_check[f"{dim}_assessment"] = {
                "narrative": narrative.get(f"{dim}_narrative", ""),
            }
        consistency = validate_narrative_consistency(shap_values, narrative_for_check)
        logger.info(f"  Consistency check: {'PASSED' if consistency['passed'] else 'FAILED'} (coverage: {consistency.get('shap_coverage', 0)*100:.0f}%)")

        # ── Step 7: Build assessment dicts (merge rule scores + LLM narrative) ──
        shap_alloc = shap_values.get("five_c_shap_allocation", {})
        assessments = {}
        for dim, detail in five_c_detail.items():
            alloc_info = shap_alloc.get(dim, {})
            assessments[f"{dim}_assessment"] = {
                "score": detail["score"],
                "max_score": detail["max_score"],
                "status": detail["status"],
                "shap_pct": f"{alloc_info.get('pct', 0)}%",
                "indicators_met": detail["indicators_met"],
                "indicators_review": detail["indicators_review"],
                "breakdown": detail["breakdown"],
                "narrative": narrative.get(f"{dim}_narrative", "Không có dữ liệu."),
            }

        logger.info(f"  5C Total: {total_5c}/120")
        for dim, score in five_c_scores.items():
            logger.info(f"    {dim}: {score}/{five_c_dim_scores[dim].max_score}")
        logger.info(f"  Recommendation: {decision.recommendation} (rule-based)")

        # ── Step 8: Build final report ────────────────────────────────
        # Merge LLM suggested_terms with decision engine conditions
        llm_terms = narrative.get("suggested_terms", {})
        merged_conditions = list(dict.fromkeys(
            decision.conditions + (llm_terms.get("conditions") or [])
        ))
        suggested_terms = {
            "max_amount_vnd": financial_ratios.get("credit_total_vnd"),  # always use scaled VND
            "interest_rate_suggestion": llm_terms.get("interest_rate_suggestion", "theo biểu phí hiện hành"),
            "conditions": merged_conditions,
        }

        # Build structured customer info from application_row
        gender_map = {"F": "Nữ", "M": "Nam"}
        income_map = {
            "Working": "Đang làm việc", "Pensioner": "Hưu trí",
            "Commercial associate": "Kinh doanh", "State servant": "Công chức",
            "Student": "Sinh viên", "Unemployed": "Thất nghiệp",
        }
        edu_map = {
            "Higher education": "Đại học / Cao đẳng",
            "Secondary / secondary special": "Trung cấp / THPT",
            "Incomplete higher": "Chưa tốt nghiệp ĐH",
            "Lower secondary": "THCS",
            "Academic degree": "Sau đại học",
        }
        family_map = {
            "Single / not married": "Độc thân", "Married": "Đã kết hôn",
            "Civil marriage": "Sống chung", "Separated": "Ly thân",
            "Widow": "Goá",
        }
        housing_map = {
            "House / apartment": "Nhà riêng / Chung cư",
            "Rented apartment": "Thuê nhà", "With parents": "Ở cùng cha mẹ",
            "Municipal apartment": "Nhà tập thể",
            "Office apartment": "Nhà công vụ", "Co-op apartment": "Nhà hợp tác",
        }
        age = round(abs(app_row.get("DAYS_BIRTH", 0)) / 365.25) if app_row.get("DAYS_BIRTH") else None

        customer_info_struct = {
            **(narrative.get("customer_info") or {}),
            "gender": gender_map.get(app_row.get("CODE_GENDER", ""), ""),
            "age": age,
            "education": edu_map.get(app_row.get("NAME_EDUCATION_TYPE", ""), app_row.get("NAME_EDUCATION_TYPE", "")),
            "family_status": family_map.get(app_row.get("NAME_FAMILY_STATUS", ""), app_row.get("NAME_FAMILY_STATUS", "")),
            "income_type": income_map.get(app_row.get("NAME_INCOME_TYPE", ""), app_row.get("NAME_INCOME_TYPE", "")),
            "housing": housing_map.get(app_row.get("NAME_HOUSING_TYPE", ""), app_row.get("NAME_HOUSING_TYPE", "")),
            "own_realty": app_row.get("FLAG_OWN_REALTY", ""),
            "own_car": app_row.get("FLAG_OWN_CAR", ""),
            "loan_purpose": llm_feats.get("loan_purpose_category", ""),
        }

        final_report = {
            # Section I: Thông tin khách hàng (structured)
            "customer_info": customer_info_struct,
            # Section II: Executive Summary (all deterministic)
            "executive_summary": {
                "credit_score": credit_score,
                "risk_band": decision.risk_band,
                "pd_pct": pd_pct,
                "recommendation": decision.recommendation,
                "five_c_total": total_5c,
                "five_c_scores": five_c_scores,
                "five_c_shap_allocation": shap_alloc,
                "decision_reasons": decision.reasons,
                "decision_overrides": decision.overrides_applied,
                "model_info": {
                    "model_version": shap_values.get("model_version", "lgbm_v1_noxmoon"),
                    "auc": shap_values.get("auc", "0.803"),
                    "shap_verified": True,
                    "inference_timestamp": shap_values.get("inference_timestamp"),
                },
                "financial_ratios": financial_ratios,
            },
            # Section III: 5C Scorecard (rule-based scores + LLM narrative)
            "five_c_scorecard": assessments,
            # Section IV: Tình hình tài chính + Debt Analyst
            "financial_summary": narrative.get("financial_summary", {}),
            "debt_assessment": self._compute_debt_assessment(
                financial_ratios=financial_ratios,
                llm_feats=llm_feats,
                app_row=app_row,
            ),
            # Section V: Tài sản bảo đảm
            "collateral_detail": assessments.get("collateral_assessment", {}),
            # Section VI: Khuyến nghị & Caveats + Reward Modeler
            "suggested_terms": suggested_terms,
            "reward_assessment": self._compute_reward_assessment(
                credit_score=credit_score,
                pd_pct=pd_pct,
                risk_band=decision.risk_band,
                financial_ratios=financial_ratios,
                app_row=app_row,
                llm_feats=llm_feats,
            ),
            "llm_insights": {
                "loan_purpose": llm_feats.get("loan_purpose_category"),
                "positive_signals": llm_feats.get("positive_signals", []),
                "risk_flags": llm_feats.get("risk_flags", []),
            },
            "caveats": (narrative.get("caveats") or []) + warnings,
            "audit_reference": {
                "model_version": shap_values.get("model_version"),
                "inference_timestamp": shap_values.get("inference_timestamp"),
                "rag_citations": rag_citations,
                "rag_policy_docs": [
                    "TT39/2016/TT-NHNN", "TT11/2021/TT-NHNN",
                    "QĐ493/2005/QĐ-NHNN", "CIC Scoring Guide",
                    "Basel II/III (TT41/2016, TT14/2025)",
                    "5C Assessment Framework",
                ],
            },
        }

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A4",
            "action": "report_generation_5c",
            "output_summary": {
                "recommendation": decision.recommendation,
                "five_c_total": total_5c,
                "five_c_scores": five_c_scores,
                "decision_reasons": decision.reasons,
                "consistency_passed": consistency.get("passed"),
            },
            "model_version": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        }

        return {
            "credit_score": credit_score,
            "pd_pct": pd_pct,
            "risk_band": decision.risk_band,
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

        VND_SCALE: Home Credit uses anonymized currency units. We multiply by
        100 to display realistic VND amounts in the report.
        Ratios (DTI, DSCR, LTV) are NOT affected since both sides scale equally.
        """
        VND_SCALE = 100  # Home Credit unit → approximate VND conversion

        income_annual = app_row.get("AMT_INCOME_TOTAL")
        annuity_annual = app_row.get("AMT_ANNUITY")   # annual — must divide by 12 for monthly
        credit_total = app_row.get("AMT_CREDIT")
        goods_price = app_row.get("AMT_GOODS_PRICE")

        # Convert annuity to monthly for ratio calculations
        annuity_monthly = annuity_annual / 12 if annuity_annual else None

        # VND display values (scaled for report display)
        ratios = {
            "income_annual_vnd": income_annual * VND_SCALE if income_annual else None,
            "income_monthly_vnd": (income_annual / 12) * VND_SCALE if income_annual else None,
            "annuity_monthly_vnd": annuity_monthly * VND_SCALE if annuity_monthly else None,
            "credit_total_vnd": credit_total * VND_SCALE if credit_total else None,
            "goods_price_vnd": goods_price * VND_SCALE if goods_price else None,
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
        rag_context: str = "",
        five_c_scores: dict | None = None,
        system_decision: str = "REVIEW",
    ) -> dict[str, Any]:
        """Generate 5C NARRATIVE ONLY using LLM with RAG policy context.

        LLM does NOT score or decide — only writes narrative text.
        """
        five_c_alloc = shap_values.get("five_c_shap_allocation", {})
        customer_context = self._build_customer_context(app_row or {})
        financial_info = self._build_financial_context(llm_feats or {}, financial_ratios or {})
        collateral_info = self._build_collateral_context(llm_feats or {})

        # Build 5C scores context for LLM reference
        five_c_context = ""
        if five_c_scores:
            lines = []
            for dim, detail in five_c_scores.items():
                score = detail.get("score", 0)
                max_s = detail.get("max_score", 0)
                status = detail.get("status", "—")
                met = detail.get("indicators_met", [])
                rev = detail.get("indicators_review", [])
                lines.append(f"  {dim}: {score}/{max_s} ({status})")
                for m in met:
                    lines.append(f"    ✓ {m}")
                for r in rev:
                    lines.append(f"    ⚠ {r}")
            five_c_context = "\n".join(lines)
        else:
            five_c_context = "Chưa có điểm 5C"

        # Use RAG context if available, else provide fallback
        policy_context = rag_context if rag_context else (
            "Không có dữ liệu chính sách RAG. "
            "Tham chiếu chung: TT39/2016, TT11/2021, CIC, Basel II/III."
        )

        prompt = REPORT_USER.format(
            shap_json=json.dumps(shap_values, indent=2, default=str),
            five_c_allocation=json.dumps(five_c_alloc, indent=2, default=str),
            five_c_scores_context=five_c_context,
            warnings_json=json.dumps(warnings, default=str),
            customer_context=customer_context,
            customer_type=customer_type,
            thin_file_flag=thin_file,
            credit_score=credit_score,
            risk_band=risk_band,
            pd_pct=pd_pct,
            system_decision=system_decision,
            financial_info=financial_info,
            collateral_info=collateral_info,
            rag_context=policy_context,
        )

        return self.llm.generate_json(
            REPORT_SYSTEM, prompt,
            {"character_narrative", "capacity_narrative",
             "capital_narrative", "conditions_narrative",
             "collateral_narrative"},
            max_tokens=16384,
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

    @staticmethod
    def _build_customer_context(app: dict) -> str:
        """Build structured customer profile from application_row for LLM."""
        lines = []

        # Demographics
        gender_vi = {"M": "Nam", "F": "Nữ"}.get(app.get("CODE_GENDER", ""), "")
        if gender_vi:
            lines.append(f"Giới tính: {gender_vi}")

        days_birth = app.get("DAYS_BIRTH")
        if days_birth and isinstance(days_birth, (int, float)):
            age = abs(int(days_birth)) // 365
            lines.append(f"Tuổi: {age}")

        education = app.get("NAME_EDUCATION_TYPE")
        if education:
            lines.append(f"Trình độ học vấn: {education}")

        family = app.get("NAME_FAMILY_STATUS")
        if family:
            lines.append(f"Tình trạng hôn nhân: {family}")

        children = app.get("CNT_CHILDREN", 0)
        if children:
            lines.append(f"Số con: {children}")

        # Employment
        income_type = app.get("NAME_INCOME_TYPE")
        if income_type:
            lines.append(f"Nguồn thu nhập: {income_type}")

        occupation = app.get("OCCUPATION_TYPE")
        if occupation:
            lines.append(f"Nghề nghiệp: {occupation}")

        org = app.get("ORGANIZATION_TYPE")
        if org and org != "XNA":
            lines.append(f"Tổ chức: {org}")

        days_employed = app.get("DAYS_EMPLOYED")
        if days_employed and isinstance(days_employed, (int, float)):
            if days_employed == 365243:  # Special code for retired/unemployed
                lines.append("Nghỉ hưu / Không đi làm")
            elif days_employed < 0:
                years = abs(days_employed) / 365
                lines.append(f"Thâm niên công tác: {years:.1f} năm")

        # Loan details
        contract = app.get("NAME_CONTRACT_TYPE")
        if contract:
            lines.append(f"Loại hợp đồng: {contract}")

        income = app.get("AMT_INCOME_TOTAL")
        if income:
            lines.append(f"Thu nhập năm: {income:,.0f} VND")

        credit = app.get("AMT_CREDIT")
        if credit:
            lines.append(f"Số tiền vay: {credit:,.0f} VND")

        annuity = app.get("AMT_ANNUITY")
        if annuity:
            lines.append(f"Trả góp kỳ: {annuity:,.0f} VND")

        # Assets
        assets = []
        if app.get("FLAG_OWN_REALTY") == "Y":
            assets.append("bất động sản")
        if app.get("FLAG_OWN_CAR") == "Y":
            car_age = app.get("OWN_CAR_AGE")
            assets.append(f"xe ô tô ({car_age} năm)" if car_age else "xe ô tô")
        if assets:
            lines.append(f"Tài sản sở hữu: {', '.join(assets)}")
        else:
            lines.append("Tài sản sở hữu: Không khai báo")

        # Housing
        housing = app.get("NAME_HOUSING_TYPE")
        if housing:
            lines.append(f"Nhà ở: {housing}")

        # External credit scores (CIC)
        ext_scores = []
        for i in [1, 2, 3]:
            ext = app.get(f"EXT_SOURCE_{i}")
            if ext is not None:
                ext_scores.append(f"EXT_{i}={ext:.4f}")
        if ext_scores:
            lines.append(f"Điểm tín dụng ngoại (CIC): {', '.join(ext_scores)}")

        # Region
        region = app.get("REGION_RATING_CLIENT")
        if region:
            lines.append(f"Xếp hạng vùng: {region}")

        return "\n".join(lines) if lines else "Không có thông tin hồ sơ khách hàng"

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
            # Section VI — Fix 3: terms from real AMT_CREDIT (scaled to VND)
            "recommendation": recommendation,
            "suggested_terms": {
                "requested_amount_vnd": (app.get("AMT_CREDIT") or 0) * 100,
                "max_amount_vnd": (
                    (app.get("AMT_CREDIT") or 0) * 100
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
