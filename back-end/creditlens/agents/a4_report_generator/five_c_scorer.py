"""
CreditLens A4 — Deterministic 5C Scorer.

Rule-based scoring for the 5C credit assessment framework:
  C1 Character  (0-30): Credit history, payment behaviour, identity
  C2 Capacity   (0-40): Income, DTI, DSCR, employment stability
  C3 Capital    (0-20): Net worth, assets, bureau debt
  C4 Conditions (0-10): Loan purpose, education, regional factors
  C5 Collateral (0-20): LTV, asset type, ownership

Design principle: Same input → Same output. No LLM involved.
SHAP allocation is used for DISPLAY only, not for scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Score Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single 5C dimension."""
    score: int
    max_score: int
    status: str        # ĐẠT | XEM_XET | CHƯA_ĐẠT
    breakdown: list[dict] = field(default_factory=list)  # [{criterion, points, max, detail}]
    indicators_met: list[str] = field(default_factory=list)
    indicators_review: list[str] = field(default_factory=list)

    @property
    def pct(self) -> float:
        return self.score / self.max_score if self.max_score else 0


def _status_from_pct(pct: float) -> str:
    """Derive status from score percentage."""
    if pct >= 0.70:
        return "ĐẠT"
    if pct >= 0.40:
        return "XEM_XÉT"
    return "CHƯA_ĐẠT"


# ─────────────────────────────────────────────────────────────────────────────
# Helper extractors
# ─────────────────────────────────────────────────────────────────────────────

def _ext_source_avg(app: dict) -> float | None:
    """Average of available EXT_SOURCE scores (CIC proxy)."""
    vals = []
    for i in (1, 2, 3):
        v = app.get(f"EXT_SOURCE_{i}")
        if v is not None and isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def _age_years(app: dict) -> int | None:
    days = app.get("DAYS_BIRTH")
    if days and isinstance(days, (int, float)):
        return abs(int(days)) // 365
    return None


def _employment_years(app: dict) -> float | None:
    days = app.get("DAYS_EMPLOYED")
    if days and isinstance(days, (int, float)):
        if days == 365243:  # retired / unemployed sentinel
            return None
        if days < 0:
            return abs(days) / 365.0
    return None


def _is_pensioner(app: dict) -> bool:
    return (app.get("NAME_INCOME_TYPE") or "").lower() == "pensioner"


def _has_dpd(shap_factors: list[dict]) -> bool:
    """Check if any SHAP factor indicates DPD (days past due) issues."""
    for f in shap_factors:
        feat = (f.get("feature") or "").lower()
        if "dpd" in feat and float(f.get("shap_value", f.get("shap", 0)) or 0) > 0.02:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Character (0-30)
# ─────────────────────────────────────────────────────────────────────────────

def score_character(
    app: dict, shap: dict, llm_feats: dict, **_
) -> DimensionScore:
    """
    C1 Character — Uy tín / Tư cách (max 30).

    Criteria:
      - CIC External Scores (EXT_SOURCE avg)     : 0-10 pts
      - Payment history (no DPD)                  : 0-7 pts
      - Identity & documentation consistency      : 0-5 pts
      - Previous loan history                     : 0-5 pts
      - Social default circle                     : 0-3 pts
    """
    pts = 0
    breakdown = []
    met = []
    review = []

    # 1. CIC scores (10 pts)
    ext_avg = _ext_source_avg(app)
    if ext_avg is not None:
        if ext_avg >= 0.80:
            p = 10; met.append(f"Điểm CIC ngoại tốt ({ext_avg:.2f})")
        elif ext_avg >= 0.65:
            p = 7; met.append(f"Điểm CIC ngoại khá ({ext_avg:.2f})")
        elif ext_avg >= 0.50:
            p = 5; review.append(f"Điểm CIC ngoại trung bình ({ext_avg:.2f})")
        elif ext_avg >= 0.30:
            p = 3; review.append(f"Điểm CIC ngoại thấp ({ext_avg:.2f})")
        else:
            p = 1; review.append(f"Điểm CIC ngoại rất thấp ({ext_avg:.2f})")
        pts += p
        breakdown.append({"criterion": "CIC External Score", "points": p, "max": 10,
                          "detail": f"EXT_SOURCE avg = {ext_avg:.4f}"})
    else:
        # Thin file — no CIC
        p = 2
        pts += p
        review.append("Thin-file: không có điểm CIC ngoại")
        breakdown.append({"criterion": "CIC External Score", "points": p, "max": 10,
                          "detail": "Thin file — no EXT_SOURCE"})

    # 2. Payment history — DPD (7 pts)
    all_factors = shap.get("top_positive_factors", []) + shap.get("top_negative_factors", [])
    has_dpd_issue = _has_dpd(all_factors)
    social_def = app.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) or 0

    if not has_dpd_issue:
        p = 7; met.append("Không ghi nhận nợ quá hạn (DPD)")
    else:
        p = 2; review.append("Ghi nhận chỉ số DPD trong SHAP — cần kiểm tra")
    pts += p
    breakdown.append({"criterion": "Lịch sử thanh toán (DPD)", "points": p, "max": 7,
                      "detail": f"DPD issue: {has_dpd_issue}"})

    # 3. Identity & documentation (5 pts)
    id_pts = 0
    if app.get("FLAG_DOCUMENT_3") == 1 or app.get("FLAG_DOCUMENT_6") == 1:
        id_pts += 2
    # Days since ID publish — longer = more stable identity
    days_id = app.get("DAYS_ID_PUBLISH")
    if days_id and isinstance(days_id, (int, float)) and abs(days_id) > 1000:
        id_pts += 2; met.append("Giấy tờ định danh ổn định")
    else:
        id_pts += 1
    # Phone available
    if app.get("FLAG_CONT_MOBILE") == 1 or app.get("FLAG_MOBIL") == 1:
        id_pts += 1
    id_pts = min(id_pts, 5)
    pts += id_pts
    breakdown.append({"criterion": "Định danh & hồ sơ", "points": id_pts, "max": 5,
                      "detail": "Document flags + ID publish stability"})

    # 4. Previous loan history (5 pts)
    bureau_req_year = app.get("AMT_REQ_CREDIT_BUREAU_YEAR", 0) or 0
    bureau_req_mon = app.get("AMT_REQ_CREDIT_BUREAU_MON", 0) or 0
    if bureau_req_year <= 2 and bureau_req_mon <= 1:
        p = 5; met.append("Ít yêu cầu tín dụng gần đây")
    elif bureau_req_year <= 5:
        p = 3; review.append(f"Có {int(bureau_req_year)} yêu cầu CIC trong năm")
    else:
        p = 1; review.append(f"Nhiều yêu cầu tín dụng ({int(bureau_req_year)}/năm)")
    pts += p
    breakdown.append({"criterion": "Lịch sử tín dụng trước", "points": p, "max": 5,
                      "detail": f"Bureau requests: {bureau_req_year}/year, {bureau_req_mon}/month"})

    # 5. Social circle defaults (3 pts)
    if social_def == 0:
        p = 3; met.append("Vòng xã hội không có nợ xấu")
    elif social_def <= 1:
        p = 2
    else:
        p = 0; review.append(f"Vòng xã hội có {int(social_def)} case nợ xấu")
    pts += p
    breakdown.append({"criterion": "Vòng xã hội (social circle)", "points": p, "max": 3,
                      "detail": f"DEF_30_CNT_SOCIAL_CIRCLE = {social_def}"})

    pts = min(pts, 30)
    status = _status_from_pct(pts / 30)
    return DimensionScore(pts, 30, status, breakdown, met, review)


# ─────────────────────────────────────────────────────────────────────────────
# C2 — Capacity (0-40)
# ─────────────────────────────────────────────────────────────────────────────

def score_capacity(
    app: dict, financial_ratios: dict, llm_feats: dict, **_
) -> DimensionScore:
    """
    C2 Capacity — Năng lực trả nợ (max 40).

    Criteria:
      - DTI ratio                    : 0-14 pts
      - DSCR                         : 0-10 pts
      - Income type & stability      : 0-8 pts
      - Employment duration          : 0-8 pts
    """
    fr = financial_ratios or {}
    pts = 0
    breakdown = []
    met = []
    review = []

    # 1. DTI (14 pts)
    dti = fr.get("dti")
    if dti is not None:
        if dti < 0.25:
            p = 14; met.append(f"DTI {dti*100:.1f}% — rất tốt")
        elif dti < 0.35:
            p = 11; met.append(f"DTI {dti*100:.1f}% — tốt")
        elif dti < 0.40:
            p = 8; met.append(f"DTI {dti*100:.1f}% — chấp nhận được")
        elif dti < 0.50:
            p = 5; review.append(f"DTI {dti*100:.1f}% — gần ngưỡng cao")
        else:
            p = 2; review.append(f"DTI {dti*100:.1f}% — vượt ngưỡng 50%")
        pts += p
        breakdown.append({"criterion": "DTI (Nợ/Thu nhập)", "points": p, "max": 14,
                          "detail": f"DTI = {dti*100:.1f}%"})
    else:
        review.append("Không tính được DTI")
        breakdown.append({"criterion": "DTI", "points": 0, "max": 14, "detail": "N/A"})

    # 2. DSCR (10 pts)
    dscr = fr.get("dscr")
    if dscr is not None:
        if dscr >= 3.0:
            p = 10; met.append(f"DSCR {dscr:.2f} — dư dả")
        elif dscr >= 2.0:
            p = 8; met.append(f"DSCR {dscr:.2f} — tốt")
        elif dscr >= 1.5:
            p = 6; met.append(f"DSCR {dscr:.2f} — khá")
        elif dscr >= 1.2:
            p = 4; review.append(f"DSCR {dscr:.2f} — sát ngưỡng")
        else:
            p = 1; review.append(f"DSCR {dscr:.2f} — dưới ngưỡng an toàn")
        pts += p
        breakdown.append({"criterion": "DSCR (Dòng tiền/Nợ)", "points": p, "max": 10,
                          "detail": f"DSCR = {dscr:.2f}"})
    else:
        breakdown.append({"criterion": "DSCR", "points": 0, "max": 10, "detail": "N/A"})

    # 3. Income type & stability (8 pts)
    income_type = app.get("NAME_INCOME_TYPE", "")
    income_score_map = {
        "Working": 8, "Commercial associate": 7, "State servant": 8,
        "Pensioner": 5, "Student": 2, "Unemployed": 1, "Businessman": 6,
        "Maternity leave": 3,
    }
    p = income_score_map.get(income_type, 4)
    if income_type == "Pensioner":
        review.append("Thu nhập hưu trí — cần xác minh tính bền vững")
    elif p >= 7:
        met.append(f"Nguồn thu nhập ổn định ({income_type})")
    pts += p
    breakdown.append({"criterion": "Loại thu nhập", "points": p, "max": 8,
                      "detail": f"NAME_INCOME_TYPE = {income_type}"})

    # 4. Employment duration (8 pts)
    emp_years = _employment_years(app)
    if emp_years is not None:
        if emp_years >= 5:
            p = 8; met.append(f"Thâm niên công tác {emp_years:.1f} năm")
        elif emp_years >= 3:
            p = 6; met.append(f"Thâm niên {emp_years:.1f} năm")
        elif emp_years >= 1:
            p = 4
        else:
            p = 2; review.append(f"Thâm niên ngắn ({emp_years:.1f} năm)")
        pts += p
        breakdown.append({"criterion": "Thâm niên công tác", "points": p, "max": 8,
                          "detail": f"{emp_years:.1f} years"})
    elif _is_pensioner(app):
        p = 4  # pensioner — no employment but has income
        pts += p
        breakdown.append({"criterion": "Thâm niên công tác", "points": p, "max": 8,
                          "detail": "Pensioner — retired"})
    else:
        p = 2
        pts += p
        review.append("Không xác định được thâm niên công tác")
        breakdown.append({"criterion": "Thâm niên công tác", "points": p, "max": 8,
                          "detail": "Unknown"})

    pts = min(pts, 40)
    status = _status_from_pct(pts / 40)
    return DimensionScore(pts, 40, status, breakdown, met, review)


# ─────────────────────────────────────────────────────────────────────────────
# C3 — Capital (0-20)
# ─────────────────────────────────────────────────────────────────────────────

def score_capital(
    app: dict, financial_ratios: dict, **_
) -> DimensionScore:
    """
    C3 Capital — Vốn tự có (max 20).

    Criteria:
      - Owns real estate                : 0-7 pts
      - Owns car                        : 0-3 pts
      - Credit/Income ratio (leverage)  : 0-6 pts
      - Bureau debt exposure            : 0-4 pts
    """
    fr = financial_ratios or {}
    pts = 0
    breakdown = []
    met = []
    review = []

    # 1. Real estate (7 pts)
    if app.get("FLAG_OWN_REALTY") == "Y":
        p = 7; met.append("Sở hữu bất động sản")
    else:
        p = 1; review.append("Không sở hữu bất động sản")
    pts += p
    breakdown.append({"criterion": "Bất động sản", "points": p, "max": 7,
                      "detail": f"FLAG_OWN_REALTY = {app.get('FLAG_OWN_REALTY')}"})

    # 2. Car (3 pts)
    if app.get("FLAG_OWN_CAR") == "Y":
        p = 3; met.append("Sở hữu ô tô")
    else:
        p = 0
    pts += p
    breakdown.append({"criterion": "Ô tô", "points": p, "max": 3,
                      "detail": f"FLAG_OWN_CAR = {app.get('FLAG_OWN_CAR')}"})

    # 3. Credit/Income ratio — leverage (6 pts)
    income = fr.get("income_annual_vnd")
    credit = fr.get("credit_total_vnd")
    if income and credit and income > 0:
        ratio = credit / income
        if ratio < 2:
            p = 6; met.append(f"Đòn bẩy tài chính thấp (vay/thu nhập = {ratio:.1f}x)")
        elif ratio < 4:
            p = 4
        elif ratio < 6:
            p = 2; review.append(f"Đòn bẩy tài chính cao ({ratio:.1f}x thu nhập)")
        else:
            p = 0; review.append(f"Đòn bẩy tài chính rất cao ({ratio:.1f}x thu nhập)")
    else:
        p = 2  # cannot compute — neutral
    pts += p
    breakdown.append({"criterion": "Đòn bẩy (Vay/Thu nhập)", "points": p, "max": 6,
                      "detail": f"Ratio = {credit}/{income}" if income else "N/A"})

    # 4. Bureau debt exposure (4 pts)
    bureau_req = (app.get("AMT_REQ_CREDIT_BUREAU_YEAR") or 0)
    if bureau_req <= 1:
        p = 4; met.append("Ít khoản vay ngoài")
    elif bureau_req <= 3:
        p = 3
    elif bureau_req <= 5:
        p = 2; review.append(f"Nhiều khoản tín dụng ({int(bureau_req)} trong năm)")
    else:
        p = 0; review.append(f"Quá nhiều khoản tín dụng ({int(bureau_req)}/năm)")
    pts += p
    breakdown.append({"criterion": "Gánh nặng nợ ngoài (CIC)", "points": p, "max": 4,
                      "detail": f"Bureau requests/year = {bureau_req}"})

    pts = min(pts, 20)
    status = _status_from_pct(pts / 20)
    return DimensionScore(pts, 20, status, breakdown, met, review)


# ─────────────────────────────────────────────────────────────────────────────
# C4 — Conditions (0-10)
# ─────────────────────────────────────────────────────────────────────────────

def score_conditions(
    app: dict, llm_feats: dict, financial_ratios: dict, **_
) -> DimensionScore:
    """
    C4 Conditions — Điều kiện vay (max 10).

    Criteria:
      - Loan purpose clarity          : 0-3 pts
      - Education level               : 0-3 pts
      - Regional risk rating          : 0-2 pts
      - Age & family stability        : 0-2 pts
    """
    feats = llm_feats or {}
    pts = 0
    breakdown = []
    met = []
    review = []

    # 1. Loan purpose (3 pts)
    purpose = (feats.get("loan_purpose_category") or "UNCLEAR").upper()
    purpose_pts = {"PRODUCTION": 3, "INVESTMENT": 3, "CONSUMPTION": 2,
                   "REFINANCING": 1, "UNCLEAR": 0}
    p = purpose_pts.get(purpose, 0)
    if p >= 2:
        met.append(f"Mục đích vay rõ ràng ({purpose})")
    else:
        review.append(f"Mục đích vay chưa rõ ({purpose})")
    pts += p
    breakdown.append({"criterion": "Mục đích vay", "points": p, "max": 3,
                      "detail": f"Purpose = {purpose}"})

    # 2. Education (3 pts)
    edu = (app.get("NAME_EDUCATION_TYPE") or "").lower()
    if "higher" in edu or "academic" in edu:
        p = 3; met.append("Trình độ học vấn cao")
    elif "secondary" in edu or "complete" in edu:
        p = 2
    elif "incomplete" in edu:
        p = 1; review.append("Trình độ học vấn chưa hoàn tất")
    else:
        p = 1
    pts += p
    breakdown.append({"criterion": "Trình độ học vấn", "points": p, "max": 3,
                      "detail": f"Education = {app.get('NAME_EDUCATION_TYPE')}"})

    # 3. Regional rating (2 pts)
    region = app.get("REGION_RATING_CLIENT")
    if region is not None:
        if region <= 1:
            p = 2; met.append("Khu vực kinh tế tốt")
        elif region == 2:
            p = 1
        else:
            p = 0; review.append(f"Khu vực có xếp hạng rủi ro {region}")
    else:
        p = 1
    pts += p
    breakdown.append({"criterion": "Xếp hạng vùng", "points": p, "max": 2,
                      "detail": f"REGION_RATING = {region}"})

    # 4. Age & family stability (2 pts)
    age = _age_years(app)
    if age is not None:
        if 25 <= age <= 60:
            p = 2; met.append(f"Độ tuổi phù hợp ({age} tuổi)")
        elif 22 <= age <= 65:
            p = 1
        else:
            p = 0; review.append(f"Tuổi ngoài khoảng lý tưởng ({age})")
    else:
        p = 1
    pts += p
    breakdown.append({"criterion": "Tuổi & ổn định gia đình", "points": p, "max": 2,
                      "detail": f"Age = {age}"})

    pts = min(pts, 10)
    status = _status_from_pct(pts / 10)
    return DimensionScore(pts, 10, status, breakdown, met, review)


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Collateral (0-20)
# ─────────────────────────────────────────────────────────────────────────────

def score_collateral(
    app: dict, financial_ratios: dict, llm_feats: dict, **_
) -> DimensionScore:
    """
    C5 Collateral — Tài sản bảo đảm (max 20).

    Criteria:
      - LTV ratio                     : 0-10 pts
      - Property ownership            : 0-5 pts
      - Collateral quality proxy      : 0-5 pts
    """
    fr = financial_ratios or {}
    pts = 0
    breakdown = []
    met = []
    review = []

    # 1. LTV (10 pts)
    ltv = fr.get("ltv")
    if ltv is not None:
        if ltv < 0.50:
            p = 10; met.append(f"LTV {ltv*100:.0f}% — rất an toàn")
        elif ltv < 0.60:
            p = 8; met.append(f"LTV {ltv*100:.0f}% — tốt")
        elif ltv < 0.70:
            p = 6; met.append(f"LTV {ltv*100:.0f}% — chấp nhận")
        elif ltv < 0.80:
            p = 4; review.append(f"LTV {ltv*100:.0f}% — gần ngưỡng")
        elif ltv < 1.0:
            p = 2; review.append(f"LTV {ltv*100:.0f}% — vượt ngưỡng 70%")
        else:
            p = 0; review.append(f"LTV {ltv*100:.0f}% — vượt 100%, rủi ro cao")
        pts += p
        breakdown.append({"criterion": "LTV (Vay/TSBĐ)", "points": p, "max": 10,
                          "detail": f"LTV = {ltv*100:.1f}%"})
    else:
        review.append("Không xác định được LTV — thiếu dữ liệu TSBĐ")
        breakdown.append({"criterion": "LTV", "points": 0, "max": 10, "detail": "N/A"})

    # 2. Property ownership (5 pts)
    if app.get("FLAG_OWN_REALTY") == "Y":
        p = 5; met.append("Sở hữu bất động sản")
    elif app.get("FLAG_OWN_CAR") == "Y":
        p = 3; met.append("Sở hữu ô tô (làm TSBĐ phụ)")
    else:
        p = 0; review.append("Không có tài sản sở hữu làm TSBĐ")
    pts += p
    breakdown.append({"criterion": "Quyền sở hữu tài sản", "points": p, "max": 5,
                      "detail": f"Realty={app.get('FLAG_OWN_REALTY')}, Car={app.get('FLAG_OWN_CAR')}"})

    # 3. Collateral quality proxy (5 pts) — housing info
    housing = (app.get("NAME_HOUSING_TYPE") or "").lower()
    walls = (app.get("WALLSMATERIAL_MODE") or "").lower()
    total_area = app.get("TOTALAREA_MODE")

    quality_pts = 0
    if "house" in housing or "apartment" in housing:
        quality_pts += 2
    if "stone" in walls or "brick" in walls or "panel" in walls:
        quality_pts += 2
    if total_area and total_area > 0.15:
        quality_pts += 1
    quality_pts = min(quality_pts, 5)

    if quality_pts >= 4:
        met.append("Chất lượng tài sản tốt (nhà xây kiên cố)")
    elif quality_pts >= 2:
        pass  # neutral
    else:
        review.append("Chất lượng tài sản thấp hoặc không rõ")

    pts += quality_pts
    breakdown.append({"criterion": "Chất lượng TSBĐ", "points": quality_pts, "max": 5,
                      "detail": f"Housing={housing}, Walls={walls}, Area={total_area}"})

    pts = min(pts, 20)
    status = _status_from_pct(pts / 20)
    return DimensionScore(pts, 20, status, breakdown, met, review)


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_five_c_scores(
    app_row: dict,
    financial_ratios: dict,
    shap_values: dict,
    llm_feats: dict,
) -> dict[str, DimensionScore]:
    """
    Compute deterministic 5C scores.

    Returns dict: {dimension_name: DimensionScore}
    """
    ctx = dict(
        app=app_row or {},
        financial_ratios=financial_ratios or {},
        shap=shap_values or {},
        llm_feats=llm_feats or {},
    )

    scores = {
        "character":  score_character(**ctx),
        "capacity":   score_capacity(**ctx),
        "capital":    score_capital(**ctx),
        "conditions": score_conditions(**ctx),
        "collateral": score_collateral(**ctx),
    }

    total = sum(s.score for s in scores.values())
    logger.info(f"  5C Rule-based Total: {total}/120")
    for dim, s in scores.items():
        logger.info(f"    {dim}: {s.score}/{s.max_score} ({s.status})")

    return scores


def five_c_to_dict(scores: dict[str, DimensionScore]) -> dict:
    """Convert DimensionScore objects to plain dict for JSON serialization."""
    return {
        dim: {
            "score": s.score,
            "max_score": s.max_score,
            "status": s.status,
            "pct": f"{s.pct*100:.0f}%",
            "indicators_met": s.indicators_met,
            "indicators_review": s.indicators_review,
            "breakdown": s.breakdown,
        }
        for dim, s in scores.items()
    }
