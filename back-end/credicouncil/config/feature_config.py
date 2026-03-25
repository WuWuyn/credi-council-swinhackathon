"""
CREDICOUNCIL Feature Configuration.

Defines feature tiers, confidence thresholds, weights, and
the mapping from features to 4C dimensions (Character/Capacity/Capital/Conditions).
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


# ─── Feature Tiers ────────────────────────────────────────────────────────────


class FeatureTier(str, Enum):
    """Classification tier for field importance in the confidence system."""

    CRITICAL = "CRITICAL"  # Min confidence ≥ 0.85 → HALT if below
    IMPORTANT = "IMPORTANT"  # Min confidence ≥ 0.70 → IMPUTE if below
    OPTIONAL = "OPTIONAL"  # Min confidence ≥ 0.50 → skip if not available


# ─── Confidence Thresholds ────────────────────────────────────────────────────


CONFIDENCE_THRESHOLDS = {
    FeatureTier.CRITICAL: 0.85,
    FeatureTier.IMPORTANT: 0.70,
    FeatureTier.OPTIONAL: 0.50,
}

# Tier weights for overall_confidence formula
TIER_WEIGHTS = {
    FeatureTier.CRITICAL: 3,
    FeatureTier.IMPORTANT: 2,
    FeatureTier.OPTIONAL: 1,
}

# Overall confidence routing thresholds
OVERALL_CONFIDENCE_AUTO_PROCEED = 0.85
OVERALL_CONFIDENCE_PROCEED_WITH_WARNINGS = 0.70  # 0.70-0.85
# Below 0.70 → ESCALATE_TO_HUMAN


# ─── Field Tier Classification ────────────────────────────────────────────────


@dataclass
class FieldDefinition:
    """Definition of a single field with its tier and metadata."""

    name: str
    tier: FeatureTier
    description: str
    default_value: object = None


FIELD_DEFINITIONS: dict[str, FieldDefinition] = {
    # 🔴 CRITICAL — HALT if below threshold
    "identity_verified": FieldDefinition(
        name="identity_verified",
        tier=FeatureTier.CRITICAL,
        description="Identity verification from CCCD/CMND document",
    ),
    "monthly_income_or_inflow": FieldDefinition(
        name="monthly_income_or_inflow",
        tier=FeatureTier.CRITICAL,
        description="Monthly income from salary or bank statement inflow",
    ),
    "debt_group": FieldDefinition(
        name="debt_group",
        tier=FeatureTier.CRITICAL,
        description="CIC debt classification group (or thin_file_flag)",
    ),
    # 🟡 IMPORTANT — IMPUTE via A2 if below threshold
    "employment_duration": FieldDefinition(
        name="employment_duration",
        tier=FeatureTier.IMPORTANT,
        description="Employment duration in months",
    ),
    "collateral_value": FieldDefinition(
        name="collateral_value",
        tier=FeatureTier.IMPORTANT,
        description="Value of collateral/TSBĐ (if applicable)",
    ),
    "income_stability_index": FieldDefinition(
        name="income_stability_index",
        tier=FeatureTier.IMPORTANT,
        description="Income stability index from bank statements (0-1)",
    ),
    "debt_service_behavior": FieldDefinition(
        name="debt_service_behavior",
        tier=FeatureTier.IMPORTANT,
        description="Debt repayment behavior from transaction analysis",
    ),
    # 🟢 OPTIONAL — skip if not available
    "regular_bill_payment": FieldDefinition(
        name="regular_bill_payment",
        tier=FeatureTier.OPTIONAL,
        description="Regular bill payment ratio (electricity, water, internet)",
    ),
    "overdraft_count": FieldDefinition(
        name="overdraft_count",
        tier=FeatureTier.OPTIONAL,
        description="Number of overdraft events in 6 months",
    ),
    "transaction_network": FieldDefinition(
        name="transaction_network",
        tier=FeatureTier.OPTIONAL,
        description="Transaction network signals",
    ),
}


# ─── Credit Score Mapping ─────────────────────────────────────────────────────


@dataclass
class RiskBandDefinition:
    """Credit score → risk band mapping definition."""

    band: str
    score_min: int
    score_max: int
    pd_min: float  # PD percentage
    pd_max: float
    auto_decision: str
    description_vi: str


RISK_BANDS: list[RiskBandDefinition] = [
    RiskBandDefinition("AAA", 720, 850, 0.0, 2.0, "AUTO_APPROVE", "Xuất sắc"),
    RiskBandDefinition("AA", 640, 719, 2.0, 8.0, "APPROVE_REVIEW", "Tốt"),
    RiskBandDefinition("A", 560, 639, 8.0, 18.0, "FULL_REVIEW", "Khá"),
    RiskBandDefinition("BBB", 460, 559, 18.0, 35.0, "CONDITIONAL", "Trung bình"),
    RiskBandDefinition("CC", 300, 459, 35.0, 100.0, "REJECT", "Rủi ro cao"),
]


# ─── Feature to 5C Mapping ───────────────────────────────────────────────────
# 5C: Character, Capacity, Capital, Conditions, Collateral (chuẩn NH VN)

FEATURE_TO_5C_MAPPING: dict[str, str] = {
    # Character (uy tín — trustworthiness & payment behavior)
    "salary_pattern_detected": "character",
    "regular_bill_payment_ratio": "character",
    "debt_service_behavior": "character",
    "cic_score": "character",
    "debt_group": "character",
    "identity_verified": "character",
    # Capacity (năng lực trả nợ — ability to repay)
    "avg_monthly_inflow_vnd": "capacity",
    "income_stability_index": "capacity",
    "stated_income_consistency": "capacity",
    "dti_ratio": "capacity",
    "employment_duration": "capacity",
    "inflow_outflow_ratio": "capacity",
    "overdraft_count_6m": "capacity",
    # Capital (vốn tự có — net worth)
    "num_active_loans": "capital",
    "total_outstanding": "capital",
    # Conditions (điều kiện — loan purpose & external factors)
    "loan_purpose_category": "conditions",
    "repayment_plan_quality": "conditions",
    "business_legitimacy_score": "conditions",
    "term_months": "conditions",
    "thin_file_flag": "conditions",
    # Collateral (tài sản bảo đảm — NEW in 5C)
    "collateral_value": "collateral",
    "loan_amount_vnd": "collateral",
}

# Backward compat alias
FEATURE_TO_4C_MAPPING = FEATURE_TO_5C_MAPPING


# ─── SHAP Feature → 5C Dimension Mapping (for ML model features) ─────────────
# Maps actual model feature name prefixes to 5C dimensions

SHAP_5C_PREFIX_MAPPING: dict[str, str] = {
    # Character — uy tín, hành vi tín dụng
    "EXT_SOURCE": "character",
    "bureau_active": "character",
    "bureau_recent": "character",
    "bureau_sum_CREDIT_ACTIVE": "character",
    "bureau_sum_STATUS": "character",
    "inst_mean_DPD": "character",
    "inst_max_DPD": "character",
    "inst_DAYS_LAST_LATE": "character",
    "pos_max_SK_DPD": "character",
    "pos_mean_SK_DPD": "character",
    "cc_SK_DPD": "character",
    "cc_max_SK_DPD": "character",
    "DEF_30": "character",
    "DEF_60": "character",
    "OBS_60": "character",
    # Capacity — năng lực trả nợ
    "AMT_INCOME": "capacity",
    "AMT_ANNUITY": "capacity",
    "AMT_CREDIT": "capacity",
    "DAYS_EMPLOYED": "capacity",
    "DAYS_BIRTH": "capacity",
    "NEW_EMPLOY_TO_BIRTH": "capacity",
    "NEW_CREDIT_TO_INCOME": "capacity",
    "AMT_ANNUITY_INCOME": "capacity",
    "inst_mean_AMT": "capacity",
    "inst_var_AMT": "capacity",
    "cc_mean_AMT_PAYMENT": "capacity",
    "cc_mean_AMT_INST": "capacity",
    "prev_avg_AMT": "capacity",
    "prev_max_AMT": "capacity",
    # Capital — vốn tự có
    "bureau_sum_AMT_CREDIT_SUM": "capital",
    "bureau_AMT_DEBT": "capital",
    "bureau_AMT_LIMIT": "capital",
    "Total_CREDIT": "capital",
    "Total_AMT": "capital",
    "Total_active": "capital",
    "cc_mean_AMT_BALANCE": "capital",
    "cc_mean_AMT_CREDIT_LIMIT": "capital",
    "cc_mean_AMT_CREDIT_USE": "capital",
    # Conditions — điều kiện khoản vay
    "NAME_CONTRACT_TYPE": "conditions",
    "NAME_INCOME_TYPE": "conditions",
    "NAME_EDUCATION_TYPE": "conditions",
    "ORGANIZATION_TYPE": "conditions",
    "prev_sum_NAME": "conditions",
    "prev_recent_NAME": "conditions",
    "REGION": "conditions",
    # Collateral — tài sản bảo đảm
    "FLAG_OWN_CAR": "collateral",
    "FLAG_OWN_REALTY": "collateral",
    "OWN_CAR_AGE": "collateral",
    "APARTMENTS": "collateral",
    "LIVINGAREA": "collateral",
    "LANDAREA": "collateral",
    "TOTALAREA": "collateral",
    "HOUSETYPE": "collateral",
    "WALLSMATERIAL": "collateral",
    "FLOORSMAX": "collateral",
}


def get_5c_dimension(feature_name: str) -> str:
    """Map a model feature to its 5C dimension.

    Uses prefix matching against SHAP_5C_PREFIX_MAPPING.
    Falls back to 'conditions' if no match found.
    """
    # Direct match first
    if feature_name in FEATURE_TO_5C_MAPPING:
        return FEATURE_TO_5C_MAPPING[feature_name]
    # Prefix match
    for prefix, dim in SHAP_5C_PREFIX_MAPPING.items():
        if feature_name.startswith(prefix):
            return dim
    return "conditions"  # default fallback


# ─── Thin-file Alternative Scoring Weights ────────────────────────────────────

THIN_FILE_FEATURE_WEIGHTS: dict[str, float] = {
    "income_stability_index": 0.30,
    "salary_pattern_detected": 0.25,
    "debt_service_behavior": 0.25,
    "regular_bill_payment_ratio": 0.15,
    "inflow_outflow_ratio": 0.05,
}

THIN_FILE_MIN_MONTHS = 3  # Minimum months of bank statement required


# ─── Unified Feature Vector — 25 features, 6 groups ──────────────────────────


@dataclass
class FeatureGroup:
    """Definition of a feature group in the unified feature vector."""

    name: str
    features: list[str]
    home_credit_source: str
    pilot_features: list[str] = field(default_factory=list)


FEATURE_GROUPS: list[FeatureGroup] = [
    FeatureGroup(
        name="Identity & KYC",
        features=["age", "gender", "id_verified"],
        home_credit_source="application_train: CODE_GENDER, DAYS_BIRTH, FLAG_OWN_CAR",
        pilot_features=["age", "gender", "id_verified"],
    ),
    FeatureGroup(
        name="Credit Bureau",
        features=["cic_score", "debt_group", "num_active_loans", "thin_file_flag"],
        home_credit_source="bureau.csv: CREDIT_ACTIVE, DAYS_CREDIT, AMT_CREDIT_SUM_OVERDUE",
        pilot_features=["cic_score", "debt_group", "num_active_loans", "thin_file_flag"],
    ),
    FeatureGroup(
        name="Transaction Behavioral",
        features=[
            "avg_monthly_inflow_vnd",
            "income_stability_index",
            "salary_pattern_detected",
            "regular_bill_payment_ratio",
            "debt_service_behavior",
            "overdraft_count_6m",
            "inflow_outflow_ratio",
            "max_single_outflow_ratio",
        ],
        home_credit_source="installments_payments.csv + credit_card_balance.csv",
        pilot_features=[
            "avg_monthly_inflow_vnd",
            "income_stability_index",
            "salary_pattern_detected",
            "regular_bill_payment_ratio",
            "debt_service_behavior",
            "overdraft_count_6m",
        ],
    ),
    FeatureGroup(
        name="LLM Semantic (A2-A)",
        features=[
            "loan_purpose_category",
            "repayment_plan_quality",
            "stated_income_consistency",
            "transaction_purpose_distribution",
            "business_legitimacy_score",
        ],
        home_credit_source="Extracted from documents (not in Home Credit)",
        pilot_features=["loan_purpose_category", "repayment_plan_quality", "stated_income_consistency"],
    ),
    FeatureGroup(
        name="Imputed Fields (A2-B)",
        features=["income_imputed_flag", "imputation_confidence"],
        home_credit_source="Proxy for EXT_SOURCE_1/2/3 in application_train",
        pilot_features=["income_imputed_flag", "imputation_confidence"],
    ),
    FeatureGroup(
        name="Loan Terms",
        features=["loan_amount_vnd", "term_months", "dti_ratio"],
        home_credit_source="application_train: AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL",
        pilot_features=["loan_amount_vnd", "term_months", "dti_ratio"],
    ),
]

# Total feature count
TOTAL_FEATURES = sum(len(g.features) for g in FEATURE_GROUPS)  # 25
PILOT_FEATURES = sum(len(g.pilot_features) for g in FEATURE_GROUPS)  # 10 core


# ─── SHAP Vietnamese Labels ──────────────────────────────────────────────────

SHAP_LABEL_VI: dict[str, str] = {
    # ── Application — Personal & Identity ──
    "DAYS_BIRTH": "Tuổi khách hàng",
    "CODE_GENDER": "Giới tính",
    "DAYS_EMPLOYED": "Thời gian công tác (ngày)",
    "DAYS_REGISTRATION": "Thời gian đăng ký cư trú",
    "DAYS_ID_PUBLISH": "Thời gian cấp CCCD",
    "DAYS_LAST_PHONE_CHANGE": "Thời gian đổi SĐT gần nhất",
    "CNT_CHILDREN": "Số con",
    "CNT_FAM_MEMBERS": "Số thành viên gia đình",
    "FLAG_OWN_CAR": "Có xe ô tô",
    "FLAG_OWN_REALTY": "Có bất động sản",
    "OWN_CAR_AGE": "Tuổi xe ô tô",
    "FLAG_EMAIL": "Có email",
    "FLAG_PHONE": "Có SĐT bàn",
    "FLAG_WORK_PHONE": "Có SĐT nơi làm việc",
    "DOCUMENT_CNT": "Số tài liệu đã nộp",
    # ── Application — Loan & Income ──
    "AMT_INCOME_TOTAL": "Thu nhập hàng năm",
    "AMT_CREDIT": "Số tiền tín dụng",
    "AMT_ANNUITY": "Khoản trả hàng kỳ",
    "AMT_GOODS_PRICE": "Giá trị hàng hóa/tài sản mua",
    "NAME_CONTRACT_TYPE": "Loại hợp đồng vay",
    "NAME_INCOME_TYPE": "Loại thu nhập",
    "NAME_EDUCATION_TYPE": "Trình độ học vấn",
    "NAME_FAMILY_STATUS": "Tình trạng hôn nhân",
    "NAME_HOUSING_TYPE": "Loại nhà ở",
    "OCCUPATION_TYPE": "Nghề nghiệp",
    "ORGANIZATION_TYPE": "Loại tổ chức nơi làm việc",
    "REGION_POPULATION_RELATIVE": "Mật độ dân số khu vực",
    "REGION_RATING_CLIENT_W_CITY": "Xếp hạng khu vực",
    # ── Application — External scores (CIC) ──
    "EXT_SOURCE_1": "Điểm tín dụng ngoại 1 (CIC)",
    "EXT_SOURCE_2": "Điểm tín dụng ngoại 2 (CIC)",
    "EXT_SOURCE_3": "Điểm tín dụng ngoại 3 (CIC)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Số người quen nợ xấu 30 ngày",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Số người quen nợ xấu 60 ngày",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Số người quen bị quan sát 60 ngày",
    # ── Application — Engineered ratios ──
    "NEW_CREDIT_TO_INCOME_RATIO": "Tỷ lệ tín dụng/thu nhập",
    "AMT_ANNUITY_INCOME_RATE": "Tỷ lệ trả góp/thu nhập",
    "AMT_CREDIT_GOODS_PERC": "Tỷ lệ tín dụng/giá trị hàng",
    "NEW_EMPLOY_TO_BIRTH_RATIO": "Tỷ lệ thời gian công tác/tuổi",
    "NEW_EXT_SOURCES_MEAN": "Trung bình điểm tín dụng ngoại",
    "NEW_SCORES_STD": "Độ lệch chuẩn điểm tín dụng ngoại",
    "NEW_SOURCES_PROD": "Tích điểm tín dụng ngoại",
    "AMT_PAY_YEAR": "Khoản trả hàng năm",
    "AGE_EMPLOYED": "Tuổi khi bắt đầu làm việc",
    "NEW_INC_PER_CHLD": "Thu nhập trên mỗi con",
    "NEW_DOC_IND_KURT": "Kurtosis tài liệu đã nộp",
    "NEW_PHONE_TO_BIRTH_RATIO": "Tỷ lệ đổi SĐT/tuổi",
    # ── Application — Housing ──
    "TOTALAREA_MODE": "Tổng diện tích nhà",
    "LIVINGAREA_MEDI": "Diện tích sử dụng",
    "LANDAREA_MEDI": "Diện tích đất",
    "APARTMENTS_MODE": "Chất lượng căn hộ",
    "FLOORSMAX_MODE": "Số tầng tối đa tòa nhà",
    # ── Application — Group relative ──
    "region_mean_income": "Thu nhập TB theo khu vực",
    "region_mean_income_rel": "Thu nhập so với TB khu vực",
    "gender_mean_income_rel": "Thu nhập so với TB giới tính",
    "family_status_mean_income_rel": "Thu nhập so với TB tình trạng HN",
    # ── Application — Total credit summary ──
    "Total_CREDIT": "Tổng dư nợ tín dụng",
    "Total_AMT_ANNUITY": "Tổng trả góp hàng kỳ",
    "Total_active_acc": "Số tài khoản đang hoạt động",
    "Total_CREDIT_INCOME_RATIO": "Tỷ lệ tổng nợ/thu nhập",
    "Total_ANNUITY_INCOME_RATIO": "Tỷ lệ tổng trả góp/thu nhập",
    # ── Bureau aggregated ──
    "bureau_count": "Số bản ghi CIC",
    "bureau_avg_DAYS_CREDIT": "TB thời gian khoản vay CIC",
    "bureau_sum_AMT_CREDIT_SUM": "Tổng dư nợ CIC",
    "bureau_sum_AMT_CREDIT_SUM_DEBT": "Tổng nợ hiện tại CIC",
    "bureau_sum_AMT_CREDIT_SUM_OVERDUE": "Tổng nợ quá hạn CIC",
    "bureau_sum_CNT_CREDIT_PROLONG": "Số lần gia hạn nợ CIC",
    "bureau_avg_DAYS_CREDIT_UPDATE": "TB ngày cập nhật CIC",
    "bureau_used_other_currency": "Có vay ngoại tệ",
    # ── Bureau active ──
    "bureau_active_count": "Số khoản vay CIC đang hoạt động",
    "bureau_active_avg_AMT_CREDIT_SUM": "TB dư nợ khoản vay đang HĐ",
    "bureau_active_avg_AMT_CREDIT_SUM_DEBT": "TB nợ hiện tại khoản vay đang HĐ",
    "bureau_active_sum_AMT_CREDIT_SUM": "Tổng dư nợ các khoản đang HĐ",
    # ── Bureau recent ──
    "bureau_recent_CREDIT_ACTIVE": "Trạng thái khoản vay CIC gần nhất",
    "bureau_recent_AMT_CREDIT_SUM": "Dư nợ khoản vay CIC gần nhất",
    "bureau_recent_DAYS_CREDIT": "Ngày mở khoản vay CIC gần nhất",
    # ── Bureau one-hot ──
    "bureau_sum_CREDIT_ACTIVE_Active": "Số khoản Active (CIC)",
    "bureau_sum_CREDIT_ACTIVE_Closed": "Số khoản Closed (CIC)",
    "bureau_sum_STATUS_TCNT_DPD_SUM": "Tổng tháng quá hạn (tất cả CIC)",
    "bureau_sum_STATUS_12CNT_DPD_SUM": "Tổng tháng quá hạn (12 tháng gần)",
    # ── Previous application ──
    "prev_count": "Số đơn vay trước",
    "prev_DEFALUTED_RATIO": "Tỷ lệ vỡ nợ đơn vay trước",
    "prev_avg_AMT_ANNUITY": "TB trả góp đơn vay trước",
    "prev_avg_AMT_CREDIT": "TB tín dụng đơn vay trước",
    "prev_avg_DAYS_DECISION": "TB thời gian quyết định",
    "prev_approved_SK_ID_PREV_COUNT": "Số đơn vay trước đã duyệt",
    "prev_refused_SK_ID_PREV_COUNT": "Số đơn vay trước bị từ chối",
    "prev_recent_NAME_CONTRACT_STATUS": "Trạng thái đơn vay gần nhất",
    "prev_recent_DAYS_DECISION": "Ngày quyết định đơn vay gần nhất",
    # ── Credit card ──
    "cc_mean_AMT_BALANCE": "TB dư nợ thẻ tín dụng",
    "cc_mean_AMT_CREDIT_LIMIT_ACTUAL": "TB hạn mức tín dụng thẻ",
    "cc_mean_AMT_CREDIT_USE_RATIO": "TB tỷ lệ sử dụng tín dụng thẻ",
    "cc_mean_AMT_DRAWINGS_CURRENT": "TB rút tiền thẻ hiện tại",
    "cc_mean_AMT_PAYMENT_CURRENT": "TB thanh toán thẻ hiện tại",
    "cc_max_SK_DPD": "Ngày quá hạn tối đa (thẻ)",
    "cc_max_SK_DPD_DEF": "Ngày quá hạn nợ xấu tối đa (thẻ)",
    "cc_var_AMT_BALANCE": "Biến động dư nợ thẻ",
    # ── POS/Cash ──
    "pos_count": "Số khoản POS/cash",
    "pos_max_SK_DPD": "Ngày quá hạn tối đa (POS)",
    "pos_mean_SK_DPD": "TB ngày quá hạn (POS)",
    "pos_recent_SK_DPD": "Ngày quá hạn gần nhất (POS)",
    # ── Installments ──
    "inst_count": "Số kỳ trả góp",
    "inst_mean_DPD": "TB ngày trả chậm",
    "inst_max_DPD": "Ngày trả chậm tối đa",
    "inst_mean_AMT_PAYMENT": "TB số tiền thanh toán",
    "inst_mean_AMT_PAYMENT_DIFF": "TB chênh lệch thanh toán vs phải trả",
    "inst_DAYS_LAST_LATE": "Ngày trả chậm gần nhất",
    "inst_DAYS_LAST_UNDERPAID": "Ngày trả thiếu gần nhất",
    # ── Cross-table ──
    "AMT_ANNUITY_to_prev_approved": "Trả góp hiện tại / TB đơn đã duyệt",
    "AMT_CREDIT_to_prev_approved": "Tín dụng hiện tại / TB đơn đã duyệt",
    "AMT_ANNUITY_to_prev_refused": "Trả góp hiện tại / TB đơn bị từ chối",
    "AMT_CREDIT_to_prev_refused": "Tín dụng hiện tại / TB đơn bị từ chối",
    # ── Behavioral (bank statement) ──
    "salary_pattern_detected": "Phát hiện giao dịch lương đều đặn",
    "income_stability_index": "Thu nhập ổn định 6 tháng",
    "regular_bill_payment_ratio": "Thanh toán hóa đơn đúng hạn",
    "avg_monthly_inflow_vnd": "Dòng tiền vào TB hàng tháng",
    "debt_service_behavior": "Hành vi trả nợ",
    "overdraft_count_6m": "Số lần tài khoản âm (6 tháng)",
    "inflow_outflow_ratio": "Tỷ lệ thu/chi",
    "max_single_outflow_ratio": "Giao dịch chi lớn nhất / thu nhập",
    # ── Others ──
    "thin_file_flag": "Không có lịch sử tín dụng CIC",
    "imputation_confidence": "Độ tin cậy dữ liệu ước tính",
    "income_imputed_flag": "Thu nhập được ước tính từ sao kê",
    "dti_ratio": "Tỷ lệ nợ/thu nhập (DTI)",
    "loan_amount_vnd": "Số tiền vay",
    "term_months": "Thời hạn vay (tháng)",
    "collateral_value": "Giá trị tài sản bảo đảm",
    "loan_purpose_category": "Mục đích vay",
    "repayment_plan_quality": "Chất lượng kế hoạch trả nợ",
    "stated_income_consistency": "Thu nhập khai báo khớp sao kê",
    "business_legitimacy_score": "Tính hợp pháp doanh nghiệp",
}


# ─── Auto-generate Vietnamese label for any feature ──────────────────────────

# Prefix → Vietnamese group name
_PREFIX_NAME_VI: dict[str, str] = {
    "bureau_active_avg_": "TB (khoản CIC đang HĐ) ",
    "bureau_active_sum_": "Tổng (khoản CIC đang HĐ) ",
    "bureau_recent_": "Gần nhất (CIC) ",
    "bureau_avg_": "TB (CIC) ",
    "bureau_sum_": "Tổng (CIC) ",
    "bureau_min_": "Min (CIC) ",
    "bureau_max_": "Max (CIC) ",
    "prev_approved_": "Đơn đã duyệt — ",
    "prev_refused_": "Đơn bị từ chối — ",
    "prev_recent_": "Đơn gần nhất — ",
    "prev_active_sum_": "Đơn đang HĐ tổng ",
    "prev_avg_": "TB đơn vay trước ",
    "prev_max_": "Max đơn vay trước ",
    "prev_sum_": "Tổng đơn vay trước ",
    "cc_mean_": "TB thẻ tín dụng ",
    "cc_mean4_": "TB thẻ (4 tháng) ",
    "cc_mean12_": "TB thẻ (12 tháng) ",
    "cc_mean36_": "TB thẻ (36 tháng) ",
    "cc_max_": "Max thẻ tín dụng ",
    "cc_var_": "Biến động thẻ ",
    "cc_scale_sum_": "Scale sum thẻ ",
    "cc_scale_mean_scale_sum_": "Scale mean thẻ ",
    "cc_min_": "Min thẻ ",
    "pos_": "POS/Cash ",
    "inst_": "Trả góp ",
    "NEW_": "Chỉ số dẫn xuất ",
    "AMT_REQ_CREDIT_BUREAU_": "Số truy vấn CIC ",
    "FLAG_DOCUMENT_": "Tài liệu loại ",
}

# Common field name → Vietnamese
_FIELD_NAME_VI: dict[str, str] = {
    "AMT_ANNUITY": "khoản trả góp",
    "AMT_CREDIT": "tín dụng",
    "AMT_APPLICATION": "số tiền đề nghị",
    "AMT_INCOME_TOTAL": "thu nhập",
    "AMT_BALANCE": "dư nợ",
    "AMT_CREDIT_LIMIT_ACTUAL": "hạn mức tín dụng",
    "AMT_CREDIT_USE_RATIO": "tỷ lệ sử dụng tín dụng",
    "AMT_DRAWINGS_CURRENT": "rút tiền hiện tại",
    "AMT_PAYMENT_CURRENT": "thanh toán hiện tại",
    "AMT_DOWN_PAYMENT": "trả trước",
    "AMT_CREDIT_SUM": "tổng dư nợ",
    "AMT_CREDIT_SUM_DEBT": "nợ hiện tại",
    "AMT_CREDIT_SUM_OVERDUE": "nợ quá hạn",
    "AMT_CREDIT_MAX_OVERDUE": "nợ quá hạn tối đa",
    "DAYS_CREDIT": "ngày mở khoản vay",
    "DAYS_CREDIT_ENDDATE": "ngày kết thúc khoản vay",
    "DAYS_DECISION": "ngày quyết định",
    "CREDIT_DAY_OVERDUE": "ngày quá hạn",
    "SK_DPD": "ngày quá hạn (DPD)",
    "SK_DPD_DEF": "ngày quá hạn nợ xấu",
    "CNT_PAYMENT": "số kỳ thanh toán",
    "RATE_DOWN_PAYMENT": "tỷ lệ trả trước",
}


def get_label_vi(feature_name: str) -> str:
    """Get Vietnamese label for a feature name.

    Priority: direct lookup → prefix match → auto-generate from name.
    """
    # 1. Direct lookup
    if feature_name in SHAP_LABEL_VI:
        return SHAP_LABEL_VI[feature_name]

    # 2. Prefix match (longest prefix first)
    for prefix in sorted(_PREFIX_NAME_VI, key=len, reverse=True):
        if feature_name.startswith(prefix):
            remainder = feature_name[len(prefix):]
            # Try to translate the remainder
            vi_remainder = _FIELD_NAME_VI.get(remainder, remainder)
            return f"{_PREFIX_NAME_VI[prefix]}{vi_remainder}"

    # 3. Auto-generate: split underscores, capitalize
    parts = feature_name.replace("_", " ").strip()
    return parts.capitalize()
