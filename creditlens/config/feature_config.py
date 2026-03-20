"""
CreditLens Feature Configuration.

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
OVERALL_CONFIDENCE_AUTO_PROCEED = 0.80
OVERALL_CONFIDENCE_PROCEED_WITH_WARNINGS = 0.65  # 0.65-0.80
# Below 0.65 → ESCALATE_TO_HUMAN


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


# ─── Feature to 4C Mapping ───────────────────────────────────────────────────

FEATURE_TO_4C_MAPPING: dict[str, str] = {
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
    # Capital (vốn — net worth & collateral)
    "collateral_value": "capital",
    "num_active_loans": "capital",
    "total_outstanding": "capital",
    "loan_amount_vnd": "capital",
    # Conditions (điều kiện — loan purpose & external factors)
    "loan_purpose_category": "conditions",
    "repayment_plan_quality": "conditions",
    "business_legitimacy_score": "conditions",
    "term_months": "conditions",
    "thin_file_flag": "conditions",
}


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
    # Transaction behavioral
    "salary_pattern_detected": "Phát hiện giao dịch lương đều đặn",
    "income_stability_index": "Thu nhập ổn định 6 tháng",
    "regular_bill_payment_ratio": "Thanh toán hóa đơn đúng hạn",
    "avg_monthly_inflow_vnd": "Dòng tiền vào trung bình hàng tháng",
    "debt_service_behavior": "Hành vi trả nợ",
    "overdraft_count_6m": "Số lần tài khoản âm trong 6 tháng",
    "inflow_outflow_ratio": "Tỷ lệ thu/chi",
    "max_single_outflow_ratio": "Giao dịch chi lớn nhất so với thu nhập",
    # Credit bureau
    "cic_score": "Điểm CIC (Trung tâm Thông tin Tín dụng)",
    "debt_group": "Nhóm nợ CIC",
    "num_active_loans": "Số khoản vay đang hoạt động",
    "thin_file_flag": "Không có lịch sử tín dụng CIC",
    # LLM semantic
    "loan_purpose_category": "Mục đích vay",
    "repayment_plan_quality": "Chất lượng kế hoạch trả nợ",
    "stated_income_consistency": "Thu nhập khai báo khớp sao kê",
    "business_legitimacy_score": "Tính hợp pháp doanh nghiệp",
    # Loan terms
    "dti_ratio": "Tỷ lệ nợ/thu nhập",
    "loan_amount_vnd": "Số tiền vay",
    "term_months": "Thời hạn vay (tháng)",
    # Imputation
    "imputation_confidence": "Một số trường được ước tính",
    "income_imputed_flag": "Thu nhập được ước tính từ sao kê",
    # Identity
    "age": "Tuổi",
    "gender": "Giới tính",
    "id_verified": "Xác minh danh tính",
    # Other
    "employment_duration": "Thời gian công tác",
    "collateral_value": "Giá trị tài sản bảo đảm",
}
