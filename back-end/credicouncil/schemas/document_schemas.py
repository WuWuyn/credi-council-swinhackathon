"""
CREDICOUNCIL — Document Extraction Schemas (Pydantic).

Defines structured schemas for each Vietnamese banking document type.
Used with Gemini structured output (response_schema) for LLM-based extraction.
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── 1. CCCD / CMND ─────────────────────────────────────────────────────────────

class CCCDExtraction(BaseModel):
    """Thông tin trích xuất từ Căn cước công dân / CMND."""
    full_name: Optional[str] = Field(None, description="Họ và tên đầy đủ")
    id_number: Optional[str] = Field(None, description="Số CCCD hoặc CMND (12 chữ số)")
    date_of_birth: Optional[str] = Field(None, description="Ngày sinh, format YYYY-MM-DD")
    gender: Optional[Literal["Nam", "Nữ"]] = Field(None, description="Giới tính: Nam hoặc Nữ")
    nationality: Optional[str] = Field(None, description="Quốc tịch")
    permanent_address: Optional[str] = Field(None, description="Nơi thường trú / Địa chỉ thường trú")
    current_address: Optional[str] = Field(None, description="Địa chỉ hiện tại (nếu khác thường trú)")
    id_issue_date: Optional[str] = Field(None, description="Ngày cấp CCCD, format YYYY-MM-DD")
    id_issue_place: Optional[str] = Field(None, description="Nơi cấp CCCD")
    registration_date: Optional[str] = Field(None, description="Ngày đăng ký, format YYYY-MM-DD")


# ── 2. Hợp đồng lao động ───────────────────────────────────────────────────────

class EmploymentExtraction(BaseModel):
    """Thông tin trích xuất từ Hợp đồng lao động."""
    full_name: Optional[str] = Field(None, description="Họ và tên người lao động")
    employer_name: Optional[str] = Field(None, description="Tên doanh nghiệp / tổ chức sử dụng lao động")
    employer_address: Optional[str] = Field(None, description="Địa chỉ doanh nghiệp")
    employer_tax_id: Optional[str] = Field(None, description="Mã số thuế doanh nghiệp")
    employer_phone: Optional[str] = Field(None, description="Số điện thoại doanh nghiệp")
    position: Optional[str] = Field(None, description="Chức danh / vị trí công việc")
    department: Optional[str] = Field(None, description="Phòng ban")
    contract_type: Optional[str] = Field(
        None,
        description="Loại hợp đồng: 'Không xác định thời hạn', 'Xác định thời hạn', 'Thời vụ', etc."
    )
    employment_start_date: Optional[str] = Field(None, description="Ngày bắt đầu làm việc, format YYYY-MM-DD")
    base_salary: Optional[float] = Field(None, description="Lương cơ bản VND/tháng (số nguyên, không dấu chấm phẩy)")
    allowance: Optional[float] = Field(None, description="Phụ cấp VND/tháng")
    total_monthly_income: Optional[float] = Field(None, description="Tổng thu nhập tháng VND")
    annual_income: Optional[float] = Field(None, description="Thu nhập năm VND. Nếu chỉ có lương tháng thì nhân 12")
    social_insurance_number: Optional[str] = Field(None, description="Số sổ BHXH")


# ── 3. Sổ hộ khẩu ──────────────────────────────────────────────────────────────

class HouseholdExtraction(BaseModel):
    """Thông tin trích xuất từ Sổ hộ khẩu."""
    head_of_household: Optional[str] = Field(None, description="Họ tên chủ hộ")
    family_members_count: Optional[int] = Field(None, description="Tổng số nhân khẩu (số nguyên)")
    children_count: Optional[int] = Field(None, description="Số con (số nguyên)")
    marital_status: Optional[str] = Field(
        None,
        description="Tình trạng hôn nhân chủ hộ: 'Married', 'Single / not married', 'Separated', 'Widow', 'Civil marriage'"
    )


# ── 4. Biên bản thẩm định nhà ở / TSBĐ ─────────────────────────────────────────

class HousingSurveyExtraction(BaseModel):
    """Thông tin trích xuất từ Biên bản thẩm định nhà ở / Tài sản bảo đảm."""

    # Basic property info
    housing_type: Optional[str] = Field(
        None, description="Loại hình nhà ở: 'House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'"
    )
    housetype_mode: Optional[str] = Field(
        None, description="Loại tòa nhà: 'block of flats', 'terraced house', 'specific housing'"
    )
    living_area: Optional[float] = Field(None, description="Diện tích sử dụng (m²)")
    year_built: Optional[int] = Field(None, description="Năm xây dựng (nếu có)")
    max_floors: Optional[int] = Field(None, description="Số tầng của tòa nhà")
    has_elevator: Optional[bool] = Field(None, description="Có thang máy không?")
    wall_material: Optional[str] = Field(
        None, description="Vật liệu tường: 'Stone, brick', 'Panel', 'Wooden', 'Mixed', 'Block', 'Monolithic', 'Others'"
    )
    emergency_state: Optional[str] = Field(
        None, description="Tình trạng khẩn cấp: 'No' nếu bình thường, 'Yes' nếu khẩn cấp"
    )
    estimated_value: Optional[float] = Field(None, description="Giá trị ước tính BĐS (VND)")
    fond_kapremont: Optional[str] = Field(
        None, description="Quỹ sửa chữa: 'reg oper account', 'reg oper spec account', 'not specified', 'org spec account'. Null nếu không có"
    )

    # Region ratings
    region_population_relative: Optional[float] = Field(None, description="Mật độ dân số tương đối (0 đến 0.1)")
    region_rating: Optional[int] = Field(None, description="Xếp hạng khu vực (1, 2, hoặc 3)")
    region_rating_w_city: Optional[int] = Field(None, description="Xếp hạng khu vực kèm thành phố (1, 2, hoặc 3)")

    # Cross-check booleans
    reg_live_same_region: Optional[bool] = Field(None, description="Đăng ký cùng vùng sống?")
    reg_work_same_city: Optional[bool] = Field(None, description="Đăng ký cùng TP làm việc?")
    live_work_same_region: Optional[bool] = Field(None, description="Sống cùng vùng làm việc?")
    reg_city_same_live_city: Optional[bool] = Field(None, description="Đăng ký cùng TP sống?")
    reg_city_same_work_city: Optional[bool] = Field(None, description="Đăng ký cùng TP làm việc (TP)?")
    live_city_same_work_city: Optional[bool] = Field(None, description="Sống cùng TP làm việc (TP)?")

    # Normalized housing metrics (0-1 range from thẩm định report)
    apartments_norm: Optional[float] = Field(None, description="Diện tích căn hộ (normalized 0-1)")
    apartments_mode_norm: Optional[float] = Field(None, description="Diện tích căn hộ MODE (normalized 0-1)")
    apartments_medi_norm: Optional[float] = Field(None, description="Diện tích căn hộ MEDI (normalized 0-1)")
    basementarea_norm: Optional[float] = Field(None, description="Diện tích tầng hầm (normalized 0-1)")
    basementarea_mode_norm: Optional[float] = Field(None, description="Diện tích tầng hầm MODE (normalized 0-1)")
    basementarea_medi_norm: Optional[float] = Field(None, description="Diện tích tầng hầm MEDI (normalized 0-1)")
    years_beginexpluatation_norm: Optional[float] = Field(None, description="Năm bắt đầu sử dụng (normalized 0-1)")
    years_beginexpluatation_mode_norm: Optional[float] = Field(None, description="Năm bắt đầu sử dụng MODE (normalized 0-1)")
    years_beginexpluatation_medi_norm: Optional[float] = Field(None, description="Năm bắt đầu sử dụng MEDI (normalized 0-1)")
    years_build_norm: Optional[float] = Field(None, description="Năm xây dựng (normalized 0-1)")
    years_build_mode_norm: Optional[float] = Field(None, description="Năm xây dựng MODE (normalized 0-1)")
    years_build_medi_norm: Optional[float] = Field(None, description="Năm xây dựng MEDI (normalized 0-1)")
    commonarea_norm: Optional[float] = Field(None, description="Diện tích chung (normalized 0-1)")
    commonarea_mode_norm: Optional[float] = Field(None, description="Diện tích chung MODE (normalized 0-1)")
    commonarea_medi_norm: Optional[float] = Field(None, description="Diện tích chung MEDI (normalized 0-1)")
    elevators_norm: Optional[float] = Field(None, description="Thang máy (normalized 0-1)")
    elevators_mode_norm: Optional[float] = Field(None, description="Thang máy MODE (normalized 0-1)")
    elevators_medi_norm: Optional[float] = Field(None, description="Thang máy MEDI (normalized 0-1)")
    entrances_norm: Optional[float] = Field(None, description="Số lối vào (normalized 0-1)")
    entrances_mode_norm: Optional[float] = Field(None, description="Số lối vào MODE (normalized 0-1)")
    entrances_medi_norm: Optional[float] = Field(None, description="Số lối vào MEDI (normalized 0-1)")
    floorsmax_norm: Optional[float] = Field(None, description="Số tầng max (normalized 0-1)")
    floorsmax_mode_norm: Optional[float] = Field(None, description="Số tầng max MODE (normalized 0-1)")
    floorsmax_medi_norm: Optional[float] = Field(None, description="Số tầng max MEDI (normalized 0-1)")
    floorsmin_norm: Optional[float] = Field(None, description="Số tầng min (normalized 0-1)")
    floorsmin_mode_norm: Optional[float] = Field(None, description="Số tầng min MODE (normalized 0-1)")
    floorsmin_medi_norm: Optional[float] = Field(None, description="Số tầng min MEDI (normalized 0-1)")
    landarea_norm: Optional[float] = Field(None, description="Diện tích đất (normalized 0-1)")
    landarea_mode_norm: Optional[float] = Field(None, description="Diện tích đất MODE (normalized 0-1)")
    landarea_medi_norm: Optional[float] = Field(None, description="Diện tích đất MEDI (normalized 0-1)")
    livingapartments_norm: Optional[float] = Field(None, description="Diện tích ở (căn hộ, normalized 0-1)")
    livingapartments_mode_norm: Optional[float] = Field(None, description="Diện tích ở MODE (normalized 0-1)")
    livingapartments_medi_norm: Optional[float] = Field(None, description="Diện tích ở MEDI (normalized 0-1)")
    livingarea_norm: Optional[float] = Field(None, description="Diện tích sống (normalized 0-1)")
    livingarea_mode_norm: Optional[float] = Field(None, description="Diện tích sống MODE (normalized 0-1)")
    livingarea_medi_norm: Optional[float] = Field(None, description="Diện tích sống MEDI (normalized 0-1)")
    nonlivingapartments_norm: Optional[float] = Field(None, description="Diện tích phi ở phòng (normalized 0-1)")
    nonlivingapartments_mode_norm: Optional[float] = Field(None, description="Diện tích phi ở phòng MODE (normalized 0-1)")
    nonlivingapartments_medi_norm: Optional[float] = Field(None, description="Diện tích phi ở phòng MEDI (normalized 0-1)")
    nonlivingarea_norm: Optional[float] = Field(None, description="Diện tích phi ở (normalized 0-1)")
    nonlivingarea_mode_norm: Optional[float] = Field(None, description="Diện tích phi ở MODE (normalized 0-1)")
    nonlivingarea_medi_norm: Optional[float] = Field(None, description="Diện tích phi ở MEDI (normalized 0-1)")
    totalarea_norm: Optional[float] = Field(None, description="Tổng diện tích (normalized 0-1)")


# ── 5. Đơn đề nghị vay vốn ──────────────────────────────────────────────────────

class LoanApplicationExtraction(BaseModel):
    """Thông tin trích xuất từ Đơn đề nghị vay vốn."""
    contract_type: Optional[str] = Field(
        None, description="Loại hợp đồng: 'Cash loans' hoặc 'Revolving loans'"
    )
    loan_amount: Optional[float] = Field(None, description="Số tiền vay VND (số nguyên)")
    loan_term_months: Optional[int] = Field(None, description="Kỳ hạn vay (tháng)")
    monthly_payment: Optional[float] = Field(None, description="Trả hàng tháng dự kiến VND")
    goods_price: Optional[float] = Field(None, description="Giá trị hàng hóa VND")
    loan_purpose: Optional[str] = Field(None, description="Mục đích vay")
    education_type: Optional[str] = Field(
        None,
        description="Trình độ học vấn: 'Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'"
    )
    marital_status: Optional[str] = Field(
        None,
        description="Tình trạng hôn nhân: 'Married', 'Single / not married', 'Separated', 'Widow', 'Civil marriage'"
    )
    has_car: Optional[bool] = Field(None, description="Có sở hữu xe ô tô không?")
    car_age: Optional[int] = Field(None, description="Tuổi xe (năm). Null nếu không có xe")
    has_realty: Optional[bool] = Field(None, description="Có sở hữu bất động sản không?")
    type_suite: Optional[str] = Field(
        None,
        description="Người đồng hành khi nộp đơn: 'Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_A', 'Other_B', 'Group of people'"
    )
    flag_cont_mobile: Optional[bool] = Field(None, description="Số di động liên lạc được?")
    flag_phone: Optional[bool] = Field(None, description="Có số điện thoại bàn?")
    flag_email: Optional[bool] = Field(None, description="Có email?")
    flag_emp_phone: Optional[int] = Field(None, description="Có SĐT nơi làm việc? (0 hoặc 1)")
    flag_work_phone: Optional[int] = Field(None, description="Có SĐT bàn công ty? (0 hoặc 1)")
    days_last_phone_change_info: Optional[str] = Field(
        None, description="Thông tin đổi SĐT gần nhất (ngày YYYY-MM-DD hoặc mô tả thời gian)"
    )
    weekday_appr: Optional[str] = Field(
        None, description="Ngày nộp đơn trong tuần: 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'"
    )
    hour_appr: Optional[int] = Field(None, description="Giờ nộp đơn (0-23)")

    # Document flags (FLAG_DOCUMENT_2 to FLAG_DOCUMENT_21)
    flag_document_2: Optional[int] = Field(None, description="Tài liệu 2 đã nộp? (0 hoặc 1)")
    flag_document_3: Optional[int] = Field(None, description="Tài liệu 3 đã nộp? (0 hoặc 1)")
    flag_document_4: Optional[int] = Field(None, description="Tài liệu 4 đã nộp? (0 hoặc 1)")
    flag_document_5: Optional[int] = Field(None, description="Tài liệu 5 đã nộp? (0 hoặc 1)")
    flag_document_6: Optional[int] = Field(None, description="Tài liệu 6 đã nộp? (0 hoặc 1)")
    flag_document_7: Optional[int] = Field(None, description="Tài liệu 7 đã nộp? (0 hoặc 1)")
    flag_document_8: Optional[int] = Field(None, description="Tài liệu 8 đã nộp? (0 hoặc 1)")
    flag_document_9: Optional[int] = Field(None, description="Tài liệu 9 đã nộp? (0 hoặc 1)")
    flag_document_10: Optional[int] = Field(None, description="Tài liệu 10 đã nộp? (0 hoặc 1)")
    flag_document_11: Optional[int] = Field(None, description="Tài liệu 11 đã nộp? (0 hoặc 1)")
    flag_document_12: Optional[int] = Field(None, description="Tài liệu 12 đã nộp? (0 hoặc 1)")
    flag_document_13: Optional[int] = Field(None, description="Tài liệu 13 đã nộp? (0 hoặc 1)")
    flag_document_14: Optional[int] = Field(None, description="Tài liệu 14 đã nộp? (0 hoặc 1)")
    flag_document_15: Optional[int] = Field(None, description="Tài liệu 15 đã nộp? (0 hoặc 1)")
    flag_document_16: Optional[int] = Field(None, description="Tài liệu 16 đã nộp? (0 hoặc 1)")
    flag_document_17: Optional[int] = Field(None, description="Tài liệu 17 đã nộp? (0 hoặc 1)")
    flag_document_18: Optional[int] = Field(None, description="Tài liệu 18 đã nộp? (0 hoặc 1)")
    flag_document_19: Optional[int] = Field(None, description="Tài liệu 19 đã nộp? (0 hoặc 1)")
    flag_document_20: Optional[int] = Field(None, description="Tài liệu 20 đã nộp? (0 hoặc 1)")
    flag_document_21: Optional[int] = Field(None, description="Tài liệu 21 đã nộp? (0 hoặc 1)")


# ── Schema registry ─────────────────────────────────────────────────────────────

DOC_SCHEMAS: dict[str, type[BaseModel]] = {
    "cccd": CCCDExtraction,
    "employment": EmploymentExtraction,
    "household": HouseholdExtraction,
    "housing": HousingSurveyExtraction,
    "loan_application": LoanApplicationExtraction,
}

DOC_TYPE_VI: dict[str, str] = {
    "cccd": "Căn cước công dân (CCCD/CMND)",
    "employment": "Hợp đồng lao động",
    "household": "Sổ hộ khẩu",
    "housing": "Biên bản thẩm định nhà ở / Tài sản bảo đảm",
    "loan_application": "Đơn đề nghị vay vốn",
}
