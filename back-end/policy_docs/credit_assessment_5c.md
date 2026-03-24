# Hướng dẫn Đánh giá Tín dụng 5C — Chuẩn Ngân hàng Việt Nam

## 1. Tổng quan

Framework 5C là phương pháp đánh giá tín dụng chuẩn trong ngành ngân hàng Việt Nam, được quy định tại Thông tư 39/2016/TT-NHNN Điều 17 (thẩm định khoản vay). 5C bao gồm: Character, Capacity, Capital, Conditions, Collateral.

CreditLens sử dụng 5C framework kết hợp SHAP attribution — mỗi tiêu chí có điểm số, SHAP contribution %, và narrative được grounded trên dữ liệu.

## 2. Thang điểm 5C — CreditLens

| Tiêu chí | Điểm tối đa | Trọng số | Mô tả |
|---|---|---|---|
| **C1 — Character** | 30 | 25% | Uy tín, tư cách người vay |
| **C2 — Capacity** | 40 | 33% | Năng lực trả nợ |
| **C3 — Capital** | 20 | 17% | Vốn tự có, tài sản ròng |
| **C4 — Conditions** | 10 | 8% | Điều kiện khoản vay, mục đích |
| **C5 — Collateral** | 20 | 17% | Tài sản bảo đảm |
| **TỔNG** | **120** | 100% | |

## 3. C1 — Character (Uy tín / Tư cách) — 30 điểm

### 3.1 Các chỉ số đánh giá

| Chỉ số | Nguồn dữ liệu | Điểm | Tiêu chí |
|---|---|---|---|
| Lịch sử CIC | CIC API (A1) | 0–10 | Nhóm 1 = 10đ, Nhóm 2 = 5đ, Nhóm 3+ = 0đ |
| Hành vi thanh toán | Bank statement (A1) | 0–8 | Bill payment ratio ≥ 90% = 8đ, 70–89% = 5đ |
| Lịch sử vay trước | previous_application (NoxMoon) | 0–7 | agg_prev_score_mean ≥ 0.7 = 7đ |
| Định danh nhất quán | OCR cross-check (A1) | 0–5 | identity_consistency = OK: 5đ, MISMATCH: 0đ |

### 3.2 Trạng thái đánh giá

| Điểm | Trạng thái | Ý nghĩa |
|---|---|---|
| 22–30 | ĐẠT | Uy tín tốt — đủ điều kiện |
| 15–21 | XEM XÉT | Cần thẩm định thêm |
| 0–14 | KHONG ĐẠT | Rủi ro cao — cần override hoặc reject |

### 3.3 Quy định liên quan
- TT11/2021 Điều 10: Phân loại nhóm nợ CIC → ảnh hưởng trực tiếp Character score
- QĐ493/2005: Hard override — CIC Nhóm 4–5 → Character = 0 điểm → REJECT

## 4. C2 — Capacity (Năng lực trả nợ) — 40 điểm

### 4.1 Các chỉ số đánh giá

| Chỉ số | Công thức | Điểm | Ngưỡng |
|---|---|---|---|
| DTI | Trả nợ tháng / Thu nhập tháng | 0–15 | < 30% = 15đ, 30–39% = 10đ, 40–49% = 5đ, ≥ 50% = 0đ |
| DSCR | Thu nhập tháng / Trả nợ tháng | 0–10 | ≥ 1.5 = 10đ, 1.2–1.49 = 7đ, 1.0–1.19 = 3đ, < 1.0 = 0đ |
| Thu nhập ổn định | income_stability_index (A1) | 0–8 | ≥ 0.8 = 8đ, 0.6–0.79 = 5đ, < 0.6 = 2đ |
| Salary detected | salary_pattern (A1) | 0–5 | Có pattern lương: 5đ, không: 0đ |
| Imputation flag | imputation_confidence (A2) | 0–2 | Không imputed: 2đ, imputed conf > 0.7: 1đ, < 0.7: 0đ |

### 4.2 Ngưỡng tài chính theo TT39/2016

| Chỉ số | An toàn | Cảnh báo | Rủi ro cao | Từ chối |
|---|---|---|---|---|
| DTI | < 30% | 30–40% | 40–50% | > 50% |
| DSCR | > 1.50 | 1.20–1.50 | 1.00–1.20 | < 1.00 |

### 4.3 Quy định liên quan
- TT39/2016 Điều 7: Khả năng tài chính là điều kiện bắt buộc
- Basel II: DTI là input cho tính toán RWA
- DSCR < 1.0: Không đủ dòng tiền trả nợ → cần TSBĐ bổ sung

## 5. C3 — Capital (Vốn tự có) — 20 điểm

### 5.1 Các chỉ số đánh giá

| Chỉ số | Nguồn | Điểm | Tiêu chí |
|---|---|---|---|
| Inflow/Outflow ratio | Bank statement (A1) | 0–8 | ≥ 1.3 = 8đ, 1.1–1.29 = 5đ, < 1.1 = 2đ |
| Savings behavior | Transaction pattern (A1) | 0–5 | Có tích lũy đều đặn = 5đ |
| Net worth estimate | TSBĐ + tiết kiệm − nợ | 0–5 | Net worth > 2x khoản vay = 5đ |
| Overdraft frequency | overdraft_count_6m (A1) | 0–2 | 0 lần = 2đ, 1–2 lần = 1đ, > 2 lần = 0đ |

### 5.2 Quy định liên quan
- Basel III TT14/2025: Equity ratio ảnh hưởng xếp hạng nội bộ
- TT39/2016: Tài sản ròng là yếu tố thẩm định

## 6. C4 — Conditions (Điều kiện) — 10 điểm

### 6.1 Các chỉ số đánh giá

| Chỉ số | Nguồn | Điểm | Tiêu chí |
|---|---|---|---|
| Mục đích vay | loan_purpose_category (A2) | 0–4 | PRODUCTION = 4đ, INVESTMENT = 3đ, CONSUMPTION = 2đ, UNCLEAR = 0đ |
| Kế hoạch trả nợ | repayment_plan_quality (A2) | 0–3 | DETAILED = 3đ, GENERAL = 2đ, VAGUE = 1đ, NONE = 0đ |
| Điều kiện thị trường | Sector risk assessment | 0–2 | Ngành ổn định = 2đ, biến động = 1đ |
| SME growth signal | sme_growth_signal (A2) | 0–1 | GROWING = 1đ, DECLINING = 0đ |

### 6.2 Quy định liên quan
- TT39/2016 Điều 8: Mục đích vay phải hợp pháp
- TT39/2016 Điều 7: Phương án sử dụng vốn khả thi

## 7. C5 — Collateral (Tài sản bảo đảm) — 20 điểm

### 7.1 Các chỉ số đánh giá

| Chỉ số | Nguồn | Điểm | Tiêu chí |
|---|---|---|---|
| LTV ratio | AMT_CREDIT / collateral_value | 0–8 | < 50% = 8đ, 50–69% = 6đ, 70–80% = 3đ, > 80% = 0đ |
| Loại TSBĐ | collateral_type (A1) | 0–5 | BĐS = 5đ, Xe = 3đ, Khác = 1đ, Không có = 0đ |
| Tình trạng pháp lý | Document verification (A1) | 0–4 | Hợp lệ đầy đủ = 4đ, thiếu = 2đ |
| Thanh khoản TSBĐ | Market assessment | 0–3 | Dễ bán = 3đ, trung bình = 2đ, khó = 1đ |

### 7.2 Tỷ lệ LTV theo loại TSBĐ (TT11/2021)

| Loại TSBĐ | LTV tối đa khuyến nghị | Tỷ lệ khấu trừ DPRR |
|---|---|---|
| Bất động sản nhà ở | 70% | 50% |
| Bất động sản thương mại | 60% | 50% |
| Ô tô | 70% | 50% |
| Xe máy | 80% | 30% |
| Máy móc thiết bị | 60% | 30% |
| Sổ tiết kiệm | 95% | 95-100% |

### 7.3 Yêu cầu tái định giá
- Tái định giá **hàng năm** cho BĐS
- Tái định giá **6 tháng** cho phương tiện giao thông
- Tái định giá **ngay** khi có sự kiện ảnh hưởng giá trị (thiên tai, sự cố)

## 8. Quyết định tổng hợp

### 8.1 Ma trận quyết định

| Tổng 5C | Trạng thái | Hành động | Điều kiện |
|---|---|---|---|
| 96–120 | XUẤT SẮC | AUTO APPROVE | Tất cả C ≥ ĐẠT |
| 72–95 | TỐT | APPROVE + REVIEW | Không quá 1 C ở XEM XÉT |
| 54–71 | TRUNG BÌNH | FULL REVIEW | Cần thẩm định chi tiết |
| 36–53 | DƯỚI TB | CONDITIONAL | Yêu cầu TSBĐ bổ sung |
| 0–35 | YẾU | REJECT | Trừ đặc cách cấp cao |

### 8.2 Hard Override Rules

| Điều kiện | Hành động | Bất kể 5C score |
|---|---|---|
| CIC Nhóm 4–5 | REJECT | QĐ493, TT11 |
| Khoản vay > 10 tỷ VND | ESCALATE | Thẩm quyền hội sở |
| thin_file + Score < 560 | Tăng yêu cầu TSBĐ | Chính sách nội bộ |
| DTI > 60% | REJECT | TT39 Điều 7 |
| Fraud flag | REJECT + báo cáo | Luật phòng chống rửa tiền |
