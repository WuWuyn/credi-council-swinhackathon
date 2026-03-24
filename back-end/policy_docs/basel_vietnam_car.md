# Basel II/III tại Việt Nam — Tỷ lệ An toàn Vốn và Quản lý Rủi ro Tín dụng

## 1. Tổng quan

NHNN áp dụng chuẩn mực Basel qua các thông tư:
- **TT41/2016**: Basel II — CAR ≥ 8% (hiệu lực 01/01/2020)
- **TT13/2018**: Basel II Pillar 2 — ICAAP
- **TT14/2025**: Basel III — CAR ≥ 10.5% (hiệu lực 01/07/2025, đầy đủ 01/01/2030)

## 2. Basel II — TT41/2016

### 2.1 Tỷ lệ An toàn Vốn (CAR)

```
CAR = Vốn tự có / Tài sản có rủi ro ≥ 8%
```

### 2.2 Trọng số rủi ro tín dụng

| Loại tài sản | Trọng số | Ví dụ |
|---|---|---|
| Trái phiếu Chính phủ VN | 0% | Nghĩa vụ Chính phủ |
| Cho vay BĐS (LTV ≤ 70%) | 50% | Vay mua nhà |
| Cho vay BĐS (LTV > 70%) | 100% | Vay mua nhà LTV cao |
| Cho vay cá nhân/SME | 100% | Vay tiêu dùng |
| Nợ quá hạn > 90 ngày | 150% | Nợ dưới tiêu chuẩn |

### 2.3 Ba trụ cột Basel II

| Trụ cột | Nội dung | Thông tư |
|---|---|---|
| Pillar 1 | Vốn tối thiểu CAR ≥ 8% | TT41/2016 |
| Pillar 2 | ICAAP, stress testing | TT13/2018 |
| Pillar 3 | Kỷ luật thị trường, công bố thông tin | TT41/2016 |

## 3. Basel III — TT14/2025

### 3.1 Nâng cao yêu cầu vốn

| Chỉ số | Basel II | Basel III |
|---|---|---|
| CAR tổng thể | ≥ 8% | ≥ **10.5%** |
| CET1 (Common Equity Tier 1) | Không riêng | ≥ **4.5%** (7% với buffer) |
| Tier 1 | Không riêng | ≥ **6%** |
| Capital Conservation Buffer | Không có | **2.5%** |
| Countercyclical Buffer | Không có | **0–2.5%** |

### 3.2 Phương pháp tính RWA

- **Standardized Approach (SA)**: Trọng số cố định — bắt buộc tất cả TCTD
- **IRB Approach**: TCTD tự ước tính PD từ mô hình nội bộ — cần NHNN phê duyệt. CreditLens ML scoring output PD có thể dùng cho IRB.

### 3.3 Timeline

| Mốc | Yêu cầu |
|---|---|
| 01/07/2025 | TT14 có hiệu lực |
| 2027–2028 | CCB 2.5% bắt buộc |
| 01/01/2030 | Đầy đủ Basel III |

## 4. Quản lý Rủi ro Tín dụng theo Basel

### 4.1 Thành phần rủi ro

| Thành phần | Ký hiệu | Mô tả | CreditLens |
|---|---|---|---|
| Probability of Default | PD | Xác suất vỡ nợ | pd_pct output |
| Loss Given Default | LGD | Tỷ lệ tổn thất | 45% chuẩn ngành |
| Exposure at Default | EAD | Dư nợ khi vỡ nợ | AMT_CREDIT |

### 4.2 Expected Loss

```
EL = PD × LGD × EAD
```

Ví dụ: PD=5.8%, LGD=45%, EAD=300M → EL = 7,830,000 VND

### 4.3 RAROC

```
RAROC = (Gross Interest Income − Expected Loss) / Loan Amount
```

| RAROC | Đánh giá |
|---|---|
| > 8% | Tốt — approve |
| 4–8% | Chấp nhận — review |
| 0–4% | Thấp — conditional |
| < 0% | Không khả thi — reject |

## 5. Stress Testing (Pillar 2)

TCTD thực hiện stress testing hàng năm:

| Kịch bản | GDP | Lạm phát | NPL |
|---|---|---|---|
| Baseline | +6% | 3% | Bình thường |
| Adverse | +2% | 6% | Tăng 50% |
| Severe | -2% | 10% | Tăng 200% |

## 6. Ứng dụng trong CreditLens A4

| Section | Trích dẫn | Nội dung |
|---|---|---|
| Executive Summary | PD, EL | "PD 5.8% — EL theo Basel II" |
| Capacity | DTI vs CAR | "DTI phù hợp an toàn vốn" |
| Reward | RAROC | "RAROC 8.5% — đạt ngưỡng" |
| Audit | Model validation | "ML PD phù hợp IRB approach" |
