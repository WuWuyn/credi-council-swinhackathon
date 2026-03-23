# CreditLens AI — Hệ thống Đánh giá Tín dụng AI

> **AI-powered credit scoring pipeline** xây dựng trên kiến trúc MASCA (Multi-Agent System for Credit Assessment), sử dụng LightGBM + SHAP + Gemini LLM để tạo báo cáo tín dụng 5C theo chuẩn ngân hàng Việt Nam.

---

## Mục lục

- [1. Cài đặt & Chạy](#1-cài-đặt--chạy)
- [2. Test & Verify](#2-test--verify)
- [3. Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
- [4. Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
- [5. Pipeline chi tiết: A1 → A2 → A3 → A4](#5-pipeline-chi-tiết-a1--a2--a3--a4)
  - [5.1 A1 — Data Ingestion Agent](#51-a1--data-ingestion-agent)
  - [5.2 A2 — Feature Engineer Agent](#52-a2--feature-engineer-agent)
  - [5.3 A3 — ML Scoring Agent](#53-a3--ml-scoring-agent)
  - [5.4 A4 — Report Generator Agent](#54-a4--report-generator-agent)
- [6. API Endpoints](#6-api-endpoints)
- [7. Cấu hình](#7-cấu-hình)
- [8. Training Model](#8-training-model)

---

## 1. Cài đặt & Chạy

### Yêu cầu hệ thống

- Python 3.10+
- Conda (khuyến nghị)
- ~2GB RAM cho model LightGBM

### Bước 1: Tạo môi trường

```bash
conda create -n swinburn_hackathon python=3.10 -y
conda activate swinburn_hackathon
```

### Bước 2: Cài dependencies

```bash
pip install -r requirements.txt
```

Các thư viện chính:

| Nhóm | Thư viện | Mục đích |
|---|---|---|
| ML | `lightgbm`, `shap`, `scikit-learn` | Scoring model & giải thích |
| LLM | `google-genai` | Gemini API cho semantic extraction |
| PDF | `reportlab` | Tạo báo cáo PDF |
| API | `fastapi`, `uvicorn` | REST API server |
| Data | `pandas`, `numpy` | Xử lý dữ liệu |

### Bước 3: Cấu hình `.env`

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here   # Google AI Studio API key
USE_MOCK=false                             # true = không gọi Gemini API
MODEL_PATH=models/lgbm_ref_v1.pkl         # Đường dẫn model đã train
FE_STATS_PATH=models/fe_stats.pkl         # Feature engineering stats
```

> **Mock mode** (`USE_MOCK=true`): Chạy không cần Gemini API. A2 sẽ bỏ qua LLM extraction, A4 tạo report bằng logic deterministic. Phù hợp cho demo nhanh.

### Bước 4: Chạy server

```bash
conda activate swinburn_hackathon
uvicorn creditlens.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập:
- **Dashboard**: http://localhost:8000/app
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

---

## 2. Test & Verify

### Test pipeline đơn lẻ (1 customer)

```bash
conda activate swinburn_hackathon
python test_pipeline.py
```

### Test toàn bộ 4 demo customers

```bash
conda activate swinburn_hackathon
python test_all_customers.py
```

Script này chạy pipeline A1→A4 cho 4 customers, so sánh kết quả và kiểm tra:
- Score spread (phải > 50 points giữa min/max)
- High-risk < Standard (TARGET=1 phải có score thấp hơn TARGET=0)
- 5C totals phải khác nhau giữa các customers
- Consistency check phải PASS cho tất cả

**Output**: `data/mock/pipeline_test_summary.json`

### Test qua API

```bash
# Score mock customer 001
curl -X POST http://localhost:8000/score/mock -d "customer_id=001"

# Xem PDF report
curl http://localhost:8000/v1/report/001/pdf -o report_001.pdf
```

### Pytest

```bash
make test              # Chạy tất cả tests
make test-unit         # Unit tests
make test-integration  # Integration tests
make test-cov          # Coverage report
```

---

## 3. Kiến trúc hệ thống

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  A1          │   │  A2          │   │  A3          │   │  A4          │
│  Ingestion   │──▶│  Feature     │──▶│  Scoring     │──▶│  Report      │
│              │   │  Engineer    │   │  (LightGBM)  │   │  Generator   │
│ • PDF parse  │   │ • Semantic   │   │ • PD predict │   │ • 5C assess  │
│ • CIC API    │   │   extraction │   │ • PD → Score │   │ • Debt Anlst │
│ • Bank stmt  │   │ • Imputation │   │ • SHAP       │   │ • Reward Mdl │
│ • Internal DB│   │ • 753 feats  │   │ • Decision   │   │ • PDF render │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
       │                 │                  │                  │
       │         ┌───────┴──────────────────┴──────────────────┘
       │         │
       ▼         ▼
┌─────────────────────────┐     ┌──────────────────┐
│  FastAPI Service         │     │  Frontend (HTML)  │
│  /score/mock             │◀────│  Dashboard UI     │
│  /v1/report/{id}/pdf     │     │  PDF Viewer       │
└─────────────────────────┘     └──────────────────┘
```

**3-Tier Explainability:**
1. **SHAP Attribution** — Feature-level importance từ LightGBM
2. **Grounded LLM Narrative** — Gemini viết nhận xét, chỉ trích dẫn SHAP factors
3. **Audit Trail** — Timestamp, model version, consistency check cho mỗi step

---

## 4. Cấu trúc thư mục

```
swinburn_new/
├── creditlens/                    # Core application package
│   ├── agents/                   # 4 pipeline agents
│   │   ├── a1_ingestion/         # Data ingestion
│   │   │   ├── agent.py          # IngestionAgent — main orchestrator
│   │   │   ├── document_parser.py # PDF → structured fields (PyMuPDF)
│   │   │   ├── cic_service.py    # CIC API client (mock JSON)
│   │   │   └── internal_db_reader.py # Internal DB → DataFrames
│   │   ├── a2_feature_engineer/  # Feature engineering
│   │   │   ├── agent.py          # FeatureEngineerAgent — orchestrator
│   │   │   ├── semantic_extractor.py # LLM semantic extraction
│   │   │   ├── imputer.py        # Missing value imputation
│   │   │   └── single_customer_fe.py # 218 raw → 753 ML features
│   │   ├── a3_scoring/           # ML scoring
│   │   │   ├── agent.py          # ScoringAgent — LightGBM + SHAP
│   │   │   ├── model.py          # Model wrapper (load/predict)
│   │   │   ├── score_mapper.py   # PD% → Credit Score (300-850)
│   │   │   └── decision_rules.py # Hard override rules
│   │   └── a4_report_generator/  # Report generation
│   │       ├── agent.py          # ReportGeneratorAgent — LLM narrative
│   │       ├── pdf_generator.py  # PDF rendering (ReportLab)
│   │       └── consistency_validator.py # SHAP-narrative consistency
│   ├── api/
│   │   └── main.py               # FastAPI app + endpoints
│   ├── config/
│   │   ├── feature_config.py     # Feature→5C mapping, risk bands, labels
│   │   ├── feature_source_schema.json # 122-field schema
│   │   ├── prompts.py            # LLM prompt templates
│   │   └── settings.py           # App settings
│   ├── services/
│   │   └── llm_service.py        # Gemini API wrapper
│   ├── orchestrator/
│   │   └── graph.py              # LangGraph orchestration (optional)
│   └── state/                    # State management
├── models/                       # Trained model artifacts
│   ├── lgbm_ref_v1.pkl           # LightGBM model (46MB)
│   ├── fe_stats.pkl              # Feature engineering statistics
│   ├── feature_names.json        # 753 feature names
│   └── feature_importance.csv    # SHAP feature importance
├── training/                     # Model training code
│   ├── train_pipeline.py         # Main training script
│   ├── feature_engineering.py    # Full FE pipeline (218→753)
│   └── precompute_fe_stats.py    # Pre-compute FE stats for inference
├── data/
│   └── mock/                     # 4 demo customers
│       ├── customer_001/         # TARGET=0 (good), Score ~694
│       ├── customer_002/         # TARGET=0 (good), Score ~700+
│       ├── customer_003/         # TARGET=1 (bad), Score ~400s
│       └── customer_004/         # TARGET=1 (bad), Score ~300s
├── front-end/app/                # Frontend dashboard (HTML/JS)
├── test_pipeline.py              # Single customer test
├── test_all_customers.py         # 4-customer comparison test
├── requirements.txt
├── Makefile
└── .env.example
```

---

## 5. Pipeline chi tiết: A1 → A2 → A3 → A4

### 5.1 A1 — Data Ingestion Agent

**File**: `creditlens/agents/a1_ingestion/agent.py`
**Class**: `IngestionAgent`

#### Nhiệm vụ
Thu thập và chuẩn hóa dữ liệu từ 4 kênh khác nhau thành format tương thích Home Credit dataset.

#### Input

Thư mục khách hàng chứa:

| File | Kênh | Mô tả |
|---|---|---|
| `01_cccd.pdf` | PDF → OCR | Căn cước công dân |
| `02_hop_dong_lao_dong.pdf` | PDF → OCR | Hợp đồng lao động |
| `03_so_ho_khau.pdf` | PDF → OCR | Sổ hộ khẩu |
| `04_tham_dinh_nha_o.pdf` | PDF → OCR | Phiếu thẩm định nhà ở |
| `05_don_vay.pdf` | PDF → OCR | Đơn đề nghị vay vốn |
| `07_cic_api_response.json` | CIC API | Bureau records + EXT_SOURCE scores |
| `08_internal_db.json` | Internal DB | Lịch sử vay nội bộ |
| `application_row.json` | **Fast-path** | Dữ liệu gốc 122 cột (bỏ qua OCR) |

#### Output — `dict`

```python
{
    "application_id": str,          # ID duy nhất cho hồ sơ
    "application_row": dict,        # 122 cột matching application_train
    "bureau_df": pd.DataFrame,      # Bureau records (matching bureau.csv)
    "bureau_balance_df": DataFrame, # Monthly bureau status
    "previous_application_df": DataFrame,  # Lịch sử đơn vay trước
    "pos_cash_df": DataFrame,       # POS cash balance
    "installments_df": DataFrame,   # Installments payments
    "credit_card_df": DataFrame,    # Credit card balance
    "confidence_map": dict,         # Confidence per extracted field
    "identity_consistency_flag": str, # "OK" | "MISMATCH" | "MISSING"
    "thin_file_flag": bool,         # True nếu không có lịch sử CIC
    "raw_texts": dict,              # OCR text từ từng document
    "audit_trail": list[dict],      # Audit entries
}
```

#### Logic xử lý

1. **Fast-path check**: Nếu `application_row.json` tồn tại → load trực tiếp 122 cột, bỏ qua OCR (100% coverage, 10x nhanh hơn)
2. **PDF Parsing** (khi không có fast-path): PyMuPDF extract text → regex + rule-based parsing → map sang Home Credit columns
3. **CIC API**: Load `07_cic_api_response.json` → extract `EXT_SOURCE_1/2/3`, bureau records, thin_file_flag
4. **Internal DB**: Load `08_internal_db.json` → convert to DataFrames (previous_application, POS, installments, credit_card)
5. **Cross-validation**: So sánh tên trên các tài liệu → `identity_consistency_flag`

#### Các cột quan trọng trong `application_row`

| Cột | Ý nghĩa | Nguồn |
|---|---|---|
| `AMT_INCOME_TOTAL` | Thu nhập hàng năm | HĐLĐ / application_row.json |
| `AMT_CREDIT` | Tổng số tiền vay | Đơn vay |
| `AMT_ANNUITY` | Tổng trả nợ hàng năm | Đơn vay |
| `AMT_GOODS_PRICE` | Giá trị hàng hóa/TSBĐ | Đơn vay |
| `EXT_SOURCE_1/2/3` | Điểm CIC bên ngoài (0–1) | CIC API |
| `DAYS_BIRTH` | Số ngày từ sinh đến nay (âm) | CCCD |
| `DAYS_EMPLOYED` | Số ngày làm việc (âm) | HĐLĐ |
| `NAME_CONTRACT_TYPE` | Loại hợp đồng | Đơn vay |

---

### 5.2 A2 — Feature Engineer Agent

**File**: `creditlens/agents/a2_feature_engineer/agent.py`
**Class**: `FeatureEngineerAgent`

#### Nhiệm vụ
Chuyển đổi 218 cột raw data từ A1 thành 753 features cho ML model, kết hợp semantic extraction từ LLM.

#### Input — `a1_output: dict`

Output từ A1 (xem mục 5.1).

#### Output — `dict`

```python
{
    "feature_vector": pd.Series,   # 753 ML features (float)
    "application_row": dict,       # Pass-through từ A1
    "llm_feats": {                 # Semantic features từ LLM
        "loan_purpose_category": str,      # "CONSUMPTION"|"PRODUCTION"|"INVESTMENT"|"REFINANCING"|"UNCLEAR"
        "positive_signals": list[str],     # Tín hiệu tích cực
        "risk_flags": list[str],           # Cảnh báo rủi ro
        "income_stability_index": float,   # Chỉ số ổn định thu nhập (0-1)
        "inflow_outflow_ratio": float,     # Tỷ lệ thu/chi
        "collateral_type": str,            # Loại TSBĐ
        "collateral_value_vnd": float,     # Giá trị TSBĐ
        "thin_file_flag": bool,            # Thin-file flag
        "income_imputed_flag": int,        # 1 nếu thu nhập được impute
        "imputation_confidence": float,    # Confidence (0-1)
    },
    "imputation_log": list[dict],  # Chi tiết imputation
    "warnings": list[str],         # Cảnh báo
    "audit_trail": list[dict],
}
```

#### Logic xử lý

1. **Semantic Extraction** (LLM):
   - Gộp OCR text từ `raw_texts` → gửi Gemini AI → extract `loan_purpose_category`, `positive_signals`, `risk_flags`
   - **Fallback**: Nếu LLM trả UNCLEAR → suy ra từ `NAME_CONTRACT_TYPE`:
     - `"Revolving loans"` / `"Cash loans"` → `CONSUMPTION`
     - `NAME_INCOME_TYPE = "Businessman"` → `PRODUCTION`
     - `NAME_INCOME_TYPE = "Commercial associate"` → `INVESTMENT`

2. **Feature Engineering** (deterministic):
   - `SingleCustomerFE` class reuse logic từ `training/feature_engineering.py`
   - **218 raw columns → 753 ML features** qua:
     - Bureau aggregations (mean, max, min, count per credit type)
     - Previous application aggregations
     - POS/installment/credit card balance features
     - One-hot encoding các categorical columns
     - Derived ratios (income/credit, annuity/income, etc.)
   - Sử dụng `fe_stats.pkl` (pre-computed từ training data) cho imputation và encoding

---

### 5.3 A3 — ML Scoring Agent

**File**: `creditlens/agents/a3_scoring/agent.py`
**Class**: `ScoringAgent`

#### Nhiệm vụ
Chấm điểm tín dụng bằng LightGBM, tạo SHAP explanation, áp dụng decision rules.

#### Input — `a2_output: dict`

Output từ A2 (xem mục 5.2).

#### Output — `dict`

```python
{
    "credit_score": int,           # 300–850
    "pd_pct": float,               # Xác suất vỡ nợ (%)
    "pd_prob": float,              # PD probability (0–1)
    "risk_band": str,              # "AAA"|"AA"|"A"|"BBB"|"BB"|"B"|"CCC"|"CC"
    "shap_values": {               # SHAP explanation
        "credit_score": int,
        "pd_prob": float,
        "risk_band": str,
        "top_positive_factors": [   # Top 10 yếu tố tăng rủi ro
            {
                "feature": str,         # Tên feature
                "shap_value": float,    # SHAP value
                "value": float,         # Giá trị feature thực tế
                "label_vi": str,        # Nhãn tiếng Việt
                "dimension_5c": str,    # character|capacity|capital|conditions|collateral
                "direction": str,       # positive_for_default
            }, ...
        ],
        "top_negative_factors": [...],  # Top 10 yếu tố giảm rủi ro
        "five_c_shap_allocation": {     # Phân bổ SHAP theo 5C
            "character":  {"shap_sum": float, "pct": int},
            "capacity":   {"shap_sum": float, "pct": int},
            "capital":    {"shap_sum": float, "pct": int},
            "conditions": {"shap_sum": float, "pct": int},
            "collateral": {"shap_sum": float, "pct": int},
        },
        "model_version": str,
        "inference_timestamp": str,     # ISO 8601
    },
    "routing": str,                # "APPROVE"|"REVIEW"|"REJECT"
    "decision_details": dict,      # Chi tiết decision rules
    "features_df": pd.DataFrame,   # Feature vector đã dùng
    "audit_trail": list[dict],
}
```

#### Logic xử lý

1. **Build feature DataFrame**: Align 753 features từ A2 với `model.feature_names`. Missing features = 0.0.
2. **LightGBM predict**: `model.predict_proba(features_df)` → PD probability (0–1)
3. **Score Mapping** (piecewise linear trong log-PD space):

   | PD% | Credit Score | Risk Band |
   |---|---|---|
   | ≤ 0.5% | 850 | AAA |
   | 2% | 720 | AA |
   | 8% | 640 | A |
   | 18% | 560 | BBB |
   | 35% | 460 | BB/B |
   | ≥ 100% | 300 | CC |

4. **SHAP TreeExplainer**: Tính SHAP values cho mỗi feature → sort by `|shap_value|` → top 10 positive + top 10 negative. Phân bổ vào 5C dimensions theo `feature_config.py`.
5. **Hard Override Rules**: Kiểm tra EXT_SOURCE thấp, income quá nhỏ, thin-file → có thể override quyết định.

---

### 5.4 A4 — Report Generator Agent ⭐

**File**: `creditlens/agents/a4_report_generator/agent.py`
**Class**: `ReportGeneratorAgent`

#### Nhiệm vụ
Tạo báo cáo tín dụng 5C đầy đủ bằng tiếng Việt (6 phần), bao gồm Debt Analyst và Reward Modeler.

#### Input

```python
def generate(
    a3_output: dict,      # Output từ A3 (credit_score, shap_values, pd_pct, ...)
    a2_output: dict,      # Output từ A2 (llm_feats, warnings, ...)
    a1_output: dict,      # Output từ A1 (application_row với AMT_* fields)
) -> dict
```

#### Output — `dict`

```python
{
    "credit_score": int,
    "pd_pct": float,
    "risk_band": str,
    "five_c_scores": {             # Điểm 5C (tối đa 120)
        "character": int,          # 0–30
        "capacity": int,           # 0–40
        "capital": int,            # 0–20
        "conditions": int,         # 0–10
        "collateral": int,         # 0–20
    },
    "narrative": dict,             # LLM-generated narrative (raw)
    "consistency_check": {         # Kiểm tra tính nhất quán
        "passed": bool,
        "fabricated_features": list,  # Features LLM bịa ra (nếu có)
    },
    "final_report": {              # BÁO CÁO CHÍNH — 6 PHẦN
        # ─── Section I ───
        "customer_info": {"summary": str},

        # ─── Section II ───
        "executive_summary": {
            "credit_score": int,
            "risk_band": str,
            "pd_pct": float,
            "recommendation": str,   # "APPROVE"|"REVIEW"|"REJECT"
            "five_c_total": int,     # /120
            "five_c_scores": dict,
            "five_c_shap_allocation": dict,
            "model_info": {
                "model_version": str,
                "auc": str,
                "shap_verified": bool,
                "inference_timestamp": str,
            },
            "financial_ratios": dict,  # DTI, DSCR, LTV ...
        },

        # ─── Section III: Đánh giá 5C ───
        "five_c_scorecard": {
            "<dim>_assessment": {  # cho mỗi dim: character, capacity, capital, conditions, collateral
                "score": int,
                "status": str,        # "DAT"|"XEM_XET"|"KHONG_DAT"
                "shap_pct": int,      # % SHAP contribution
                "indicators_met": list[str],
                "indicators_review": list[str],
                "narrative": str,     # 100-150 chữ
            }, ...
        },

        # ─── Section IV: Tài chính + Debt Analyst ───
        "financial_summary": {
            "income_analysis": str,
            "debt_analysis": str,
            "key_ratios": {"dti": str, "dscr": str, "ltv": str},
        },
        "debt_assessment": {           # ⭐ DEBT ANALYST
            "score": int,              # 0–100
            "max_score": 100,
            "score_pct": str,          # e.g. "75%"
            "overall_status": str,     # "ĐẠT"|"XEM_XET"|"KHONG_DAT"
            "overall_color": str,      # "green"|"orange"|"red"
            "metrics": [               # Bảng chi tiết
                {
                    "name": str,       # "DTI (Nợ/Thu nhập)"
                    "value": str,      # "14.3%"
                    "threshold": str,  # "< 40%"
                    "status": str,     # "Tốt"
                    "flag": str,       # "OK" | "!!"
                }, ...                 # DTI(40pts) + DSCR(35pts) + LTV(15pts) + Mục đích vay(10pts)
            ],
            "summary": str,           # Tóm tắt narrative
        },

        # ─── Section V: TSBĐ ───
        "collateral_detail": dict,

        # ─── Section VI: Khuyến nghị + Reward Modeler ───
        "suggested_terms": {
            "requested_amount_vnd": float,
            "max_amount_vnd": float,
            "requested_term_months": int,
            "interest_rate_suggestion": str,
            "conditions": list[str],
            "dti_at_approval": str,
        },
        "reward_assessment": {          # ⭐ REWARD MODELER
            "interest_rate_pct": str,   # "9.5%" (theo risk band)
            "loan_amount_fmt": str,     # "900 triệu VND"
            "term_months": int,
            "gross_income_fmt": str,    # Thu nhập lãi ước tính
            "expected_loss_fmt": str,   # PD × LGD(45%)
            "risk_adj_income_fmt": str, # Gross - Expected Loss
            "raroc_pct": str,          # RAROC = risk_adj / loan_amount
            "verdict": str,            # "Tốt"|"Chấp nhận"|"Thấp"|"Không khả thi"
            "verdict_flag": str,       # "OK"|"!!"
            "verdict_color": str,      # "green"|"orange"|"red"
            "customer_segment": str,   # "Premium"|"Mid-tier"|"Mass"|"Sub-prime"
            "upsell_opportunities": list[str],
            "summary": str,
        },
        "llm_insights": dict,
        "caveats": list[str],
        "audit_reference": dict,
    },
    "warnings": list[str],
    "audit_trail": list[dict],
}
```

#### Logic xử lý chi tiết

##### 1. Tính Financial Ratios (`_compute_financial_ratios`)

Từ `application_row` (A1 output):

```
monthly_income = AMT_INCOME_TOTAL / 12
monthly_annuity = AMT_ANNUITY / 12     # AMT_ANNUITY là ANNUAL, chia 12

DTI = monthly_annuity / monthly_income
DSCR = monthly_income / monthly_annuity
LTV = AMT_CREDIT / AMT_GOODS_PRICE
```

##### 2. Generate LLM Narrative (`_generate_narrative`)

- **Real mode**: Gửi SHAP values + financial context + collateral context cho Gemini → nhận JSON 5C assessment
- **Mock mode**: Logic deterministic dựa trên `credit_score`:
  - ≥ 700: Character 28/30, Capacity 35/40, ... → APPROVE
  - ≥ 600: Giảm tương ứng → REVIEW
  - ≥ 460: → CONDITIONAL
  - < 460: → REJECT

##### 3. Consistency Validation (`consistency_validator.py`)

Kiểm tra LLM narrative chỉ trích dẫn features có trong SHAP output. Nếu phát hiện feature bịa đặt → `fabricated_features` list, `passed = false`.

##### 4. Debt Analyst (`_compute_debt_assessment`)

**100% deterministic, không dùng LLM.**

| Metric | Điểm tối đa | Scoring |
|---|---|---|
| DTI (Nợ/Thu nhập) | 40 pts | < 30%: 40, < 40%: 30, < 50%: 15, ≥ 50%: 0 |
| DSCR (Dòng tiền/Nợ) | 35 pts | ≥ 1.5: 35, ≥ 1.2: 25, ≥ 1.0: 10, < 1.0: 0 |
| LTV (Vay/TSBĐ) | 15 pts | < 70%: 15, < 80%: 8, ≥ 80%: 0 |
| Mục đích vay | 10 pts | PRODUCTION: 10, INVESTMENT: 8, CONSUMPTION: 6, REFINANCING: 4, UNCLEAR: 0 |

**Overall:**
- ≥ 70%: ĐẠT (green)
- ≥ 45%: XEM_XET (orange)
- < 45%: KHONG_DAT (red)

##### 5. Reward Modeler (`_compute_reward_assessment`)

**100% deterministic, không dùng LLM.**

```
interest_rate = f(risk_band)         # AAA:8.5%, AA:9.5%, A:11%, BBB:13%, ...
term_months = AMT_CREDIT / AMT_ANNUITY (clamped 6–360)
gross_income = loan_amount × interest_rate × (term/12)
expected_loss = loan_amount × PD × LGD(45%)
RAROC = (gross_income - expected_loss) / loan_amount
```

| RAROC | Verdict | Color |
|---|---|---|
| ≥ 8% | Tốt | green |
| ≥ 4% | Chấp nhận được | orange |
| > 0% | Thấp | orange |
| ≤ 0% | Không khả thi | red |

**Customer Segments:**

| Credit Score | Segment | Upsell |
|---|---|---|
| ≥ 720 | Premium | Bảo hiểm nhân thọ, Thẻ vàng, Quỹ tiết kiệm |
| ≥ 640 | Mid-tier | Bảo hiểm tài sản, Thẻ cơ bản |
| ≥ 560 | Mass | Bảo hiểm khoản vay |
| < 560 | Sub-prime | — |

##### 6. PDF Generation (`pdf_generator.py`)

Render `final_report` thành PDF báo cáo tín dụng chuyên nghiệp (6 phần), sử dụng ReportLab:

| Section | Nội dung | Dữ liệu nguồn |
|---|---|---|
| I | Thông tin khách hàng | `customer_info` |
| II | Tóm tắt đánh giá + Scorecard | `executive_summary`, 5C scores, model info |
| III | Đánh giá 5C chi tiết | `five_c_scorecard` (5 sub-sections) |
| IV | Tài chính & Phân tích nợ | `financial_summary` + `debt_assessment` |
| V | Tài sản bảo đảm | `collateral_detail` |
| VI | Khuyến nghị & Điều kiện | `reward_assessment` + `suggested_terms` + `caveats` |

---

## 6. API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/score/mock` | Score demo customer (001–004) |
| `POST` | `/score/customer-folder` | Score from folder path |
| `POST` | `/score/upload` | Score from uploaded files |
| `POST` | `/score/json` | Score from raw JSON |
| `GET` | `/v1/report/{id}/pdf` | PDF report (generate/cached) |

**Scoring response** (`ScoringResult`):

```json
{
  "credit_score": 694,
  "pd_probability": 3.13,
  "risk_band": "AA",
  "decision": "REVIEW",
  "shap_top_positive": [...],
  "shap_top_negative": [...],
  "five_c_scores": {"character": 25, "capacity": 28, ...},
  "five_c_total": 90,
  "recommendation": "REVIEW",
  "consistency_check": true,
  "audit_trail": [...],
  "warnings": [...]
}
```

---

## 7. Cấu hình

### Environment Variables (`.env`)

| Biến | Default | Mô tả |
|---|---|---|
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `USE_MOCK` | `true` | `true` = mock LLM, `false` = real Gemini |
| `MODEL_PATH` | `models/lgbm_ref_v1.pkl` | Đường dẫn model file |
| `FE_STATS_PATH` | `models/fe_stats.pkl` | Feature engineering stats |

### Feature Config (`creditlens/config/feature_config.py`)

- **122 features** với label tiếng Việt (`get_label_vi()`)
- **5C dimension mapping** cho mỗi feature (`get_5c_dimension()`)
- **Risk band definitions** (AAA → CC) với score ranges và auto_decision

---

## 8. Training Model

### Pre-compute FE Statistics

```bash
python training/precompute_fe_stats.py --data-dir home-credit-default-risk/
```

Tạo `models/fe_stats.pkl` chứa imputation values + encoding mappings.

### Train LightGBM

```bash
python training/train_pipeline.py --data-dir home-credit-default-risk/
```

Pipeline:
1. Load 7 bảng từ Home Credit dataset
2. Feature engineering: 218 raw → 753 features
3. NoxMoon + Downsampling strategy
4. LightGBM training với 5-fold CV
5. Save model → `models/lgbm_ref_v1.pkl`

**Metrics**: AUC ~0.803 trên validation set.

---

## Demo Customers

| ID | SK_ID_CURR | TARGET | Score | Band | Profile |
|---|---|---|---|---|---|
| 001 | 418735 | 0 (pass) | ~694 | AA | Pensioner, Revolving loan, EXT high |
| 002 | 394570 | 0 (pass) | ~700+ | AA | Working, Cash loan, EXT high |
| 003 | 272483 | 1 (fail) | ~400s | B/CCC | Low EXT scores |
| 004 | 169206 | 1 (fail) | ~300s | CC | Lowest EXT scores |
