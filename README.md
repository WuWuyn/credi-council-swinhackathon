# Credicouncil AI — Hệ thống Đánh giá Tín dụng AI

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
- [9. Evaluation — Đánh giá Model](#9-evaluation--đánh-giá-model)
- [10. Demo Customers](#10-demo-customers)

---

## 1. Cài đặt & Chạy

### Yêu cầu hệ thống

- Python 3.10+
- Conda (khuyến nghị)
- ~2GB RAM cho model LightGBM
- NVIDIA GPU (tùy chọn, cho Docling OCR nhanh hơn)

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
| LLM | `google-genai`, `pydantic` | Gemini structured extraction |
| OCR | `docling`, `easyocr`, `PyMuPDF` | PDF text extraction (smart dual-mode) |
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
USE_OCR=true                               # true = parse PDFs, false = read JSON
USE_DOCLING=true                           # true = Docling+LLM, false = PyMuPDF+regex
DOCLING_DEVICE=cpu                         # cpu | cuda | mps
MODEL_PATH=models/lgbm_ref_v1.pkl         # Đường dẫn model đã train
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_RAG_MODEL=gemini-3.1-flash-lite-preview
FILE_SEARCH_STORE_NAME=your_gemini_api_key_here
```

### Bước 4: Khởi tạo RAG Policy Store (chỉ chạy 1 lần)

```bash
cd back-end
python policy_docs/init_policy_store.py
```

Script sẽ upload các tài liệu chính sách ngân hàng (TT39, QĐ493, QĐ18, Basel...) lên Gemini FileSearchStore. Store name tự động thêm vào `.env`.

### Test pipeline đơn lẻ (1 customer)

```bash
conda activate swinburn_hackathon
cd back-end
python test_pipeline.py
```

### Bước 5: Chạy server

```bash
conda activate swinburn_hackathon
cd back-end
uvicorn credicouncil.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập:
- **Dashboard**: http://localhost:8000/app

---

## 3. Kiến trúc hệ thống

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────┐   ┌──────────────────┐
│  A1              │   │  A2              │   │  A3          │   │  A4               │
│  Ingestion       │──▶│  Feature         │──▶│  Scoring     │──▶│  Report           │
│                  │   │  Engineer        │   │  (LightGBM)  │   │  Generator        │
│ • Smart OCR      │   │ • Semantic LLM   │   │ • PD predict │   │ • 5C assessment   │
│   (PyMuPDF/      │   │   extraction     │   │ • PD → Score │   │ • RAG policy cite │
│    Docling)      │   │ • Pydantic valid. │   │ • SHAP       │   │ • Debt Analyst    │
│ • LLM extraction │   │ • Imputation     │   │ • Decision   │   │ • Reward Modeler  │
│ • CIC API        │   │ • 753 features   │   │              │   │ • PDF render      │
│ • Internal DB    │   │                  │   │              │   │                   │
└─────────────────┘   └─────────────────┘   └─────────────┘   └──────────────────┘
       │                      │                   │                     │
       │              ┌───────┴───────────────────┴─────────────────────┘
       │              │
       ▼              ▼
┌─────────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  FastAPI Service         │     │  Frontend (HTML)  │     │  Gemini RAG      │
│  /score/mock             │◀────│  Dashboard UI     │     │  FileSearchStore │
│  /v1/report/{id}/pdf     │     │  PDF Viewer       │     │  (Policy docs)   │
└─────────────────────────┘     └──────────────────┘     └──────────────────┘
```

### Core Technologies

| Component | Technology | Mục đích |
|---|---|---|
| **OCR** | PyMuPDF + Docling + EasyOCR | Smart dual-mode text extraction |
| **LLM Extraction** | Gemini + Pydantic `response_schema` | Type-safe field extraction |
| **ML Scoring** | LightGBM (5-fold bagging) | Probability of Default |
| **Explainability** | SHAP TreeExplainer | Feature-level attribution |
| **RAG** | Gemini FileSearchStore | Policy citation (TT39, QĐ493, Basel...) |
| **Report** | Gemini + ReportLab | 5C narrative + PDF rendering |

### 3-Tier Explainability

1. **SHAP Attribution** — Feature-level importance từ LightGBM
2. **Grounded LLM Narrative** — Gemini viết nhận xét, chỉ trích dẫn SHAP factors + policy từ RAG
3. **Audit Trail** — Timestamp, model version, consistency check cho mỗi step

---

## 4. Cấu trúc thư mục

```
swinburn_new/
├── back-end/                          # Main application
│   ├── credicouncil/                    # Core application package
│   │   ├── agents/                   # 4 pipeline agents
│   │   │   ├── a1_ingestion/         # Data ingestion
│   │   │   │   ├── agent.py          # IngestionAgent — main orchestrator
│   │   │   │   ├── llm_field_extractor.py # Gemini+Pydantic field extraction
│   │   │   │   ├── document_parser.py # PDF → structured fields (regex fallback)
│   │   │   │   ├── cic_service.py    # CIC API client (mock JSON)
│   │   │   │   └── internal_db_reader.py # Internal DB → DataFrames
│   │   │   ├── a2_feature_engineer/  # Feature engineering
│   │   │   │   ├── agent.py          # FeatureEngineerAgent — orchestrator
│   │   │   │   ├── semantic_extractor.py # LLM semantic extraction (Pydantic)
│   │   │   │   ├── imputer.py        # Missing value imputation
│   │   │   │   └── single_customer_fe.py # 218 raw → 753 ML features
│   │   │   ├── a3_scoring/           # ML scoring
│   │   │   │   ├── agent.py          # ScoringAgent — LightGBM + SHAP
│   │   │   │   ├── model.py          # Model wrapper (load/predict)
│   │   │   │   ├── score_mapper.py   # PD% → Credit Score (300-850)
│   │   │   │   └── decision_rules.py # Hard override rules
│   │   │   └── a4_report_generator/  # Report generation
│   │   │       ├── agent.py          # ReportGeneratorAgent — 5C + RAG
│   │   │       ├── pdf_generator.py  # PDF rendering (ReportLab)
│   │   │       └── consistency_validator.py # SHAP-narrative consistency
│   │   ├── api/
│   │   │   └── main.py               # FastAPI app + endpoints
│   │   ├── config/
│   │   │   ├── feature_config.py     # Feature→5C mapping, risk bands
│   │   │   ├── feature_source_schema.json # 122-field schema
│   │   │   ├── prompts.py            # LLM prompt templates
│   │   │   └── settings.py           # App settings (USE_DOCLING, etc.)
│   │   ├── schemas/
│   │   │   └── document_schemas.py   # Pydantic schemas (5 doc types)
│   │   └── services/
│   │       ├── llm_service.py        # Unified Gemini API (JSON/text/structured)
│   │       ├── docling_ocr_service.py # Smart OCR (PyMuPDF→Docling fallback)
│   │       └── policy_rag_service.py # RAG FileSearchStore service
│   ├── policy_docs/                  # Vietnamese banking policy documents
│   │   ├── tt39_2016_cho_vay.md      # TT39/2016 — Quy định cho vay
│   │   ├── qd493_2005_phan_loai_no.md # QĐ493 — Phân loại nợ
│   │   ├── qd18_2007_xep_hang_tin_dung.md # QĐ18 — Xếp hạng tín dụng
│   │   ├── basel_vietnam_car.md      # Basel III — CAR ratio
│   │   └── init_policy_store.py      # Script khởi tạo Gemini RAG store
│   ├── models/                       # Trained model artifacts
│   │   ├── lgbm_ref_v1.pkl           # LightGBM model (46MB)
│   │   ├── fe_stats.pkl              # Feature engineering statistics
│   │   └── feature_names.json        # 753 feature names
│   ├── training/                     # Model training code
│   │   ├── train_pipeline.ipynb         # Main training script
│   │   ├── feature_engineering.py    # Full FE pipeline (218→753)
│   │   └── precompute_fe_stats.py    # Pre-compute FE stats for inference
│   ├── data/
│   │   └── mock/                     # 50 demo customers
│   │       ├── customer_001/ ... customer_050/
│   │       ├── customer_map.json     # SK_ID → dir mapping
│   │       └── extract_real_customers.py # Generate demo data from dataset
│   ├── tests/
│   │   └── unit/                     # Unit tests
│   │       ├── test_docling_coverage.py # Docling+LLM vs ground truth
│   │       └── test_ocr_coverage.py  # PyMuPDF+regex vs ground truth
│   ├── test_pipeline.py              # Single customer pipeline test
│   └── test_all_customers.py         # Multi-customer comparison test
├── front-end/app/                    # Frontend dashboard (HTML/JS)
├── .env.example                      # Environment template
└── requirements.txt
```

---

## 5. Pipeline chi tiết: A1 → A2 → A3 → A4

### 5.1 A1 — Data Ingestion Agent

**File**: `credicouncil/agents/a1_ingestion/agent.py`
**Class**: `IngestionAgent`

#### Nhiệm vụ
Thu thập và chuẩn hóa dữ liệu từ 4 kênh khác nhau thành format tương thích Home Credit dataset.

#### Dual-Mode OCR Engine

A1 hỗ trợ 2 extraction engine, điều khiển bởi `USE_DOCLING`:

| | Path A: Docling+LLM (`USE_DOCLING=true`) | Path B: PyMuPDF+regex (`USE_DOCLING=false`) |
|---|---|---|
| **Text extraction** | Smart 3-tier: PyMuPDF → Docling+EasyOCR → fallback | PyMuPDF only |
| **Field extraction** | Gemini + Pydantic `response_schema` | Regex + rule-based |
| **Accuracy** | 93/121 fields (76.9%) | 93/121 fields (76.9%) |
| **Speed** (text-layer PDF) | 0.12s + ~20s LLM = ~20s | ~0.5s |
| **Speed** (scanned PDF) | ~10s OCR + ~20s LLM = ~30s | N/A (regex chỉ hỗ trợ text) |
| **Ưu điểm** | Robust, hỗ trợ PDF scan, type-safe | Nhanh, không cần API |

##### Smart 3-Tier OCR (`DoclingOCRService`)

```
PDF input
   ↓
Tier 1: PyMuPDF (0.003s) — text layer extraction
   ↓ (nếu text ≥ 50 chars → thành công, bỏ qua Tier 2)
Tier 2: Docling + EasyOCR (5-10s) — cho scanned/image PDFs
   ↓ (nếu thất bại)
Tier 3: PyMuPDF fallback
```

- **`DOCLING_DEVICE`**: `cpu` | `cuda` | `mps` — chọn thiết bị tính toán cho Docling layout model, TableFormer, và EasyOCR
- Docling chỉ lazy-load khi thực sự cần (scanned PDF) → không tốn RAM khởi tạo

##### LLM Field Extraction (`LLMFieldExtractor`)

Sử dụng `LLMService.generate_structured()` — shared Gemini client:

```
OCR text → Gemini API (response_schema=PydanticModel) → Pydantic validation → typed dict
```

5 Pydantic schemas cho 5 loại tài liệu:
- `CCCDSchema` — Căn cước công dân (10 fields)
- `EmploymentSchema` — Hợp đồng lao động (14 fields)
- `HouseholdSchema` — Sổ hộ khẩu (4 fields)
- `HousingSurveySchema` — Phiếu thẩm định nhà ở (62 fields)
- `LoanApplicationSchema` — Đơn vay (40 fields)

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
    "raw_texts": dict,              # OCR text từ từng document (dùng cho A2)
    "audit_trail": list[dict],      # Audit entries
}
```

---

### 5.2 A2 — Feature Engineer Agent

**File**: `credicouncil/agents/a2_feature_engineer/agent.py`
**Class**: `FeatureEngineerAgent`

#### Nhiệm vụ
Chuyển đổi 218 cột raw data từ A1 thành 753 features cho ML model, kết hợp semantic extraction từ LLM.

#### Semantic Extraction (`SemanticExtractor`)

Sử dụng `LLMService.generate_structured()` với Pydantic schema `SemanticFeatures`:

```python
class SemanticFeatures(BaseModel):
    loan_purpose_category: str      # PRODUCTION|CONSUMPTION|INVESTMENT|REFINANCING|UNCLEAR
    repayment_plan_quality: str     # DETAILED|GENERAL|VAGUE|NONE
    stated_income_consistency: bool # Income matches employment docs?
    risk_flags: list[str]           # Risk indicators
    positive_signals: list[str]     # Positive indicators
    extraction_confidence: float    # 0.0-1.0
```

**Smart text source selection:**
- Khi `raw_texts` có sẵn (từ OCR) → dùng trực tiếp, không build lại
- Khi `raw_texts` rỗng (fast-path) → tự build summary từ `application_row`

#### Output — `dict`

```python
{
    "feature_vector": pd.Series,   # 753 ML features (float)
    "application_row": dict,       # Pass-through từ A1
    "llm_feats": {                 # Semantic features từ LLM
        "loan_purpose_category": str,
        "loan_purpose_category_encoded": int,
        "repayment_plan_quality": str,
        "repayment_plan_quality_encoded": int,
        "stated_income_consistency": bool,
        "risk_flags": list[str],
        "risk_flag_count": int,
        "positive_signals": list[str],
        "extraction_confidence": float,
    },
    "imputation_log": list[dict],
    "warnings": list[str],
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

**File**: `credicouncil/agents/a3_scoring/agent.py`
**Class**: `ScoringAgent`

#### Nhiệm vụ
Chấm điểm tín dụng bằng LightGBM, tạo SHAP explanation, áp dụng decision rules.

#### Output — `dict`

```python
{
    "credit_score": int,           # 300–850
    "pd_pct": float,               # Xác suất vỡ nợ (%)
    "pd_prob": float,              # PD probability (0–1)
    "risk_band": str,              # "AAA"|"AA"|"A"|"BBB"|"BB"|"B"|"CCC"|"CC"
    "shap_values": {               # SHAP explanation
        "top_positive_factors": [...],  # Top 10 yếu tố tăng rủi ro
        "top_negative_factors": [...],  # Top 10 yếu tố giảm rủi ro
        "five_c_shap_allocation": {...}, # Phân bổ SHAP theo 5C
    },
    "routing": str,                # "APPROVE"|"REVIEW"|"REJECT"
    "decision_details": dict,
    "features_df": pd.DataFrame,
    "audit_trail": list[dict],
}
```

#### Score Mapping (piecewise linear trong log-PD space)

| PD% | Credit Score | Risk Band |
|---|---|---|
| ≤ 0.5% | 850 | AAA |
| 2% | 720 | AA |
| 8% | 640 | A |
| 18% | 560 | BBB |
| 35% | 460 | BB/B |
| ≥ 100% | 300 | CC |

---

### 5.4 A4 — Report Generator Agent ⭐

**File**: `credicouncil/agents/a4_report_generator/agent.py`
**Class**: `ReportGeneratorAgent`

#### Nhiệm vụ
Tạo báo cáo tín dụng 5C đầy đủ bằng tiếng Việt, với RAG policy citation.

#### RAG Pipeline (Policy Documents)

A4 sử dụng **Gemini FileSearchStore** để trích dẫn quy định ngân hàng Việt Nam:

```
SHAP context + Financial ratios
        ↓
PolicyRAGService.query() → Gemini FileSearchStore
        ↓
Policy excerpts + citations (TT39, QĐ493, QĐ18, Basel...)
        ↓
LLM narrative + grounded policy references
```

**Policy documents:**
- `tt39_2016_cho_vay.md` — TT39/2016 NHNN: Quy định cho vay
- `qd493_2005_phan_loai_no.md` — QĐ493: Phân loại nợ
- `qd18_2007_xep_hang_tin_dung.md` — QĐ18: Xếp hạng tín dụng
- `basel_vietnam_car.md` — Basel III: Tỷ lệ an toàn vốn (CAR)

#### Report Output — 6 Sections

| Section | Nội dung | Dữ liệu nguồn |
|---|---|---|
| I | Thông tin khách hàng | `customer_info` |
| II | Tóm tắt đánh giá + Scorecard | `executive_summary`, 5C scores, model info |
| III | Đánh giá 5C chi tiết | `five_c_scorecard` (5 sub-sections) |
| IV | Tài chính & Phân tích nợ | `financial_summary` + `debt_assessment` |
| V | Tài sản bảo đảm | `collateral_detail` |
| VI | Khuyến nghị & Điều kiện | `reward_assessment` + `suggested_terms` + `caveats` |

#### Debt Analyst (100% deterministic)

| Metric | Điểm tối đa | Scoring |
|---|---|---|
| DTI (Nợ/Thu nhập) | 40 pts | < 30%: 40, < 40%: 30, < 50%: 15, ≥ 50%: 0 |
| DSCR (Dòng tiền/Nợ) | 35 pts | ≥ 1.5: 35, ≥ 1.2: 25, ≥ 1.0: 10, < 1.0: 0 |
| LTV (Vay/TSBĐ) | 15 pts | < 70%: 15, < 80%: 8, ≥ 80%: 0 |
| Mục đích vay | 10 pts | PRODUCTION: 10, INVESTMENT: 8, CONSUMPTION: 6, REFINANCING: 4, UNCLEAR: 0 |

#### Reward Modeler (100% deterministic)

```
interest_rate = f(risk_band)         # AAA:8.5%, AA:9.5%, A:11%, BBB:13%, ...
term_months = AMT_CREDIT / AMT_ANNUITY (clamped 6–360)
gross_income = loan_amount × interest_rate × (term/12)
expected_loss = loan_amount × PD × LGD(45%)
RAROC = (gross_income - expected_loss) / loan_amount
```

---

## 6. API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/score/mock` | Score demo customer (001–050) |
| `POST` | `/score/customer-folder` | Score from folder path |
| `POST` | `/score/upload` | Score from uploaded files |
| `POST` | `/score/json` | Score from raw JSON |
| `GET` | `/v1/report/{id}/pdf` | PDF report (generate/cached) |

---

## 7. Cấu hình

### Environment Variables (`.env`)

| Biến | Default | Mô tả |
|---|---|---|
| `GEMINI_API_KEY` | — | Google AI Studio API key (bắt buộc) |
| `USE_OCR` | `true` | `true` = parse PDFs, `false` = read JSON directly |
| `USE_DOCLING` | `true` | `true` = Docling+LLM extraction, `false` = PyMuPDF+regex |
| `DOCLING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` — device cho AI models |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model cho field + semantic extraction |
| `GEMINI_RAG_MODEL` | `gemini-2.5-flash` | Gemini model cho RAG policy query |
| `MODEL_PATH` | `models/lgbm_ref_v1.pkl` | Đường dẫn model file |
| `FILE_SEARCH_STORE_NAME` | — | Gemini FileSearchStore name (từ init script) |

### LLM Service Architecture

```
LLMService (shared Gemini client, auto-loads .env)
  ├── generate_json()        → A4 report, free-form JSON
  ├── generate_text()        → A4 narrative text
  └── generate_structured()  → A1 field extraction + A2 semantic (Pydantic)
        ├── LLMFieldExtractor (5 document schemas)
        └── SemanticExtractor (SemanticFeatures schema)
```

### Feature Config (`credicouncil/config/feature_config.py`)

- **122 features** với label tiếng Việt (`get_label_vi()`)
- **5C dimension mapping** cho mỗi feature (`get_5c_dimension()`)
- **Risk band definitions** (AAA → CC) với score ranges và auto_decision

---

## 8. Training Model

```bash
conda activate swinburn_hackathon
cd back-end/training
```

### Pre-compute FE Statistics

```bash
python precompute_fe_stats.py --data-dir ../home-credit-default-risk/
```

Tạo `models/fe_stats.pkl` chứa imputation values + encoding mappings.

### Train LightGBM (Notebook)

Mở `train_pipeline.ipynb` bằng Jupyter và chạy toàn bộ cells:

```bash
jupyter notebook train_pipeline.ipynb
```

> **Lưu ý**: Trong notebook, `data_dir` mặc định là `../home-credit-default-risk/`, `output_dir` là `../models/`. Chỉnh trực tiếp trong cell cuối nếu cần.

Pipeline:
1. Load 7 bảng từ Home Credit dataset
2. Feature engineering: 218 raw → 753 features
3. NoxMoon + Downsampling strategy
4. LightGBM training với 5-fold CV
5. Save model → `models/lgbm_ref_v1.pkl`

**Metrics**: AUC ~0.803 trên validation set.

---

## 9. Evaluation — Đánh giá Model

Module `evaluation/` cung cấp pipeline đánh giá toàn diện cho ML Core (A3 — LightGBM) trên tập Home Credit dataset.

### Cấu trúc thư mục

```
back-end/evaluation/
├── evaluate.py          # Main runner — orchestrate toàn bộ evaluation
├── metrics.py           # Tính AUC-ROC, Gini, KS, PR-AUC, per-band breakdown
├── plots.py             # Vẽ ROC, PR, Score Distribution, Calibration
├── shap_analysis.py     # SHAP feature importance, beeswarm, 5C allocation
└── results/             # Output: JSON, CSV, PNG
```

### Cách chạy

> **Yêu cầu**: Cần có thư mục Home Credit dataset (chứa các file CSV: `application_train.csv`, `bureau.csv`, v.v.).

```bash
conda activate swinburn_hackathon
cd back-end/evaluation
```

#### Evaluate model đã train (nhanh, không SHAP)

```bash
python evaluate.py --data-dir ../home-credit-default-risk/ --model-path ../models/lgbm_ref_v1.pkl --no-shap
```

#### Evaluate model đã train (đầy đủ, có SHAP analysis)

```bash
python evaluate.py --data-dir ../home-credit-default-risk/ --model-path ../models/lgbm_ref_v1.pkl
```

#### Train model mới rồi evaluate

```bash
python evaluate.py --data-dir ../home-credit-default-risk/ --train --no-shap
```

#### Giới hạn sample test set (chạy nhanh hơn)

```bash
python evaluate.py --data-dir ../home-credit-default-risk/ --model-path ../models/lgbm_ref_v1.pkl --no-shap --sample 10000
```

### CLI Arguments

| Argument | Bắt buộc | Default | Mô tả |
|---|---|---|---|
| `--data-dir` | ✅ | — | Thư mục chứa Home Credit CSV files |
| `--model-path` | ❌ | `None` | Path tới model `.pkl` đã train |
| `--train` | ❌ | `false` | Train model mới trước khi evaluate |
| `--no-shap` | ❌ | `false` | Bỏ qua SHAP analysis (nhanh hơn đáng kể) |
| `--output-dir` | ❌ | `evaluation/results` | Thư mục lưu kết quả |
| `--sample` | ❌ | `None` | Giới hạn số sample trong test set |
| `--test-size` | ❌ | `0.20` | Tỷ lệ train/test split |

> **Lưu ý**: Phải cung cấp `--model-path` hoặc `--train` (một trong hai).

### Pipeline evaluation (6 bước)

1. **Build feature matrix** — Load 7 bảng Home Credit, tạo 753 features
2. **Stratified split** — 80/20 train/test (stratified theo TARGET)
3. **Load/Train model** — Load model `.pkl` hoặc train mới
4. **Inference** — Chạy predict trên test set, map PD → Credit Score → Risk Band
5. **Compute metrics** — AUC-ROC, Gini, KS, PR-AUC, per risk-band breakdown + vẽ plots
6. **SHAP analysis** (optional) — Feature importance, beeswarm plot, 5C allocation

### Output Files

Kết quả được lưu trong `evaluation/results/`:

| File | Mô tả |
|---|---|
| `metrics_summary.json` | Tất cả metrics dạng JSON (AUC, Gini, KS, classification report...) |
| `metrics_summary.csv` | Summary dạng bảng, dễ đọc |
| `riskband_breakdown.csv` | Metrics per risk band (AAA/AA/A/BBB/CC) |
| `classification_report.json` | Precision, Recall, F1, Confusion Matrix |
| `roc_curve.png` | ROC Curve với AUC annotation + KS point |
| `pr_curve.png` | Precision-Recall Curve |
| `score_distribution.png` | Phân phối credit score theo class + pie chart risk band |
| `calibration_plot.png` | Calibration plot (predicted vs actual PD) |
| `shap_feature_importance.csv` | Global feature importance (mean \|SHAP\|) |
| `shap_feature_importance.png` | Bar chart top 20 features |
| `shap_beeswarm.png` | SHAP beeswarm summary plot |
| `shap_5c_allocation.csv` | SHAP allocation theo 5C dimensions |
| `shap_5c_allocation.png` | Pie chart SHAP contribution per 5C |

### Metrics giải thích

| Metric | Target | Ý nghĩa |
|---|---|---|
| **AUC-ROC** | ≥ 0.77 | Khả năng phân biệt default vs non-default |
| **Gini** | ≥ 0.54 | 2×AUC − 1, đo discriminatory power |
| **KS Statistic** | ≥ 0.35 | Max separation giữa CDF default và non-default |
| **PR-AUC** | — | Average Precision, quan trọng với imbalanced data |
| **SHAP Coverage** | ≥ 85% | % tổng SHAP được cover bởi top 5 features |

---

## 10. Demo Customers

**50 demo customers** (25 pass + 25 fail) được tạo từ Home Credit dataset thật:

```bash
cd back-end
python data/mock/extract_real_customers.py
```

Mỗi customer folder chứa:
- `01_cccd.pdf` — Căn cước công dân
- `02_hop_dong_lao_dong.pdf` — Hợp đồng lao động
- `03_so_ho_khau.pdf` — Sổ hộ khẩu
- `04_tham_dinh_nha_o.pdf` — Phiếu thẩm định nhà ở
- `05_don_vay.pdf` — Đơn đề nghị vay vốn
- `07_cic_api_response.json` — CIC credit bureau response
- `08_internal_db.json` — Internal loan history
- `application_row.json` — Ground truth (122 columns)

Selection strategy: Diverse PD score spread (from lowest to highest risk) across both TARGET groups.
