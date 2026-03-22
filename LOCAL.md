# CreditLens — Local Architecture

> Phiên bản local sử dụng PyMuPDF + Gemini API thay cho AWS Textract + Bedrock Claude.
> Cập nhật: 2026-03-22

---

## Tổng quan hệ thống

CreditLens là hệ thống AI chấm điểm tín dụng cá nhân, gồm **4 Agent** chạy tuần tự:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│    A1    │───▶│    A2    │───▶│    A3    │───▶│    A4    │
│ Ingestion│    │ Feature  │    │   ML     │    │  Report  │
│  4 kênh  │    │ Engineer │    │ Scoring  │    │ Generator│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
  PDF/CSV/JSON    LLM+FE         LightGBM        LLM+5C
                                  +SHAP          Narrative
```

**Output cuối cùng**: `credit_report.json` — báo cáo tín dụng 5C tiếng Việt.

---

## Cấu trúc thư mục

```
creditlens/
├── agents/
│   ├── a1_ingestion/           # Agent thu thập dữ liệu
│   │   ├── agent.py            # Orchestrator 4 kênh
│   │   ├── document_parser.py  # PyMuPDF – đọc PDF tiếng Việt
│   │   ├── cic_service.py      # Mock CIC API (JSON)
│   │   ├── bank_statement_parser.py  # Parse CSV sao kê
│   │   ├── internal_db_reader.py     # Đọc lịch sử nội bộ (JSON)
│   │   └── textract_service.py       # [Unused] AWS Textract placeholder
│   │
│   ├── a2_feature_engineer/    # Agent tiền xử lý features
│   │   ├── agent.py            # Orchestrator: semantic + FE + imputation
│   │   ├── semantic_extractor.py  # LLM trích xuất ngữ nghĩa từ OCR text
│   │   ├── single_customer_fe.py  # Feature engineering (753 features)
│   │   ├── imputer.py             # Imputation fields thiếu
│   │   └── thin_file_handler.py   # Xử lý hồ sơ thiếu dữ liệu
│   │
│   ├── a3_scoring/             # Agent chấm điểm ML
│   │   ├── agent.py            # Orchestrator: model → SHAP → decision
│   │   ├── model.py            # Load LightGBM BaggingClassifier
│   │   ├── shap_explainer.py   # SHAP TreeExplainer + 5C allocation
│   │   ├── score_mapper.py     # PD → Credit Score → Risk Band
│   │   └── decision_rules.py   # Routing logic (APPROVE/REVIEW/REJECT)
│   │
│   └── a4_report_generator/    # Agent sinh báo cáo
│       ├── agent.py            # LLM → 5C narrative + 6-section report
│       └── consistency_validator.py  # Kiểm tra narrative ↔ SHAP
│
├── api/main.py                 # FastAPI REST endpoint
├── config/
│   ├── feature_config.py       # 5C mapping, SHAP labels tiếng Việt
│   ├── prompts.py              # System/user prompts cho LLM
│   └── settings.py             # App config từ .env
├── orchestrator/graph.py       # LangGraph workflow
├── services/llm_service.py     # Gemini 2.5 Flash Lite client
├── state/credit_state.py       # TypedDict schema (CreditState, 5C types)
└── data/loader.py              # Data utilities
```

---

## Chi tiết từng module

### Agent A1 — Data Ingestion

**Mục đích**: Thu thập dữ liệu thô từ 4 kênh, chuyển sang định dạng Home Credit dataset.

| Kênh | File | Input | Output |
|------|------|-------|--------|
| **1. PDF Documents** | `document_parser.py` | 5 file PDF tiếng Việt | `application_row` (121 fields), `raw_texts` |
| **2. CIC API** | `cic_service.py` | `07_cic_api_response.json` | `bureau_df`, `EXT_SOURCE_1/2/3`, `thin_file_flag` |
| **3. Bank Statement** | `bank_statement_parser.py` | `06_sao_ke_ngan_hang.csv` | 8 behavioral features |
| **4. Internal DB** | `internal_db_reader.py` | `08_internal_db.json` | `previous_application_df`, `pos_cash_df`, `installments_df`, `credit_card_df` |

**Kênh 1 — PDF OCR** (đang hoạt động thật):

Sử dụng **PyMuPDF (fitz)** đọc text từ PDF. Các file PDF được sinh bởi `data/mock/generate_mock_data.py` với text embedded (không phải scan ảnh), nên PyMuPDF extract được trực tiếp:

| File PDF | Loại tài liệu | Fields trích xuất |
|----------|---------------|-------------------|
| `01_cccd.pdf` | Căn cước công dân | `full_name`, `gender`, `date_of_birth`, `id_number` |
| `02_hop_dong_lao_dong.pdf` | Hợp đồng lao động | `employer_name`, `base_salary`, `position`, `start_date` |
| `03_so_ho_khau.pdf` | Sổ hộ khẩu | `family_members_count`, `children_count`, `marital_status` |
| `04_tham_dinh_nha_o.pdf` | Thẩm định nhà ở | `living_area`, `year_built`, `apartment_quality`, `has_elevator` |
| `05_don_vay.pdf` | Đơn vay | `loan_amount`, `monthly_payment`, `contract_type`, `purpose` |

**Kênh 3 — Bank Statement** (đang hoạt động thật):

Parse CSV sao kê ngân hàng, trích xuất 8 features hành vi:

| Feature | Ý nghĩa |
|---------|---------|
| `avg_monthly_inflow_vnd` | Dòng tiền vào trung bình/tháng |
| `avg_monthly_outflow_vnd` | Dòng tiền ra trung bình/tháng |
| `inflow_outflow_ratio` | Tỷ lệ thu/chi |
| `salary_pattern_detected` | Có phát hiện lương đều? |
| `income_stability_index` | Chỉ số ổn định thu nhập (0-1) |
| `n_unique_counterparties` | Số đối tác giao dịch |
| `debt_service_behavior` | Hành vi trả nợ (`ON_TIME`/`LATE`/`NONE`) |
| `avg_monthly_balance_vnd` | Số dư trung bình/tháng |

---

### Agent A2 — Feature Engineering

**Mục đích**: Chuyển dữ liệu thô → 753 features cho ML model.

```
OCR text ──► [Semantic Extractor (LLM)] ──► loan_purpose, positive_signals, risk_flags
                                              │
application_row ──► [Single Customer FE] ──► 753 features (match training schema)
                         │
missing fields ──► [Imputer] ──► fill median/mode values
```

| Sub-module | File | Chức năng |
|-----------|------|-----------|
| **Semantic Extractor** | `semantic_extractor.py` | Gemini phân tích OCR text → `loan_purpose_category`, `positive_signals`, `risk_flags`, `thin_file_flag` |
| **Single Customer FE** | `single_customer_fe.py` | Feature engineering: aggregate bureau, previous apps, installments, credit card → đúng 753 features khớp model schema |
| **Imputer** | `imputer.py` | Fill missing fields bằng median/mode từ training data |
| **Thin File Handler** | `thin_file_handler.py` | Xử lý hồ sơ thiếu CIC data |

---

### Agent A3 — ML Scoring

**Mục đích**: Chấm điểm tín dụng bằng LightGBM + giải thích bằng SHAP.

```
753 features ──► [LightGBM BaggingClassifier x5] ──► PD probability
                         │
                    [Score Mapper] ──► Credit Score (300-850) + Risk Band
                         │
                   [SHAP Explainer] ──► Top 10 positive/negative factors
                         │                + 5C SHAP allocation
                   [Decision Rules] ──► APPROVE / APPROVE_REVIEW / REJECT
```

| Sub-module | File | Chức năng |
|-----------|------|-----------|
| **Model** | `model.py` | Load `lgbm_ref_v1.pkl` — 5-fold BaggingClassifier, 753 features |
| **Score Mapper** | `score_mapper.py` | PD → Credit Score (logistic mapping 300-850) → Risk Band (AAA→D) |
| **SHAP Explainer** | `shap_explainer.py` | TreeExplainer → top factors với `label_vi`, `value`, `dimension_5c`, `five_c_shap_allocation` |
| **Decision Rules** | `decision_rules.py` | PD thresholds: <7% APPROVE, 7-20% REVIEW, 20-35% CONDITIONAL, >35% REJECT |

**Output schema** (deterministic, không LLM):
```json
{
  "credit_score": 687,
  "pd_pct": 11.60,
  "risk_band": "AA",
  "routing": "APPROVE_REVIEW",
  "shap_values": {
    "top_positive_factors": [...],
    "top_negative_factors": [...],
    "five_c_shap_allocation": {
      "character": {"shap_sum": 0.86, "pct": 11},
      "capacity":  {"shap_sum": 0.81, "pct": 10},
      "capital":   {"shap_sum": 0.23, "pct": 3},
      "conditions": {"shap_sum": 5.71, "pct": 74},
      "collateral": {"shap_sum": 0.11, "pct": 1}
    }
  }
}
```

---

### Agent A4 — Report Generator

**Mục đích**: Sinh báo cáo tín dụng 5C tiếng Việt, grounded bởi SHAP values.

```
A3 output (SHAP) ──►┐
                     ├──► [Gemini LLM] ──► 5C narrative JSON
A2 output (context) ─┘         │
                          [Consistency Validator] ──► SHAP grounding check
                               │
                          [Score Clamper] ──► Enforce 5C bounds
                               │
                          [Report Builder] ──► 9-key final_report
```

**5C Score Bounds** (clamped tự động):

| Dimension | Thang điểm | Ý nghĩa |
|-----------|:---------:|---------|
| Character | 0 – 30 | Uy tín tín dụng, lịch sử thanh toán |
| Capacity | 0 – 40 | Năng lực trả nợ, thu nhập, DTI |
| Capital | 0 – 20 | Vốn tự có, tài sản ròng |
| Conditions | 0 – 10 | Mục đích vay, điều kiện thị trường |
| Collateral | 0 – 20 | Tài sản bảo đảm |
| **Tổng** | **0 – 120** | |

**Final report** gồm 9 keys:
1. `customer_info` — Tóm tắt hồ sơ
2. `executive_summary` — Credit Score + 5C + Recommendation
3. `five_c_scorecard` — Đánh giá chi tiết 5 chiều
4. `financial_summary` — Phân tích tài chính
5. `collateral_detail` — Chi tiết TSBĐ
6. `suggested_terms` — Đề xuất điều khoản
7. `llm_insights` — Nhận định từ LLM
8. `caveats` — Lưu ý và rủi ro
9. `audit_reference` — Model version, timestamp

**Consistency Validator**: Kiểm tra narrative phải reference ≥1 SHAP label cho mỗi dimension → đảm bảo LLM không bịa thông tin.

---

### Services & Infrastructure

| Module | File | Chức năng |
|--------|------|-----------|
| **LLM Service** | `services/llm_service.py` | Gemini 2.5 Flash Lite client, JSON generation + markdown stripping |
| **Feature Config** | `config/feature_config.py` | `FEATURE_TO_5C_MAPPING`, `SHAP_LABEL_VI` (120+ labels), `get_5c_dimension()`, `get_label_vi()` |
| **Credit State** | `state/credit_state.py` | TypedDict: `CreditState`, `FiveCAllocation`, `FiveCAssessment`, `CreditReport` |
| **API** | `api/main.py` | FastAPI endpoint: `POST /score` → full pipeline |
| **Orchestrator** | `orchestrator/graph.py` | LangGraph DAG: A1→A2→A3→A4 |

---

## 📋 Mock vs Real — Tổng quan các module

| Module | Component | Mock mode | Real mode | Ghi chú |
|--------|-----------|:---------:|:---------:|---------|
| **A1** | PDF OCR | ❌ Không mock — PyMuPDF đọc PDF thật | ← Giống | PDF có text embedded, không cần OCR scan |
| **A1** | CIC API | 📄 Đọc `07_cic_api_response.json` | ← Giống | Luôn dùng mock JSON (chưa có CIC API thật) |
| **A1** | Bank Statement | ❌ Không mock — parse CSV thật | ← Giống | `06_sao_ke_ngan_hang.csv` → 8 features |
| **A1** | Internal DB | 📄 Đọc `08_internal_db.json` | ← Giống | Luôn dùng mock JSON |
| **A2** | Semantic Extraction | 🔸 Trả empty dict `{}` | 🤖 **Gemini** phân tích OCR text | Mock → `purpose=CONSUMPTION` (hardcode). Real → `purpose=UNCLEAR` + 6 signals chi tiết |
| **A2** | Feature Engineering | ❌ Không mock — chạy thật | ← Giống | `single_customer_fe.py` → 753 features |
| **A2** | Imputation | 🔸 Median/mode defaults | ← Giống | Cả mock và real đều dùng statistical imputation |
| **A3** | ML Scoring | ❌ Không mock — LightGBM thật | ← Giống | `lgbm_ref_v1.pkl`, 5-fold bagging |
| **A3** | SHAP | ❌ Không mock — TreeExplainer thật | ← Giống | Deterministic, output giống nhau mỗi lần |
| **A4** | Report Narrative | 🔸 Template tĩnh (hardcode scores) | 🤖 **Gemini** sinh narrative | Mock → consistency FAIL. Real → SHAP-grounded, consistency PASS |
| **A4** | Consistency Check | ✅ Chạy thật | ← Giống | Validate narrative ↔ SHAP labels |

### Ảnh hưởng Mock vs Real đến kết quả

| Metric | Mock | Real (Gemini 2.5 Flash Lite) |
|--------|:----:|:----------------------------:|
| Credit Score | 687 | 687 (giống — ML deterministic) |
| PD% | 11.60% | 11.60% (giống) |
| Risk Band | AA | AA (giống) |
| SHAP values | Giống | Giống (deterministic) |
| 5C Total | 90/120 (hardcode) | 73/120 (LLM đánh giá thận trọng hơn) |
| Positive Signals | 1 generic | 6 chi tiết (company name, income, insurance...) |
| Consistency | ❌ FAIL (2 violations) | ✅ PASS (75% coverage) |
| Report quality | Template, thiếu SHAP reference | Narrative SHAP-grounded tiếng Việt |

> **Kết luận**: ML output (Score/PD/SHAP) không bị ảnh hưởng bởi mock/real — luôn deterministic. Sự khác biệt chỉ ở A2 semantic features và A4 narrative quality.

---

## Dữ liệu Customer

```
data/mock/customer_001/
├── 01_cccd.pdf                  # CCCD — Nguyễn Văn An, 1990-05-15
├── 02_hop_dong_lao_dong.pdf     # HĐLĐ — Công ty ABC, 40M/tháng
├── 03_so_ho_khau.pdf            # Hộ khẩu — 4 thành viên
├── 04_tham_dinh_nha_o.pdf       # Nhà ở — 65m², 7.5/10
├── 05_don_vay.pdf               # Đơn vay — 300M, 36 tháng
├── 06_sao_ke_ngan_hang.csv      # Sao kê — 6 tháng, 80 giao dịch
├── 07_cic_api_response.json     # CIC — 2 bureau records, EXT_SOURCE
├── 08_internal_db.json          # Nội bộ — previous apps, installments
└── credit_report.json           # [OUTPUT] Báo cáo tín dụng 5C
```

---

## Chạy Pipeline

```bash
# Mock mode (không cần API key)
USE_MOCK=true python test_pipeline.py

# Real mode (cần GEMINI_API_KEY trong .env)
USE_MOCK=false python test_pipeline.py

# API server
uvicorn creditlens.api.main:app --reload
```

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| ML Model | LightGBM (BaggingClassifier) | 4.x |
| Explainability | SHAP TreeExplainer | 0.45+ |
| LLM | Google Gemini 2.5 Flash Lite | via `google-generativeai` |
| PDF OCR | PyMuPDF (fitz) | 1.24+ |
| API | FastAPI | 0.110+ |
| Orchestrator | LangGraph | 0.2+ |
| Data | pandas, numpy | — |

---

## ML vs LLM — Phân tách vai trò

```
┌─────────────────────────────────────────────────────┐
│                  DETERMINISTIC (ML)                  │
│  Credit Score, PD%, Risk Band, SHAP values           │
│  → Không thay đổi giữa các lần chạy                 │
│  → Không bị ảnh hưởng bởi LLM                       │
│  → Ground truth cho quyết định tín dụng              │
└──────────────────────┬──────────────────────────────┘
                       │ (read-only input)
                       ▼
┌─────────────────────────────────────────────────────┐
│                 INTERPRETIVE (LLM)                   │
│  5C Scores, Narrative, Positive Signals              │
│  → Có thể dao động giữa các lần chạy                │
│  → Diễn giải SHAP thành ngôn ngữ cán bộ tín dụng   │
│  → Bổ sung, không thay thế ML output                │
└─────────────────────────────────────────────────────┘
```

LLM **chỉ đọc** output ML, **không bao giờ thay đổi** Credit Score, PD, hay SHAP values.
5C scores là layer bổ sung theo framework ngân hàng, được clamp vào thang chuẩn (max 120).
