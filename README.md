# CrediCouncil AI — AI Credit Assessment System

> **AI-powered credit scoring pipeline** built on the MASCA architecture (Multi-Agent System for Credit Assessment), using LightGBM + SHAP + Gemini LLM to generate 5C credit reports compliant with Vietnamese banking standards.

---

## Table of Contents

- [1. Installation & Running](#1-installation--running)
- [2. Test & Verify](#2-test--verify)
- [3. System Architecture](#3-system-architecture)
- [4. Directory Structure](#4-directory-structure)
- [5. Detailed Pipeline: A1 → A2 → A3 → A4](#5-detailed-pipeline-a1--a2--a3--a4)

---

## 1. Installation & Running

### Requirements

- Python 3.10+ (Conda recommended)
- Node.js 18+ (for frontend)
- ~2GB RAM for LightGBM model

### Step 1: Create Environment

```bash
conda create -n swinburne_hackathon python=3.10 -y
conda activate swinburne_hackathon
```

### Step 2: Install Dependencies

```bash
cd back-end
pip install -r requirements.txt
```

### Step 3: Configure `.env`

```bash
cd back-end
cp .env.example .env
```

Edit `.env` — replace the placeholder values:

```env
GEMINI_API_KEY=your_gemini_api_key_here            # Google AI Studio API key (required)
DOCLING_DEVICE=cpu                                  # cpu | cuda | mps
MODEL_PATH=models/lgbm_ref_v1.pkl                   # Model path
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_RAG_MODEL=gemini-3.1-flash-lite-preview
FILE_SEARCH_STORE_NAME=your_file_search_store_name   # From Step 4
```

### Step 4: Initialize RAG Policy Store (run once only)

```bash
cd back-end
python policy_docs/init_policy_store.py
```

Uploads banking policy documents (TT39, TT11, QĐ18, Basel...) to the Gemini FileSearchStore. The store name is automatically added to `.env`.

### Step 5: Run Backend

```bash
conda activate swinburne_hackathon
cd back-end
uvicorn credicouncil.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI**: http://localhost:8000/docs

### Step 6: Run Frontend

```bash
cd front-end
npm install
npm run dev
```

- **Dashboard**: http://localhost:5173

---

## 2. Test & Verify

### Single Pipeline Test (1 customer)

```bash
conda activate swinburne_hackathon
cd back-end
python test_pipeline.py
```

### Batch Pipeline Test (5 customers in parallel)

```bash
python test_batch_pipeline.py
```

### Evaluation — A3 Model Assessment

> Requirement: Home Credit dataset directory (`application_train.csv`, `bureau.csv`, etc.) is needed.

```bash
cd back-end
python evaluation/a3_scoring/evaluate.py --data-dir home-credit-default-risk/ --model-path models/lgbm_ref_v1.pkl --no-shap
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | — | Home Credit CSV directory (required) |
| `--model-path` | `None` | Path to model `.pkl` file |
| `--train` | `false` | Train a new model before evaluation |
| `--no-shap` | `false` | Skip SHAP analysis (faster) |
| `--sample` | `None` | Limit test set size |

**Target Metrics**: AUC-ROC ≥ 0.77 · Gini ≥ 0.54 · KS ≥ 0.35

---

## 3. System Architecture

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────┐   ┌──────────────────┐
│  A1              │   │  A2              │   │  A3          │   │  A4               │
│  Ingestion       │──▶│  Feature         │──▶│  Scoring     │──▶│  Report           │
│                  │   │  Engineer        │   │  (LightGBM)  │   │  Generator        │
│ • Docling OCR    │   │ • Semantic LLM   │   │ • PD predict │   │ • 5C assessment   │
│ • LLM extraction │   │ • Pydantic valid │   │ • PD → Score │   │ • RAG policy cite │
│ • CIC Bureau     │   │ • Imputation     │   │ • SHAP       │   │ • Decision Engine │
│ • Internal DB    │   │ • 753 features   │   │ • Decision   │   │ • PDF render      │
└─────────────────┘   └─────────────────┘   └─────────────┘   └──────────────────┘
       │                      │                   │                     │
       └──────────────────────┴───────────────────┴─────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  FastAPI + WebSocket │  │  React + Vite    │  │  Gemini RAG          │
│  /v1/score           │  │  Dashboard UI    │  │  FileSearchStore     │
│  /ws/pipeline        │  │  PDF Viewer      │  │  (Policy docs)       │
└──────────────────────┘  └──────────────────┘  └──────────────────────┘
```

### Core Technologies

| Component | Technology | Purpose |
|---|---|---|
| **OCR** | Docling + EasyOCR | Smart text extraction |
| **LLM** | Gemini + Pydantic `response_schema` | Type-safe field extraction |
| **ML Scoring** | LightGBM (5-fold bagging) | Probability of Default |
| **Explainability** | SHAP TreeExplainer | Feature-level attribution |
| **RAG** | Gemini FileSearchStore | Policy citation (TT39, TT11, Basel...) |
| **Report** | Gemini + ReportLab | 5C narrative + PDF rendering |
| **Frontend** | React + Vite + TailwindCSS | Dashboard & Credit Report UI |

### 3-Tier Explainability

1. **SHAP Attribution** — Feature-level importance from LightGBM
2. **Grounded LLM Narrative** — Gemini generates commentary, citing SHAP factors + policy RAG
3. **Audit Trail** — Timestamp, model version, consistency check for each step

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/v1/score/mock` | Score demo customer (001–005) |
| `POST` | `/v1/score/upload` | Score from uploaded files |
| `POST` | `/v1/score/json` | Score from raw JSON |
| `GET` | `/v1/report/{id}/pdf` | PDF report |
| `GET` | `/v1/customers` | List customers & results |
| `WS` | `/ws/pipeline` | Realtime pipeline progress |

---

## 4. Directory Structure

```
cridi-council-swinhackathon/
├── back-end/
│   ├── credicouncil/                    # Core application package
│   │   ├── agents/
│   │   │   ├── a1_ingestion/            # Data ingestion
│   │   │   │   ├── agent.py             # IngestionAgent
│   │   │   │   ├── llm_field_extractor.py # Gemini+Pydantic extraction
│   │   │   │   ├── cic_service.py       # CIC Bureau API
│   │   │   │   └── internal_db_reader.py
│   │   │   ├── a2_feature_engineer/     # Feature engineering
│   │   │   │   ├── agent.py             # FeatureEngineerAgent
│   │   │   │   ├── semantic_extractor.py # LLM semantic extraction
│   │   │   │   ├── imputer.py           # Missing value imputation
│   │   │   │   └── single_customer_fe.py # 218 raw → 753 ML features
│   │   │   ├── a3_scoring/              # ML scoring
│   │   │   │   ├── agent.py             # ScoringAgent (LightGBM+SHAP)
│   │   │   │   ├── model.py             # Model wrapper
│   │   │   │   ├── score_mapper.py      # PD% → Credit Score (300-850)
│   │   │   │   └── decision_rules.py    # Hard override rules
│   │   │   └── a4_report_generator/     # Report generation
│   │   │       ├── agent.py             # ReportGeneratorAgent (5C+RAG)
│   │   │       ├── five_c_scorer.py     # 5C scoring engine
│   │   │       ├── decision_engine.py   # Band-based decision logic
│   │   │       ├── pdf_generator.py     # PDF rendering (ReportLab)
│   │   │       └── consistency_validator.py
│   │   ├── api/
│   │   │   ├── main.py                  # FastAPI app entry point
│   │   │   ├── config.py               # Settings, paths, .env loading
│   │   │   ├── pipeline.py             # Pipeline execution logic
│   │   │   ├── routes_scoring.py       # Score endpoints
│   │   │   ├── routes_report.py        # Report endpoints
│   │   │   ├── routes_customers.py     # Customer list endpoints
│   │   │   ├── routes_ws.py            # WebSocket pipeline
│   │   │   └── schemas.py              # Request/Response models
│   │   ├── orchestrator/
│   │   │   ├── graph.py                # Pipeline orchestration graph
│   │   │   └── confidence_gate.py      # Confidence-based routing
│   │   ├── config/
│   │   │   ├── feature_config.py       # Feature→5C mapping, labels
│   │   │   ├── feature_source_schema.json # 122-field schema
│   │   │   ├── prompts.py              # LLM prompt templates
│   │   │   └── settings.py             # App settings
│   │   ├── schemas/
│   │   │   └── document_schemas.py     # Pydantic schemas (5 doc types)
│   │   ├── services/
│   │   │   ├── llm_service.py          # Gemini API (JSON/text/structured)
│   │   │   ├── docling_ocr_service.py  # Smart OCR service
│   │   │   └── policy_rag_service.py   # RAG FileSearchStore service
│   │   └── state/
│   │       └── credit_state.py         # Pipeline state management
│   ├── policy_docs/                     # Vietnamese banking policy docs
│   │   ├── tt39_2016_lending.md
│   │   ├── tt11_2021_debt_classification.md
│   │   ├── credit_assessment_5c.md
│   │   ├── cic_scoring_guide.md
│   │   ├── basel_vietnam_car.md
│   │   └── init_policy_store.py
│   ├── models/                          # Trained model artifacts
│   │   ├── lgbm_ref_v1.pkl             # LightGBM model (~46MB)
│   │   ├── fe_stats.pkl                # FE statistics for inference
│   │   └── feature_names.json          # 753 feature names
│   ├── training/                        # Model training
│   │   ├── train_pipeline.ipynb
│   │   ├── train_pipeline.py
│   │   ├── feature_engineering.py       # Full FE pipeline (218→753)
│   │   └── precompute_fe_stats.py
│   ├── evaluation/                      # Model evaluation
│   │   ├── a3_scoring/                  # A3 ML Core evaluation
│   │   │   ├── evaluate.py / metrics.py / plots.py / shap_analysis.py
│   │   │   └── results/
│   │   └── e2e/                         # End-to-end pipeline evaluation
│   ├── data/mock/                       # 5 demo customers
│   │   ├── customer_001/ ... customer_005/
│   │   └── customer_map.json
│   ├── test_pipeline.py                 # Single customer test
│   ├── test_batch_pipeline.py           # Batch test (5 customers)
│   ├── .env.example
│   └── requirements.txt
├── front-end/                           # React + Vite SPA
│   └── src/
│       ├── pages/
│       │   ├── DashboardPage.jsx        # Main dashboard
│       │   └── CreditReportDetailPage.jsx
│       ├── components/
│       │   ├── Header.jsx
│       │   └── ExtractedDataReviewModal.jsx  # Human-in-the-loop review
│       ├── services/apiService.js
│       └── config/api.js
└── README.md
```

---

## 5. Detailed Pipeline: A1 → A2 → A3 → A4

### 5.1 A1 — Data Ingestion Agent

**File**: `credicouncil/agents/a1_ingestion/agent.py`

Collects and normalizes data from 4 channels into a format compatible with the Home Credit dataset.

**Input**: Customer directory containing:

| File | Channel | Description |
|---|---|---|
| `01_cccd.pdf` | PDF → OCR → LLM | National ID Card |
| `02_hop_dong_lao_dong.pdf` | PDF → OCR → LLM | Employment Contract |
| `03_so_ho_khau.pdf` | PDF → OCR → LLM | Household Registration Book |
| `04_tham_dinh_nha_o.pdf` | PDF → OCR → LLM | Housing Appraisal Form |
| `05_don_vay.pdf` | PDF → OCR → LLM | Loan Application Form |
| `07_cic_api_response.json` | CIC API | Bureau records + EXT_SOURCE scores |
| `08_internal_db.json` | Internal DB | Internal loan history |
| `application_row.json` | **Fast-path** | Original 122-column data (skip OCR) |

**OCR Pipeline**: Docling + EasyOCR for text extraction → Gemini + Pydantic `response_schema` for field extraction (5 document schemas: CCCD, Employment, Household, HousingSurvey, LoanApplication).

**Output**:

```python
{
    "application_id": str,
    "application_row": dict,         # 122 columns matching Home Credit format
    "bureau_df": pd.DataFrame,       # Bureau records
    "previous_application_df": DataFrame,
    "confidence_map": dict,          # Confidence per extracted field
    "thin_file_flag": bool,          # True if no CIC history
    "raw_texts": dict,               # OCR text (used by A2)
    "audit_trail": list[dict],
}
```

---

### 5.2 A2 — Feature Engineer Agent

**File**: `credicouncil/agents/a2_feature_engineer/agent.py`

Transforms 218 raw columns from A1 into **753 features** for the ML model, combining semantic extraction from LLM.

**Semantic Extraction** (via `SemanticExtractor`): Sends OCR text → Gemini → extracts `loan_purpose_category`, `positive_signals`, `risk_flags` with Pydantic validation.

**Deterministic FE** (via `SingleCustomerFE`): Bureau aggregations, previous app features, one-hot encoding, derived ratios — mirrors logic from `training/feature_engineering.py`, uses `fe_stats.pkl` for imputation.

**Output**:

```python
{
    "feature_vector": pd.Series,     # 753 ML features
    "application_row": dict,         # Pass-through from A1
    "llm_feats": {                   # Semantic features
        "loan_purpose_category": str,
        "risk_flags": list[str],
        "positive_signals": list[str],
        "extraction_confidence": float,
    },
    "audit_trail": list[dict],
}
```

---

### 5.3 A3 — ML Scoring Agent

**File**: `credicouncil/agents/a3_scoring/agent.py`

Performs credit scoring using LightGBM, generates SHAP explanations, and applies decision rules.

**Score Mapping** (PD → Credit Score 300–850 → Risk Band):

| PD% | Score | Band | Auto Decision |
|---|---|---|---|
| ≤ 0.5% | 850 | AAA | APPROVE |
| 2% | 720 | AA | APPROVE |
| 8% | 640 | A | REVIEW |
| 18% | 560 | BBB | REVIEW |
| 35% | 460 | BB/B | REJECT |
| ≥ 100% | 300 | CC | REJECT |

**Output**:

```python
{
    "credit_score": int,             # 300–850
    "pd_pct": float,                 # Probability of Default (%)
    "risk_band": str,                # "AAA"|"AA"|"A"|"BBB"|"CC"
    "shap_values": {
        "top_positive_factors": [...],     # Increase risk
        "top_negative_factors": [...],     # Decrease risk
        "five_c_shap_allocation": {...},   # SHAP allocation by 5C
    },
    "routing": str,                  # "APPROVE"|"REVIEW"|"REJECT"
    "audit_trail": list[dict],
}
```

---

### 5.4 A4 — Report Generator Agent

**File**: `credicouncil/agents/a4_report_generator/agent.py`

Generates a comprehensive 5C credit report in Vietnamese, combining RAG policy citation + deterministic scoring.

**Sub-components**:

| Module | Logic | Description |
|---|---|---|
| `five_c_scorer.py` | Deterministic | 5C scoring (Character/Capacity/Capital/Conditions/Collateral) |
| `decision_engine.py` | Deterministic | Band-based decision + suggested terms |
| `policy_rag_service.py` | LLM + RAG | Regulatory citation (TT39, TT11, Basel...) |
| `consistency_validator.py` | Deterministic | Validate SHAP ↔ narrative consistency |
| `pdf_generator.py` | Deterministic | ReportLab PDF rendering |

**Report Output — 6 Sections**:

| Section | Content |
|---|---|
| I | Customer Information |
| II | Assessment Summary + Scorecard |
| III | Detailed 5C Assessment |
| IV | Financial & Debt Analysis (DTI, DSCR, LTV) |
| V | Collateral Assets |
| VI | Recommendations & Conditions |
