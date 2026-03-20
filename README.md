# 🔍 CreditLens AI

**Credit Scoring & Creditworthiness Assessment for Underbanked Customers & Micro-SMEs**

A multi-agent AI system that combines **LightGBM** (deterministic ML scoring), **SHAP** (mathematical explainability), and **Claude LLM** (semantic feature engineering & report generation) — orchestrated by **LangGraph** — to deliver transparent, auditable credit decisions.

---

## 🎯 Problem Statement

Over **1.4 billion adults** worldwide lack access to formal credit. Traditional scoring models reject thin-file customers outright due to missing credit history. CreditLens addresses this by:

- Scoring customers with **zero credit bureau history** using alternative data (bank transactions, salary patterns)
- Providing **grounded explainability** — every narrative statement is mathematically backed by SHAP values
- Maintaining an **immutable audit trail** for regulatory compliance

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                       │
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐ │
│  │    A1     │    │     A2     │    │    A3    │    │   A4    │ │
│  │ Ingestion │───▶│ LLM Feature│───▶│   ML    │───▶│ Report  │ │
│  │  Agent    │    │  Engineer  │    │ Scoring  │    │Generator│ │
│  └──────────┘    └────────────┘    └──────────┘    └─────────┘ │
│       │               │                │                │       │
│   Textract        Claude 3.5       LightGBM +        Claude +  │
│   CIC API        Bedrock           SHAP             RAG        │
│   Bank CSV       (A2-A, A2-B)    TreeExplainer     4C Report   │
└─────────────────────────────────────────────────────────────────┘
```

### 4-Agent Pipeline

| Agent | Role | Technology | Output |
|-------|------|-----------|--------|
| **A1** — Data Ingestion | 3-channel data collection (OCR, CIC, bank statement) | AWS Textract, CIC API, CSV parser | `structured_feats`, `confidence_map` |
| **A2** — LLM Feature Engineer | Semantic extraction (A) + Intelligent imputation (B) | Claude 3.5 Sonnet (Bedrock) | `llm_feats`, `imputation_log` |
| **A3** — ML Scoring Engine | Deterministic credit scoring + explainability | LightGBM, SHAP TreeExplainer | `credit_score`, `shap_values`, `risk_band` |
| **A4** — Report Generator | 4C credit narrative grounded in SHAP | Claude + RAG (OpenSearch) | `final_report`, `consistency_check` |

### Key Design Principles

- **Grounded Explainability**: LLM narratives are constrained to only reference factors present in SHAP output, preventing hallucination
- **Confidence-Gated Routing**: Pipeline halts on critical field failures ( < 85% confidence), escalates on overall low confidence ( < 65%)
- **Thin-File Innovation**: Customers without CIC history are scored via alternative data with reweighted features

## 📁 Project Structure

```
swinburn_new/
├── creditlens/                        # Core Python package
│   ├── state/credit_state.py          # LangGraph state schema (TypedDict)
│   ├── config/
│   │   ├── settings.py                # Environment config (Pydantic)
│   │   ├── feature_config.py          # Feature tiers, 4C mapping, thresholds
│   │   └── prompts.py                 # LLM prompt templates (A2, A4)
│   ├── data/
│   │   ├── loader.py                  # Home Credit 8-table loader
│   │   ├── feature_engineering.py     # 25-feature production vector
│   │   └── preprocessing.py           # Split, clean, ADASYN
│   ├── agents/
│   │   ├── a1_ingestion/              # Textract + CIC + bank statement
│   │   ├── a2_feature_engineer/       # Semantic extraction + imputation
│   │   ├── a3_scoring/                # LightGBM + SHAP + score mapper
│   │   └── a4_report_generator/       # 4C report + consistency validator
│   └── orchestrator/
│       ├── confidence_gate.py         # HALT / PROCEED / ESCALATE router
│       └── graph.py                   # LangGraph StateGraph (6 nodes)
├── training/
│   └── train.py                       # Full training pipeline + Optuna
├── api/
│   ├── main.py                        # FastAPI entry point
│   └── routes/score.py                # POST /v1/score
├── tests/unit/                        # Unit test suites
├── home-credit-default-risk/          # Dataset (8 CSV tables)
├── requirements.txt
├── pyproject.toml
├── Makefile
└── .env.example
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone & install
cd swinburn_new
pip install -r requirements.txt

# Or using Make
make install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials (or use mock mode)
```

### 3. Train the Model

```bash
# Basic training
python training/train.py --data-dir home-credit-default-risk/

# With Optuna hyperparameter tuning
python training/train.py --data-dir home-credit-default-risk/ --tune --n-trials 50

# Pilot features only (10 core features)
python training/train.py --data-dir home-credit-default-risk/ --feature-set pilot
```

### 4. Run Tests

```bash
make test
# Or directly:
pytest tests/ -v
```

### 5. Start API Server

```bash
make api
# Or directly:
uvicorn api.main:app --reload --port 8000
```

### 6. Score an Application

```bash
curl -X POST http://localhost:8000/v1/score \
  -F "applicant_id=TEST001" \
  -F "customer_type=INDIVIDUAL" \
  -F "bank_statement=@bank_statement.csv"
```

## 📊 Dataset

Uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset:

| Table | Rows | Key Features |
|-------|------|-------------|
| `application_train` | 307,511 | Demographics, EXT_SOURCE, TARGET |
| `bureau` | 1.7M | Credit bureau history |
| `bureau_balance` | 27.3M | Monthly bureau balances |
| `installments_payments` | 13.6M | Payment consistency |
| `credit_card_balance` | 3.8M | Credit utilization |
| `POS_CASH_balance` | 10M | Days past due |
| `previous_application` | 1.67M | Loan purpose history |

**Class imbalance**: ~8% default rate → handled via ADASYN oversampling (target 20%)

## 🧮 Credit Score Mapping

| Risk Band | Score Range | PD Range | Auto Decision |
|-----------|-----------|----------|---------------|
| **AAA** | 720–850 | 0–2% | Auto Approve |
| **AA** | 640–719 | 2–8% | Approve + Review |
| **A** | 560–639 | 8–18% | Conditional |
| **BBB** | 460–559 | 18–35% | Manual Underwrite |
| **CC** | 300–459 | 35–100% | Decline |

## 🛡️ Hard Override Rules

These policy rules **always override** ML decisions:

1. **CIC Nhóm 4-5** → `REJECT` (regardless of ML score)
2. **Loan > 10 tỷ VND** → `ESCALATE` to head office
3. **Overall confidence < 65%** → `HUMAN REVIEW`
4. **Thin-file + score < 560** → Require additional collateral

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | LightGBM + SHAP TreeExplainer |
| LLM | Claude 3.5 Sonnet (AWS Bedrock) |
| Orchestration | LangGraph (StateGraph) |
| API | FastAPI |
| OCR | AWS Textract |
| Search/RAG | Amazon OpenSearch |
| Storage | DynamoDB + S3 |
| Oversampling | ADASYN (imbalanced-learn) |
| HP Tuning | Optuna |

## 📈 Target Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| AUC-ROC | ≥ 0.77 | Discrimination power |
| Gini | ≥ 0.54 | Model quality |
| SHAP coverage | ≥ 85% | Explainability grounding |
| Consistency pass | ≥ 95% | Anti-hallucination |
| Latency (P95) | ≤ 8s | User experience |

## 📄 API Reference

### `POST /v1/score`

Score a credit application.

**Request** (multipart/form-data):
- `applicant_id` (string) — Unique identifier
- `customer_type` (string) — `INDIVIDUAL` or `SME`
- `bank_statement` (file) — 6-month bank statement CSV
- `documents` (files) — PDF documents (CCCD, contracts)

**Response**:
```json
{
  "application_id": "abc123",
  "credit_score": 672,
  "pd_pct": 5.8,
  "risk_band": "AA",
  "recommendation": "APPROVE_REVIEW",
  "overall_confidence": 0.82,
  "four_c_scores": {
    "character": 28,
    "capacity": 31,
    "capital": 16,
    "conditions": 9
  },
  "warnings": [],
  "report": { ... }
}
```

### `GET /health`

Health check endpoint.

## 📝 License

Academic project — Swinburne University of Technology.

## 👥 Authors

CreditLens AI Team — Applied Project 2025.
