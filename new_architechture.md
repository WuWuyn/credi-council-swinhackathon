## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                                │
│  ┌──────────────┐   ┌────────────────┐   ┌───────────────────────┐ │
│  │  PDF / Scan  │   │  CIC API JSON  │   │  HC Internal CSV/DB   │ │
│  │  (Đơn vay,   │   │  (bureau.csv   │   │  (prev_app, cc, pos,  │ │
│  │   CCCD, HĐ)  │   │  + EXT_SOURCE) │   │   installments)       │ │
│  └──────┬───────┘   └───────┬────────┘   └──────────┬────────────┘ │
└─────────┼───────────────────┼────────────────────────┼─────────────┘
          │                   │                        │
          ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODULE A1 — DATA INGESTION                       │
│                                                                     │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │ PDF Parser   │   │ CIC JSON Parser  │   │  HC Data Loader      │ │
│  │ (pdfplumber/ │   │ → bureau rows   │   │  → prev, cc, pos,    │ │
│  │  PyPDF2)     │   │ → bureau_balance │   │    installments rows  │ │
│  └──────┬───────┘   └───────┬──────────┘   └──────────┬───────────┘ │
│         │                   │                          │             │
│         ▼                   └──────────────────────────┘             │
│  ┌──────────────────┐                 │                              │
│  │ raw_ocr_text     │         ┌───────▼────────┐                     │
│  │ (noisy strings)  │         │  raw_tables    │                     │
│  └──────┬───────────┘         │  (DataFrames)  │                     │
└─────────┼──────────────────────┼───────────────┼─────────────────────┘
          │                      │               │
          ▼                      │               │
┌─────────────────────────────────┼───────────────┼─────────────────────┐
│              MODULE A2 — LLM FEATURE EXTRACTION                       │
│                                                                       │
│  Input: raw_ocr_text + partial structured fields                      │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Gemini 2.5 Flash Lite                                          │  │
│  │  Prompt: Extract JSON {AMT_INCOME_TOTAL, OCCUPATION_TYPE,      │  │
│  │          ORGANIZATION_TYPE, DAYS_EMPLOYED, CODE_GENDER,        │  │
│  │          NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, ...}        │  │
│  │  → normalize: "15tr" → 15000000, "kế toán" → "Accountants"   │  │
│  │  → impute missing fields with confidence scores                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Output: application_row (dict, all 122 raw fields filled/imputed)   │
│          + imputation_log {field: confidence}                         │
└────────────────────────────────┼───────────────────┬──────────────────┘
                                 │                   │
                                 ▼                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│              MODULE A1 (PART 2) — NoxMoon Feature Engineering          │
│                                                                        │
│  Uses: fe_stats.pkl (pre-computed from training data)                  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  SingleCustomerFE(application_row, bureau_rows, bubl_rows,       │ │
│  │                   prev_rows, cc_rows, pos_rows, inst_rows)       │ │
│  │                                                                  │ │
│  │  fe_stats.pkl provides:                                          │ │
│  │    ✓ inc_by_org          → NEW_INC_BY_ORG (no groupby needed)  │ │
│  │    ✓ group_medians       → gender/car/realty/region/family means│ │
│  │    ✓ factorize_maps      → label encode categoricals            │ │
│  │    ✓ global_scores_std_mean → fillna for NEW_SCORES_STD         │ │
│  │    ✓ mean_encode_maps    → map 9 features to mean target        │ │
│  │    ✓ feature_names       → align to 753-col model input         │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  Output: feature_vector (1 × 755 → align to 753 cols)                │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│              MODULE A3 — ML SCORING ENGINE                             │
│                                                                        │
│  ┌──────────────┐   ┌──────────────────────┐   ┌──────────────────┐  │
│  │  LightGBM    │   │  Credit Score Mapper  │   │  SHAP Explainer  │  │
│  │  lgb1.pkl    │→  │  PD → [300-850]       │→  │  TreeExplainer   │  │
│  │  predict_proba│  │  (log-odds, PDO=20)   │   │  top_factors     │  │
│  └──────────────┘   └──────────────────────┘   └──────────────────┘  │
│                                                                        │
│  Output: {credit_score, pd_pct, risk_band, shap_values, top_factors}  │
└────────────────────────────────────────────────────────────────────────┘