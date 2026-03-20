
TECHNICAL DESIGN DOCUMENT
CreditLens AI
Credit Scoring & Creditworthiness Assessment
Giải bài toán underbanked & micro SME bằng ML + LLM Explainability


Dataset Home Credit Default Risk (307,511 records, 8 tables)	Core Stack LightGBM + SHAP + Claude + LangGraph + AWS Bedrock	Timeline 4 tuần build Pilot: 40 test cases ≥ 95%
0. Bài Toán & Luận Điểm Giải Pháp
0.1 Problem Statement (BTC)
Ban Tổ Chức:
"Traditional credit scoring models rely heavily on historical financial data and static rules, limiting their ability to accurately assess creditworthiness for underbanked customers, especially micro SMEs, and to adapt to changing borrower behavior. This results in biased decisions, limited assessments, and restricted access to financial services."

0.2 Ba điểm đau — ba giải pháp kỹ thuật
#	Pain Point	Root cause	Giải pháp kỹ thuật
P1	Thin-file exclusion ~70% dân số VN không đủ lịch sử CIC	Scoring chỉ dùng credit bureau — loại bỏ người không có lịch sử vay trước	Alternative Data Pipeline: transaction behavioral features từ sao kê thay thế CIC
P2	Static rules không thích nghi Không bắt được tín hiệu từ hành vi số mới	Scorecard cứng không học từ dữ liệu mới; không xử lý được unstructured docs	LLM Feature Engineering: trích xuất semantic features từ văn bản không có cấu trúc
P3	Black-box, không giải thích Biased decisions, không audit được	AI scoring không trace được decision → vi phạm adverse action notice requirements	Grounded XAI Stack: SHAP → LLM narrative có ràng buộc → audit trail bất biến

0.3 Tại sao Home Credit Default Risk?
Một bộ dataset — không cần mở rộng. Home Credit Default Risk là lựa chọn tốt nhất và duy nhất cần thiết cho giai đoạn này:
•	Đúng population: Home Credit Việt Nam phục vụ chính xác underbanked population — không có lịch sử tín dụng đầy đủ, đúng core của bài toán BTC
•	307,511 records: đủ lớn để train LightGBM ổn định mà không cần augmentation
•	8 bảng quan hệ: bao gồm bureau data, installment payments, POS cash, credit card — phản ánh alternative data signals thực tế
•	Default rate 8%: class imbalance thực tế của credit data — cần ADASYN, nhưng manageable
•	Kaggle top AUC ~0.794: baseline đã được cộng đồng benchmark kỹ — chúng ta có thể so sánh improvement rõ ràng
1. Kiến Trúc Hệ Thống
1.1 Tổng quan — 4 Agent, 3 lớp công nghệ
Hệ thống được tổ chức thành 4 agent chuyên biệt điều phối bởi LangGraph. Nguyên tắc thiết kế: mỗi agent làm đúng việc mà nó giỏi nhất — không agent nào làm thay công việc của agent khác.

	Agent	Input	Processing	Output
A1	Data Ingestion Agent (Tool-calling agent)	PDF/scan, CIC API call, bank statement CSV	Textract OCR → field extraction → transaction aggregation → confidence scoring	structured_features dict + confidence_map
A2	LLM Feature Engineer (LLM agent w/ structured output)	OCR text + structured_features (with missing flags)	Semantic extraction (Variant A) + intelligent imputation (Variant B)	llm_features dict + imputation_flags + field_confidence
A3	ML Scoring Engine (Deterministic — NOT an LLM)	Unified feature vector (A1 + A2 merged)	LightGBM predict_proba → credit score mapping → SHAP TreeExplainer	credit_score, pd_pct, risk_band, shap_values JSON
A4	Report Generator (LLM agent w/ RAG + constraint)	shap_values JSON + RAG policy context + warnings	4C Assessment + grounded narrative + consistency validation	Full credit report (JSON + human-readable PDF)

Nguyên tắc cốt lõi	ML làm scoring (A3) — không bao giờ để LLM làm. LLM làm 2 việc: (1) trích xuất features từ text (A2), và (2) diễn giải SHAP thành narrative (A4). Tính agentic đến từ: mỗi agent tự quyết định tool nào cần gọi, khi nào cần impute vs flag, và orchestrator tự route dựa trên confidence — không phải từ số lượng LLM calls.

1.2 LangGraph State Schema
class CreditState(TypedDict):
    # Core identifiers
    application_id:   str              # SHA-256(applicant_id + timestamp)
    customer_type:    Literal["INDIVIDUAL", "SME"]
    
    # A1 outputs
    raw_ocr_text:     dict[str, str]   # {doc_type: extracted_text}
    structured_feats: dict[str, Any]   # {feature_name: value}
    confidence_map:   dict[str, float] # {feature_name: confidence_0_to_1}
    missing_fields:   list[str]        # critical/important fields below threshold
    
    # A2 outputs
    llm_feats:        dict[str, Any]   # semantic features + imputed values
    imputation_log:   list[dict]       # [{field, method, confidence, source}]
    warnings:         list[str]        # human-readable warning messages
    overall_confidence: float          # weighted mean across all fields
    
    # A3 outputs
    credit_score:     int              # 300-850
    pd_pct:           float            # probability of default %
    risk_band:        str              # AAA/AA/A/BBB/BB/C
    shap_values:      dict             # full SHAP JSON (see Section 3.4)
    
    # A4 outputs
    four_c_scores:    dict[str, float] # {character, capacity, capital, conditions}
    narrative:        dict[str, str]   # LLM text per 4C dimension
    consistency_check: dict            # narrative vs SHAP validation result
    final_report:     dict             # complete structured report
    
    # Routing & audit
    routing:          str              # AUTO_APPROVE|REVIEW|REJECT|ESCALATE|HALT
    audit_trail:      list[dict]       # immutable append-only log

1.3 LangGraph Node Graph & Routing Logic
Node	Type	Entry condition	Exit → next node
ingest_documents	Tool-calling	START	→ check_cic (parallel) + analyze_transactions (parallel)
check_cic	Tool (API)	After ingest	→ confidence_gate (join)
analyze_transactions	Code executor	After ingest	→ confidence_gate (join)
confidence_gate	Conditional router	After CIC + transactions done	HALT if critical field missing; PROCEED otherwise
llm_feature_engineer	LLM (Claude)	routing == PROCEED	→ ml_score
ml_score	SageMaker API	After llm_feature_engineer	→ report_generator
report_generator	LLM (Claude)	After ml_score	→ consistency_validator
consistency_validator	Code (deterministic)	After report_generator	→ decision_router
decision_router	Policy rules	After consistency check	→ END (with routing label)
2. Agent A1 — Data Ingestion & Feature Pipeline
Mục tiêu: Nhận hồ sơ thô và trả về structured feature vector. Đây là nơi alternative data được khai thác để giải quyết thin-file problem — không phụ thuộc vào CIC score như traditional scoring.
2.1 Input — 3 kênh tiếp nhận
Kênh 1: Tài liệu PDF/Scan
	Xử lý	Output fields
Loại tài liệu	CCCD/CMND, Hợp đồng lao động, Sổ hộ khẩu, Giấy tờ TSBĐ, GPKD (SME)	identity fields, employment fields, collateral fields
AWS Textract	Analyze Lending API → auto page classification → key-value pair extraction → confidence per field	Mỗi field kèm extraction_confidence ∈ [0,1]
Validation	Cross-check: tên trên CCCD vs hợp đồng vs đơn vay; date range checks; format validation	identity_consistency_flag ∈ {OK, MISMATCH, MISSING}

Kênh 2: CIC API
Thin-file handling quan trọng: Nếu CIC không có record, hệ thống KHÔNG từ chối — chuyển sang Alternative Scoring Path với weight cao hơn cho transaction data.
# CIC API Response → Mapped features
cic_score:          int     # 150-750; null nếu thin-file
debt_group:         int     # 1=current, 2=watchlist, 3-5=bad debt
num_active_loans:   int     # số khoản vay đang hoạt động
total_outstanding:  float   # tổng dư nợ VND
worst_ever_group:   int     # nhóm nợ xấu nhất trong lịch sử
thin_file_flag:     bool    # True → activate alternative scoring path

Kênh 3: Bank Statement (Alternative Data — Core Innovation)
Đây là tính năng cốt lõi phân biệt hệ thống với traditional scoring. 6 tháng sao kê ngân hàng tiết lộ hành vi tài chính thực tế của underbanked customers — đặc biệt những người không có lịch sử CIC.

Feature	Type	Công thức & Ý nghĩa tín dụng
avg_monthly_inflow_vnd	float	Mean(monthly_credit_sum, 6M). Proxy thu nhập thực tế — quan trọng hơn lương khai báo với self-employed
income_stability_index	float [0,1]	1 - std(monthly_inflows)/mean. Gần 1 = ổn định. Gig workers thường thấp hơn nhưng vẫn creditworthy
salary_pattern_detected	bool	Rule: credit ≈ same_amount (±5%), ngày 1-5 hàng tháng, nội dung chứa regex(LUONG|SALARY|THU NHAP). Xác nhận employment status không cần doc
regular_bill_payment_ratio	float [0,1]	% tháng có debit khớp pattern: điện (DIEN), nước (NUOC), internet (VTC|FPT|VNPT) đúng hạn (không trễ >5 ngày). Alternative credit signal cực mạnh với thin-file
debt_service_behavior	enum	NLP detect existing loan repayments trong nội dung giao dịch: ON_TIME / LATE_1_30 / LATE_31_60 / MISSING. Hành vi trả nợ thực tế — direct predictor
overdraft_count_6m	int	Số lần balance < 500,000 VND hoặc < 0. Financial stress indicator — negative signal
inflow_outflow_ratio	float	Mean(monthly_inflow) / Mean(monthly_outflow). > 1.2 = healthy cash buffer; < 1.0 = spending > income
max_single_outflow_ratio	float	Max(single_debit) / avg_monthly_inflow. Phát hiện giao dịch bất thường lớn — rủi ro thanh khoản

2.2 Critical Field Threshold System
Confidence scoring không chỉ là "dữ liệu thiếu thì cảnh báo". Hệ thống phân cấp rõ ràng: CRITICAL fields fail → HALT (yêu cầu bổ sung), không được bypass. IMPORTANT fields fail → Chuyển sang A2 imputation. OPTIONAL fields fail → bỏ qua.

Tier	Fields	Min confidence	Action nếu dưới ngưỡng
CRITICAL	identity_verified, monthly_income_or_inflow, debt_group (CIC hoặc thin_file_flag)	≥ 0.85	HALT: Dừng pipeline, yêu cầu tài liệu bổ sung, không estimate
IMPORTANT	employment_duration, collateral_value, income_stability_index, debt_service_behavior	≥ 0.70	IMPUTE: Chuyển A2 LLM Imputer + ghi imputation_flag + warning trong report
OPTIONAL	regular_bill_payment, overdraft_count, transaction network signals	≥ 0.50	USE_IF_AVAILABLE: Dùng nếu có, bỏ qua trong feature vector nếu không có

Overall Confidence Formula	overall_confidence = Σ(weight_i × confidence_i) / Σ(weight_i). Weights: CRITICAL=3, IMPORTANT=2, OPTIONAL=1. ≥ 0.80 → AUTO_PROCEED | 0.65–0.80 → PROCEED_WITH_WARNINGS | < 0.65 → ESCALATE_TO_HUMAN.
3. Agent A2 — LLM Feature Engineer
Mục tiêu: Khai thác tín hiệu từ văn bản phi cấu trúc mà rule-based hoàn toàn không làm được, và impute missing fields có căn cứ. LLM ở đây đóng vai data transformer, không phải decision maker.
3.1 Variant A — Semantic Feature Extraction
Khi nào: Luôn luôn — với mọi hồ sơ có raw_ocr_text.
Input → Output mapping
Input text	Feature được trích xuất	Type & Encoding
Nội dung đơn vay (mục đích, kế hoạch)	loan_purpose_category	Categorical → one-hot: PRODUCTION/CONSUMPTION/INVESTMENT/REFINANCING/UNCLEAR
Mô tả kế hoạch trả nợ	repayment_plan_quality	Ordinal → integer: DETAILED=3, GENERAL=2, VAGUE=1, NONE=0
So sánh thu nhập khai báo vs dòng tiền	stated_income_consistency	Binary: 1 nếu |stated - inflow| < 20%, 0 nếu > 20% hoặc mâu thuẫn
Nội dung 50 giao dịch gần nhất	transaction_purpose_distribution	Dict of floats: {salary, rent, business, retail, transfer} summing to 1.0
Mô tả doanh nghiệp (SME, từ GPKD + web)	business_legitimacy_score	Float [0,1]: tổng hợp từ: reg_age, web_presence, industry_risk, description_quality
Top-5 rủi ro phát hiện trong hồ sơ	risk_flag_count + risk_flags_list	Int + List[str]: ML nhận count, LLM report dùng list để generate narrative

Prompt design (Loan Application Extraction)
# System prompt (truncated)
system = """
You are a Vietnamese bank credit analyst assistant.
Analyze the loan application text and return ONLY valid JSON.
DO NOT add explanations. DO NOT invent information not in the text.
If a field cannot be determined from the text, set it to null.
"""

# User prompt template
user = f"""
LOAN APPLICATION TEXT:
{ocr_text[:3000]}

Return JSON with exactly these fields:
{{
  "loan_purpose_category": "PRODUCTION|CONSUMPTION|INVESTMENT|REFINANCING|UNCLEAR",
  "repayment_plan_quality": "DETAILED|GENERAL|VAGUE|NONE",
  "stated_income_consistency": true|false|null,
  "risk_flags": ["list", "of", "concern", "strings"],
  "positive_signals": ["list", "of", "strength", "strings"],
  "extraction_confidence": 0.0-1.0
}}
"""

# Parsing with validation
response = claude.messages.create(model="claude-3-5-sonnet", ...)
result = json.loads(response.content[0].text)  # strict parse
assert set(result.keys()) == REQUIRED_KEYS      # schema validation

3.2 Variant B — Intelligent Imputation
Khi nào: Khi IMPORTANT fields bị thiếu (confidence < 0.70). Thay vì dùng mean/median imputation, LLM suy luận có căn cứ từ context.

Trường hợp	Cách LLM impute
monthly_income = null nhưng có bank statement	Context cung cấp cho LLM: avg_monthly_inflow=14.8M, salary_pattern=True, employment=FULL_TIME, deductions visible Prompt: "Estimate net monthly income. Return: {estimated_value, confidence, reasoning, source}" Output: {monthly_income_imputed: 14000000, confidence: 0.81, source: "inferred_from_6mo_bank_statement"}
employment_duration = null nhưng có hợp đồng không rõ ngày	Context: employment_type=FULL_TIME, salary_months_detected=6, company_age_from_reg Prompt: "Estimate employment duration in months. Return with confidence." Output: {employment_duration_months_imputed: 24, confidence: 0.65, source: "lower_bound_from_salary_history"}

Explainability safeguard cho imputation	imputation_flag = True được ghi vào state. Trong báo cáo cuối: section "Cảnh báo dữ liệu" liệt kê mọi field imputed với: giá trị ước tính, confidence, và lý do. Chuyên viên luôn biết đâu là dữ liệu thực, đâu là ước tính. SHAP cũng nhận imputation_confidence như một feature riêng.

3.3 Thin-file Alternative Scoring Path
Khi thin_file_flag = True, A2 kích hoạt Thin-file Path: 
Traditional scoring	CreditLens thin-file path	Evidence basis
Từ chối hoặc yêu cầu tài sản thế chấp bổ sung cao	Activate alternative feature weights: transaction data ×2, bill payment ×2, income stability ×1.5	CGAP 2023: Indian fintechs đạt AUC ~0.70+ chỉ từ transaction data với 80% thin-file customers
•	Minimum requirement: ≥ 3 tháng sao kê liên tục. Dưới mức này → ESCALATE (không đủ dữ liệu để score reliably)
•	Feature weights trong thin-file path: income_stability (30%), salary_pattern (25%), debt_service_behavior (25%), bill_payment_ratio (15%), inflow_outflow_ratio (5%)
•	Report flag: "Khách hàng được đánh giá theo hướng thin-file. Kết quả dựa trên dữ liệu giao dịch thay thế, không có lịch sử tín dụng từ CIC."
4. Agent A3 — ML Scoring Engine
Mục tiêu: Nhận unified feature vector, trả về credit score với mathematical explainability. Đây là component deterministic duy nhất trong pipeline — same input luôn cho same output, không có randomness của LLM.
4.1 Unified Feature Vector — Home Credit alignment
Tất cả features được thiết kế để map trực tiếp vào cấu trúc dữ liệu của Home Credit Default Risk, đảm bảo training data và production data aligned:

Feature Group	Count	Home Credit mapping	Pilot features (10 core)
Identity & KYC	3	application_train: CODE_GENDER, DAYS_BIRTH, FLAG_OWN_CAR	age, gender, id_verified
Credit Bureau	4	bureau.csv: CREDIT_ACTIVE, DAYS_CREDIT, AMT_CREDIT_SUM_OVERDUE	cic_score, debt_group, num_active_loans, thin_file_flag
Transaction Behavioral ★ Alt. Data	8	Engineered từ installments_payments.csv + credit_card_balance.csv	avg_inflow, income_stability, salary_detected, bill_payment_ratio, debt_service, overdraft_count
LLM Semantic (A2-A)	5	Không có trong Home Credit (extracted từ documents)	loan_purpose_cat, repayment_quality, stated_income_consistency
Imputed fields (A2-B)	2	Proxy cho EXT_SOURCE_1/2/3 trong application_train	income_imputed_flag, imputation_confidence
Loan Terms	3	application_train: AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL	loan_amount_vnd, term_months, dti_ratio
TỔNG	25	Full feature set cho production	10 core features cho pilot

4.2 Training Pipeline trên Home Credit
Data preparation
# Home Credit Default Risk — training pipeline
import lightgbm as lgb
import shap
from imblearn.over_sampling import ADASYN

# 1. Load & merge tables
app = pd.read_csv("application_train.csv")         # 307,511 rows
bureau = pd.read_csv("bureau.csv")                 # aggregated per SK_ID_CURR
installments = pd.read_csv("installments_payments.csv")

# 2. Feature engineering — transaction behavioral features
installments_feats = installments.groupby("SK_ID_CURR").agg({
    "NUM_INSTALMENT_NUMBER": "max",   # loan tenure proxy
    "DAYS_INSTALMENT": "mean",        # payment timing
    "AMT_INSTALMENT": ["mean","std"], # payment stability
    "AMT_PAYMENT": "sum",             # total paid
}).reset_index()
installments_feats["payment_consistency"] = 1 - (
    installments_feats[("AMT_INSTALMENT","std")] / 
    installments_feats[("AMT_INSTALMENT","mean")]
)  # ≈ income_stability_index in production

# 3. Class imbalance: ADASYN moderate oversampling
adasyn = ADASYN(sampling_strategy=0.2, random_state=42)  # target 5:1 ratio
X_res, y_res = adasyn.fit_resample(X_train, y_train)

# 4. LightGBM with tuned hyperparameters (via Optuna)
params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "is_unbalance": True,
    "random_state": 42
}
model = lgb.LGBMClassifier(**params)
model.fit(X_res, y_res, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

4.3 Credit Score Mapping
Risk Band	Credit Score	PD Range	Auto decision	Điều kiện cộng thêm
AAA — Xuất sắc	720–850	< 2%	AUTO APPROVE	+ CIC Nhóm 1 + DTI < 40%
AA — Tốt	640–719	2%–8%	APPROVE + REVIEW	Chuyên viên xem nhanh báo cáo
A — Khá	560–639	8%–18%	FULL REVIEW	Đánh giá đầy đủ 4C
BBB — Trung bình	460–559	18%–35%	CONDITIONAL	Cần thêm TSBĐ hoặc guarantor
CC/C — Rủi ro cao	300–459	> 35%	REJECT	Trừ đặc cách có thẩm quyền cao

Hard Override Rules (Policy-based)	CIC Nhóm 4-5 → REJECT bất kể ML score. Loan_amount > 10 tỷ VND → ESCALATE to head office. overall_confidence < 0.65 → HUMAN REVIEW trước khi quyết định. thin_file_flag=True + score < 560 → tăng yêu cầu tài sản thế chấp.

4.4 SHAP Output Schema — Bridge sang A4
SHAP được tính ngay tại A3 và serialize thành JSON chuẩn. A4 bắt buộc phải dùng SHAP JSON này làm ground truth — không được tự ý thêm reasoning ngoài SHAP.

shap_output = {
    "credit_score": 672,
    "pd_pct": 5.8,
    "risk_band": "AA",
    "model_version": "lgbm_v1.2_homecredit",
    "inference_timestamp": "2026-03-18T10:23:41Z",

    "top_positive_factors": [
        {"feature": "salary_pattern_detected",     "shap": +0.089, "value": true,   "label_vi": "Phát hiện giao dịch lương đều đặn"},
        {"feature": "income_stability_index",      "shap": +0.072, "value": 0.81,  "label_vi": "Thu nhập ổn định 6 tháng (index 0.81)"},
        {"feature": "regular_bill_payment_ratio",  "shap": +0.051, "value": 0.90,  "label_vi": "Thanh toán hóa đơn đúng hạn 90%"},
        {"feature": "repayment_plan_quality",      "shap": +0.038, "value": 3,     "label_vi": "Kế hoạch trả nợ chi tiết"},
        {"feature": "stated_income_consistency",   "shap": +0.029, "value": true,  "label_vi": "Thu nhập khai báo khớp sao kê"}
    ],

    "top_negative_factors": [
        {"feature": "dti_ratio",              "shap": -0.063, "value": 0.48, "label_vi": "Tỷ lệ nợ/thu nhập ở mức cao (48%)"},
        {"feature": "overdraft_count_6m",     "shap": -0.031, "value": 2,   "label_vi": "Số dư về âm 2 lần trong 6 tháng"},
        {"feature": "imputation_confidence",  "shap": -0.018, "value": 0.81,"label_vi": "Một số trường được ước tính (confidence 81%)"}
    ],

    "4c_shap_allocation": {
        "character":  {"shap_sum": 0.118, "pct": 28},
        "capacity":   {"shap_sum": 0.172, "pct": 41},
        "capital":    {"shap_sum": 0.080, "pct": 19},
        "conditions": {"shap_sum": 0.050, "pct": 12}
    },

    "all_features_shap": { ... }  # full dict lưu vào DynamoDB cho audit
}
5. Agent A4 — Report Generator & Explainability Stack
Mục tiêu: Biến SHAP output thành báo cáo tín dụng dễ đọc, có căn cứ, và auditable. Đây là nơi chúng ta duy trì contribution của MASCA (transparent, explainable) nhưng với foundation mạnh hơn: grounded trên ML math, không phải LLM reasoning.
5.1 Ba tầng Explainability
Tầng	Mechanism	Ai dùng được
Tầng 1 SHAP Feature Attribution	SHAP TreeExplainer — toán học đảm bảo: consistency, efficiency, additivity. Mỗi feature có exact contribution đến PD.	Kiểm toán nội bộ, compliance officer, quản lý rủi ro
Tầng 2 Grounded LLM Narrative	LLM diễn giải SHAP values thành tiếng Việt — có constraint: chỉ được đề cập factors trong SHAP, không được thêm reasoning ngoài data.	Chuyên viên tín dụng, giám đốc chi nhánh
Tầng 3 Immutable Audit Trail	DynamoDB append-only log mọi agent action: input, output, timestamp, model_version, confidence. Không thể modify sau khi ghi.	Regulatory compliance, NĐ 94/2025 sandbox audit

5.2 RAG Pipeline — Grounding trong Policy
•	Knowledge Base: Thông tư NHNN về phân loại nợ và trích lập dự phòng, quy định CIC 2024, điều kiện sản phẩm vay từng loại, hướng dẫn xét duyệt tín dụng
•	Vector Store: AWS OpenSearch Serverless, embedding model: Amazon Titan Text V2
•	Query: A4 tạo query dựa trên context hồ sơ → retrieve top-3 relevant policy clauses → inject vào prompt dưới dạng "Policy Context"
•	Citation requirement: LLM phải cite số thông tư/điều khoản cụ thể khi đề cập policy — không được generalize

5.3 Report Generation Prompt (A4 Core)
# System prompt — đặt ra hard constraints
SYSTEM = """
You are a Vietnamese bank credit report writer.

HARD RULES (violations = invalid report):
1. ONLY discuss risk factors that appear in the SHAP values provided.
   Do NOT invent new risk factors not supported by SHAP data.
2. CITE specific policy clauses from the Policy Context section.
3. For each negative factor, provide one specific, actionable improvement suggestion.
4. Write in formal Vietnamese banking language.
5. Do NOT reveal model weights, training data, or technical internals.
"""

# User prompt
USER = f"""
SHAP Analysis: {json.dumps(shap_output)}

Policy Context (retrieved from RAG):
{rag_context}

Warnings: {json.dumps(warnings)}

Return JSON:
{{
  "character_assessment": {{
    "score": 0-30,
    "status": "DAT|XEM_XET|CHUA_DAT",
    "indicators_met": ["list"],
    "indicators_review": ["list with action"],
    "narrative": "100-150 word Vietnamese text, SHAP-grounded"
  }},
  "capacity_assessment": {{ ... }},
  "capital_assessment":  {{ ... }},
  "conditions_assessment": {{ ... }},
  "recommendation": "APPROVE|CONDITIONAL|REVIEW|REJECT",
  "suggested_terms": {{"max_amount_vnd": int, "max_term_months": int}},
  "caveats": ["imputation warnings", "data quality notes"]
}}
"""

5.4 Consistency Validator (sau A4)
Đây là cơ chế không có trong MASCA — một bước deterministic kiểm tra LLM narrative có trung thực với SHAP data không:
def validate_narrative_consistency(shap_output, narrative):
    """
    Check that LLM narrative only references factors in top SHAP features.
    Returns: {passed: bool, violations: list[str], confidence: float}
    """
    top_shap_features = {f["feature"] for f in
        shap_output["top_positive_factors"] + shap_output["top_negative_factors"]}
    top_shap_labels = {f["label_vi"] for f in
        shap_output["top_positive_factors"] + shap_output["top_negative_factors"]}

    violations = []
    for dimension in ["character", "capacity", "capital", "conditions"]:
        text = narrative[f"{dimension}_assessment"]["narrative"]

        # Check: every major claim traceable to SHAP
        claim_has_shap_support = any(
            label.lower() in text.lower() for label in top_shap_labels
        )
        if not claim_has_shap_support:
            violations.append(f"{dimension}: narrative lacks SHAP grounding")

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "shap_coverage": len(top_shap_labels & extract_mentioned_topics(text)) / len(top_shap_labels)
    }

# If consistency check fails → re-prompt A4 with violation list
# Max 2 retries; if still fails → flag report for human review

5.5 Output Format — Cấu trúc Báo cáo
Nguyên tắc: "Điểm trước — chi tiết sau." Phần Executive Summary chiếm 1/3 trang đầu. Chuyên viên nhìn vào là biết quyết định ngay.

Section 1 — Executive Summary
Credit Score 672 / 850	Risk Band AA — Rủi ro Thấp	Khuyến nghị PHÊ DUYỆT	Xác suất vỡ nợ 5.8% Confidence: 87%

Section 2 — 4C Scorecard
Tiêu chí	Điểm	Trạng thái	SHAP contribution / Tóm tắt
Character (Uy tín)	28/30	ĐẠT	SHAP +28% | Không nợ xấu, thanh toán hóa đơn đúng hạn 90%, lịch sử giao dịch nhất quán
Capacity (Năng lực)	31/40	XEM XÉT	SHAP +41% | Thu nhập ổn định. DTI 48% ở mức cao. 2 lần overdraft cần giải trình
Capital (Vốn)	16/20	ĐẠT	SHAP +19% | TSBĐ: xe ô tô định giá 450M. LTV 67% (ngưỡng 70%). Đủ buffer
Conditions (Điều kiện)	9/10	TỐT	SHAP +12% | Ngành ổn định. Mục đích vay sản xuất rõ ràng. Kế hoạch trả nợ chi tiết

Section 3 — Phân tích Chi tiết mỗi 4C Dimension
Mỗi dimension có 3 phần:
•	Indicators MET: Danh sách chỉ tiêu đạt, kèm số liệu và SHAP contribution cụ thể
•	Indicators NEEDS REVIEW: Chỉ tiêu cần xem xét, lý do, và action recommendation cụ thể (e.g., "DTI 48% > ngưỡng tốt 40% → Đề xuất: giảm hạn mức 20% hoặc yêu cầu chứng minh thu nhập bổ sung")
•	LLM Narrative: Đoạn 100–150 chữ bằng tiếng Việt, grounded hoàn toàn trên SHAP values, có cite policy clause

Section 4 — Data Warnings & Caveats
Liệt kê tất cả: imputed fields, low-confidence extractions, thin-file flag nếu có. Chuyên viên luôn biết độ tin cậy của từng phần dữ liệu.
Section 5 — Audit Reference
application_id, model_version, inference_timestamp, SHAP JSON hash, RAG chunks used — đủ để tái hiện lại bất kỳ quyết định nào.
6. Dataset — Home Credit Default Risk
6.1 Cấu trúc Dataset
Bảng	Rows	Mô tả & Features sử dụng
application_train.csv	307,511	Bảng chính: thông tin cá nhân, tài chính, khoản vay. TARGET = 0/1. Features: AMT_CREDIT, AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED, EXT_SOURCE_1/2/3, FLAG_OWN_CAR, CODE_GENDER
bureau.csv	1.7M	Lịch sử tín dụng từ credit bureau. Features: CREDIT_ACTIVE, DAYS_CREDIT, AMT_CREDIT_SUM, AMT_CREDIT_SUM_OVERDUE → map sang cic_score proxy
installments_payments.csv	13.6M	Lịch sử trả góp từng kỳ. DAYS_INSTALMENT vs DAYS_ENTRY_PAYMENT = payment timing. AMT_INSTALMENT stability = income_stability_index proxy
credit_card_balance.csv	3.8M	Thẻ tín dụng: AMT_BALANCE/AMT_CREDIT_LIMIT_ACTUAL → credit utilization. AMT_DRAWINGS = spending pattern proxy
pos_cash_balance.csv	10.0M	POS và tiền mặt: DPD (Days Past Due) → overdraft_count proxy. SK_DPD_DEF → debt_service_behavior proxy
previous_application.csv	1.67M	Lịch sử đơn vay trước: NAME_CONTRACT_STATUS (Approved/Refused) → loan purpose history

6.2 Feature Engineering từ Home Credit → Production mapping
Mỗi alternative data feature trong production được map từ Home Credit tables, đảm bảo training data và production pipeline aligned:
Production Feature	Home Credit Proxy	Engineering logic
income_stability_index	installments_payments: AMT_INSTALMENT CV	1 - std(AMT_INSTALMENT)/mean(AMT_INSTALMENT) per SK_ID_CURR
salary_pattern_detected	Không trực tiếp — dùng DAYS_EMPLOYED + income regularity	DAYS_EMPLOYED > 0 AND income_stability > 0.7 → proxy True
debt_service_behavior	pos_cash_balance: SK_DPD, SK_DPD_DEF	max(SK_DPD) per applicant: 0→ON_TIME, 1-30→LATE_1_30, >30→LATE_31_60
overdraft_count_6m	pos_cash_balance: NAME_CONTRACT_STATUS	Count months where SK_DPD > 0 in last 6 entries
inflow_outflow_ratio	credit_card: AMT_DRAWINGS / AMT_BALANCE	Proxy: (AMT_INCOME_TOTAL/12) / (AMT_ANNUITY + avg_drawings)
cic_score (proxy)	EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3	Weighted mean: 0.5×EXT_SOURCE_2 + 0.3×EXT_SOURCE_3 + 0.2×EXT_SOURCE_1, scale to 150-750

6.3 Training / Validation / Test Split
Split	Records	Mục đích	Lưu ý
Train	215,258 (70%)	LightGBM training	Stratified by TARGET, ADASYN applied
Validation	46,127 (15%)	Hyperparameter tuning, early stopping	No ADASYN — real distribution
Test (held-out)	46,126 (15%)	Final AUC, KS, Gini evaluation	Locked — không xem trong quá trình build
Pilot test set (40 cases)	40 (expert curated)	Final pilot evaluation — 25 individual + 15 SME proxy	Stratified: 50% default / 50% non-default per group. Expert blind rating.

6.4 Baseline Benchmarks (Kaggle community)
Approach	AUC-ROC	Method	Reference
Logistic Regression (baseline)	~0.68	application_train only, top features	Kaggle community
LightGBM + bureau features	~0.77	Multi-table merge, basic FE	Kaggle EDA notebooks
LightGBM + full feature engineering	~0.794	All 8 tables, 500+ features	Kaggle top public
CreditLens target	> 0.80	+ LLM semantic features (A2-A) + alternative data weights	Ablation study sẽ quantify incremental gain
7. Explainability Stack — So sánh với MASCA
MASCA's contribution là tính transparent qua chain-of-thought LLM reasoning. Hệ thống của chúng ta duy trì và cải thiện transparency đó bằng 3 lớp grounding mà MASCA không có.

Dimension	MASCA approach	CreditLens approach
Source of explanation	LLM chain-of-thought — post-hoc rationalization không guaranteed consistent	SHAP TreeExplainer — mathematical attribution, same input always same SHAP
Traceability	Cannot trace claim → specific data point. LLM may confabulate	Every SHAP value traceable đến feature value và training data distribution
Consistency	Changing demographic input → LLM reasoning changes. Accuracy drops 6.96% when gender flipped (MASCA paper)	LightGBM deterministic. SHAP mathematically consistent. LLM only translates, does not reason
Auditability	No structured audit trail. LLM text not reproducible	Immutable DynamoDB audit trail. SHAP JSON hash verifiable. Full replay possible
Policy grounding	LLM cites policy từ training knowledge — may be outdated or hallucinated	RAG từ current policy docs với citation verification. Consistency validator flags mismatch
Regulatory compliance	Hard to satisfy adverse action notice requirements khi LLM explanations vary	SHAP-grounded explanation = specific, accurate reasons — satisfies ECOA/VN equivalent

Key distinction	MASCA: "I think DTI is high risk BECAUSE income is volatile" (LLM creates the causal link). CreditLens: "DTI contributed −0.063 to credit score, income_stability contributed +0.072" (SHAP calculates the causal link, LLM translates it). The first is an opinion. The second is a mathematical fact.
8. Chiến Lược Đánh Giá
8.1 Quantitative metrics
Metric	Target	Tại sao	Cách đo
AUC-ROC	> 0.80	Không bị ảnh hưởng class imbalance 8%	sklearn on held-out 15% test set
Pilot accuracy (40 cases)	≥ 38/40 = 95%	Yêu cầu cụ thể, demo-ready	Binary correct/incorrect on full pipeline output
Gini Coefficient	> 0.60	Standard credit scoring metric	2 × AUC − 1
Thin-file sub-AUC	> 0.72	Validate alternative data path	Sub-group AUC on records where thin_file_flag=True
SHAP consistency score	> 0.80	Validate explainability quality	shap_coverage in consistency_validator output

8.2 Ablation Study — Quantify mỗi component
"Tại sao cần alternative data? Tại sao cần LLM features?" — Ablation study trả lời bằng số:
Experiment	AUC expected	Δ AUC	Component được justify
E0: Logistic Regression, CIC only (traditional baseline)	~0.65	—	Traditional scoring as lower bound
E1: LightGBM, tabular features only (no alt data)	~0.75	+0.10	Value of ML over logistic
E2: + Transaction alternative data features	~0.79	+0.04	Value of alternative data (thin-file path)
E3: + LLM Semantic features (A2-A)	~0.81	+0.02	Value of unstructured document analysis
E4: + LLM Imputation (A2-B) — Full System	> 0.82	+0.01	Value of smart imputation
9. Kế Hoạch Build 4 Tuần
Thứ tự critical path: A3 (ML model) trước tiên — phải có working model trước khi build A4 report generator. A2 (LLM features) được thêm sau khi A3 baseline chạy ổn định.

Tuần	Focus	Deliverables cụ thể	Claude Code strategy
1 Data + A3	A3 — ML Core Home Credit + Feature Engineering	• Download & EDA: 307K records, 8 tables • Transaction feature engineering pipeline • Alternative data feature creation (8 features) • LightGBM baseline train → AUC > 0.77 • SHAP output schema defined & tested • SageMaker endpoint deployment	FE pipeline code, SHAP JSON schema, SageMaker boilerplate
2 A1 + A3 tune	A1 — Ingestion + A3 hyperparameter tuning	• Textract integration cho PDF docs • CIC API mock (hoặc real nếu có) • Bank statement parser (CSV → transaction features) • Optuna tuning A3 → AUC > 0.80 • Confidence gate logic + HALT/PROCEED routing • FastAPI wrapper: POST /v1/score	Textract wrapper, CSV parser, FastAPI routes, DynamoDB schema
3 A2 + A4	A2 + A4 — LLM Features + Report	• A2-A: Semantic extraction prompts + validation • A2-B: Imputation prompts per field type • A4: RAG setup (policy docs → OpenSearch) • A4: Report generation prompt + constraints • Consistency validator implementation • LangGraph StateGraph full assembly	LangGraph graph, prompt templates, RAG ingestion, consistency validator
4 Demo	Integration + Demo + Ablation study	• 40 pilot cases end-to-end test • Thin-file path: 5 cases không có CIC • Ablation E0→E4 results documented • Next.js dashboard: upload → report view • SHAP bar chart visualization • Video demo recording	Dashboard UI, SHAP visualization component, video script

Demo 3 moments (BGK cần thấy)	(1) Thin-file demo: khách hàng không có CIC → hệ thống vẫn cho score dựa trên 6 tháng sao kê. (2) Explainability: click vào score → SHAP bar chart → từng bar tương ứng đoạn text trong báo cáo. (3) Consistency: thử thay 1 feature → score thay đổi, SHAP thay đổi, narrative thay đổi nhất quán.
10. Hạ Tầng AWS
Service	Role	Agent	Pilot cost/month
Amazon Textract	OCR + Analyze Lending API	A1	~$15
Amazon Bedrock (Claude 3.5 Sonnet)	LLM inference — A2 extraction + A4 report	A2, A4	~$30–50
Amazon SageMaker RT Endpoint	LightGBM model hosting + SHAP	A3	~$50 (ml.t3.medium)
Amazon OpenSearch Serverless	RAG vector store cho policy docs	A4	~$100
AWS Lambda + API Gateway	REST API endpoints, LangGraph runner	All	~$5
Amazon DynamoDB	CreditState store + audit trail (append-only)	All	~$5
Amazon S3	Document storage, model artifacts, SHAP JSON	All	~$3
TỔNG PILOT			~$210–230/month
Phụ Lục — Tài liệu Tham khảo
Nghiên cứu nền tảng
•	Home Credit Default Risk Dataset — kaggle.com/competitions/home-credit-default-risk
•	GPT-LGBM: ChatGPT-Based Framework for Credit Scoring — Yu et al. (2023), Knowledge and Information Systems [LLM→Feature→ML pattern]
•	Cash Flow Underwriting with Bank Transaction Data — Ng et al. (2025), arXiv:2510.16066 [AUC 0.782 từ transaction data only, Malaysia MSME]
•	Leveraging Transactional Data for Micro and Small Enterprise Lending — CGAP (2023) [AUC ~0.70+ với 80% thin-file customers]
•	MASCA: LLM-based Multi-Agent System for Credit Assessment — Jajoo et al. (2025), arXiv:2507.22758 [Multi-agent, nhưng AUC thấp do no ML backbone]
•	SHAP and LIME: An Evaluation of Discriminative Power in Credit Risk — Gramegna & Giudici (2021), Frontiers in AI [SHAP > LIME cho credit scoring]
•	Machine Learning-Based Empirical Investigation for Credit Scoring in Vietnam — Springer (2021) [LightGBM best trên data Việt Nam]
•	Fintech Credit Risk Assessment for SMEs: Evidence from China — IMF Working Paper (2020) [MYbank 76 variables, thin-file SME]
Documentation
•	LangGraph Documentation — langchain-ai.github.io/langgraph
•	AWS Bedrock — Build digital lending solution — AWS ML Blog (Jan 2025)
•	SHAP TreeExplainer — shap.readthedocs.io
•	SDV (Synthetic Data Vault) — sdv.dev [nếu cần augment sau này]
