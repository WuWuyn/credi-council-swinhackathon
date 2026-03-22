
TECHNICAL ARCHITECTURE GUIDE
CreditLens AI
Hệ thống Chấm điểm Tín dụng Thông minh
Tích hợp NoxMoon (Home Credit) · MASCA · SHAP · LLM Grounded Explainability


Dataset Home Credit Default Risk 307,511 records · 8 tables	Core Stack LightGBM + SHAP + Claude API + LangGraph + AWS Bedrock	References NoxMoon (rank #17/7198) MASCA arXiv:2507.22758 Tờ trình tín dụng VN chuẩn
0. Bài Toán & Tổng Quan Giải Pháp
0.1 Problem Statement
Ban Tổ Chức:
"Traditional credit scoring models rely heavily on historical financial data and static rules, limiting their ability to accurately assess creditworthiness for underbanked customers, especially micro SMEs, and to adapt to changing borrower behavior. This results in biased decisions, limited assessments, and restricted access to financial services."

0.2 Kiến trúc giải pháp — 3 module chính
CreditLens AI kết hợp ba nguồn học thuật: (1) NoxMoon pipeline — feature engineering tinh vi từ solution rank #17 Kaggle — làm nền cho ML scoring; (2) MASCA agent architecture — phân công vai trò chuyên biệt cho từng agent; (3) SHAP Grounded Explainability — mọi quyết định có thể audit đến từng data point. Không có confidence gate — pipeline luôn chạy đến cuối và xuất báo cáo đầy đủ.

	Module	Input	Processing	Output
A1	Data Ingestion & Feature Pipeline	Hồ sơ PDF, CIC API, sao kê CSV	Textract OCR + transaction aggregation + NoxMoon statistical features	structured_feats + confidence_map
A2	LLM Feature Engineer	raw_ocr_text + structured_feats	Semantic extraction (Var.A) + intelligent imputation (Var.B)	llm_feats + imputation_log
A3	ML Scoring Engine	Unified feature vector (A1+A2) + NoxMoon sub-model predictions	LightGBM predict_proba → log-odds credit score → SHAP TreeExplainer	credit_score + pd_pct + risk_band + shap_values JSON
A4	Report Generator (MASCA + SHAP grounding)	shap_values JSON + RAG policy + 4C scoring	LLM narrative grounded strictly on SHAP + RAG → consistency validator	Tờ trình tín dụng đầy đủ (JSON + PDF)
1. Module A1 — Data Ingestion & Feature Pipeline
Mục tiêu: Nhận hồ sơ thô và xuất ra unified feature vector. Đây là module thực thi toàn bộ feature engineering của NoxMoon — phần tạo ra sự khác biệt về AUC so với baseline. Không dùng LLM ở bước này.
1.1 Input — 3 kênh
Kênh 1: Tài liệu PDF / Scan
	Xử lý	Output fields
Tài liệu	CCCD, CMND, Hợp đồng lao động, Sổ HKTT, GPKD (SME), Giấy tờ TSBĐ	identity_verified, employment_type, collateral_type, collateral_value_vnd
AWS Textract	Analyze Lending API → auto page classification → key-value pair extraction → confidence per field	Mỗi field kèm extraction_confidence ∈ [0,1]
Cross-check	Đối chiếu tên trên CCCD vs HĐ vs đơn vay; date range; format validation	identity_consistency_flag: OK | MISMATCH | MISSING

Kênh 2: CIC API
Thin-file handling: Nếu CIC không có record → thin_file_flag=True → hệ thống KHÔNG từ chối mà chuyển sang alternative scoring path với transaction data được weight cao hơn.
# CIC API Response → Feature mapping
cic_score          : int    # 300–850; null nếu thin-file
debt_group         : int    # 1=Nợ đủ tiêu chuẩn … 5=Nợ mất vốn
num_active_loans   : int    # số khoản vay đang hoạt động
total_outstanding  : float  # tổng dư nợ VND
worst_ever_group   : int    # nhóm nợ xấu nhất trong lịch sử
thin_file_flag     : bool   # True → activate alternative scoring path

Kênh 3: Bank Statement CSV (Alternative Data — Core Innovation)
Đây là điểm cốt lõi giải bài toán thin-file/underbanked. 6 tháng sao kê tiết lộ hành vi tài chính thực tế mà CIC không có. Mỗi feature được thiết kế map 1-1 với NoxMoon's installments_payments.csv để đảm bảo training-production alignment.

Feature (Production)	Home Credit Mapping	Công thức & Ý nghĩa
avg_monthly_inflow_vnd	AMT_INCOME_TOTAL / 12	Mean(monthly_credit_sum, 6M). Proxy thu nhập thực tế — quan trọng hơn lương khai báo với self-employed
income_stability_index	1 - CV(AMT_INSTALMENT)	1 - std(inflows)/mean. Gần 1 = ổn định. Gig workers thường thấp hơn nhưng vẫn creditworthy
salary_pattern_detected	DAYS_EMPLOYED > 0 + stability > 0.7	Rule: credit ≈ same amount (±5%), ngày 1–5 hàng tháng, nội dung regex(LUONG|SALARY). Confirm employment không cần doc
debt_service_behavior	SK_DPD từ POS_CASH_balance	NLP detect repayment trong nội dung giao dịch → ON_TIME / LATE_1_30 / LATE_31_60 / MISSING
regular_bill_payment_ratio	(Không có trực tiếp)	% tháng có debit khớp pattern điện/nước/internet đúng hạn. Alternative credit signal cực mạnh với thin-file
overdraft_count_6m	SK_DPD > 0 count	Số lần balance < 500K VND hoặc < 0. Financial stress indicator — negative signal
inflow_outflow_ratio	AMT_INCOME / (AMT_ANNUITY + drawings)	Mean(inflow) / Mean(outflow). > 1.2 = healthy. < 1.0 = spending > income
max_single_outflow_ratio	max(AMT_DRAWINGS) / income	Max(single_debit) / avg_monthly_inflow. Phát hiện giao dịch bất thường lớn
business_revenue_trend	Linear slope(monthly_inflows)	SME only: slope > 0 = tăng trưởng. Thay thế báo cáo tài chính cho micro SME

1.2 NoxMoon Feature Engineering — Sub-model Predictions
Đây là breakthrough chính của NoxMoon (đưa từ silver medal lên gold, +0.003 AUC). Thay vì chỉ aggregate mean/sum từ bảng phụ, train một LightGBM nhỏ trên từng previous record rồi aggregate predictions — model tự học record nào quan trọng.

Sub-model	Training data (Home Credit)	Output feature
prev_training.ipynb	previous_application.csv: mỗi previous record là 1 training sample, target = TARGET của current application	agg_prev_score_mean, _max, _min, _std → merge by SK_ID_CURR
buro_training.ipynb	bureau.csv: mỗi bureau record → train LightGBM → predict → agg by SK_ID_CURR	agg_buro_score_mean, _max, _min → merge by SK_ID_CURR
month_training.ipynb	installments + POS + CC monthly: group by SK_ID_PREV trước, train LightGBM → agg by SK_ID_CURR (cần special KFold để tránh leak)	agg_month_score_mean, _max → merge by SK_ID_CURR
inst-ts.ipynb / pos-ts.ipynb cc-ts.ipynb / bubl-ts.ipynb	Time series features: GRU network train trên mỗi sequence → extract prediction làm features. AUC 0.55–0.61 per table	ts_inst_pred, ts_pos_pred, ts_cc_pred, ts_bubl_pred → merge by SK_ID_CURR
house-doc-feats.ipynb	~20 binary document flags + house features → train Logistic Regression → extract prediction	house_doc_pred → merge by SK_ID_CURR

Data Leak Warning	Monthly records của cùng 1 customer thường share cùng giá trị (same monthly payment). Phải dùng StratifiedGroupKFold với groups=SK_ID_CURR — early stopping sẽ trigger đúng lúc khi model bắt đầu exploit shared information. Nếu không: CV tăng ảo, production AUC thấp hơn nhiều.

1.3 Unified Feature Vector — Input cho A3
Feature Group	Count	Home Credit Source	Key features
Identity & KYC	4	application_train: CODE_GENDER, DAYS_BIRTH, FLAG_OWN_CAR, FLAG_OWN_REALTY	age, gender, id_verified, identity_consistency_flag
Credit Bureau (CIC proxy)	5	bureau.csv: CREDIT_ACTIVE, DAYS_CREDIT, AMT_CREDIT_SUM_OVERDUE + EXT_SOURCE_1/2/3	cic_score_proxy, debt_group, num_active_loans, worst_ever_group, thin_file_flag
Transaction Behavioral ★	9	installments_payments, credit_card_balance, POS_CASH_balance	avg_inflow, income_stability, salary_detected, debt_service, bill_payment, overdraft, inflow_outflow, max_outflow, revenue_trend(SME)
NoxMoon Sub-model Scores	8	prev_training + buro_training + month_training + GRU 4 tables	agg_prev_score, agg_buro_score, agg_month_score, ts_inst_pred, ts_pos_pred, ts_cc_pred, ts_bubl_pred, house_doc_pred
LLM Semantic (từ A2)	6	Không có trong dataset — LLM extract từ documents	loan_purpose_cat, repayment_quality, stated_income_consistency, risk_flag_count, business_legitimacy (SME), sme_growth_signal (SME)
LLM Imputed (từ A2)	2	Proxy EXT_SOURCE — khi fields thiếu	income_imputed_flag, imputation_confidence
Loan Terms	3	application_train: AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL	loan_amount_vnd, term_months, dti_ratio
TỔNG	37	Full feature set. Pilot dùng 10 core features	Chạy NoxMoon notebooks theo thứ tự: prev_training → buro_training → month_training → ts → house-doc → lgb1/2/3
2. Module A2 — LLM Feature Engineer
Mục tiêu: Khai thác tín hiệu từ văn bản phi cấu trúc và impute missing fields có căn cứ. LLM ở đây là data transformer, không phải decision maker. Mọi output phải có structured JSON schema để A3 có thể consume.
2.1 Variant A — Semantic Feature Extraction
Input
•	raw_ocr_text["loan_application"]: nội dung đơn vay (mục đích, kế hoạch trả nợ)
•	raw_ocr_text["contract"]: hợp đồng lao động / GPKD
•	transaction_descriptions: 50 nội dung giao dịch gần nhất từ A1
•	web_crawl_text (SME): kết quả crawl từ Cổng ĐKKD + Google

Prompt Design — Loan Application Extraction
SYSTEM = """
You are a Vietnamese bank credit analyst assistant.
Analyze the provided text and return ONLY valid JSON.
DO NOT add explanations. DO NOT invent information not in the text.
If a field cannot be determined from the text, set it to null.
"""

USER = f"""
LOAN APPLICATION TEXT: {ocr_text[:3000]}

Return JSON with exactly these fields:
{
  "loan_purpose_category": "PRODUCTION|CONSUMPTION|INVESTMENT|REFINANCING|UNCLEAR",
  "repayment_plan_quality": "DETAILED|GENERAL|VAGUE|NONE",
  "stated_income_consistency": true|false|null,
  "risk_flags": ["list of concern strings — max 5"],
  "positive_signals": ["list of strength strings — max 5"],
  "extraction_confidence": 0.0-1.0
}
"""

# Parse với strict schema validation
result = json.loads(response.content[0].text)
assert set(result.keys()) == REQUIRED_KEYS  # fail fast nếu sai schema

Output Features (Variant A)
Feature	Type / Encoding	Ý nghĩa & Home Credit Mapping
loan_purpose_category	Categorical → one-hot (5 classes)	Mục đích vay. Map → NAME_CONTRACT_TYPE + NAME_GOODS_CATEGORY trong previous_application.csv
repayment_plan_quality	Ordinal int: 3/2/1/0	DETAILED=3, GENERAL=2, VAGUE=1, NONE=0. Không có trong Home Credit — đây là giá trị LLM thêm vào
stated_income_consistency	Binary bool	True nếu |khai báo - inflow| < 20%. Cross-check AMT_INCOME_TOTAL vs avg_monthly_inflow
risk_flag_count	Integer [0..5]	Số cờ đỏ phát hiện trong văn bản. ML nhận count, A4 dùng list để generate narrative
business_legitimacy_score (SME)	Float [0,1]	Tổng hợp: web_presence + reg_age + industry_risk + description_quality
sme_growth_signal (SME)	Categorical GROWING/STABLE/DECLINING	Kết hợp business_revenue_trend (từ A1) + web crawl context

2.2 Variant B — Intelligent Imputation (khi field thiếu)
Khi nào dùng: Khi field quan trọng bị thiếu. Thay vì mean/median imputation, LLM suy luận có căn cứ từ context — output kèm confidence và source để ML và báo cáo dùng.

Field thiếu	Cách LLM impute
monthly_income = null nhưng có sao kê	Context: avg_monthly_inflow=14.8M, salary_pattern=True, employment=FULL_TIME Prompt: "Estimate net monthly income. Return {estimated_value, confidence, reasoning, source}" Output: {monthly_income_imputed: 14000000, confidence: 0.81, source: "inferred_from_6mo_bank_statement"}
employment_duration = null nhưng có HĐ không rõ ngày	Context: employment_type=FULL_TIME, salary_months_detected=6, company_reg_date Prompt: "Estimate employment duration in months with confidence." Output: {employment_duration_months: 24, confidence: 0.65, source: "lower_bound_from_salary_history"}

Imputation Explainability Safeguard	imputation_flag = True được ghi vào state. Trong báo cáo A4: section "Cảnh báo dữ liệu" liệt kê mọi field imputed với giá trị ước tính + confidence + lý do. imputation_confidence được đưa vào ML model như feature độc lập — model học: nhiều fields imputed = rủi ro tăng nhẹ. Chuyên viên luôn biết đâu là data thực, đâu là ước tính.
3. Module A3 — ML Scoring Engine
Mục tiêu: Nhận unified feature vector, xuất credit score với SHAP explanation. Đây là component deterministic duy nhất — same input luôn cho same output. Train trên Home Credit Default Risk (307K records, 8 tables) theo đúng NoxMoon pipeline.
3.1 Training Pipeline — NoxMoon method
Bước 1: Data preparation
# 1. Load & merge — chạy notebooks theo đúng thứ tự NoxMoon
app       = pd.read_csv("application_train.csv")          # 307,511 rows
prev_pred = pd.read_csv("output/prev_score.csv")          # từ prev_training.ipynb
buro_pred = pd.read_csv("output/buro_score.csv")          # từ buro_training.ipynb
month_pred= pd.read_csv("output/month_score.csv")         # từ month_training.ipynb
ts_feats  = pd.read_csv("output/ts_features.csv")         # từ inst/pos/cc/bubl ts notebooks
doc_pred  = pd.read_csv("output/house_doc_pred.csv")      # từ house-doc-feats.ipynb

# 2. Merge tất cả về application level
df = app.merge(prev_pred, on="SK_ID_CURR", how="left")
       .merge(buro_pred,  on="SK_ID_CURR", how="left")
       .merge(month_pred, on="SK_ID_CURR", how="left")
       .merge(ts_feats,   on="SK_ID_CURR", how="left")
       .merge(doc_pred,   on="SK_ID_CURR", how="left")

# 3. Transaction behavioral features từ installments_payments
inst = pd.read_csv("installments_payments.csv")
inst_agg = inst.groupby("SK_ID_CURR").agg({
    "NUM_INSTALMENT_NUMBER": "max",
    "AMT_INSTALMENT": ["mean","std","max"],
    "AMT_PAYMENT": ["mean","sum"],
    "DAYS_ENTRY_PAYMENT": "mean"
}).reset_index()
inst_agg["income_stability_index"] = 1 - (
    inst_agg[("AMT_INSTALMENT","std")] / inst_agg[("AMT_INSTALMENT","mean")]
)  # proxy cho income_stability_index trong production

Bước 2: Class imbalance — NoxMoon downsampling strategy
# Target distribution: ~8% default (imbalanced)
# NoxMoon strategy: downsampling major class (tốt hơn oversampling cho dataset này)

from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

def train_fold_with_downsampling(X_tr, y_tr, X_val, y_val, n_splits_major=3):
    """
    NoxMoon approach: chia major class thành 3 phần, train 3 runs,
    mỗi run dùng all minor + 1/3 major, average kết quả
    """
    major_idx = np.where(y_tr == 0)[0]
    minor_idx = np.where(y_tr == 1)[0]
    chunk_size = len(major_idx) // n_splits_major
    preds = []
    for i in range(n_splits_major):
        major_chunk = major_idx[i*chunk_size:(i+1)*chunk_size]
        idx = np.concatenate([minor_idx, major_chunk])
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X_tr[idx], y_tr[idx],
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
        preds.append(model.predict_proba(X_val)[:, 1])
    return np.mean(preds, axis=0)

Bước 3: LightGBM hyperparameters (Bayesian-optimized)
# NoxMoon finding: shallow trees + low feature_fraction = key
PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.003,   # NoxMoon: 0.003
    "max_depth":        5,       # NoxMoon finding: shallow (4/5) optimal
    "num_leaves":       31,
    "feature_fraction": 0.3,     # NoxMoon finding: KEY parameter
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples":20,
    "random_state":     42,
    "n_jobs":           -1,
}

# Validation: StratifiedKFold 5-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# CRITICAL: với monthly records training, dùng GroupKFold(groups=SK_ID_CURR)
# để tránh data leak từ records cùng customer

3.2 Credit Score Mapping — Log-odds Scaling
Sau khi LightGBM trả về PD (probability of default), áp dụng công thức chuẩn ngành ngân hàng để chuyển thành credit score [300–850] trên thang log-odds linear.
def pd_to_credit_score(pd_value, pdo=20, base_score=600, base_odds=50):
    """
    pd_value  : float — output của LightGBM predict_proba[:,1]
    pdo       : Points to Double the Odds (chuẩn ngành = 20)
    base_score: 600 điểm tại Odds = 50:1 (PD ≈ 2%)
    base_odds : 50 — mốc calibration
    """
    pd_value = np.clip(pd_value, 1e-6, 1 - 1e-6)  # tránh log(0)
    factor   = pdo / np.log(2)                      # = 28.85
    offset   = base_score - factor * np.log(base_odds)  # = 487.2
    odds     = (1 - pd_value) / pd_value
    score    = offset + factor * np.log(odds)
    return int(np.clip(score, 300, 850))

# Ví dụ: PD = 0.058 → Score = 487.2 + 28.85 × ln(16.24) = 568

3.3 Risk Band & Decision Rules
Risk Band	Credit Score	PD Range	Decision	Điều kiện cộng thêm
AAA — Xuất sắc	720–850	< 2%	AUTO APPROVE	CIC Nhóm 1 + DTI < 40%
AA — Tốt	640–719	2–8%	APPROVE + REVIEW	Chuyên viên xem nhanh báo cáo
A — Khá	560–639	8–18%	FULL REVIEW	Đánh giá đầy đủ 4C + điều kiện
BBB — Trung bình	460–559	18–35%	CONDITIONAL	Cần bổ sung TSBĐ hoặc guarantor
CC/C — Rủi ro cao	300–459	> 35%	REJECT	Trừ đặc cách có thẩm quyền cao

Hard Override Rules	CIC Nhóm 4-5 → REJECT bất kể ML score. Loan > 10 tỷ VND → ESCALATE to head office. thin_file_flag = True + Score < 560 → tăng yêu cầu tài sản thế chấp.

3.4 SHAP Output Schema — Bridge sang A4
SHAP được tính tại A3 và serialize thành JSON chuẩn. A4 bắt buộc dùng SHAP JSON này làm ground truth — không được thêm reasoning ngoài SHAP. Đây là điểm phân biệt với MASCA (LLM reasoning) → grounded XAI.
shap_output = {
    "credit_score":   672,
    "pd_pct":         5.8,
    "risk_band":      "AA",
    "model_version":  "lgbm_v1_noxmoon_homecredit",
    "inference_ts":   "2026-03-21T10:23:41Z",

    "top_positive_factors": [
        {"feature":"salary_pattern_detected",    "shap":+0.089,"value":true,  "label_vi":"Phát hiện giao dịch lương đều đặn"},
        {"feature":"income_stability_index",     "shap":+0.072,"value":0.81, "label_vi":"Thu nhập ổn định 6 tháng (index 0.81)"},
        {"feature":"agg_prev_score_mean",        "shap":+0.058,"value":0.71, "label_vi":"Điểm lịch sử vay trước đây tốt"},
        {"feature":"regular_bill_payment_ratio", "shap":+0.051,"value":0.90, "label_vi":"Thanh toán hóa đơn đúng hạn 90%"},
        {"feature":"repayment_plan_quality",     "shap":+0.038,"value":3,    "label_vi":"Kế hoạch trả nợ chi tiết (DETAILED)"}
    ],
    "top_negative_factors": [
        {"feature":"dti_ratio",             "shap":-0.063,"value":0.48,"label_vi":"Tỷ lệ nợ/thu nhập ở mức cao (48%)"},
        {"feature":"overdraft_count_6m",    "shap":-0.031,"value":2,  "label_vi":"Số dư về âm 2 lần trong 6 tháng"},
        {"feature":"imputation_confidence", "shap":-0.018,"value":0.81,"label_vi":"Một số trường ước tính (conf 81%)"}
    ],
    "4c_shap_allocation": {
        "character":  {"shap_sum":0.118,"pct":28},
        "capacity":   {"shap_sum":0.172,"pct":41},
        "capital":    {"shap_sum":0.080,"pct":19},
        "conditions": {"shap_sum":0.050,"pct":12}
    },
    "all_features_shap": { ... }  // full dict → DynamoDB audit log
}
4. Module A4 — Report Generator & Explainability Stack
Mục tiêu: Biến SHAP output thành Tờ trình tín dụng chuẩn ngân hàng Việt Nam — có căn cứ, traceable, và actionable. LLM ở đây đóng vai translator, không phải analyst. Áp dụng kiến trúc MASCA (4 tầng agent, contrastive Risk/Reward, Signaling Game Theory) nhưng với SHAP làm ground truth thay vì LLM chain-of-thought.
4.1 MASCA Agent Architecture — Adapted
Agent (MASCA)	Role gốc (MASCA)	Implementation trong CreditLens
Data Analyst	Chuẩn bị raw data, formatting	→ A1: Textract + CIC API + CSV parser (code, không phải LLM)
Contextualizer	Tổng hợp customer persona từ data	→ LLM đọc NoxMoon features đã structured → synthesize narrative context
Feature Engineer	LLM tính DTI, DAR, ratios	→ THAY BẰNG NoxMoon pipeline (code Python) — deterministic, reproducible, AUC 0.80+
Risk Modeler	LLM assess credit risk	→ THAY BẰNG LightGBM + SHAP — accuracy 60% (MASCA) vs AUC 0.80+ (LightGBM)
Income & Stability Analyst	LLM assess income stability	→ income_stability_index từ A1 (từ installments_payments) + SHAP contribution
Debt Analyst	LLM assess loan purpose + DTI	→ GIỮ NGUYÊN: LLM giỏi hơn ở loan purpose context. Input: loan_purpose_cat từ A2 + dti_ratio từ A1
Reward Modeler	LLM assess profitability	→ GIỮ NGUYÊN: đánh giá profitability từ customer segment + credit score + loan margin
Strategic Optimizer	Risk-Reward ratio + scenarios	→ SHAP 4C allocation làm Risk signal. Reward signal từ Reward Modeler. LLM synthesize final recommendation
Decision Orchestrator	LLM final approve/reject	→ THAY BẰNG Credit Score + Business Rules (Policy-based routing). LLM chỉ viết narrative, không quyết định

4.2 3-Tier Explainability Stack
Tầng	Mechanism	Ai dùng
Tier 1 SHAP Attribution	SHAP TreeExplainer — toán học đảm bảo consistency + efficiency + additivity. Mỗi feature có exact contribution đến PD. Same input → same SHAP values	Kiểm toán, compliance officer, model risk management
Tier 2 Grounded LLM Narrative	LLM translate SHAP → tiếng Việt. Hard constraint: chỉ được mention factors trong SHAP top features. Consistency validator check sau khi generate	Chuyên viên tín dụng, giám đốc chi nhánh
Tier 3 Audit Trail	DynamoDB append-only log: mọi agent action — input, output, timestamp, model_version, SHAP JSON hash. Không thể modify sau khi ghi	Regulatory compliance, SBV audit, adverse action notice

4.3 Report Generation Prompt — Hard Constraints
SYSTEM = """
You are a Vietnamese bank credit report writer (chuyên viên thẩm định tín dụng).

HARD RULES — violations = invalid report, re-generate required:
1. ONLY discuss risk/positive factors that appear in the SHAP values provided.
   DO NOT invent new factors not supported by SHAP data.
2. CITE specific policy clauses from the Policy Context (RAG) section.
3. For each negative factor: provide ONE specific, actionable improvement.
4. Write in formal Vietnamese banking language (văn phong ngân hàng).
5. DO NOT reveal model weights, training data, or internal architecture.
"""

USER = f"""
SHAP Analysis JSON: {json.dumps(shap_output, ensure_ascii=False)}
Policy Context (RAG): {rag_context}
Data Warnings: {json.dumps(warnings, ensure_ascii=False)}
Customer Profile Summary: {customer_summary}

Generate JSON report with sections:
  executive_summary, character_assessment, capacity_assessment,
  capital_assessment, conditions_assessment,
  reward_assessment, final_recommendation, caveats
"""

# Consistency Validator — chạy sau khi LLM generate
def validate(shap_output, narrative):
    top_labels = {f["label_vi"] for f in
        shap_output["top_positive_factors"] + shap_output["top_negative_factors"]}
    text = " ".join([narrative[k]["narrative"] for k in narrative])
    coverage = sum(1 for l in top_labels if l.lower() in text.lower())
    shap_coverage = coverage / len(top_labels)
    return {"passed": shap_coverage > 0.6, "shap_coverage": shap_coverage}
# Nếu failed → re-prompt với violation list, max 2 retries
5. Format Báo Cáo Tín Dụng — Tờ Trình Tín Dụng CreditLens
Thiết kế theo chuẩn Tờ trình cấp tín dụng Việt Nam (Circular 39/2016/TT-NHNN, Decision 493/2005/QĐ-NHNN) kết hợp SHAP waterfall visualization và 5C framework. Nguyên tắc: "Quyết định ở trên — chi tiết ở dưới." Chuyên viên nhìn vào trang đầu là biết đề xuất ngay.

5.1 Cấu trúc báo cáo 6 phần
#	Phần báo cáo	Nội dung & Mục tiêu
I	THÔNG TIN KHÁCH HÀNG	Thông tin định danh (họ tên, CCCD, ngày sinh, địa chỉ), nghề nghiệp/doanh nghiệp, quan hệ với ngân hàng, sản phẩm đang sử dụng
II	TÓM TẮT ĐÁNH GIÁ (EXECUTIVE SUMMARY)	Credit Score + Risk Band + Đề xuất + PD% — TRÌNH BÀY ĐẦU TRANG, chiếm 1/3 trang đầu. SHAP waterfall chart (top 5 positive, top 5 negative). Khoản vay đề nghị + điều khoản đề xuất
III	ĐÁNH GIÁ 5C (CHI TIẾT)	Mỗi tiêu chí có: điểm số, trạng thái ĐẠT/XEM XÉT/CHƯA ĐẠT, SHAP contribution %, indicators met/review, LLM narrative 100–150 chữ grounded trên SHAP
IV	TÌNH HÌNH TÀI CHÍNH	Bảng cân đối tài sản đơn giản, thu nhập và chi tiêu, phân tích dòng tiền 6 tháng (từ bank statement), các chỉ số tài chính chính: DTI, DSCR, LTV
V	TÀI SẢN BẢO ĐẢM	Loại TSBĐ, mô tả, giá trị thẩm định, LTV ratio, tình trạng pháp lý, hạn chế (nếu có)
VI	KHUYẾN NGHỊ & CAVEATS	Đề xuất phê duyệt/từ chối/có điều kiện, điều kiện tiên quyết, điều khoản đặc biệt, danh sách cảnh báo dữ liệu, model_version + SHAP hash + RAG chunks → audit reference

5.2 Phần II — Executive Summary (Chi tiết)
Đây là phần quan trọng nhất — chuyên viên tín dụng phải nhìn vào trang đầu là biết quyết định. Thiết kế theo chuẩn credit memo best practices: decision first, details later.

Block A — Scorecard 4 ô (top of page)
ĐIỂM TÍN DỤNG672 / 850
AA — Rủi ro Thấp
	ĐỀ XUẤTPHÊ DUYỆT
Chờ xác nhận điều kiện
	XÁC SUẤT VỠ NỢ5.8%
Nhóm nợ dự kiến: 1
	MÔ HÌNH SỬ DỤNGLightGBM v1
AUC: 0.803 | SHAP verified


Block B — SHAP Waterfall Summary
SHAP waterfall được render inline trong báo cáo HTML/PDF. Bảng text version cho PDF:
Yếu tố	SHAP	Diễn giải (từ label_vi)
+ salary_pattern_detected	+0.089	Phát hiện giao dịch lương đều đặn — xác nhận employment
+ income_stability_index	+0.072	Thu nhập ổn định 6 tháng gần nhất (index 0.81)
+ agg_prev_score_mean	+0.058	Lịch sử vay trước đây — NoxMoon sub-model score tốt (0.71)
+ regular_bill_payment_ratio	+0.051	Thanh toán hóa đơn điện/nước/internet đúng hạn 90% tháng
– dti_ratio	–0.063	DTI = 48% ở mức cao (ngưỡng tốt: < 40%). Cần theo dõi
– overdraft_count_6m	–0.031	Số dư về âm 2 lần trong 6 tháng → dấu hiệu áp lực tài chính nhẹ
Baseline PD (trung bình portfolio)	0.082	Điểm xuất phát (E[f(x)] = 8.2% default rate của dataset)
PD cuối cùng (f(x))	0.058	5.8% → Credit Score: 672 → Risk Band: AA

5.3 Phần III — Đánh giá 5C Chi tiết
5C framework chuẩn ngân hàng Việt Nam (Character, Capacity, Capital, Conditions, Collateral) với SHAP grounding cho từng dimension. Phân bổ SHAP theo 4C (Capital = Collateral trong context này):

Tiêu chí 5C	Điểm	Trạng thái	SHAP % + Nội dung
C1 — Character (Uy tín / Tư cách)	28/30	ĐẠT	SHAP 28% | Không có nợ xấu CIC. Hành vi thanh toán hóa đơn đúng hạn 90%. Định danh nhất quán. Lịch sử vay trước (agg_prev_score_mean=0.71 → SHAP +0.058) cho thấy trách nhiệm tài chính tốt.
→ Indicators NEEDS REVIEW: Thin-file flag (không có CIC history đủ dài → không đủ để kết luận chắc chắn)
C2 — Capacity (Năng lực trả nợ)	31/40	XEM XÉT	SHAP 41% | Thu nhập ổn định (income_stability=0.81 → SHAP +0.072). Lương phát hiện đều đặn (→ SHAP +0.089). DTI = 48% ở mức cao (SHAP –0.063) — gần ngưỡng 50%.
→ Đề xuất: Giảm hạn mức 10–15% hoặc yêu cầu chứng minh thu nhập bổ sung. DSCR = 1.18 (ngưỡng tối thiểu: 1.2).
C3 — Capital (Vốn tự có)	16/20	ĐẠT	SHAP 19% | Inflow/Outflow ratio = 1.24 → healthy cash buffer. Savings behavior tốt. Không có giao dịch bất thường lớn (max_single_outflow_ratio = 0.31).
→ Không có real estate collateral — xem C5 Collateral.
C4 — Conditions (Điều kiện)	9/10	TỐT	SHAP 12% | Mục đích vay PRODUCTION (→ SHAP contribution dương). Kế hoạch trả nợ chi tiết (repayment_quality=DETAILED → SHAP +0.038). Ngành nghề ổn định, điều kiện thị trường thuận lợi.
C5 — Collateral (Tài sản bảo đảm)	14/20	XEM XÉT	(Đánh giá riêng - xem Phần V) | Xe ô tô định giá 450M VND. LTV = 67% (ngưỡng: 70%). Giấy tờ sở hữu hợp lệ. Rủi ro: giá trị tài sản có thể giảm theo thời gian.
→ Đề xuất: Tái định giá sau 12 tháng. Yêu cầu bảo hiểm tài sản.

5.4 Phần VI — Khuyến nghị & Điều kiện
Mục	Nội dung
Đề xuất	PHÊ DUYỆT CÓ ĐIỀU KIỆN — Risk Band AA, PD 5.8%
Số tiền: 300,000,000 VND | Kỳ hạn: 36 tháng | Lãi suất: theo biểu phí hiện hành
Điều kiện tiên quyết	1. Chứng minh thu nhập bổ sung (slip lương 3 tháng gần nhất) 2. Mua bảo hiểm tài sản thế chấp trước giải ngân 3. DTI sau khi vay không vượt 50%
Cảnh báo dữ liệu	• Thu nhập được ước tính từ dữ liệu giao dịch (confidence 81%) — chưa xác minh từ hợp đồng chính thức • 2 trường được impute bởi LLM (xem imputation_log trong audit trail)
Audit Reference	Model: lgbm_v1_noxmoon_homecredit | SHAP hash: sha256:a3f7b2... | RAG chunks: TT39/2016 Điều 17, QĐ493/2005 | Timestamp: 2026-03-21T10:23:41Z
6. LangGraph Orchestration
6.1 State Schema
class CreditState(TypedDict):
    application_id  : str               # SHA-256(applicant_id + timestamp)
    customer_type   : Literal["IND","SME"]
    # A1
    raw_ocr_text    : dict[str, str]     # {doc_type: text}
    structured_feats: dict[str, Any]     # tabular features từ A1
    confidence_map  : dict[str, float]   # per-field
    thin_file_flag  : bool
    # A2
    llm_feats       : dict[str, Any]     # semantic + imputed
    imputation_log  : list[dict]
    warnings        : list[str]
    # A3
    credit_score    : int                # 300–850
    pd_pct          : float
    risk_band       : str
    shap_values     : dict               # full SHAP JSON
    routing         : str                # AUTO_APPROVE|REVIEW|REJECT|ESCALATE
    # A4
    five_c_scores   : dict[str, float]
    narrative       : dict[str, str]
    consistency_check: dict
    final_report    : dict
    audit_trail     : list[dict]         # append-only

6.2 Node Graph
Node	Type	Entry condition	Exit → next
ingest_documents	Tool (Textract)	START	→ check_cic (parallel) + analyze_transactions (parallel)
check_cic	Tool (API)	After ingest	→ join (wait for transactions)
analyze_transactions	Code (Pandas)	After ingest	→ join
llm_feature_engineer	LLM (Claude)	After join	→ noxmoon_submodel_features
noxmoon_submodel_features	Code (LightGBM small models)	After llm_feature_engineer	→ ml_score
ml_score	SageMaker API	After submodel	→ report_generator
report_generator	LLM (Claude)	After ml_score	→ consistency_validator
consistency_validator	Code (deterministic)	After report	→ decision_router
decision_router	Policy rules	After validation	→ END với routing label

6.3 REST API Endpoints
Endpoint	Method	Mô tả
POST /v1/assess	POST	Submit hồ sơ → async pipeline → trả về application_id
GET /v1/assess/{id}	GET	Lấy full report khi hoàn thành (status=COMPLETED)
GET /v1/assess/{id}/status	GET	PROCESSING | REVIEW_NEEDED | COMPLETED | FAILED
POST /v1/human-review/{id}	POST	Chuyên viên submit quyết định cuối → append vào audit_trail
GET /v1/assess/{id}/audit	GET	Full audit trail (SHAP hash, RAG chunks, all agent actions)
7. Demo Design — 4 Scenarios, 8 Phút
Mục tiêu demo: BGK thấy được 3 điểm khác biệt cốt lõi so với traditional scoring: (1) thin-file không bị loại, (2) mọi quyết định traceable đến data point cụ thể, (3) hệ thống có discriminative power — không approve hết.

7.1 Kịch bản demo 8 phút
Phút	Scenario	Nội dung demo	"Wow moment" cần thể hiện
0–1	Slide bài toán	70% dân số VN underbanked. CIC chỉ cover 30.8M người. 100 hồ sơ/ngày/chuyên viên → 2–3 ngày xử lý	Set context — đây là bài toán thực tế, không phải academic
1–3	Demo 1: Thin-file Freelancer	Upload hồ sơ Nguyễn Văn An (freelancer, không CIC). Pipeline chạy: thin_file_flag=TRUE xuất hiện → hệ thống không từ chối mà chuyển alternative scoring path → 90s sau: Score 634, Band A	"Khách hàng này sẽ bị từ chối ngay bởi bất kỳ ngân hàng nào dùng CIC truyền thống"
3–5	Demo 2: SHAP Click-through	Từ báo cáo case 1 hoặc 2 (approved): click vào SHAP bar chart → mỗi bar hiện tooltip giải thích → click vào bar → scroll đến đoạn narrative tương ứng trong báo cáo. Thay đổi 1 feature → score thay đổi, narrative thay đổi nhất quán	"Không phải AI nói chung chung — đây là con số toán học. Bạn có thể audit từng quyết định"
5–6	Demo 3: SME Micro Business	Upload hồ sơ Cửa hàng Hoa Lan (micro SME, 3 năm). LLM web crawl → business profile → revenue_trend từ sao kê → Score 588, Band A, có cảnh báo revenue declining 15%	"LLM đọc được business context mà scorecard cứng không làm được"
6–7	Demo 4: High-risk REJECT	Upload hồ sơ Lê Minh Cường (sinh viên mới đi làm, DTI 71%, overdraft 5 lần). Score 421, Band CC → REJECT với specific reasons từ SHAP	"Hệ thống có discriminative power thực sự — không approve hết để làm vui"
7–8	Ablation Study Slide	Bảng E0→E4: LR baseline (0.65) → LightGBM (0.75) → + alt data (0.79) → + LLM features (0.81) → full system (0.82+). Quantify contribution mỗi component	"Mọi component đều có lý do tồn tại — không phải thêm AI vào cho có"

7.2 4 Test Cases — Thiết kế từ Home Credit
Case	Profile	Home Credit analog	Expected output & why
Case 1	Nguyễn Văn An Freelancer, không CIC 6 tháng sao kê	SK_ID = thin-file records trong Home Credit. EXT_SOURCE = null. income_stability_index từ installments_payments	Score ~620–650 (Band A). Alternative path. Thin-file flag visible trong report. Moment: BGK thấy hệ thống serve underbanked
Case 2	Trần Thị Bình NV ngân hàng CIC đầy đủ, vay mua xe	Standard application. Tốt mọi chiều. EXT_SOURCE cao. installments history sạch	Score ~700–720 (Band AA). AUTO APPROVE. Moment: SHAP chart đẹp, mọi bars đều dương hoặc nhỏ âm
Case 3	Cửa hàng Hoa Lan Micro SME, 3 năm Vay vốn lưu động	SME proxy: NAME_CONTRACT_TYPE=Revolving, business_age từ DAYS_EMPLOYED proxy	Score ~575–600 (Band A). XEM XÉT. Revenue declining flag. Moment: LLM web crawl + business profile trong báo cáo
Case 4	Lê Minh Cường SV mới đi làm 4 tháng DTI 71%, overdraft 5 lần	High-risk pattern: DAYS_EMPLOYED thấp, AMT_ANNUITY/INCOME cao, DPD > 0 nhiều	Score ~400–440 (Band CC). REJECT. SHAP bars đỏ chiếm ưu thế. Moment: hệ thống reject đúng, không approve hết

7.3 Tech Stack Demo — Next.js Dashboard
•	Frontend: Next.js + Tailwind CSS + Recharts. Upload form → processing animation → report viewer
•	SHAP visualization: Recharts horizontal bar chart — bars màu xanh (positive) / đỏ (negative), click bar → scroll đến narrative tương ứng
•	Processing animation: Hiện từng bước pipeline (A1→A2→A3→A4) với timing thực tế. thin_file_flag=TRUE làm bước CIC đổi màu amber
•	Report viewer: Tờ trình tín dụng 6 phần, có thể print/export PDF. Mobile-friendly cho BGK dùng điện thoại
•	2 điều cần tránh: (1) Không demo real-time 90s — dùng pre-computed results với animation. (2) Không dùng data quá lý tưởng — Case 3 (declining revenue) và Case 4 (reject) chứng minh hệ thống realistic
8. Chiến Lược Đánh Giá
8.1 Quantitative Metrics
Metric	Target	Tại sao	Cách đo
AUC-ROC	> 0.80	Benchmark NoxMoon single model. Không bị ảnh hưởng 8% class imbalance	sklearn trên 15% held-out test set từ Home Credit
Gini Coefficient	> 0.60	Standard credit scoring metric	= 2 × AUC − 1
KS Statistic	> 0.40	Separation power giữa good/bad credit	max(TPR − FPR) qua tất cả thresholds
Pilot Accuracy (40 cases)	≥ 38/40 = 95%	Yêu cầu cụ thể hackathon	Full pipeline — so sánh routing decision với expert label
SHAP Consistency Score	> 0.70	Validate explainability quality	shap_coverage trong consistency_validator
Thin-file sub-AUC	> 0.72	Validate alternative data path	Sub-group AUC trên records có thin_file_flag=True

8.2 Ablation Study — Justify mọi component
Experiment	AUC expected	Δ AUC	Component được justify
E0: Logistic Regression, CIC only (traditional baseline)	~0.65	—	Traditional scoring lower bound
E1: LightGBM, tabular features only (no alt data)	~0.75	+0.10	Value of ML over logistic regression
E2: + Transaction alternative data + NoxMoon sub-models	~0.79	+0.04	Value of NoxMoon FE + alternative data
E3: + LLM Semantic features (A2-A)	~0.81	+0.02	Value of unstructured doc analysis
E4: + LLM Imputation (A2-B) — Full System	> 0.82	+0.01	Value of intelligent imputation
9. Hạ Tầng AWS — Pilot Stack
AWS Service	Role	Module	Cost/month (pilot)
Amazon Textract	OCR + Analyze Lending API	A1	~$15
Amazon Bedrock (Claude 3.5 Sonnet)	LLM inference — A2 extraction + A4 report	A2, A4	~$30–50
Amazon SageMaker RT Endpoint	LightGBM hosting + SHAP inference	A3	~$50 (ml.t3.medium)
Amazon OpenSearch Serverless	RAG vector store — policy docs	A4	~$100
AWS Lambda + API Gateway	REST endpoints + LangGraph runner	All	~$5
Amazon DynamoDB	CreditState store + audit trail	All	~$5
Amazon S3	Document storage + model artifacts	All	~$3
TỔNG PILOT			~$210–230/month
Phụ Lục — Tài liệu & Mã nguồn Tham khảo
Mã nguồn trực tiếp sử dụng
•	NoxMoon Home Credit: github.com/NoxMoon/home-credit-default-risk — Rank #17/7198. Notebooks: prev_training, buro_training, month_training, inst-ts, pos-ts, cc-ts, bubl-ts, house-doc-feats, lgb1/2/3, ensembling
•	MASCA Paper: arXiv:2507.22758 — LLM Multi-Agent System. Lấy: kiến trúc agent layers, 5C assessment framework, contrastive Risk/Reward, Signaling Game Theory
•	SHAP: shap.readthedocs.io — TreeExplainer, waterfall plots, global/local explanation
Quy định & Chuẩn mực ngân hàng Việt Nam
•	Thông tư 39/2016/TT-NHNN — Hoạt động cho vay của TCTD
•	Quyết định 493/2005/QĐ-NHNN — Phân loại nợ và trích lập dự phòng rủi ro
•	Quyết định 18/2007/QĐ-NHNN — Xếp hạng tín dụng nội bộ (sửa đổi 493)
•	CIC Trung tâm Thông tin Tín dụng Quốc gia — Thang điểm 300–850, 5 nhóm nợ
Nghiên cứu nền tảng
•	Cash Flow Underwriting with Bank Transaction Data — Ng et al. (2025), arXiv:2510.16066
•	Explaining Deep Learning Models for Credit Scoring with SHAP — MDPI Journal of Risk and Financial Management (2023)
•	Credit Scoring Approaches Guidelines — World Bank Group (2019)
•	Algorithmic Credit Scoring in Vietnam: A Legal Proposal — Asian Journal of Law and Society, Cambridge (2023)
