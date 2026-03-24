// Mock Data for Frontend Testing and Fallback

export const CUSTOMERS = [
  { id: '001', label: 'Customer #001', folderId: 'customer_001', info: { gender: 'Female', age: 60, income: 'Retired', loan: 'Consumption', amount: '900k VND' } },
  { id: '002', label: 'Customer #002', folderId: 'customer_002', info: { gender: 'Male', age: 35, income: 'Business', loan: 'Real Estate', amount: '2.5M VND' } },
  { id: '003', label: 'Customer #003', folderId: 'customer_003', info: { gender: 'Female', age: 42, income: 'Employed', loan: 'Consumption', amount: '450k VND' } },
  { id: '004', label: 'Customer #004', folderId: 'customer_004', info: { gender: 'Male', age: 28, income: 'Salary', loan: 'Auto Loan', amount: '1.2M VND' } },
  { id: '005', label: 'Customer #005', folderId: 'customer_005', info: { gender: 'Male', age: 50, income: 'Freelance', loan: 'Business', amount: '3M VND' } },
]

export const PIPELINE_LAYERS = [
  {
    id: 'A1', title: 'Layer 01: Data Ingestion & Statistical Feature Pipeline', sub: 'Dataset ground truth verification', color: '#0f9d58', bgColor: '#e6f4ea', badgeBg: '#0f9d58',
    nodes: [
      { icon: 'description', label: 'App Row JSON', detail: '122 cột' },
      { icon: 'grid_view', label: 'CIC API', detail: 'Bureau' },
      { icon: 'storage', label: 'Internal DB', detail: 'Prev Loans' },
    ],
  },
  {
    id: 'A2', title: 'Layer 02: LLM Feature Engineer Layer', sub: 'Gemini Pro 1.5 Synthesis', color: '#0f9d58', bgColor: '#e6f4ea', badgeBg: '#f4b400',
    nodes: [
      { icon: 'search', label: 'Semantic', detail: 'LLM Extract' },
      { icon: 'upload_file', label: 'Impute', detail: 'Điền NaN' },
      { icon: 'layers', label: 'FE Build', detail: '210+753' },
      { icon: 'task_alt', label: 'Purpose', detail: 'Loan Type' },
    ],
  },
  {
    id: 'A3', title: 'Layer 03: Core ML Scoring Engine', sub: 'Ensemble Probability Model', color: '#d23f31', bgColor: '#fce8e6', badgeBg: '#00897b',
    nodes: [
      { icon: 'bar_chart', label: 'LightGBM', detail: 'Dự đoán' },
      { icon: 'my_location', label: 'Score Map', detail: 'PD+300-850' },
      { icon: 'hexagon', label: 'Decision', detail: 'Hard Rules' },
      { icon: 'view_in_ar', label: 'SHAP', detail: 'Giải thích' },
      { icon: 'shield', label: 'Risk Band', detail: 'AA' },
    ],
  },
  {
    id: 'A4', title: 'Layer 04: Report Generator Agent (A4) & Policy RAG', sub: 'Multi-dimensional Audit & Synthesis Service', color: '#5f6368', bgColor: '#f1f3f4', badgeBg: '#5f6368',
    nodes: [
      { icon: 'search', label: 'Policy RAG', detail: 'OPENSEARCH EMBEDS' },
      { icon: 'document_scanner', label: 'Contextual', detail: 'RETRIEVAL ENGINE' },
      { icon: 'translate', label: '5C Narrative', detail: '11 SECTIONS [VN]' },
      { icon: 'fact_check', label: 'Consistency', detail: 'SHAP-GROUNDED VAL' },
      { icon: 'picture_as_pdf', label: 'PDF Output', detail: 'FINAL DOSSIER' },
    ],
  },
]

export const reportFallbackData = {
  report_data: {
    customer_info: {
      name: "Local Mock User",
      income_type: "Tech Lead",
      loan_purpose: "Hacker House Building",
      gender: "Male",
      education: "University",
      housing: "Rented Apartment",
      summary: "Highly reliable candidate with strong technical background and stable income streams."
    },
    executive_summary: {
      credit_score: 792,
      risk_band: "A",
      pd_pct: 1.15,
      five_c_total: 94,
      five_c_scores: { character: 28, capacity: 33, capital: 15, collateral: 10, conditions: 8 }
    }
  },
  shap_data: {
    top_positive_factors: [
      { label_vi: "Excellent payment history", shap_value: 0.22 },
      { label_vi: "Low debt utilization", shap_value: 0.15 },
      { label_vi: "Long credit history", shap_value: 0.08 }
    ],
    top_negative_factors: [
      { label_vi: "Recent credit inquiries", shap_value: -0.06 },
      { label_vi: "High unsecured loan count", shap_value: -0.03 }
    ]
  }
}
