const CUSTOMERS = [
  {
    id: "CL-2026-001",
    name: "Nguyễn Văn Minh",
    initials: "NV",
    avatarColor: "#1565c0",
    tag: "Standard",
    tagClass: "tag-standard",
    info: "NV Ngân hàng · 35 tuổi · 300M VNĐ",
    result: { score: 642, band: "AA", pd: "7.73%", decision: "APPROVE", decClass: "badge-approve", scoreClass: "score-green", totalTime: "8.2s" },
    pipeline: [
      {
        label: "A1", cls: "a1", title: "Data Ingestion", meta: "121 fields · 3 src", scoreNum: "121", scoreLabel: "fields ✓", scoreClass: "green",
        steps: [
          { icon: 'doc', name: "PDF → OCR", sub: "45 fields" },
          { icon: 'grid', name: "CIC API", sub: "Score · nợ" },
          { icon: 'db', name: "Bank CSV", sub: "6 features" },
          { icon: 'info', name: "EXT_SRC", sub: "Synthesis" }
        ]
      },
      {
        label: "A2", cls: "a2", title: "LLM Feature Engineering", meta: "753 features · Gemini", scoreNum: "753", scoreLabel: "feats ✓", scoreClass: "yellow",
        steps: [
          { icon: 'search', name: "Semantic", sub: "purpose" },
          { icon: 'home', name: "Impute", sub: "81% conf" },
          { icon: 'layers', name: "NixMoon", sub: "FE slice" },
          { icon: 'check', name: "Validate", sub: "Check ✓" }
        ]
      },
      {
        label: "A3", cls: "a3", title: "ML Scoring", meta: "Score: 642 · PD: 7.73%", scoreNum: "642", scoreLabel: "Score ✓", scoreClass: "green",
        steps: [
          { icon: 'bars', name: "LightGBM", sub: "PD 7.73%" },
          { icon: 'target', name: "Score Map", sub: "642/800" },
          { icon: 'cube', name: "MASCA", sub: "adjust" },
          { icon: 'shap', name: "SHAP", sub: "+2 feat ↑" },
          { icon: 'shield', name: "Risk Band", sub: "AA" }
        ]
      },
      {
        label: "A4", cls: "a4", title: "Report Generation", meta: "5C: 84/125 · PDF ready", scoreNum: "84", scoreLabel: "5C pre ✓", scoreClass: "yellow",
        steps: [
          { icon: 'monitor', name: "MASCA", sub: "variables" },
          { icon: 'star', name: "5C Score", sub: "84/125" },
          { icon: 'checkfull', name: "Consistency", sub: "PASSED" },
          { icon: 'filedoc', name: "PDF Gen", sub: "ready" }
        ]
      }
    ],
    modal: {
      sub: "NV Ngân hàng · Standard · Mã hồ sơ: CL-2026-001",
      scoreCards: [
        { label: "ĐIỂM TÍN DỤNG", value: "642", sub: "AA — Rủi ro Thấp", cls: "blue" },
        { label: "ĐỀ XUẤT", value: "APPROVE", sub: "Chờ xác nhận", cls: "green" },
        { label: "XÁC SUẤT VỠ NỢ", value: "7.73%", sub: "Nhóm nợ: 1", cls: "orange" },
        { label: "MÔ HÌNH", value: "LightGBM", sub: "AUC: 0.803", cls: "purple" }
      ],
      fiveC: [
        { name: "Character", score: "22/30", cls: "good", tag: "ĐẠT", tagCls: "pass" },
        { name: "Capacity", score: "28/40", cls: "ok", tag: "XEM XÉT", tagCls: "review" },
        { name: "Capital", score: "10/20", cls: "good", tag: "ĐẠT", tagCls: "pass" },
        { name: "Conditions", score: "8/10", cls: "good", tag: "TỐT", tagCls: "pass" },
        { name: "Collateral", score: "5/20", cls: "bad", tag: "YẾU", tagCls: "fail" }
      ],
      shapPos: [
        { feat: "salary_pattern", val: "+0.089", pct: 89, desc: "Thu nhập ổn định" },
        { feat: "income_stability", val: "+0.072", pct: 72, desc: "Index = 0.81" },
        { feat: "prev_score", val: "+0.058", pct: 58, desc: "Lịch sử tốt" },
        { feat: "bill_payment", val: "+0.051", pct: 51, desc: "Đúng hạn 90%" }
      ],
      shapNeg: [
        { feat: "dti_ratio", val: "-0.063", pct: 63, desc: "DTI = 48%" },
        { feat: "overdraft_cnt", val: "-0.031", pct: 31, desc: "Âm 2 lần" },
        { feat: "impute_conf", val: "-0.018", pct: 18, desc: "Conf 81%" }
      ]
    }
  },
  {
    id: "CL-2026-002",
    name: "Phạm Thị Lan",
    initials: "PT",
    avatarColor: "#6a1b9a",
    tag: "Thin-file",
    tagClass: "tag-thin-file",
    info: "Freelancer · không CIC · 150M VNĐ",
    result: { score: 600, band: "A", pd: "11.94%", decision: "REVIEW", decClass: "badge-review", scoreClass: "score-yellow", totalTime: "9.4s" },
    pipeline: [
      {
        label: "A1", cls: "a1", title: "A1 · Data Ingestion", meta: "87 fields · 2 src (no CIC)", scoreNum: "87", scoreLabel: "fields ✓", scoreClass: "green",
        steps: [
          { icon: 'doc', name: "PDF → OCR", sub: "42 fields" },
          { icon: 'grid', name: "CIC API", sub: "No data" },
          { icon: 'db', name: "Bank CSV", sub: "5 features" },
          { icon: 'info', name: "EXT_SRC", sub: "Imputed" }
        ]
      },
      {
        label: "A2", cls: "a2", title: "A2 · LLM Feature Engineering", meta: "753 features · NaN fill", scoreNum: "753", scoreLabel: "feats ~", scoreClass: "yellow",
        steps: [
          { icon: 'search', name: "Semantic", sub: "freelance" },
          { icon: 'home', name: "Impute", sub: "68% conf" },
          { icon: 'layers', name: "NixMoon", sub: "NaN paths" },
          { icon: 'check', name: "Validate", sub: "Warn ⚠" }
        ]
      },
      {
        label: "A3", cls: "a3", title: "A3 · ML Scoring", meta: "Score: 600 · PD: 11.94%", scoreNum: "600", scoreLabel: "Score ~", scoreClass: "yellow",
        steps: [
          { icon: 'bars', name: "LightGBM", sub: "PD 11.94%" },
          { icon: 'target', name: "Score Map", sub: "600/800" },
          { icon: 'cube', name: "MASCA", sub: "adjust" },
          { icon: 'shap', name: "SHAP", sub: "NaN flag" },
          { icon: 'shield', name: "Risk Band", sub: "A" }
        ]
      },
      {
        label: "A4", cls: "a4", title: "A4 · Report Generation", meta: "5C: 68/125 · Thin-file note", scoreNum: "68", scoreLabel: "5C pre ~", scoreClass: "yellow",
        steps: [
          { icon: 'monitor', name: "MASCA", sub: "thin-file" },
          { icon: 'star', name: "5C Score", sub: "68/125" },
          { icon: 'checkfull', name: "Consistency", sub: "WARN" },
          { icon: 'filedoc', name: "PDF Gen", sub: "ready" }
        ]
      }
    ],
    modal: {
      sub: "Freelancer · Thin-file · Mã hồ sơ: CL-2026-002",
      scoreCards: [
        { label: "ĐIỂM TÍN DỤNG", value: "600", sub: "A — Khá", cls: "blue" },
        { label: "ĐỀ XUẤT", value: "REVIEW", sub: "Cần xem xét thêm", cls: "orange" },
        { label: "XÁC SUẤT VỠ NỢ", value: "11.94%", sub: "Nhóm nợ: 2", cls: "orange" },
        { label: "MÔ HÌNH", value: "LightGBM", sub: "AUC: 0.803", cls: "purple" }
      ],
      fiveC: [
        { name: "Character", score: "18/30", cls: "ok", tag: "XEM XÉT", tagCls: "review" },
        { name: "Capacity", score: "20/40", cls: "bad", tag: "YẾU", tagCls: "fail" },
        { name: "Capital", score: "12/20", cls: "good", tag: "ĐẠT", tagCls: "pass" },
        { name: "Conditions", score: "8/10", cls: "good", tag: "TỐT", tagCls: "pass" },
        { name: "Collateral", score: "10/25", cls: "bad", tag: "YẾU", tagCls: "fail" }
      ],
      shapPos: [
        { feat: "capital_ratio", val: "+0.071", pct: 71, desc: "Vốn đủ" },
        { feat: "loan_purpose", val: "+0.055", pct: 55, desc: "Mục đích rõ" },
        { feat: "age_feature", val: "+0.038", pct: 38, desc: "Tuổi phù hợp" }
      ],
      shapNeg: [
        { feat: "cic_missing", val: "-0.091", pct: 91, desc: "Không có CIC" },
        { feat: "income_var", val: "-0.072", pct: 72, desc: "Thu nhập biến động" },
        { feat: "impute_nan", val: "-0.041", pct: 41, desc: "NaN 14 features" },
        { feat: "dti_ratio", val: "-0.035", pct: 35, desc: "DTI = 52%" }
      ]
    }
  },
  {
    id: "CL-2026-003",
    name: "Trần Văn Đức",
    initials: "TV",
    avatarColor: "#2e7d32",
    tag: "SME",
    tagClass: "tag-sme",
    info: "Chủ cửa hàng · SME · 500M VNĐ",
    result: { score: 593, band: "A", pd: "12.82%", decision: "REVIEW", decClass: "badge-review", scoreClass: "score-yellow", totalTime: "10.1s" },
    pipeline: [
      {
        label: "A1", cls: "a1", title: "A1 · Data Ingestion", meta: "134 fields · 3 src + biz", scoreNum: "134", scoreLabel: "fields ✓", scoreClass: "green",
        steps: [
          { icon: 'doc', name: "PDF → OCR", sub: "52 fields" },
          { icon: 'grid', name: "CIC API", sub: "Score B+" },
          { icon: 'db', name: "Bank CSV", sub: "10 feats" },
          { icon: 'info', name: "BIZ DATA", sub: "Revenue" }
        ]
      },
      {
        label: "A2", cls: "a2", title: "A2 · LLM Feature Engineering", meta: "753 features · SME logic", scoreNum: "753", scoreLabel: "feats ✓", scoreClass: "yellow",
        steps: [
          { icon: 'search', name: "Semantic", sub: "business" },
          { icon: 'home', name: "Impute", sub: "74% conf" },
          { icon: 'layers', name: "NixMoon", sub: "SME path" },
          { icon: 'check', name: "Validate", sub: "Check ✓" }
        ]
      },
      {
        label: "A3", cls: "a3", title: "A3 · ML Scoring", meta: "Score: 593 · PD: 12.82%", scoreNum: "593", scoreLabel: "Score ~", scoreClass: "yellow",
        steps: [
          { icon: 'bars', name: "LightGBM", sub: "PD 12.82%" },
          { icon: 'target', name: "Score Map", sub: "593/800" },
          { icon: 'cube', name: "MASCA", sub: "SME adj" },
          { icon: 'shap', name: "SHAP", sub: "biz feat" },
          { icon: 'shield', name: "Risk Band", sub: "A" }
        ]
      },
      {
        label: "A4", cls: "a4", title: "A4 · Report Generation", meta: "5C: 76/125 · SME note", scoreNum: "76", scoreLabel: "5C pre ~", scoreClass: "yellow",
        steps: [
          { icon: 'monitor', name: "MASCA", sub: "SME vars" },
          { icon: 'star', name: "5C Score", sub: "76/125" },
          { icon: 'checkfull', name: "Consistency", sub: "PASSED" },
          { icon: 'filedoc', name: "PDF Gen", sub: "ready" }
        ]
      }
    ],
    modal: {
      sub: "Chủ cửa hàng · SME · Mã hồ sơ: CL-2026-003",
      scoreCards: [
        { label: "ĐIỂM TÍN DỤNG", value: "593", sub: "A — Khá", cls: "blue" },
        { label: "ĐỀ XUẤT", value: "REVIEW", sub: "Kiểm tra thêm", cls: "orange" },
        { label: "XÁC SUẤT VỠ NỢ", value: "12.82%", sub: "Nhóm nợ: 2", cls: "orange" },
        { label: "MÔ HÌNH", value: "LightGBM", sub: "AUC: 0.803", cls: "purple" }
      ],
      fiveC: [
        { name: "Character", score: "20/30", cls: "good", tag: "ĐẠT", tagCls: "pass" },
        { name: "Capacity", score: "22/40", cls: "ok", tag: "XEM XÉT", tagCls: "review" },
        { name: "Capital", score: "14/20", cls: "good", tag: "ĐẠT", tagCls: "pass" },
        { name: "Conditions", score: "8/10", cls: "good", tag: "TỐT", tagCls: "pass" },
        { name: "Collateral", score: "12/25", cls: "ok", tag: "XEM XÉT", tagCls: "review" }
      ],
      shapPos: [
        { feat: "biz_revenue", val: "+0.081", pct: 81, desc: "Doanh thu ổn" },
        { feat: "cic_history", val: "+0.062", pct: 62, desc: "Lịch sử vay B+" },
        { feat: "bank_balance", val: "+0.044", pct: 44, desc: "Số dư tốt" }
      ],
      shapNeg: [
        { feat: "dti_biz", val: "-0.078", pct: 78, desc: "DTI biz = 55%" },
        { feat: "cashflow_var", val: "-0.055", pct: 55, desc: "Dòng tiền biến động" },
        { feat: "collateral_gap", val: "-0.032", pct: 32, desc: "Tài sản đảm bảo thấp" }
      ]
    }
  },
  {
    id: "CL-2026-004",
    name: "Lê Minh Cường",
    initials: "LM",
    avatarColor: "#b71c1c",
    tag: "High-risk",
    tagClass: "tag-high-risk",
    info: "SV mới đi làm · vay tiêu dùng · 50M",
    result: { score: 406, band: "CC", pd: "50.61%", decision: "REJECT", decClass: "badge-reject", scoreClass: "score-red", totalTime: "7.8s" },
    pipeline: [
      {
        label: "A1", cls: "a1", title: "A1 · Data Ingestion", meta: "78 fields · 2 src", scoreNum: "78", scoreLabel: "fields ✓", scoreClass: "green",
        steps: [
          { icon: 'doc', name: "PDF → OCR", sub: "38 fields" },
          { icon: 'grid', name: "CIC API", sub: "Nợ xấu" },
          { icon: 'db', name: "Bank CSV", sub: "4 feats" },
          { icon: 'info', name: "EXT_SRC", sub: "Low conf" }
        ]
      },
      {
        label: "A2", cls: "a2", title: "A2 · LLM Feature Engineering", meta: "753 features · high-risk flags", scoreNum: "753", scoreLabel: "feats ⚠", scoreClass: "yellow",
        steps: [
          { icon: 'search', name: "Semantic", sub: "consumer" },
          { icon: 'home', name: "Impute", sub: "61% conf" },
          { icon: 'layers', name: "NixMoon", sub: "HR path" },
          { icon: 'check', name: "Validate", sub: "Warn ⚠" }
        ]
      },
      {
        label: "A3", cls: "a3", title: "A3 · ML Scoring", meta: "Score: 406 · PD: 50.61%", scoreNum: "406", scoreLabel: "Score ✗", scoreClass: "red-score",
        steps: [
          { icon: 'bars', name: "LightGBM", sub: "PD 50.61%" },
          { icon: 'target', name: "Score Map", sub: "406/800" },
          { icon: 'cube', name: "MASCA", sub: "no adjust" },
          { icon: 'shap', name: "SHAP", sub: "-5 feat ↓" },
          { icon: 'shield', name: "Risk Band", sub: "CC" }
        ]
      },
      {
        label: "A4", cls: "a4", title: "A4 · Report Generation", meta: "5C: 38/125 · REJECT", scoreNum: "38", scoreLabel: "5C FAIL", scoreClass: "red-score",
        steps: [
          { icon: 'monitor', name: "MASCA", sub: "risk vars" },
          { icon: 'star', name: "5C Score", sub: "38/125" },
          { icon: 'checkfull', name: "Consistency", sub: "FAILED" },
          { icon: 'filedoc', name: "PDF Gen", sub: "reject" }
        ]
      }
    ],
    modal: {
      sub: "SV mới đi làm · High-risk · Mã hồ sơ: CL-2026-004",
      scoreCards: [
        { label: "ĐIỂM TÍN DỤNG", value: "406", sub: "CC — Rủi ro Cao", cls: "blue" },
        { label: "ĐỀ XUẤT", value: "REJECT", sub: "Không đủ điều kiện", cls: "orange" },
        { label: "XÁC SUẤT VỠ NỢ", value: "50.61%", sub: "Nhóm nợ: 4", cls: "orange" },
        { label: "MÔ HÌNH", value: "LightGBM", sub: "AUC: 0.803", cls: "purple" }
      ],
      fiveC: [
        { name: "Character", score: "8/30", cls: "bad", tag: "YẾU", tagCls: "fail" },
        { name: "Capacity", score: "10/40", cls: "bad", tag: "YẾU", tagCls: "fail" },
        { name: "Capital", score: "5/20", cls: "bad", tag: "YẾU", tagCls: "fail" },
        { name: "Conditions", score: "7/10", cls: "ok", tag: "ĐẠT", tagCls: "pass" },
        { name: "Collateral", score: "8/25", cls: "bad", tag: "YẾU", tagCls: "fail" }
      ],
      shapPos: [
        { feat: "loan_purpose", val: "+0.041", pct: 41, desc: "Mục đích rõ" },
        { feat: "age_range", val: "+0.022", pct: 22, desc: "Tuổi trẻ" }
      ],
      shapNeg: [
        { feat: "bad_debt_hist", val: "-0.142", pct: 100, desc: "Nợ xấu nhóm 4" },
        { feat: "income_low", val: "-0.098", pct: 98, desc: "Thu nhập < 7M" },
        { feat: "ext_source_low", val: "-0.081", pct: 81, desc: "EXT thấp" },
        { feat: "dti_ratio", val: "-0.071", pct: 71, desc: "DTI = 78%" },
        { feat: "overdraft_freq", val: "-0.055", pct: 55, desc: "Âm 6 lần" }
      ]
    }
  }
];
