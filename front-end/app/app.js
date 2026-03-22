// === ICON SVG LIBRARY ===
const ICONS = {
  doc: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
  grid: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>`,
  db: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>`,
  info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  home: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="22 11 12 2 2 11"/><path d="M7 11v10a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V11"/></svg>`,
  layers: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  bars: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`,
  target: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>`,
  cube: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>`,
  shap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
  shield: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
  monitor: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>`,
  star: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>`,
  checkfull: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  filedoc: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="15" x2="15" y2="15"/><line x1="9" y1="11" x2="15" y2="11"/></svg>`,
  arrow: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>`,
  checkmark: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>`,
  chartline: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
  download: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
  shield2: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
  plus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
  minus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
  warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
};

// === CONFIG ===
const API_BASE = 'http://localhost:8000';

// === STATE ===
let selected = new Set([0]);
let lastResults = [];
let currentModalData = null;

// Customer metadata (static demo info — real scores come from API)
const CUSTOMER_META = [
  {
    id: "001", name: "Nguyễn Văn Minh", initials: "NV", avatarColor: "#1565c0",
    tag: "Standard", tagClass: "tag-standard",
    fields: [
      { label:"Nghề nghiệp",   value:"NV Ngân hàng" },
      { label:"Tuổi",          value:"35 tuổi" },
      { label:"Thu nhập/T",    value:"22M VNĐ",    cls:"green" },
      { label:"Khoản vay",     value:"300M VNĐ" },
      { label:"CIC",           value:"Có hồ sơ",   cls:"green" },
      { label:"PD dự kiến",    value:"~7-12%",     cls:"yellow" },
    ]
  },
  {
    id: "002", name: "Phạm Thị Lan", initials: "PT", avatarColor: "#6a1b9a",
    tag: "Thin-file", tagClass: "tag-thin-file",
    fields: [
      { label:"Nghề nghiệp",   value:"Freelancer" },
      { label:"Tuổi",          value:"28 tuổi" },
      { label:"Thu nhập/T",    value:"~12M VNĐ",   cls:"yellow" },
      { label:"Khoản vay",     value:"150M VNĐ" },
      { label:"CIC",           value:"Không có",   cls:"red" },
      { label:"PD dự kiến",    value:"~10-20%",    cls:"yellow" },
    ]
  },
  {
    id: "003", name: "Trần Văn Đức", initials: "TV", avatarColor: "#2e7d32",
    tag: "SME", tagClass: "tag-sme",
    fields: [
      { label:"Nghề nghiệp",   value:"Chủ cửa hàng" },
      { label:"Tuổi",          value:"42 tuổi" },
      { label:"Doanh thu/T",   value:"~80M VNĐ",   cls:"green" },
      { label:"Khoản vay",     value:"500M VNĐ" },
      { label:"CIC",           value:"Có (B+)",    cls:"yellow" },
      { label:"PD dự kiến",    value:"~12-18%",    cls:"yellow" },
    ]
  },
  {
    id: "004", name: "Lê Minh Cường", initials: "LM", avatarColor: "#b71c1c",
    tag: "High-risk", tagClass: "tag-high-risk",
    fields: [
      { label:"Nghề nghiệp",   value:"Mới đi làm" },
      { label:"Tuổi",          value:"23 tuổi" },
      { label:"Thu nhập/T",    value:"~6M VNĐ",    cls:"red" },
      { label:"Khoản vay",     value:"50M VNĐ" },
      { label:"CIC",           value:"Nợ xấu G4",  cls:"red" },
      { label:"PD dự kiến",    value:">40%",        cls:"red" },
    ]
  },
];

// Pipeline stage definitions (static - visual only)
const PIPELINE_STAGES = [
  { label:"A1", cls:"a1", title:"Data Ingestion", metaFn: (r) => `${r._a1_fields||'—'} fields · nguồn dữ liệu`,
    steps: [{icon:'doc',name:"PDF → OCR",sub:"Trích xuất"},{icon:'grid',name:"CIC API",sub:"Lịch sử"},{icon:'db',name:"Bank CSV",sub:"Sao kê"},{icon:'info',name:"EXT_SRC",sub:"Tổng hợp"}],
    scoreFn: (r) => ({ num: r._a1_fields||'✓', label:"fields ✓", cls:"green" }) },
  { label:"A2", cls:"a2", title:"LLM Feature Engineering", metaFn: (r) => `${r._a2_feats||753} features · Gemini`,
    steps: [{icon:'search',name:"Semantic",sub:"Phân tích"},{icon:'home',name:"Impute",sub:"Điền NaN"},{icon:'layers',name:"NixMoon",sub:"FE slice"},{icon:'check',name:"Validate",sub:"Kiểm tra"}],
    scoreFn: (r) => ({ num:"753", label:"feats ✓", cls:"yellow" }) },
  { label:"A3", cls:"a3", title:"ML Scoring", metaFn: (r) => `Score: ${r.credit_score} · PD: ${(r.pd_probability*100).toFixed(2)}%`,
    steps: [{icon:'bars',name:"LightGBM",sub:"Dự đoán"},{icon:'target',name:"Score Map",sub:"Chuẩn hóa"},{icon:'cube',name:"MASCA",sub:"Điều chỉnh"},{icon:'shap',name:"SHAP",sub:"Giải thích"},{icon:'shield',name:"Risk Band",sub:r => r.risk_band||'—'}],
    scoreFn: (r) => ({ num: r.credit_score, label:"Score ✓", cls: r.credit_score>=600?"green":r.credit_score>=450?"yellow":"red-score" }) },
  { label:"A4", cls:"a4", title:"Report Generation", metaFn: (r) => `5C: ${r.five_c_total||'—'}/125 · ${r.recommendation}`,
    steps: [{icon:'monitor',name:"MASCA",sub:"Biến đầu vào"},{icon:'star',name:"5C Score",sub:"Đánh giá"},{icon:'checkfull',name:"Consistency",sub: r => r.consistency_check?"PASSED":"WARN"},{icon:'filedoc',name:"PDF Gen",sub:"Xuất báo cáo"}],
    scoreFn: (r) => ({ num: r.five_c_total||'—', label:"5C pts ✓", cls: r.five_c_total>=80?"green":r.five_c_total>=50?"yellow":"red-score" }) },
];

// Map API decision → badge class
function decisionBadge(dec) {
  if (!dec) return { cls:'badge-review', label:'REVIEW' };
  const d = dec.toUpperCase();
  if (d === 'APPROVE') return { cls:'badge-approve', label:'APPROVE' };
  if (d === 'REJECT')  return { cls:'badge-reject',  label:'REJECT' };
  return { cls:'badge-review', label:'REVIEW' };
}

function scoreClass(score) {
  if (score >= 620) return 'score-green';
  if (score >= 500) return 'score-yellow';
  if (score >= 400) return 'score-orange';
  return 'score-red';
}

// Resolve step sub (can be a function of result)
function resolveStepSub(sub, result) {
  return typeof sub === 'function' ? sub(result) : sub;
}

// === INIT ===
document.addEventListener('DOMContentLoaded', () => {
  renderCustomerList();
  setupSelectAll();
});

function renderCustomerList() {
  const list = document.getElementById('customerList');
  list.innerHTML = CUSTOMER_META.map((c, i) => `
    <div class="customer-card ${selected.has(i) ? 'active' : ''}" onclick="toggleCustomer(${i}, this)">
      <div class="customer-card-header">
        <input type="checkbox" class="customer-check" ${selected.has(i) ? 'checked' : ''} onclick="event.stopPropagation(); toggleCheckbox(${i}, this)">
        <div class="customer-avatar" style="background:${c.avatarColor}">${c.initials}</div>
        <span class="customer-name">${c.name}</span>
        <span class="customer-tag ${c.tagClass}">${c.tag}</span>
      </div>
      <div class="customer-meta">
        ${c.fields.map(f => `
          <div class="customer-meta-item">
            <span class="customer-meta-label">${f.label}</span>
            <span class="customer-meta-value ${f.cls||''}">${f.value}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}

function toggleCustomer(i, card) {
  if (selected.has(i)) { if (selected.size > 1) selected.delete(i); }
  else selected.add(i);
  renderCustomerList();
}
function toggleCheckbox(i, cb) {
  if (cb.checked) selected.add(i);
  else { if (selected.size > 1) selected.delete(i); else cb.checked = true; }
  renderCustomerList();
}
function setupSelectAll() {
  document.getElementById('selectAll').addEventListener('change', function() {
    if (this.checked) CUSTOMER_META.forEach((_, i) => selected.add(i));
    else { selected.clear(); selected.add(0); }
    renderCustomerList();
  });
}

// === PIPELINE ===
async function runPipeline() {
  const btn = document.getElementById('runBtn');
  btn.classList.add('loading');
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px;animation:spin 1s linear infinite"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Đang xử lý...`;
  if (!document.querySelector('style#spin')) {
    const s = document.createElement('style'); s.id='spin';
    s.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
    document.head.appendChild(s);
  }

  const content = document.getElementById('pipelineContent');
  const progressTrack = document.getElementById('progressTrack');
  const meta = document.getElementById('pipelineMeta');
  content.innerHTML = '';
  progressTrack.style.display = 'none';
  document.getElementById('resultList').innerHTML = '<div class="empty-state-sm">Đang xử lý...</div>';

  const toRunIdx = [...selected];
  const toRun = toRunIdx.map(i => CUSTOMER_META[i]);
  lastResults = [];

  const startTime = Date.now();

  // Check API health first
  const apiOnline = await checkApiHealth();
  if (!apiOnline) {
    showApiError(content);
    resetBtn(btn);
    return;
  }

  // Show progress bar for first customer's pipeline
  progressTrack.style.display = 'block';
  document.getElementById('progressLabels').innerHTML = PIPELINE_STAGES.map(s =>
    `<div class="progress-label">A${s.label.slice(1)}: <span>${s.title}</span></div>`
  ).join('');

  // Run all selected customers in parallel
  const promises = toRun.map(c => callScoringAPI(c.id));
  const results = await Promise.allSettled(promises);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  // Animate progress bar
  setTimeout(() => { document.getElementById('progressFill').style.width = '100%'; }, 50);

  // Update meta
  const successCount = results.filter(r => r.status === 'fulfilled').length;
  meta.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    Hoàn thành ${successCount}/${toRun.length} • <strong>${elapsed}s</strong>`;

  // Use first successful result for pipeline visualization
  let primaryResult = null;
  for (let i = 0; i < results.length; i++) {
    if (results[i].status === 'fulfilled') {
      primaryResult = { meta: toRun[i], api: results[i].value };
      break;
    }
  }

  // Render pipeline stages (animated)
  if (primaryResult) {
    for (let idx = 0; idx < PIPELINE_STAGES.length; idx++) {
      await delay(300 + idx * 150);
      renderStageBlock(content, PIPELINE_STAGES[idx], primaryResult.api);
    }
  }

  // Collect all results for right panel
  lastResults = results.map((r, i) => ({
    meta: toRun[i],
    api: r.status === 'fulfilled' ? r.value : null,
    error: r.status === 'rejected' ? r.reason?.message : null,
  }));

  await delay(200);
  renderResults(lastResults);

  resetBtn(btn);
}

async function checkApiHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return resp.ok;
  } catch {
    return false;
  }
}

async function callScoringAPI(customerId) {
  const formData = new FormData();
  formData.append('customer_id', customerId);
  const resp = await fetch(`${API_BASE}/score/mock`, {
    method: 'POST',
    body: formData,
    signal: AbortSignal.timeout(60000),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

function showApiError(container) {
  container.innerHTML = `
    <div class="api-error">
      ${ICONS.warning}
      <h4>Backend chưa khởi động</h4>
      <p>Hãy chạy lệnh sau để start server:</p>
      <code>conda activate swinburn_hackathon<br>uvicorn creditlens.api.main:app --host 0.0.0.0 --port 8000 --reload</code>
      <p style="margin-top:8px;font-size:11px;color:#9e9e9e">API endpoint: <strong>http://localhost:8000</strong></p>
    </div>
  `;
}

function renderStageBlock(container, stage, apiResult) {
  const scoreInfo = stage.scoreFn(apiResult);
  const block = document.createElement('div');
  block.className = 'stage-block';
  block.innerHTML = `
    <div class="stage-header">
      <div class="stage-label ${stage.cls}">${stage.label}</div>
      <div>
        <div class="stage-title">${stage.label} · ${stage.title}</div>
        <div class="stage-meta">${stage.metaFn(apiResult)}</div>
      </div>
      <div class="stage-check">${ICONS.checkmark}</div>
    </div>
    <div class="stage-steps">
      ${stage.steps.map(st => `
        <div class="step">
          <div class="step-icon-wrap">${ICONS[st.icon] || ICONS.info}</div>
          <div class="step-name">${st.name}</div>
          <div class="step-sub">${resolveStepSub(st.sub, apiResult)}</div>
        </div>
        <div class="step-arrow">${ICONS.arrow}</div>
      `).join('')}
      <div class="step-score ${scoreInfo.cls}">
        <div class="step-score-num">${scoreInfo.num}</div>
        <div class="step-score-label">${scoreInfo.label}</div>
      </div>
    </div>
  `;
  container.appendChild(block);
}

function renderResults(results) {
  const list = document.getElementById('resultList');
  list.innerHTML = results.map((r, i) => {
    if (r.error || !r.api) {
      return `
        <div class="result-card result-card-error">
          <div class="result-score score-red">ERR</div>
          <div class="result-info">
            <div class="result-name">${r.meta.name}</div>
            <div class="result-detail" style="color:#e53935">${r.error || 'Lỗi không xác định'}</div>
          </div>
        </div>`;
    }
    const api = r.api;
    const dec = decisionBadge(api.recommendation || api.decision);
    const sc = scoreClass(api.credit_score);
    const pdPct = (api.pd_probability * 100).toFixed(2);
    return `
      <div class="result-card" onclick="openModal(${i})">
        <div class="result-score ${sc}">${api.credit_score}</div>
        <div class="result-info">
          <div class="result-name">${r.meta.name}</div>
          <div class="result-detail">${api.risk_band} — PD ${pdPct}%</div>
        </div>
        <span class="result-badge ${dec.cls}">${dec.label}</span>
        <div class="result-chevron">${ICONS.arrow}</div>
      </div>`;
  }).join('');
}

// === MODAL ===
function openModal(idx) {
  const r = lastResults[idx];
  if (!r || !r.api) return;
  currentModalData = r;
  const api = r.api;
  const meta = r.meta;
  const dec = decisionBadge(api.recommendation || api.decision);
  const pdPct = (api.pd_probability * 100).toFixed(2);

  document.getElementById('modalTitle').textContent = `${meta.name} — Chi tiết đánh giá`;
  document.getElementById('modalSub').textContent = `${meta.tag} · Mã KH: ${meta.id}`;

  // Build 5C data
  const fiveC = api.five_c_scores || {};
  // Map API keys → display names + max values
  const fiveCMap = [
    { key:'character', name:'Character', max:30 },
    { key:'capacity',  name:'Capacity',  max:40 },
    { key:'capital',   name:'Capital',   max:25 },
    { key:'conditions',name:'Conditions',max:10 },
    { key:'collateral',name:'Collateral',max:20 },
  ];

  function cScoreClass(val, max) {
    const ratio = val / max;
    if (ratio >= 0.7) return 'good';
    if (ratio >= 0.4) return 'ok';
    return 'bad';
  }
  function cTagClass(val, max) {
    const ratio = val / max;
    if (ratio >= 0.7) return { cls:'pass', label:'ĐẠT' };
    if (ratio >= 0.4) return { cls:'review', label:'XEM XÉT' };
    return { cls:'fail', label:'YẾU' };
  }

  // SHAP factors
  const shapPos = (api.shap_top_positive || []).slice(0, 5);
  const shapNeg = (api.shap_top_negative || []).slice(0, 5);
  const maxShapPos = Math.max(...shapPos.map(s => Math.abs(s.shap_value || s.value || 0)), 0.001);
  const maxShapNeg = Math.max(...shapNeg.map(s => Math.abs(s.shap_value || s.value || 0)), 0.001);

  // Warnings
  const warnings = api.warnings || [];

  const body = document.getElementById('modalBody');
  body.innerHTML = `
    <!-- Score Summary -->
    <div>
      <div class="section-title">${ICONS.shield2} Tổng quan kết quả</div>
      <div class="score-cards">
        <div class="score-card blue">
          <div class="score-card-label">ĐIỂM TÍN DỤNG</div>
          <div class="score-card-value">${api.credit_score}</div>
          <div class="score-card-sub">${api.risk_band} — Band xếp hạng</div>
        </div>
        <div class="score-card ${dec.label==='APPROVE'?'green':'orange'}">
          <div class="score-card-label">ĐỀ XUẤT</div>
          <div class="score-card-value" style="font-size:18px">${dec.label}</div>
          <div class="score-card-sub">${api.consistency_check ? 'Consistency ✓' : 'Consistency ⚠'}</div>
        </div>
        <div class="score-card orange">
          <div class="score-card-label">XÁC SUẤT VỠ NỢ</div>
          <div class="score-card-value">${pdPct}%</div>
          <div class="score-card-sub">PD Probability</div>
        </div>
        <div class="score-card purple">
          <div class="score-card-label">TỔNG 5C</div>
          <div class="score-card-value">${api.five_c_total || '—'}</div>
          <div class="score-card-sub">/ 125 điểm</div>
        </div>
      </div>
    </div>

    ${warnings.length > 0 ? `
    <!-- Warnings -->
    <div>
      <div class="section-title" style="color:#e65100">${ICONS.warning} Cảnh báo hệ thống</div>
      <div class="warnings-list">
        ${warnings.map(w => `<div class="warning-item">${ICONS.warning} ${w}</div>`).join('')}
      </div>
    </div>` : ''}

    <!-- 5C Assessment -->
    <div>
      <div class="section-title">${ICONS.star} Đánh giá 5C</div>
      <div class="fiveC-row">
        ${fiveCMap.map(f => {
          const val = fiveC[f.key] ?? fiveC[f.name] ?? 0;
          const tag = cTagClass(val, f.max);
          return `
          <div class="fiveC-card">
            <div class="fiveC-name">${f.name}</div>
            <div class="fiveC-score ${cScoreClass(val, f.max)}">${val}/${f.max}</div>
            <span class="fiveC-tag ${tag.cls}">${tag.label}</span>
          </div>`;
        }).join('')}
      </div>
    </div>

    <!-- SHAP -->
    <div>
      <div class="section-title">${ICONS.chartline} Phân tích SHAP — Yếu tố ảnh hưởng</div>
      <div class="shap-grid">
        <div class="shap-panel">
          <div class="shap-panel-title positive">${ICONS.plus} Yếu tố tích cực</div>
          ${shapPos.length === 0 ? '<div style="color:#bdbdbd;font-size:11px">Không có dữ liệu</div>' :
            shapPos.map(s => {
              const val = s.shap_value ?? s.value ?? 0;
              const feat = s.feature_name || s.feature || s.name || '—';
              const desc = s.description || s.interpretation || '';
              const pct = Math.round(Math.abs(val) / maxShapPos * 100);
              return `
              <div class="shap-item">
                <span class="shap-feat">${feat}</span>
                <div class="shap-bar-wrap"><div class="shap-bar-fill pos" style="width:${pct}%"></div></div>
                <span class="shap-val pos">+${Math.abs(val).toFixed(3)}</span>
                <span class="shap-desc">${desc}</span>
              </div>`;
            }).join('')}
        </div>
        <div class="shap-panel">
          <div class="shap-panel-title negative">${ICONS.minus} Yếu tố rủi ro</div>
          ${shapNeg.length === 0 ? '<div style="color:#bdbdbd;font-size:11px">Không có dữ liệu</div>' :
            shapNeg.map(s => {
              const val = s.shap_value ?? s.value ?? 0;
              const feat = s.feature_name || s.feature || s.name || '—';
              const desc = s.description || s.interpretation || '';
              const pct = Math.round(Math.abs(val) / maxShapNeg * 100);
              return `
              <div class="shap-item">
                <span class="shap-feat">${feat}</span>
                <div class="shap-bar-wrap"><div class="shap-bar-fill neg" style="width:${pct}%"></div></div>
                <span class="shap-val neg">-${Math.abs(val).toFixed(3)}</span>
                <span class="shap-desc">${desc}</span>
              </div>`;
            }).join('')}
        </div>
      </div>
    </div>
  `;

  document.getElementById('modalOverlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modalOverlay')) return;
  document.getElementById('modalOverlay').classList.remove('open');
  document.body.style.overflow = '';
}

// === PDF EXPORT ===
function exportPDF() {
  const name = currentModalData?.meta?.name || 'Report';
  alert(`📄 Xuất PDF: ${name}\n\nTích hợp endpoint: POST /generate-pdf`);
}

// === HELPERS ===
function resetBtn(btn) {
  btn.classList.remove('loading');
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" style="width:16px;height:16px"><polygon points="5,3 19,12 5,21"/></svg> Chạy Pipeline`;
}
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
