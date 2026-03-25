import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { PIPELINE_LAYERS } from '../data/mockData'
import {
  fetchCustomers,
  runScorePipeline,
  runBatchScorePipeline,
  clearOutputData,
  checkBackendHealth,
  isBackendAvailable,
  getPdfPreviewUrl,
  downloadPdf,
} from '../services/apiService'
import './DashboardPage.css'

export default function DashboardPage() {
  const navigate = useNavigate()

  // Customer data — loaded from API or fallback
  const [customers, setCustomers] = useState([])
  const [dataSource, setDataSource] = useState(null) // 'backend' | 'fallback'
  const [loadingCustomers, setLoadingCustomers] = useState(true)

  const [selected, setSelected] = useState(new Set())
  const [expanded, setExpanded] = useState(null)
  const [running, setRunning] = useState(false)
  const [activeLayer, setActiveLayer] = useState(-1)
  const [layerProgress, setLayerProgress] = useState(0)
  const [completedLayers, setCompletedLayers] = useState(new Set())
  const [layerData, setLayerData] = useState({})
  const [customerResults, setCustomerResults] = useState({})
  const [currentCustomer, setCurrentCustomer] = useState(null)
  const [pipelineMeta, setPipelineMeta] = useState('Awaiting input — Select profiles and run pipeline')
  const [previewPdf, setPreviewPdf] = useState(null)

  // Batch progress tracking
  const [batchProgress, setBatchProgress] = useState(null) // { completed: N, total: N }

  // Clear output confirm dialog
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [clearing, setClearing] = useState(false)

  // ── Load customers on mount ──
  useEffect(() => {
    async function loadData() {
      setLoadingCustomers(true)
      await checkBackendHealth()
      const result = await fetchCustomers()
      setCustomers(result.customers)
      setDataSource(result.source)
      setLoadingCustomers(false)
    }
    loadData()
  }, [])

  const toggleSelect = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const toggleExpand = (id) => {
    setExpanded(prev => prev === id ? null : id)
  }

  const toggleAll = (checked) => {
    setSelected(checked ? new Set(customers.map(c => c.id)) : new Set())
  }

  /* ── Clear output data for demo reset ── */
  const handleClearOutput = useCallback(async () => {
    setClearing(true)
    try {
      const result = await clearOutputData()
      // Reset all local state
      setCustomerResults({})
      setCustomers(prev => prev.map(c => ({ ...c, hasOutput: false, scoreData: null })))
      setCompletedLayers(new Set())
      setLayerData({})
      setPipelineMeta(`Output cleared — ${result.cleared} items removed`)
      setShowClearConfirm(false)
    } catch (err) {
      console.error('[Clear] Failed:', err)
      setPipelineMeta(`Clear failed — ${err.message}`)
    } finally {
      setClearing(false)
    }
  }, [])

  /* ── Pipeline execution (BATCH PARALLEL) ── */
  const runPipeline = useCallback(async () => {
    if (selected.size === 0 || running) return
    setRunning(true)
    setCustomerResults({})
    setCompletedLayers(new Set())
    setLayerData({})

    const selectedCustomers = customers.filter(c => selected.has(c.id))
    const total = selectedCustomers.length
    const customerIds = selectedCustomers.map(c => c.id)

    // Show batch meta
    setCurrentCustomer(null)
    setPipelineMeta(`Batch processing ${total} profiles in parallel...`)
    setBatchProgress({ completed: 0, total })

    // ── Start animation (runs independently, purely cosmetic) ──
    const animationPromise = (async () => {
      const totalAnimTime = PIPELINE_LAYERS.length * 1200 // ~4.8s total animation
      for (let li = 0; li < PIPELINE_LAYERS.length; li++) {
        setActiveLayer(li)
        setLayerProgress(0)

        const stepDuration = li === 2 ? 1800 : 800
        const steps = 20
        for (let s = 0; s <= steps; s++) {
          setLayerProgress((s / steps) * 100)
          await new Promise(r => setTimeout(r, stepDuration / steps))
        }
        setCompletedLayers(prev => new Set([...prev, li]))
      }
    })()

    // ── Fire batch API call in parallel with animation ──
    let batchResult = null
    let batchError = null
    const apiPromise = (async () => {
      try {
        batchResult = await runBatchScorePipeline(customerIds)
      } catch (err) {
        console.warn('[Pipeline] Batch API failed:', err.message)
        batchError = err
      }
    })()

    // ── Wait for BOTH animation and API to finish ──
    await Promise.all([animationPromise, apiPromise])

    // ── Process results ──
    if (batchResult && batchResult.results) {
      const batchResults = batchResult.results

      for (const customer of selectedCustomers) {
        const result = batchResults[customer.id]

        if (result && result.status !== 'FAILED') {
          setCustomerResults(prev => ({ ...prev, [customer.id]: result }))

          // Update hasOutput flag and scoreData for persistence
          setCustomers(prev => prev.map(cust =>
            cust.id === customer.id ? {
              ...cust,
              hasOutput: true,
              // Score data ONLY from output/ (never mock)
              scoreData: {
                creditScore: result.credit_score,
                riskBand: result.risk_band,
                pdPct: result.pd_pct,
                recommendation: result.recommendation,
                fiveCTotal: Object.values(result.four_c_scores || {}).reduce((a, b) => a + b, 0),
                fiveCScores: result.four_c_scores,
              },
            } : cust
          ))
        } else {
          // Per-customer fallback if batch had individual failures
          const fallbackResult = result || {
            application_id: customer.id,
            credit_score: 0,
            risk_band: 'ERR',
            pd_pct: 0,
            recommendation: 'ERROR',
            four_c_scores: {},
            error: true,
          }
          setCustomerResults(prev => ({ ...prev, [customer.id]: fallbackResult }))
        }
      }

      // Update layer badges with data from last successful result
      const anySuccess = Object.values(batchResults).find(r => r.status === 'SUCCESS')
      if (anySuccess) {
        setLayerData({
          A1: { value: '122', label: 'Fields ✓' },
          A2: { value: anySuccess.four_c_scores ? '753' : '—', label: 'Feats ✓' },
          A3: { value: anySuccess.credit_score || '—', label: 'Score ✓' },
          A4: {
            value: anySuccess.four_c_scores
              ? Object.values(anySuccess.four_c_scores).reduce((a, b) => a + b, 0).toFixed(0) : '—',
            label: '5C pts ✓',
          },
        })
      }

      const duration = batchResult.duration_s || 0
      setPipelineMeta(
        `Completed ${batchResult.success_count}/${total} profiles in ${duration}s (parallel)`
      )
    } else {
      // Entire batch failed — fallback per-customer using preloaded scoreData
      for (const customer of selectedCustomers) {
        if (customer.scoreData) {
          setCustomerResults(prev => ({
            ...prev,
            [customer.id]: {
              application_id: customer.id,
              credit_score: customer.scoreData.creditScore,
              pd_pct: customer.scoreData.pdPct,
              risk_band: customer.scoreData.riskBand,
              recommendation: customer.scoreData.recommendation,
              four_c_scores: customer.scoreData.fiveCScores || {},
              fallback: true,
            },
          }))
        } else {
          setCustomerResults(prev => ({
            ...prev,
            [customer.id]: {
              application_id: customer.id,
              credit_score: 0,
              risk_band: 'ERR',
              pd_pct: 0,
              recommendation: 'OFFLINE',
              four_c_scores: {},
              error: true,
            },
          }))
        }
      }

      setPipelineMeta(`Batch failed — ${batchError?.message || 'Unknown error'}`)
    }

    setActiveLayer(-1)
    setCurrentCustomer(null)
    setBatchProgress(null)
    setRunning(false)
  }, [selected, running, customers])

  const getCustomerStatus = (customerId) => {
    // Check in-memory pipeline result first (from current session)
    const result = customerResults[customerId]
    if (result) {
      if (result.error) return 'error'
      const rec = (result.recommendation || '').toUpperCase()
      if (rec.includes('APPROVE')) return 'approved'
      if (rec === 'REJECT') return 'rejected'
      return 'review'
    }

    // Check persisted output data (survives F5 refresh)
    const customer = customers.find(c => c.id === customerId)
    if (customer?.scoreData) {
      const rec = (customer.scoreData.recommendation || '').toUpperCase()
      if (rec.includes('APPROVE')) return 'approved'
      if (rec === 'REJECT') return 'rejected'
      return 'review'
    }

    return 'pending'
  }

  const getStatusLabel = (status) => {
    switch (status) {
      case 'approved': return 'Approved'
      case 'rejected': return 'Rejected'
      case 'review': return 'Review'
      case 'error': return 'Error'
      default: return 'Pending'
    }
  }

  return (
    <div className="dashboard-root">
      {/* TOPBAR */}
      <div className="topbar">
        <div className="logo">
          <div className="logo-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e53935" strokeWidth="2.5">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          CrediCouncil AI
        </div>
        <span className="topbar-title">Credit Scoring Dashboard</span>
        <div className="topbar-spacer"></div>
        {/* Backend connection indicator */}
        <div className={`live-badge ${running ? 'active' : ''}`}>
          <div className={`live-dot ${dataSource === 'backend' ? '' : 'offline'}`}></div>
          {running ? 'Processing...' : (
            dataSource === 'backend' ? 'Backend Connected' :
              dataSource === 'fallback' ? 'Offline Mode' : 'Connecting...'
          )}
        </div>
      </div>

      {/* MAIN LAYOUT — 2 columns */}
      <div className="layout">

        {/* ═══ LEFT SIDEBAR ═══ */}
        <div className="sidebar-left">
          <div className="sidebar-header">
            <div className="sidebar-header-top">
              <h3>Evaluation Queue</h3>
              <button
                className="clear-output-btn"
                onClick={() => setShowClearConfirm(true)}
                disabled={running || clearing}
                title="Clear all pipeline output data"
              >
                <span className="material-symbols-outlined" style={{ fontSize: 13 }}>delete_sweep</span>
                Reset
              </button>
            </div>
            <div className="sidebar-header-actions">
              <label className="select-all" style={{ margin: 0 }}>
                <input
                  type="checkbox"
                  checked={customers.length > 0 && selected.size === customers.length}
                  onChange={(e) => toggleAll(e.target.checked)}
                  disabled={loadingCustomers}
                />
                {' '}Select All
              </label>
              <span className="sidebar-subtitle">
                {dataSource === 'backend'
                  ? `${customers.length} profiles`
                  : `${customers.length} (offline)`
                }
              </span>
            </div>
          </div>

          {/* Clear Output Confirm Dialog */}
          {showClearConfirm && (
            <div className="clear-confirm-overlay">
              <div className="clear-confirm-dialog">
                <div className="clear-confirm-icon">
                  <span className="material-symbols-outlined">warning</span>
                </div>
                <div className="clear-confirm-text">
                  <strong>Reset Output Data?</strong>
                  <p>This will delete all pipeline results. All customer statuses will revert to Pending.</p>
                </div>
                <div className="clear-confirm-actions">
                  <button className="clear-cancel-btn" onClick={() => setShowClearConfirm(false)} disabled={clearing}>
                    Cancel
                  </button>
                  <button className="clear-delete-btn" onClick={handleClearOutput} disabled={clearing}>
                    {clearing ? <><span className="spinner white"></span> Clearing...</> : 'Clear All'}
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="customer-list">
            {loadingCustomers ? (
              <div className="loading-state">
                <span className="spinner"></span>
                <p>Loading customer profiles...</p>
              </div>
            ) : (
              customers.map(c => {
                const status = getCustomerStatus(c.id)
                const isExpanded = expanded === c.id
                const result = customerResults[c.id]
                const isProcessing = running && selected.has(c.id) && !result

                return (
                  <div key={c.id} className={`customer-card ${status} ${isProcessing ? 'processing' : ''}`}>
                    {/* Main row */}
                    <div className="customer-row">
                      <input
                        type="checkbox"
                        checked={selected.has(c.id)}
                        onChange={() => toggleSelect(c.id)}
                        disabled={running}
                      />
                      <div className="customer-main" onClick={() => toggleExpand(c.id)}>
                        <div className="customer-name-row">
                          <span className="customer-name">{c.label}</span>
                          <span className={`status-badge ${status}`}>
                            {isProcessing ? (
                              <><span className="spinner"></span> Processing</>
                            ) : (
                              getStatusLabel(status)
                            )}
                          </span>
                        </div>
                        <span className="customer-id">ID: {c.folderId}</span>
                      </div>
                      {/* View report button — only shown when output data exists */}
                      {(c.hasOutput || result) && (
                        <button
                          className="view-btn"
                          onClick={(e) => { e.stopPropagation(); navigate(`/report/${c.id}`) }}
                          title="View Full Report"
                        >
                          <span className="material-symbols-outlined">visibility</span>
                        </button>
                      )}
                      {/* Expand arrow */}
                      <button className={`expand-btn ${isExpanded ? 'open' : ''}`} onClick={() => toggleExpand(c.id)}>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </button>
                    </div>

                    {/* Expandable info */}
                    {isExpanded && (
                      <div className="customer-detail">
                        <div className="detail-grid">
                          <div className="detail-item"><span className="dt">Gender/Age</span><span className="dd">{c.info.gender}, {c.info.age}</span></div>
                          <div className="detail-item"><span className="dt">Income</span><span className="dd">{c.info.income}</span></div>
                          <div className="detail-item full"><span className="dt">Request</span><span className="dd">{c.info.amount} ({c.info.loan})</span></div>
                          {c.info.education && c.info.education !== 'N/A' && (
                            <div className="detail-item"><span className="dt">Education</span><span className="dd">{c.info.education}</span></div>
                          )}
                          {c.info.housing && c.info.housing !== 'N/A' && (
                            <div className="detail-item"><span className="dt">Housing</span><span className="dd">{c.info.housing}</span></div>
                          )}
                        </div>

                        {/* Show pre-loaded score data if available */}
                        {c.scoreData && !result && (
                          <div className="customer-result-summary preloaded">
                            <div className="result-mini">
                              <span className="result-mini-label">Credit Score</span>
                              <span className="result-mini-value">{c.scoreData.creditScore}</span>
                            </div>
                            <div className="result-mini">
                              <span className="result-mini-label">Risk Band</span>
                              <span className="result-mini-value">{c.scoreData.riskBand}</span>
                            </div>
                            <div className="result-mini">
                              <span className="result-mini-label">PD</span>
                              <span className="result-mini-value">{c.scoreData.pdPct}%</span>
                            </div>
                          </div>
                        )}

                        {/* Show pipeline result if available */}
                        {result && !result.error && (
                          <>
                            <div className="customer-result-summary">
                              <div className="result-mini">
                                <span className="result-mini-label">Credit Score</span>
                                <span className="result-mini-value">{result.credit_score}</span>
                              </div>
                              <div className="result-mini">
                                <span className="result-mini-label">Risk Band</span>
                                <span className="result-mini-value">{result.risk_band}</span>
                              </div>
                              <div className="result-mini">
                                <span className="result-mini-label">PD</span>
                                <span className="result-mini-value">{result.pd_pct}%</span>
                              </div>
                            </div>
                            <div className="customer-detail-actions mt-2 pt-2 border-t border-slate-100 flex justify-end">
                              <button className="flex items-center gap-1.5 text-xs font-bold text-accent bg-accent-light px-3 py-1.5 rounded hover:bg-red-100 transition-colors" onClick={() => setPreviewPdf(c.id)}>
                                <span className="material-symbols-outlined text-[14px]">picture_as_pdf</span> PDF Preview
                              </button>
                            </div>
                          </>
                        )}

                        {/* Show error state */}
                        {result && result.error && (
                          <div className="customer-result-summary error-state">
                            <span className="material-symbols-outlined text-error">error</span>
                            <span className="text-xs text-slate-500">Backend unavailable — Run pipeline when server is online</span>
                          </div>
                        )}

                        {/* Show fallback indicator */}
                        {result && result.fallback && (
                          <div className="fallback-notice">
                            <span className="material-symbols-outlined text-[12px]">info</span>
                            Showing cached report data (backend was offline)
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>

          <button
            className={`run-btn ${running ? 'running' : ''} ${selected.size === 0 ? 'disabled' : ''}`}
            onClick={runPipeline}
            disabled={running || selected.size === 0}
          >
            {running ? (
              <><span className="spinner white"></span> Processing...</>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21" /></svg>
                Run Pipeline ({selected.size})
              </>
            )}
          </button>
        </div>

        {/* ═══ CENTER: PIPELINE VISUALIZATION ═══ */}
        <div className="main">
          <div className="pipeline-header">
            <div className="pipeline-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              MULTI-LAYER PIPELINE ENGINE
            </div>
            <div className="pipeline-meta">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              {pipelineMeta}
            </div>
          </div>

          <div className="pipeline-layers">
            {PIPELINE_LAYERS.map((layer, li) => {
              const isActive = activeLayer === li
              const isCompleted = completedLayers.has(li)
              const isPending = !isActive && !isCompleted
              const data = layerData[layer.id]

              return (
                <div
                  key={layer.id}
                  className={`pipe-layer ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''} ${isPending ? 'pending' : ''}`}
                >
                  {/* Layer header */}
                  <div className="pipe-layer-header">
                    <div className="pipe-layer-left">
                      <div className="pipe-layer-badge" style={{
                        background: isPending ? '#f5f5f5' : layer.bgColor,
                        color: isPending ? '#bdbdbd' : layer.color,
                        borderColor: isPending ? '#e0e0e0' : layer.color + '30',
                      }}>
                        {layer.id}
                      </div>
                      <div className="pipe-layer-info">
                        <h3>
                          {layer.title}
                          {isCompleted && <span className="material-symbols-outlined check-icon" style={{ color: layer.color }}>check_circle</span>}
                          {isActive && <span className="pipe-processing-tag">PROCESSING {Math.round(layerProgress)}%</span>}
                        </h3>
                        <p className="pipe-layer-sub">
                          {layer.sub}
                          {data && isCompleted && layer.id === 'A3' && ` · PD: ${customerResults[currentCustomer]?.pd_pct || '—'}%`}
                          {data && isCompleted && layer.id === 'A4' && ` · ${customerResults[currentCustomer]?.recommendation || 'REVIEW'}`}
                        </p>
                      </div>
                    </div>
                    {/* Badge with value */}
                    {data && isCompleted ? (
                      <div className="pipe-layer-value" style={{ background: layer.badgeBg || layer.color }}>
                        <span className="val">{data.value}</span>
                        <span className="lbl">{data.label}</span>
                      </div>
                    ) : isPending ? (
                      <div className="pipe-layer-value pending-val" style={{ background: '#f5f5f5', color: '#9e9e9e', opacity: 0.8, borderRadius: '99px', padding: '4px 12px', minWidth: 'auto' }}>
                        <span className="lbl" style={{ fontSize: '9px', fontWeight: 600 }}>PENDING SEQUENCE</span>
                      </div>
                    ) : null}
                  </div>

                  {/* Nodes */}
                  <div className={`pipe-nodes ${isPending ? 'dimmed' : ''}`}>
                    {layer.nodes.map((node, ni) => (
                      <div key={ni} className={`pipe-node ${ni < layer.nodes.length - 1 ? 'has-arrow' : ''}`}>
                        <div className={`pipe-node-icon ${isActive && ni === 0 ? 'active-node' : ''}`}
                          style={isActive && ni === 0 ? { borderColor: layer.color + '40', color: layer.color } : {}}>
                          <span className="material-symbols-outlined">{node.icon}</span>
                        </div>
                        <div className="pipe-node-text">
                          <span className="pipe-node-label">{node.label}</span>
                          <span className="pipe-node-detail">{node.detail}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Progress bar for active layer */}
                  {isActive && (
                    <div className="pipe-progress">
                      <div className="pipe-progress-fill" style={{
                        width: `${layerProgress}%`,
                        background: layer.color,
                        boxShadow: `0 0 8px ${layer.color}60`,
                      }}></div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* PDF PREVIEW MODAL */}
      {previewPdf && (
        <div className="modal-overlay" onClick={() => setPreviewPdf(null)}>
          <div className="pdf-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header flex justify-between items-center p-4 border-b border-border">
              <div>
                <div className="font-bold text-sm">PDF Preview - Credit Report</div>
                <div className="text-xs text-text-secondary">Customer ID: {previewPdf}</div>
              </div>
              <button className="modal-close p-1 hover:bg-slate-100 rounded text-slate-500" onClick={() => setPreviewPdf(null)}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <div className="modal-body w-full h-[65vh] bg-slate-100 p-0">
              <iframe
                src={getPdfPreviewUrl(previewPdf)}
                title="PDF Preview"
                className="w-full h-full border-0"
              />
            </div>
            <div className="modal-footer border-t border-border p-3 bg-white flex justify-end gap-3 rounded-b-lg">
              <button
                className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-sm font-bold rounded flex items-center gap-2 transition-colors"
                onClick={() => setPreviewPdf(null)}
              >
                Close
              </button>
              <button
                className="px-4 py-2 bg-accent hover:bg-accent-dark text-white text-sm font-bold rounded flex items-center gap-2 transition-colors shadow-sm"
                onClick={() => downloadPdf(previewPdf)}
              >
                <span className="material-symbols-outlined text-[16px]">download</span>
                Download PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
