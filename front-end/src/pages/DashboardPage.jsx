import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { PIPELINE_LAYERS } from '../data/mockData'
import {
  fetchCustomers,
  runScorePipeline,
  runIngestion,
  runProcessing,
  runBatchScorePipeline,
  clearOutputData,
  checkBackendHealth,
  isBackendAvailable,
  getPdfPreviewUrl,
  downloadPdf,
  connectProcessingWebSocket,
  connectBatchWebSocket,
} from '../services/apiService'
import ExtractedDataReviewModal from '../components/ExtractedDataReviewModal'
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

  // Human-in-the-Loop review state
  const [reviewData, setReviewData] = useState(null)
  const [reviewCustomerId, setReviewCustomerId] = useState(null)
  const [reviewCustomerLabel, setReviewCustomerLabel] = useState('')
  const [isProcessingApproval, setIsProcessingApproval] = useState(false)

  // Clear output confirm dialog
  const [showClearConfirm, setShowClearConfirm] = useState(false)
  const [clearing, setClearing] = useState(false)

  // Pipeline metrics (runtime + token usage) — single customer
  const [pipelineMetrics, setPipelineMetrics] = useState(null)

  // ═══ BATCH STATE ═══
  const [batchPhase, setBatchPhase] = useState('idle')
  // 'idle' | 'ingestion' | 'review' | 'processing' | 'done'
  const [batchProgressPct, setBatchProgressPct] = useState(0)
  const [batchProgressLabel, setBatchProgressLabel] = useState('')
  const [batchIngestionResults, setBatchIngestionResults] = useState({})
  // Map<customerId, ingestionResult>
  const [reviewQueue, setReviewQueue] = useState([])
  // Array of { customerId, label, ingestionData }
  const [currentReviewIndex, setCurrentReviewIndex] = useState(0)
  const [approvedCustomers, setApprovedCustomers] = useState([])
  // Array of { customer_id, application_row, ... }
  const [batchSummary, setBatchSummary] = useState(null)
  const [showBatchSummary, setShowBatchSummary] = useState(false)

  // Ref to hold WebSocket sendAction for batch CG communication
  const batchWsRef = useRef(null)

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

  // ═══════════════════════════════════════════════════════════════════════════
  // SINGLE CUSTOMER PIPELINE (existing behavior for 1 selected)
  // ═══════════════════════════════════════════════════════════════════════════

  const runSingleCustomerPipeline = useCallback(async (customer) => {
    setRunning(true)
    setCustomerResults({})
    setCompletedLayers(new Set())
    setLayerData({})
    setCurrentCustomer(customer.id)
    setPipelineMeta(`Phase 1: Extracting data — ${customer.label}`)
    setPipelineMetrics(null)

    // Animate A1
    setActiveLayer(0)
    setLayerProgress(0)
    const a1Anim = setInterval(() => {
      setLayerProgress(prev => Math.min(prev + 5, 95))
    }, 200)

    try {
      const ingestionResult = await runIngestion(customer.id)

      clearInterval(a1Anim)
      setLayerProgress(100)
      setCompletedLayers(prev => new Set([...prev, 0]))
      const row = ingestionResult.application_row || {}
      const totalFields = Object.keys(row).length
      const filledFields = Object.values(row).filter(v => v !== null && v !== undefined && v !== '').length
      setLayerData(prev => ({
        ...prev,
        A1: { value: `${filledFields}/${totalFields}`, label: 'Fields ✓' }
      }))

      // Animate CG layer
      setActiveLayer(1)
      setLayerProgress(0)
      for (let s = 0; s <= 10; s++) {
        setLayerProgress((s / 10) * 100)
        await new Promise(r => setTimeout(r, 60))
      }
      setCompletedLayers(prev => new Set([...prev, 1]))
      setLayerData(prev => ({ ...prev, CG: { value: '✓', label: 'PROCEED' } }))
      setActiveLayer(-1)
      setPipelineMeta(`Chờ xác nhận dữ liệu — ${customer.label}`)

      // Show review popup and wait
      const processResult = await new Promise((resolve, reject) => {
        setReviewData(ingestionResult)
        setReviewCustomerId(customer.id)
        setReviewCustomerLabel(customer.label)
        window.__hitlResolve = resolve
        window.__hitlReject = reject
      })

      setReviewData(null)
      setReviewCustomerId(null)

      if (processResult) {
        setCustomerResults(prev => ({ ...prev, [customer.id]: processResult }))
        setCustomers(prev => prev.map(cust =>
          cust.id === customer.id ? {
            ...cust,
            hasOutput: true,
            scoreData: {
              creditScore: processResult.credit_score,
              riskBand: processResult.risk_band,
              pdPct: processResult.pd_pct,
              recommendation: processResult.recommendation,
              fiveCTotal: Object.values(processResult.four_c_scores || {}).reduce((a, b) => a + b, 0),
              fiveCScores: processResult.four_c_scores,
            },
          } : cust
        ))
        setLayerData(prev => ({
          ...prev,
          A3: { value: processResult.credit_score || '—', label: 'Score ✓' },
          A4: {
            value: processResult.four_c_scores
              ? Object.values(processResult.four_c_scores).reduce((a, b) => a + b, 0).toFixed(0) : '—',
            label: '5C pts ✓',
          },
        }))
        setCompletedLayers(new Set([0, 1, 2, 3, 4]))
      }
    } catch (err) {
      clearInterval(a1Anim)
      setReviewData(null)
      setReviewCustomerId(null)
      if (err.message === 'USER_CANCELLED') {
        setActiveLayer(-1)
        setLayerProgress(0)
        setCompletedLayers(new Set())
        setLayerData({})
        setPipelineMeta('Pipeline cancelled by user')
        setIsProcessingApproval(false)
      } else {
        setCustomerResults(prev => ({
          ...prev,
          [customer.id]: {
            application_id: customer.id,
            credit_score: 0, risk_band: 'ERR', pd_pct: 0,
            recommendation: 'ERROR', four_c_scores: {}, error: true,
          },
        }))
      }
    }

    setActiveLayer(-1)
    setCurrentCustomer(null)
    setRunning(false)
  }, [customers])

  // ═══════════════════════════════════════════════════════════════════════════
  // BATCH PIPELINE (multiple customers — progress bar + batched CG)
  // ═══════════════════════════════════════════════════════════════════════════

  const runBatchPipeline = useCallback(async (selectedCustomers) => {
    setRunning(true)
    setCustomerResults({})
    setCompletedLayers(new Set())
    setLayerData({})
    setCurrentCustomer(null)
    setPipelineMetrics(null)
    setBatchSummary(null)
    setShowBatchSummary(false)
    setApprovedCustomers([])
    setBatchIngestionResults({})
    setReviewQueue([])
    setCurrentReviewIndex(0)

    const customerIds = selectedCustomers.map(c => c.id)
    const total = customerIds.length

    // ── Phase 1: Ingestion ──
    setBatchPhase('ingestion')
    setBatchProgressPct(0)
    setBatchProgressLabel(`Đang trích xuất dữ liệu... (0/${total})`)
    setPipelineMeta(`Batch Phase 1: Ingesting ${total} customers`)

    try {
      const { promise, sendAction, close } = connectBatchWebSocket(customerIds, (event) => {
        // Handle events from WebSocket
        if (event.event === 'a1_started') {
          const pct = Math.round((event.index / total) * 100)
          setBatchProgressPct(pct)
          const cust = selectedCustomers.find(c => c.id === event.customer_id)
          setBatchProgressLabel(
            `Đang trích xuất: ${cust?.label || event.customer_id} (${event.index + 1}/${total})`
          )
        }

        if (event.event === 'a1_completed') {
          const pct = Math.round(((event.index + 1) / total) * 100)
          setBatchProgressPct(pct)
          // Store ingestion result
          setBatchIngestionResults(prev => ({
            ...prev,
            [event.customer_id]: event.data,
          }))
        }

        if (event.event === 'a1_error') {
          // Mark this customer as failed
          setBatchIngestionResults(prev => ({
            ...prev,
            [event.customer_id]: { error: true, errorMessage: event.error },
          }))
        }

        if (event.event === 'phase_ingestion_done') {
          setBatchProgressPct(100)
          setBatchProgressLabel(`Trích xuất hoàn tất — ${event.results_count} hồ sơ`)
        }

        if (event.event === 'phase_review_start') {
          // Build review queue from successful ingestions
          const queue = []
          for (const cid of event.customer_ids) {
            const cust = selectedCustomers.find(c => c.id === cid)
            queue.push({
              customerId: cid,
              label: cust?.label || `Customer #${cid}`,
            })
          }
          setReviewQueue(queue)
          setCurrentReviewIndex(0)

          // Switch to review phase
          setBatchPhase('review')
          setBatchProgressPct(0)
          setBatchProgressLabel('')

          // Show first review popup
          if (queue.length > 0) {
            const firstItem = queue[0]
            // We need to get the ingestionData from our stored results
            // Delay slightly to ensure state is updated
            setTimeout(() => {
              setBatchIngestionResults(prev => {
                const data = prev[firstItem.customerId]
                if (data && !data.error) {
                  setReviewData(data)
                  setReviewCustomerId(firstItem.customerId)
                  setReviewCustomerLabel(
                    `${firstItem.label} (1/${queue.length})`
                  )
                }
                return prev
              })
            }, 100)
          }
        }

        // ── Phase 3 events ──
        if (event.event === 'processing_started') {
          setBatchPhase('processing')
          setBatchProgressPct(0)
          setBatchProgressLabel(`Đang xử lý A2→A4... (0/${event.total})`)
        }

        if (event.event === 'batch_progress') {
          const pct = Math.round((event.completed / event.total) * 100)
          setBatchProgressPct(pct)
          setBatchProgressLabel(
            `Đang xử lý A2→A4... (${event.completed}/${event.total})`
          )
        }

        if (event.event === 'customer_done') {
          // Update customer result in state
          setCustomerResults(prev => ({
            ...prev,
            [event.customer_id]: event.result,
          }))
          setCustomers(prev => prev.map(cust =>
            cust.id === event.customer_id ? {
              ...cust,
              hasOutput: true,
              scoreData: {
                creditScore: event.result.credit_score,
                riskBand: event.result.risk_band,
                pdPct: event.result.pd_pct,
                recommendation: event.result.recommendation,
                fiveCTotal: Object.values(event.result.four_c_scores || {}).reduce((a, b) => a + b, 0),
                fiveCScores: event.result.four_c_scores,
              },
            } : cust
          ))
        }

        if (event.event === 'customer_error') {
          setCustomerResults(prev => ({
            ...prev,
            [event.customer_id]: {
              application_id: event.customer_id,
              credit_score: 0, risk_band: 'ERR', pd_pct: 0,
              recommendation: 'ERROR', four_c_scores: {}, error: true,
            },
          }))
        }

        if (event.event === 'batch_done') {
          setBatchPhase('done')
          setBatchProgressPct(100)
          setBatchProgressLabel('Batch hoàn tất!')
          setBatchSummary(event.summary)
          setShowBatchSummary(true)
        }
      })

      // Store sendAction ref for use in CG handlers
      batchWsRef.current = { sendAction, close }

      // Wait for batch completion
      const summary = await promise

      // If batch_done was not triggered via event (fallback)
      if (!showBatchSummary) {
        setBatchSummary(summary)
        setShowBatchSummary(true)
      }

    } catch (err) {
      console.error('[Batch] WebSocket error:', err)
      setPipelineMeta(`Batch error: ${err.message}`)
    } finally {
      setRunning(false)
      setBatchPhase('idle')
      setBatchProgressPct(0)
      setBatchProgressLabel('')
      batchWsRef.current = null
    }
  }, [customers])

  // ── Main pipeline entry point ──
  const runPipeline = useCallback(async () => {
    if (selected.size === 0 || running) return

    const selectedCustomers = customers.filter(c => selected.has(c.id))

    if (selectedCustomers.length === 1) {
      // Single customer → use existing detailed pipeline UI
      await runSingleCustomerPipeline(selectedCustomers[0])
    } else {
      // Multiple customers → batch pipeline with progress bar
      await runBatchPipeline(selectedCustomers)
    }
  }, [selected, running, customers, runSingleCustomerPipeline, runBatchPipeline])

  // ═══════════════════════════════════════════════════════════════════════════
  // HITL: Handle approval from review popup (single customer mode — WebSocket)
  // ═══════════════════════════════════════════════════════════════════════════

  const handleReviewApprove = useCallback(async (editedRow, metadata) => {
    // Check if we're in batch review mode
    if (batchPhase === 'review' && batchWsRef.current) {
      // ── BATCH MODE: Send approve via WebSocket, advance to next ──
      const cid = reviewCustomerId
      batchWsRef.current.sendAction({
        action: 'approve',
        customer_id: cid,
        application_row: editedRow,
        metadata: {
          raw_texts: metadata.raw_texts,
          thin_file_flag: metadata.thin_file_flag,
          identity_consistency_flag: metadata.identity_consistency_flag,
        },
      })

      // Close current popup
      setReviewData(null)
      setReviewCustomerId(null)

      // Track locally
      setApprovedCustomers(prev => [...prev, {
        customer_id: cid,
        application_row: editedRow,
        ...metadata,
      }])

      // Advance to next in queue
      const nextIdx = currentReviewIndex + 1
      setCurrentReviewIndex(nextIdx)

      if (nextIdx < reviewQueue.length) {
        // Show next review popup
        const nextItem = reviewQueue[nextIdx]
        const data = batchIngestionResults[nextItem.customerId]
        if (data && !data.error) {
          setTimeout(() => {
            setReviewData(data)
            setReviewCustomerId(nextItem.customerId)
            setReviewCustomerLabel(
              `${nextItem.label} (${nextIdx + 1}/${reviewQueue.length})`
            )
          }, 200)
        } else {
          // Skip errored customer, auto-cancel
          batchWsRef.current.sendAction({
            action: 'cancel',
            customer_id: nextItem.customerId,
          })
          // Recursively advance (via state update triggering effect)
          setCurrentReviewIndex(nextIdx + 1)
        }
      } else {
        // All reviews done → start processing
        setPipelineMeta('Tất cả hồ sơ đã được xác nhận. Đang xử lý...')
        batchWsRef.current.sendAction({ action: 'start_processing' })
      }

      return
    }

    // ── SINGLE CUSTOMER MODE (existing behavior) ──
    setReviewData(null)
    setIsProcessingApproval(true)
    setPipelineMetrics(null)
    setPipelineMeta(`Đang xử lý A2→A3→A4 cho ${reviewCustomerLabel || reviewCustomerId}...`)

    const STEP_TO_INDEX = { A2: 2, A3: 3, A4: 4 }

    const wsParams = {
      customer_id: reviewCustomerId,
      application_row: editedRow,
      raw_texts: metadata.raw_texts,
      thin_file_flag: metadata.thin_file_flag,
      identity_consistency_flag: metadata.identity_consistency_flag,
    }

    try {
      const { promise } = connectProcessingWebSocket(wsParams, (event) => {
        const layerIdx = STEP_TO_INDEX[event.step]

        if (event.event === 'started' && layerIdx !== undefined) {
          setActiveLayer(layerIdx)
          setLayerProgress(0)
          setPipelineMeta(`Đang xử lý ${event.step} cho ${reviewCustomerLabel || reviewCustomerId}...`)
          const speed = event.step === 'A4' ? { increment: 0.8, interval: 150 } : { increment: 2, interval: 100 }
          const animInterval = setInterval(() => {
            setLayerProgress(prev => Math.min(prev + speed.increment, 95))
          }, speed.interval)
          window[`__anim_${event.step}`] = animInterval
        }

        if (event.event === 'completed' && layerIdx !== undefined) {
          if (window[`__anim_${event.step}`]) {
            clearInterval(window[`__anim_${event.step}`])
            delete window[`__anim_${event.step}`]
          }
          setLayerProgress(100)
          setCompletedLayers(prev => new Set([...prev, layerIdx]))

          if (event.step === 'A2' && event.data) {
            setLayerData(prev => ({
              ...prev,
              A2: { value: event.data.features_count || '—', label: 'Feats ✓' },
            }))
          } else if (event.step === 'A3' && event.data) {
            setLayerData(prev => ({
              ...prev,
              A3: { value: event.data.credit_score || '—', label: 'Score ✓' },
            }))
            setPipelineMeta(`A3 hoàn tất — Score: ${event.data.credit_score}, PD: ${event.data.pd_pct}%`)
          } else if (event.step === 'A4' && event.data) {
            setLayerData(prev => ({
              ...prev,
              A4: { value: event.data.five_c_total || '—', label: '5C pts ✓' },
            }))
          }
        }

        if (event.event === 'done' && event.metrics) {
          setPipelineMetrics(event.metrics)
        }
      })

      const result = await promise
      setActiveLayer(-1)
      setIsProcessingApproval(false)

      if (window.__hitlResolve) {
        window.__hitlResolve(result)
        window.__hitlResolve = null
        window.__hitlReject = null
      }
    } catch (err) {
      console.warn('[HITL] WebSocket failed, falling back to HTTP:', err.message)

      try {
        setActiveLayer(2)
        setLayerProgress(0)
        const fallbackAnim = setInterval(() => {
          setLayerProgress(prev => Math.min(prev + 1, 95))
        }, 200)

        const result = await runProcessing(wsParams)

        clearInterval(fallbackAnim)
        setLayerProgress(100)
        setCompletedLayers(new Set([0, 1, 2, 3, 4]))
        setActiveLayer(-1)
        setIsProcessingApproval(false)

        if (window.__hitlResolve) {
          window.__hitlResolve(result)
          window.__hitlResolve = null
          window.__hitlReject = null
        }
      } catch (httpErr) {
        setActiveLayer(-1)
        setIsProcessingApproval(false)
        console.error('[HITL] HTTP fallback also failed:', httpErr)
        if (window.__hitlResolve) {
          window.__hitlResolve(null)
          window.__hitlResolve = null
          window.__hitlReject = null
        }
      }
    }
  }, [reviewCustomerId, reviewCustomerLabel, batchPhase, currentReviewIndex, reviewQueue, batchIngestionResults])

  // ═══════════════════════════════════════════════════════════════════════════
  // HITL: Handle cancel from review popup
  // ═══════════════════════════════════════════════════════════════════════════

  const handleReviewCancel = useCallback(() => {
    // Check if we're in batch review mode
    if (batchPhase === 'review' && batchWsRef.current) {
      // ── BATCH MODE: Send cancel via WebSocket, advance to next ──
      const cid = reviewCustomerId
      batchWsRef.current.sendAction({
        action: 'cancel',
        customer_id: cid,
      })

      setReviewData(null)
      setReviewCustomerId(null)

      // Advance to next
      const nextIdx = currentReviewIndex + 1
      setCurrentReviewIndex(nextIdx)

      if (nextIdx < reviewQueue.length) {
        const nextItem = reviewQueue[nextIdx]
        const data = batchIngestionResults[nextItem.customerId]
        if (data && !data.error) {
          setTimeout(() => {
            setReviewData(data)
            setReviewCustomerId(nextItem.customerId)
            setReviewCustomerLabel(
              `${nextItem.label} (${nextIdx + 1}/${reviewQueue.length})`
            )
          }, 200)
        }
      } else {
        // All reviews done → start processing
        setPipelineMeta('Bắt đầu xử lý các hồ sơ đã duyệt...')
        batchWsRef.current.sendAction({ action: 'start_processing' })
      }

      return
    }

    // ── SINGLE CUSTOMER MODE ──
    setReviewData(null)
    setReviewCustomerId(null)
    if (window.__hitlReject) {
      window.__hitlReject(new Error('USER_CANCELLED'))
      window.__hitlResolve = null
      window.__hitlReject = null
    }
  }, [batchPhase, reviewCustomerId, currentReviewIndex, reviewQueue, batchIngestionResults])

  // ── Effect: handle advancing past errored customers in review queue ──
  useEffect(() => {
    if (batchPhase !== 'review' || !batchWsRef.current || reviewQueue.length === 0) return
    if (currentReviewIndex >= reviewQueue.length && !reviewData) {
      // All done, start processing
      setPipelineMeta('Bắt đầu xử lý các hồ sơ đã duyệt...')
      batchWsRef.current.sendAction({ action: 'start_processing' })
    }
  }, [currentReviewIndex, reviewQueue, batchPhase, reviewData])

  const getCustomerStatus = (customerId) => {
    const result = customerResults[customerId]
    if (result) {
      if (result.error) return 'error'
      const rec = (result.recommendation || '').toUpperCase()
      if (rec.includes('APPROVE')) return 'approved'
      if (rec === 'REJECT') return 'rejected'
      return 'review'
    }

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

  // Is batch mode active? (show progress bar instead of pipeline viz)
  const isBatchMode = batchPhase !== 'idle' && batchPhase !== 'done'

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
                      {(c.hasOutput || result) && (
                        <button
                          className="view-btn"
                          onClick={(e) => { e.stopPropagation(); navigate(`/report/${c.id}`) }}
                          title="View Full Report"
                        >
                          <span className="material-symbols-outlined">visibility</span>
                        </button>
                      )}
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

                        {result && result.error && (
                          <div className="customer-result-summary error-state">
                            <span className="material-symbols-outlined text-error">error</span>
                            <span className="text-xs text-slate-500">Backend unavailable — Run pipeline when server is online</span>
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

        {/* ═══ CENTER: PIPELINE VISUALIZATION or BATCH PROGRESS ═══ */}
        <div className="main">
          <div className="pipeline-header">
            <div className="pipeline-header-left">
              <div className="pipeline-title">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
                {isBatchMode ? 'BATCH PROCESSING ENGINE' : 'MULTI-LAYER PIPELINE ENGINE'}
              </div>
              <div className="pipeline-meta">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12 6 12 12 16 14" />
                </svg>
                {pipelineMeta}
              </div>
            </div>
            {pipelineMetrics && !isBatchMode && (
              <div className="pipeline-metrics">
                <span className="metric-badge time">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  {pipelineMetrics.runtime_seconds >= 60
                    ? `${Math.floor(pipelineMetrics.runtime_seconds / 60)}m ${Math.round(pipelineMetrics.runtime_seconds % 60)}s`
                    : `${pipelineMetrics.runtime_seconds}s`
                  }
                </span>
                <span className="metric-badge tokens">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                  {pipelineMetrics.total_tokens >= 1000
                    ? `${(pipelineMetrics.total_tokens / 1000).toFixed(1)}K`
                    : pipelineMetrics.total_tokens
                  } tokens
                </span>
              </div>
            )}
          </div>

          {/* ═══ BATCH PROGRESS BAR (shown during batch mode) ═══ */}
          {isBatchMode && (
            <div className="batch-progress-container">
              <div className="batch-phase-indicator">
                <div className={`batch-phase-step ${batchPhase === 'ingestion' ? 'active' : batchPhase === 'review' || batchPhase === 'processing' ? 'completed' : ''}`}>
                  <div className="phase-dot">1</div>
                  <span>Trích xuất (A1)</span>
                </div>
                <div className="phase-connector"></div>
                <div className={`batch-phase-step ${batchPhase === 'review' ? 'active' : batchPhase === 'processing' ? 'completed' : ''}`}>
                  <div className="phase-dot">2</div>
                  <span>Xác nhận (CG)</span>
                </div>
                <div className="phase-connector"></div>
                <div className={`batch-phase-step ${batchPhase === 'processing' ? 'active' : ''}`}>
                  <div className="phase-dot">3</div>
                  <span>Xử lý (A2→A4)</span>
                </div>
              </div>

              {batchPhase !== 'review' && (
                <div className="batch-progress-bar-wrapper">
                  <div className="batch-progress-bar">
                    <div
                      className="batch-progress-fill"
                      style={{ width: `${batchProgressPct}%` }}
                    ></div>
                  </div>
                  <div className="batch-progress-info">
                    <span className="batch-progress-label">{batchProgressLabel}</span>
                    <span className="batch-progress-pct">{batchProgressPct}%</span>
                  </div>
                </div>
              )}

              {batchPhase === 'review' && (
                <div className="batch-review-status">
                  <span className="material-symbols-outlined" style={{ color: '#f4b400', fontSize: 28 }}>fact_check</span>
                  <div>
                    <strong>Đang chờ xác nhận dữ liệu</strong>
                    <p>Vui lòng xác nhận hoặc huỷ từng hồ sơ trong popup</p>
                    <span className="text-xs text-slate-400">
                      {currentReviewIndex + 1} / {reviewQueue.length} hồ sơ
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ═══ PIPELINE LAYERS (shown in single mode or idle) ═══ */}
          {!isBatchMode && (
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
          )}
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

      {/* BATCH SUMMARY POPUP */}
      {showBatchSummary && batchSummary && (
        <div className="modal-overlay" onClick={() => setShowBatchSummary(false)}>
          <div className="batch-summary-modal" onClick={e => e.stopPropagation()}>
            <div className="batch-summary-header">
              <div className="batch-summary-title">
                <span className="material-symbols-outlined" style={{ color: '#0f9d58', fontSize: 24 }}>check_circle</span>
                <h3>Batch Processing Complete</h3>
              </div>
              <button className="modal-close-btn" onClick={() => setShowBatchSummary(false)}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div className="batch-summary-stats">
              <div className="stat-card">
                <span className="material-symbols-outlined">timer</span>
                <div>
                  <div className="stat-value">
                    {batchSummary.total_time_seconds >= 60
                      ? `${Math.floor(batchSummary.total_time_seconds / 60)}m ${Math.round(batchSummary.total_time_seconds % 60)}s`
                      : `${batchSummary.total_time_seconds}s`
                    }
                  </div>
                  <div className="stat-label">Total Time</div>
                </div>
              </div>
              <div className="stat-card">
                <span className="material-symbols-outlined">token</span>
                <div>
                  <div className="stat-value">
                    {batchSummary.total_tokens >= 1000
                      ? `${(batchSummary.total_tokens / 1000).toFixed(1)}K`
                      : batchSummary.total_tokens
                    }
                  </div>
                  <div className="stat-label">Total Tokens</div>
                </div>
              </div>
              <div className="stat-card">
                <span className="material-symbols-outlined">group</span>
                <div>
                  <div className="stat-value">{batchSummary.success_count}/{batchSummary.total_customers}</div>
                  <div className="stat-label">Succeeded</div>
                </div>
              </div>
            </div>

            <div className="batch-summary-table-wrapper">
              <table className="batch-summary-table">
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Risk Band</th>
                    <th>PD %</th>
                  </tr>
                </thead>
                <tbody>
                  {(batchSummary.customers || []).map(c => {
                    const cust = customers.find(cu => cu.id === c.customer_id)
                    return (
                      <tr key={c.customer_id} className={c.status === 'FAILED' ? 'failed-row' : ''}>
                        <td className="customer-cell">
                          <span className="customer-name-sm">{cust?.label || `#${c.customer_id}`}</span>
                        </td>
                        <td>
                          <span className={`table-status ${c.status.toLowerCase()}`}>
                            {c.status}
                          </span>
                        </td>
                        <td className="mono">{c.credit_score || '—'}</td>
                        <td>{c.risk_band || '—'}</td>
                        <td className="mono">{c.pd_pct ? `${c.pd_pct}%` : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="batch-summary-footer">
              <button className="batch-summary-close-btn" onClick={() => setShowBatchSummary(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* HUMAN-IN-THE-LOOP REVIEW MODAL */}
      {reviewData && (
        <ExtractedDataReviewModal
          ingestionData={reviewData}
          customerId={reviewCustomerLabel || reviewCustomerId}
          onApprove={handleReviewApprove}
          onCancel={handleReviewCancel}
          isProcessing={isProcessingApproval}
        />
      )}
    </div>
  )
}
