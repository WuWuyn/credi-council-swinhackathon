/**
 * API Service Layer — Centralized fetch with fallback to mock data.
 * 
 * This module provides a single source of truth for all API calls.
 * When the backend is offline, it automatically falls back to mock data.
 * 
 * In dev mode (Vite), requests to /v1/* and /health are proxied to the backend.
 * In production, API_CONFIG.BASE_URL is used.
 */

import { API_CONFIG } from '../config/api'
import { CUSTOMERS_FALLBACK, PIPELINE_LAYERS, reportFallbackData } from '../data/mockData'

// ── URL Helpers ──────────────────────────────────────────────────────────
// In development with Vite proxy, use relative paths (empty string).
// In production builds, use the configured BASE_URL.
const isDev = import.meta.env.DEV
const API_BASE = isDev ? '' : API_CONFIG.BASE_URL

// ── Health & Status ──────────────────────────────────────────────────────
let _backendAvailable = null // cache

/**
 * Check if the backend server is reachable.
 * Caches result for the session, can be refreshed.
 */
export async function checkBackendHealth(forceRefresh = false) {
  if (_backendAvailable !== null && !forceRefresh) return _backendAvailable
  
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 3000)
    
    const res = await fetch(`${API_BASE}${API_CONFIG.ENDPOINTS.HEALTH}`, {
      signal: controller.signal,
    })
    clearTimeout(timeout)
    
    _backendAvailable = res.ok
    return _backendAvailable
  } catch {
    _backendAvailable = false
    return false
  }
}

/**
 * Returns the cached backend availability status.
 */
export function isBackendAvailable() {
  return _backendAvailable === true
}


// ── Clear Output (Demo Reset) ────────────────────────────────────────────

/**
 * Clear all pipeline output data for a fresh demo.
 * @returns {Promise<{cleared: number, message: string}>}
 */
export async function clearOutputData() {
  const res = await fetch(
    `${API_BASE}${API_CONFIG.ENDPOINTS.CLEAR_OUTPUT}`,
    { method: 'DELETE' }
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}


// ── Customer Listing ─────────────────────────────────────────────────────

/**
 * Fetch list of available customers.
 * Falls back to hardcoded data if backend is offline.
 */
export async function fetchCustomers() {
  try {
    const res = await fetch(`${API_BASE}${API_CONFIG.ENDPOINTS.CUSTOMERS}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const json = await res.json()
    
    // Transform backend data to frontend format
    return {
      customers: json.customers.map(c => ({
        id: c.id,
        label: c.label,
        folderId: c.folder_id,
        skIdCurr: c.sk_id_curr,
        target: c.target,
        targetLabel: c.target_label,
        hasOutput: c.has_output || false,
        info: {
          gender: c.gender || 'N/A',
          age: c.age || 0,
          income: c.income_type || 'N/A',
          loan: c.loan_purpose || 'N/A',
          amount: c.amt_credit ? formatVND(c.amt_credit) : 'N/A',
          education: c.education || 'N/A',
          housing: c.housing || 'N/A',
          familyStatus: c.family_status || 'N/A',
          ownRealty: c.own_realty || 'N',
          ownCar: c.own_car || 'N',
        },
        // Score data ONLY from output/ (never mock)
        scoreData: c.has_output ? {
          creditScore: c.credit_score,
          riskBand: c.risk_band,
          pdPct: c.pd_pct,
          recommendation: c.recommendation,
          fiveCTotal: c.five_c_total,
          fiveCScores: c.five_c_scores,
        } : null,
      })),
      total: json.total,
      source: 'backend',
    }
  } catch (err) {
    console.warn('[API] Backend offline for customers, using fallback data:', err.message)
    return {
      customers: CUSTOMERS_FALLBACK,
      total: CUSTOMERS_FALLBACK.length,
      source: 'fallback',
    }
  }
}


// ── Report Data ──────────────────────────────────────────────────────────

/**
 * Fetch detailed credit report JSON for a given customer.
 * Falls back to mock data if backend is unavailable.
 */
export async function fetchReportJSON(customerId) {
  try {
    const res = await fetch(
      `${API_BASE}${API_CONFIG.ENDPOINTS.REPORT_JSON(customerId)}`
    )
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const json = await res.json()
    return { data: json, source: 'backend' }
  } catch (err) {
    console.warn(`[API] Report fallback for customer ${customerId}:`, err.message)
    return { data: reportFallbackData, source: 'fallback' }
  }
}


// ── Pipeline Scoring ─────────────────────────────────────────────────────

/**
 * Submit a customer for credit scoring via the pipeline.
 */
export async function runScorePipeline(customerId, customerType = 'INDIVIDUAL') {
  const formData = new FormData()
  formData.append('applicant_id', customerId)
  formData.append('customer_type', customerType)

  const res = await fetch(
    `${API_BASE}${API_CONFIG.ENDPOINTS.SCORE}`,
    { method: 'POST', body: formData }
  )
  
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

/**
 * Submit multiple customers for parallel batch scoring.
 * Backend runs all pipelines concurrently with staggered starts.
 * 
 * @param {string[]} customerIds - Array of customer IDs to process
 * @param {number} staggerDelay - Seconds between each pipeline launch (default: 2)
 * @returns {Promise<{results: Object, total: number, success_count: number, duration_s: number}>}
 */
export async function runBatchScorePipeline(customerIds, staggerDelay = 2.0) {
  const res = await fetch(
    `${API_BASE}${API_CONFIG.ENDPOINTS.SCORE_BATCH}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_ids: customerIds,
        stagger_delay: staggerDelay,
      }),
    }
  )

  if (!res.ok) throw new Error(`Batch HTTP ${res.status}`)
  return await res.json()
}


// ── 2-Phase Pipeline (Human-in-the-Loop) ─────────────────────────────────

/**
 * Phase 1: Run OCR + LLM extraction only.
 * Returns extracted features + confidence for human review.
 */
export async function runIngestion(customerId) {
  const formData = new FormData()
  formData.append('applicant_id', customerId)

  const res = await fetch(
    `${API_BASE}${API_CONFIG.ENDPOINTS.INGEST}`,
    { method: 'POST', body: formData }
  )

  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

/**
 * Phase 2: Submit approved/edited data to complete pipeline (A2→A3→A4).
 *
 * @param {Object} params - { customer_id, application_row, raw_texts, thin_file_flag, identity_consistency_flag }
 */
export async function runProcessing(params) {
  const res = await fetch(
    `${API_BASE}${API_CONFIG.ENDPOINTS.PROCESS}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }
  )

  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

/**
 * Get the URL for PDF preview (inline).
 * Uses the Vite proxy in dev mode to avoid cross-origin iframe issues.
 */
export function getPdfPreviewUrl(customerId) {
  return `${API_BASE}${API_CONFIG.ENDPOINTS.REPORT_PDF(customerId, false)}`
}

/**
 * Get the URL for PDF download.
 */
export function getPdfDownloadUrl(customerId) {
  return `${API_BASE}${API_CONFIG.ENDPOINTS.REPORT_PDF(customerId, true)}`
}

/**
 * Download PDF as a file. Fetches the PDF as blob and triggers browser download.
 * This ensures the file is saved with .pdf extension regardless of browser behavior.
 */
export async function downloadPdf(customerId) {
  const url = `${API_BASE}${API_CONFIG.ENDPOINTS.REPORT_PDF(customerId, true)}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to download PDF: HTTP ${res.status}`)
  
  const blob = await res.blob()
  const blobUrl = URL.createObjectURL(blob)
  
  const a = document.createElement('a')
  a.href = blobUrl
  a.download = `credit_report_${customerId}.pdf`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(blobUrl)
}


// ── Helpers ──────────────────────────────────────────────────────────────

// Home Credit uses anonymized currency units → multiply by 100 for approximate VND display
const VND_SCALE = 100

function formatVND(rawAmount) {
  const amount = rawAmount * VND_SCALE
  if (amount >= 1e9) return `${(amount / 1e9).toFixed(1)}B VND`
  if (amount >= 1e6) return `${(amount / 1e6).toFixed(1)}M VND`
  if (amount >= 1e3) return `${(amount / 1e3).toFixed(0)}K VND`
  return `${amount.toLocaleString()} VND`
}

// Re-export constants
export { PIPELINE_LAYERS }
