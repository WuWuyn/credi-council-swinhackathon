export const API_CONFIG = {
  // Use Vite environment variable, fallback to localhost
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  ENDPOINTS: {
    // Health check
    HEALTH: '/health',

    // Customer listing
    CUSTOMERS: '/v1/customers',
    CUSTOMER_DETAIL: (customerId) => `/v1/customers/${customerId}`,

    // Pipeline Scoring
    SCORE: '/v1/score',
    SCORE_BATCH: '/v1/score/batch',
    CLEAR_OUTPUT: '/v1/output',

    // 2-Phase Pipeline (Human-in-the-Loop)
    INGEST: '/v1/ingest',       // Phase 1: OCR + LLM extraction only
    PROCESS: '/v1/process',     // Phase 2: A2→A3→A4 with approved data
    
    // Credit Report details
    REPORT_JSON: (customerId) => `/v1/report/${customerId}/json`,
    
    // PDF Generation and streaming
    REPORT_PDF: (customerId, download = false) => 
      `/v1/report/${customerId}/pdf${download ? '?download=1' : ''}`,
  }
}
