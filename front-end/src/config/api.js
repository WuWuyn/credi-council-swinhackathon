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
    
    // Credit Report details
    REPORT_JSON: (customerId) => `/v1/report/${customerId}/json`,
    
    // PDF Generation and streaming
    REPORT_PDF: (customerId, download = false) => 
      `/v1/report/${customerId}/pdf${download ? '?download=1' : ''}`,
  }
}
