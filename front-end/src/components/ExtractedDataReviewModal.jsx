/**
 * ExtractedDataReviewModal — Human-in-the-Loop Review Popup
 *
 * Displays the ~121 extracted features grouped by document source
 * after A1 Ingestion (OCR + LLM). Staff can review, edit values,
 * and approve before the pipeline continues to A2→A3→A4.
 */

import { useState, useMemo, useCallback } from 'react'
import './ExtractedDataReviewModal.css'

// ── Confidence color helpers ─────────────────────────────────────────────

function getConfidenceClass(conf) {
  if (conf >= 0.90) return 'conf-high'
  if (conf >= 0.70) return 'conf-mid'
  return 'conf-low'
}

function getConfidenceLabel(conf) {
  if (conf >= 0.90) return 'Cao'
  if (conf >= 0.70) return 'Trung bình'
  return 'Thấp'
}

// ── VND scaling (display only) ────────────────────────────────────────────
// Home Credit raw units are anonymized → multiply ×100 for approximate VND
const VND_SCALE = 100

// Fields that represent monetary values (AMT_* prefix)
const MONETARY_FIELDS = new Set([
  'AMT_INCOME_TOTAL',
  'AMT_CREDIT',
  'AMT_ANNUITY',
  'AMT_GOODS_PRICE',
])

function isMonetaryField(fieldName) {
  return MONETARY_FIELDS.has(fieldName)
}

/** Scale raw value → display value (for show only) */
function toDisplayValue(fieldName, rawValue) {
  if (rawValue === null || rawValue === undefined) return rawValue
  if (isMonetaryField(fieldName) && typeof rawValue === 'number') {
    return rawValue * VND_SCALE
  }
  return rawValue
}

/** Scale display value → raw value (to store back) */
function fromDisplayValue(fieldName, displayValue) {
  if (displayValue === null || displayValue === undefined) return displayValue
  if (isMonetaryField(fieldName) && typeof displayValue === 'number') {
    return displayValue / VND_SCALE
  }
  return displayValue
}

// ── Format helpers ───────────────────────────────────────────────────────

function formatValue(value) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') {
    if (Math.abs(value) >= 1000) return value.toLocaleString('vi-VN')
    return String(value)
  }
  return String(value)
}

function formatVNDShort(value) {
  if (value === null || value === undefined) return '—'
  const v = Math.abs(value)
  if (v >= 1e9) return `${(value / 1e9).toFixed(1)} tỷ`
  if (v >= 1e6) return `${(value / 1e6).toFixed(1)} triệu`
  if (v >= 1e3) return `${(value / 1e3).toFixed(0)}K`
  return value.toLocaleString('vi-VN')
}

// ══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════════

export default function ExtractedDataReviewModal({
  ingestionData,     // IngestionResponse from backend
  customerId,        // customer display label
  onApprove,         // callback(editedApplicationRow, metadata) 
  onCancel,          // callback()
  isProcessing = false,
}) {
  const { application_row, confidence_map, field_metadata, raw_texts } = ingestionData

  // ── Editable state ─────────────────────────────────────────────────
  const [editedRow, setEditedRow] = useState(() => ({ ...application_row }))
  const [editedFields, setEditedFields] = useState(new Set())
  const [expandedGroups, setExpandedGroups] = useState(() => {
    // Auto-expand groups that have low-confidence fields
    const expanded = new Set()
    for (const group of field_metadata || []) {
      const hasLowConf = group.fields?.some(f => f.confidence < 0.85 && f.value !== null)
      if (hasLowConf) expanded.add(group.group_id)
    }
    // Always expand at least identity and loan
    expanded.add('identity')
    expanded.add('loan')
    return expanded
  })

  // Filter modes
  const [filterMode, setFilterMode] = useState('all') // 'all' | 'low_confidence' | 'edited'

  // ── Statistics ─────────────────────────────────────────────────────
  const stats = useMemo(() => {
    let totalFields = 0
    let filledFields = 0
    let lowConfFields = 0
    let highConfFields = 0
    for (const group of field_metadata || []) {
      for (const f of group.fields || []) {
        totalFields++
        if (f.value !== null && f.value !== undefined && f.value !== '') filledFields++
        if (f.confidence < 0.70 && f.value !== null) lowConfFields++
        if (f.confidence >= 0.90) highConfFields++
      }
    }
    return { totalFields, filledFields, lowConfFields, highConfFields, editedCount: editedFields.size }
  }, [field_metadata, editedFields])

  // ── Handlers ───────────────────────────────────────────────────────
  const handleFieldChange = useCallback((fieldName, newValue) => {
    setEditedRow(prev => ({ ...prev, [fieldName]: newValue }))
    setEditedFields(prev => new Set([...prev, fieldName]))
  }, [])

  const toggleGroup = useCallback((groupId) => {
    setExpandedGroups(prev => {
      const next = new Set(prev)
      next.has(groupId) ? next.delete(groupId) : next.add(groupId)
      return next
    })
  }, [])

  const handleApprove = useCallback(() => {
    if (onApprove) {
      onApprove(editedRow, {
        raw_texts: raw_texts || {},
        thin_file_flag: ingestionData.thin_file_flag,
        identity_consistency_flag: ingestionData.identity_consistency_flag,
        edited_fields: [...editedFields],
      })
    }
  }, [editedRow, editedFields, raw_texts, ingestionData, onApprove])

  // ── Filter logic ──────────────────────────────────────────────────
  const filteredGroups = useMemo(() => {
    if (!field_metadata) return []
    return field_metadata.map(group => {
      let fields = group.fields
      if (filterMode === 'low_confidence') {
        fields = fields.filter(f => f.confidence < 0.85 && f.value !== null)
      } else if (filterMode === 'edited') {
        fields = fields.filter(f => editedFields.has(f.field_name))
      }
      return { ...group, fields }
    }).filter(g => g.fields.length > 0)
  }, [field_metadata, filterMode, editedFields])

  // ══════════════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════════════

  return (
    <div className="review-overlay" onClick={(e) => e.target === e.currentTarget && !isProcessing && onCancel?.()}>
      <div className="review-modal">
        {/* ── Header ────────────────────────────────────────────── */}
        <div className="review-header">
          <div className="review-header-left">
            <div className="review-header-icon">
              <span className="material-symbols-outlined">fact_check</span>
            </div>
            <div>
              <h2>Xác Nhận Dữ Liệu Trích Xuất</h2>
              <p className="review-header-sub">
                Kiểm tra và xác nhận dữ liệu OCR trước khi chấm điểm tín dụng · {customerId}
              </p>
            </div>
          </div>
          <button
            className="review-close-btn"
            onClick={onCancel}
            disabled={isProcessing}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* ── Stats bar ─────────────────────────────────────────── */}
        <div className="review-stats-bar">
          <div className="review-stat">
            <span className="material-symbols-outlined" style={{ color: '#4caf50', fontSize: 16 }}>check_circle</span>
            <span className="stat-value">{stats.filledFields}</span>
            <span className="stat-label">/ {stats.totalFields} trường</span>
          </div>
          <div className="review-stat">
            <span className="material-symbols-outlined" style={{ color: '#ff9800', fontSize: 16 }}>warning</span>
            <span className="stat-value warn">{stats.lowConfFields}</span>
            <span className="stat-label">cần xác nhận</span>
          </div>
          <div className="review-stat">
            <span className="material-symbols-outlined" style={{ color: '#2196f3', fontSize: 16 }}>edit</span>
            <span className="stat-value edit">{stats.editedCount}</span>
            <span className="stat-label">đã sửa</span>
          </div>

          {/* Filter buttons */}
          <div className="review-filters">
            <button
              className={`filter-btn ${filterMode === 'all' ? 'active' : ''}`}
              onClick={() => setFilterMode('all')}
            >Tất cả</button>
            <button
              className={`filter-btn ${filterMode === 'low_confidence' ? 'active' : ''}`}
              onClick={() => setFilterMode('low_confidence')}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>warning</span>
              Cần xác nhận
            </button>
            <button
              className={`filter-btn ${filterMode === 'edited' ? 'active' : ''}`}
              onClick={() => setFilterMode('edited')}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>edit</span>
              Đã sửa ({editedFields.size})
            </button>
          </div>
        </div>

        {/* ── Body: Field groups ─────────────────────────────────── */}
        <div className="review-body">
          {filteredGroups.map(group => {
            const isExpanded = expandedGroups.has(group.group_id)
            const lowCount = group.fields.filter(f => f.confidence < 0.70 && f.value !== null).length
            const editCount = group.fields.filter(f => editedFields.has(f.field_name)).length

            return (
              <div key={group.group_id} className={`review-group ${isExpanded ? 'expanded' : ''}`}>
                <div className="review-group-header" onClick={() => toggleGroup(group.group_id)}>
                  <div className="review-group-left">
                    <span className="material-symbols-outlined group-icon">{group.icon}</span>
                    <span className="group-title">{group.group_label}</span>
                    <span className="group-source">{group.source_document}</span>
                    {lowCount > 0 && (
                      <span className="group-badge warn">
                        <span className="material-symbols-outlined" style={{ fontSize: 11 }}>warning</span>
                        {lowCount}
                      </span>
                    )}
                    {editCount > 0 && (
                      <span className="group-badge edit">
                        <span className="material-symbols-outlined" style={{ fontSize: 11 }}>edit</span>
                        {editCount}
                      </span>
                    )}
                  </div>
                  <div className="review-group-right">
                    <span className="group-field-count">{group.fields.length} trường</span>
                    <svg
                      className={`group-arrow ${isExpanded ? 'open' : ''}`}
                      width="12" height="12" viewBox="0 0 24 24"
                      fill="none" stroke="currentColor" strokeWidth="2.5"
                    >
                      <polyline points="6 9 12 15 18 9" />
                    </svg>
                  </div>
                </div>

                {isExpanded && (
                  <div className="review-group-body">
                    <div className="review-fields-grid">
                      {group.fields.map(field => {
                        const isEdited = editedFields.has(field.field_name)
                        const confClass = getConfidenceClass(field.confidence)
                        const currentValue = editedRow[field.field_name]

                        return (
                          <div
                            key={field.field_name}
                            className={`review-field ${confClass} ${isEdited ? 'edited' : ''}`}
                          >
                            <div className="field-header">
                              <label className="field-label">
                                {field.label_vi}
                                {isMonetaryField(field.field_name) && (
                                  <span className="field-vnd-tag">VND ×100</span>
                                )}
                                <span className="field-name">{field.field_name}</span>
                              </label>
                              <div className={`confidence-badge ${confClass}`}>
                                {Math.round(field.confidence * 100)}%
                                <span className="conf-text">{getConfidenceLabel(field.confidence)}</span>
                              </div>
                            </div>

                            <div className="field-input-wrap">
                              {field.field_type === 'enum' && field.options ? (
                                <select
                                  value={currentValue ?? ''}
                                  onChange={(e) => handleFieldChange(field.field_name, e.target.value)}
                                  className={`field-input field-select ${isEdited ? 'is-edited' : ''}`}
                                >
                                  <option value="">— chọn —</option>
                                  {field.options.map(opt => (
                                    <option key={opt} value={opt}>{opt}</option>
                                  ))}
                                </select>
                              ) : field.field_type === 'number' ? (
                                <>
                                  <input
                                    type="number"
                                    value={toDisplayValue(field.field_name, currentValue) ?? ''}
                                    onChange={(e) => {
                                      const displayNum = e.target.value === '' ? null : parseFloat(e.target.value)
                                      const rawNum = fromDisplayValue(field.field_name, displayNum)
                                      handleFieldChange(field.field_name, rawNum)
                                    }}
                                    className={`field-input ${isEdited ? 'is-edited' : ''}`}
                                    placeholder="—"
                                    step="any"
                                  />
                                  {isMonetaryField(field.field_name) && currentValue != null && (
                                    <span className="field-vnd-display">
                                      {formatVNDShort(currentValue * VND_SCALE)} VND
                                    </span>
                                  )}
                                </>
                              ) : (
                                <input
                                  type="text"
                                  value={currentValue ?? ''}
                                  onChange={(e) => handleFieldChange(field.field_name, e.target.value || null)}
                                  className={`field-input ${isEdited ? 'is-edited' : ''}`}
                                  placeholder="—"
                                />
                              )}
                              {isEdited && (
                                <span className="edited-indicator" title="Đã chỉnh sửa">
                                  <span className="material-symbols-outlined" style={{ fontSize: 14 }}>edit</span>
                                </span>
                              )}
                            </div>

                            {/* Show original value if edited */}
                            {isEdited && application_row[field.field_name] !== undefined && (
                              <div className="field-original">
                                Gốc: <strong>{formatValue(toDisplayValue(field.field_name, application_row[field.field_name]))}</strong>
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* ── Footer ────────────────────────────────────────────── */}
        <div className="review-footer">
          <div className="review-footer-info">
            {editedFields.size > 0 && (
              <span className="footer-edited-note">
                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>edit_note</span>
                {editedFields.size} trường đã được chỉnh sửa
              </span>
            )}
          </div>
          <div className="review-footer-actions">
            <button
              className="review-btn cancel"
              onClick={onCancel}
              disabled={isProcessing}
            >
              Hủy bỏ
            </button>
            <button
              className="review-btn approve"
              onClick={handleApprove}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <>
                  <span className="spinner white" />
                  Đang xử lý A2→A3→A4...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
                  Phê duyệt & Tiếp tục
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
