import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Header from '../components/Header'
import { reportFallbackData } from '../data/mockData'
import { API_CONFIG } from '../config/api'

export default function CreditReportDetailPage() {
  const navigate = useNavigate()
  const { id } = useParams()
  const [showPdf, setShowPdf] = useState(false)
  
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchReport() {
      try {
        const res = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REPORT_JSON(id)}`)
        if (!res.ok) throw new Error(`HTTP error ${res.status}`)
        const json = await res.json()
        setData(json)
      } catch (err) {
        console.warn('Backend not running or failed. Falling back to offline mock data!', err)
        setData(reportFallbackData)
      } finally {
        setLoading(false)
      }
    }
    fetchReport()
  }, [id])

  if (loading) return <div className="p-8 text-center mt-20 text-slate-500 font-bold">Loading credit report data...</div>
  if (error || !data) return <div className="p-8 text-center text-red-500 mt-20 font-bold">Data load error: {error || 'No data found'}</div>

  const rd = data.report_data || {}
  const sd = data.shap_data || {}
  const cinfo = rd.customer_info || {}
  const exec = rd.executive_summary || {}

  const profile = {
    name: cinfo.name || `Customer #${id}`,
    id: id,
    occupation: cinfo.income_type || 'N/A',
    income: 'N/A',
    residency: 'N/A',
    location: 'N/A',
    assetType: cinfo.housing || 'N/A',
  }
  
  const score = {
    value: exec.credit_score || 0,
    label: exec.risk_band || 'N/A',
    change: 'New',
    defaultProb: `${exec.pd_pct || 0}%`,
  }

  const fiveC = {
    character: exec.five_c_scores?.character || 0,
    capacity: exec.five_c_scores?.capacity || 0,
    capital: exec.five_c_scores?.capital || 0,
    collateral: exec.five_c_scores?.collateral || 0,
    conditions: exec.five_c_scores?.conditions || 0,
  }

  const positiveFactors = (sd.top_positive_factors || []).map(f => ({
    label: f.label_vi || f.feature,
    points: `+${(f.shap_value * 100).toFixed(1)} pts`,
    width: `${Math.min(f.shap_value * 100 * 5, 100)}%`
  }))

  const negativeFactors = (sd.top_negative_factors || []).map(f => ({
    label: f.label_vi || f.feature,
    points: `${(f.shap_value * 100).toFixed(1)} pts`,
    width: `${Math.min(Math.abs(f.shap_value * 100 * 5), 100)}%`
  }))

  // Dynamic SVG Map for 5C (max 40)
  const maxC = 40
  const calcPoint = (val, max, angleOffset) => {
    const angle = angleOffset * (Math.PI / 180)
    const r = (val / max) * 45 // 45 is the max inner radius in the 100x100 viewBox
    const cx = 50 + r * Math.sin(angle)
    const cy = 50 - r * Math.cos(angle)
    return { cx, cy, pt: `${cx},${cy}` }
  }

  const ptCharacter = calcPoint(fiveC.character, maxC, 0)
  const ptCapacity = calcPoint(fiveC.capacity, maxC, 72)
  const ptCapital = calcPoint(fiveC.capital, maxC, 144)
  const ptCollateral = calcPoint(fiveC.collateral, maxC, 216)
  const ptConditions = calcPoint(fiveC.conditions, maxC, 288)

  const radarPoints = `${ptCharacter.pt} ${ptCapacity.pt} ${ptCapital.pt} ${ptCollateral.pt} ${ptConditions.pt}`

  // Standard static transactions array for demo as backend doesnt return it yet
  const transactions = [
    {
      icon: 'credit_score',
      type: 'AI Credit Evaluation Request',
      date: 'Today',
      amount: 'N/A',
      institution: 'CrediCouncil System',
      status: 'Completed',
      statusColor: 'bg-green-100 text-green-700',
    }
  ]

  // SVG circle calculations
  const radius = 70
  const circumference = 2 * Math.PI * radius
  const scoreRatio = score.value / 900  // max score ~900
  const dashOffset = circumference - (circumference * scoreRatio)

  return (
    <div className="min-h-screen bg-surface transition-colors duration-300">
      <Header
        variant="report"
        title="Credit Report Detail"
        onBack={() => navigate(-1)}
        actions={{ onPdfClick: () => setShowPdf(true) }}
      />

      {/* Content Area */}
      <div className="pt-12 pb-12 px-8 max-w-7xl mx-auto space-y-8">
        {/* Hero Info & Score */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* User Profile Card */}
          <div className="lg:col-span-2 bg-surface-container-lowest p-8 rounded-xl shadow-sm border border-outline-variant/15 flex flex-col md:flex-row gap-8 items-center">
            <div className="relative">
              <div className="w-32 h-32 rounded-full border-4 border-surface-container-high bg-gradient-to-br from-primary/10 to-primary/5 flex items-center justify-center">
                <span className="material-symbols-outlined text-primary text-5xl">person</span>
              </div>
              <div className="absolute bottom-0 right-0 bg-primary text-on-primary p-1.5 rounded-full border-4 border-surface-container-lowest">
                <span className="material-symbols-outlined text-sm block" style={{ fontVariationSettings: "'FILL' 1" }}>verified</span>
              </div>
            </div>
            <div className="flex-1 text-center md:text-left">
              <div className="flex flex-col md:flex-row md:items-end gap-2 md:gap-4 mb-4">
                <h3 className="font-headline text-3xl font-bold tracking-tight text-on-surface">{profile.name}</h3>
                <span className="text-slate-500 font-label text-sm tracking-widest uppercase pb-1">ID: {profile.id}</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-y-4 gap-x-8">
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Income Type</p>
                  <p className="text-on-surface font-medium">{profile.occupation}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Loan Purpose</p>
                  <p className="text-on-surface font-medium">{cinfo.loan_purpose || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Gender</p>
                  <p className="text-on-surface font-medium">{cinfo.gender || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Education</p>
                  <p className="text-on-surface font-medium">{cinfo.education || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Asset Type</p>
                  <p className="text-on-surface font-medium">{profile.assetType}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Score Radial Display */}
          <div className="bg-primary text-on-primary p-8 rounded-xl shadow-lg flex flex-col items-center justify-center text-center relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16 blur-3xl"></div>
            <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/5 rounded-full -ml-12 -mb-12 blur-2xl"></div>
            <p className="font-label text-[11px] font-bold uppercase tracking-[0.2em] mb-4 opacity-80">Credit Score Index</p>
            <div className="relative flex items-center justify-center">
              <svg className="w-40 h-40">
                <circle className="text-white/10" cx="80" cy="80" fill="transparent" r={radius} stroke="currentColor" strokeWidth="8" />
                <circle
                  className="text-white"
                  cx="80" cy="80"
                  fill="transparent"
                  r={radius}
                  stroke="currentColor"
                  strokeDasharray={circumference}
                  strokeDashoffset={dashOffset}
                  strokeLinecap="round"
                  strokeWidth="8"
                  style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%', transition: 'stroke-dashoffset 1s ease-in-out' }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="font-headline text-5xl font-black tracking-tighter">{score.value}</span>
                <span className="text-xs font-medium opacity-80 uppercase tracking-widest">{score.label}</span>
              </div>
            </div>
            <p className="mt-6 text-sm opacity-90 leading-relaxed px-4">
              <span className="font-bold">{cinfo.summary}</span>
            </p>
          </div>
        </section>

        {/* Detailed Analysis Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 5C Radar Analysis */}
          <div className="bg-surface-container-low p-8 rounded-xl border border-outline-variant/15 flex flex-col">
            <div className="flex justify-between items-start mb-10">
              <div>
                <h4 className="font-headline text-xl font-bold text-on-surface">5C Model Analysis</h4>
                <p className="text-slate-500 text-sm">Component Scores (Total: {exec.five_c_total || 0})</p>
              </div>
              <span className="bg-surface-container-lowest text-primary text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-tighter shadow-sm border border-primary/10">
                AI Computed
              </span>
            </div>
            <div className="flex-1 flex items-center justify-center relative py-12">
              {/* Custom Radar Chart */}
              <div className="relative w-72 h-72">
                {/* Background Pentagons */}
                <div className="absolute inset-0 radar-grid bg-slate-200/30 scale-100"></div>
                <div className="absolute inset-0 radar-grid bg-slate-200/50 scale-75"></div>
                <div className="absolute inset-0 radar-grid bg-slate-200/70 scale-50"></div>
                <div className="absolute inset-0 radar-grid bg-slate-200/90 scale-25"></div>
                {/* Active Data Polygon */}
                <svg className="absolute inset-0 w-full h-full drop-shadow-xl" viewBox="0 0 100 100">
                  <polygon
                    fill="rgba(227, 24, 55, 0.2)"
                    points={radarPoints}
                    stroke="#E31837"
                    strokeWidth="1.5"
                  />
                  {/* Nodes */}
                  <circle cx={ptCharacter.cx} cy={ptCharacter.cy} fill="#E31837" r="2.5" />
                  <circle cx={ptCapacity.cx} cy={ptCapacity.cy} fill="#E31837" r="2.5" />
                  <circle cx={ptCapital.cx} cy={ptCapital.cy} fill="#E31837" r="2.5" />
                  <circle cx={ptCollateral.cx} cy={ptCollateral.cy} fill="#E31837" r="2.5" />
                  <circle cx={ptConditions.cx} cy={ptConditions.cy} fill="#E31837" r="2.5" />
                </svg>
                {/* Labels */}
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] font-bold uppercase tracking-widest">Character</div>
                <div className="absolute top-1/4 -right-12 text-[10px] font-bold uppercase tracking-widest">Capacity</div>
                <div className="absolute -bottom-6 right-0 text-[10px] font-bold uppercase tracking-widest">Capital</div>
                <div className="absolute -bottom-6 left-0 text-[10px] font-bold uppercase tracking-widest">Collateral</div>
                <div className="absolute top-1/4 -left-12 text-[10px] font-bold uppercase tracking-widest">Conditions</div>
              </div>
            </div>
            <div className="grid grid-cols-5 gap-2 mt-8">
              {[
                { value: fiveC.character, label: 'CHR' },
                { value: fiveC.capacity, label: 'CAP' },
                { value: fiveC.capital, label: 'CPL' },
                { value: fiveC.collateral, label: 'COL' },
                { value: fiveC.conditions, label: 'CON' },
              ].map((item) => (
                <div key={item.label} className="text-center">
                  <span className="block text-primary font-bold text-lg">{item.value}</span>
                  <span className="text-[9px] text-slate-400 font-bold uppercase">{item.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Feature Importance (XAI) */}
          <div className="bg-surface-container-low p-8 rounded-xl border border-outline-variant/15 flex flex-col">
            <div className="mb-8">
              <h4 className="font-headline text-xl font-bold text-on-surface">Feature Importance (XAI)</h4>
              <p className="text-slate-500 text-sm">AI Score Explanation</p>
            </div>
            <div className="space-y-6 flex-1">
              {/* Positive Factors */}
              <div className="space-y-4">
                <h5 className="text-[10px] font-bold text-primary uppercase tracking-widest flex items-center gap-2">
                  <span className="material-symbols-outlined text-sm">trending_up</span>
                  Key Positive Factors
                </h5>
                <div className="space-y-3">
                  {positiveFactors.map((factor, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-on-surface-variant">{factor.label}</span>
                        <span className="font-bold text-primary">{factor.points}</span>
                      </div>
                      <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-primary rounded-full transition-all duration-1000" style={{ width: factor.width }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {/* Negative Factors */}
              <div className="space-y-4 pt-4 border-t border-slate-200/50">
                <h5 className="text-[10px] font-bold text-error uppercase tracking-widest flex items-center gap-2">
                  <span className="material-symbols-outlined text-sm">trending_down</span>
                  Key Risk Factors
                </h5>
                <div className="space-y-3">
                  {negativeFactors.map((factor, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-on-surface-variant">{factor.label}</span>
                        <span className="font-bold text-error">{factor.points}</span>
                      </div>
                      <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-error rounded-full transition-all duration-1000" style={{ width: factor.width }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-8 p-4 bg-surface-container-highest/30 rounded-lg">
              <p className="text-[11px] italic text-slate-500 leading-relaxed">
                * CrediCouncil AI v2.1 uses Ensemble Learning averaging 248 variables. Area under the curve (AUC): 0.94
              </p>
            </div>
          </div>
        </section>

        {/* Transaction History Table */}
        <section className="bg-surface-container-lowest p-8 rounded-xl shadow-sm border border-outline-variant/15 overflow-hidden">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
            <div>
              <h4 className="font-headline text-xl font-bold text-on-surface">Transaction & Credit History</h4>
              <p className="text-slate-500 text-sm">Detailed activities in the last 12 months</p>
            </div>
            <div className="flex gap-2">
              <button className="px-4 py-2 bg-surface-container-low text-xs font-bold uppercase tracking-wider rounded-lg hover:bg-surface-container-high transition-all">
                Filter
              </button>
              <button className="px-4 py-2 bg-surface-container-low text-xs font-bold uppercase tracking-wider rounded-lg hover:bg-surface-container-high transition-all">
                Fullscreen
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low/50">
                  <th className="px-6 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Transaction Type</th>
                  <th className="px-6 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Date</th>
                  <th className="px-6 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Amount (VND)</th>
                  <th className="px-6 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Financial Institution</th>
                  <th className="px-6 py-4 text-[10px] font-bold text-slate-400 uppercase tracking-widest">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100/50">
                {transactions.map((tx, i) => (
                  <tr key={i} className="hover:bg-slate-50 transition-colors group">
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-3">
                        <span className={`material-symbols-outlined ${tx.isError ? 'text-error bg-error-container' : 'text-primary bg-primary/10'} p-2 rounded-lg`}>
                          {tx.icon}
                        </span>
                        <span className="text-sm font-semibold text-on-surface">{tx.type}</span>
                      </div>
                    </td>
                    <td className="px-6 py-5 text-sm text-slate-600 font-mono">{tx.date}</td>
                    <td className="px-6 py-5 text-sm font-bold text-on-surface font-mono">{tx.amount}</td>
                    <td className="px-6 py-5 text-sm text-slate-600">{tx.institution}</td>
                    <td className="px-6 py-5">
                      <span className={`${tx.statusColor} text-[10px] font-bold px-2 py-1 rounded uppercase`}>
                        {tx.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-8 pb-12">
        <div className="bg-surface-container-high/40 p-6 rounded-xl border border-dashed border-outline-variant flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="bg-white p-3 rounded-full shadow-sm">
              <span className="material-symbols-outlined text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>info</span>
            </div>
            <div>
              <p className="font-bold text-on-surface">This report is automatically generated by CrediCouncil AI Engine.</p>
              <p className="text-sm text-slate-500">Data synchronized with national credit bureau (CIC) at 08:30 AM today.</p>
            </div>
          </div>
          <div className="flex gap-4">
            <button className="text-slate-500 font-bold text-sm hover:text-primary transition-colors">Report Discrepancy</button>
            <button className="bg-white border border-outline-variant/30 px-6 py-2 rounded-lg text-sm font-bold shadow-sm hover:shadow-md transition-all active:scale-95">
              Direct Print
            </button>
          </div>
        </div>
      </footer>

      {/* PDF PREVIEW MODAL */}
      {showPdf && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200" onClick={() => setShowPdf(false)}>
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden animate-in slide-in-from-bottom-8 duration-300" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center px-6 py-4 border-b border-outline-variant/30 bg-surface-container-lowest">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg text-primary">
                  <span className="material-symbols-outlined text-xl">description</span>
                </div>
                <div>
                  <h3 className="font-bold text-lg text-on-surface">Credit Report Preview (PDF)</h3>
                  <p className="text-sm text-slate-500 font-mono">ID: {id}</p>
                </div>
              </div>
              <button 
                className="p-2 hover:bg-surface-container-low rounded-full transition-colors text-slate-500"
                onClick={() => setShowPdf(false)}
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            
            <div className="flex-1 bg-slate-100 p-0 relative overflow-hidden min-h-[500px]">
              <iframe 
                src={`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REPORT_PDF(id)}`} 
                title="Credit Report PDF"
                className="absolute inset-0 w-full h-full border-0"
              />
            </div>
            
            <div className="px-6 py-4 bg-surface-container-lowest border-t border-outline-variant/30 flex justify-end gap-4">
              <button 
                className="px-6 py-2.5 rounded-lg font-bold text-slate-600 hover:bg-slate-100 transition-colors"
                onClick={() => setShowPdf(false)}
              >
                Close
              </button>
              <button 
                className="px-6 py-2.5 rounded-lg font-bold text-white bg-primary hover:opacity-90 transition-opacity flex items-center gap-2 shadow-sm shadow-primary/30"
                onClick={() => window.open(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.REPORT_PDF(id, true)}`, '_blank')}
              >
                <span className="material-symbols-outlined text-[18px]">download</span>
                Download Official PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
