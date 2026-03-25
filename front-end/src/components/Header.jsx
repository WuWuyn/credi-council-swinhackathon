import { Link, useLocation } from 'react-router-dom'

export default function Header({ variant = 'default', title, onBack, actions }) {
  const location = useLocation()

  const navLinks = [
    { label: 'Dashboard', href: '/batch', icon: 'grid_view' },
    { label: 'Batch', href: '/batch', icon: 'layers' },
    { label: 'Pipeline', href: '/pipeline', icon: 'account_tree' },
    { label: 'History', href: '/batch', icon: 'history' },
  ]

  if (variant === 'report') {
    return (
      <header className="sticky top-0 w-full z-40 h-16 bg-white/80 backdrop-blur-xl flex justify-center items-center border-b border-slate-200/50 shadow-sm">
        <div className="max-w-7xl w-full px-8 flex justify-between items-center">
          <div className="flex items-center gap-4">
              {onBack && (
                <button
                  onClick={onBack}
                  className="text-slate-500 hover:bg-slate-200/50 p-2 rounded-lg transition-all active:scale-95"
                >
                  <span className="material-symbols-outlined">arrow_back</span>
                </button>
              )}
              <h2 className="font-headline font-bold tracking-tight text-slate-900 text-lg">
                {title || 'Credit Report Detail'}
              </h2>
            </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={actions?.onPdfClick}
              className="bg-primary text-on-primary flex items-center gap-2 px-5 py-2 rounded-full font-semibold transition-all duration-200 ease-in-out active:scale-95 shadow-md hover:opacity-90">
              <span className="material-symbols-outlined text-[20px]">picture_as_pdf</span>
              <span className="text-sm">Export PDF Report</span>
            </button>
            <div className="h-8 w-[1px] bg-slate-200 mx-2"></div>
            <button className="text-slate-500 hover:bg-slate-200/50 p-2 rounded-lg transition-all">
              <span className="material-symbols-outlined">notifications</span>
            </button>
            <button className="text-slate-500 hover:bg-slate-200/50 p-2 rounded-lg transition-all">
              <span className="material-symbols-outlined">settings</span>
            </button>
          </div>
        </div>
      </header>
    )
  }

  if (variant === 'pipeline') {
    return (
      <header className="w-full bg-white border-b border-slate-100 h-14 px-6 shrink-0">
        <div className="h-full flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 bg-primary rounded flex items-center justify-center">
              <span className="material-symbols-outlined text-white text-lg">account_balance</span>
            </div>
            <Link to="/batch" className="text-lg font-extrabold font-headline text-slate-900 tracking-tight hover:text-primary transition-colors">
              CrediCouncil AI
            </Link>
            <span className="w-px h-3 bg-slate-200 mx-1"></span>
            <p className="text-[9px] text-slate-400 font-bold uppercase tracking-[0.2em] mt-0.5">Multi-Layer Pipeline Engine</p>
          </div>
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2 px-3 py-1 bg-green-50 rounded-full border border-green-100">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 status-pulse"></span>
              <span className="text-[9px] font-bold text-green-700 uppercase tracking-wider">Intelligence Active</span>
            </div>
            <div className="flex items-center gap-3 border-l border-slate-100 pl-5">
              <div className="text-right">
                <p className="text-[9px] font-bold text-slate-900 leading-tight">Admin Console</p>
                <p className="text-[8px] text-slate-400">Node Cluster HKG-01</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-slate-100 border border-slate-200 overflow-hidden flex items-center justify-center">
                <span className="material-symbols-outlined text-slate-400">person</span>
              </div>
            </div>
          </div>
        </div>
      </header>
    )
  }

  // Default header (Batch Processing)
  return (
    <header className="w-full top-0 sticky z-40 bg-[#f7fafc] shadow-none">
      <div className="flex justify-between items-center w-full px-8 py-4 max-w-full mx-auto">
        <div className="flex items-center gap-8">
          <Link to="/batch" className="text-xl font-extrabold tracking-tighter text-primary font-headline">
            CrediCouncil AI
          </Link>
          <nav className="hidden md:flex gap-6 items-center">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.href && link.label === 'Batch'
              return (
                <Link
                  key={link.label}
                  to={link.href}
                  className={`font-headline text-sm font-semibold tracking-tight transition-colors duration-200 ${
                    isActive
                      ? 'text-primary font-bold border-b-2 border-primary pb-1'
                      : 'text-[#181c1e] font-medium hover:text-primary'
                  }`}
                >
                  {link.label}
                </Link>
              )
            })}
          </nav>
        </div>
        <div className="flex items-center gap-4">
          <button className="p-2 text-on-surface-variant hover:bg-surface-container transition-colors rounded-full">
            <span className="material-symbols-outlined">notifications</span>
          </button>
          <button className="p-2 text-on-surface-variant hover:bg-surface-container transition-colors rounded-full">
            <span className="material-symbols-outlined">settings</span>
          </button>
          <div className="h-8 w-8 rounded-full bg-slate-100 border border-slate-200 overflow-hidden flex items-center justify-center">
            <span className="material-symbols-outlined text-slate-400">person</span>
          </div>
        </div>
      </div>
      <div className="bg-[#e0e3e5] h-[1px]"></div>
    </header>
  )
}
