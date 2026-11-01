import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity, CheckCircle2, ClipboardCheck, CloudUpload, FileText, Gauge, LogOut, Menu, Microscope,
  ShieldCheck, SlidersHorizontal, Wrench, X, BadgeCheck,
} from 'lucide-react';
import { api, getToken, setToken, setUnauthorizedHandler, errMsg } from './api';
import { Landing, AuthPage } from './pages/Public';
import { GetStarted, Validate, RunChecks, Verification } from './pages/Workflow';
import { Overview, Troubleshooting, Monitor } from './pages/Insights';
import { ModelLab } from './pages/Lab';
import { FixVerification } from './pages/Fix';
import { Reports } from './pages/Reports';
import { Loading } from './ui';

const NAV = [
  ['Get Started', CloudUpload], ['Validate Model', ShieldCheck], ['Run Checks', ClipboardCheck],
  ['Model Overview', Gauge], ['Model Lab', Microscope], ['Troubleshooting', Wrench],
  ['Monitor & Logs', Activity], ['Verification', BadgeCheck], ['Fix Verification', SlidersHorizontal],
  ['Reports', FileText],
];
const PUBLIC = ['Landing Page', 'Login', 'Register'];

export default function App() {
  const [page, setPage] = useState('Landing Page');
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(!!getToken());
  const [models, setModels] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [fixResult, setFixResult] = useState(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [toast, setToast] = useState('');

  const notify = useCallback((msg) => { setToast(msg); setTimeout(() => setToast(''), 3500); }, []);
  const go = useCallback((p) => { setPage(p); setMobileOpen(false); window.scrollTo(0, 0); }, []);

  const refreshModels = useCallback(async (selectId) => {
    const list = await api.models();
    setModels(list);
    setActiveId((cur) => (selectId ?? (list.some((m) => m.id === cur) ? cur : list[0]?.id ?? null)));
    return list;
  }, []);

  const logout = useCallback(() => {
    setToken(''); setUser(null); setModels([]); setActiveId(null); setFixResult(null); setPage('Landing Page');
  }, []);
  useEffect(() => { setUnauthorizedHandler(() => { logout(); }); }, [logout]);

  useEffect(() => {
    if (!getToken()) return;
    api.me().then((r) => { setUser(r.user); setPage('Get Started'); return refreshModels(); })
      .catch(() => setToken('')).finally(() => setBooting(false));
  }, [refreshModels]);

  const onAuthed = async (token, u) => {
    setToken(token); setUser(u);
    try { await refreshModels(); } catch (e) { notify(errMsg(e)); }
    go('Get Started');
  };

  if (booting) return <div className="boot"><Loading text="Starting…" /></div>;
  if (!user) {
    if (page === 'Login' || page === 'Register') return <AuthPage key={page} mode={page} go={go} onAuthed={onAuthed} />;
    return <Landing go={go} />;
  }

  const ctx = { models, activeId, setActiveId, go, notify, refreshModels, fixResult, setFixResult, user };
  const body = () => {
    switch (page) {
      case 'Validate Model': return <Validate ctx={ctx} />;
      case 'Run Checks': return <RunChecks ctx={ctx} />;
      case 'Model Overview': return <Overview ctx={ctx} />;
      case 'Model Lab': return <ModelLab ctx={ctx} />;
      case 'Troubleshooting': return <Troubleshooting ctx={ctx} />;
      case 'Monitor & Logs': return <Monitor ctx={ctx} />;
      case 'Verification': return <Verification ctx={ctx} />;
      case 'Fix Verification': return <FixVerification key={fixResult ? 'fx' : 'free'} ctx={ctx} />;
      case 'Reports': return <Reports ctx={ctx} />;
      default: return <GetStarted ctx={ctx} />;
    }
  };

  return (
    <div className="app-layout">
      <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
        <div className="brand"><div className="brand-icon">⬡</div><span>ML Model<br />Troubleshooter</span>
          <button className="close-menu" onClick={() => setMobileOpen(false)}><X size={18} /></button></div>
        <div className="nav-list">
          {NAV.map(([label, Icon]) => (
            <button key={label} className={`nav-item ${page === label ? 'active' : ''}`} onClick={() => go(label)}><Icon size={16} /><span>{label}</span></button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className="user-chip"><div className="avatar">{user.name?.[0]?.toUpperCase() || 'U'}</div><div><b>{user.name}</b><small>{user.email}</small></div></div>
          <button className="nav-item" onClick={logout}><LogOut size={16} />Logout</button>
        </div>
      </aside>
      <main className="main-area">
        <div className="mobile-top"><button onClick={() => setMobileOpen(true)}><Menu /></button><b>ML Model Troubleshooter</b></div>
        {page !== 'Get Started' && page !== 'Reports' || models.length ? (
          <div className="model-bar">
            <span>Active model</span>
            <select value={activeId ?? ''} onChange={(e) => { setActiveId(Number(e.target.value)); setFixResult(null); }}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
            <span className="online-dot-wrap"><span className="online-dot" /> API connected</span>
          </div>
        ) : null}
        <div className="content" key={`${page}-${activeId}`}>{body()}</div>
        <footer className="app-footer">ML Model Troubleshooter <span>All metrics are computed live from your models</span></footer>
      </main>
      {mobileOpen && <div className="scrim" onClick={() => setMobileOpen(false)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
