import React from 'react';
import { Loader2, AlertTriangle } from 'lucide-react';

export const Header = ({ title, desc, right }) => (
  <div className="page-head">
    <div>
      <div className="eyebrow">ML MODEL TROUBLESHOOTER / WORKSPACE</div>
      <h1>{title}</h1>
      <p>{desc}</p>
    </div>
    {right}
  </div>
);

export const Card = ({ children, className = '', ...rest }) => (
  <section className={`card ${className}`} {...rest}>{children}</section>
);

export const Metric = ({ label, value, sub, icon: Icon, tone }) => (
  <Card className="metric">
    <div className="metric-top"><span>{label}</span>{Icon && <span className="metric-icon"><Icon size={17} /></span>}</div>
    <strong className={tone ? `tone-${tone}` : ''}>{value ?? '—'}</strong>
    {sub && <small className={tone ? `tone-${tone}` : ''}>{sub}</small>}
  </Card>
);

export const Button = ({ children, onClick, secondary = false, disabled = false, busy = false, type = 'button', full = false }) => (
  <button type={type} className={`btn ${secondary ? 'btn-secondary' : ''} ${full ? 'full' : ''}`} onClick={onClick} disabled={disabled || busy}>
    {busy && <Loader2 size={14} className="spin" />}{children}
  </button>
);

export const Loading = ({ text = 'Loading…' }) => (
  <div className="loading"><Loader2 className="spin" size={20} /> {text}</div>
);

export const ErrorBox = ({ children }) => (
  <div className="error-box"><AlertTriangle size={16} /> <span>{children}</span></div>
);

export const Empty = ({ title, children }) => (
  <Card className="empty-card"><h3>{title}</h3><p className="muted">{children}</p></Card>
);

export const Steps = ({ current }) => (
  <div className="steps">
    {['Upload', 'Validate', 'Run Checks', 'Results'].map((s, i) => (
      <div className={`step ${i === current ? 'current' : i < current ? 'done' : ''}`} key={s}>
        <b>{i < current ? '✓' : i + 1}</b>{s}
      </div>
    ))}
  </div>
);

export const statusClass = (s) => (s === 'Passed' ? 'good' : s === 'Warning' || s === 'Passed with warnings' ? 'warn' : 'bad');
export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`);
export const signed = (v, d = 3) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}`);
