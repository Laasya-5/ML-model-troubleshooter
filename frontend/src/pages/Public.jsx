import React, { useState } from 'react';
import {
  ChevronRight, ShieldCheck, BarChart3, AlertTriangle, Activity, Wrench, FileText, Mail, Lock, User, Eye, EyeOff,
} from 'lucide-react';
import { api, errMsg } from '../api';
import { Button, ErrorBox } from '../ui';

const FEATURES = [
  [ShieldCheck, 'Model Validation'], [BarChart3, 'Performance Evaluation'], [AlertTriangle, 'Issue Diagnosis'],
  [Activity, 'Monitoring & Logs'], [Wrench, 'Fix Verification'], [FileText, 'Reports'],
];

function Nav({ go }) {
  return (
    <header className="landing-nav">
      <div className="brand"><div className="brand-icon">⬡</div><b>ML Model Troubleshooter</b></div>
      <div className="landing-links">
        <a href="#features">Features</a><a href="#workflow">Workflow</a><a href="#about">About</a>
        <button className="btn btn-secondary" onClick={() => go('Login')}>Login</button>
        <Button onClick={() => go('Register')}>Get Started</Button>
      </div>
    </header>
  );
}

export function Landing({ go }) {
  return (
    <div className="auth-shell">
      <Nav go={go} />
      <main className="hero">
        <div className="hero-copy">
          <span className="pill">AI MODEL HEALTH PLATFORM</span>
          <h1>Deploy. Monitor.<br /><em>Troubleshoot. Improve.</em></h1>
          <p>A unified platform to validate, diagnose and optimize your ML models — before and after deployment.</p>
          <div className="hero-actions">
            <Button onClick={() => go('Register')}>Get Started <ChevronRight size={17} /></Button>
            <button className="hero-learn" onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}>Learn More</button>
          </div>
          <div className="hero-stats">
            <div><b>2</b><small>Demo models included</small></div>
            <div><b>Real</b><small>Metrics, never mocked</small></div>
            <div><b>A/B</b><small>Model comparison</small></div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="glow" />
          <div className="mock-laptop">
            <div className="mock-top"><span>● ● ●</span><span>Model Analytics</span><span>•••</span></div>
            <div className="mock-body">
              <div className="mock-side">▰<br />▤<br />◈<br />⌁</div>
              <div className="mock-chart">
                <b>Model performance</b>
                <div className="chart-lines"><span /><span /><span />
                  <svg viewBox="0 0 400 150"><polyline points="0,120 65,92 125,105 190,55 250,67 315,25 400,15" fill="none" stroke="#a66bff" strokeWidth="4" /></svg>
                </div>
                <div className="mock-kpis"><i /><i /><i /></div>
              </div>
            </div>
            <div className="verified">✓</div>
          </div>
        </div>
      </main>
      <section id="features" className="feature-strip">
        {FEATURES.map(([I, t]) => <div key={t}><I size={21} /><span>{t}</span></div>)}
      </section>
      <footer className="landing-footer" id="about">© 2026 ML Model Troubleshooter · Built for healthier machine learning models</footer>
    </div>
  );
}

export function AuthPage({ mode, go, onAuthed }) {
  const isLogin = mode === 'Login';
  const [f, setF] = useState({ name: '', email: '', password: '' });
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!f.email || !f.password || (!isLogin && !f.name)) { setError('Please complete all fields.'); return; }
    setBusy(true);
    try {
      const r = isLogin ? await api.login({ email: f.email, password: f.password }) : await api.register(f);
      onAuthed(r.token, r.user);
    } catch (err) { setError(errMsg(err)); } finally { setBusy(false); }
  };

  return (
    <div className="auth-shell">
      <Nav go={go} />
      <div className="auth-card">
        <div className="auth-art">
          <div className="brand-icon big">⬡</div>
          <h2>ML Model<br />Troubleshooter</h2>
          <div className="art-orb"><div className="art-symbol">{isLogin ? '⌁' : '♙+'}</div></div>
          <p>Understand your model. Improve its performance.</p>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <h2>{isLogin ? 'Welcome Back' : 'Create Your Account'}</h2>
          <p>{isLogin ? 'Sign in to continue to your dashboard' : 'Start your ML troubleshooting journey'}</p>
          {!isLogin && (
            <label>Full Name<div className="input-icon"><User size={15} /><input value={f.name} onChange={set('name')} placeholder="Enter your full name" autoComplete="name" /></div></label>
          )}
          <label>Email Address<div className="input-icon"><Mail size={15} /><input type="email" value={f.email} onChange={set('email')} placeholder="Enter your email" autoComplete="email" /></div></label>
          <label>Password
            <div className="input-icon"><Lock size={15} />
              <input type={show ? 'text' : 'password'} value={f.password} onChange={set('password')}
                placeholder={isLogin ? 'Enter your password' : 'Create a strong password (6+ characters)'} autoComplete={isLogin ? 'current-password' : 'new-password'} />
              <button type="button" className="eye" onClick={() => setShow(!show)} aria-label="Show password">{show ? <EyeOff size={15} /> : <Eye size={15} />}</button>
            </div>
          </label>
          {error && <ErrorBox>{error}</ErrorBox>}
          <Button type="submit" full busy={busy}>{isLogin ? 'Sign In' : 'Register'}</Button>
          <div className="auth-switch">
            {isLogin ? "Don't have an account? " : 'Already have an account? '}
            <b onClick={() => go(isLogin ? 'Register' : 'Login')}>{isLogin ? 'Register' : 'Login'}</b>
          </div>
        </form>
      </div>
      <footer className="landing-footer">© 2026 ML Model Troubleshooter</footer>
    </div>
  );
}
