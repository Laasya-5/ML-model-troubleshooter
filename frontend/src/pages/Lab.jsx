import React, { useEffect, useState } from 'react';
import { ShieldCheck, Gauge, Activity, BarChart3, AlertTriangle } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { api, errMsg } from '../api';
import { Header, Card, Metric, Button, Loading, ErrorBox, Empty, fmt, pct } from '../ui';
import { ComparisonView, Confusion } from './Fix';

const TABS = ['Make Prediction', 'Evaluate Model', 'Compare Versions'];

function ResultCard({ r, tag, agree }) {
  const cls = /high/i.test(r.label) ? 'risk-high' : /low/i.test(r.label) ? 'risk-low' : 'risk-neutral';
  return (
    <Card className="prediction-card">
      {tag && <span className="pill dark">{tag}</span>}
      <h3>Prediction Result</h3>
      <div className="prediction-symbol"><ShieldCheck size={38} /></div>
      <div><span className="muted">{r.modelName}</span></div>
      <span className={`risk ${cls}`}>{r.label}</span>
      {r.confidence != null && <p>Confidence: <b>{r.confidence.toFixed(2)}</b></p>}
      {r.probabilities && (
        <div className="probs">{Object.entries(r.probabilities).map(([k, v]) => (
          <div key={k}><span>{k}</span><div className="bar"><i style={{ width: `${v * 100}%` }} /></div><b>{(v * 100).toFixed(1)}%</b></div>))}</div>
      )}
      <small className="muted">Computed by the model in {r.latencyMs} ms</small>
      {agree != null && <span className={`status ${agree ? 'good' : 'warn'}`}>{agree ? 'Both models agree' : 'Models disagree'}</span>}
      {r.warnings?.map((w) => <div className="warn-line" key={w}><AlertTriangle size={13} /> {w}</div>)}
    </Card>
  );
}

function PredictTab({ ctx, info }) {
  const { models, activeId } = ctx;
  const active = models.find((m) => m.id === activeId);
  const [vals, setVals] = useState({});
  const [other, setOther] = useState('');
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const v = {}; info.features.forEach((f) => { v[f.key] = String(f.default); }); setVals(v); setOut(null); setError(''); setOther('');
  }, [info]);
  const compatible = models.filter((m) => m.id !== activeId && m.nFeatures === active?.nFeatures && JSON.stringify(m.featureNames) === JSON.stringify(active?.featureNames));

  const go = async () => {
    setBusy(true); setError('');
    try {
      if (other) { const r = await api.predictCompare([activeId, Number(other)], vals); setOut(r.results); }
      else { const r = await api.predict(activeId, vals); setOut([r]); }
    } catch (e) { setError(errMsg(e)); setOut(null); } finally { setBusy(false); }
  };
  return (
    <div className="two-col">
      <Card>
        <h3>Input Features</h3>
        <p className="muted">Enter values to test a prediction. The result comes from the loaded model.</p>
        <div className="form-stack">
          {info.features.map((f) => (
            <label key={f.key}>{f.label}{f.min != null && <small className="range"> (data range {f.min} – {f.max})</small>}
              <input type="number" step="any" value={vals[f.key] ?? ''} onChange={(e) => setVals({ ...vals, [f.key]: e.target.value })} /></label>
          ))}
          {compatible.length > 0 && (
            <label>Compare with another model (optional)
              <select value={other} onChange={(e) => { setOther(e.target.value); setOut(null); }}><option value="">— none —</option>{compatible.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
          )}
          {error && <ErrorBox>{error}</ErrorBox>}
          <Button onClick={go} busy={busy} full>Predict</Button>
        </div>
      </Card>
      <div className={out?.length === 2 ? 'stack-2' : ''}>
        {!out && <Card className="prediction-card"><h3>Prediction Result</h3><div className="empty-result">Enter input features and select Predict to see the model's output.</div></Card>}
        {out?.map((r, i) => <ResultCard key={r.modelId} r={r} agree={out.length === 2 ? r.agree : null} tag={out.length === 2 ? (i === 0 ? 'Model A' : 'Model B') : null} />)}
      </div>
    </div>
  );
}

function EvaluateTab({ ctx }) {
  const { activeId } = ctx;
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { setData(null); setError(''); api.overview(activeId).then(setData).catch((e) => setError(errMsg(e))); }, [activeId]);
  if (error) return <ErrorBox>{error}</ErrorBox>;
  if (!data) return <Loading text="Evaluating…" />;
  const m = data.metrics;
  if (!m) return <Empty title="No evaluation data">{data.note}</Empty>;
  const dist = (m.classes || []).map((c) => ({ name: c, actual: m.actual_distribution[c], predicted: m.predicted_distribution[c] }));
  return (
    <Card>
      <h3>Evaluate Model</h3>
      <p className="muted">Measured on: {data.evalSource}</p>
      {m.task === 'Regression' ? (
        <div className="metric-grid four"><Metric label="R²" value={fmt(m.r2)} icon={Gauge} /><Metric label="MAE" value={fmt(m.mae)} icon={ShieldCheck} /><Metric label="RMSE" value={fmt(m.rmse)} icon={Activity} /><Metric label="Rows" value={m.n} icon={BarChart3} /></div>
      ) : (<>
        <div className="metric-grid">
          <Metric label="Accuracy" value={fmt(m.accuracy)} icon={Gauge} /><Metric label="Precision" value={fmt(m.precision)} icon={ShieldCheck} />
          <Metric label="Recall" value={fmt(m.recall)} icon={Activity} /><Metric label="F1 Score" value={fmt(m.f1)} icon={BarChart3} />
          <Metric label="ROC AUC" value={fmt(m.roc_auc)} sub={m.ece != null ? `calibration error ${fmt(m.ece)}` : undefined} icon={Gauge} />
        </div>
        <div className="cm-row"><Confusion m={m} title="Confusion matrix" />
          <div className="cm"><b>Actual vs predicted class counts</b>
            <div className="chart small"><ResponsiveContainer><BarChart data={dist}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Legend />
              <Bar dataKey="actual" name="Actual" fill="#c7a6ff" /><Bar dataKey="predicted" name="Predicted" fill="#792cf2" /></BarChart></ResponsiveContainer></div></div></div>
      </>)}
    </Card>
  );
}

function CompareTab({ ctx }) {
  const { models, activeId } = ctx;
  const other = models.find((m) => m.id !== activeId)?.id || '';
  // Put the older model in slot A so "B vs A" reads as new-vs-original.
  const [a, setA] = useState(other && other < activeId ? other : activeId);
  const [b, setB] = useState(other && other < activeId ? activeId : other);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const run = async () => {
    setBusy(true); setError('');
    try { setRes(await api.compare(Number(a), Number(b))); } catch (e) { setError(errMsg(e)); setRes(null); } finally { setBusy(false); }
  };
  useEffect(() => { if (a && b && a !== b) run(); }, []); // eslint-disable-line
  return (
    <>
      <Card>
        <h3>Compare Versions</h3>
        <p className="muted">Both models are scored on exactly the same evaluation rows.</p>
        <div className="picker-row">
          <label>Model A<select value={a} onChange={(e) => setA(Number(e.target.value))}>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
          <label>Model B<select value={b} onChange={(e) => setB(Number(e.target.value))}><option value="">Select…</option>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
          <Button onClick={run} busy={busy} disabled={!a || !b || a === b}>Compare</Button>
        </div>
      </Card>
      {error && <ErrorBox>{error}</ErrorBox>}
      {busy && !res && <Loading text="Scoring both models…" />}
      {res && <ComparisonView res={res} aTitle="Model A" bTitle="Model B" />}
    </>
  );
}

export function ModelLab({ ctx }) {
  const { activeId, models } = ctx;
  const [tab, setTab] = useState('Make Prediction');
  const [info, setInfo] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { if (!activeId) return; setInfo(null); setError(''); api.info(activeId).then(setInfo).catch((e) => setError(errMsg(e))); }, [activeId]);
  if (!activeId) return <Empty title="No model selected">Upload or select a model on the Get Started page.</Empty>;
  return (
    <>
      <Header title="Model Lab" desc="Test predictions, evaluate performance and compare versions." />
      <div className="tabs">{TABS.map((x) => <button className={tab === x ? 'selected' : ''} onClick={() => setTab(x)} key={x}>{x}</button>)}</div>
      {error && <ErrorBox>{error}</ErrorBox>}
      {tab === 'Make Prediction' && (info ? <PredictTab ctx={ctx} info={info} /> : !error && <Loading />)}
      {tab === 'Evaluate Model' && <EvaluateTab ctx={ctx} />}
      {tab === 'Compare Versions' && (models.length < 2 ? <Empty title="Need two models">Register a second model to compare.</Empty> : <CompareTab ctx={ctx} />)}
    </>
  );
}
