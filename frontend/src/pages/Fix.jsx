import React, { useEffect, useState } from 'react';
import { CheckCircle2, AlertTriangle, XCircle, MinusCircle, ChevronRight } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { api, errMsg } from '../api';
import { Header, Card, Button, Loading, ErrorBox, Empty, fmt, signed, pct } from '../ui';

const CLS_KEYS = [['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1 Score']];
const REG_KEYS = [['r2', 'R²'], ['mae', 'MAE'], ['rmse', 'RMSE']];
const LOWER_BETTER = new Set(['mae', 'rmse']);

export const VERDICTS = {
  improved: { cls: 'ok', icon: CheckCircle2, title: 'Improvement Verified', text: 'Primary metrics improved on the same evaluation rows.' },
  mixed: { cls: 'mid', icon: AlertTriangle, title: 'Mixed Result', text: 'Some metrics improved while others got worse — review the trade-off.' },
  regressed: { cls: 'bad', icon: XCircle, title: 'Regression', text: 'The new version scores worse than the original.' },
  unchanged: { cls: 'flat', icon: MinusCircle, title: 'No Significant Change', text: 'Metrics are within ±0.002 of each other.' },
};

export function metricKeys(m) { return m?.task === 'Regression' ? REG_KEYS : CLS_KEYS; }

export function Confusion({ m, title }) {
  if (!m?.confusion) return null;
  const mx = Math.max(...m.confusion.flat(), 1);
  return (
    <div className="cm">
      <b>{title}</b>
      <table><thead><tr><th /><th colSpan={m.classes.length}>Predicted</th></tr><tr><th /><>{m.classes.map((c) => <th key={c}>{c}</th>)}</></tr></thead>
        <tbody>{m.confusion.map((row, i) => (
          <tr key={i}><th>Actual {m.classes[i]}</th>{row.map((v, j) => <td key={j} className={i === j ? 'diag' : 'off'} style={{ opacity: 0.35 + 0.65 * (v / mx) }}>{v}</td>)}</tr>
        ))}</tbody></table>
    </div>
  );
}

export function ComparisonView({ res, aTitle, bTitle, showConfusion = true }) {
  const keys = metricKeys(res.a.metrics);
  const v = VERDICTS[res.verdict];
  const chart = keys.map(([k, label]) => ({ name: label, a: res.a.metrics[k], b: res.b.metrics[k] }));
  const winner = (k) => {
    const d = res.b.metrics[k] - res.a.metrics[k];
    if (Math.abs(d) <= 0.002) return '';
    return (LOWER_BETTER.has(k) ? d < 0 : d > 0) ? 'b' : 'a';
  };
  return (
    <>
      <div className="before-after">
        <Card className="ba before">
          <h3>{aTitle}</h3><p className="muted">{res.a.model.name}</p>
          {keys.map(([k, l]) => <div className="compare-metric" key={k}><span>{l}</span><b className={winner(k) === 'a' ? 'green-text' : ''}>{fmt(res.a.metrics[k])}</b></div>)}
        </Card>
        <div className="ba-arrow">→</div>
        <Card className="ba after">
          <h3>{bTitle}</h3><p className="muted">{res.b.model.name}</p>
          {keys.map(([k, l]) => <div className="compare-metric" key={k}><span>{l}</span><b className={winner(k) === 'b' ? 'green-text' : ''}>{fmt(res.b.metrics[k])} <small className={`delta ${(LOWER_BETTER.has(k) ? -res.deltas[k] : res.deltas[k]) >= 0 ? 'up' : 'down'}`}>{signed(res.deltas[k])}</small></b></div>)}
        </Card>
      </div>
      <div className={`verdict ${v.cls}`}>
        <v.icon size={26} /><div><b>{v.title}</b><small>{v.text}</small>
          <small>Scored on {res.n.toLocaleString()} rows · {res.source}.
            {res.pValue != null && ` Models disagree on ${(100 - res.agreement * 100).toFixed(1)}% of rows; of those, A alone is right on ${res.onlyACorrect} and B alone on ${res.onlyBCorrect} (McNemar p ${res.pValue < 0.0001 ? '< 0.0001' : '= ' + res.pValue}${res.significant ? ', statistically significant' : ', not significant'}).`}</small></div>
      </div>
      <Card>
        <h3>Metric Comparison</h3>
        <div className="chart"><ResponsiveContainer><BarChart data={chart}>
          <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis domain={[0, 'auto']} /><Tooltip /><Legend />
          <Bar dataKey="a" name={aTitle} fill="#c7a6ff" /><Bar dataKey="b" name={bTitle} fill="#792cf2" />
        </BarChart></ResponsiveContainer></div>
        {showConfusion && res.a.metrics.confusion && <div className="cm-row"><Confusion m={res.a.metrics} title={aTitle} /><Confusion m={res.b.metrics} title={bTitle} /></div>}
      </Card>
    </>
  );
}

export function FixVerification({ ctx }) {
  const { models, activeId, fixResult, setFixResult, notify, go } = ctx;
  const active = models.find((m) => m.id === activeId);
  const parent = active?.parentId ? active.parentId : null;
  const [a, setA] = useState(fixResult ? fixResult.comparison.a.model.id : parent || activeId);
  const [b, setB] = useState(fixResult ? fixResult.comparison.b.model.id : parent ? activeId : '');
  const [res, setRes] = useState(fixResult ? fixResult.comparison : null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const run = async (x = a, y = b) => {
    if (!x || !y || x === y) { setError('Pick two different models.'); return; }
    setBusy(true); setError('');
    try { const r = await api.compare(Number(x), Number(y)); setRes(r); setFixResult(null); notify('Verification complete'); }
    catch (e) { setError(errMsg(e)); setRes(null); } finally { setBusy(false); }
  };
  useEffect(() => { if (!fixResult && a && b && a !== b) run(a, b); }, []); // eslint-disable-line

  if (!models.length) return <Empty title="No models">Upload a model first.</Empty>;
  return (
    <>
      <Header title="Fix Verification" desc="Re-score the original and the fixed model on the same rows and compare the results." />
      {fixResult && <p className="demo-note">Fix applied: {fixResult.note}</p>}
      <Card>
        <div className="picker-row">
          <label>Before (original)<select value={a} onChange={(e) => setA(Number(e.target.value))}>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
          <label>After (fixed)<select value={b} onChange={(e) => setB(Number(e.target.value))}><option value="">Select…</option>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
          <Button onClick={() => run()} busy={busy}>Run Verification</Button>
        </div>
        <small className="muted">Tip: apply a fix from Troubleshooting to create a fixed version automatically, or upload your own new version and compare it here.</small>
      </Card>
      {error && <ErrorBox>{error}</ErrorBox>}
      {busy && !res && <Loading text="Scoring both models on the same rows…" />}
      {res && <ComparisonView res={res} aTitle="Before Fix" bTitle="After Fix" />}
      {!res && !busy && !error && <Empty title="Nothing to compare yet">Choose a before and an after model, then run verification.{' '}<Button secondary onClick={() => go('Troubleshooting')}>Open Troubleshooting <ChevronRight size={14} /></Button></Empty>}
    </>
  );
}
