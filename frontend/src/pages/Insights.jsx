import React, { useCallback, useEffect, useState } from 'react';
import {
  Gauge, ShieldCheck, Activity, BarChart3, HeartPulse, AlertTriangle, ChevronRight, X, Wrench, Trash2,
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, PieChart, Pie, Cell,
} from 'recharts';
import { api, errMsg } from '../api';
import { Header, Card, Metric, Button, Loading, ErrorBox, Empty, fmt, pct } from '../ui';

const noModel = <Empty title="No model selected">Upload or select a model on the Get Started page.</Empty>;

function useLoad(fn, deps) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const reload = useCallback(async () => {
    setBusy(true); setError('');
    try { setData(await fn()); } catch (e) { setError(errMsg(e)); } finally { setBusy(false); }
  }, deps); // eslint-disable-line
  useEffect(() => { setData(null); reload(); }, [reload]);
  return { data, error, busy, reload };
}

/* ---------------------------------------------------------------- Overview */
export function Overview({ ctx }) {
  const { activeId, go } = ctx;
  const { data, error, busy } = useLoad(() => (activeId ? api.overview(activeId) : Promise.resolve(null)), [activeId]);
  if (!activeId) return noModel;
  const m = data?.metrics;
  const reg = m?.task === 'Regression';
  const tone = { Healthy: 'green', 'Needs Attention': 'amber', Unhealthy: 'red' }[data?.health];
  const trend = (m?.batches || []).map((v, i) => ({ t: `Batch ${i + 1}`, v }));
  const dist = Object.entries(data?.distribution || {}).map(([name, value]) => ({ name, value }));
  const total = dist.reduce((a, b) => a + b.value, 0);
  const COLORS = ['#792cf2', '#c7a6ff', '#4fb3a1', '#f2a65a'];
  return (
    <>
      <Header title="Model Overview" desc="A quick summary of your model's health and performance." />
      {error && <ErrorBox>{error}</ErrorBox>}
      {busy && !data && <Loading text="Evaluating the model on its holdout data…" />}
      {data && !m && <Empty title="No performance numbers yet">{data.note} <Button secondary onClick={() => go('Get Started')}>Upload a model with a CSV</Button></Empty>}
      {m && (<>
        <div className="metric-grid">
          <Metric label="Model Health" value={data.health} tone={tone} sub={data.issueCount ? `${data.issueCount} issue(s) found` : 'No issues found'} icon={HeartPulse} />
          {reg ? (<>
            <Metric label="R²" value={fmt(m.r2)} icon={Gauge} /><Metric label="MAE" value={fmt(m.mae)} icon={ShieldCheck} />
            <Metric label="RMSE" value={fmt(m.rmse)} icon={Activity} /><Metric label="Rows scored" value={m.n} icon={BarChart3} />
          </>) : (<>
            <Metric label="Accuracy" value={fmt(m.accuracy)} icon={Gauge} /><Metric label="Precision" value={fmt(m.precision)} icon={ShieldCheck} />
            <Metric label="Recall" value={fmt(m.recall)} icon={Activity} /><Metric label="F1 Score" value={fmt(m.f1)} icon={BarChart3} />
          </>)}
        </div>
        <p className="demo-note">Measured on: {data.evalSource}{m.roc_auc != null ? ` · ROC AUC ${fmt(m.roc_auc)}` : ''}</p>
        <div className="two-col">
          <Card>
            <h3>Performance Trend</h3>
            <p className="muted">{reg ? 'R²' : 'Accuracy'} across {trend.length} sequential batches of the evaluation data</p>
            <div className="chart"><ResponsiveContainer><LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="t" /><YAxis domain={['auto', 'auto']} /><Tooltip />
              <Line type="monotone" dataKey="v" name={reg ? 'R²' : 'Accuracy'} stroke="#8238f5" strokeWidth={3} dot={{ r: 4 }} />
            </LineChart></ResponsiveContainer></div>
          </Card>
          <Card>
            <h3>Prediction Distribution</h3>
            <p className="muted">{total.toLocaleString()} predictions · from {data.distributionSource}</p>
            {dist.length ? (<>
              <div className="chart donut">
                <ResponsiveContainer><PieChart><Pie data={dist} dataKey="value" nameKey="name" innerRadius={65} outerRadius={95} paddingAngle={3}>
                  {dist.map((x, i) => <Cell key={x.name} fill={COLORS[i % COLORS.length]} />)}
                </Pie><Tooltip /></PieChart></ResponsiveContainer>
                <div className="donut-center"><b>{total.toLocaleString()}</b><small>Predictions</small></div>
              </div>
              <div className="legend">{dist.map((d, i) => <span key={d.name}><i style={{ color: COLORS[i % COLORS.length] }}>●</i> {d.name} ({pct(d.value / total)})</span>)}</div>
            </>) : <p className="muted">No class distribution for a regression model.</p>}
          </Card>
        </div>
      </>)}
    </>
  );
}

/* ---------------------------------------------------------------- Troubleshooting */
const SEV = { High: 'red', Medium: 'orange', Low: 'blue' };

export function Troubleshooting({ ctx }) {
  const { activeId, go, notify, refreshModels, setFixResult } = ctx;
  const { data, error, busy } = useLoad(() => (activeId ? api.issues(activeId) : Promise.resolve(null)), [activeId]);
  const [sel, setSel] = useState(null);
  const [fixing, setFixing] = useState(false);
  const [fixErr, setFixErr] = useState('');
  if (!activeId) return noModel;

  const apply = async () => {
    setFixing(true); setFixErr('');
    try {
      const r = await api.fix(activeId, sel.fixAction);
      setFixResult(r);
      await refreshModels(r.newModel.id, { keepPage: true });
      setSel(null); notify('Fix applied — verifying'); go('Fix Verification');
    } catch (e) { setFixErr(errMsg(e)); } finally { setFixing(false); }
  };

  return (
    <>
      <Header title="Troubleshooting" desc="Identify issues, understand causes and get actionable fixes." />
      {error && <ErrorBox>{error}</ErrorBox>}
      {busy && !data && <Loading text="Diagnosing the model…" />}
      {data && data.note && <Empty title="Nothing to diagnose yet">{data.note}</Empty>}
      {data && !data.note && data.issues.length === 0 && (
        <Card className="verified-box"><ShieldCheck size={26} /><b>No issues detected</b><small>Every diagnostic passed on the evaluation data{data.metrics?.accuracy ? ` (accuracy ${fmt(data.metrics.accuracy)}, F1 ${fmt(data.metrics.f1)})` : ''}.</small></Card>
      )}
      {data && data.issues.length > 0 && (
        <div className="issue-list">
          {data.issues.map((i) => (
            <Card className="issue" key={i.id}>
              <div className={`issue-icon ${SEV[i.severity]}`}><AlertTriangle /></div>
              <div className="issue-copy"><h3>{i.title}</h3><p className="muted">{i.description}</p></div>
              <span className={`severity ${SEV[i.severity]}`}>{i.severity}</span>
              <Button secondary onClick={() => { setSel(i); setFixErr(''); }}>View Fix <ChevronRight size={15} /></Button>
            </Card>
          ))}
        </div>
      )}
      {sel && (
        <div className="modal-backdrop" onClick={() => setSel(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-x" onClick={() => setSel(null)}><X /></button>
            <span className="pill">RECOMMENDED ACTION</span>
            <h2>{sel.title}</h2>
            <p>{sel.description}</p>
            <h3>Suggested steps</h3>
            <ol>{sel.steps.map((s) => <li key={s}>{s}</li>)}</ol>
            {fixErr && <ErrorBox>{fixErr}</ErrorBox>}
            {sel.fixAction ? (
              <><p className="demo-note">This will retrain a new version on the built-in training data, register it, and score both versions on the same holdout so you can see whether it really helped.</p>
                <Button onClick={apply} busy={fixing}><Wrench size={14} /> Apply fix &amp; verify</Button></>
            ) : (
              <><p className="demo-note">This fix can't be applied automatically. Apply it in your own pipeline, upload the new version, then compare both in Fix Verification.</p>
                <Button secondary onClick={() => { setSel(null); go('Fix Verification'); }}>Go to Fix Verification</Button></>
            )}
          </div>
        </div>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- Monitor */
export function Monitor({ ctx }) {
  const { activeId, notify } = ctx;
  const mon = useLoad(() => (activeId ? api.monitor(activeId) : Promise.resolve(null)), [activeId]);
  const logs = useLoad(() => api.logs(), []);
  const [sending, setSending] = useState('');
  if (!activeId) return noModel;
  const d = mon.data;

  const send = async (drift) => {
    setSending(drift ? 'drift' : 'ok');
    try {
      const r = await api.testTraffic(activeId, { n: drift ? 60 : 40, drift });
      notify(`Sent ${r.sent} ${drift ? 'drifted ' : ''}test requests`);
      await Promise.all([mon.reload(), logs.reload()]);
    } catch (e) { notify(errMsg(e)); } finally { setSending(''); }
  };
  const clear = async () => { await api.clearPredictions(activeId); notify('Prediction history cleared'); mon.reload(); };
  const dr = d?.drift;
  const driftTone = dr?.status === 'Stable' ? 'green' : dr?.status === 'Drift detected' ? 'red' : undefined;

  return (
    <>
      <Header title="Monitor & Logs" desc="Track model performance, prediction activity and logs — measured from real requests." />
      {mon.error && <ErrorBox>{mon.error}</ErrorBox>}
      {!d && !mon.error && <Loading />}
      {d && (<>
        <div className="metric-grid four">
          <Metric label="Predictions" value={d.total.toLocaleString()} sub={d.test_traffic ? `${d.test_traffic} from test traffic` : 'all manual'} icon={Activity} />
          <Metric label="Avg. Latency" value={d.avg_latency != null ? `${d.avg_latency} ms` : '—'} sub={d.p95_latency != null ? `p95 ${d.p95_latency} ms` : 'no requests yet'} icon={Gauge} />
          <Metric label="Error Rate" value={d.total ? pct(d.error_rate) : '—'} sub={d.total ? `${d.total - d.ok} of ${d.total} failed` : 'no requests yet'} icon={AlertTriangle} />
          <Metric label="Drift Status" value={dr.status} tone={driftTone} sub={dr.note || `${dr.n} recent inputs vs reference`} icon={ShieldCheck} />
        </div>
        <div className="toolbar">
          <Button secondary onClick={() => send(false)} busy={sending === 'ok'}>Send 40 test requests</Button>
          <Button secondary onClick={() => send(true)} busy={sending === 'drift'}>Send 60 drifted test requests</Button>
          <button className="link-btn" onClick={clear}><Trash2 size={13} /> Clear history</button>
          <small className="muted">Test requests are real rows from the evaluation data run through the model one at a time (drifted = every feature shifted +1 std).</small>
        </div>
        <div className="two-col">
          <Card>
            <h3>Prediction Latency (ms)</h3><p className="muted">Last {d.series.length} successful requests</p>
            {d.series.length ? <div className="chart"><ResponsiveContainer><LineChart data={d.series}>
              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="t" /><YAxis /><Tooltip /><Line dataKey="v" name="ms" stroke="#8238f5" strokeWidth={2} dot={false} />
            </LineChart></ResponsiveContainer></div> : <p className="empty-result">No requests yet. Make a prediction in Model Lab or send test requests.</p>}
          </Card>
          <Card>
            <h3>Recent Logs</h3>
            {(logs.data || []).slice(0, 8).map((l, i) => (
              <div className="log-row" key={i}><span className={l.status === 'error' ? 'log-error' : 'log-ok'}>{l.status === 'error' ? '!' : '✓'}</span><span>{l.message}</span><small>{l.time}</small></div>
            ))}
            {logs.data && !logs.data.length && <p className="empty-result">No log entries yet.</p>}
          </Card>
        </div>
        {dr.features.length > 0 && (
          <Card>
            <h3>Drift by feature</h3><p className="muted">Two-sample KS test of recent inputs vs the reference data (flagged when p &lt; 0.01 and mean shift &gt; 0.3 std).</p>
            <div className="table-wrap"><table className="tbl"><thead><tr><th>Feature</th><th>KS statistic</th><th>p-value</th><th>Mean shift (std)</th><th>Status</th></tr></thead>
              <tbody>{dr.features.map((f) => (
                <tr key={f.feature}><td>{f.feature}</td><td>{fmt(f.ks)}</td><td>{f.p_value < 0.0001 ? '<0.0001' : fmt(f.p_value, 4)}</td><td>{fmt(f.mean_shift_sd, 2)}</td>
                  <td><span className={`status ${f.drifted ? 'bad' : 'good'}`}>{f.drifted ? 'Drifted' : 'Stable'}</span></td></tr>))}</tbody></table></div>
          </Card>
        )}
      </>)}
    </>
  );
}
