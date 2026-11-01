import React, { useEffect, useRef, useState } from 'react';
import { CloudUpload, CheckCircle2, ChevronRight, AlertTriangle, XCircle, Trash2, FileSpreadsheet } from 'lucide-react';
import { api, errMsg } from '../api';
import { Header, Card, Button, Loading, ErrorBox, Empty, Steps, statusClass } from '../ui';

/* ---------------------------------------------------------------- Get Started */
export function GetStarted({ ctx }) {
  const { models, activeId, setActiveId, go, notify, refreshModels } = ctx;
  const [file, setFile] = useState(null);
  const [csv, setCsv] = useState(null);
  const [name, setName] = useState('');
  const [framework, setFramework] = useState('Scikit-learn');
  const [taskType, setTaskType] = useState('Classification');
  const [target, setTarget] = useState('');
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState('');
  const input = useRef();

  const pick = (f) => { if (f) { setFile(f); setError(''); if (!name) setName(f.name.replace(/\.[^.]+$/, '')); } };

  const submit = async (e) => {
    e.preventDefault();
    if (!file) { setError('Choose a model file first.'); return; }
    const fd = new FormData();
    fd.append('file', file); fd.append('modelName', name); fd.append('framework', framework); fd.append('taskType', taskType);
    if (csv) { fd.append('dataset', csv); if (target) fd.append('target', target); }
    setBusy(true); setError('');
    try {
      const r = await api.upload(fd);
      await refreshModels(r.model.id);
      notify('Model loaded successfully');
      go('Validate Model');
    } catch (err) { setError(errMsg(err)); } finally { setBusy(false); }
  };

  const remove = async (m) => {
    try { await api.deleteModel(m.id); await refreshModels(); notify('Model deleted'); } catch (err) { notify(errMsg(err)); }
  };

  return (
    <>
      <Header title="Get Started" desc="Upload and register your machine learning model — or pick one of the built-in demo models." />
      <Steps current={0} />
      <div className="two-col">
        <Card>
          <h3>Upload Model File</h3>
          <p className="muted">Drag and drop your model file here or browse to upload.</p>
          <div className={`dropzone ${drag ? 'drag' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
            onClick={() => input.current.click()}>
            <CloudUpload size={42} />
            <b>{file ? file.name : 'Drag & drop your model file here'}</b>
            <span>or click to browse</span>
            <input ref={input} type="file" accept=".pkl,.joblib,.onnx,.h5" onChange={(e) => pick(e.target.files[0])} />
            <span className="btn">Choose File</span>
            <span>Supported: .pkl, .joblib (scikit-learn) · .onnx</span>
          </div>
          <p className="demo-note">⚠ .pkl / .joblib files are Python pickles and run code when loaded. Only upload models you trust.</p>
        </Card>
        <Card>
          <h3>Model Configuration</h3>
          <form onSubmit={submit} className="form-stack">
            <label>Model Name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Risk Classification Model" /></label>
            <label>Framework<select value={framework} onChange={(e) => setFramework(e.target.value)}><option>Scikit-learn</option><option>ONNX</option><option>XGBoost</option><option>LightGBM</option><option>TensorFlow</option><option>PyTorch</option></select></label>
            <label>Task Type<select value={taskType} onChange={(e) => setTaskType(e.target.value)}><option>Classification</option><option>Regression</option></select></label>
            <label>Evaluation data (optional CSV with a label column)
              <div className="file-row">
                <FileSpreadsheet size={16} />
                <input type="file" accept=".csv" onChange={(e) => setCsv(e.target.files[0] || null)} />
              </div>
            </label>
            {csv && <label>Label column name (blank = "target"/"label"/"y" or the last column)<input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="target" /></label>}
            <small className="muted">Without a labelled CSV, accuracy / precision / recall can't be measured for your own model, so the app won't show them.</small>
            {error && <ErrorBox>{error}</ErrorBox>}
            <Button type="submit" full busy={busy}>Register Model <ChevronRight size={16} /></Button>
          </form>
        </Card>
      </div>
      <Card>
        <h3>Your models</h3>
        <p className="muted">Pick the model every other page works on.</p>
        {models.map((m) => (
          <div className={`model-row ${m.id === activeId ? 'active' : ''}`} key={m.id}>
            <div><b>{m.name}</b><small>{m.framework} · {m.taskType} · {m.featureText} · {m.size} MB{m.isDemo ? ' · built-in demo' : ''}{m.parentId ? ' · fixed version' : ''}</small></div>
            {m.id === activeId ? <span className="status good">Active</span> : <Button secondary onClick={() => { setActiveId(m.id); notify(`Using ${m.name}`); }}>Use</Button>}
            {!m.isDemo && !m.parentId && <button className="icon-btn" title="Delete" onClick={() => remove(m)}><Trash2 size={15} /></button>}
          </div>
        ))}
      </Card>
    </>
  );
}

/* ---------------------------------------------------------------- Validate */
export function Validate({ ctx }) {
  const { activeId, go } = ctx;
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!activeId) return; setData(null); setError('');
    api.validate(activeId).then(setData).catch((e) => setError(errMsg(e)));
  }, [activeId]);
  if (!activeId) return <Empty title="No model selected">Upload or select a model on the Get Started page.</Empty>;
  return (
    <>
      <Header title="Validate Model" desc="Check model file, compatibility and required information." />
      <Steps current={1} />
      {error && <ErrorBox>{error}</ErrorBox>}
      {!data && !error && <Loading text="Loading the model and inspecting it…" />}
      {data && (
        <Card>
          <div className="card-title-row">
            <div><h3>Validation Results</h3><p className="muted">Read directly from the loaded model object.</p></div>
            <span className={`status ${statusClass(data.status)}`}>{data.status}</span>
          </div>
          {data.items.map((it) => (
            <div className="validation-row" key={it.label}>
              {it.ok ? <CheckCircle2 size={19} /> : <AlertTriangle size={19} className="warn-ico" />}
              <b>{it.label}</b><span>{it.value}</span>
              <strong className={it.ok ? '' : 'warn-ico'}>{it.ok ? '✓' : '!'}</strong>
            </div>
          ))}
          <div className="actions">
            <Button secondary onClick={() => go('Get Started')}>← Back</Button>
            <Button onClick={() => go('Run Checks')} disabled={data.status === 'Failed'}>Continue to Run Checks <ChevronRight size={16} /></Button>
          </div>
        </Card>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- Run Checks */
export function CheckList({ checks }) {
  return (
    <div className="check-grid">
      {checks.map((c) => (
        <Card className="check-card" key={c.name}>
          <div className={`check-icon ${statusClass(c.status)}`}>{c.status === 'Passed' ? <CheckCircle2 /> : c.status === 'Warning' ? <AlertTriangle /> : <XCircle />}</div>
          <div><h3>{c.name}</h3><p className="muted">{c.description}</p><p className="detail">{c.detail}</p></div>
          <span className={`status ${statusClass(c.status)}`}>{c.status}</span>
        </Card>
      ))}
    </div>
  );
}

export function RunChecks({ ctx }) {
  const { activeId, go, notify } = ctx;
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const run = async () => {
    setBusy(true); setError('');
    try { const r = await api.checks(activeId); setData(r); } catch (e) { setError(errMsg(e)); } finally { setBusy(false); }
  };
  useEffect(() => { if (activeId) { setData(null); run(); } }, [activeId]); // eslint-disable-line
  if (!activeId) return <Empty title="No model selected">Upload or select a model on the Get Started page.</Empty>;
  return (
    <>
      <Header title="Run Model Checks" desc="Execute runtime, inference and output checks on the real model." />
      <Steps current={2} />
      {error && <ErrorBox>{error}</ErrorBox>}
      {busy && !data && <Loading text="Running checks against the model…" />}
      {data && (<>
        <p className="summary-line">Overall: <span className={`status ${statusClass(data.status)}`}>{data.status}</span> · {data.checks.filter((c) => c.status === 'Passed').length} of {data.checks.length} checks passed</p>
        <CheckList checks={data.checks} />
      </>)}
      <div className="actions">
        <Button secondary onClick={() => go('Validate Model')}>← Back</Button>
        <Button secondary onClick={run} busy={busy}>Run Checks Again</Button>
        <Button onClick={() => { notify('Checks complete'); go('Model Overview'); }}>View Results <ChevronRight size={16} /></Button>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- Verification (full re-verify) */
export function Verification({ ctx }) {
  const { activeId, go } = ctx;
  const [val, setVal] = useState(null);
  const [chk, setChk] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const run = async () => {
    setBusy(true); setError('');
    try { const [v, c] = await Promise.all([api.validate(activeId), api.checks(activeId)]); setVal(v); setChk(c); }
    catch (e) { setError(errMsg(e)); } finally { setBusy(false); }
  };
  useEffect(() => { if (activeId) { setVal(null); setChk(null); run(); } }, [activeId]); // eslint-disable-line
  if (!activeId) return <Empty title="No model selected">Select a model first.</Empty>;
  const all = val && chk ? [
    ...val.items.map((i) => ({ name: i.label, detail: i.value, status: i.ok ? 'Passed' : 'Warning', group: 'Validation' })),
    ...chk.checks.map((c) => ({ name: c.name, detail: c.detail, status: c.status, group: 'Check' })),
  ] : [];
  const bad = all.filter((a) => a.status !== 'Passed').length;
  return (
    <>
      <Header title="Verification" desc="Re-run every validation and runtime check on the active model and confirm it is still sound." />
      {error && <ErrorBox>{error}</ErrorBox>}
      {busy && !val && <Loading text="Re-verifying…" />}
      {val && chk && (
        <Card>
          <div className="card-title-row">
            <div><h3>Verification Results</h3><p className="muted">{all.length} items verified just now.</p></div>
            <span className={`status ${bad === 0 ? 'good' : 'warn'}`}>{bad === 0 ? 'All passed' : `${bad} need attention`}</span>
          </div>
          {all.map((a) => (
            <div className="validation-row" key={a.group + a.name}>
              {a.status === 'Passed' ? <CheckCircle2 size={19} /> : <AlertTriangle size={19} className="warn-ico" />}
              <b>{a.name}</b><span>{a.detail}</span><strong className={a.status === 'Passed' ? '' : 'warn-ico'}>{a.status === 'Passed' ? '✓' : '!'}</strong>
            </div>
          ))}
          <div className="actions"><Button secondary onClick={run} busy={busy}>Re-run verification</Button><Button onClick={() => go('Fix Verification')}>Compare before / after a fix <ChevronRight size={16} /></Button></div>
        </Card>
      )}
    </>
  );
}
