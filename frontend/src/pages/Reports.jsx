import React, { useEffect, useState } from 'react';
import { FileText, Download } from 'lucide-react';
import { api, errMsg, saveBlob } from '../api';
import { Header, Card, Button, ErrorBox, Empty } from '../ui';

const SECTIONS = ['Model Summary', 'Validation & Check Results', 'Issues & Recommendations', 'Verification Results', 'Monitoring & Logs'];
const TYPES = {
  'Full Diagnostic Report': SECTIONS,
  'Model Summary': ['Model Summary'],
  'Performance Report': ['Model Summary', 'Validation & Check Results'],
  'Troubleshooting Report': ['Model Summary', 'Issues & Recommendations', 'Verification Results'],
};

export function Reports({ ctx }) {
  const { activeId, models, notify } = ctx;
  const [type, setType] = useState('Full Diagnostic Report');
  const [sections, setSections] = useState(SECTIONS);
  const [recent, setRecent] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const load = () => api.reports().then(setRecent).catch(() => {});
  useEffect(() => { load(); }, []);

  const generate = async () => {
    setBusy(true); setError('');
    try {
      const r = await api.report({ modelId: activeId, type, sections });
      saveBlob(await api.downloadReport(r.id), r.filename);
      notify('Report generated'); load();
    } catch (e) { setError(errMsg(e)); } finally { setBusy(false); }
  };
  const dl = async (r) => { try { saveBlob(await api.downloadReport(r.id), r.filename); } catch (e) { notify(errMsg(e)); } };
  const active = models.find((m) => m.id === activeId);
  if (!activeId) return <Empty title="No model selected">Select a model first.</Empty>;
  return (
    <>
      <Header title="Reports" desc="Generate and export a complete diagnostic report (PDF) built from live results." />
      <div className="two-col">
        <Card>
          <h3>Report Configuration</h3>
          <div className="form-stack">
            <label>Model<input value={active?.name || ''} disabled /></label>
            <label>Report Type<select value={type} onChange={(e) => { setType(e.target.value); setSections(TYPES[e.target.value]); }}>{Object.keys(TYPES).map((t) => <option key={t}>{t}</option>)}</select></label>
            <b>Include Sections</b>
            {SECTIONS.map((s) => (
              <label className="check-label" key={s}><input type="checkbox" checked={sections.includes(s)} onChange={(e) => setSections(e.target.checked ? SECTIONS.filter((x) => x === s || sections.includes(x)) : sections.filter((x) => x !== s))} />{s}</label>
            ))}
            {error && <ErrorBox>{error}</ErrorBox>}
            <Button onClick={generate} busy={busy} full disabled={!sections.length}>Generate Report <Download size={16} /></Button>
          </div>
        </Card>
        <Card>
          <h3>Recent Reports</h3>
          {recent.map((r) => (
            <div className="report-row" key={r.id}>
              <div className="report-icon"><FileText /></div>
              <div><b>{r.filename}</b><small>{r.ts.slice(0, 16)} · {r.size_bytes > 1048576 ? (r.size_bytes / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(r.size_bytes / 1024)) + ' KB'} · {r.model_name}</small></div>
              <button className="icon-btn" onClick={() => dl(r)} title="Download"><Download size={17} /></button>
            </div>
          ))}
          {!recent.length && <p className="empty-result">No reports generated yet.</p>}
        </Card>
      </div>
    </>
  );
}
