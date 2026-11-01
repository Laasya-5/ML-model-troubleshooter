import axios from 'axios';

export const API_URL = import.meta.env?.VITE_API_URL || 'http://127.0.0.1:5000/api';
const TOKEN_KEY = 'mlt_token';

export const getToken = () => localStorage.getItem(TOKEN_KEY) || '';
export const setToken = (t) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY));

const http = axios.create({ baseURL: API_URL, timeout: 120000 });
let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => (onUnauthorized = fn);

http.interceptors.request.use((cfg) => {
  const t = getToken();
  if (t) cfg.headers.Authorization = `Bearer ${t}`;
  return cfg;
});
http.interceptors.response.use(
  (r) => r,
  (e) => {
    if (e.response?.status === 401 && getToken()) onUnauthorized();
    return Promise.reject(e);
  }
);

/** Turn any axios failure into a readable message. */
export function errMsg(e) {
  if (e?.response?.data instanceof Blob) return 'The server rejected the request.';
  return e?.response?.data?.error || (e?.code === 'ERR_NETWORK' || !e?.response
    ? `Cannot reach the Python backend at ${API_URL}. Start it with "python app.py" in the backend folder.`
    : e.message || 'Something went wrong.');
}

export const api = {
  register: (b) => http.post('/auth/register', b).then((r) => r.data),
  login: (b) => http.post('/auth/login', b).then((r) => r.data),
  me: () => http.get('/auth/me').then((r) => r.data),
  models: () => http.get('/models').then((r) => r.data),
  upload: (fd) => http.post('/models/upload', fd).then((r) => r.data),
  attachDataset: (id, fd) => http.post(`/models/${id}/dataset`, fd).then((r) => r.data),
  deleteModel: (id) => http.delete(`/models/${id}`).then((r) => r.data),
  info: (id) => http.get(`/models/${id}/info`).then((r) => r.data),
  validate: (id) => http.post(`/models/${id}/validate`).then((r) => r.data),
  checks: (id) => http.post(`/models/${id}/checks`).then((r) => r.data),
  overview: (id) => http.get(`/models/${id}/overview`).then((r) => r.data),
  issues: (id) => http.get(`/models/${id}/issues`).then((r) => r.data),
  predict: (id, features) => http.post(`/models/${id}/predict`, { features }).then((r) => r.data),
  predictCompare: (modelIds, features) => http.post('/predict-compare', { modelIds, features }).then((r) => r.data),
  compare: (a, b) => http.post('/compare', { a, b }).then((r) => r.data),
  fix: (id, action) => http.post(`/models/${id}/fix`, { action }).then((r) => r.data),
  monitor: (id) => http.get(`/models/${id}/monitor`).then((r) => r.data),
  testTraffic: (id, body) => http.post(`/models/${id}/test-traffic`, body).then((r) => r.data),
  clearPredictions: (id) => http.delete(`/models/${id}/predictions`).then((r) => r.data),
  logs: () => http.get('/logs').then((r) => r.data),
  report: (body) => http.post('/report', body).then((r) => r.data),
  reports: () => http.get('/reports').then((r) => r.data),
  downloadReport: (id) => http.get(`/reports/${id}/download`, { responseType: 'blob' }).then((r) => r.data),
};

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
