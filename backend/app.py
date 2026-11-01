"""ML Model Troubleshooter API.

Real behaviour:
  * uploaded .pkl/.joblib (scikit-learn) and .onnx models are actually loaded and executed
  * validation, checks, metrics, predictions, comparison, drift and latency are measured, never hard-coded
  * two demo models are trained on a synthetic dataset the first time the server starts

SECURITY: .pkl/.joblib files are Python pickles - loading one runs code. Only upload models you trust,
and run this server locally / in a sandbox.
"""
import io, os, random, secrets, time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import joblib, numpy as np, pandas as pd
from flask import Flask, g, jsonify, request, send_file
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sklearn.base import clone
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import db, demo_data, ml_core
from ml_core import ModelLoadError
from report_pdf import build_pdf

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"; UPLOADS.mkdir(exist_ok=True)
MODELS_DIR = db.DATA / "models"; MODELS_DIR.mkdir(exist_ok=True)
REPORTS = db.DATA / "reports"; REPORTS.mkdir(exist_ok=True)
ALLOWED = {".pkl", ".joblib", ".onnx", ".h5"}


class Provider(DefaultJSONProvider):
    @staticmethod
    def default(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        return DefaultJSONProvider.default(o)


app = Flask(__name__)
app.json = Provider(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}}, expose_headers=["Content-Disposition"])

secret_file = db.DATA / "secret.key"
if not secret_file.exists():
    secret_file.write_text(secrets.token_hex(32))
signer = URLSafeTimedSerializer(secret_file.read_text())
db.init()

_models = {}   # id -> Wrapped
_eval = {}     # id -> eval data


# ------------------------------------------------------------------ helpers
def err(msg, code=400):
    return jsonify({"error": msg}), code


def auth(fn):
    @wraps(fn)
    def w(*a, **k):
        h = request.headers.get("Authorization", "")
        tok = h[7:] if h.startswith("Bearer ") else request.args.get("token", "")
        try:
            uid = signer.loads(tok, max_age=60 * 60 * 24 * 14)["uid"]
        except (BadSignature, SignatureExpired, KeyError):
            return err("Please sign in again.", 401)
        u = db.q("SELECT id,name,email FROM users WHERE id=?", (uid,), one=True)
        if not u:
            return err("Account not found.", 401)
        g.user = u
        return fn(*a, **k)
    return w


def get_row(mid):
    r = db.q("SELECT * FROM models WHERE id=?", (mid,), one=True)
    if not r or r["owner"] not in (0, g.user["id"]):
        return None
    return r


def get_wrapped(row):
    if row["id"] not in _models:
        ext = Path(row["path"]).suffix
        _models[row["id"]] = ml_core.load_model(row["path"], ext)
    return _models[row["id"]]


def get_eval(row, w):
    if row["id"] not in _eval:
        _eval[row["id"]] = ml_core.eval_data(row, w)
    return _eval[row["id"]]


def feature_names(row, w):
    return w.feature_names or [f"f{i}" for i in range(w.n_features or 0)]


def feature_specs(row, w, ev):
    names = feature_names(row, w)
    specs = []
    for i, n in enumerate(names):
        s = {"key": n, "label": demo_data.FEATURE_LABELS.get(n, n if w.feature_names else f"Feature {i+1}"),
             "default": 0, "min": None, "max": None}
        if ev is not None:
            col = ev["X"][:, i]
            s.update(default=round(float(np.median(col)), 2), min=round(float(col.min()), 2), max=round(float(col.max()), 2))
        specs.append(s)
    return specs


def class_label(row, c):
    lab = db.jd(row["class_labels"], {}) or {}
    return lab.get(str(c), str(c))


def model_dict(row):
    names = db.jd(row["feature_names"], None)
    return {"id": row["id"], "name": row["name"], "filename": row["filename"], "size": row["size_mb"],
            "framework": row["framework_detected"], "frameworkDeclared": row["framework_declared"],
            "taskType": row["task_type"], "nFeatures": row["n_features"], "featureNames": names,
            "featureText": (", ".join(names) if names else f"{row['n_features']} unnamed features"),
            "isDemo": bool(row["is_demo"]), "parentId": row["parent_id"], "fixNote": row["fix_note"],
            "hasDataset": bool(row["dataset_path"]), "canFix": bool(row["trained_on_demo"]), "created": row["created"]}


def train_acc(row, w):
    if not row["trained_on_demo"] or w.kind != "sklearn":
        return None
    Xtr, _, ytr, _ = demo_data.split()
    return float((w.predict(Xtr.values) == ytr).mean())


def record(row, values, out, ms, ok, source):
    db.x("INSERT INTO predictions(model_id,ts,inputs,output,latency_ms,ok,source) VALUES(?,?,?,?,?,?,?)",
         (row["id"], db.now(), db.jl(values), db.jl(out), ms, int(ok), source))


def parse_values(row, w, raw):
    names = feature_names(row, w)
    if not isinstance(raw, dict):
        raise ValueError("Send the feature values as an object.")
    miss = [n for n in names if n not in raw or raw[n] in (None, "")]
    if miss:
        raise ValueError(f"Missing required feature(s): {', '.join(miss)}")
    vals = []
    for n in names:
        try:
            v = float(raw[n])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{n}' must be a number.")
        if not np.isfinite(v):
            raise ValueError(f"Feature '{n}' must be a finite number.")
        vals.append(v)
    return names, vals


def run_prediction(row, w, ev, raw, source="manual"):
    try:
        names, vals = parse_values(row, w, raw)
    except ValueError as e:
        record(row, raw if isinstance(raw, dict) else {}, {"error": str(e)}, 0, False, source)
        raise
    X = np.array([vals])
    warnings = []
    if ev is not None:
        for i, n in enumerate(names):
            lo, hi = float(ev["X"][:, i].min()), float(ev["X"][:, i].max())
            if vals[i] < lo or vals[i] > hi:
                warnings.append(f"{n}={vals[i]:g} is outside the range seen in the reference data ({lo:g} to {hi:g}); "
                                "the prediction is an extrapolation.")
    t = time.perf_counter()
    try:
        pred = w.predict(X)[0]
        proba = w.predict_proba(X) if w.task_type == "Classification" else None
    except Exception as e:
        ms = (time.perf_counter() - t) * 1000
        record(row, dict(zip(names, vals)), {"error": str(e)}, ms, False, source)
        raise RuntimeError(f"Model failed to run: {e}")
    ms = (time.perf_counter() - t) * 1000
    pred = pred.item() if hasattr(pred, "item") else pred
    res = {"modelId": row["id"], "modelName": row["name"], "latencyMs": round(ms, 2), "warnings": warnings}
    if w.task_type == "Regression":
        res.update(value=round(float(pred), 4), label=f"{float(pred):.4g}", confidence=None)
    else:
        res["label"] = class_label(row, pred)
        res["rawClass"] = str(pred)
        if proba is not None:
            classes = w.classes or list(range(proba.shape[1]))
            res["probabilities"] = {class_label(row, c): round(float(p), 4) for c, p in zip(classes, proba[0])}
            res["confidence"] = round(float(proba[0].max()), 4)
        else:
            res["confidence"] = None
    record(row, dict(zip(names, vals)), {"label": res["label"], "confidence": res.get("confidence")}, ms, True, source)
    return res


def monitor_data(row, w, ev):
    preds = db.q("SELECT * FROM predictions WHERE model_id=? ORDER BY id DESC LIMIT 500", (row["id"],))
    total = db.q("SELECT COUNT(*) c, SUM(ok) s FROM predictions WHERE model_id=?", (row["id"],), one=True)
    n = total["c"] or 0
    okn = total["s"] or 0
    lat = [p["latency_ms"] for p in preds if p["ok"]]
    names = feature_names(row, w)
    rows = []
    for p in preds:
        if p["ok"]:
            d = db.jd(p["inputs"], {})
            if all(k in d for k in names):
                rows.append([d[k] for k in names])
    drift = ml_core.drift_report(ev["dataset"] if ev is not None else pd.DataFrame(columns=names), names, rows) \
        if ev is not None else {"status": "No reference data", "n": len(rows), "features": [],
                                "note": "Attach an evaluation dataset to enable drift detection."}
    series = [{"t": p["ts"][11:19], "v": round(p["latency_ms"], 2)} for p in reversed(preds) if p["ok"]][-30:]
    return {"total": n, "ok": okn, "error_rate": (n - okn) / n if n else 0.0,
            "avg_latency": round(float(np.mean(lat)), 2) if lat else None,
            "p95_latency": round(float(np.percentile(lat, 95)), 2) if lat else None,
            "series": series, "drift": drift, "test_traffic": sum(1 for p in preds if p["source"] != "manual"),
            "class_counts": _pred_counts(preds, row)}


def _pred_counts(preds, row):
    c = {}
    for p in preds:
        if p["ok"]:
            l = db.jd(p["output"], {}).get("label")
            c[l] = c.get(l, 0) + 1
    return c


def full_metrics(row, w, ev):
    if ev is None:
        return None, None, None
    m, pred, proba = ml_core.evaluate(w, ev["X"], ev["y"])
    return m, pred, proba


def compare_models(rowa, rowb):
    wa, wb = get_wrapped(rowa), get_wrapped(rowb)
    eva = get_eval(rowa, wa)
    src = rowa
    if eva is None:
        eva, src = get_eval(rowb, wb), rowb
    if eva is None:
        raise ValueError("Neither model has an evaluation dataset. Attach a CSV to one of them (Get Started) so both "
                         "can be scored on the same rows.")
    if wa.task_type != wb.task_type:
        raise ValueError("Models solve different task types (classification vs regression) and cannot be compared.")
    try:
        Xa = ml_core.select_X(eva["dataset"], wa)
        Xb = ml_core.select_X(eva["dataset"], wb)
    except ValueError as e:
        raise ValueError("These two models take different input features, so they cannot be scored on the same rows "
                         f"({e})")
    if wa.task_type == "Classification" and wa.classes and wb.classes and list(wa.classes) != list(wb.classes):
        raise ValueError("Models predict different class sets, so they are not comparable.")
    y = eva["y"]
    ma, pa, _ = ml_core.evaluate(wa, Xa, y)
    mb, pb, _ = ml_core.evaluate(wb, Xb, y)
    v, deltas = ml_core.verdict(ma, mb)
    out = {"source": eva["source"], "evalOwner": src["id"], "a": {"model": model_dict(rowa), "metrics": ma},
           "b": {"model": model_dict(rowb), "metrics": mb}, "deltas": deltas, "verdict": v, "n": int(len(y))}
    if wa.task_type == "Classification":
        ao, bo, p = ml_core.mcnemar(np.asarray(y), pa, pb)
        out.update(agreement=round(float((pa == pb).mean()), 4), onlyACorrect=ao, onlyBCorrect=bo, pValue=round(p, 5),
                   significant=bool(p < 0.05))
    return out


# ------------------------------------------------------------------ demo seeding
def seed_demo():
    if db.q("SELECT id FROM models WHERE is_demo=1 LIMIT 1", one=True):
        return
    Xtr, Xte, ytr, yte = demo_data.split()
    for name, est, desc in demo_data.demo_estimators():
        est.fit(Xtr.values if False else Xtr, ytr)
        path = MODELS_DIR / (secure_filename(name.lower().replace(" ", "_")) + ".joblib")
        joblib.dump(est, path)
        w = ml_core.Wrapped("sklearn", est)
        db.x("""INSERT INTO models(owner,name,filename,path,size_mb,framework_declared,framework_detected,task_type,kind,
                n_features,feature_names,classes,class_labels,trained_on_demo,is_demo,fix_note,created)
                VALUES(0,?,?,?,?,?,?,?,?,?,?,?,?,1,1,?,?)""",
             (name, path.name, str(path), round(path.stat().st_size / 1048576, 3), "Scikit-learn", "Scikit-learn",
              "Classification", "sklearn", w.n_features, db.jl(w.feature_names), db.jl(w.classes),
              db.jl(demo_data.CLASS_LABELS), desc, db.now()))


seed_demo()


# ------------------------------------------------------------------ auth
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "message": "ML Troubleshooter API is running"})


def token_for(u):
    return signer.dumps({"uid": u["id"]})


@app.post("/api/auth/register")
def register():
    d = request.get_json(silent=True) or {}
    name, email, pw = (d.get("name") or "").strip(), (d.get("email") or "").strip().lower(), d.get("password") or ""
    if not name or "@" not in email or "." not in email.split("@")[-1]:
        return err("Enter your name and a valid email address.")
    if len(pw) < 6:
        return err("Password must be at least 6 characters.")
    if db.q("SELECT id FROM users WHERE email=?", (email,), one=True):
        return err("An account with this email already exists. Please log in.", 409)
    uid = db.x("INSERT INTO users(name,email,pw_hash,created) VALUES(?,?,?,?)",
               (name, email, generate_password_hash(pw), db.now()))
    u = {"id": uid, "name": name, "email": email}
    db.log(uid, "Account created")
    return jsonify({"token": token_for(u), "user": {"name": name, "email": email}})


@app.post("/api/auth/login")
def login():
    d = request.get_json(silent=True) or {}
    email, pw = (d.get("email") or "").strip().lower(), d.get("password") or ""
    u = db.q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if not u or not check_password_hash(u["pw_hash"], pw):
        return err("Incorrect email or password.", 401)
    db.log(u["id"], "Signed in")
    return jsonify({"token": token_for(u), "user": {"name": u["name"], "email": u["email"]}})


@app.get("/api/auth/me")
@auth
def me():
    return jsonify({"user": {"name": g.user["name"], "email": g.user["email"]}})


# ------------------------------------------------------------------ models
@app.get("/api/models")
@auth
def list_models():
    rows = db.q("SELECT * FROM models WHERE owner IN (0,?) ORDER BY id", (g.user["id"],))
    return jsonify([model_dict(r) for r in rows])


@app.post("/api/models/upload")
@auth
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("Please choose a model file.")
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return err("Unsupported file type. Use .pkl, .joblib, .onnx or .h5.")
    owner_dir = UPLOADS / str(g.user["id"]); owner_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = owner_dir / f"{stamp}_{secure_filename(f.filename)}"
    f.save(path)
    try:
        w = ml_core.load_model(path, ext)
    except ModelLoadError as e:
        path.unlink(missing_ok=True)
        db.log(g.user["id"], f"Upload rejected ({f.filename}): {e}", "error")
        return err(str(e), 422)
    ds_path, ds_target = None, None
    d = request.files.get("dataset")
    if d and d.filename:
        ds_path = owner_dir / f"{stamp}_dataset_{secure_filename(d.filename)}"
        d.save(ds_path)
        try:
            ml_core.select_X(ml_core.read_dataset(ds_path, request.form.get("target") or None)[0], w)
        except Exception as e:
            path.unlink(missing_ok=True); ds_path.unlink(missing_ok=True)
            return err(f"Evaluation dataset problem: {e}", 422)
        ds_target = request.form.get("target") or None
    name = (request.form.get("modelName") or "").strip() or Path(f.filename).stem
    det = ml_core.detect_framework(w.obj, w.kind)
    mid = db.x("""INSERT INTO models(owner,name,filename,path,size_mb,framework_declared,framework_detected,task_type,kind,
                  n_features,feature_names,classes,class_labels,dataset_path,dataset_target,created)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (g.user["id"], name, secure_filename(f.filename), str(path), round(path.stat().st_size / 1048576, 3),
                request.form.get("framework", "Scikit-learn"), det, w.task_type, w.kind, w.n_features,
                db.jl(w.feature_names), db.jl(w.classes), db.jl({}), str(ds_path) if ds_path else None, ds_target, db.now()))
    _models[mid] = w
    db.log(g.user["id"], f"Model uploaded and loaded: {name}")
    return jsonify({"message": "Model uploaded successfully", "model": model_dict(db.q("SELECT * FROM models WHERE id=?", (mid,), one=True))})


@app.post("/api/models/<int:mid>/dataset")
@auth
def attach_dataset(mid):
    row = get_row(mid)
    if not row or row["is_demo"]:
        return err("Model not found.", 404)
    f = request.files.get("dataset")
    if not f or not f.filename:
        return err("Choose a CSV file.")
    w = get_wrapped(row)
    p = UPLOADS / str(g.user["id"]); p.mkdir(exist_ok=True)
    path = p / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_dataset_{secure_filename(f.filename)}"
    f.save(path)
    try:
        ml_core.select_X(ml_core.read_dataset(path, request.form.get("target") or None)[0], w)
        ml_core._encode_y(ml_core.read_dataset(path, request.form.get("target") or None)[1], w)
    except Exception as e:
        path.unlink(missing_ok=True)
        return err(f"Dataset problem: {e}", 422)
    db.x("UPDATE models SET dataset_path=?, dataset_target=? WHERE id=?", (str(path), request.form.get("target") or None, mid))
    _eval.pop(mid, None)
    db.log(g.user["id"], f"Evaluation dataset attached to {row['name']}")
    return jsonify({"ok": True})


@app.delete("/api/models/<int:mid>")
@auth
def delete_model(mid):
    row = get_row(mid)
    if not row or row["is_demo"] or row["owner"] != g.user["id"]:
        return err("Only your own uploaded models can be deleted.", 403)
    Path(row["path"]).unlink(missing_ok=True)
    db.x("DELETE FROM models WHERE id=?", (mid,)); db.x("DELETE FROM predictions WHERE model_id=?", (mid,))
    _models.pop(mid, None); _eval.pop(mid, None)
    return jsonify({"ok": True})


@app.get("/api/models/<int:mid>/info")
@auth
def info(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    try:
        w = get_wrapped(row)
    except ModelLoadError as e:
        return err(str(e), 422)
    ev = get_eval(row, w) if not _safe_eval_err(row, w) else None
    return jsonify({"model": model_dict(row), "features": feature_specs(row, w, ev),
                    "evalSource": ev["source"] if ev else None, "classes": [class_label(row, c) for c in (w.classes or [])]})


def _safe_eval_err(row, w):
    try:
        get_eval(row, w); return None
    except Exception as e:
        return str(e)


@app.post("/api/models/<int:mid>/validate")
@auth
def validate(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    try:
        w = get_wrapped(row)
    except ModelLoadError as e:
        return jsonify({"status": "Failed", "items": [{"label": "Load model", "value": str(e), "ok": False}]})
    items = ml_core.validation_report(row, w, row["size_mb"])
    items[3]["ok"] = items[3]["ok"] and (row["task_type"] == w.task_type or row["is_demo"])
    try:
        ev = get_eval(row, w)
        items.append({"label": "Evaluation Data", "value": ev["source"] if ev else
                      "None attached - upload a CSV to get real metrics for this model", "ok": ev is not None})
    except Exception as e:
        items.append({"label": "Evaluation Data", "value": str(e), "ok": False})
    hard_fail = any(not i["ok"] for i in items if i["label"] in ("Required Input", "Framework Compatibility"))
    status = "Failed" if hard_fail else ("Passed" if all(i["ok"] for i in items) else "Passed with warnings")
    db.log(g.user["id"], f"Validation {status.lower()}: {row['name']}", "success" if status != "Failed" else "error")
    return jsonify({"status": status, "items": items})


@app.post("/api/models/<int:mid>/checks")
@auth
def checks(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    try:
        w = get_wrapped(row)
        ev = get_eval(row, w) if not _safe_eval_err(row, w) else None
    except ModelLoadError as e:
        return err(str(e), 422)
    res = ml_core.run_checks(w, ev)
    st = "Failed" if any(c["status"] == "Failed" for c in res) else ("Warning" if any(c["status"] == "Warning" for c in res) else "Passed")
    db.log(g.user["id"], f"Checks {st.lower()} ({sum(c['status']=='Passed' for c in res)}/{len(res)} passed): {row['name']}",
           "success" if st != "Failed" else "error")
    return jsonify({"status": st, "checks": res})


@app.get("/api/models/<int:mid>/overview")
@auth
def overview(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    try:
        w = get_wrapped(row)
    except ModelLoadError as e:
        return err(str(e), 422)
    ee = _safe_eval_err(row, w)
    ev = None if ee else get_eval(row, w)
    base = {"model": model_dict(row), "evalSource": ev["source"] if ev else None, "evalError": ee}
    if ev is None:
        base.update(metrics=None, health="Unknown", issues=[], note=ee or "No evaluation data. Attach a labelled CSV to see accuracy, "
                    "precision, recall and F1 for this model.")
        return jsonify(base)
    m, pred, _ = full_metrics(row, w, ev)
    mon = monitor_data(row, w, ev)
    issues = ml_core.diagnose(row, w, ev, m, train_acc(row, w), mon["drift"], mon["error_rate"], mon["total"])
    dist_src = "logged predictions" if mon["class_counts"] else "evaluation data"
    if mon["class_counts"]:
        dist = mon["class_counts"]
    elif "predicted_distribution" in m:
        dist = {class_label(row, c): n for c, n in m["predicted_distribution"].items()}
    else:
        dist = {}
    base.update(metrics=m, health=ml_core.health(m, issues), issueCount=len(issues), distribution=dist, distributionSource=dist_src,
                predictionsLogged=mon["total"])
    return jsonify(base)


@app.get("/api/models/<int:mid>/issues")
@auth
def issues(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    w = get_wrapped(row)
    ee = _safe_eval_err(row, w)
    ev = None if ee else get_eval(row, w)
    if ev is None:
        return jsonify({"issues": [], "note": ee or "No evaluation data attached, so no diagnosis is possible yet."})
    m, _, _ = full_metrics(row, w, ev)
    mon = monitor_data(row, w, ev)
    iss = ml_core.diagnose(row, w, ev, m, train_acc(row, w), mon["drift"], mon["error_rate"], mon["total"])
    return jsonify({"issues": iss, "metrics": m, "canFix": bool(row["trained_on_demo"]), "note": None})


@app.post("/api/models/<int:mid>/predict")
@auth
def predict(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    try:
        w = get_wrapped(row)
        ev = get_eval(row, w) if not _safe_eval_err(row, w) else None
        res = run_prediction(row, w, ev, (request.get_json(silent=True) or {}).get("features"))
    except ValueError as e:
        return err(str(e))
    except (RuntimeError, ModelLoadError) as e:
        return err(str(e), 500)
    db.log(g.user["id"], f"Prediction: {row['name']} -> {res['label']}" + (f" ({res['confidence']:.2f})" if res.get("confidence") is not None else ""))
    return jsonify(res)


@app.post("/api/predict-compare")
@auth
def predict_compare():
    d = request.get_json(silent=True) or {}
    ids = d.get("modelIds") or []
    if len(ids) != 2 or ids[0] == ids[1]:
        return err("Pick two different models.")
    out = []
    for mid in ids:
        row = get_row(int(mid))
        if not row: return err("Model not found.", 404)
        try:
            w = get_wrapped(row)
            ev = get_eval(row, w) if not _safe_eval_err(row, w) else None
            out.append(run_prediction(row, w, ev, d.get("features"), source="compare"))
        except ValueError as e:
            return err(f"{row['name']}: {e}")
        except RuntimeError as e:
            return err(f"{row['name']}: {e}", 500)
    out[0]["agree"] = out[1]["agree"] = out[0]["label"] == out[1]["label"]
    return jsonify({"results": out})


@app.post("/api/compare")
@auth
def compare():
    d = request.get_json(silent=True) or {}
    try:
        a, b = get_row(int(d.get("a", 0))), get_row(int(d.get("b", 0)))
    except (TypeError, ValueError):
        return err("Pick two models.")
    if not a or not b or a["id"] == b["id"]:
        return err("Pick two different models.")
    try:
        res = compare_models(a, b)
    except ValueError as e:
        return err(str(e), 422)
    db.log(g.user["id"], f"Compared {a['name']} vs {b['name']}: {res['verdict']}")
    return jsonify(res)


@app.post("/api/models/<int:mid>/fix")
@auth
def fix(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    if not row["trained_on_demo"]:
        return err("Automatic fixes retrain the model, which needs its training data. This is only available for the "
                   "built-in demo models and their fixed versions. For your own model, apply the fix yourself, "
                   "upload the new version, and compare the two in Fix Verification.", 422)
    action = (request.get_json(silent=True) or {}).get("action")
    w = get_wrapped(row)
    est, note = ml_core.build_fixed(w.obj, action)
    if est is None:
        return err(note, 422)
    Xtr, _, ytr, _ = demo_data.split()
    est.fit(Xtr, ytr)
    label = {"increase_capacity": "more capacity", "balance_classes": "balanced classes", "regularize": "regularized"}[action]
    name = f"{row['name'].split(' [')[0]} [fixed: {label}]"
    path = MODELS_DIR / f"u{g.user['id']}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.joblib"
    joblib.dump(est, path)
    nw = ml_core.Wrapped("sklearn", est)
    nid = db.x("""INSERT INTO models(owner,name,filename,path,size_mb,framework_declared,framework_detected,task_type,kind,
                  n_features,feature_names,classes,class_labels,parent_id,trained_on_demo,fix_note,created)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)""",
               (g.user["id"], name, path.name, str(path), round(path.stat().st_size / 1048576, 3), "Scikit-learn",
                "Scikit-learn", "Classification", "sklearn", nw.n_features, db.jl(nw.feature_names), db.jl(nw.classes),
                row["class_labels"], mid, note, db.now()))
    _models[nid] = nw
    new = db.q("SELECT * FROM models WHERE id=?", (nid,), one=True)
    db.log(g.user["id"], f"Fix applied to {row['name']}: {note}")
    return jsonify({"newModel": model_dict(new), "note": note, "comparison": compare_models(row, new)})


# ------------------------------------------------------------------ monitoring
@app.get("/api/models/<int:mid>/monitor")
@auth
def monitor(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    w = get_wrapped(row)
    ee = _safe_eval_err(row, w)
    ev = None if ee else get_eval(row, w)
    return jsonify(monitor_data(row, w, ev))


@app.post("/api/models/<int:mid>/test-traffic")
@auth
def test_traffic(mid):
    """Sends real rows from the evaluation data through the model one request at a time (measured latency).
    Optional drift=true shifts every feature by +1 std to exercise the drift detector. Logged as test traffic."""
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    w = get_wrapped(row)
    ee = _safe_eval_err(row, w)
    ev = None if ee else get_eval(row, w)
    if ev is None:
        return err("Test traffic needs evaluation data to sample from. Attach a CSV first.", 422)
    d = request.get_json(silent=True) or {}
    n = max(1, min(int(d.get("n", 40)), 100))
    drift = bool(d.get("drift"))
    rng = np.random.default_rng()
    idx = rng.choice(len(ev["X"]), size=n, replace=False)
    sd = ev["X"].std(axis=0)
    names = feature_names(row, w)
    ok = 0
    for i in idx:
        x = ev["X"][i] + (sd if drift else 0)
        try:
            run_prediction(row, w, None, dict(zip(names, x.tolist())), source="test-drift" if drift else "test")
            ok += 1
        except Exception:
            pass
    db.log(g.user["id"], f"Sent {ok} {'drifted ' if drift else ''}test requests to {row['name']}")
    return jsonify({"sent": ok})


@app.delete("/api/models/<int:mid>/predictions")
@auth
def clear_predictions(mid):
    row = get_row(mid)
    if not row: return err("Model not found.", 404)
    db.x("DELETE FROM predictions WHERE model_id=?", (mid,))
    return jsonify({"ok": True})


@app.get("/api/logs")
@auth
def logs():
    rows = db.q("SELECT ts,message,status FROM logs WHERE owner=? ORDER BY id DESC LIMIT 40", (g.user["id"],))
    return jsonify([{"time": r["ts"][11:16], "ts": r["ts"], "message": r["message"], "status": r["status"]} for r in rows])


# ------------------------------------------------------------------ reports
REPORT_SECTIONS = ["Model Summary", "Validation & Check Results", "Issues & Recommendations",
                   "Verification Results", "Monitoring & Logs"]


@app.post("/api/report")
@auth
def report():
    d = request.get_json(silent=True) or {}
    row = get_row(int(d.get("modelId", 0) or 0))
    if not row: return err("Choose a model first.")
    rtype = d.get("type", "Full Diagnostic Report")
    sections = [s for s in (d.get("sections") or []) if s in REPORT_SECTIONS]
    if not sections:
        return err("Select at least one section.")
    try:
        w = get_wrapped(row)
    except ModelLoadError as e:
        return err(str(e), 422)
    ee = _safe_eval_err(row, w)
    ev = None if ee else get_eval(row, w)
    metrics = full_metrics(row, w, ev)[0] if ev else None
    mon = monitor_data(row, w, ev)
    iss = ml_core.diagnose(row, w, ev, metrics, train_acc(row, w), mon["drift"], mon["error_rate"], mon["total"]) if metrics else []
    data = {"rtype": rtype, "model": model_dict(row), "generated": db.now(), "eval_source": ev["source"] if ev else None,
            "sections": sections, "metrics": metrics, "health": ml_core.health(metrics, iss), "issues": iss, "monitor": mon,
            "checks": ml_core.run_checks(w, ev), "verification": None,
            "logs": db.q("SELECT ts,message FROM logs WHERE owner=? ORDER BY id DESC LIMIT 15", (g.user["id"],))}
    if row["parent_id"]:
        par = db.q("SELECT * FROM models WHERE id=?", (row["parent_id"],), one=True)
        try:
            c = compare_models(par, row)
            data["verification"] = {"before_name": par["name"], "after_name": row["name"], "verdict": c["verdict"],
                                    "deltas": c["deltas"], "before": c["a"]["metrics"], "after": c["b"]["metrics"]}
        except ValueError:
            pass
    fname = f"{secure_filename(row['name']) or 'model'}_{rtype.split()[0].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = REPORTS / fname
    build_pdf(path, data)
    rid = db.x("INSERT INTO reports(owner,filename,path,ts,size_bytes,rtype,model_name) VALUES(?,?,?,?,?,?,?)",
               (g.user["id"], fname, str(path), db.now(), path.stat().st_size, rtype, row["name"]))
    db.log(g.user["id"], f"Report generated: {fname}")
    return jsonify({"id": rid, "filename": fname})


@app.get("/api/reports")
@auth
def reports():
    rows = db.q("SELECT id,filename,ts,size_bytes,rtype,model_name FROM reports WHERE owner=? ORDER BY id DESC LIMIT 10", (g.user["id"],))
    return jsonify(rows)


@app.get("/api/reports/<int:rid>/download")
@auth
def download(rid):
    r = db.q("SELECT * FROM reports WHERE id=? AND owner=?", (rid, g.user["id"]), one=True)
    if not r or not Path(r["path"]).exists():
        return err("Report not found.", 404)
    return send_file(r["path"], mimetype="application/pdf", as_attachment=True, download_name=r["filename"])


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
