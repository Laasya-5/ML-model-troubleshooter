"""Model loading, inference, evaluation, diagnostics. Everything returned here is measured, not invented."""
import math, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import joblib
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, brier_score_loss, r2_score, mean_absolute_error,
                             mean_squared_error)
import demo_data

warnings.filterwarnings("ignore")


class ModelLoadError(Exception):
    pass


FRAMEWORKS = {"sklearn": "Scikit-learn", "xgboost": "XGBoost", "lightgbm": "LightGBM", "torch": "PyTorch",
              "tensorflow": "TensorFlow", "keras": "TensorFlow", "onnx": "ONNX"}


class Wrapped:
    """Uniform interface over sklearn-style estimators and ONNX sessions."""

    def __init__(self, kind, obj):
        self.kind, self.obj = kind, obj
        self.feature_names, self.n_features, self.classes, self.task_type = None, None, None, "Classification"
        if kind == "sklearn":
            o = obj
            fn = getattr(o, "feature_names_in_", None)
            self.feature_names = [str(v) for v in fn] if fn is not None else None
            self.n_features = getattr(o, "n_features_in_", None)
            if self.n_features is None and self.feature_names:
                self.n_features = len(self.feature_names)
            cl = getattr(o, "classes_", None)
            self.classes = [c.item() if hasattr(c, "item") else c for c in cl] if cl is not None else None
            try:
                self.task_type = "Regression" if (is_regressor(o) and not is_classifier(o)) else "Classification"
            except Exception:
                self.task_type = "Classification" if hasattr(o, "predict_proba") else "Regression"
            self.has_proba = hasattr(o, "predict_proba")
        else:  # onnx
            inp = obj.get_inputs()[0]
            shp = inp.shape
            self.n_features = shp[1] if len(shp) == 2 and isinstance(shp[1], int) else None
            self.input_name = inp.name
            outs = obj.get_outputs()
            self.has_proba = len(outs) > 1
            self.task_type = "Classification" if len(outs) > 1 or "int" in outs[0].type else "Regression"
            self.has_proba = len(outs) > 1

    # ---- inference -------------------------------------------------------
    def _frame(self, X):
        X = np.asarray(X, dtype=float)
        if self.feature_names and X.shape[1] == len(self.feature_names):
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    def predict(self, X):
        if self.kind == "sklearn":
            return np.asarray(self.obj.predict(self._frame(X)))
        out = self.obj.run(None, {self.input_name: np.asarray(X, dtype=np.float32)})
        return np.asarray(out[0]).reshape(len(X), -1)[:, 0] if np.asarray(out[0]).ndim > 1 else np.asarray(out[0])

    def predict_proba(self, X):
        if self.kind == "sklearn":
            if not self.has_proba:
                return None
            return np.asarray(self.obj.predict_proba(self._frame(X)))
        if not self.has_proba:
            return None
        out = self.obj.run(None, {self.input_name: np.asarray(X, dtype=np.float32)})[1]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            keys = sorted(out[0].keys())
            if self.classes is None:
                self.classes = keys
            return np.array([[row[k] for k in keys] for row in out], dtype=float)
        return np.asarray(out, dtype=float)


def detect_framework(obj, kind):
    if kind == "onnx":
        return "ONNX"
    mod = type(obj).__module__.split(".")[0]
    if mod in FRAMEWORKS:
        return FRAMEWORKS[mod]
    # Pipelines etc. live in sklearn; unknown third-party estimators fall through
    return FRAMEWORKS.get(mod, mod or "Unknown")


def load_model(path, ext):
    ext = ext.lower()
    if ext in (".pkl", ".joblib"):
        try:
            obj = joblib.load(path)
        except Exception as e:  # corrupt file, missing class/library, etc.
            raise ModelLoadError("Could not load this file. It is not a valid pickle/joblib model, or it was saved "
                                 f"with an incompatible library version or one that is not installed here ({type(e).__name__}: {e}).")
        if isinstance(obj, dict) or not hasattr(obj, "predict"):
            raise ModelLoadError("File loaded but the object has no predict() method "
                                 f"(found {type(obj).__name__}). Save the fitted model or Pipeline itself.")
        if is_classifier(obj) or is_regressor(obj) or hasattr(obj, "n_features_in_") or hasattr(obj, "classes_"):
            pass
        return Wrapped("sklearn", obj)
    if ext == ".onnx":
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelLoadError(f"Could not load ONNX model: {e}")
        return Wrapped("onnx", sess)
    if ext == ".h5":
        raise ModelLoadError("Keras .h5 models need TensorFlow, which is not installed in this backend. "
                             "Export the model to ONNX (.onnx) or scikit-learn (.joblib) instead.")
    raise ModelLoadError(f"Unsupported file type {ext}")


# ---------------------------------------------------------------------------------------
# evaluation data
# ---------------------------------------------------------------------------------------
TARGET_NAMES = ("target", "label", "y", "class", "outcome")


def read_dataset(path, target=None):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The CSV has no rows.")
    cols = list(df.columns)
    if target and target in cols:
        t = target
    else:
        t = next((c for c in cols if str(c).lower() in TARGET_NAMES), cols[-1])
    feats = df.drop(columns=[t]).apply(pd.to_numeric, errors="coerce")
    bad = feats.isna().any(axis=1) | df[t].isna()
    feats, y = feats[~bad], df[t][~bad]
    if len(feats) < 20:
        raise ValueError("Need at least 20 clean numeric rows to evaluate.")
    return feats.reset_index(drop=True), y.reset_index(drop=True), t


def select_X(feats: pd.DataFrame, w: Wrapped):
    """Pick/ordering of columns for the model. Raises ValueError when it can't be done honestly."""
    if w.feature_names:
        missing = [c for c in w.feature_names if c not in feats.columns]
        if missing:
            raise ValueError(f"Dataset is missing feature columns the model expects: {missing}")
        return feats[w.feature_names].values.astype(float)
    if w.n_features and feats.shape[1] != w.n_features:
        raise ValueError(f"Model expects {w.n_features} features but the dataset has {feats.shape[1]} feature columns.")
    return feats.values.astype(float)


def eval_data(row, w: Wrapped):
    """Returns dict(X, y, names, source) for this model, or None when no honest evaluation set exists."""
    if row.get("dataset_path") and Path(row["dataset_path"]).exists():
        feats, y, t = read_dataset(row["dataset_path"], row.get("dataset_target"))
        X = select_X(feats, w)
        names = w.feature_names or list(feats.columns)
        return dict(X=X, y=_encode_y(y, w), names=names, source=f"Uploaded dataset ({len(y)} rows, target '{t}')", dataset=feats)
    if w.feature_names == demo_data.FEATURES:
        _, Xte, _, yte = demo_data.split()
        return dict(X=Xte.values, y=yte, names=demo_data.FEATURES,
                    source=f"Built-in synthetic holdout ({len(yte)} rows, not used for training)", dataset=Xte)
    return None


def _encode_y(y, w):
    y = pd.Series(y)
    if w.task_type == "Regression":
        return y.astype(float).values
    if w.classes:
        cl = list(w.classes)
        if y.dtype.kind in "iuf" and all(isinstance(c, (int, float, np.integer)) for c in cl):
            return y.astype(type(cl[0]) if not isinstance(cl[0], float) else float).values
        # string classes: match on string form
        m = {str(c): c for c in cl}
        out = y.astype(str).map(m)
        if out.isna().any():
            raise ValueError("Target values do not match the model's classes: "
                             f"dataset has {sorted(y.astype(str).unique())[:6]}, model has {[str(c) for c in cl][:6]}")
        return out.values
    return y.values


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------
def r4(v):
    return None if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else round(float(v), 4)


def evaluate(w: Wrapped, X, y, n_batches=6):
    pred = w.predict(X)
    out = {"n": int(len(y)), "task": w.task_type}
    if w.task_type == "Regression":
        pred = pred.astype(float)
        out.update(r2=r4(r2_score(y, pred)), mae=r4(mean_absolute_error(y, pred)),
                   rmse=r4(math.sqrt(mean_squared_error(y, pred))))
        idx = np.array_split(np.arange(len(y)), n_batches)
        out["batches"] = [r4(r2_score(y[i], pred[i])) for i in idx if len(i) > 2]
        return out, pred, None
    classes = list(w.classes) if w.classes else sorted(set(np.unique(y)) | set(np.unique(pred)))
    binary = len(classes) == 2
    avg = "binary" if binary else "weighted"
    kw = dict(average=avg, zero_division=0)
    if binary:
        kw["pos_label"] = classes[1]
    out.update(accuracy=r4(accuracy_score(y, pred)), precision=r4(precision_score(y, pred, **kw)),
               recall=r4(recall_score(y, pred, **kw)), f1=r4(f1_score(y, pred, **kw)))
    proba = None
    try:
        proba = w.predict_proba(X)
    except Exception:
        proba = None
    if proba is not None and proba.shape[1] == len(classes):
        try:
            if binary:
                out["roc_auc"] = r4(roc_auc_score((np.asarray(y) == classes[1]).astype(int), proba[:, 1]))
                out["brier"] = r4(brier_score_loss((np.asarray(y) == classes[1]).astype(int), proba[:, 1]))
                out["ece"] = r4(_ece((np.asarray(y) == classes[1]).astype(int), proba[:, 1]))
            else:
                out["roc_auc"] = r4(roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=classes))
        except Exception:
            pass
        out["mean_confidence"] = r4(float(np.mean(proba.max(axis=1))))
    cm = confusion_matrix(y, pred, labels=classes)
    out["classes"] = [str(c) for c in classes]
    out["confusion"] = cm.tolist()
    out["actual_distribution"] = {str(c): int((np.asarray(y) == c).sum()) for c in classes}
    out["predicted_distribution"] = {str(c): int((pred == c).sum()) for c in classes}
    idx = np.array_split(np.arange(len(y)), n_batches)
    out["batches"] = [r4(accuracy_score(np.asarray(y)[i], pred[i])) for i in idx if len(i)]
    return out, pred, proba


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def mcnemar(y, pa, pb):
    """Exact McNemar test on paired correctness. Returns (a_only_correct, b_only_correct, p_value)."""
    from scipy.stats import binomtest
    ca, cb = (pa == y), (pb == y)
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p


def verdict(a, b):
    """Compare metric dicts a (before) and b (after). Primary metrics: accuracy + F1 (R2 for regression);
    a large drop in any secondary metric makes the verdict 'mixed'."""
    reg = a.get("task") == "Regression"
    keys = ["r2", "mae", "rmse"] if reg else ["accuracy", "precision", "recall", "f1"]
    lower_better = {"mae", "rmse"}
    deltas, gain = {}, {}
    for k in keys:
        if a.get(k) is None or b.get(k) is None:
            continue
        deltas[k] = round(b[k] - a[k], 4)
        gain[k] = -deltas[k] if k in lower_better else deltas[k]
    primary = ["r2"] if reg else ["accuracy", "f1"]
    pg = [gain[k] for k in primary if k in gain]
    if not pg:
        return "unchanged", deltas
    secondary_drop = any(v < -0.03 for k, v in gain.items() if k not in primary)
    if all(v > 0.002 for v in pg) and not secondary_drop:
        return "improved", deltas
    if all(v < -0.002 for v in pg):
        return "regressed", deltas
    if all(abs(v) <= 0.002 for v in gain.values()):
        return "unchanged", deltas
    if any(v > 0.002 for v in pg) and not any(v < -0.002 for v in pg) and not secondary_drop:
        return "improved", deltas
    return "mixed", deltas


# ---------------------------------------------------------------------------------------
# validation + checks
# ---------------------------------------------------------------------------------------
def sample_input(w, ev):
    if ev is not None:
        return ev["X"][:1], "first row of the evaluation dataset"
    if w.n_features:
        return np.zeros((1, w.n_features)), "an all-zero vector (no evaluation data available)"
    return None, None


def run_checks(w: Wrapped, ev):
    import sklearn, numpy
    checks = []

    def add(name, desc, status, detail):
        checks.append({"name": name, "description": desc, "status": status, "detail": detail})

    # 1 runtime
    try:
        import onnxruntime
        ov = onnxruntime.__version__
    except Exception:
        ov = "n/a"
    add("Runtime Environment", "Model loaded and runtime libraries available", "Passed",
        f"{type(w.obj).__name__ if w.kind=='sklearn' else 'ONNX session'} loaded · scikit-learn {sklearn.__version__} · "
        f"numpy {numpy.__version__} · onnxruntime {ov}")
    # 2 input validation
    if not w.n_features:
        add("Input Validation", "Input format and feature consistency", "Warning",
            "Model does not declare how many features it expects, so input width cannot be verified.")
    else:
        try:
            w.predict(np.zeros((1, w.n_features + 1)))
            add("Input Validation", "Input format and feature consistency", "Failed",
                f"Model accepted {w.n_features + 1} features although it declares {w.n_features}.")
        except Exception:
            add("Input Validation", "Input format and feature consistency", "Passed",
                f"Declares {w.n_features} features" + (f" ({', '.join(w.feature_names)})" if w.feature_names else " (unnamed)")
                + "; correctly rejects the wrong number of inputs.")
    # 3 inference
    X1, how = sample_input(w, ev)
    out1 = None
    if X1 is None:
        add("Inference Test", "Run a sample prediction", "Failed", "Cannot build a sample input: feature count unknown.")
    else:
        try:
            t = time.perf_counter(); out1 = w.predict(X1); ms = (time.perf_counter() - t) * 1000
            add("Inference Test", "Run a sample prediction", "Passed",
                f"Predicted {out1[0]!r} for {how} in {ms:.1f} ms.")
        except Exception as e:
            add("Inference Test", "Run a sample prediction", "Failed", f"{type(e).__name__}: {e}")
    # 4 output validation
    if out1 is not None:
        probs, problems = None, []
        try:
            probs = w.predict_proba(X1)
        except Exception as e:
            problems.append(f"predict_proba failed: {e}")
        if out1.shape[0] != 1:
            problems.append(f"expected 1 output row, got shape {out1.shape}")
        if w.task_type == "Regression" and not np.isfinite(out1.astype(float)).all():
            problems.append("regression output is not finite")
        if w.classes and out1[0] not in w.classes:
            problems.append("predicted label not in model.classes_")
        if probs is not None:
            if not np.isfinite(probs).all() or (probs < -1e-9).any() or (probs > 1 + 1e-9).any():
                problems.append("probabilities outside [0,1]")
            if abs(float(probs.sum()) - 1) > 1e-3:
                problems.append(f"probabilities sum to {probs.sum():.4f}, not 1")
        add("Output Validation", "Verify prediction output and shape", "Failed" if problems else "Passed",
            "; ".join(problems) if problems else
            f"Output shape {tuple(out1.shape)}" + (f", probabilities valid (sum=1) over {probs.shape[1]} classes" if probs is not None else
                                                     ", no probability output available"))
    # 5 determinism
    if X1 is not None and out1 is not None:
        try:
            again = [w.predict(X1)[0] for _ in range(3)]
            same = all(a == out1[0] for a in again)
            add("Determinism", "Same input should give the same output", "Passed" if same else "Warning",
                "Identical output on 4 repeated calls." if same else f"Outputs varied across repeats: {again}")
        except Exception as e:
            add("Determinism", "Same input should give the same output", "Failed", str(e))
    # 6 latency + batch
    if ev is not None:
        try:
            Xb = ev["X"][:200]
            t = time.perf_counter(); w.predict(Xb); ms = (time.perf_counter() - t) * 1000
            add("Batch Throughput", f"Predict {len(Xb)} rows in one call", "Passed",
                f"{ms:.1f} ms total ({ms/len(Xb):.3f} ms per row).")
        except Exception as e:
            add("Batch Throughput", "Predict a batch", "Failed", str(e))
    # 7 prediction sensitivity: inputs matter
    if ev is not None:
        try:
            p = w.predict(ev["X"][:300])
            uniq = len(np.unique(p))
            if w.task_type == "Classification":
                add("Output Sensitivity", "Different inputs should not always give the same output",
                    "Passed" if uniq > 1 else "Warning",
                    f"{uniq} distinct predictions across 300 evaluation rows." if uniq > 1 else
                    "Model predicted a single class for every input - it may be degenerate.")
        except Exception:
            pass
    return checks


def validation_report(row, w, size_mb, ev_err=None):
    ext = Path(row["filename"]).suffix.lower()
    decl = row["framework_declared"]
    det = row["framework_detected"]
    compat = (decl == det) or (decl == "Scikit-learn" and det == "Scikit-learn")
    items = [
        {"label": "File Format", "value": f"{ext} (Supported)", "ok": True},
        {"label": "Model File Size", "value": f"{size_mb} MB", "ok": size_mb < 100},
        {"label": "Framework Compatibility",
         "value": f"{det}" + ("" if compat else f" - you selected {decl}"), "ok": compat},
        {"label": "Model Type", "value": f"{w.task_type}" + (f" ({len(w.classes)} classes)" if w.classes else ""),
         "ok": True},
        {"label": "Required Input",
         "value": (f"{w.n_features} features" if w.n_features else "Unknown (model does not declare it)"),
         "ok": bool(w.n_features)},
        {"label": "Feature Names",
         "value": ", ".join(w.feature_names) if w.feature_names else "Not stored in model (inputs will be labelled Feature 1..n)",
         "ok": bool(w.feature_names)},
        {"label": "Probability Output", "value": "Available" if w.has_proba else "Not available (labels only)",
         "ok": True},
    ]
    return items


# ---------------------------------------------------------------------------------------
# drift + diagnostics
# ---------------------------------------------------------------------------------------
def drift_report(ref_df: pd.DataFrame, names, rows):
    """Compare recent logged inputs against the reference data with a two-sample KS test."""
    from scipy.stats import ks_2samp
    if len(rows) < 20:
        return {"status": "Insufficient data", "n": len(rows), "features": [],
                "note": f"Needs at least 20 logged predictions (have {len(rows)})."}
    recent = pd.DataFrame(rows, columns=names)
    feats, drifted = [], []
    for i, n in enumerate(names):
        ref = ref_df.iloc[:, i] if n not in ref_df.columns else ref_df[n]
        stat, p = ks_2samp(ref.values, recent[n].values)
        sd = float(ref.std()) or 1.0
        shift = (float(recent[n].mean()) - float(ref.mean())) / sd
        d = bool(p < 0.01 and abs(shift) > 0.3)
        feats.append({"feature": n, "ks": r4(stat), "p_value": r4(p), "mean_shift_sd": r4(shift), "drifted": d})
        if d:
            drifted.append(n)
    return {"status": "Drift detected" if drifted else "Stable", "n": len(rows), "features": feats,
            "drifted": drifted, "note": None}


def diagnose(row, w, ev, metrics, train_acc, drift, err_rate, n_preds):
    issues = []

    def add(id_, title, desc, sev, steps, fix=None):
        issues.append({"id": id_, "title": title, "description": desc, "severity": sev, "steps": steps,
                       "fixAction": fix if (fix and row.get("trained_on_demo")) else None})

    if metrics is None:
        return issues
    if metrics["task"] == "Regression":
        r2 = metrics.get("r2")
        if r2 is not None and r2 < 0.5:
            add("low_r2", "Low explanatory power", f"R² on evaluation data is {r2:.3f}.", "High" if r2 < 0.2 else "Medium",
                ["Add informative features", "Try a higher-capacity model", "Check for target leakage or scaling bugs"])
        return issues
    acc, rec, prec, f1 = metrics["accuracy"], metrics["recall"], metrics["precision"], metrics["f1"]
    if acc < 0.8:
        add("low_acc", "High error rate detected", f"Error rate is {(1-acc)*100:.1f}% (accuracy {acc:.3f}); recommended maximum is 20%.", "High",
            ["Inspect misclassified rows in the confusion matrix", "Increase model capacity or add features",
             "Re-verify on the same holdout after the change"], "increase_capacity")
    elif acc < 0.88:
        add("mid_acc", "Elevated error rate", f"Error rate is {(1-acc)*100:.1f}% (accuracy {acc:.3f}); recommended maximum is 12%.", "Medium",
            ["Inspect misclassified rows", "Try a higher-capacity model or tuned hyper-parameters",
             "Re-verify on the same holdout after the change"], "increase_capacity")
    if len(metrics["classes"]) == 2:
        act = metrics["actual_distribution"]; prd = metrics["predicted_distribution"]
        c0, c1 = metrics["classes"]
        n = sum(act.values())
        act_pos, pred_pos = act[c1] / n, prd[c1] / n
        if rec < 0.8:
            add("low_recall", "Positive cases are being missed",
                f"Recall for class {c1} is {rec:.3f}: the model misses {(1-rec)*100:.0f}% of true positives.",
                "High" if rec < 0.7 else "Medium",
                ["Use class_weight='balanced' (or re-sample)", "Lower the decision threshold", "Re-verify recall after the change"],
                "balance_classes")
        if prec < 0.8:
            add("low_prec", "Many false alarms", f"Precision for class {c1} is {prec:.3f}.", "Medium",
                ["Raise the decision threshold", "Add discriminative features"])
        if min(act_pos, 1 - act_pos) < 0.25:
            add("imbalance", "Class imbalance", f"Minority class makes up {min(act_pos,1-act_pos)*100:.0f}% of the evaluation data.", "Low",
                ["Prefer F1/recall over accuracy", "Use stratified sampling and class weights"], "balance_classes")
        if pred_pos < act_pos - 0.06:
            add("under_pred", "Model under-predicts the positive class",
                f"It predicts class {c1} for {pred_pos*100:.1f}% of rows but {act_pos*100:.1f}% are truly {c1}.", "Medium",
                ["Rebalance classes or tune the threshold"], "balance_classes")
    if train_acc is not None:
        gap = train_acc - acc
        if gap > 0.08:
            add("overfit", "Possible overfitting", f"Training accuracy {train_acc:.3f} vs holdout {acc:.3f} (gap {gap:.3f}).", "Medium",
                ["Increase regularization (shallower trees, larger min leaf)", "Gather more data"], "regularize")
        elif train_acc < 0.88:
            add("underfit", "Possible underfitting", f"Training accuracy is only {train_acc:.3f}; the model is too simple for the data.", "Medium",
                ["Increase model capacity (depth / estimators)", "Engineer interaction features"], "increase_capacity")
    if metrics.get("ece") is not None and metrics["ece"] > 0.08:
        add("calibration", "Poorly calibrated probabilities", f"Expected calibration error is {metrics['ece']:.3f}.", "Low",
            ["Apply Platt scaling or isotonic calibration"])
    if drift and drift.get("status") == "Drift detected":
        add("drift", "Possible data drift", f"Recent inputs differ from the reference data in: {', '.join(drift['drifted'])}.", "Medium",
            ["Compare recent inputs with the training baseline", "Retrain with recent data"])
    if n_preds >= 10 and err_rate > 0.05:
        add("errors", "Prediction errors in production", f"{err_rate*100:.1f}% of {n_preds} logged requests failed (e.g. feature mismatch).", "High",
            ["Check the logs for rejected inputs", "Ensure clients send all required features"])
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(issues, key=lambda i: order[i["severity"]])


def health(metrics, issues):
    if metrics is None:
        return "Unknown"
    if any(i["severity"] == "High" for i in issues):
        return "Unhealthy"
    if any(i["severity"] == "Medium" for i in issues):
        return "Needs Attention"
    return "Healthy"


# ---------------------------------------------------------------------------------------
# fixes (real retraining on the built-in training split)
# ---------------------------------------------------------------------------------------
def build_fixed(est, action):
    est = clone(est)
    p = est.get_params()
    if action == "balance_classes":
        if "class_weight" not in p:
            return None, "This estimator has no class_weight parameter."
        est.set_params(class_weight="balanced")
        return est, "Retrained with class_weight='balanced'."
    if action == "increase_capacity":
        if "max_depth" in p and p["max_depth"] is not None:
            nd = p["max_depth"] + 3
            est.set_params(max_depth=nd)
            if "class_weight" in p:
                est.set_params(class_weight="balanced")
            return est, f"Retrained with max_depth {p['max_depth']} -> {nd}" + (" and balanced class weights." if "class_weight" in p else ".")
        if "n_estimators" in p:
            est.set_params(n_estimators=p["n_estimators"] * 2)
            return est, f"Retrained with n_estimators {p['n_estimators']} -> {p['n_estimators']*2}."
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=0), "Replaced with a GradientBoostingClassifier (higher capacity)."
    if action == "regularize":
        ch = []
        if "max_depth" in p:
            nd = 8 if p["max_depth"] is None else max(2, p["max_depth"] - 2)
            est.set_params(max_depth=nd); ch.append(f"max_depth -> {nd}")
        if "min_samples_leaf" in p:
            nl = max(10, p["min_samples_leaf"] * 2)
            est.set_params(min_samples_leaf=nl); ch.append(f"min_samples_leaf -> {nl}")
        if not ch:
            return None, "No regularization parameter available for this estimator."
        return est, "Retrained with " + ", ".join(ch) + "."
    return None, "Unknown fix."
