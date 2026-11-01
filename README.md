# ML Model Troubleshooter 

Upload a machine-learning model, validate it, run real checks, evaluate it, diagnose problems,
apply a fix, **compare two models side by side**, monitor it and export a PDF report.
Every number in the app is **measured by running the model** - nothing is hard-coded or simulated.

* **Backend:** Python / Flask (`backend/`) - loads and runs the models (scikit-learn, ONNX)
* **Frontend:** React + Vite (`frontend/`)

## Run it

Terminal 1 - backend (http://127.0.0.1:5000):
```bash
cd backend
python -m venv venv
# Windows PowerShell: .\venv\Scripts\Activate.ps1     Windows CMD: venv\Scripts\activate
# macOS/Linux:       source venv/bin/activate
pip install -r requirements.txt
python app.py
```
The first start trains two demo models (a shallow Decision Tree and a Random Forest) and
creates `backend/data/`.

Terminal 2 - frontend (http://localhost:5173):
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints, click **Get Started**, create an account (stored in a local SQLite file,
passwords hashed) and sign in. To point the frontend at another backend URL, set `VITE_API_URL`
(e.g. in `frontend/.env`: `VITE_API_URL=http://127.0.0.1:5000/api`).

## What each page really does

| Page | Behaviour |
|---|---|
| Get Started | Upload `.pkl` / `.joblib` (scikit-learn, incl. Pipelines) or `.onnx`. The file is **actually loaded**; corrupt/unsupported files are rejected with a reason. Optional labelled CSV gives your own model real metrics. |
| Validate Model | Reads the loaded object: file type, size, detected framework (vs. what you selected), task type, number of input features and their names, probability support, evaluation data. |
| Run Checks | Executes the model: loads, rejects wrong-width input, predicts, validates output shape / probabilities sum to 1, determinism, batch speed, "not the same output for every input". Shows measured values. |
| Model Overview | Accuracy / precision / recall / F1 (or R², MAE, RMSE), ROC AUC, calibration, computed on a hold-out set. Trend = accuracy over 6 sequential hold-out batches. Health is derived from the diagnosed issues. |
| Model Lab | **Make Prediction** (real model output + class probabilities, latency, out-of-range warnings, optional side-by-side with a second model) - **Evaluate Model** (confusion matrix, actual vs predicted counts) - **Compare Versions** (two models on identical rows with deltas, agreement and an exact McNemar significance test). |
| Troubleshooting | Diagnostics computed from the metrics (high error, missed positives, false alarms, imbalance, over/under-fitting, calibration, drift, production errors). "Apply fix & verify" really retrains a new version (built-in demo models) and registers it. |
| Monitor & Logs | Prediction count, average / p95 latency, error rate and per-feature drift (KS test) from logged requests. "Send test requests" pushes real hold-out rows through the model; "drifted" shifts every feature +1 std to prove the drift detector works. |
| Verification | Re-runs every validation and runtime check right now. |
| Fix Verification | Before vs after on the same rows; verdict is *Improvement Verified / Mixed / Regression / No change* - computed, so a fix that doesn't help says so. |
| Reports | Builds a real PDF (model summary, checks, issues, verification, monitoring, logs) from live results. |

## The demo models and data
The two built-in models are trained on a **synthetic** risk dataset (age, BMI, blood pressure,
cholesterol -> Low/High risk) generated from a formula plus noise, so it works out of the box.
It is *not* medical data and the models are *not* medical tools. The baseline tree genuinely
under-fits (about 0.84 accuracy) while the forest reaches about 0.92 - and Troubleshooting finds that
from the measurements.

## Try your own models
```bash
cd backend
python make_sample_models.py     # writes backend/sample_models/ (real breast-cancer data, 8 features)
```
On Get Started upload `cancer_logreg.joblib` with `cancer_holdout.csv` (label column `target`), then
upload `cancer_gb.joblib` too and compare them in Model Lab. For your own model, attach a CSV whose
columns are the features (names must match the model's `feature_names_in_` if it has them, otherwise
the column count must match) plus one label column.

## Honest limitations
* **Security:** `.pkl`/`.joblib` are Python pickles and execute code when loaded. Only upload models you
  trust and run this app locally / in a sandbox.
* Accuracy/precision/recall for **your own** model need a labelled CSV - without one the app tells you
  instead of inventing numbers. Drift needs that CSV too (it is the reference distribution).
* Keras `.h5` is rejected (TensorFlow isn't a dependency) - export to ONNX or scikit-learn.
  ONNX loading is implemented with onnxruntime but was not exercised with a real `.onnx` file during development.
* Automatic fixes retrain only the built-in demo models (the app has their training data). For your
  own models: fix them yourself, upload the new version, compare in Fix Verification.
* Models are shared per account for your uploads; the two demo models are shared by all accounts.
  This is a local-first app, not a hardened multi-tenant service.
