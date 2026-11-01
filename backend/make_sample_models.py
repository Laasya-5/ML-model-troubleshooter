"""Creates sample files you can upload to try the full workflow with your own files:

    python make_sample_models.py

Output (in ./sample_models):
  cancer_logreg.joblib   - scikit-learn Pipeline (scaler + logistic regression), 8 named features
  cancer_gb.joblib       - gradient boosting classifier on the same 8 features
  cancer_holdout.csv     - labelled hold-out rows (column "target": 1 = benign, 0 = malignant)

Upload cancer_logreg.joblib together with cancer_holdout.csv on the Get Started page, upload
cancer_gb.joblib too (it can reuse the same CSV), then compare them in Model Lab > Compare Versions.
Data: scikit-learn's bundled Wisconsin breast-cancer dataset (a real dataset, for demonstration only).
"""
from pathlib import Path
import joblib, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLS = ["mean radius", "mean texture", "mean smoothness", "mean compactness",
        "mean concavity", "mean symmetry", "worst radius", "worst texture"]
out = Path(__file__).parent / "sample_models"; out.mkdir(exist_ok=True)
d = load_breast_cancer(as_frame=True)
X, y = d.data[COLS], d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
gb = GradientBoostingClassifier(random_state=0, n_estimators=60, max_depth=2).fit(Xtr, ytr)
joblib.dump(lr, out / "cancer_logreg.joblib"); joblib.dump(gb, out / "cancer_gb.joblib")
df = Xte.copy(); df["target"] = yte.values; df.to_csv(out / "cancer_holdout.csv", index=False)
print("Wrote", [p.name for p in out.iterdir()])
print("holdout accuracy  logreg: %.3f   gb: %.3f" % (lr.score(Xte, yte), gb.score(Xte, yte)))
