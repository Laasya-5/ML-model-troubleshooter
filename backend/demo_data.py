"""Synthetic risk dataset + the two demo models (trained for real, with scikit-learn).

The data is SYNTHETIC (generated from a known formula plus noise) so the app works out of the box.
Every number the app shows for these models is measured on a held-out split - nothing is hard-coded.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["age", "bmi", "blood_pressure", "cholesterol"]
FEATURE_LABELS = {"age": "Age", "bmi": "BMI", "blood_pressure": "Blood Pressure", "cholesterol": "Cholesterol"}
CLASS_LABELS = {"0": "Low Risk", "1": "High Risk"}

def make_dataset(n=6000, seed=42):
    r = np.random.default_rng(seed)
    age = r.normal(52, 14, n).clip(18, 90)
    bmi = (r.normal(27.5, 5, n) + 0.02 * (age - 52)).clip(15, 48)
    bp = (100 + 0.35 * (age - 50) + 0.8 * (bmi - 27) + r.normal(0, 12, n) + 18).clip(85, 210)
    chol = (185 + 0.7 * (age - 50) + 1.2 * (bmi - 27) + r.normal(0, 32, n)).clip(110, 360)
    za, zb, zp, zc = (age - 52) / 14, (bmi - 27.5) / 5, (bp - 120) / 17, (chol - 190) / 38
    logit = (-0.55 + 0.9 * za + 0.7 * zb + 1.0 * zp + 0.8 * zc
             + 0.9 * za * zb + 0.7 * np.maximum(zp - 0.5, 0) ** 2 * 2 - 0.6 * (zc < -0.5))
    p = 1 / (1 + np.exp(-2.6 * logit))
    y = (r.random(n) < p).astype(int)
    df = pd.DataFrame({"age": age, "bmi": bmi, "blood_pressure": bp, "cholesterol": chol})
    return df.round(1), y

def split(seed=42):
    X, y = make_dataset()
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

def demo_estimators():
    return [
        ("Baseline Decision Tree (v1)", DecisionTreeClassifier(max_depth=3, random_state=0),
         "Shallow decision tree - fast but limited capacity."),
        ("Random Forest (v2)", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                                      random_state=0, n_jobs=1),
         "Ensemble of 200 trees - higher capacity."),
    ]
