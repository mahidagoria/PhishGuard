#!/usr/bin/env python3
"""
Internal one-shot script: regenerate dataset → retrain RF → save artifact + metadata.
Run from project root: python3 scripts/_retrain.py
"""
import sys, json, random, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- Import dataset generator pools ----
from scripts.generate_dataset import (
    LEGITIMATE_SHORT, LEGITIMATE_BASES, HARD_NEGATIVES_LEGIT,
    PHISHING_PATTERNS, HARD_POSITIVES_PHISH, SHORTENERS,
    SUSPICIOUS_TLDS, BRANDS, SUBS,
    random_hex, random_short,
    generate_legitimate, generate_phishing,
)

random.seed(42)
N = 1000
rows = generate_legitimate(N) + generate_phishing(N)
random.shuffle(rows)

DATA_PATH = ROOT / "data" / "url_dataset.csv"
DATA_PATH.parent.mkdir(exist_ok=True)
with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["url", "label"])
    w.writerows(rows)
print(f"[1/3] Dataset: {len(rows)} URLs → {DATA_PATH}")

# ---- Train ----
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
import joblib
from src.feature_extractor import FEATURE_NAMES, extract_features

df = pd.read_csv(DATA_PATH)
X = pd.DataFrame([extract_features(u) for u in df["url"]], columns=FEATURE_NAMES)
y = df["label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

MODELS = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=6, min_samples_split=8, min_samples_leaf=4, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=80, max_depth=10, min_samples_split=6,
        min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.08, subsample=0.85,
        min_samples_split=6, min_samples_leaf=3, random_state=42),
}

results = {}
for name, m in MODELS.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    cv = cross_val_score(m, X_train, y_train, cv=5, scoring="f1").mean()
    results[name] = {
        "accuracy":   round(accuracy_score(y_test, preds) * 100, 2),
        "precision":  round(precision_score(y_test, preds) * 100, 2),
        "recall":     round(recall_score(y_test, preds) * 100, 2),
        "f1_score":   round(f1_score(y_test, preds) * 100, 2),
        "cv_f1_mean": round(cv * 100, 2),
    }
    print(f"        {name:16s}  Acc={results[name]['accuracy']:6.2f}%  "
          f"F1={results[name]['f1_score']:6.2f}%  CV={results[name]['cv_f1_mean']:6.2f}%")

rf = MODELS["RandomForest"]
preds_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, preds_rf)
tn, fp, fn, tp = cm.ravel()
report = classification_report(
    y_test, preds_rf, target_names=["Legitimate", "Phishing"], digits=4)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
artifact = {
    "model": rf, "feature_names": FEATURE_NAMES,
    "model_name": "RandomForest",
    "train_timestamp": pd.Timestamp.now("UTC").isoformat(),
}
joblib.dump(artifact, MODELS_DIR / "phishing_model.joblib")

meta = {
    "model_name": "RandomForest",
    "feature_count": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "dataset_size": len(df),
    "train_test_split": {"train": len(X_train), "test": len(X_test)},
    "hyperparameters": {
        "n_estimators": 80, "max_depth": 10,
        "min_samples_split": 6, "min_samples_leaf": 3, "max_features": "sqrt",
    },
    "benchmarks": results,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_labels": {
        "TN_legit_correct": int(tn),
        "FP_legit_flagged_false_alarm": int(fp),
        "FN_phish_missed": int(fn),
        "TP_phish_caught": int(tp),
    },
}
(MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
print(f"[2/3] Artifact saved → {MODELS_DIR / 'phishing_model.joblib'}")
print(f"      CM: TN={tn} FP={fp} FN={fn} TP={tp}  "
      f"DetRate={round(tp/(tp+fn)*100,1)}%  FARate={round(fp/(fp+tn)*100,1)}%")

# ---- Smoke test ----
import src.predict as _p; _p._artifact = None
from src.predict import predict_url

SMOKE = [
    ("SAFE", "https://www.google.com"),
    ("SAFE", "https://www.gecajmer.ac.in"),
    ("SAFE", "https://github.com"),
    ("SAFE", "https://www.microsoft.com"),
    ("SAFE", "https://docs.python.org/3/"),
    ("HIGH", "http://secure-login-8847.verify-account.tk/signin?user=admin"),
    ("HIGH", "https://www-paypa1-9921.secure-update.ml/login"),
    ("HIGH", "http://192.168.45.12/paypal/verify/update"),
    ("HIGH", "https://netflix-billing-4421.account-suspend.work/update-payment"),
    ("HIGH", "http://banking-hdfc-7731.update-now.cf/secure/login"),
]
print("\n[3/3] Smoke test:")
all_pass = True
for expected, url in SMOKE:
    r = predict_url(url)
    ok = r["risk_level"] == expected or (expected == "SAFE" and r["phishing_probability"] < 50)
    status = "✓" if ok else "✗ FAIL"
    if not ok: all_pass = False
    print(f"  {status}  {r['risk_level']:8s} {r['phishing_probability']:5.1f}%  {url[:60]}")

print()
if all_pass:
    print("All smoke tests passed. Ready for demo.")
else:
    print("Some smoke tests FAILED — check training data coverage.")
