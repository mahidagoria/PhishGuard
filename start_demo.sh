#!/bin/bash
# One-command setup and start for PhishGuard AI Demo
set -e
cd "$(dirname "$0")"

PY=python3

echo ""
echo "  PhishGuard AI — URL Threat Detection Engine"
echo "  Govt. Engineering College, Ajmer · B.Tech Final Year Project"
echo "  ============================================================"
echo ""

# Step 1 — Virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    $PY -m venv venv
else
    echo "[1/4] Virtual environment — OK"
fi

source venv/bin/activate
PY=python3

# Step 2 — Dependencies
echo "[2/4] Installing dependencies..."
python3 -m pip install -r requirements.txt ss

# Step 3 — Dataset + model
if [ ! -f "data/url_dataset.csv" ] || [ ! -f "models/phishing_model.joblib" ]; then
    echo "[3/4] Generating dataset and training model (~30s)..."
    $PY scripts/generate_dataset.py
    $PY -m src.train_model
else
    echo "[3/4] Dataset and model — OK"
fi

# Step 4 — Smoke test
echo "[4/4] Running smoke test..."
$PY -c "
import sys; sys.path.insert(0,'.')
import src.predict as p; p._artifact = None
from src.predict import predict_url
urls = [('SAFE','https://www.google.com'),('SAFE','https://www.gecajmer.ac.in'),('HIGH','http://secure-login-8847.verify-account.tk/signin')]
ok = all(
    (exp=='SAFE' and predict_url(u)['phishing_probability'] < 50) or
    (exp=='HIGH' and predict_url(u)['risk_level']=='HIGH')
    for exp,u in urls
)
print('  Smoke test:', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
"

echo ""
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  READY — Open in browser:  http://127.0.0.1:5000/dashboard  │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""

$PY src/api/app.py
