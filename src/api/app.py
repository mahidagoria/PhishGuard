
    
"""
Flask REST API — Real-Time Phishing URL Detection
Serves ML predictions on localhost:5000
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import csv

from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_features  # noqa: E402
from src.predict import predict_url  # noqa: E402

STATIC_DIR = ROOT / "src" / "api" / "static"
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "url_dataset.csv"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)

scan_logs: list[dict] = []
_demo_seeded = False

DEMO_SEED_URLS = [
    "https://www.google.com",
    "https://docs.python.org/3/",
    "https://github.com/scikit-learn/scikit-learn",
    "http://secure-login-8847.verify-account.tk/signin?user=admin",
    "https://www-paypa1-9921.secure-update.ml/login",
    "http://192.168.45.12/paypal/verify/update",
    "https://netflix-billing-4421.account-suspend.work/update-payment",
    "https://www.microsoft.com/en-in",
    "http://banking-hdfc-7731.update-now.cf/secure/login",
    "https://docs.python.org/3/library/urllib.parse.html",
]


@app.route("/")
def home():
    return redirect("/dashboard")


@app.route("/api")
def api_info():
    return jsonify({
        "project": "Real-Time AI-Powered Phishing & Deceptive URL Detection Engine",
        "institution": "Govt. Engineering College, Ajmer",
        "demo": "Demo 1 — Feature Extraction + ML Training + REST API + Dashboard",
        "dashboard": "/dashboard",
        "endpoints": {
            "GET /dashboard": "Web UI",
            "GET /health": "Health check",
            "POST /predict": "Classify single URL",
            "POST /predict/batch": "Classify multiple URLs",
            "GET /features": "Extract features without prediction",
            "GET /logs": "Recent scan logs",
            "GET /metrics": "API usage metrics",
            "GET /dataset/stats": "Training dataset statistics",
            "GET /dataset/preview": "Sample rows from training dataset",
        },
    })


@app.route("/health")
def health():
    try:
        from src.predict import load_model
        load_model()
        model_status = "loaded"
    except FileNotFoundError:
        model_status = "not_trained"

    return jsonify({
        "status": "ok",
        "model": model_status,
        "feature_count": len(FEATURE_NAMES),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    try:
        result = predict_url(url)
        _log_scan(result)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls", [])

    if not urls or not isinstance(urls, list):
        return jsonify({"error": "Missing 'urls' array in request body"}), 400

    start = time.perf_counter()
    results = []
    for url in urls[:50]:
        try:
            r = predict_url(str(url).strip())
            _log_scan(r)
            results.append(r)
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    elapsed = (time.perf_counter() - start) * 1000
    return jsonify({
        "count": len(results),
        "total_latency_ms": round(elapsed, 3),
        "avg_latency_ms": round(elapsed / max(len(results), 1), 3),
        "results": results,
    })


@app.route("/features")
def features():
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing ?url= parameter"}), 400
    return jsonify({
        "url": url,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "features": extract_features(url),
    })


@app.route("/logs")
def logs():
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify({"count": len(scan_logs), "logs": scan_logs[-limit:][::-1]})


@app.route("/metrics")
def metrics():
    if not scan_logs:
        return jsonify({"total_scans": 0, "message": "No scans yet"})

    phishing = sum(1 for l in scan_logs if l.get("label") == 1)
    latencies = [l["latency_ms"] for l in scan_logs if "latency_ms" in l]
    return jsonify({
        "total_scans": len(scan_logs),
        "phishing_detected": phishing,
        "legitimate_detected": sum(1 for l in scan_logs if l.get("label") == 0),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "min_latency_ms": round(min(latencies), 3) if latencies else 0,
        "max_latency_ms": round(max(latencies), 3) if latencies else 0,
    })


def _log_scan(result: dict) -> None:
    scan_logs.append({
        "url": result.get("url"),
        "label": result.get("label"),
        "label_text": result.get("label_text"),
        "confidence": result.get("confidence"),
        "phishing_probability": result.get("phishing_probability"),
        "risk_level": result.get("risk_level"),
        "latency_ms": result.get("latency_ms"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    if len(scan_logs) > 500:
        scan_logs.pop(0)


@app.route("/dashboard")
def dashboard():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/model-metadata")
def model_metadata():
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        return jsonify({"error": "Model metadata not found. Run training first."}), 404
    data = json.loads(meta_path.read_text())
    return jsonify(data)


@app.route("/sample-urls")
def sample_urls():
    return jsonify({
        "legitimate": [
            "https://www.google.com",
            "https://github.com/scikit-learn/scikit-learn",
            "https://docs.python.org/3/",
            "https://www.microsoft.com/en-in",
            "https://docs.python.org/3/library/urllib.parse.html",
        ],
        "phishing": [
            "http://secure-login-8847.verify-account.tk/signin?user=admin",
            "https://www-paypa1-9921.secure-update.ml/login",
            "http://192.168.45.12/paypal/verify/update",
            "https://netflix-billing-4421.account-suspend.work/update-payment",
            "http://banking-hdfc-7731.update-now.cf/secure/login",
        ],
    })


def _load_dataset_rows() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    rows = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"url": row["url"], "label": int(row["label"])})
    return rows


@app.route("/dataset/stats")
def dataset_stats():
    rows = _load_dataset_rows()
    if not rows:
        return jsonify({"error": "Dataset not found. Run: python scripts/generate_dataset.py"}), 404

    legit = sum(1 for r in rows if r["label"] == 0)
    phish = sum(1 for r in rows if r["label"] == 1)

    tld_counts: dict[str, int] = {"legitimate": 0, "phishing": 0}
    https_legit = https_phish = ip_legit = ip_phish = 0

    for row in rows:
        feats = extract_features(row["url"])
        key = "phishing" if row["label"] == 1 else "legitimate"
        if feats["is_suspicious_tld"]:
            tld_counts[key] += 1
        if feats["has_https"]:
            if row["label"]:
                https_phish += 1
            else:
                https_legit += 1
        if feats["has_ip_address"]:
            if row["label"]:
                ip_phish += 1
            else:
                ip_legit += 1

    avg_len = {
        "legitimate": round(sum(len(r["url"]) for r in rows if r["label"] == 0) / max(legit, 1), 1),
        "phishing": round(sum(len(r["url"]) for r in rows if r["label"] == 1) / max(phish, 1), 1),
    }

    return jsonify({
        "total": len(rows),
        "legitimate": legit,
        "phishing": phish,
        "balance_ratio": f"{legit}:{phish}",
        "sources": ["PhishTank (simulated)", "OpenPhish (simulated)", "Kaggle URL datasets (simulated)"],
        "avg_url_length": avg_len,
        "suspicious_tld_count": tld_counts,
        "https_usage": {"legitimate": https_legit, "phishing": https_phish},
        "ip_based_urls": {"legitimate": ip_legit, "phishing": ip_phish},
        "feature_count": len(FEATURE_NAMES),
    })


@app.route("/dataset/preview")
def dataset_preview():
    rows = _load_dataset_rows()
    if not rows:
        return jsonify({"error": "Dataset not found"}), 404

    label_filter = request.args.get("label")
    limit = min(int(request.args.get("limit", 25)), 100)
    offset = max(int(request.args.get("offset", 0)), 0)

    if label_filter in ("0", "1"):
        rows = [r for r in rows if r["label"] == int(label_filter)]

    page = rows[offset: offset + limit]
    enriched = []
    for row in page:
        feats = extract_features(row["url"])
        enriched.append({
            **row,
            "label_text": "phishing" if row["label"] == 1 else "legitimate",
            "url_length": feats["url_length"],
            "is_suspicious_tld": bool(feats["is_suspicious_tld"]),
            "has_ip_address": bool(feats["has_ip_address"]),
            "shannon_entropy": round(feats["shannon_entropy"], 2),
        })

    return jsonify({
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "rows": enriched,
    })


def seed_demo_scans() -> None:
    global _demo_seeded
    if _demo_seeded or scan_logs:
        return
    try:
        from src.predict import load_model
        load_model()
    except FileNotFoundError:
        return

    for url in DEMO_SEED_URLS:
        try:
            result = predict_url(url)
            _log_scan(result)
        except Exception:
            pass
    _demo_seeded = True


@app.before_request
def _ensure_demo_data():
    if request.path.startswith("/static"):
        return
    seed_demo_scans()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print("\n🛡️  PhishGuard AI — Phishing Detection Engine")
    print("=" * 50)
    print(f"📊  Dashboard:   http://127.0.0.1:{port}/dashboard")
    print(f"🔌  REST API:    http://127.0.0.1:{port}/api")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
