# PhishGuard AI — URL Threat Detection Engine

**Govt. Engineering College, Ajmer** · B.Tech Final Year Project · VII & VIII Semester  
**Domain:** Cybersecurity · Machine Learning · Web Security

---

## Overview

A lightweight, privacy-preserving phishing URL detection engine that classifies any URL as legitimate or phishing in under 10 milliseconds — entirely on-device with no external API calls.

The system extracts 22 structural, lexical, and entropy signals from a URL and runs them through a trained Random Forest classifier, returning a risk score and classification with sub-10ms latency.

## Quick Start

```bash
# Clone / open the project
cd phishing-detection-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Generate training dataset
python3 scripts/generate_dataset.py

# Train ML models
python3 -m src.train_model

# Start API server
python3 src/api/app.py
```

Open `http://127.0.0.1:5000/dashboard` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard` | Web dashboard UI |
| GET | `/health` | Engine health check |
| POST | `/predict` | Classify a single URL |
| POST | `/predict/batch` | Classify up to 50 URLs |
| GET | `/features?url=` | Extract 22 features without prediction |
| GET | `/logs` | Recent scan history |
| GET | `/metrics` | Session performance metrics |
| GET | `/dataset/stats` | Training dataset statistics |
| GET | `/dataset/preview` | Sample rows from training data |
| GET | `/model-metadata` | Model benchmark results |

### Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://secure-login-8847.verify-account.tk/signin"}'
```

```json
{
  "url": "http://secure-login-8847.verify-account.tk/signin",
  "label": 1,
  "label_text": "phishing",
  "confidence": 99.98,
  "phishing_probability": 99.98,
  "risk_level": "HIGH",
  "latency_ms": 2.1
}
```

## 22 Extracted Features

| Category | Features |
|----------|----------|
| Length | url_length, hostname_length, path_length, query_length |
| Character Analysis | num_dots, num_hyphens, num_underscores, num_slashes, num_digits |
| Ratios | digit_ratio, letter_ratio |
| Structural | has_ip_address, subdomain_count, tld_length, has_https |
| Risk Flags | is_suspicious_tld, is_shortening_service, has_at_symbol |
| Obfuscation | has_double_slash_redirect, has_hex_encoding, has_suspicious_keywords |
| Entropy | shannon_entropy |

## Model Performance

Trained on 2,000 balanced URLs (1,000 legitimate · 1,000 phishing) with hard edge-cases included.

| Classifier | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Decision Tree | 94.75% | 93.82% | 95.50% | 94.65% |
| Gradient Boosting | 96.50% | 95.91% | 97.00% | 96.45% |
| **Random Forest** | **97.25%** | **96.80%** | **97.50%** | **97.15%** |

**Deployed model:** Random Forest · False alarm rate: 2% · Detection rate: 96.5%

## Project Structure

```
phishing-detection-engine/
├── data/
│   └── url_dataset.csv          # Training dataset (2,000 URLs)
├── models/
│   ├── phishing_model.joblib    # Trained model artifact
│   └── model_metadata.json      # Benchmark results
├── src/
│   ├── feature_extractor.py     # 22-signal extraction engine
│   ├── train_model.py           # ML training pipeline
│   ├── predict.py               # Inference module
│   └── api/
│       ├── app.py               # Flask REST API
│       └── static/
│           └── index.html       # Web dashboard
├── scripts/
│   ├── generate_dataset.py      # Dataset generator
│   └── demo.py                  # CLI demo script
├── docs/
│   └── DEMO1_PRESENTATION.md    # Demo presentation guide
└── requirements.txt
```

## Tech Stack

- **Python 3.10+** — Feature engineering & ML pipeline
- **scikit-learn** — Random Forest, Decision Tree, Gradient Boosting
- **Flask + Flask-CORS** — Local REST API
- **Chart.js + Tailwind CSS** — Dashboard UI

---

*100% local · Zero cloud cost · Privacy-preserving*
