# PhishGuard AI — URL Threat Detection Engine

**Govt. Engineering College, Ajmer** · B.Tech Final Year Project · VII & VIII Semester  
**Domain:** Cybersecurity · Machine Learning · Web Security

---

## Overview

A lightweight phishing URL detection engine that classifies URL structure and checks whether the hostname resolves via the system DNS resolver. It does not request or open the submitted site.

The system extracts 22 structural, lexical, and entropy signals from a URL and benchmarks Decision Tree, Random Forest, and Gradient Boosting classifiers. The selected model's measured grouped-holdout accuracy is 99.52%. Prediction latency includes the local system DNS lookup and can vary with resolver behavior.

The included training corpus contains 200,000 balanced URLs prepared from a local PhiUSIIL dataset export. Dataset source, label mapping, and cleaning counts are recorded with the data. This corpus does not claim separate PhishTank or OpenPhish feed ingestion. Low-risk URLs with resolvable hostnames are marked safe; unresolved low-risk domains are undefined, while strong phishing indicators are still flagged.

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

# Optional: regenerate the included balanced dataset from the source CSV
python3 -m scripts.prepare_dataset \
  --source "/path/to/PhiUSIIL_Phishing_URL_Dataset.csv"

# Train ML models
python3 -m src.train_model

# Start API server
python3 src/api/app.py
```

Open `http://127.0.0.1:5000/dashboard` in your browser.

The prepared dataset is included at `data/url_dataset.csv`, so the preparation step is only needed when rebuilding it from the source CSV.

Run API and dataset tests with:

```bash
python3 -m unittest discover -s tests -v
```

After starting the API, run the presentation smoke test with `python3 scripts/demo.py`.

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

Trained on 200,000 balanced URLs (100,000 legitimate and 100,000 phishing) prepared from the locally supplied PhiUSIIL dataset. Evaluation holds out entire hostnames so URLs from one host do not occur in both train and test partitions.

| Classifier | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Decision Tree | 99.46% | 99.95% | 98.96% | 99.45% |
| Random Forest | 99.52% | 99.92% | 99.11% | 99.51% |
| **Gradient Boosting** | **99.52%** | **99.88%** | **99.15%** | **99.51%** |

The selected model achieved 99.15% phishing recall and a 0.12% false-positive rate on 39,412 held-out URLs from 36,969 hostnames. Metrics describe this dataset and split and are not a guarantee of future performance. Some path-heavy benign URLs are false positives; see [the Semester VII report](docs/semester_vii_report.md) for limitations and methodology.

## Project Structure

```
phishing-detection-engine/
├── data/
│   ├── url_dataset.csv          # Clean balanced training data (200,000 URLs)
│   └── dataset_metadata.json    # Source, mapping, and cleaning record
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
│   ├── prepare_dataset.py       # Clean/balance a PhiUSIIL source CSV
│   └── demo.py                  # CLI demo script
├── tests/
│   ├── test_api.py
│   └── test_dataset_pipeline.py
├── docs/
│   └── semester_vii_report.md   # Project-I scope, methods, results, limitations
└── requirements.txt
```

## Tech Stack

- **Python 3.10+** — Feature engineering & ML pipeline
- **scikit-learn** — Random Forest, Decision Tree, Gradient Boosting
- **Flask + Flask-CORS** — Local REST API
- **Chart.js + Tailwind CSS** — Dashboard UI

---

*Local model inference · No submitted webpage is opened*
