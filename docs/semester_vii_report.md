# Semester VII (Project-I) Work

## Scope

This report covers only the Semester VII deliverables in the project synopsis:

1. Dataset acquisition, cleaning, and preparation.
2. URL feature extraction and model training/benchmarking.
3. Local Flask API implementation and endpoint testing.
4. Project-I documentation and a reproducible mid-term demo path.

Chrome Manifest V3 interception and other Semester VIII work are outside this scope. The existing dashboard is retained; no Semester VIII features were added.

## Dataset

The project dataset contains 200,000 URLs, balanced at 100,000 legitimate and 100,000 phishing examples. It was prepared from the local `PhiUSIIL_Phishing_URL_Dataset.csv` file:

| Cleaning step | Result |
|---|---:|
| Source rows | 235,795 |
| Missing, blank, or invalid-label rows removed | 0 |
| Conflicting URL labels removed | 0 |
| Exact duplicate URLs removed | 425 |
| Unique rows after cleaning | 235,370 |
| Balanced rows retained | 200,000 |

PhiUSIIL uses `0 = phishing` and `1 = legitimate`; the preparation script maps this to the project convention `0 = legitimate` and `1 = phishing`. Sampling is deterministic (`random_state=42`). The source CSV is not bundled; the cleaned dataset and a provenance/cleaning record are in `data/`.

The current corpus is from one local PhiUSIIL source file. It does **not** claim separate, independently collected PhishTank or OpenPhish feeds. If the evaluator requires those named feeds individually, that acquisition and source-level deduplication remains a follow-up.

Rebuild the included dataset with:

```bash
python3 -m scripts.prepare_dataset \
  --source "/path/to/PhiUSIIL_Phishing_URL_Dataset.csv"
```

## Feature extraction and training

`src/feature_extractor.py` calculates 22 lexical, structural, and entropy features directly from each URL. `src/train_model.py` benchmarks Decision Tree, Random Forest, and Gradient Boosting (or XGBoost when importable).

Evaluation uses a deterministic 80/20 `GroupShuffleSplit` grouped by full hostname, so URL variants from one hostname cannot appear on both sides. Five-fold `StratifiedGroupKFold` is used for cross-validation within the training partition.

| Model | Accuracy | Precision | Recall | F1 | 5-fold CV F1 |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 99.46% | 99.95% | 98.96% | 99.45% | 99.45% |
| Random Forest | 99.52% | 99.92% | 99.11% | 99.51% | 99.51% |
| Gradient Boosting (selected) | 99.52% | 99.88% | 99.15% | 99.51% | 99.52% |

The selected model's held-out confusion matrix is:

| | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Actual legitimate | 20,022 | 24 |
| Actual phishing | 164 | 19,202 |

This corresponds to a 99.15% phishing recall and a 0.12% false-positive rate on 39,412 held-out URLs from 36,969 hostnames. Metrics describe this dataset and split; they are not a guarantee of performance on future phishing campaigns. In particular, a URL-only classifier cannot inspect page content, and hostname grouping does not prevent every form of related-domain leakage.

Manual edge-case checks also exposed false positives on some path-heavy benign URLs (for example, a GitHub repository URL, a localized Microsoft URL, and a deep Python documentation URL). Their simpler homepage URLs pass. The demo samples use homepage URLs accordingly; this observed limitation is reported rather than hidden by the aggregate score.

## API and verification

The local Flask service exposes health, single and batch prediction, feature extraction, scan logs, metrics, and dataset inspection endpoints. It runs on `127.0.0.1:5000`.

Validation completed:

- `python3 -m unittest discover -s tests -v` — API and dataset-preparation tests.
- Prediction smoke checks correctly classified Google, GitHub, and Microsoft homepage examples as legitimate, and multiple suspicious examples as phishing. The test suite mocks DNS so it does not depend on network availability.
- After DNS cache warm-up, 100 local `predict_url` calls (including hostname resolution) measured 1.50 ms median, 2.64 ms p95, and 3.24 ms maximum on this machine. A cold DNS lookup can take substantially longer; HTTP/browser overhead is excluded.
- The model artifact and benchmark metadata are generated together in `models/`.

Start the API and run the demo with:

```bash
python3 src/api/app.py
python3 scripts/demo.py
```

Dashboard: `http://127.0.0.1:5000/dashboard`

## Semester boundary and remaining caveat

No browser extension, live-navigation interception, or Semester VIII integration was implemented as part of this work. The original synopsis/progress report records the literature survey as completed; the references supplied there should be independently checked before external publication. Separate PhishTank/OpenPhish feed collection remains the dataset limitation noted above.
