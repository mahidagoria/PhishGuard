"""Load trained model and classify URLs."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.feature_extractor import FEATURE_NAMES, extract_feature_vector

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"
PHISHING_THRESHOLD = 0.85

_artifact = None


def load_model():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python -m src.train_model"
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def predict_url(url: str) -> dict:
    """
    Classify a single URL.
    Returns label, confidence, risk_score, features, and latency_ms.
    """
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact.get("feature_names", FEATURE_NAMES)

    start = time.perf_counter()
    vector_values = extract_feature_vector(url)

    try:
        import pandas as pd
        vector_df = pd.DataFrame([vector_values], columns=feature_names)
        proba = model.predict_proba(vector_df)[0]
        label = int(model.predict(vector_df)[0])
    except Exception:
        vector = np.array([vector_values])
        proba = model.predict_proba(vector)[0]
        label = int(model.predict(vector)[0])

    elapsed_ms = (time.perf_counter() - start) * 1000

    phishing_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    is_phishing = phishing_prob >= PHISHING_THRESHOLD

    from src.feature_extractor import extract_features

    return {
        "url": url,
        "label": label,
        "label_text": "phishing" if label == 1 else "legitimate",
        "confidence": round(max(proba) * 100, 2),
        "phishing_probability": round(phishing_prob * 100, 2),
        "is_phishing": is_phishing,
        "risk_level": _risk_level(phishing_prob),
        "latency_ms": round(elapsed_ms, 3),
        "features": extract_features(url),
    }


def _risk_level(prob: float) -> str:
    if prob >= 0.85:
        return "HIGH"
    if prob >= 0.60:
        return "MEDIUM"
    if prob >= 0.40:
        return "LOW"
    return "SAFE"
