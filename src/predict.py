
"""Phishing URL prediction with an UNVERIFIED state."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

from src.feature_extractor import FEATURE_NAMES, extract_feature_vector, extract_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"

PHISHING_THRESHOLD = 0.85
SUSPICIOUS_THRESHOLD = 0.60

_artifact = None


def load_model():
    global _artifact

    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run python -m src.train_model"
            )

        _artifact = joblib.load(MODEL_PATH)

    return _artifact


def normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please enter a URL.")

    url = url.strip()

    if len(url) > 2048:
        raise ValueError("URL is too long.")

    if any(char.isspace() for char in url):
        raise ValueError("URL contains invalid spaces.")

    if "://" not in url:
        url = "https://" + url

    parsed = urlsplit(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    hostname = parsed.hostname

    if not hostname:
        raise ValueError("Invalid URL: hostname missing.")

    try:
        hostname = hostname.encode("idna").decode("ascii").lower()
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL or port.") from exc

    if len(hostname) > 253:
        raise ValueError("Invalid hostname.")

    if hostname != "localhost" and "." not in hostname:
        raise ValueError("Invalid domain format.")

    for part in hostname.rstrip(".").split("."):
        if (
            not part
            or len(part) > 63
            or part.startswith("-")
            or part.endswith("-")
            or not all(c.isalnum() or c == "-" for c in part)
        ):
            raise ValueError("Invalid hostname format.")

    return url


def predict_url(url: str) -> dict:
    start = time.perf_counter()

    url = normalize_url(url)
    hostname = urlsplit(url).hostname

    domain_resolves = True
    try:
        socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        domain_resolves = False

    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact.get("feature_names", FEATURE_NAMES)

    vector_values = extract_feature_vector(url)

    vector_df = pd.DataFrame(
        [vector_values],
        columns=feature_names
    )

    probabilities = model.predict_proba(vector_df)[0]
    classes = list(model.classes_)

    phishing_probability = (
        float(probabilities[classes.index(1)])
        if 1 in classes else 0.0
    )

    if phishing_probability >= PHISHING_THRESHOLD:
        risk_level = "HIGH"
        verdict = "PHISHING"
        label = 1
        label_text = "phishing"

    elif phishing_probability >= SUSPICIOUS_THRESHOLD:
        risk_level = "MEDIUM"
        verdict = "SUSPICIOUS"
        label = 0
        label_text = "suspicious"

    elif not domain_resolves:
        risk_level = "UNDEFINED"
        verdict = "UNDEFINED"
        label = None
        label_text = "undefined"

    else:
        risk_level = "SAFE"
        verdict = "SAFE"
        label = 0
        label_text = "legitimate"

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "url": url,
        "label": label,
        "label_text": label_text,
        "verdict": verdict,
        "confidence": round(
            max(float(p) for p in probabilities) * 100, 2
        ),
        "phishing_probability": round(
            phishing_probability * 100, 2
        ),
        "is_phishing": risk_level == "HIGH",
        "risk_level": risk_level,
        "domain_status": "resolved" if domain_resolves else "unresolved",
        "latency_ms": round(elapsed_ms, 3),
        "features": extract_features(url),
    }


def _risk_level(prob: float) -> str:
    if prob >= PHISHING_THRESHOLD:
        return "HIGH"
    if prob >= SUSPICIOUS_THRESHOLD:
        return "MEDIUM"
    return "SAFE"
