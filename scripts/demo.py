#!/usr/bin/env python3
"""Quick demo script for live presentation — run after starting the API."""

import json
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

import os
BASE = os.environ.get("API_BASE", "http://127.0.0.1:5000")

TEST_URLS = [
    ("https://www.google.com", "Legitimate"),
    ("https://www.github.com", "Legitimate"),
    ("https://www.microsoft.com", "Legitimate"),
    ("http://secure-login-8847.verify-account.tk/signin?user=admin", "Phishing"),
    ("https://www-paypa1-9921.secure-update.ml/login", "Phishing"),
    ("http://192.168.45.12/paypal/verify/update", "Phishing"),
    ("https://netflix-billing-4421.account-suspend.work/update-payment", "Phishing"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    all_passed = True
    section("1. Health Check")
    r = requests.get(f"{BASE}/health", timeout=5)
    print(json.dumps(r.json(), indent=2))

    section("2. Feature Extraction Demo")
    url = TEST_URLS[3][0]
    r = requests.get(f"{BASE}/features", params={"url": url}, timeout=5)
    data = r.json()
    print(f"URL: {url}\n")
    for k, v in data["features"].items():
        print(f"  {k:30s}: {v}")

    section("3. Single URL Prediction")
    for url, expected in TEST_URLS:
        r = requests.post(f"{BASE}/predict", json={"url": url}, timeout=5)
        d = r.json()
        status = "✓" if (
            (expected == "Phishing" and d.get("label") == 1)
            or (expected == "Legitimate" and d.get("label") == 0)
        ) else "✗"
        all_passed = all_passed and status == "✓"
        print(f"{status} [{d.get('risk_level', '?'):6s}] {d.get('latency_ms', 0):6.2f}ms | "
              f"{d.get('phishing_probability', 0):5.1f}% phishing | {url[:55]}")

    section("4. Batch Prediction + Latency Benchmark")
    urls = [u for u, _ in TEST_URLS]
    start = time.perf_counter()
    r = requests.post(f"{BASE}/predict/batch", json={"urls": urls}, timeout=15)
    elapsed = (time.perf_counter() - start) * 1000
    data = r.json()
    print(f"Batch size: {data['count']}")
    print(f"Total latency: {data['total_latency_ms']:.2f} ms")
    print(f"Avg per URL:   {data['avg_latency_ms']:.2f} ms")

    section("5. API Metrics")
    r = requests.get(f"{BASE}/metrics", timeout=5)
    print(json.dumps(r.json(), indent=2))

    print(f"\n{'=' * 60}")
    if not all_passed:
        print("  Demo failed — one or more sample URLs were misclassified.")
        sys.exit(1)
    print("  Demo complete — all sample URLs matched their expected labels.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except requests.ConnectionError:
        print("\n❌ API not running. Start it first:")
        print("   python src/api/app.py\n")
        sys.exit(1)
