"""
Offline feature extraction module for URL-based phishing detection.
Extracts 22 lexical, structural, and entropy metrics from URL strings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

# High-risk TLDs commonly abused in phishing campaigns
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    "buzz", "rest", "fit", "cam", "loan", "win", "party", "review",
}

# Known URL shortening / redirect services
SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "cutt.ly", "rb.gy", "shorturl.at",
}

IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"
)


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def count_digits(text: str) -> int:
    return sum(c.isdigit() for c in text)


def has_ip_address(hostname: str) -> int:
    """Return 1 if hostname is an IP address."""
    if not hostname:
        return 0
    hostname = hostname.split(":")[0]
    return 1 if IP_PATTERN.match(hostname) else 0


def get_subdomain_count(hostname: str) -> int:
    if not hostname:
        return 0
    parts = hostname.lower().split(".")
    return max(0, len(parts) - 2)


def get_tld(hostname: str) -> str:
    if not hostname or "." not in hostname:
        return ""
    return hostname.lower().split(".")[-1]


def is_suspicious_tld(hostname: str) -> int:
    return 1 if get_tld(hostname) in SUSPICIOUS_TLDS else 0


def is_shortening_service(hostname: str) -> int:
    if not hostname:
        return 0
    host = hostname.lower().removeprefix("www.")
    return 1 if host in SHORTENING_SERVICES else 0


def has_at_symbol(url: str) -> int:
    """@ in URL often indicates credential-stuffing obfuscation."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return 1 if "@" in (parsed.netloc + parsed.path) else 0


def has_double_slash_redirect(url: str) -> int:
    """Multiple // after protocol may hide real destination."""
    if "://" in url:
        rest = url.split("://", 1)[1]
        return 1 if "//" in rest else 0
    return 0


def has_hex_encoding(url: str) -> int:
    return 1 if re.search(r"%[0-9a-fA-F]{2}", url) else 0


def has_suspicious_keywords(url: str) -> int:
    keywords = [
        "login", "verify", "secure", "account", "update", "confirm",
        "banking", "password", "wallet", "signin", "suspend", "urgent",
    ]
    lower = url.lower()
    return 1 if any(kw in lower for kw in keywords) else 0


FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "digit_ratio",
    "letter_ratio",
    "has_ip_address",
    "subdomain_count",
    "tld_length",
    "shannon_entropy",
    "has_https",
    "is_suspicious_tld",
    "is_shortening_service",
    "has_at_symbol",
    "has_double_slash_redirect",
    "has_hex_encoding",
    "has_suspicious_keywords",
]


def extract_features(url: str) -> dict[str, Any]:
    """
    Extract all URL features for ML classification.
    Returns a dictionary keyed by feature name.
    """
    raw = url.strip()
    if not raw:
        return {name: 0 for name in FEATURE_NAMES}

    normalized = raw if "://" in raw else f"http://{raw}"
    parsed = urlparse(normalized)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_host_path = hostname + path + query

    url_len = len(raw)
    num_digits = count_digits(raw)
    num_letters = sum(c.isalpha() for c in raw)

    features = {
        "url_length": url_len,
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_digits": num_digits,
        "digit_ratio": num_digits / url_len if url_len else 0.0,
        "letter_ratio": num_letters / url_len if url_len else 0.0,
        "has_ip_address": has_ip_address(hostname),
        "subdomain_count": get_subdomain_count(hostname),
        "tld_length": len(get_tld(hostname)),
        "shannon_entropy": shannon_entropy(full_host_path),
        "has_https": 1 if parsed.scheme == "https" else 0,
        "is_suspicious_tld": is_suspicious_tld(hostname),
        "is_shortening_service": is_shortening_service(hostname),
        "has_at_symbol": has_at_symbol(raw),
        "has_double_slash_redirect": has_double_slash_redirect(raw),
        "has_hex_encoding": has_hex_encoding(raw),
        "has_suspicious_keywords": has_suspicious_keywords(raw),
    }
    return features


def extract_feature_vector(url: str) -> list[float]:
    """Return feature values in consistent order for model input."""
    features = extract_features(url)
    return [float(features[name]) for name in FEATURE_NAMES]
