import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api import app as api
from scripts.demo import TEST_URLS


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api.app.config.update(TESTING=True)
        api._demo_seeded = True

    def setUp(self):
        api.scan_logs.clear()
        self.client = api.app.test_client()

    def test_health_and_feature_extraction(self):
        health = self.client.get("/health")
        features = self.client.get(
            "/features", query_string={"url": "https://example.com/login"}
        )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["feature_count"], 22)
        self.assertEqual(features.status_code, 200)
        self.assertEqual(features.get_json()["feature_count"], 22)
        self.assertIn("shannon_entropy", features.get_json()["features"])

    def test_predict_requires_url_and_logs_valid_result(self):
        missing = self.client.post("/predict", json={})
        result = {
            "url": "https://example.com",
            "label": 0,
            "label_text": "legitimate",
            "confidence": 99.0,
            "phishing_probability": 1.0,
            "risk_level": "SAFE",
            "latency_ms": 1.0,
        }
        with patch.object(api, "predict_url", return_value=result):
            valid = self.client.post("/predict", json={"url": result["url"]})

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(self.client.get("/logs").get_json()["count"], 1)

    def test_batch_prediction_is_capped_at_50_urls(self):
        result = {
            "url": "https://example.com",
            "label": 0,
            "label_text": "legitimate",
            "confidence": 99.0,
            "phishing_probability": 1.0,
            "risk_level": "SAFE",
            "latency_ms": 1.0,
        }
        with patch.object(api, "predict_url", return_value=result) as predict:
            response = self.client.post(
                "/predict/batch",
                json={"urls": [f"https://example-{i}.com" for i in range(60)]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 50)
        self.assertEqual(predict.call_count, 50)

    def test_dataset_stats_reports_recorded_source(self):
        rows = [
            {"url": "https://example.com", "label": 0},
            {"url": "http://verify-login.example.xyz/account", "label": 1},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "url_dataset.csv"
            metadata_path = data_path.with_name("dataset_metadata.json")
            metadata_path.write_text(
                json.dumps({
                    "dataset_name": "PhiUSIIL Phishing URL Dataset",
                    "source_file": "source.csv",
                    "source_label_mapping": {
                        "0": "phishing",
                        "1": "legitimate",
                    },
                }),
                encoding="utf-8",
            )
            with (
                patch.object(api, "DATA_PATH", data_path),
                patch.object(api, "_load_dataset_rows", return_value=rows),
            ):
                response = self.client.get("/dataset/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["source_dataset"],
            "PhiUSIIL Phishing URL Dataset",
        )
        self.assertEqual(response.get_json()["total"], 2)

    def test_demo_urls_match_expected_labels(self):
        with patch("src.predict.socket.getaddrinfo", return_value=[]):
            for url, expected in TEST_URLS:
                with self.subTest(url=url):
                    result = api.predict_url(url)
                    expected_label = 1 if expected == "Phishing" else 0
                    self.assertEqual(result["label"], expected_label)


if __name__ == "__main__":
    unittest.main()
