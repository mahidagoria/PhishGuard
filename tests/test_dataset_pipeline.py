import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_dataset import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_cleans_relabels_deduplicates_and_balances(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.csv"
            output = directory / "url_dataset.csv"
            rows = [
                ("https://legit-one.example/", 1),
                ("https://legit-one.example/", 1),
                ("https://legit-two.example/", 1),
                ("https://legit-three.example/", 1),
                ("https://phish-one.example/", 0),
                ("https://phish-two.example/", 0),
                ("https://phish-three.example/", 0),
                ("https://phish-four.example/", 0),
                ("https://conflict.example/", 1),
                ("https://conflict.example/", 0),
                ("   ", 1),
            ]
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["URL", "label"])
                writer.writerows(rows)

            metadata = prepare_dataset(source, output, samples_per_class=2)
            dataset = pd.read_csv(output)
            counts = dataset["label"].value_counts().to_dict()

            self.assertEqual(len(dataset), 4)
            self.assertEqual(counts, {0: 2, 1: 2})
            self.assertNotIn("https://conflict.example/", set(dataset["url"]))
            self.assertEqual(metadata["rows_removed"]["duplicate_urls"], 1)
            self.assertEqual(metadata["rows_removed"]["conflicting_url_labels"], 1)
            saved_metadata = json.loads(
                (directory / "dataset_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_metadata["source_label_mapping"]["0"], "phishing")

    def test_rejects_source_and_output_collision(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "same.csv"
            with self.assertRaisesRegex(ValueError, "must be different"):
                prepare_dataset(path, path, samples_per_class=1)


if __name__ == "__main__":
    unittest.main()
