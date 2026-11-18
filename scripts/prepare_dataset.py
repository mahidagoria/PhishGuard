"""Clean and balance a local PhiUSIIL phishing URL dataset export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LABELS = {0: 1, 1: 0}  # PhiUSIIL: 0=phishing, 1=legitimate


def prepare_dataset(
    source_path: Path,
    output_path: Path,
    samples_per_class: int = 100_000,
    random_state: int = 42,
) -> dict:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Source and output paths must be different")

    df = pd.read_csv(source_path, usecols=["URL", "label"])
    source_rows = len(df)

    df["URL"] = df["URL"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    valid = df["URL"].ne("") & df["label"].isin(SOURCE_LABELS)
    removed_invalid = int((~valid).sum())
    df = df.loc[valid, ["URL", "label"]].copy()
    df["label"] = df["label"].astype(int).map(SOURCE_LABELS)

    label_counts = df.groupby("URL")["label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)
    removed_conflicting = len(conflicting_urls)
    if conflicting_urls:
        df = df[~df["URL"].isin(conflicting_urls)]

    before_deduplication = len(df)
    df = df.drop_duplicates(subset="URL", keep="first")
    removed_duplicates = before_deduplication - len(df)

    counts = df["label"].value_counts().to_dict()
    if any(counts.get(label, 0) < samples_per_class for label in (0, 1)):
        raise ValueError(
            "Not enough unique rows per class for a balanced sample of "
            f"{samples_per_class}: {counts}"
        )

    balanced = pd.concat(
        [
            group.sample(n=samples_per_class, random_state=random_state)
            for _, group in df.groupby("label", sort=True)
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)
    balanced = balanced.rename(columns={"URL": "url"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    balanced.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)

    metadata = {
        "dataset_name": "PhiUSIIL Phishing URL Dataset",
        "source_file": source_path.name,
        "source_rows": source_rows,
        "output_rows": len(balanced),
        "project_label_mapping": {
            "0": "legitimate",
            "1": "phishing",
        },
        "source_label_mapping": {
            "0": "phishing",
            "1": "legitimate",
        },
        "rows_removed": {
            "invalid_url_or_label": removed_invalid,
            "conflicting_url_labels": removed_conflicting,
            "duplicate_urls": removed_duplicates,
        },
        "samples_per_class": samples_per_class,
        "random_state": random_state,
        "preprocessing": [
            "Trim URL whitespace",
            "Remove missing, blank, and non-binary labeled rows",
            "Remove URLs with conflicting labels",
            "Deduplicate exact URLs",
            "Invert the PhiUSIIL label convention to project labels",
            "Deterministically sample equal class counts",
        ],
    }
    metadata_path = output_path.with_name("dataset_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the downloaded PhiUSIIL CSV (must contain URL and label columns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "url_dataset.csv",
        help="Balanced project dataset output path",
    )
    parser.add_argument("--samples-per-class", type=int, default=100_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metadata = prepare_dataset(
        args.source,
        args.output,
        samples_per_class=args.samples_per_class,
        random_state=args.random_state,
    )
    print(f"Prepared {metadata['output_rows']} URLs: {args.output}")
    print(f"  Legitimate: {metadata['samples_per_class']}")
    print(f"  Phishing:   {metadata['samples_per_class']}")
    print(f"  Metadata:   {args.output.with_name('dataset_metadata.json')}")


if __name__ == "__main__":
    main()
