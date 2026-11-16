
"""
Phishing Detection Engine
Model training, benchmarking and evaluation pipeline.

Features:
- Removes duplicate URLs before train/test split.
- Removes URLs having conflicting labels.
- Uses stratified train/test split.
- Evaluates a constrained Decision Tree classifier.
- Saves the best model and benchmark metadata.
- Reports actual accuracy, precision, recall and F1 score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier

# Project paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_extractor import FEATURE_NAMES, extract_features

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load, validate and clean URL dataset."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {"url", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain url and label columns."
        )

    df = df[["url", "label"]].copy()

    df = df.dropna(subset=["url", "label"])

    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(
        df["label"], errors="coerce"
    )

    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            "Dataset labels must be 0 (legitimate) "
            "or 1 (phishing)."
        )

    df = df[df["url"] != ""].copy()

    original_count = len(df)

    # Remove URLs which have conflicting labels.
    label_counts = df.groupby("url")["label"].nunique()

    conflicting_urls = label_counts[
        label_counts > 1
    ].index

    if len(conflicting_urls):
        print(
            f"Conflicting URLs removed: {len(conflicting_urls)}"
        )

        df = df[
            ~df["url"].isin(conflicting_urls)
        ].copy()

    # Remove exact duplicate URLs.
    df = df.drop_duplicates(
        subset=["url"],
        keep="first"
    ).reset_index(drop=True)

    removed = original_count - len(df)

    print(f"Original records: {original_count}")
    print(f"Removed records: {removed}")
    print(f"Unique URLs: {len(df)}")

    if df["label"].nunique() != 2:
        raise ValueError(
            "Both legitimate and phishing classes "
            "must be present."
        )

    return df

def build_feature_matrix(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Extract URL features."""

    rows = [
        extract_features(url)
        for url in df["url"]
    ]

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES
    ).astype(float)

def get_models():
    """Create the intentionally constrained deployment model."""
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=2,
            max_features=5,
            splitter="random",
            random_state=42,
        )
    }

def benchmark_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train and evaluate candidate models."""

    models = get_models()

    results = {}

    best_model = None
    best_name = None
    best_f1 = -1

    class_counts = y_train.value_counts()

    cv_folds = min(5, int(class_counts.min()))

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test, predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        cv_f1 = None

        if cv_folds >= 2:
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=42,
            )

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1",
                n_jobs=1,
            )

            cv_f1 = round(
                float(scores.mean()) * 100, 2
            )

        results[name] = {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "cv_f1_mean": cv_f1,
        }

        print(f"Accuracy  : {accuracy * 100:.2f}%")
        print(f"Precision : {precision * 100:.2f}%")
        print(f"Recall    : {recall * 100:.2f}%")
        print(f"F1 Score  : {f1 * 100:.2f}%")

        # Select model by F1, not accuracy alone.
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    return {
        "benchmarks": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "test_predictions": best_model.predict(X_test),
        "y_test": y_test,
    }

def main():

    data_path = ROOT / "data" / "url_dataset.csv"

    models_dir = ROOT / "models"
    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PHISHING DETECTION - MODEL TRAINING")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset(data_path)

    print("\nDataset summary")

    legit_count = int(
        (df["label"] == 0).sum()
    )

    phishing_count = int(
        (df["label"] == 1).sum()
    )

    print(f"Legitimate URLs : {legit_count}")
    print(f"Phishing URLs   : {phishing_count}")
    print(f"Total URLs      : {len(df)}")

    if min(legit_count, phishing_count) < 2:
        raise ValueError(
            "At least two samples per class are required."
        )

    # 2. Feature extraction
    print("\nExtracting URL features...")

    X = build_feature_matrix(df)
    y = df["label"]

    print(f"Feature count: {len(FEATURE_NAMES)}")

    # 3. Train/test split
    # Dataset is already deduplicated.
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print("\nDataset split")

    print(f"Training records: {len(X_train)}")
    print(f"Testing records : {len(X_test)}")

    # 4. Model benchmarking
    outcome = benchmark_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n" + "=" * 60)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 60)

    for name, metrics in outcome["benchmarks"].items():

        print(f"\n{name}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    best_name = outcome["best_model_name"]
    best_model = outcome["best_model"]

    best_metrics = outcome["benchmarks"][best_name]

    print("\nBest model:", best_name)
    print(
        "Actual test accuracy:",
        best_metrics["accuracy"],
        "%"
    )

    # 5. Classification report
    predictions = outcome["test_predictions"]

    report = classification_report(
        outcome["y_test"],
        predictions,
        labels=[0, 1],
        target_names=[
            "Legitimate",
            "Phishing",
        ],
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        outcome["y_test"],
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # 6. Save model
    artifact = {
        "model": best_model,
        "feature_names": FEATURE_NAMES,
        "model_name": best_name,
        "train_timestamp": pd.Timestamp.utcnow().isoformat(),
    }

    model_path = (
        models_dir / "phishing_model.joblib"
    )

    joblib.dump(
        artifact,
        model_path
    )

    # 7. Save metadata
    metadata = {
        "model_name": best_name,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "dataset_size": len(df),

        "train_test_split": {
            "train": len(X_train),
            "test": len(X_test),
        },

        "hyperparameters": {
            "max_depth": best_model.max_depth,
            "max_features": best_model.max_features,
            "splitter": best_model.splitter,
        },

        "benchmarks": outcome["benchmarks"],

        "classification_report": report,

        "confusion_matrix": cm.tolist(),

        "confusion_matrix_labels": {
            "TN_legit_correct": int(tn),
            "FP_legit_flagged_false_alarm": int(fp),
            "FN_phish_missed": int(fn),
            "TP_phish_caught": int(tp),
        },

        "evaluation_note": (
            "Metrics are calculated on a deduplicated "
            "stratified holdout split. They are not a "
            "guarantee of real-world performance. A shallow "
            "randomized decision tree is used to target the "
            "requested approximately 80% accuracy."
        ),
    }

    metadata_path = (
        models_dir / "model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    # 8. Final report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Best model: {best_name}")

    print(
        f"Test accuracy: "
        f"{best_metrics['accuracy']}%"
    )

    print(
        f"Precision: "
        f"{best_metrics['precision']}%"
    )

    print(
        f"Recall: "
        f"{best_metrics['recall']}%"
    )

    print(
        f"F1 Score: "
        f"{best_metrics['f1_score']}%"
    )

    print("\nClassification Report")
    print(report)

    print("Confusion Matrix")
    print(cm)

    fpr = (
        fp / (tn + fp) * 100
        if (tn + fp) > 0
        else 0
    )

    print(f"False Positive Rate: {fpr:.2f}%")

    print("\nSaved model:", model_path)
    print("Saved metadata:", metadata_path)

    print("\nTraining pipeline finished successfully.")

if __name__ == "__main__":
    main()