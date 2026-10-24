"""Evaluate a trained sentiment classification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data_validation import validate_dataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_validation_dataset(path: str | Path) -> pd.DataFrame:
    report = validate_dataset(path)
    if not report.is_valid:
        details = " ".join(report.errors)
        raise ValueError(f"Validation dataset failed validation: {details}")
    return pd.read_csv(path, dtype="string", encoding="utf-8")


def _save_confusion_matrix(matrix: object, labels: list[str], output_path: Path) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(7, 6))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title("Sentiment Classification Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_model(
    model_path: str | Path,
    validation_input_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Evaluate a persisted model against labeled validation data."""

    pipeline_path = Path(model_path)
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {pipeline_path}")

    validation_frame = _load_validation_dataset(validation_input_path)
    pipeline = joblib.load(pipeline_path)
    labels = pipeline.named_steps["classifier"].classes_.tolist()
    actual = validation_frame["sentiment"]
    predicted = pipeline.predict(validation_frame["text"])

    accuracy = float(accuracy_score(actual, predicted))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, average="weighted", zero_division=0
    )
    matrix_array = confusion_matrix(actual, predicted, labels=labels)
    matrix = matrix_array.tolist()
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0
    )

    metrics: dict[str, object] = {
        "model_path": str(model_path),
        "validation_input_path": str(validation_input_path),
        "validation_rows": len(validation_frame),
        "labels": labels,
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": matrix,
    }

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (destination / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _save_confusion_matrix(matrix_array, labels, destination / "confusion_matrix.png")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained sentiment classification pipeline."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/models/sentiment_pipeline.joblib"),
        help="Path to the trained pipeline.",
    )
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=Path("data/processed/validation.csv"),
        help="Path to the processed validation CSV.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Directory for evaluation artifacts.",
    )
    args = parser.parse_args()

    try:
        metrics = evaluate_model(
            model_path=args.model_path,
            validation_input_path=args.validation_input,
            output_directory=args.output_directory,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Model evaluation failed: {error}")
        return 1

    print("Model evaluation completed.")
    print(f"Validation rows: {metrics['validation_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
