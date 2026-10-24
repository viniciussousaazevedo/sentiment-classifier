import json

import pandas as pd

from src.model_evaluation import evaluate_model
from src.model_training import train_model


def write_dataset(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["text", "sentiment"]).to_csv(path, index=False, encoding="utf-8")
    return path


def test_evaluate_model_writes_metrics_and_confusion_matrix(tmp_path):
    train_rows = [
        ("excellent product", "positive"),
        ("excellent service", "positive"),
        ("excellent delivery", "positive"),
        ("poor product", "negative"),
        ("poor service", "negative"),
        ("poor delivery", "negative"),
        ("average product", "neutral"),
        ("average service", "neutral"),
        ("average delivery", "neutral"),
    ]
    validation_rows = [
        ("excellent quality", "positive"),
        ("poor quality", "negative"),
        ("average quality", "neutral"),
    ]
    train_path = write_dataset(tmp_path, "train.csv", train_rows)
    validation_path = write_dataset(tmp_path, "validation.csv", validation_rows)
    model_directory = tmp_path / "models"
    evaluation_directory = tmp_path / "evaluation"
    train_model(train_path, model_directory)

    metrics = evaluate_model(
        model_directory / "sentiment_pipeline.joblib",
        validation_path,
        evaluation_directory,
    )

    assert metrics["validation_rows"] == 3
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 3
    assert (evaluation_directory / "confusion_matrix.png").is_file()
    assert json.loads((evaluation_directory / "metrics.json").read_text())["labels"] == [
        "negative",
        "neutral",
        "positive",
    ]
