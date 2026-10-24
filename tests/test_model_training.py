import json

import joblib
import pandas as pd

from src.model_training import train_model


def write_training_dataset(tmp_path):
    path = tmp_path / "train.csv"
    rows = [
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
    pd.DataFrame(rows, columns=["text", "sentiment"]).to_csv(path, index=False, encoding="utf-8")
    return path


def test_train_model_saves_complete_pipeline(tmp_path):
    train_path = write_training_dataset(tmp_path)
    output_directory = tmp_path / "models"

    metadata = train_model(train_path, output_directory)

    pipeline = joblib.load(output_directory / "sentiment_pipeline.joblib")
    saved_metadata = json.loads((output_directory / "training_metadata.json").read_text())

    assert list(pipeline.named_steps) == ["text_cleaner", "tfidf", "classifier"]
    assert pipeline.named_steps["text_cleaner"].transform(["EXCELLENT\tSERVICE"]) == [
        "excellent service"
    ]
    assert metadata["classes"] == ["negative", "neutral", "positive"]
    assert saved_metadata["vocabulary_size"] == metadata["vocabulary_size"]
    assert pipeline.predict(["excellent service"])[0] == "positive"
