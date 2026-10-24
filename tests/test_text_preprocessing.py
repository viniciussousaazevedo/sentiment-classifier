import json

import pandas as pd
import pytest

from src.text_preprocessing import clean_text, clean_text_batch, preprocess_dataset


def write_dataset(tmp_path, rows):
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows, columns=["text", "sentiment"]).to_csv(path, index=False, encoding="utf-8")
    return path


def test_clean_text_normalizes_without_removing_sentiment_signals():
    text = "  AMAZING\tservice! Visit HTTPS://example.com <b>NOW</b> &amp; enjoy.  "

    assert clean_text(text) == "amazing service! visit <url> now & enjoy."


def test_clean_text_batch_supports_pipeline_input():
    assert clean_text_batch(["EXCELLENT\tSERVICE", "Poor delivery"]) == [
        "excellent service",
        "poor delivery",
    ]


def test_preprocess_dataset_creates_reproducible_stratified_split(tmp_path):
    rows = []
    for sentiment in ("negative", "neutral", "positive"):
        for index in range(10):
            rows.append((f"{sentiment} review number {index}", sentiment))

    input_path = write_dataset(tmp_path, rows)
    train_path = tmp_path / "processed" / "train.csv"
    validation_path = tmp_path / "processed" / "validation.csv"
    metadata_path = tmp_path / "processed" / "metadata.json"

    metadata = preprocess_dataset(
        input_path,
        train_path,
        validation_path,
        metadata_path,
        validation_size=0.2,
        random_seed=42,
    )

    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)

    assert len(train_frame) == 24
    assert len(validation_frame) == 6
    assert metadata["train_distribution"] == {"negative": 8, "neutral": 8, "positive": 8}
    assert metadata["validation_distribution"] == {"negative": 2, "neutral": 2, "positive": 2}
    assert set(train_frame["cleaned_text"]).isdisjoint(validation_frame["cleaned_text"])
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["random_seed"] == 42


def test_preprocess_dataset_rejects_text_that_becomes_empty(tmp_path):
    rows = [
        ("<b> </b>", "positive"),
        ("Average item.", "neutral"),
        ("Broken item.", "negative"),
    ]
    input_path = write_dataset(tmp_path, rows)

    with pytest.raises(ValueError, match="empty text"):
        preprocess_dataset(
            input_path,
            tmp_path / "train.csv",
            tmp_path / "validation.csv",
            tmp_path / "metadata.json",
        )
