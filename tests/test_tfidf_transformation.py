import json

import joblib
import pandas as pd
from scipy.sparse import load_npz

from src.tfidf_transformation import transform_tfidf_features


def write_processed_dataset(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["cleaned_text", "sentiment"]).to_csv(
        path, index=False, encoding="utf-8"
    )
    return path


def test_tfidf_transformation_fits_only_on_training_text(tmp_path):
    train_path = write_processed_dataset(
        tmp_path,
        "train.csv",
        [
            ("excellent delivery", "positive"),
            ("excellent product", "positive"),
            ("poor delivery", "negative"),
            ("poor service", "negative"),
            ("average product", "neutral"),
            ("average service", "neutral"),
        ],
    )
    validation_path = write_processed_dataset(
        tmp_path,
        "validation.csv",
        [
            ("excellent validationonlyword", "positive"),
            ("poor validationonlyword", "negative"),
            ("average validationonlyword", "neutral"),
        ],
    )
    output_directory = tmp_path / "features"

    metadata = transform_tfidf_features(train_path, validation_path, output_directory)

    vectorizer = joblib.load(output_directory / "tfidf_vectorizer.joblib")
    train_matrix = load_npz(output_directory / "train_features.npz")
    validation_matrix = load_npz(output_directory / "validation_features.npz")

    assert "validationonlyword" not in vectorizer.vocabulary_
    assert metadata["train_matrix_shape"] == [6, 6]
    assert metadata["validation_matrix_shape"] == [3, 6]
    assert train_matrix.shape == (6, 6)
    assert validation_matrix.shape == (3, 6)
    assert json.loads((output_directory / "tfidf_metadata.json").read_text())[
        "ngram_range"
    ] == [1, 2]


def test_tfidf_transformation_rejects_overlapping_text(tmp_path):
    train_path = write_processed_dataset(
        tmp_path,
        "train.csv",
        [("excellent delivery", "positive"), ("poor delivery", "negative")],
    )
    validation_path = write_processed_dataset(
        tmp_path,
        "validation.csv",
        [("excellent delivery", "positive"), ("average delivery", "neutral")],
    )

    try:
        transform_tfidf_features(train_path, validation_path, tmp_path / "features")
    except ValueError as error:
        assert "overlapping" in str(error)
    else:
        raise AssertionError("Expected overlapping text validation to fail.")
