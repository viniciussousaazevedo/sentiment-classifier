import pandas as pd
import pytest

from src.data_validation import validate_dataset


def write_csv(tmp_path, rows, columns=("text", "sentiment")):
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8")
    return path


def test_valid_dataset_passes_validation(tmp_path):
    path = write_csv(
        tmp_path,
        [
            ("Excellent service.", "positive"),
            ("The order was acceptable.", "neutral"),
            ("The item arrived damaged.", "negative"),
        ],
    )

    report = validate_dataset(path)

    assert report.is_valid
    assert report.class_distribution == {"negative": 1, "neutral": 1, "positive": 1}


def test_invalid_content_is_reported(tmp_path):
    path = write_csv(
        tmp_path,
        [
            ("", "positive"),
            ("Works fine.", "good"),
            ("Works fine.", "negative"),
            ("Works fine.", "negative"),
        ],
    )

    report = validate_dataset(path)

    assert not report.is_valid
    assert report.empty_text_rows == 1
    assert report.invalid_sentiments == {"good": 1}
    assert report.exact_duplicate_rows == 1
    assert report.conflicting_duplicate_texts == 1


def test_missing_required_column_fails(tmp_path):
    path = write_csv(tmp_path, [("Excellent service.",)], columns=("text",))

    with pytest.raises(ValueError, match="sentiment"):
        validate_dataset(path)
