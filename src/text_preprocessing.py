"""Clean review text and create reproducible train/validation splits."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_validation import validate_dataset


DEFAULT_VALIDATION_SIZE = 0.20
DEFAULT_RANDOM_SEED = 42
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply conservative, deterministic text normalization.

    The function preserves punctuation, numbers, emojis, contractions, and
    negations because they can carry sentiment information.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = html.unescape(normalized)
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = URL_PATTERN.sub(" <url> ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip().casefold()


def clean_text_batch(texts: object) -> list[str]:
    """Clean an iterable of texts for use inside a scikit-learn pipeline."""

    return [clean_text(text) for text in texts]


def _validate_processed_frame(frame: pd.DataFrame) -> None:
    if frame["cleaned_text"].eq("").any():
        raise ValueError("Text cleaning produced one or more empty text values.")

    duplicate_count = int(frame.duplicated(subset=["cleaned_text", "sentiment"]).sum())
    if duplicate_count:
        raise ValueError(f"Text cleaning produced {duplicate_count} duplicate row(s).")

    labels_per_text = frame.groupby("cleaned_text")["sentiment"].nunique()
    conflicting_count = int(labels_per_text.gt(1).sum())
    if conflicting_count:
        raise ValueError(
            "Text cleaning produced "
            f"{conflicting_count} text value(s) with conflicting sentiment labels."
        )


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    return frame["sentiment"].value_counts().sort_index().astype(int).to_dict()


def preprocess_dataset(
    input_path: str | Path,
    train_output_path: str | Path,
    validation_output_path: str | Path,
    metadata_output_path: str | Path,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    """Validate, clean, split, and persist a sentiment dataset."""

    if not 0 < validation_size < 1:
        raise ValueError("Validation size must be greater than 0 and less than 1.")

    validation_report = validate_dataset(input_path)
    if not validation_report.is_valid:
        details = " ".join(validation_report.errors)
        raise ValueError(f"Input dataset failed validation: {details}")

    frame = pd.read_csv(input_path, dtype="string", encoding="utf-8")
    frame["cleaned_text"] = frame["text"].map(clean_text).astype("string")
    _validate_processed_frame(frame)

    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_size,
        random_state=random_seed,
        stratify=frame["sentiment"],
    )
    train_frame = train_frame.reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)

    overlap = set(train_frame["cleaned_text"]).intersection(validation_frame["cleaned_text"])
    if overlap:
        raise ValueError("Training and validation sets contain overlapping cleaned text values.")

    for output_path in (train_output_path, validation_output_path, metadata_output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    train_frame.to_csv(train_output_path, index=False, encoding="utf-8")
    validation_frame.to_csv(validation_output_path, index=False, encoding="utf-8")

    metadata: dict[str, object] = {
        "input_path": str(input_path),
        "random_seed": random_seed,
        "validation_size": validation_size,
        "input_rows": len(frame),
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "input_distribution": _distribution(frame),
        "train_distribution": _distribution(train_frame),
        "validation_distribution": _distribution(validation_frame),
    }
    Path(metadata_output_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean review text and create stratified train/validation datasets."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the validated CSV file.")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path for the processed training CSV.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/processed/validation.csv"),
        help="Path for the processed validation CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/processed/split_metadata.json"),
        help="Path for split metadata.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
        help="Fraction of rows reserved for validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed used to make the split reproducible.",
    )
    args = parser.parse_args()

    try:
        metadata = preprocess_dataset(
            input_path=args.input,
            train_output_path=args.train_output,
            validation_output_path=args.validation_output,
            metadata_output_path=args.metadata_output,
            validation_size=args.validation_size,
            random_seed=args.random_seed,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Text preprocessing failed: {error}")
        return 1

    print("Text preprocessing completed.")
    print(f"Input rows: {metadata['input_rows']}")
    print(f"Train rows: {metadata['train_rows']}")
    print(f"Validation rows: {metadata['validation_rows']}")
    print("Train distribution:")
    for label, count in metadata["train_distribution"].items():
        print(f"- {label}: {count}")
    print("Validation distribution:")
    for label, count in metadata["validation_distribution"].items():
        print(f"- {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
