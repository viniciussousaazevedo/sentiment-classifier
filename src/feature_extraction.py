"""Create TF-IDF features from the processed review datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


NGRAM_RANGE = (1, 2)
MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY = 0.95
MAX_FEATURES = 20_000
SUBLINEAR_TERM_FREQUENCY = True


def create_vectorizer() -> TfidfVectorizer:
    """Create the project's standard TF-IDF vectorizer configuration."""

    return TfidfVectorizer(
        lowercase=False,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DOCUMENT_FREQUENCY,
        max_df=MAX_DOCUMENT_FREQUENCY,
        max_features=MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TERM_FREQUENCY,
    )


def _load_processed_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed dataset was not found: {dataset_path}")

    frame = pd.read_csv(dataset_path, dtype="string", encoding="utf-8")
    required_columns = {"cleaned_text", "sentiment"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Processed dataset is missing required column(s): {names}")
    if frame.empty:
        raise ValueError("Processed dataset has no rows.")
    if frame["cleaned_text"].isna().any() or frame["cleaned_text"].str.strip().eq("").any():
        raise ValueError("Processed dataset contains empty cleaned text values.")
    return frame


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    return frame["sentiment"].value_counts().sort_index().astype(int).to_dict()


def extract_features(
    train_input_path: str | Path,
    validation_input_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Fit TF-IDF on training text and transform validation text without leakage."""

    train_frame = _load_processed_dataset(train_input_path)
    validation_frame = _load_processed_dataset(validation_input_path)

    overlap = set(train_frame["cleaned_text"]).intersection(validation_frame["cleaned_text"])
    if overlap:
        raise ValueError("Training and validation sets contain overlapping cleaned text values.")

    vectorizer = create_vectorizer()
    train_matrix = vectorizer.fit_transform(train_frame["cleaned_text"])
    validation_matrix = vectorizer.transform(validation_frame["cleaned_text"])

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, destination / "tfidf_vectorizer.joblib")
    save_npz(destination / "train_features.npz", train_matrix)
    save_npz(destination / "validation_features.npz", validation_matrix)

    metadata: dict[str, object] = {
        "train_input_path": str(train_input_path),
        "validation_input_path": str(validation_input_path),
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "train_distribution": _distribution(train_frame),
        "validation_distribution": _distribution(validation_frame),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_matrix_shape": list(train_matrix.shape),
        "validation_matrix_shape": list(validation_matrix.shape),
        "ngram_range": list(NGRAM_RANGE),
        "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
        "max_document_frequency": MAX_DOCUMENT_FREQUENCY,
        "max_features": MAX_FEATURES,
        "sublinear_term_frequency": SUBLINEAR_TERM_FREQUENCY,
    }
    (destination / "tfidf_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create TF-IDF features from processed sentiment datasets."
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to the processed training CSV.",
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
        default=Path("artifacts/features"),
        help="Directory for the vectorizer, matrices, and metadata.",
    )
    args = parser.parse_args()

    try:
        metadata = extract_features(
            train_input_path=args.train_input,
            validation_input_path=args.validation_input,
            output_directory=args.output_directory,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Feature extraction failed: {error}")
        return 1

    print("TF-IDF feature extraction completed.")
    print(f"Vocabulary size: {metadata['vocabulary_size']}")
    print(f"Training matrix shape: {tuple(metadata['train_matrix_shape'])}")
    print(f"Validation matrix shape: {tuple(metadata['validation_matrix_shape'])}")
    print(f"N-gram range: {tuple(metadata['ngram_range'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
