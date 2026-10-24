"""Train and persist the complete sentiment classification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data_validation import validate_dataset
from src.tfidf_transformation import create_tfidf_vectorizer
from src.text_preprocessing import clean_text_batch


MAX_ITERATIONS = 1_000
RANDOM_SEED = 42


def create_pipeline() -> Pipeline:
    """Create an unfitted pipeline for raw customer review text."""

    return Pipeline(
        steps=[
            ("text_cleaner", FunctionTransformer(clean_text_batch, validate=False)),
            ("tfidf", create_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_SEED),
            ),
        ]
    )


def _load_training_dataset(path: str | Path) -> pd.DataFrame:
    report = validate_dataset(path)
    if not report.is_valid:
        details = " ".join(report.errors)
        raise ValueError(f"Training dataset failed validation: {details}")

    frame = pd.read_csv(path, dtype="string", encoding="utf-8")
    return frame


def _distribution(frame: pd.DataFrame) -> dict[str, int]:
    return frame["sentiment"].value_counts().sort_index().astype(int).to_dict()


def train_model(
    train_input_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Train the pipeline on labeled reviews and write local model artifacts."""

    train_frame = _load_training_dataset(train_input_path)
    pipeline = create_pipeline()
    pipeline.fit(train_frame["text"], train_frame["sentiment"])

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination / "sentiment_pipeline.joblib")

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    metadata: dict[str, object] = {
        "train_input_path": str(train_input_path),
        "train_rows": len(train_frame),
        "train_distribution": _distribution(train_frame),
        "classes": classifier.classes_.tolist(),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "max_iterations": MAX_ITERATIONS,
        "random_seed": RANDOM_SEED,
        "pipeline_steps": list(pipeline.named_steps),
    }
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the complete TF-IDF and Logistic Regression pipeline."
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to the processed training CSV.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory for the trained pipeline and its metadata.",
    )
    args = parser.parse_args()

    try:
        metadata = train_model(args.train_input, args.output_directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"Model training failed: {error}")
        return 1

    print("Model training completed.")
    print(f"Training rows: {metadata['train_rows']}")
    print(f"Classes: {', '.join(metadata['classes'])}")
    print(f"Vocabulary size: {metadata['vocabulary_size']}")
    print(f"Pipeline steps: {', '.join(metadata['pipeline_steps'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
