"""Search TF-IDF and Logistic Regression settings with cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.model_evaluation import evaluate_model
from src.model_training import _load_training_dataset, create_pipeline


CV_FOLDS = 3
SEARCH_ITERATIONS = 24
RANDOM_SEED = 42
PARALLEL_JOBS = 2

PARAMETER_DISTRIBUTIONS = {
    "tfidf__ngram_range": [(1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3],
    "tfidf__max_df": [0.90, 0.95],
    "tfidf__max_features": [20_000, 30_000],
    "classifier__C": [0.1, 0.3, 1.0, 3.0],
}


def tune_hyperparameters(
    train_input_path: str | Path,
    validation_input_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Select the best configuration by cross-validated macro F1.

    The validation input is not used while choosing parameters. It is evaluated
    once after the winning pipeline is refitted on the complete training split.
    """

    train_frame = _load_training_dataset(train_input_path)
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    search = RandomizedSearchCV(
        estimator=create_pipeline(),
        param_distributions=PARAMETER_DISTRIBUTIONS,
        n_iter=SEARCH_ITERATIONS,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=PARALLEL_JOBS,
        random_state=RANDOM_SEED,
        refit=True,
        return_train_score=True,
        verbose=2,
    )
    search.fit(train_frame["text"], train_frame["sentiment"])

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "tuned_sentiment_pipeline.joblib"
    joblib.dump(search.best_estimator_, model_path)

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(destination / "search_results.csv", index=False, encoding="utf-8")

    evaluation_metrics = evaluate_model(
        model_path=model_path,
        validation_input_path=validation_input_path,
        output_directory=destination / "evaluation",
    )
    metadata: dict[str, object] = {
        "train_input_path": str(train_input_path),
        "validation_input_path": str(validation_input_path),
        "scoring": "f1_macro",
        "cv_folds": CV_FOLDS,
        "search_iterations": SEARCH_ITERATIONS,
        "best_cross_validation_macro_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "validation_accuracy": evaluation_metrics["accuracy"],
        "validation_macro_f1": evaluation_metrics["macro_f1"],
    }
    (destination / "tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF and Logistic Regression settings using macro F1."
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
        help="Path to the reserved validation CSV.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/tuning"),
        help="Directory for the tuned model and experiment artifacts.",
    )
    args = parser.parse_args()

    try:
        metadata = tune_hyperparameters(
            train_input_path=args.train_input,
            validation_input_path=args.validation_input,
            output_directory=args.output_directory,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Hyperparameter tuning failed: {error}")
        return 1

    print("Hyperparameter tuning completed.")
    print(f"Best cross-validation macro F1: {metadata['best_cross_validation_macro_f1']:.4f}")
    print(f"Validation macro F1: {metadata['validation_macro_f1']:.4f}")
    print(f"Validation accuracy: {metadata['validation_accuracy']:.4f}")
    print(f"Best parameters: {metadata['best_parameters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
