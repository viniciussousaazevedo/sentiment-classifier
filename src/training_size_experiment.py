"""Measure how training-set size affects the tuned sentiment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_validation import validate_dataset


RANDOM_SEED = 42
TEST_SAMPLES_PER_CLASS = 4_000
TRAIN_SAMPLES_PER_CLASS = (8_000, 12_000, 16_000)


def _load_dataset(path: str | Path) -> pd.DataFrame:
    report = validate_dataset(path)
    if not report.is_valid:
        details = " ".join(report.errors)
        raise ValueError(f"Experiment dataset failed validation: {details}")
    return pd.read_csv(path, dtype="string", encoding="utf-8")


def _create_fixed_test_set(
    frame: pd.DataFrame, labels: list[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    required_per_class = TEST_SAMPLES_PER_CLASS + max(TRAIN_SAMPLES_PER_CLASS)
    test_parts: list[pd.DataFrame] = []
    training_pools: dict[str, pd.DataFrame] = {}

    for offset, label in enumerate(labels):
        label_frame = frame[frame["sentiment"] == label]
        if len(label_frame) < required_per_class:
            raise ValueError(
                f"Class '{label}' needs at least {required_per_class} rows, "
                f"but contains {len(label_frame)}."
            )
        shuffled = label_frame.sample(frac=1, random_state=RANDOM_SEED + offset).reset_index(
            drop=True
        )
        test_parts.append(shuffled.iloc[:TEST_SAMPLES_PER_CLASS])
        training_pools[label] = shuffled.iloc[TEST_SAMPLES_PER_CLASS:]

    test_frame = pd.concat(test_parts, ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    return test_frame, training_pools


def run_training_size_experiment(
    input_path: str | Path,
    model_path: str | Path,
    output_directory: str | Path,
) -> list[dict[str, object]]:
    """Train nested sample sizes and evaluate them against one fixed test set."""

    frame = _load_dataset(input_path)
    template_pipeline = joblib.load(model_path)
    labels = template_pipeline.named_steps["classifier"].classes_.tolist()
    test_frame, training_pools = _create_fixed_test_set(frame, labels)

    results: list[dict[str, object]] = []
    for samples_per_class in TRAIN_SAMPLES_PER_CLASS:
        training_frame = pd.concat(
            [training_pools[label].iloc[:samples_per_class] for label in labels],
            ignore_index=True,
        ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        pipeline = clone(template_pipeline)
        pipeline.fit(training_frame["text"], training_frame["sentiment"])
        predictions = pipeline.predict(test_frame["text"])
        _, _, f1_scores, _ = precision_recall_fscore_support(
            test_frame["sentiment"], predictions, labels=labels, zero_division=0
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            test_frame["sentiment"], predictions, labels=labels, average="macro", zero_division=0
        )
        result: dict[str, object] = {
            "train_samples_per_class": samples_per_class,
            "train_rows": len(training_frame),
            "test_rows": len(test_frame),
            "accuracy": float(accuracy_score(test_frame["sentiment"], predictions)),
            "macro_f1": float(macro_f1),
        }
        result.update({f"f1_{label}": float(score) for label, score in zip(labels, f1_scores)})
        results.append(result)

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(destination / "training_size_results.csv", index=False)
    metadata = {
        "input_path": str(input_path),
        "model_template_path": str(model_path),
        "test_samples_per_class": TEST_SAMPLES_PER_CLASS,
        "train_samples_per_class": list(TRAIN_SAMPLES_PER_CLASS),
        "random_seed": RANDOM_SEED,
    }
    (destination / "training_size_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fixed training sizes using a shared holdout test set."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to a balanced source CSV.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/tuning/tuned_sentiment_pipeline.joblib"),
        help="Path to the tuned pipeline used as an unfitted template.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/training_size_experiment"),
        help="Directory for experiment results.",
    )
    args = parser.parse_args()

    try:
        results = run_training_size_experiment(
            input_path=args.input,
            model_path=args.model_path,
            output_directory=args.output_directory,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Training size experiment failed: {error}")
        return 1

    print("Training size experiment completed.")
    for result in results:
        print(
            f"{result['train_rows']} training rows: "
            f"accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
