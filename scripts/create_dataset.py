"""Create a balanced Yelp review sample for this project."""

from __future__ import annotations

import csv
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "Yelp/yelp_review_full"
SPLIT = "train"
SAMPLES_PER_SENTIMENT = 10_000
RANDOM_SEED = 42
OUTPUT_PATH = Path("data/reviews.csv")
METADATA_PATH = Path("data/dataset_metadata.json")


def rating_to_sentiment(label: int) -> str:
    """Map Yelp's zero-based 1-5 star label to a three-class sentiment."""

    if label in {0, 1}:
        return "negative"
    if label == 2:
        return "neutral"
    if label in {3, 4}:
        return "positive"
    raise ValueError(f"Unexpected Yelp label: {label}")


def create_balanced_sample(
    samples_per_sentiment: int = SAMPLES_PER_SENTIMENT,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Use reservoir sampling to select an even sample without loading all data."""

    if samples_per_sentiment <= 0:
        raise ValueError("Samples per sentiment must be greater than zero.")

    generator = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
    random_generator = random.Random(RANDOM_SEED)
    reservoir_size = samples_per_sentiment + 1_000
    reservoirs: dict[str, list[str]] = {label: [] for label in ("negative", "neutral", "positive")}
    seen = Counter()

    for record in generator:
        sentiment = rating_to_sentiment(record["label"])
        seen[sentiment] += 1
        bucket = reservoirs[sentiment]

        if len(bucket) < reservoir_size:
            bucket.append(record["text"])
            continue

        replacement_index = random_generator.randrange(seen[sentiment])
        if replacement_index < reservoir_size:
            bucket[replacement_index] = record["text"]

    rows: list[dict[str, str]] = []
    selected_texts: set[str] = set()
    for sentiment, texts in reservoirs.items():
        selected_count = 0
        for text in texts:
            normalized_text = text.strip().casefold()
            if not normalized_text or normalized_text in selected_texts:
                continue

            rows.append({"text": text, "sentiment": sentiment})
            selected_texts.add(normalized_text)
            selected_count += 1
            if selected_count == samples_per_sentiment:
                break

        if selected_count < samples_per_sentiment:
            raise RuntimeError(
                f"Not enough unique source examples for {sentiment}: "
                f"expected {samples_per_sentiment}, found {selected_count}."
            )

    random_generator.shuffle(rows)
    return pd.DataFrame(rows), dict(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced Yelp review sample.")
    parser.add_argument(
        "--samples-per-sentiment",
        type=int,
        default=SAMPLES_PER_SENTIMENT,
        help="Number of reviews to sample for each sentiment label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for the generated CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=METADATA_PATH,
        help="Path for generated metadata.",
    )
    args = parser.parse_args()

    frame, source_distribution = create_balanced_sample(args.samples_per_sentiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    metadata = {
        "source_dataset": DATASET_NAME,
        "source_split": SPLIT,
        "random_seed": RANDOM_SEED,
        "samples_per_sentiment": args.samples_per_sentiment,
        "source_distribution_after_mapping": source_distribution,
        "output_distribution": frame["sentiment"].value_counts().sort_index().to_dict(),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Created {args.output} with {len(frame):,} reviews.")
    print("Output distribution:")
    for sentiment, count in metadata["output_distribution"].items():
        print(f"- {sentiment}: {count:,}")


if __name__ == "__main__":
    main()
