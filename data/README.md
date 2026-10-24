# Dataset

`reviews.csv` is generated locally from the `Yelp/yelp_review_full` dataset.
It contains a balanced sample of 30,000 English customer reviews:

- 10,000 `negative` reviews (original ratings 1-2)
- 10,000 `neutral` reviews (original rating 3)
- 10,000 `positive` reviews (original ratings 4-5)

The generated file has two columns:

```text
text,sentiment
```

Generate it with:

```bash
python scripts/create_dataset.py
```

Validate it with:

```bash
python -m src.data_validation --input data/reviews.csv
```

Create cleaned, stratified training and validation files with:

```bash
python -m src.text_preprocessing --input data/reviews.csv
```

This command creates `data/processed/train.csv`,
`data/processed/validation.csv`, and `data/processed/split_metadata.json`.

Create TF-IDF features with:

```bash
python -m src.tfidf_transformation
```

This command learns the vocabulary from the training split only and writes the
vectorizer, sparse feature matrices, and metadata to `artifacts/features/`.

Train the complete sentiment pipeline with:

```bash
python -m src.model_training
```

This command fits text cleaning, TF-IDF, and Logistic Regression as one
pipeline, then writes the trained model and metadata to `artifacts/models/`.
