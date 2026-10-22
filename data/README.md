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
