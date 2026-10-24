# Sentiment Classifier

A complete supervised machine learning pipeline that classifies English customer reviews as `negative`, `neutral`, or `positive`.

## Pipeline

```text
Raw reviews
    -> data validation
    -> text cleaning
    -> TF-IDF transformation
    -> Logistic Regression
    -> evaluation and inference
```

The project uses a balanced sample of Yelp reviews. Star ratings are mapped to sentiment labels as follows:

```text
1-2 stars -> negative
3 stars   -> neutral
4-5 stars -> positive
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Workflow

Create the default balanced dataset:

```bash
python scripts/create_dataset.py
```

Validate it:

```bash
python -m src.data_validation --input data/reviews.csv
```

Create reproducible training and validation splits:

```bash
python -m src.text_preprocessing --input data/reviews.csv
```

Create standalone TF-IDF artifacts for inspection:

```bash
python -m src.tfidf_transformation
```

Train the complete pipeline:

```bash
python -m src.model_training
```

Evaluate the trained model:

```bash
python -m src.model_evaluation
```

Search TF-IDF and Logistic Regression settings with cross-validation:

```bash
python -m src.hyperparameter_tuning
```

## Evaluation Metrics

The evaluator reports metrics for each sentiment class and aggregate metrics across all classes.

### Accuracy

Accuracy is the proportion of all predictions that are correct.

```text
accuracy = correct predictions / total predictions
```

### Precision

Precision answers: *when the model predicts a class, how often is it correct?*

```text
precision = true positives / predicted positives
```

### Recall

Recall answers: *among all real examples of a class, how many did the model find?*

```text
recall = true positives / actual positives
```

### F1-score

F1-score combines precision and recall using their harmonic mean.

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

### Macro F1

Macro F1 calculates the F1-score separately for `negative`, `neutral`, and `positive`, then takes their unweighted average.

Each class has the same importance, even if the dataset is unbalanced. This is the main metric used for hyperparameter tuning in this project.

## Alternativa Approaches for this project

| Approach | Text representation | Model | Strengths | Limitations |
| --- | --- | --- | --- | --- |
| Classic baseline | Word counts | Naive Bayes | Very simple and fast | Less accurate with context |
| Current classic approach | TF-IDF | Logistic Regression | Fast, interpretable, and produces probabilities | Understands little context |
| Strong classic approach | TF-IDF | Linear SVM | Often performs very well on text | Probabilities require an extra step |
| Online / large-scale approach | TF-IDF | SGDClassifier | Trains in batches and scales well | More sensitive to hyperparameters |
| Character-based approach | Character n-grams | Logistic Regression / SVM | Handles typos and spelling variations well | Can produce many features |
| Embeddings approach | Semantic vectors | Logistic Regression / SVM / XGBoost | Captures more meaning and synonyms | Requires an additional embedding model |
| Transformer approach | Contextualized tokens | BERT / DistilBERT / RoBERTa | Understands context and negation better | Heavier and more expensive |
| LLM API approach | Natural-language prompt | Generative model | Requires little or no initial training | Cost, latency, and less predictable behavior |

