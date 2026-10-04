# Women's Safety Discussions: Tweet Sentiment Classification

A Python project comparing classical classifiers on sentiment labels generated from a static tweet dataset. The focus is the text-classification pipeline and the limitations of its evaluation.

## Implemented pipeline

The implementation is in [model_training.py](model_training.py), with cleaning in [utils.py](utils.py) and a Flask interface in [app.py](app.py).

1. Load `MeToo_tweets.csv` and normalize column names.
2. Lowercase text; remove URLs, mentions, hashtag markers, punctuation and digits; retain cleaned strings longer than five characters.
3. Apply VADER to cleaned text: compound scores at least 0.05 are positive, at most -0.05 are negative, and the remainder are neutral.
4. Construct TF-IDF features with a maximum of 5,000 features.
5. Make an 80/20 train/test split with `random_state=42`.
6. Compare Logistic Regression, Multinomial Naive Bayes, Linear SVM (`LinearSVC`), Decision Tree and Random Forest.
7. Save classification reports, confusion matrices, comparison plots, the selected model and the vectorizer under `static/`.

The committed training code uses classical models. It does not implement RoBERTa or a transformer/classical hybrid.

## What the results mean

The targets are VADER-generated positive, negative and neutral labels, not human annotations of danger, harassment or unsafe experiences. Classification performance therefore measures agreement with VADER's labeling procedure. It does not establish the ability to assess women's safety or detect real-world emergencies.

The repository's training functions generate metrics when run. This README does not present an independently reproduced accuracy or a de-duplication comparison without corresponding saved experiment records.

## Evaluation limitations

- **Feature leakage:** the current code fits TF-IDF on the complete corpus before splitting. Vocabulary and inverse-document-frequency statistics therefore include test data.
- **Duplicate handling:** the cleaning function does not remove duplicate tweets or group retweets before splitting. Overlap between train and test must be audited.
- **Model selection:** the model with the highest score on the test split is saved as the best model. A separate validation set is needed for selection, followed by evaluation on an untouched test set.
- **Reproducibility:** the split seed is fixed, but the Decision Tree and Random Forest do not set their own random seeds. Some results may vary between runs.
- **Prediction display:** the Flask route uses a hardcoded 85% fallback when a model lacks `predict_proba`. That value is not a measured confidence estimate. Prediction input also bypasses the cleaning function used during training.
- **Scope:** the code works on a CSV dataset; it does not provide a live Twitter stream, validated monthly safety trends or real-time crisis detection.
- **Data interpretation:** sentiment in this dataset should not be treated as a representative estimate of population safety or incident prevalence.

## Run the existing training pipeline

Install the dependencies from [requirements.txt](requirements.txt) in a virtual environment, then run the following from the repository root:

```python
from model_training import load_and_clean_data, perform_eda, apply_vader, train_models

df = load_and_clean_data("MeToo_tweets.csv")
perform_eda(df)
df = apply_vader(df)
train_models(df)
```

This reproduces the current pipeline structure, including its limitations; it is not a corrected evaluation protocol. The Flask routes reference HTML templates that are not included in the current top-level repository listing, so the web interface should not be assumed to run from a fresh checkout.

## Next evaluation steps

1. Record dataset provenance, row counts and class balance.
2. Define exact-duplicate and near-duplicate rules, then keep duplicate groups within a single split.
3. Split raw text before fitting TF-IDF, using a scikit-learn pipeline fitted only on training data.
4. Use a validation split for model choice and an untouched test split for final results.
5. Publish split manifests, software versions, seeds, per-class reports and baseline-versus-deduplicated results.
6. Evaluate against independently annotated human labels before making claims beyond VADER agreement.
7. Remove the placeholder confidence score and apply identical preprocessing at training and inference.

## Presentation

Related poster: **Women's Safety Insights**, Research and Innovation Day, Riyadh Elm University, 15 May 2026.

[View the research poster](https://canva.link/pb826zc6bw8vsse)

The poster represents the presentation-stage work; the limitations above describe the committed Python implementation.

## Author and license

Reda Kaleem. See [LICENSE](LICENSE).
