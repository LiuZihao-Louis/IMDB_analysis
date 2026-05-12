# Movie Success Prediction: ML Module Summary

## Objective

This machine learning module extends the Step 1 EDA from describing movie success factors to predicting whether a movie is successful from an audience-response perspective.

## Target Definition

A movie is labeled as `success_movie = 1` when `vote_average >= 7.0` and `vote_count >= 50`; otherwise it is labeled `0`.

## Dataset

- Source file: `outputs/cleaned_movies.csv`
- Rows used for modeling: 45,349
- Successful movies: 2,502
- Non-successful movies: 42,847
- Positive class rate: 5.52%

## Features

The model uses: budget, runtime, popularity, release_year, original_language, main_genre.

`vote_average` and `vote_count` are excluded from the feature set because they define the target label and would create data leakage.

## Models

- Logistic Regression
- Random Forest

Both models use class balancing to reduce the effect of class imbalance.

## Evaluation Results

```text
              model  accuracy  precision  recall    f1
      Random Forest     0.908      0.348   0.768 0.479
Logistic Regression     0.853      0.250   0.838 0.386
```

Best model by F1-score: **Random Forest**.

## Feature Importance

Top Random Forest features: popularity, runtime, budget, release_year, main_genre_Unknown.

Feature importance should be interpreted as model-based importance, not causal proof.

## Generated Outputs

- `outputs/ml_model_metrics.csv`
- `outputs/models/logistic_regression.joblib`
- `outputs/models/random_forest.joblib`
- `outputs/figures/ml_model_comparison.png`
- `outputs/figures/ml_confusion_matrix.png`
- `outputs/figures/ml_feature_importance.png`

## Limitations

- The label captures audience success, not direct financial success.
- Threshold choices affect the label distribution.
- Budget and some metadata features require imputation.
- `main_genre` simplifies multi-genre movies.
- More models and cross-validation can be added later.
