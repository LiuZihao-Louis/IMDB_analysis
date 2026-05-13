# IMDB Movie Success Prediction Summary

## Project Objective
This machine learning extension predicts whether a movie is audience-successful using the cleaned movie metadata from Step 1.
The label represents audience response, not financial success.

## Target Definition
A movie is labeled `success_movie = 1` when `vote_average >= 7.0` and `vote_count >= 50`.
`vote_average` and `vote_count` are excluded from model features to prevent data leakage.

## Class Imbalance
Rows used for modeling: 45,349.
Successful movies: 2,502 (5.52%).
Non-successful movies: 42,847.
The Dummy baseline reaches 0.945 accuracy but 0.000 recall and 0.000 F1-score, showing why accuracy is not the main metric.

## Feature Engineering
Features include budget, runtime, popularity, release year, language, main genre, has_budget, is_english, and movie_age.
Numeric missing values are handled with median imputation plus missing indicators, which is important because budget has many invalid or missing values.

## Model Comparison
The best model by F1-score is Random Forest with accuracy 0.916, precision 0.366, recall 0.728, F1-score 0.487, ROC-AUC 0.948, and PR-AUC 0.491.
Random Forest achieves precision 0.366, recall 0.728, F1-score 0.487, and PR-AUC 0.491.
This means it can identify many audience-successful movies, but precision is still limited because successful movies are rare.

## Popularity Feature Boundary
The model with popularity uses platform attention information, so it should not be described as a pure pre-release prediction model.
With popularity: F1 0.487, PR-AUC 0.491.
Without popularity: F1 0.323, PR-AUC 0.254.
The stricter model without popularity is useful as a conservative comparison.

## Threshold Tuning
Among the tested thresholds, 0.6 gives the highest F1-score (0.516).
Lower thresholds favor recall, while higher thresholds favor precision.

## Cross-Validation
Stratified 5-fold cross-validation for Random Forest gives mean F1 0.494 +/- 0.006 and mean PR-AUC 0.509 +/- 0.011.
This reduces dependence on a single train/test split.

## Feature Importance
Top Random Forest features: popularity, runtime, missingindicator_budget, movie_age, release_year.
Feature importance should be read as model association, not causal proof.

## Limitations
The success label depends on chosen rating and vote-count thresholds.
The dataset is highly imbalanced, so precision remains limited.
Popularity may include post-release attention, so the model is exploratory rather than production-level or pure pre-release prediction.
Budget is missing for many movies, and imputation cannot fully recover true budget information.

## Generated Files
- outputs/ml_model_metrics.csv
- outputs/ml_popularity_comparison.csv
- outputs/ml_threshold_tuning.csv
- outputs/ml_cross_validation_metrics.csv
- outputs/ml_feature_importance.csv
- outputs/models/dummy_baseline.joblib
- outputs/models/logistic_regression.joblib
- outputs/models/random_forest.joblib
- outputs/figures/ml_model_comparison.png
- outputs/figures/ml_confusion_matrix.png
- outputs/figures/ml_feature_importance.png
- outputs/figures/ml_popularity_comparison.png
- outputs/figures/ml_threshold_tuning.png
- outputs/figures/ml_cv_metrics.png