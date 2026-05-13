# IMDB Movies Dataset Analysis

## Project Overview

This repository supports the university Software Development Workshop II group project:
**"What Makes a Movie Successful?"**

The project currently contains three connected parts:

- **Step 1: Data cleaning, EDA, and visualization**
- **Step 2 extension: Audience-success prediction with machine learning**
- **Step 3 extension: Content-based movie recommendation**

Step 1 studies movie success from four perspectives: commercial performance, investment efficiency, audience response, and industry trend. The ML extension predicts whether a movie is audience-successful using the cleaned metadata. The recommendation module recommends content-similar movies using genres, overview, and keywords.

LLM-related features are not included in the current implementation.

## Dataset Files Used

The dataset package contains these CSV files under `data/`:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `links.csv`
- `links_small.csv`
- `ratings.csv`
- `ratings_small.csv`

Step 1 uses `movies_metadata.csv` as the main dataset. The recommendation module additionally uses `keywords.csv` together with movie metadata. In this workspace, `data/movies_metadata.csv` is malformed/truncated around row 2804, so the recommender automatically uses `outputs/cleaned_movies.csv`, the full Step 1 output generated from `movies_metadata.csv`, for movie metadata coverage. Ratings files are reserved for later collaborative-filtering work.

## Folder Structure

```text
IMDB_analysis/
|-- data/
|   |-- movies_metadata.csv
|   |-- credits.csv
|   |-- keywords.csv
|   |-- links.csv
|   |-- links_small.csv
|   |-- ratings.csv
|   `-- ratings_small.csv
|-- notebooks/
|   |-- 01_data_analysis.ipynb
|   |-- 02_rating_and_trend.ipynb
|   |-- 03_success_prediction_ml.ipynb
|   `-- 04_movie_recommendation.ipynb
|-- outputs/
|   |-- cleaned_movies.csv
|   |-- financial_movies.csv
|   |-- data_analysis_summary.md
|   |-- ml_model_metrics.csv
|   |-- ml_cross_validation_metrics.csv
|   |-- ml_feature_importance.csv
|   |-- ml_popularity_comparison.csv
|   |-- ml_success_prediction_summary.md
|   |-- ml_threshold_tuning.csv
|   |-- recommendation_summary.md
|   |-- sample_recommendations.csv
|   |-- figures/
|   |   |-- missing_values.png
|   |   |-- budget_vs_revenue.png
|   |   |-- budget_vs_profit.png
|   |   |-- roi_by_budget_group.png
|   |   |-- revenue_by_genre.png
|   |   |-- rating_by_genre.png
|   |   |-- roi_by_genre.png
|   |   |-- roi_boxplot_by_genre.png
|   |   |-- rating_vs_revenue.png
|   |   |-- vote_count_vs_revenue.png
|   |   |-- popularity_vs_revenue.png
|   |   |-- correlation_heatmap.png
|   |   |-- movies_by_year.png
|   |   |-- revenue_by_year.png
|   |   |-- budget_by_year.png
|   |   |-- rating_by_year.png
|   |   |-- ml_model_comparison.png
|   |   |-- ml_confusion_matrix.png
|   |   |-- ml_feature_importance.png
|   |   |-- ml_popularity_comparison.png
|   |   |-- ml_threshold_tuning.png
|   |   `-- ml_cv_metrics.png
|   `-- models/
|       |-- dummy_baseline.joblib
|       |-- logistic_regression.joblib
|       `-- random_forest.joblib
|-- src/
|   |-- data_cleaning.py
|   |-- visualization.py
|   |-- modeling.py
|   `-- recommender.py
|-- streamlit_app.py
|-- requirements.txt
`-- README.md
```

## Required Python Packages

- Python 3.12+
- pandas
- numpy
- matplotlib
- seaborn
- nbformat
- nbconvert
- ipykernel
- streamlit
- altair
- scikit-learn
- joblib

Install from `requirements.txt`:

```powershell
.conda-env\python.exe -m pip install -r requirements.txt
```

Or create a Conda environment:

```powershell
conda create --prefix .conda-env python=3.12 pandas matplotlib-base seaborn nbformat nbconvert ipykernel streamlit altair scikit-learn joblib
```

## How to Run

Run Step 1 EDA:

```powershell
.conda-env\python.exe -m nbconvert --execute --inplace notebooks\01_data_analysis.ipynb
```

Run the ML extension after Step 1 outputs exist:

```powershell
.conda-env\python.exe -m nbconvert --execute --inplace notebooks\03_success_prediction_ml.ipynb
```

Run the recommendation notebook:

```powershell
.conda-env\python.exe -m nbconvert --execute --inplace notebooks\04_movie_recommendation.ipynb
```

Run the Streamlit dashboard:

```powershell
.conda-env\python.exe -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Generated Outputs

Step 1 outputs:

- `outputs/cleaned_movies.csv`
- `outputs/financial_movies.csv`
- `outputs/data_analysis_summary.md`
- `outputs/figures/missing_values.png`
- `outputs/figures/budget_vs_revenue.png`
- `outputs/figures/budget_vs_profit.png`
- `outputs/figures/roi_by_budget_group.png`
- `outputs/figures/revenue_by_genre.png`
- `outputs/figures/rating_by_genre.png`
- `outputs/figures/roi_by_genre.png`
- `outputs/figures/roi_boxplot_by_genre.png`
- `outputs/figures/rating_vs_revenue.png`
- `outputs/figures/vote_count_vs_revenue.png`
- `outputs/figures/popularity_vs_revenue.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/movies_by_year.png`
- `outputs/figures/revenue_by_year.png`
- `outputs/figures/budget_by_year.png`
- `outputs/figures/rating_by_year.png`

ML extension outputs:

- `outputs/ml_model_metrics.csv`
- `outputs/ml_success_prediction_summary.md`
- `outputs/ml_popularity_comparison.csv`
- `outputs/ml_threshold_tuning.csv`
- `outputs/ml_cross_validation_metrics.csv`
- `outputs/ml_feature_importance.csv`
- `outputs/models/dummy_baseline.joblib`
- `outputs/models/logistic_regression.joblib`
- `outputs/models/random_forest.joblib`
- `outputs/figures/ml_model_comparison.png`
- `outputs/figures/ml_confusion_matrix.png`
- `outputs/figures/ml_feature_importance.png`
- `outputs/figures/ml_popularity_comparison.png`
- `outputs/figures/ml_threshold_tuning.png`
- `outputs/figures/ml_cv_metrics.png`

Recommendation outputs:

- `outputs/recommendation_summary.md`
- `outputs/sample_recommendations.csv`

## Analysis Modules

### 1. Commercial Success Analysis

Uses only movies with valid positive budget and revenue. It studies whether higher investment is associated with higher revenue and profit, and whether high-budget films always produce stronger ROI.

### 2. Genre Performance Analysis

Uses `main_genre` extracted from the metadata genre list. It compares average revenue, average rating, median ROI, and ROI distribution across genres while filtering out genres with too few samples.

### 3. Audience Response Analysis

Compares audience rating, vote count, and popularity against revenue. It also includes a correlation heatmap for success-related variables.

### 4. Time Trend Analysis

Studies movie release volume, average revenue, average budget, and average rating over time. Financial trends use only the valid financial dataset, while release count and rating trends use the cleaned main dataset.

### 5. Audience Success Prediction

Defines `success_movie` as `vote_average >= 7.0` and `vote_count >= 50`. The model excludes `vote_average` and `vote_count` from input features to prevent data leakage.

The ML module includes:

- Dummy baseline for class-imbalance comparison
- Logistic Regression and Random Forest models
- Accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC
- Median imputation with missing indicators for numeric features
- Popularity-feature comparison with and without `popularity`
- Threshold tuning for Random Forest
- Stratified 5-fold cross-validation
- Feature importance and confusion matrix visualizations

The selected Random Forest operating threshold is `0.6`, based on the threshold-tuning result with the strongest F1-score among the tested thresholds.

The model relies heavily on `popularity`, so it should be presented as an exploratory audience-success prediction model using platform attention information, not as a pure pre-release prediction system. The stricter comparison without `popularity` is included to show how predictive performance changes when that attention signal is removed.

### 6. Movie Recommendation System

Uses `movies_metadata.csv` and `keywords.csv` to build a content-based recommendation module. Each movie is represented as:

```text
content = genres + overview + keywords
```

TF-IDF converts the content text into numeric vectors, and cosine similarity ranks movies by content similarity. In Streamlit, users can choose a movie, select Top 5 / Top 10 / Top 15 recommendations, and view recommended movie title, similarity score, vote average, vote count, and main genre.

The recommender checks the raw metadata file before building recommendations. If `data/movies_metadata.csv` is malformed or truncated, it falls back to `outputs/cleaned_movies.csv` so the dashboard can recommend from the full cleaned movie set instead of a partial raw file.

Limitations:

- Content similarity does not guarantee that a user will personally like a movie.
- The first version does not use user history or personalized behavior.
- Cold-start and individual preference problems are not fully solved.
- `ratings_small.csv` can be used later for collaborative filtering.
