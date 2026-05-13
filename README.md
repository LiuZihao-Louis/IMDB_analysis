# IMDB Movies Dataset Analysis

## Project Overview

This repository supports the university Software Development Workshop II group project:
**"What Makes a Movie Successful?"**

The project currently contains two connected parts:

- **Step 1: Data cleaning, EDA, and visualization**
- **Step 2 extension: Audience-success prediction with machine learning**

Step 1 studies movie success from four perspectives: commercial performance, investment efficiency, audience response, and industry trend. The ML extension then predicts whether a movie is audience-successful using the cleaned metadata.

Recommendation-system and LLM-related features are not included in the current implementation.

## Dataset Files Used

The dataset package contains these CSV files under `data/`:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `links.csv`
- `links_small.csv`
- `ratings.csv`
- `ratings_small.csv`

Step 1 uses `movies_metadata.csv` as the main dataset. Other files are listed in the dataset inventory but are not merged into the Step 1 analysis. They are reserved for later extensions such as richer machine learning and recommendation-system work.

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
|   `-- 03_success_prediction_ml.ipynb
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
|   `-- modeling.py
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

The current ML result should be presented as an exploratory audience-success model, not as a production-level or pure pre-release prediction system.
