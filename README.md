# IMDB Movies Dataset Analysis

## Project Overview

This repository contains Step 1 of the university Software Development Workshop II group project: data cleaning, exploratory data analysis, and visualization for the theme **"What Makes a Movie Successful?"**

In this step, movie success is studied from four perspectives:

- Commercial performance
- Investment efficiency
- Audience response
- Industry trend

Machine learning, recommendation systems, and LLM-related features are intentionally not included in Step 1.

## Dataset Files Used in Step 1

The dataset package contains several CSV files under `data/`:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`
- `links.csv`
- `links_small.csv`
- `ratings.csv`
- `ratings_small.csv`

Step 1 uses `movies_metadata.csv` as the main analysis dataset. The other CSV files are only listed in the dataset inventory and are not merged into the analysis. They are not ignored; files such as `credits.csv`, `keywords.csv`, and `ratings_small.csv` are reserved for later project extensions such as machine learning and recommendation-system work.

## Folder Structure

```text
IMDB_analysis/
├── data/
│   ├── movies_metadata.csv
│   └── other dataset CSV files
├── notebooks/
│   └── 01_data_analysis.ipynb
├── outputs/
│   ├── cleaned_movies.csv
│   ├── data_analysis_summary.md
│   └── figures/
├── src/
│   ├── data_cleaning.py
│   └── visualization.py
└── README.md
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

An isolated Conda environment can be created with:

```powershell
conda create --prefix .conda-env python=3.12 pandas matplotlib-base seaborn nbformat nbconvert ipykernel
```

## How to Run the Notebook

From the project root:

```powershell
.conda-env\python.exe -m nbconvert --execute --inplace notebooks\01_data_analysis.ipynb
```

The notebook is designed to run from top to bottom. It automatically creates required output folders and regenerates the cleaned dataset, summary file, and figures.

## Generated Outputs

- `outputs/cleaned_movies.csv`: cleaned movie-level metadata for Step 1 analysis
- `outputs/financial_movies.csv`: financial-analysis subset where budget and revenue are both positive
- `outputs/data_analysis_summary.md`: presentation-friendly written summary
- `outputs/figures/`: all generated visualization files, including the missing-value data quality chart

## Analysis Modules

### 1. Commercial Success Analysis

Uses only movies with valid positive budget and revenue. It studies whether higher investment is associated with higher revenue and profit, and whether high-budget films always produce stronger ROI.

### 2. Genre Performance Analysis

Uses `main_genre` extracted from the metadata genre list. It compares average revenue, average rating, median ROI, and ROI distribution across genres while filtering out genres with too few samples.

### 3. Audience Response Analysis

Compares audience rating, vote count, and popularity against revenue. It also includes a correlation heatmap for success-related variables.

### 4. Time Trend Analysis

Studies movie release volume, average revenue, average budget, and average rating over time. Financial trends use only the valid financial dataset, while release count and rating trends use the cleaned main dataset.
