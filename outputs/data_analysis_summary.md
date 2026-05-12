# IMDB Movies Dataset Analysis: Step 1 Summary

## Project Objective

This Step 1 analysis studies **what factors are associated with movie success** using data cleaning, exploratory data analysis, and visualization. Success is considered through commercial performance, investment efficiency, audience response, and industry trends. Machine learning, recommendation systems, and LLM-related features are not included in this step.

## Dataset Overview

- Detected CSV files: credits.csv, keywords.csv, links.csv, links_small.csv, movies_metadata.csv, ratings.csv, ratings_small.csv
- Main dataset: `movies_metadata.csv`
- Shape of `movies_metadata.csv`: 45,466 rows x 24 columns
- Rows belonging to duplicated movie-id groups: 62
- Invalid budget rows for financial analysis: 36,576
- Invalid revenue rows for financial analysis: 38,058
- Rows usable for financial analysis before duplicate removal: 5,381
- Malformed/non-numeric movie id rows: 3
- Other CSV files are inventoried only in Step 1 and reserved for later extensions such as machine learning and recommendation-system work.

### Main Columns in `movies_metadata.csv`

`adult`, `belongs_to_collection`, `budget`, `genres`, `homepage`, `id`, `imdb_id`, `original_language`, `original_title`, `overview`, `popularity`, `poster_path`, `production_companies`, `production_countries`, `release_date`, `revenue`, `runtime`, `spoken_languages`, `status`, `tagline`, `title`, `video`, `vote_average`, `vote_count`

### Key Column Data Types

- `id`: str
- `title`: str
- `budget`: str
- `revenue`: float64
- `genres`: str
- `release_date`: str
- `original_language`: str
- `runtime`: float64
- `popularity`: str
- `vote_average`: float64
- `vote_count`: float64

### Missing Values in Key Columns

- `id`: 0
- `title`: 6
- `budget`: 0
- `revenue`: 6
- `genres`: 0
- `release_date`: 87
- `original_language`: 11
- `runtime`: 263
- `popularity`: 5
- `vote_average`: 6
- `vote_count`: 6

## Cleaning Summary

- Raw rows: 45,466
- Cleaned rows: 45,433
- Malformed rows removed: 3
- Duplicate rows removed: 30
- Duplicate note: 62 rows belong to duplicated movie-id groups, and 30 redundant rows were removed during cleaning.
- Rows in valid financial dataset after cleaning: 5,375
- Financial-analysis subset saved to: `outputs/financial_movies.csv`
- Zero or negative budget/revenue values were not treated as true financial values.
- Non-financial analysis keeps as many valid movie rows as possible.

## Feature Engineering Summary

Created features include `release_year`, `genre_list`, `main_genre`, `profit`, `roi`, `budget_group`, and `rating_group`. Profit and ROI are calculated only where both budget and revenue are positive.

## Finding 1: Commercial Success

Higher-budget movies tend to achieve higher revenue. The correlation between budget and revenue is 0.73, while the budget-profit correlation is 0.58. However, the ROI comparison by budget group shows that high-budget movies do not automatically produce stronger investment efficiency.

## Finding 2: Genre Performance

Genres perform differently depending on the success definition. In the filtered tables, `Animation` shows strong average revenue and `Animation` shows strong average rating, but this depends on the `main_genre` extraction method and sample composition. ROI is sensitive to outliers, so median ROI and the ROI boxplot are used as cautious investment-efficiency indicators rather than absolute genre rankings.

## Finding 3: Audience Response

Audience rating and commercial success are not identical. Among the audience variables, `vote_count` has the strongest correlation with revenue (0.77), which suggests that audience attention is more closely related to revenue than rating score alone.

## Finding 4: Time Trend

The dataset shows changes in release volume and financial scale over time. The highest release-count year in the filtered table is 2014, with 1,973 movies. Average budget and revenue trends use only valid financial rows, while average rating does not follow the same pattern as financial scale.

## Limitations

- Financial analysis is limited to movies with positive budget and revenue.
- Budget and revenue are not inflation-adjusted.
- `main_genre` simplifies multi-genre movies to the first listed genre.
- Recent years may be incomplete in the source dataset.
- Step 1 does not merge credits, keywords, links, or ratings files; these files are reserved for later project extensions.

## Future Work

Future project stages can merge additional dataset files such as credits, keywords, and ratings, add richer feature engineering, and later implement machine learning or recommendation-system components. Those later features are outside Step 1.

## Generated Figures

- `outputs/figures/budget_by_year.png`
- `outputs/figures/budget_vs_profit.png`
- `outputs/figures/budget_vs_revenue.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/missing_values.png`
- `outputs/figures/movies_by_year.png`
- `outputs/figures/popularity_vs_revenue.png`
- `outputs/figures/rating_by_genre.png`
- `outputs/figures/rating_by_year.png`
- `outputs/figures/rating_vs_revenue.png`
- `outputs/figures/revenue_by_genre.png`
- `outputs/figures/revenue_by_year.png`
- `outputs/figures/roi_boxplot_by_genre.png`
- `outputs/figures/roi_by_budget_group.png`
- `outputs/figures/roi_by_genre.png`
- `outputs/figures/vote_count_vs_revenue.png`
