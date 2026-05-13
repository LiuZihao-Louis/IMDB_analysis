# Movie Recommendation System Summary

## Objective
This module recommends content-similar movies when a user selects a movie.
It extends the project from analyzing and predicting movie success to helping users discover related movies.

## Why the IMDB Data Is Suitable
The IMDB metadata contains movie-level content fields such as genres and overview, while keywords.csv provides descriptive keyword tags.
These fields make the dataset suitable for a first content-based recommendation system.

## Metadata Source Check
Raw movies_metadata.csv rows readable in this workspace: 2,803.
Cleaned metadata rows available from Step 1: 45,433.
Metadata source used for recommendation: outputs/cleaned_movies.csv.
The raw movies_metadata.csv file in this workspace is malformed or truncated, so the recommender uses outputs/cleaned_movies.csv generated from movies_metadata.csv for full movie metadata coverage.

## Data Fields Used
- movies_metadata.csv: title, genres, overview, vote_average, vote_count, release_date
- keywords.csv: keywords

## Recommendation Method
For each movie, the system builds a content text using genres, overview, and keywords.
TF-IDF converts the content text into numeric vectors, and cosine similarity measures how similar two movies are.
Movies with higher similarity scores are recommended as more content-similar to the selected movie.

## Dataset Prepared for Recommendation
Movies available for recommendation: 45,194.
Movies with keyword text: 31,079.
TF-IDF features: 30,000.

## Sample Recommendation
The sample recommendation file uses `Batman` as the selected movie and saves the Top 10 results to outputs/sample_recommendations.csv.

## Limitations
This is based on content similarity and does not mean a user will definitely like the movie.
The current version does not use user history or personalized rating behavior.
It does not fully solve cold-start and individual preference problems.
Some movies have incomplete overviews or keywords, which can weaken similarity quality.

## Future Work
Use ratings_small.csv for collaborative filtering in a later stage.
Build a hybrid recommender that combines content similarity with user rating behavior.