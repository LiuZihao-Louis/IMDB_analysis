"""Content-based movie recommendation utilities.

The first recommendation-system version uses movie content only:
genres, overview text, and keywords. It intentionally avoids deep learning,
LLM-based recommendation, and collaborative filtering for this project stage.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_METADATA_PATH = Path("data/movies_metadata.csv")
DEFAULT_KEYWORDS_PATH = Path("data/keywords.csv")
DEFAULT_CLEANED_METADATA_PATH = Path("outputs/cleaned_movies.csv")
MIN_EXPECTED_METADATA_ROWS = 10_000

RECOMMENDATION_OUTPUT_COLUMNS = [
    "Recommended Movie",
    "Similarity Score",
    "Vote Average",
    "Vote Count",
    "Main Genre",
]


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and skip malformed lines only when the default parser fails."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    try:
        return pd.read_csv(path, low_memory=False)
    except ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def read_csv_with_status(path: str | Path) -> tuple[pd.DataFrame, str | None]:
    """Read a CSV and return a warning message when malformed rows are detected."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    try:
        return pd.read_csv(path, low_memory=False), None
    except ParserError as exc:
        fallback = pd.read_csv(path, engine="python", on_bad_lines="skip")
        return fallback, f"{type(exc).__name__}: {str(exc)}"


def load_metadata_with_fallback(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    cleaned_metadata_path: str | Path = DEFAULT_CLEANED_METADATA_PATH,
    *,
    min_expected_rows: int = MIN_EXPECTED_METADATA_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load movie metadata and use cleaned Step 1 output if the raw file is truncated."""

    metadata_path = Path(metadata_path)
    cleaned_metadata_path = Path(cleaned_metadata_path)
    if not cleaned_metadata_path.exists() and not cleaned_metadata_path.is_absolute():
        project_candidate = metadata_path.parent.parent / cleaned_metadata_path
        if project_candidate.exists():
            cleaned_metadata_path = project_candidate

    raw_metadata, raw_warning = read_csv_with_status(metadata_path)
    cleaned_metadata = pd.DataFrame()
    if cleaned_metadata_path.exists():
        cleaned_metadata = pd.read_csv(cleaned_metadata_path, low_memory=False)

    use_cleaned = (
        not cleaned_metadata.empty
        and (
            raw_warning is not None
            or (len(raw_metadata) < min_expected_rows and len(cleaned_metadata) > len(raw_metadata))
        )
    )

    if use_cleaned:
        reason = (
            "The raw movies_metadata.csv file in this workspace is malformed or truncated, "
            "so the recommender uses outputs/cleaned_movies.csv generated from movies_metadata.csv "
            "for full movie metadata coverage."
        )
        metadata = cleaned_metadata
        source = str(cleaned_metadata_path)
    else:
        reason = "The recommender uses the raw movies_metadata.csv file directly."
        metadata = raw_metadata
        source = str(metadata_path)

    status = {
        "raw_metadata_path": str(metadata_path),
        "raw_metadata_rows_read": int(len(raw_metadata)),
        "raw_metadata_parse_warning": raw_warning or "",
        "cleaned_metadata_path": str(cleaned_metadata_path),
        "cleaned_metadata_rows_available": int(len(cleaned_metadata)),
        "metadata_source": source,
        "metadata_source_reason": reason,
    }
    return metadata, status


def parse_name_list(value: Any) -> list[str]:
    """Parse Kaggle-style list-of-dict strings and return their ``name`` values."""

    if isinstance(value, list):
        raw_items = value
    elif pd.isna(value) or value == "":
        return []
    else:
        try:
            raw_items = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []

    names: list[str] = []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return names

    for item in raw_items:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = item
        if name is not None and str(name).strip():
            names.append(str(name).strip())
    return names


def normalize_token(value: str) -> str:
    """Normalize a short genre or keyword phrase into a TF-IDF-friendly token."""

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def join_tokens(values: list[str]) -> str:
    tokens = [normalize_token(value) for value in values]
    return " ".join(token for token in tokens if token)


def prepare_recommendation_data(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    keywords_path: str | Path = DEFAULT_KEYWORDS_PATH,
    cleaned_metadata_path: str | Path = DEFAULT_CLEANED_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, merge, and prepare movie content for recommendation."""

    metadata, metadata_status = load_metadata_with_fallback(metadata_path, cleaned_metadata_path)
    keywords = read_csv_safely(keywords_path)

    required_metadata = {"id", "title", "genres", "overview"}
    required_keywords = {"id", "keywords"}
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    missing_keywords = sorted(required_keywords - set(keywords.columns))
    if missing_metadata:
        raise ValueError(f"movies_metadata.csv is missing required columns: {missing_metadata}")
    if missing_keywords:
        raise ValueError(f"keywords.csv is missing required columns: {missing_keywords}")

    metadata = metadata.copy()
    keywords = keywords.copy()

    metadata["movie_id"] = pd.to_numeric(metadata["id"], errors="coerce")
    keywords["movie_id"] = pd.to_numeric(keywords["id"], errors="coerce")
    metadata = metadata.dropna(subset=["movie_id", "title"]).copy()
    keywords = keywords.dropna(subset=["movie_id"]).copy()
    metadata["movie_id"] = metadata["movie_id"].astype(int)
    keywords["movie_id"] = keywords["movie_id"].astype(int)

    metadata = metadata.drop_duplicates(subset=["movie_id"], keep="first").copy()

    metadata["genre_list"] = metadata["genres"].apply(parse_name_list)
    metadata["main_genre"] = metadata["genre_list"].apply(lambda values: values[0] if values else "Unknown")
    metadata["genres_text"] = metadata["genre_list"].apply(join_tokens)

    keywords["keyword_list"] = keywords["keywords"].apply(parse_name_list)
    keywords["keywords_text"] = keywords["keyword_list"].apply(join_tokens)
    keyword_text = (
        keywords.groupby("movie_id", as_index=False)["keywords_text"]
        .apply(lambda series: " ".join(value for value in series if value))
        .reset_index(drop=True)
    )

    recommendation_df = metadata.merge(keyword_text, on="movie_id", how="left")
    recommendation_df["keywords_text"] = recommendation_df["keywords_text"].fillna("")
    recommendation_df["overview"] = recommendation_df["overview"].fillna("").astype(str)
    recommendation_df["title"] = recommendation_df["title"].fillna("").astype(str)

    if "release_date" in recommendation_df.columns:
        release_dates = pd.to_datetime(recommendation_df["release_date"], errors="coerce")
        recommendation_df["release_year"] = release_dates.dt.year
    else:
        recommendation_df["release_year"] = pd.NA

    for column in ["vote_average", "vote_count"]:
        if column not in recommendation_df.columns:
            recommendation_df[column] = pd.NA
        recommendation_df[column] = pd.to_numeric(recommendation_df[column], errors="coerce")

    recommendation_df["content"] = (
        recommendation_df["genres_text"].fillna("")
        + " "
        + recommendation_df["overview"].fillna("")
        + " "
        + recommendation_df["keywords_text"].fillna("")
    ).str.strip()
    recommendation_df = recommendation_df[recommendation_df["content"].str.len() > 0].copy()

    recommendation_df["display_title"] = recommendation_df.apply(
        lambda row: f"{row['title']} ({int(row['release_year'])})"
        if pd.notna(row["release_year"])
        else str(row["title"]),
        axis=1,
    )

    output_columns = [
        "movie_id",
        "title",
        "display_title",
        "release_year",
        "main_genre",
        "genres_text",
        "overview",
        "keywords_text",
        "content",
        "vote_average",
        "vote_count",
    ]
    recommendation_df = recommendation_df[output_columns].sort_values("title").reset_index(drop=True)

    summary = {
        **metadata_status,
        "metadata_rows_loaded": int(len(metadata)),
        "keywords_rows_loaded": int(len(keywords)),
        "recommendation_rows": int(len(recommendation_df)),
        "movies_with_keywords": int((recommendation_df["keywords_text"].str.len() > 0).sum()),
        "movies_with_overview": int((recommendation_df["overview"].str.len() > 0).sum()),
        "movies_with_genres": int((recommendation_df["genres_text"].str.len() > 0).sum()),
    }
    return recommendation_df, summary


@dataclass
class ContentRecommender:
    """TF-IDF + cosine-similarity content recommender."""

    movies: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: Any

    def __post_init__(self) -> None:
        self.movie_id_to_index = {
            int(movie_id): index for index, movie_id in enumerate(self.movies["movie_id"].tolist())
        }

    def recommend_by_id(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return the top N content-similar movies for one movie id."""

        movie_id = int(movie_id)
        if movie_id not in self.movie_id_to_index:
            raise ValueError(f"Movie id {movie_id} is not available in the recommendation dataset.")

        top_n = max(1, int(top_n))
        source_index = self.movie_id_to_index[movie_id]
        scores = cosine_similarity(self.tfidf_matrix[source_index], self.tfidf_matrix).ravel()
        scores[source_index] = -1
        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = self.movies.iloc[top_indices].copy()
        recommendations["Similarity Score"] = scores[top_indices]
        recommendations = recommendations.rename(
            columns={
                "title": "Recommended Movie",
                "vote_average": "Vote Average",
                "vote_count": "Vote Count",
                "main_genre": "Main Genre",
            }
        )
        recommendations["Similarity Score"] = recommendations["Similarity Score"].round(4)
        recommendations["Vote Average"] = recommendations["Vote Average"].round(2)
        recommendations["Vote Count"] = recommendations["Vote Count"].fillna(0).astype(int)
        return recommendations[RECOMMENDATION_OUTPUT_COLUMNS].reset_index(drop=True)


def build_content_recommender(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    keywords_path: str | Path = DEFAULT_KEYWORDS_PATH,
    cleaned_metadata_path: str | Path = DEFAULT_CLEANED_METADATA_PATH,
    *,
    max_features: int = 30_000,
    min_df: int = 1,
) -> tuple[ContentRecommender, dict[str, Any]]:
    """Build a TF-IDF content recommender from metadata and keywords."""

    movies, summary = prepare_recommendation_data(metadata_path, keywords_path, cleaned_metadata_path)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(movies["content"])
    summary["tfidf_rows"] = int(tfidf_matrix.shape[0])
    summary["tfidf_features"] = int(tfidf_matrix.shape[1])
    return ContentRecommender(movies=movies, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix), summary


def find_movie_id_by_title(movies: pd.DataFrame, title: str) -> int:
    """Find the first movie id matching a title case-insensitively."""

    matches = movies[movies["title"].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError(f"Movie title not found: {title}")
    return int(matches.iloc[0]["movie_id"])
