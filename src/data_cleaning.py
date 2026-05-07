"""Data loading, inspection, and cleaning utilities for Step 1 analysis.

The functions in this module intentionally focus on ``movies_metadata.csv``.
Other CSV files are only inventoried so Step 1 stays within the agreed scope.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLUMNS = [
    "id",
    "title",
    "budget",
    "revenue",
    "genres",
    "release_date",
    "original_language",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
]


def load_movies_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load movies metadata safely.

    The source file contains a few malformed records. ``on_bad_lines='skip'``
    prevents hard crashes if similar issues appear in another copy of the file.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find movies metadata file: {csv_path}")

    return pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")


def to_numeric_safe(df: pd.DataFrame, column: str) -> pd.Series:
    """Convert a column to numeric values, returning missing values if absent."""

    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def parse_genres(value: Any) -> list[str]:
    """Parse the Kaggle-style genre list and return genre names.

    Examples in the source look like:
    ``[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}]``.
    Malformed or empty values return an empty list instead of raising.
    """

    if pd.isna(value):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []

    if not isinstance(parsed, list):
        return []

    names: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def remove_duplicate_movies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate movie records using movie id when available.

    If a row does not have a numeric movie id, the fallback key is
    ``title + release_date``. The returned count is the number of rows removed.
    """

    before = len(df)
    work = df.copy()

    if "movie_id" in work.columns:
        id_mask = work["movie_id"].notna()
        with_id = work[id_mask].drop_duplicates(subset=["movie_id"], keep="first")
        without_id = work[~id_mask]
    else:
        with_id = work.iloc[0:0]
        without_id = work

    fallback_columns = [col for col in ["title", "release_date"] if col in without_id.columns]
    if fallback_columns:
        without_id = without_id.drop_duplicates(subset=fallback_columns, keep="first")
    else:
        without_id = without_id.drop_duplicates(keep="first")

    cleaned = pd.concat([with_id, without_id], axis=0).sort_index()
    return cleaned, before - len(cleaned)


def add_group_column(series: pd.Series, labels: list[str]) -> pd.Series:
    """Create quantile groups, falling back gracefully when data is sparse."""

    valid = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    non_missing = valid.dropna()
    if non_missing.nunique() < len(labels):
        return result

    try:
        grouped = pd.qcut(non_missing, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        return result

    result.loc[grouped.index] = grouped.astype("object")
    return result


def clean_movies_metadata(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    financial_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clean ``movies_metadata.csv`` and return main + financial datasets.

    The main cleaned dataset keeps non-financial rows. The financial dataset is
    the subset where both budget and revenue are positive.
    """

    raw = load_movies_metadata(csv_path)
    df = raw.copy()

    numeric_columns = [
        "budget",
        "revenue",
        "popularity",
        "vote_average",
        "vote_count",
        "runtime",
    ]
    original_missing = {
        column: int(df[column].isna().sum()) for column in KEY_COLUMNS if column in df.columns
    }

    if "id" in df.columns:
        df["movie_id"] = pd.to_numeric(df["id"], errors="coerce")
    else:
        df["movie_id"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    malformed_rows = int(df["movie_id"].isna().sum())
    df = df[df["movie_id"].notna()].copy()

    for column in numeric_columns:
        if column in df.columns:
            df[column] = to_numeric_safe(df, column)

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
    else:
        df["release_date"] = pd.NaT
        df["release_year"] = pd.NA

    if "genres" in df.columns:
        df["genre_list"] = df["genres"].apply(parse_genres)
        df["main_genre"] = df["genre_list"].apply(lambda genres: genres[0] if genres else pd.NA)
    else:
        df["genre_list"] = [[] for _ in range(len(df))]
        df["main_genre"] = pd.NA

    invalid_budget_rows = (
        int((pd.to_numeric(df["budget"], errors="coerce") <= 0).sum() + df["budget"].isna().sum())
        if "budget" in df.columns
        else len(df)
    )
    invalid_revenue_rows = (
        int((pd.to_numeric(df["revenue"], errors="coerce") <= 0).sum() + df["revenue"].isna().sum())
        if "revenue" in df.columns
        else len(df)
    )

    df, duplicate_rows_removed = remove_duplicate_movies(df)

    has_financial_columns = {"budget", "revenue"}.issubset(df.columns)
    if has_financial_columns:
        financial_df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
        df["profit"] = pd.NA
        df["roi"] = pd.NA
        financial_index = financial_df.index
        df.loc[financial_index, "profit"] = df.loc[financial_index, "revenue"] - df.loc[financial_index, "budget"]
        df.loc[financial_index, "roi"] = df.loc[financial_index, "profit"] / df.loc[financial_index, "budget"]
        financial_df["profit"] = financial_df["revenue"] - financial_df["budget"]
        financial_df["roi"] = financial_df["profit"] / financial_df["budget"]
        financial_df["budget_group"] = add_group_column(
            financial_df["budget"], ["Low", "Medium", "High"]
        )
    else:
        financial_df = df.iloc[0:0].copy()
        df["profit"] = pd.NA
        df["roi"] = pd.NA

    if "budget" in df.columns:
        df["budget_group"] = add_group_column(df["budget"].where(df["budget"] > 0), ["Low", "Medium", "High"])
    else:
        df["budget_group"] = pd.NA

    if "vote_average" in df.columns:
        df["rating_group"] = add_group_column(df["vote_average"], ["Low", "Medium", "High"])
        if not financial_df.empty:
            financial_df["rating_group"] = df.loc[financial_df.index, "rating_group"]
    else:
        df["rating_group"] = pd.NA
        if not financial_df.empty:
            financial_df["rating_group"] = pd.NA

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    if financial_output_path is not None:
        financial_output_path = Path(financial_output_path)
        financial_output_path.parent.mkdir(parents=True, exist_ok=True)
        financial_df.to_csv(financial_output_path, index=False)

    key_missing_after = {
        column: int(df[column].isna().sum()) for column in KEY_COLUMNS if column in df.columns
    }

    summary: dict[str, Any] = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "cleaned_rows": int(len(df)),
        "cleaned_columns": int(df.shape[1]),
        "malformed_rows_removed": malformed_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "invalid_budget_rows": invalid_budget_rows,
        "invalid_revenue_rows": invalid_revenue_rows,
        "financial_rows": int(len(financial_df)),
        "original_missing_values": original_missing,
        "cleaned_missing_values": key_missing_after,
    }

    return df, financial_df, summary


def build_dataset_inventory(data_dir: str | Path, metadata_filename: str = "movies_metadata.csv") -> dict[str, Any]:
    """Inspect available CSV files and key metadata quality indicators."""

    data_dir = Path(data_dir)
    csv_files = sorted(path.name for path in data_dir.glob("*.csv"))
    metadata_path = data_dir / metadata_filename
    raw = load_movies_metadata(metadata_path)

    available_key_columns = [column for column in KEY_COLUMNS if column in raw.columns]
    budget = to_numeric_safe(raw, "budget")
    revenue = to_numeric_safe(raw, "revenue")
    movie_id = to_numeric_safe(raw, "id")
    financial_mask = (budget > 0) & (revenue > 0)

    inventory: dict[str, Any] = {
        "csv_files": csv_files,
        "movies_shape": tuple(raw.shape),
        "movies_columns": list(raw.columns),
        "key_dtypes": {column: str(raw[column].dtype) for column in available_key_columns},
        "missing_values": {column: int(raw[column].isna().sum()) for column in available_key_columns},
        "duplicate_movie_count": int(movie_id.duplicated(keep=False).sum()),
        "invalid_budget_rows": int(((budget <= 0) | budget.isna()).sum()),
        "invalid_revenue_rows": int(((revenue <= 0) | revenue.isna()).sum()),
        "financial_analysis_rows": int(financial_mask.sum()),
        "malformed_id_rows": int(movie_id.isna().sum()),
    }
    return inventory


def format_inventory_markdown(inventory: dict[str, Any]) -> str:
    """Return a concise markdown inventory section."""

    lines = [
        "## Dataset Overview",
        "",
        f"- Detected CSV files: {', '.join(inventory['csv_files'])}",
        f"- `movies_metadata.csv` shape: {inventory['movies_shape'][0]:,} rows x {inventory['movies_shape'][1]:,} columns",
        f"- Duplicate movie count based on numeric id: {inventory['duplicate_movie_count']:,}",
        f"- Invalid budget rows for financial analysis: {inventory['invalid_budget_rows']:,}",
        f"- Invalid revenue rows for financial analysis: {inventory['invalid_revenue_rows']:,}",
        f"- Rows usable for financial analysis: {inventory['financial_analysis_rows']:,}",
        f"- Malformed/non-numeric movie id rows: {inventory['malformed_id_rows']:,}",
        "",
        "### Columns",
        "",
        ", ".join(f"`{column}`" for column in inventory["movies_columns"]),
        "",
        "### Key Column Data Types",
        "",
    ]
    lines.extend(f"- `{column}`: {dtype}" for column, dtype in inventory["key_dtypes"].items())
    lines.extend(["", "### Missing Values in Key Columns", ""])
    lines.extend(f"- `{column}`: {count:,}" for column, count in inventory["missing_values"].items())
    return "\n".join(lines)
