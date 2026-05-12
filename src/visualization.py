"""Reusable visualization utilities for Step 1 movie success analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGURE_DPI = 160
PALETTE = ["#3b6ea8", "#d65f5f", "#4c9f70", "#c58b34", "#7c6ab0", "#4aa6a5"]


def setup_style() -> None:
    """Apply a clean plotting style suitable for notebook and slides."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "figure.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def ensure_figures_dir(output_dir: str | Path = "outputs/figures") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def required_columns_available(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(column in df.columns for column in columns)


def save_current_figure(path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return str(path)


def _sample_for_scatter(df: pd.DataFrame, columns: list[str], max_points: int = 8000) -> pd.DataFrame:
    clean = df[columns].dropna()
    if len(clean) > max_points:
        return clean.sample(max_points, random_state=42)
    return clean


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> str | None:
    if not required_columns_available(df, [x, y]):
        return None

    plot_df = _sample_for_scatter(df, [x, y])
    if log_x:
        plot_df = plot_df[plot_df[x] > 0]
    if log_y:
        plot_df = plot_df[plot_df[y] > 0]
    if plot_df.empty:
        return None

    setup_style()
    plt.figure(figsize=(8.5, 5.5))
    sns.scatterplot(data=plot_df, x=x, y=y, s=20, alpha=0.45, color=PALETTE[0], edgecolor=None)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_current_figure(output_path)


def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
    *,
    order_desc: bool = True,
    top_n: int | None = None,
    rotate: int = 35,
) -> str | None:
    if data.empty or not required_columns_available(data, [x, y]):
        return None

    plot_df = data[[x, y]].dropna()
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values(y, ascending=not order_desc)
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    setup_style()
    plt.figure(figsize=(9, 5.5))
    palette = sns.color_palette("deep", n_colors=len(plot_df))
    sns.barplot(data=plot_df, x=x, y=y, hue=x, palette=palette, legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    return save_current_figure(output_path)


def plot_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
) -> str | None:
    if data.empty or not required_columns_available(data, [x, y]):
        return None

    plot_df = data[[x, y]].dropna().sort_values(x)
    if plot_df.empty:
        return None

    setup_style()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=plot_df, x=x, y=y, color=PALETTE[0], linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_current_figure(output_path)


def plot_budget_vs_revenue(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    return plot_scatter(
        financial_df,
        "budget",
        "revenue",
        "Budget and Revenue for Movies with Valid Financial Data",
        "Budget (USD, log scale)",
        "Revenue (USD, log scale)",
        ensure_figures_dir(output_dir) / "budget_vs_revenue.png",
        log_x=True,
        log_y=True,
    )


def plot_budget_vs_profit(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    return plot_scatter(
        financial_df,
        "budget",
        "profit",
        "Budget and Profit for Movies with Valid Financial Data",
        "Budget (USD, log scale)",
        "Profit (USD)",
        ensure_figures_dir(output_dir) / "budget_vs_profit.png",
        log_x=True,
    )


def plot_roi_by_budget_group(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    if not required_columns_available(financial_df, ["budget_group", "roi"]):
        return None
    plot_df = financial_df.dropna(subset=["budget_group", "roi"]).copy()
    if plot_df.empty:
        return None

    order = ["Low", "Medium", "High"]
    setup_style()
    plt.figure(figsize=(8, 5.5))
    sns.boxplot(data=plot_df, x="budget_group", y="roi", order=order, color=PALETTE[0], showfliers=False)
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.title("ROI Distribution by Budget Group")
    plt.xlabel("Budget group")
    plt.ylabel("ROI (profit / budget)")
    return save_current_figure(ensure_figures_dir(output_dir) / "roi_by_budget_group.png")


def genre_metric_table(
    df: pd.DataFrame,
    value_col: str,
    *,
    min_count: int = 100,
    top_n: int = 10,
    metric: str = "mean",
) -> pd.DataFrame:
    if not required_columns_available(df, ["main_genre", value_col]):
        return pd.DataFrame()

    grouped = (
        df.dropna(subset=["main_genre", value_col])
        .groupby("main_genre")
        .agg(movie_count=(value_col, "size"), value=(value_col, metric))
        .reset_index()
    )
    grouped = grouped[grouped["movie_count"] >= min_count]
    grouped = grouped.sort_values("value", ascending=False).head(top_n)
    return grouped.rename(columns={"value": value_col})


def plot_missing_values(
    missing_values: dict[str, int] | pd.Series,
    output_dir: str | Path = "outputs/figures",
) -> str | None:
    """Plot missing value counts for key columns."""

    if isinstance(missing_values, dict):
        plot_df = pd.DataFrame(
            {"column": list(missing_values.keys()), "missing_count": list(missing_values.values())}
        )
    else:
        plot_df = missing_values.rename("missing_count").reset_index().rename(columns={"index": "column"})

    if plot_df.empty:
        return None

    plot_df["missing_count"] = pd.to_numeric(plot_df["missing_count"], errors="coerce").fillna(0)
    plot_df = plot_df.sort_values("missing_count", ascending=False)

    setup_style()
    plt.figure(figsize=(9, 5.5))
    palette = sns.color_palette("deep", n_colors=len(plot_df))
    sns.barplot(data=plot_df, x="column", y="missing_count", hue="column", palette=palette, legend=False)
    plt.title("Missing Values in Key Columns")
    plt.xlabel("Key column")
    plt.ylabel("Missing row count")
    plt.xticks(rotation=35, ha="right")
    return save_current_figure(ensure_figures_dir(output_dir) / "missing_values.png")


def plot_revenue_by_genre(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = genre_metric_table(financial_df, "revenue", min_count=30, top_n=10)
    return plot_bar(
        table,
        "main_genre",
        "revenue",
        "Average Revenue by Main Genre",
        "Main genre",
        "Average revenue (USD)",
        ensure_figures_dir(output_dir) / "revenue_by_genre.png",
    )


def plot_rating_by_genre(df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = genre_metric_table(df, "vote_average", min_count=100, top_n=10)
    return plot_bar(
        table,
        "main_genre",
        "vote_average",
        "Average Audience Rating by Main Genre",
        "Main genre",
        "Average vote rating",
        ensure_figures_dir(output_dir) / "rating_by_genre.png",
    )


def plot_roi_by_genre(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = genre_metric_table(financial_df, "roi", min_count=30, top_n=10, metric="median")
    return plot_bar(
        table,
        "main_genre",
        "roi",
        "Median ROI by Main Genre",
        "Main genre",
        "Median ROI",
        ensure_figures_dir(output_dir) / "roi_by_genre.png",
    )


def plot_roi_boxplot_by_genre(
    financial_df: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
    *,
    min_count: int = 30,
    top_n: int = 10,
) -> str | None:
    """Show ROI distributions for common genres, hiding extreme fliers."""

    if not required_columns_available(financial_df, ["main_genre", "roi"]):
        return None

    counts = financial_df.dropna(subset=["main_genre", "roi"])["main_genre"].value_counts()
    genres = counts[counts >= min_count].head(top_n).index.tolist()
    plot_df = financial_df[financial_df["main_genre"].isin(genres)].dropna(subset=["main_genre", "roi"])
    if plot_df.empty:
        return None

    order = (
        plot_df.groupby("main_genre")["roi"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    setup_style()
    plt.figure(figsize=(10, 5.8))
    sns.boxplot(data=plot_df, x="main_genre", y="roi", order=order, color=PALETTE[0], showfliers=False)
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.title("ROI Distribution by Main Genre")
    plt.xlabel("Main genre")
    plt.ylabel("ROI (outliers hidden)")
    plt.xticks(rotation=35, ha="right")
    return save_current_figure(ensure_figures_dir(output_dir) / "roi_boxplot_by_genre.png")


def plot_rating_vs_revenue(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    return plot_scatter(
        financial_df,
        "vote_average",
        "revenue",
        "Audience Rating and Revenue",
        "Vote average",
        "Revenue (USD, log scale)",
        ensure_figures_dir(output_dir) / "rating_vs_revenue.png",
        log_y=True,
    )


def plot_vote_count_vs_revenue(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    return plot_scatter(
        financial_df,
        "vote_count",
        "revenue",
        "Audience Vote Count and Revenue",
        "Vote count (log scale)",
        "Revenue (USD, log scale)",
        ensure_figures_dir(output_dir) / "vote_count_vs_revenue.png",
        log_x=True,
        log_y=True,
    )


def plot_popularity_vs_revenue(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    return plot_scatter(
        financial_df,
        "popularity",
        "revenue",
        "Popularity and Revenue",
        "Popularity (log scale)",
        "Revenue (USD, log scale)",
        ensure_figures_dir(output_dir) / "popularity_vs_revenue.png",
        log_x=True,
        log_y=True,
    )


def plot_correlation_heatmap(
    financial_df: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
) -> str | None:
    columns = [
        column
        for column in ["budget", "revenue", "profit", "roi", "vote_average", "vote_count", "popularity", "runtime"]
        if column in financial_df.columns
    ]
    corr_df = financial_df[columns].dropna()
    if len(columns) < 2 or corr_df.empty:
        return None

    corr = corr_df.corr(numeric_only=True)
    setup_style()
    plt.figure(figsize=(8.5, 6.5))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Between Success-Related Variables")
    return save_current_figure(ensure_figures_dir(output_dir) / "correlation_heatmap.png")


def year_filter(df: pd.DataFrame, year_col: str = "release_year") -> pd.DataFrame:
    if year_col not in df.columns:
        return pd.DataFrame()
    filtered = df.dropna(subset=[year_col]).copy()
    filtered = filtered[(filtered[year_col] >= 1900) & (filtered[year_col] <= 2020)]
    return filtered


def yearly_count_table(df: pd.DataFrame, *, min_count: int = 5) -> pd.DataFrame:
    filtered = year_filter(df)
    if filtered.empty:
        return pd.DataFrame()
    counts = filtered.groupby("release_year").size().reset_index(name="movie_count")
    return counts[counts["movie_count"] >= min_count]


def yearly_mean_table(df: pd.DataFrame, value_col: str, *, min_count: int = 5) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()
    filtered = year_filter(df).dropna(subset=[value_col])
    if filtered.empty:
        return pd.DataFrame()
    table = (
        filtered.groupby("release_year")
        .agg(movie_count=(value_col, "size"), value=(value_col, "mean"))
        .reset_index()
    )
    table = table[table["movie_count"] >= min_count]
    return table.rename(columns={"value": value_col})


def plot_movies_by_year(df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = yearly_count_table(df, min_count=5)
    return plot_line(
        table,
        "release_year",
        "movie_count",
        "Number of Movies Released by Year",
        "Release year",
        "Number of movies",
        ensure_figures_dir(output_dir) / "movies_by_year.png",
    )


def plot_revenue_by_year(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = yearly_mean_table(financial_df, "revenue", min_count=5)
    return plot_line(
        table,
        "release_year",
        "revenue",
        "Average Revenue by Release Year",
        "Release year",
        "Average revenue (USD)",
        ensure_figures_dir(output_dir) / "revenue_by_year.png",
    )


def plot_budget_by_year(financial_df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = yearly_mean_table(financial_df, "budget", min_count=5)
    return plot_line(
        table,
        "release_year",
        "budget",
        "Average Budget by Release Year",
        "Release year",
        "Average budget (USD)",
        ensure_figures_dir(output_dir) / "budget_by_year.png",
    )


def plot_rating_by_year(df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> str | None:
    table = yearly_mean_table(df, "vote_average", min_count=5)
    return plot_line(
        table,
        "release_year",
        "vote_average",
        "Average Audience Rating by Release Year",
        "Release year",
        "Average vote rating",
        ensure_figures_dir(output_dir) / "rating_by_year.png",
    )


def create_all_figures(
    cleaned_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
) -> dict[str, str | None]:
    """Create all Step 1 figures and return their paths or skipped status."""

    figure_functions = {
        "budget_vs_revenue.png": lambda: plot_budget_vs_revenue(financial_df, output_dir),
        "budget_vs_profit.png": lambda: plot_budget_vs_profit(financial_df, output_dir),
        "roi_by_budget_group.png": lambda: plot_roi_by_budget_group(financial_df, output_dir),
        "revenue_by_genre.png": lambda: plot_revenue_by_genre(financial_df, output_dir),
        "rating_by_genre.png": lambda: plot_rating_by_genre(cleaned_df, output_dir),
        "roi_by_genre.png": lambda: plot_roi_by_genre(financial_df, output_dir),
        "roi_boxplot_by_genre.png": lambda: plot_roi_boxplot_by_genre(financial_df, output_dir),
        "rating_vs_revenue.png": lambda: plot_rating_vs_revenue(financial_df, output_dir),
        "vote_count_vs_revenue.png": lambda: plot_vote_count_vs_revenue(financial_df, output_dir),
        "popularity_vs_revenue.png": lambda: plot_popularity_vs_revenue(financial_df, output_dir),
        "correlation_heatmap.png": lambda: plot_correlation_heatmap(financial_df, output_dir),
        "movies_by_year.png": lambda: plot_movies_by_year(cleaned_df, output_dir),
        "revenue_by_year.png": lambda: plot_revenue_by_year(financial_df, output_dir),
        "budget_by_year.png": lambda: plot_budget_by_year(financial_df, output_dir),
        "rating_by_year.png": lambda: plot_rating_by_year(cleaned_df, output_dir),
    }

    results: dict[str, str | None] = {}
    for figure_name, plot_func in figure_functions.items():
        results[figure_name] = plot_func()
    return results
