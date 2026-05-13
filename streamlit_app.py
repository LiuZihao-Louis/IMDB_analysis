"""Streamlit dashboard for IMDB data analysis and ML extension results.

Run from the project root:
    streamlit run streamlit_app.py

This dashboard visualizes the cleaned EDA outputs and the optional
audience-success prediction module. It does not include recommendation-system
or LLM-related features.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.modeling import MODEL_FEATURES, REFERENCE_YEAR

alt.data_transformers.disable_max_rows()


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CLEANED_PATH = OUTPUT_DIR / "cleaned_movies.csv"
FINANCIAL_PATH = OUTPUT_DIR / "financial_movies.csv"
SUMMARY_PATH = OUTPUT_DIR / "data_analysis_summary.md"
ML_METRICS_PATH = OUTPUT_DIR / "ml_model_metrics.csv"
ML_SUMMARY_PATH = OUTPUT_DIR / "ml_success_prediction_summary.md"
ML_POPULARITY_PATH = OUTPUT_DIR / "ml_popularity_comparison.csv"
ML_THRESHOLD_PATH = OUTPUT_DIR / "ml_threshold_tuning.csv"
ML_CV_PATH = OUTPUT_DIR / "ml_cross_validation_metrics.csv"
ML_IMPORTANCE_PATH = OUTPUT_DIR / "ml_feature_importance.csv"
ML_RANDOM_FOREST_PATH = OUTPUT_DIR / "models" / "random_forest.joblib"

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


st.set_page_config(
    page_title="IMDB Movie Success Dashboard",
    page_icon="IMDB",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.read_csv(CLEANED_PATH, low_memory=False)
    financial = pd.read_csv(FINANCIAL_PATH, low_memory=False)
    return cleaned, financial


@st.cache_data(show_spinner=False)
def load_summary_text() -> str:
    if SUMMARY_PATH.exists():
        return SUMMARY_PATH.read_text(encoding="utf-8")
    return ""


@st.cache_data(show_spinner=False)
def load_ml_summary_text() -> str:
    if ML_SUMMARY_PATH.exists():
        return ML_SUMMARY_PATH.read_text(encoding="utf-8")
    return ""


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_random_forest_model():
    if ML_RANDOM_FOREST_PATH.exists():
        return joblib.load(ML_RANDOM_FOREST_PATH)
    return None


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.0f}"


def base_chart(data: pd.DataFrame, height: int = 360) -> alt.Chart:
    return alt.Chart(data).properties(height=height)


def show_image_if_exists(filename: str, caption: str | None = None) -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} is not available. Re-run the analysis notebook to generate it.")


def success_probability(model, input_df: pd.DataFrame) -> float | None:
    if model is None or not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(input_df[MODEL_FEATURES])
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return float(probabilities[:, classes.index(1)][0])
    if probabilities.shape[1] >= 2:
        return float(probabilities[:, 1][0])
    return None


def filtered_data(
    cleaned: pd.DataFrame,
    financial: pd.DataFrame,
    year_range: tuple[int, int],
    selected_genres: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_filter = cleaned["release_year"].between(year_range[0], year_range[1], inclusive="both")
    fin_filter = financial["release_year"].between(year_range[0], year_range[1], inclusive="both")

    if selected_genres:
        clean_filter &= cleaned["main_genre"].isin(selected_genres)
        fin_filter &= financial["main_genre"].isin(selected_genres)

    return cleaned[clean_filter].copy(), financial[fin_filter].copy()


def insight_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.75rem 0 0.25rem 0;
    }
    .insight-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
        min-height: 116px;
    }
    .insight-title {
        color: #1f2937;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .insight-body {
        color: #4b5563;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if not CLEANED_PATH.exists() or not FINANCIAL_PATH.exists():
    st.error(
        "Required output files are missing. Run `notebooks/01_data_analysis.ipynb` first "
        "to generate `outputs/cleaned_movies.csv` and `outputs/financial_movies.csv`."
    )
    st.stop()


cleaned_movies, financial_movies = load_data()
summary_text = load_summary_text()
ml_summary_text = load_ml_summary_text()

for frame in (cleaned_movies, financial_movies):
    for column in ["release_year", "budget", "revenue", "profit", "roi", "vote_average", "vote_count", "popularity"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

available_years = cleaned_movies["release_year"].dropna()
min_year = int(max(1900, available_years.min()))
max_year = int(min(2020, available_years.max()))
genre_options = sorted(cleaned_movies["main_genre"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.header("Filters")
    year_range = st.slider("Release year range", min_year, max_year, (1980, max_year))
    selected_genres = st.multiselect("Main genre", genre_options)
    min_vote_count = st.slider("Minimum vote count", 0, 500, 20, step=10)
    st.divider()
    st.caption("Dashboard uses movie-level metadata. Other CSV files remain reserved for later recommendation and richer ML extensions.")

filtered_cleaned, filtered_financial = filtered_data(cleaned_movies, financial_movies, year_range, selected_genres)
rating_filtered = filtered_cleaned[filtered_cleaned["vote_count"].fillna(0) >= min_vote_count].copy()

st.title("IMDB Movie Success Dashboard")
st.caption(
    "Interactive visualization front-end for Step 1: data cleaning, EDA, and presentation-ready insights."
)

tabs = st.tabs(
    [
        "Overview",
        "Data Quality",
        "Commercial Success",
        "Genre Performance",
        "Audience Response",
        "Time Trends",
        "ML Prediction",
        "Summary",
    ]
)

with tabs[0]:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Cleaned movies", number(len(filtered_cleaned)))
    metric_cols[1].metric("Financial rows", number(len(filtered_financial)))
    metric_cols[2].metric("Median budget", money(filtered_financial["budget"].median()))
    metric_cols[3].metric("Median revenue", money(filtered_financial["revenue"].median()))
    metric_cols[4].metric("Median ROI", f"{filtered_financial['roi'].median():.2f}" if not filtered_financial.empty else "N/A")

    c1, c2, c3 = st.columns(3)
    with c1:
        insight_box(
            "Main question",
            "The dashboard asks what factors are associated with movie success across revenue, ROI, audience response, and industry trends.",
        )
    with c2:
        insight_box(
            "Financial caution",
            "Budget and revenue zeros are treated as invalid for financial analysis, so financial charts use the positive-budget and positive-revenue subset.",
        )
    with c3:
        insight_box(
            "Scope",
            "The main dashboard presents Step 1 EDA results. The ML tab adds the planned audience-success prediction extension without recommendations or LLM features.",
        )

    st.markdown('<div class="section-title">Movie Count by Main Genre</div>', unsafe_allow_html=True)
    genre_counts = (
        filtered_cleaned.dropna(subset=["main_genre"])
        .groupby("main_genre")
        .size()
        .reset_index(name="movie_count")
        .sort_values("movie_count", ascending=False)
        .head(12)
    )
    chart = (
        base_chart(genre_counts)
        .mark_bar()
        .encode(
            x=alt.X("movie_count:Q", title="Movies"),
            y=alt.Y("main_genre:N", sort="-x", title="Main genre"),
            tooltip=["main_genre", alt.Tooltip("movie_count:Q", format=",")],
            color=alt.Color("main_genre:N", legend=None),
        )
    )
    st.altair_chart(chart, use_container_width=True)

with tabs[1]:
    st.subheader("Data Quality")
    cols = st.columns([1.05, 1])
    with cols[0]:
        show_image_if_exists("missing_values.png", "Missing values in key columns")
    with cols[1]:
        duplicate_text = "62 rows belong to duplicated movie-id groups; 30 redundant rows were removed during cleaning."
        insight_box("Duplicate handling", duplicate_text)
        st.write("")
        insight_box(
            "Financial validity",
            "Zero or negative budget/revenue values remain in the main dataset but are excluded from financial analysis.",
        )

    missing_counts = cleaned_movies[[col for col in KEY_COLUMNS if col in cleaned_movies.columns]].isna().sum()
    missing_table = (
        missing_counts.rename("missing_rows")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rows", ascending=False)
    )
    st.dataframe(missing_table, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Commercial Success")
    c1, c2 = st.columns(2)
    with c1:
        insight_box(
            "Investment scale",
            "Higher budget is associated with higher revenue potential, but that does not mean every high-budget movie is efficient.",
        )
    with c2:
        insight_box(
            "ROI interpretation",
            "ROI can be volatile, especially for low-budget films. Use ROI as an investment-efficiency indicator, not as the only success definition.",
        )

    scatter_df = filtered_financial.dropna(subset=["budget", "revenue", "profit", "roi", "title"]).copy()
    if scatter_df.empty:
        st.info("No financial rows are available for the current filters.")
    else:
        left, right = st.columns(2)
        with left:
            chart = (
                base_chart(scatter_df, 380)
                .mark_circle(size=44, opacity=0.45)
                .encode(
                    x=alt.X("budget:Q", scale=alt.Scale(type="log"), title="Budget (log)"),
                    y=alt.Y("revenue:Q", scale=alt.Scale(type="log"), title="Revenue (log)"),
                    color=alt.Color("budget_group:N", title="Budget group"),
                    tooltip=["title", "main_genre", alt.Tooltip("budget:Q", format="$,.0f"), alt.Tooltip("revenue:Q", format="$,.0f")],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        with right:
            chart = (
                base_chart(scatter_df, 380)
                .mark_boxplot(extent="min-max")
                .encode(
                    x=alt.X("budget_group:N", sort=["Low", "Medium", "High"], title="Budget group"),
                    y=alt.Y("roi:Q", title="ROI"),
                    color=alt.Color("budget_group:N", legend=None),
                    tooltip=["budget_group", alt.Tooltip("roi:Q", format=".2f")],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    with st.expander("Show static commercial figures from notebook"):
        img_cols = st.columns(3)
        with img_cols[0]:
            show_image_if_exists("budget_vs_revenue.png")
        with img_cols[1]:
            show_image_if_exists("budget_vs_profit.png")
        with img_cols[2]:
            show_image_if_exists("roi_by_budget_group.png")

with tabs[3]:
    st.subheader("Genre Performance")
    genre_fin = filtered_financial.dropna(subset=["main_genre"])
    genre_clean = rating_filtered.dropna(subset=["main_genre"])

    revenue_by_genre = (
        genre_fin.groupby("main_genre")
        .agg(movie_count=("title", "size"), average_revenue=("revenue", "mean"), median_roi=("roi", "median"))
        .query("movie_count >= 30")
        .reset_index()
    )
    rating_by_genre = (
        genre_clean.groupby("main_genre")
        .agg(movie_count=("title", "size"), average_rating=("vote_average", "mean"))
        .query("movie_count >= 100")
        .reset_index()
    )

    c1, c2 = st.columns(2)
    with c1:
        insight_box(
            "Genre is not one ranking",
            "A genre can perform well commercially while another performs better by rating or ROI. These are different success definitions.",
        )
    with c2:
        insight_box(
            "Main-genre caution",
            "The dashboard uses the first listed genre as `main_genre`, so multi-genre movies are simplified for Step 1 clarity.",
        )

    left, right = st.columns(2)
    with left:
        chart_df = revenue_by_genre.sort_values("average_revenue", ascending=False).head(10)
        chart = (
            base_chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("average_revenue:Q", title="Average revenue"),
                y=alt.Y("main_genre:N", sort="-x", title="Main genre"),
                color=alt.Color("main_genre:N", legend=None),
                tooltip=["main_genre", "movie_count", alt.Tooltip("average_revenue:Q", format="$,.0f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        chart_df = rating_by_genre.sort_values("average_rating", ascending=False).head(10)
        chart = (
            base_chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("average_rating:Q", title="Average rating"),
                y=alt.Y("main_genre:N", sort="-x", title="Main genre"),
                color=alt.Color("main_genre:N", legend=None),
                tooltip=["main_genre", "movie_count", alt.Tooltip("average_rating:Q", format=".2f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    roi_chart_df = revenue_by_genre.sort_values("median_roi", ascending=False).head(10)
    chart = (
        base_chart(roi_chart_df)
        .mark_bar()
        .encode(
            x=alt.X("median_roi:Q", title="Median ROI"),
            y=alt.Y("main_genre:N", sort="-x", title="Main genre"),
            color=alt.Color("main_genre:N", legend=None),
            tooltip=["main_genre", "movie_count", alt.Tooltip("median_roi:Q", format=".2f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Show static genre figures from notebook"):
        img_cols = st.columns(4)
        for image_col, image_name in zip(
            img_cols,
            ["revenue_by_genre.png", "rating_by_genre.png", "roi_by_genre.png", "roi_boxplot_by_genre.png"],
        ):
            with image_col:
                show_image_if_exists(image_name)

with tabs[4]:
    st.subheader("Audience Response")
    audience_df = filtered_financial.dropna(subset=["revenue", "vote_average", "vote_count", "popularity"]).copy()
    if audience_df.empty:
        st.info("No audience/financial rows are available for the current filters.")
    else:
        corr = audience_df[["revenue", "vote_average", "vote_count", "popularity"]].corr(numeric_only=True)["revenue"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Revenue vs rating corr.", f"{corr.get('vote_average', float('nan')):.2f}")
        c2.metric("Revenue vs vote count corr.", f"{corr.get('vote_count', float('nan')):.2f}")
        c3.metric("Revenue vs popularity corr.", f"{corr.get('popularity', float('nan')):.2f}")

        insight_box(
            "Attention vs evaluation",
            "Vote count and popularity generally behave more like attention indicators, while vote average is closer to evaluation quality. They should not be interpreted as the same thing.",
        )

        left, right = st.columns(2)
        with left:
            chart = (
                base_chart(audience_df)
                .mark_circle(size=38, opacity=0.45)
                .encode(
                    x=alt.X("vote_average:Q", title="Vote average"),
                    y=alt.Y("revenue:Q", scale=alt.Scale(type="log"), title="Revenue (log)"),
                    color=alt.Color("main_genre:N", legend=None),
                    tooltip=["title", "main_genre", alt.Tooltip("vote_average:Q", format=".1f"), alt.Tooltip("revenue:Q", format="$,.0f")],
                )
            )
            st.altair_chart(chart, use_container_width=True)
        with right:
            chart = (
                base_chart(audience_df)
                .mark_circle(size=38, opacity=0.45)
                .encode(
                    x=alt.X("vote_count:Q", scale=alt.Scale(type="log"), title="Vote count (log)"),
                    y=alt.Y("revenue:Q", scale=alt.Scale(type="log"), title="Revenue (log)"),
                    color=alt.Color("main_genre:N", legend=None),
                    tooltip=["title", "main_genre", alt.Tooltip("vote_count:Q", format=","), alt.Tooltip("revenue:Q", format="$,.0f")],
                )
            )
            st.altair_chart(chart, use_container_width=True)

    with st.expander("Show static audience figures from notebook"):
        img_cols = st.columns(4)
        for image_col, image_name in zip(
            img_cols,
            ["rating_vs_revenue.png", "vote_count_vs_revenue.png", "popularity_vs_revenue.png", "correlation_heatmap.png"],
        ):
            with image_col:
                show_image_if_exists(image_name)

with tabs[5]:
    st.subheader("Time Trends")
    year_clean = filtered_cleaned.dropna(subset=["release_year"])
    year_fin = filtered_financial.dropna(subset=["release_year"])

    count_by_year = year_clean.groupby("release_year").size().reset_index(name="movie_count")
    rating_by_year = year_clean.groupby("release_year")["vote_average"].mean().reset_index(name="average_rating")
    revenue_by_year = year_fin.groupby("release_year")["revenue"].mean().reset_index(name="average_revenue")
    budget_by_year = year_fin.groupby("release_year")["budget"].mean().reset_index(name="average_budget")

    insight_box(
        "Different trend meanings",
        "Movie count reflects production volume, budget/revenue reflect financial scale, and rating reflects audience evaluation. They do not need to move together.",
    )

    left, right = st.columns(2)
    with left:
        chart = (
            base_chart(count_by_year, 330)
            .mark_line(point=False)
            .encode(
                x=alt.X("release_year:Q", title="Release year", scale=alt.Scale(zero=False)),
                y=alt.Y("movie_count:Q", title="Movie count"),
                tooltip=[alt.Tooltip("release_year:Q", format=".0f"), alt.Tooltip("movie_count:Q", format=",")],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        chart = (
            base_chart(rating_by_year, 330)
            .mark_line(point=False, color="#d65f5f")
            .encode(
                x=alt.X("release_year:Q", title="Release year", scale=alt.Scale(zero=False)),
                y=alt.Y("average_rating:Q", title="Average rating"),
                tooltip=[alt.Tooltip("release_year:Q", format=".0f"), alt.Tooltip("average_rating:Q", format=".2f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        chart = (
            base_chart(revenue_by_year, 330)
            .mark_line(point=False, color="#4c9f70")
            .encode(
                x=alt.X("release_year:Q", title="Release year", scale=alt.Scale(zero=False)),
                y=alt.Y("average_revenue:Q", title="Average revenue"),
                tooltip=[alt.Tooltip("release_year:Q", format=".0f"), alt.Tooltip("average_revenue:Q", format="$,.0f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        chart = (
            base_chart(budget_by_year, 330)
            .mark_line(point=False, color="#7c6ab0")
            .encode(
                x=alt.X("release_year:Q", title="Release year", scale=alt.Scale(zero=False)),
                y=alt.Y("average_budget:Q", title="Average budget"),
                tooltip=[alt.Tooltip("release_year:Q", format=".0f"), alt.Tooltip("average_budget:Q", format="$,.0f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

with tabs[6]:
    st.subheader("Machine Learning: Audience Success Prediction")
    st.markdown(
        """
        This module predicts `success_movie`, defined as `vote_average >= 7.0`
        and `vote_count >= 50`. The models use metadata features only, so
        `vote_average` and `vote_count` are excluded from the feature set to avoid label leakage.
        The label measures audience response, not direct financial success.
        """
    )

    if ML_METRICS_PATH.exists():
        ml_metrics = load_optional_csv(ML_METRICS_PATH)
        popularity_comparison = load_optional_csv(ML_POPULARITY_PATH)
        threshold_table = load_optional_csv(ML_THRESHOLD_PATH)
        cv_table = load_optional_csv(ML_CV_PATH)
        importance_table = load_optional_csv(ML_IMPORTANCE_PATH)

        best_row = ml_metrics.sort_values("f1", ascending=False).iloc[0]
        baseline_row = ml_metrics.loc[ml_metrics["model"] == "Dummy Baseline"].iloc[0]

        metric_cols = st.columns(6)
        metric_cols[0].metric("Best model", best_row["model"])
        metric_cols[1].metric("Accuracy", f"{best_row['accuracy']:.3f}")
        metric_cols[2].metric("Precision", f"{best_row['precision']:.3f}")
        metric_cols[3].metric("Recall", f"{best_row['recall']:.3f}")
        metric_cols[4].metric("F1-score", f"{best_row['f1']:.3f}")
        metric_cols[5].metric("PR-AUC", f"{best_row['pr_auc']:.3f}")

        c1, c2, c3 = st.columns(3)
        with c1:
            insight_box(
                "Audience-success label",
                "The target is rating-and-vote based. It should not be mixed with revenue, profit, or ROI success.",
            )
        with c2:
            insight_box(
                "Class imbalance",
                f"The Dummy baseline gets {baseline_row['accuracy']:.3f} accuracy but zero recall and zero F1-score, so accuracy is not the main metric.",
            )
        with c3:
            insight_box(
                "Popularity boundary",
                "Popularity is useful but may reflect post-release attention, so the model is exploratory rather than pure pre-release prediction.",
            )

        st.dataframe(ml_metrics.round(3), hide_index=True, use_container_width=True)

        img_cols = st.columns(3)
        for image_col, image_name in zip(
            img_cols,
            ["ml_model_comparison.png", "ml_confusion_matrix.png", "ml_feature_importance.png"],
        ):
            with image_col:
                show_image_if_exists(image_name)

        img_cols = st.columns(3)
        for image_col, image_name in zip(
            img_cols,
            ["ml_popularity_comparison.png", "ml_threshold_tuning.png", "ml_cv_metrics.png"],
        ):
            with image_col:
                show_image_if_exists(image_name)

        with st.expander("Show supporting ML tables"):
            if not popularity_comparison.empty:
                st.markdown("**With vs without popularity**")
                st.dataframe(popularity_comparison.round(3), hide_index=True, use_container_width=True)
            if not threshold_table.empty:
                st.markdown("**Threshold tuning**")
                st.dataframe(threshold_table.round(3), hide_index=True, use_container_width=True)
            if not cv_table.empty:
                st.markdown("**Stratified 5-fold cross-validation**")
                st.dataframe(cv_table.round(3), hide_index=True, use_container_width=True)
            if not importance_table.empty:
                st.markdown("**Top feature importance**")
                st.dataframe(importance_table.round(4), hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">Manual Prediction Demo</div>', unsafe_allow_html=True)
        st.caption(
            "This demo uses the saved Random Forest model and metadata-style inputs. It is for classroom explanation, not a production decision tool."
        )
        rf_model = load_random_forest_model()
        if rf_model is None:
            st.info("Random Forest model file is not available. Re-run the ML notebook to generate it.")
        else:
            language_options = (
                cleaned_movies["original_language"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(15)
                .index.tolist()
            )
            if "en" not in language_options:
                language_options.insert(0, "en")
            genre_options_for_model = sorted(cleaned_movies["main_genre"].dropna().astype(str).unique().tolist())

            input_cols = st.columns(3)
            with input_cols[0]:
                demo_budget = st.number_input("Budget", min_value=0, value=10_000_000, step=1_000_000)
                demo_runtime = st.number_input("Runtime", min_value=0, max_value=300, value=100, step=5)
            with input_cols[1]:
                demo_popularity = st.number_input("Popularity", min_value=0.0, value=10.0, step=1.0)
                demo_release_year = st.number_input("Release year", min_value=1900, max_value=2020, value=2010, step=1)
            with input_cols[2]:
                demo_language = st.selectbox("Original language", language_options, index=language_options.index("en"))
                default_genre_index = genre_options_for_model.index("Drama") if "Drama" in genre_options_for_model else 0
                demo_genre = st.selectbox("Main genre", genre_options_for_model, index=default_genre_index)

            demo_input = pd.DataFrame(
                [
                    {
                        "budget": float(demo_budget) if demo_budget > 0 else pd.NA,
                        "runtime": float(demo_runtime) if demo_runtime > 0 else pd.NA,
                        "popularity": float(demo_popularity),
                        "release_year": float(demo_release_year),
                        "has_budget": int(demo_budget > 0),
                        "is_english": int(str(demo_language).lower() == "en"),
                        "movie_age": max(0, REFERENCE_YEAR - int(demo_release_year)),
                        "original_language": demo_language,
                        "main_genre": demo_genre,
                    }
                ]
            )

            probability = success_probability(rf_model, demo_input)
            threshold_choice = st.select_slider(
                "Decision threshold",
                options=[0.3, 0.4, 0.5, 0.6, 0.7],
                value=0.6,
            )
            if probability is not None:
                result_cols = st.columns(2)
                result_cols[0].metric("Predicted success probability", f"{probability:.1%}")
                prediction_label = "Audience-successful" if probability >= threshold_choice else "Not audience-successful"
                result_cols[1].metric("Classification", prediction_label)
                st.caption(
                    "Using a higher threshold reduces false positives but may miss more successful movies. "
                    "The notebook threshold tuning table explains this trade-off."
                )
    else:
        st.info(
            "ML outputs are not available yet. Run `notebooks/03_success_prediction_ml.ipynb` "
            "to generate model metrics and figures."
        )

    if ml_summary_text:
        with st.expander("Show ML summary markdown"):
            st.markdown(ml_summary_text)

with tabs[7]:
    st.subheader("Presentation Summary")
    st.markdown(
        """
        This front-end is designed for presentation and discussion. It turns the Step 1 EDA outputs
        and the audience-success ML extension into a navigable dashboard.
        """
    )
    if summary_text:
        with st.expander("Show generated markdown summary"):
            st.markdown(summary_text)

    st.markdown('<div class="section-title">Generated Output Files</div>', unsafe_allow_html=True)
    output_rows = [
        {"Output": "Cleaned metadata", "Path": "outputs/cleaned_movies.csv", "Exists": CLEANED_PATH.exists()},
        {"Output": "Financial subset", "Path": "outputs/financial_movies.csv", "Exists": FINANCIAL_PATH.exists()},
        {"Output": "Summary markdown", "Path": "outputs/data_analysis_summary.md", "Exists": SUMMARY_PATH.exists()},
        {"Output": "Figures folder", "Path": "outputs/figures/", "Exists": FIGURES_DIR.exists()},
        {"Output": "ML metrics", "Path": "outputs/ml_model_metrics.csv", "Exists": ML_METRICS_PATH.exists()},
        {"Output": "ML summary", "Path": "outputs/ml_success_prediction_summary.md", "Exists": ML_SUMMARY_PATH.exists()},
        {"Output": "Random Forest model", "Path": "outputs/models/random_forest.joblib", "Exists": ML_RANDOM_FOREST_PATH.exists()},
    ]
    st.dataframe(pd.DataFrame(output_rows), hide_index=True, use_container_width=True)
