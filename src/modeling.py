"""Machine learning utilities for predicting audience-level movie success.

This module extends the Step 1 EDA work into a supervised classification task.
It intentionally avoids recommendation-system and LLM-related features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "success_movie"
NUMERIC_FEATURES = ["budget", "runtime", "popularity", "release_year"]
CATEGORICAL_FEATURES = ["original_language", "main_genre"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_cleaned_movies(path: str | Path) -> pd.DataFrame:
    """Load the cleaned movie dataset generated in Step 1."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned movie file: {path}")
    return pd.read_csv(path, low_memory=False)


def cap_series_outliers(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorize a numeric series using quantile caps."""

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    return numeric.clip(lower=lower, upper=upper)


def create_success_dataset(
    df: pd.DataFrame,
    *,
    rating_threshold: float = 7.0,
    vote_count_threshold: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create features and the audience-success classification label.

    ``success_movie`` is defined as:
    ``vote_average >= 7.0 and vote_count >= 50`` by default.

    ``vote_average`` and ``vote_count`` are not used as model features because
    they define the target label and would create leakage.
    """

    model_df = df.copy()

    for column in NUMERIC_FEATURES + ["vote_average", "vote_count"]:
        if column in model_df.columns:
            model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    initial_rows = len(model_df)
    required_target_cols = ["vote_average", "vote_count"]
    model_df = model_df.dropna(subset=required_target_cols).copy()

    model_df = model_df[
        (model_df["release_year"].isna())
        | ((model_df["release_year"] >= 1900) & (model_df["release_year"] <= 2020))
    ].copy()

    # Treat non-positive budget/runtime as unknown values, not real zeros.
    model_df.loc[model_df["budget"] <= 0, "budget"] = pd.NA
    model_df.loc[model_df["runtime"] <= 0, "runtime"] = pd.NA
    model_df.loc[model_df["runtime"] > 300, "runtime"] = pd.NA
    model_df.loc[model_df["popularity"] < 0, "popularity"] = pd.NA

    for column in NUMERIC_FEATURES:
        model_df[column] = cap_series_outliers(model_df[column])

    for column in CATEGORICAL_FEATURES:
        model_df[column] = model_df[column].fillna("Unknown").astype(str)

    model_df[TARGET_COLUMN] = (
        (model_df["vote_average"] >= rating_threshold)
        & (model_df["vote_count"] >= vote_count_threshold)
    ).astype(int)

    summary = {
        "initial_rows": int(initial_rows),
        "model_rows": int(len(model_df)),
        "positive_rows": int(model_df[TARGET_COLUMN].sum()),
        "negative_rows": int((model_df[TARGET_COLUMN] == 0).sum()),
        "positive_rate": float(model_df[TARGET_COLUMN].mean()),
        "rating_threshold": rating_threshold,
        "vote_count_threshold": vote_count_threshold,
        "features": MODEL_FEATURES,
    }
    return model_df, summary


def split_features(
    model_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split model features and target into train/test sets."""

    X = model_df[MODEL_FEATURES].copy()
    y = model_df[TARGET_COLUMN].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Create the classification models used in the Step 2 notebook."""

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                        min_samples_leaf=3,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, Any]]:
    """Fit all models and return fitted models, metrics table, and raw details."""

    models = build_models(random_state=random_state)
    rows = []
    details: dict[str, Any] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
        details[model_name] = metrics

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return models, metrics_df, details


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    numeric_names = NUMERIC_FEATURES
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_names = categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return numeric_names + categorical_names


def random_forest_feature_importance(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract top Random Forest feature importances."""

    classifier = model.named_steps["classifier"]
    feature_names = get_feature_names(model)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importance.head(top_n).reset_index(drop=True)


def save_metrics(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return output_path


def save_models(models: dict[str, Pipeline], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        path = output_dir / filename
        joblib.dump(model, path)
        paths[name] = path
    return paths


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = metrics_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"])
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 5.2))
    sns.barplot(data=plot_df, x="model", y="value", hue="variable")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def plot_confusion_matrix(
    matrix: Any,
    model_name: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not success", "Success"],
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", legend=False)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path
