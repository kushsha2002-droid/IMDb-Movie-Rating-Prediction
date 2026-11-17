"""Reconstructed IMDb Top 1000 multiple linear regression project.

Run: python -m src.train --data data/imdb_top_1000.csv
This is not the original December 2025 submitted notebook.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TARGET = "IMDB_Rating"
FEATURES = ["Released_Year", "No_of_Votes", "Gross", "Runtime", "Meta_score"]


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Read the original Kaggle-schema CSV and parse numeric fields."""
    df = pd.read_csv(path)
    missing = set([TARGET, *FEATURES]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = df.copy()
    for column in [TARGET, "Released_Year", "No_of_Votes", "Gross", "Meta_score"]:
        clean[column] = pd.to_numeric(
            clean[column].astype("string").str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    clean["Runtime"] = pd.to_numeric(
        clean["Runtime"].astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    clean = clean.dropna(subset=[TARGET]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("Dataset contains no valid IMDb rating labels.")
    if any(clean[c].notna().sum() == 0 for c in FEATURES):
        raise ValueError("At least one predictor has no usable numeric values.")
    return clean


def fit_explanatory_ols(df: pd.DataFrame):
    """Full-sample OLS: descriptive in-sample fit, not predictive validation."""
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(df[FEATURES])
    x = sm.add_constant(pd.DataFrame(values, columns=FEATURES, index=df.index), has_constant="add")
    return sm.OLS(df[TARGET].astype(float), x).fit()


def fit_holdout_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Fit imputation within the training split to avoid leakage."""
    X, y = df[FEATURES], df[TARGET].astype(float)
    if len(df) < 10:
        raise ValueError("Need at least 10 labeled rows for a train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("linear_regression", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "test_r2": round(float(r2_score(y_test, predictions)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": random_state,
    }
    return pipeline, metrics, y_test.to_numpy(), predictions


def plot_diagnostics(df: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET].astype(float).to_numpy(), bins=25, kde=True, ax=ax)
    ax.set(title="IMDb rating distribution", xlabel="IMDb rating", ylabel="Movie count")
    fig.tight_layout()
    fig.savefig(output / "rating_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.65, edgecolors="none")
    lower = min(float(np.min(actual)), float(np.min(predicted)))
    upper = max(float(np.max(actual)), float(np.max(predicted)))
    ax.plot([lower, upper], [lower, upper], "--", linewidth=1)
    ax.set(title="Holdout predictions vs. actual IMDb rating",
           xlabel="Actual rating", ylabel="Predicted rating")
    fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, actual - predicted, alpha=0.65, edgecolors="none")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(title="Held-out residuals", xlabel="Predicted rating",
           ylabel="Actual - predicted")
    fig.tight_layout()
    fig.savefig(output / "residuals.png", dpi=160)
    plt.close(fig)


def run(data: str | Path, output: str | Path = "outputs") -> dict:
    df = load_and_clean(data)
    ols = fit_explanatory_ols(df)
    _, held_out, actual, predicted = fit_holdout_model(df)
    metrics = {
        "source": "new reconstruction (not the original 2025 notebook)",
        "sample_rows": len(df),
        "in_sample_ols": {
            "r_squared": round(float(ols.rsquared), 4),
            "adjusted_r_squared": round(float(ols.rsquared_adj), 4),
            "f_statistic": round(float(ols.fvalue), 4),
            "f_pvalue": float(ols.f_pvalue),
        },
        "held_out_test": held_out,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "ols_summary.txt").write_text(str(ols.summary()) + "\n", encoding="utf-8")
    plot_diagnostics(df, actual, predicted, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/imdb_top_1000.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
