# IMDb Movie Rating Prediction | Multiple Linear Regression

**Python · Pandas · NumPy · statsmodels · scikit-learn · Matplotlib · Seaborn**

## Overview

Can we explain differences in IMDb ratings using attributes of movies? This project analyzes the **IMDb Top 1000** dataset using exploratory data analysis and multiple linear regression.

> **Reconstruction & verification note:** This repository reconstructs a December 2025 Group 11 course project; the original submitted notebook has not been recovered. On **November 17, 2026**, the reconstructed code was run on the supplied IMDb Top 1000 CSV and **reproduced the historical OLS metrics to the reported precision**. Separate held-out test scores were also calculated; see [verified results](reports/verified_results.md).

## Research question and features

**Target:** `IMDB_Rating`. **Numerical predictors:** `Released_Year`, `No_of_Votes`, `Gross`, `Runtime` (minutes), `Meta_score`. Genre and certificate may be used for exploration but are not part of the five-variable reported OLS specification.

## Verified regression results

The original coursework's rounded **in-sample OLS statistics have been reproduced** using the 1,000-movie dataset and the reconstructed regression pipeline:

| In-sample measure | Reproduced |
|---|---:|
| R² | **0.4434** |
| Adjusted R² | **0.4406** |
| F-statistic | **158.3518** |
| Model F-test p-value | 8.70e-124 |

**New held-out test evaluation:** An 80/20 split (800 training / 200 test rows, `random_state=42`) achieved **test R² = 0.2877**, **test MAE = 0.1700 rating points**, and **test RMSE = 0.2137 rating points**.

In-sample R² describes explained variation in the fitted data; it is **not held-out prediction accuracy**. Read the [full verified-results report](reports/verified_results.md) and the [machine-readable metrics](reports/verified_metrics.json).

## Repository contents

- `notebooks/imdb_rating_prediction.ipynb`: exploratory notebook for loading, visualizing, and modeling IMDb data
- `src/train.py`: CSV cleaning, OLS analysis, 80/20 holdout regression evaluation, and diagnostic plots
- `tests/test_pipeline.py`: synthetic-data smoke test (not validation of the original results)
- `reports/original_results.md`: historical reported metrics and their provenance
- `reports/verified_results.md`: reproduced OLS statistics, holdout validation, dataset handling, and caveats
- `reports/verified_metrics.json`: machine-readable verified metrics
- `data/README.md`: instructions for obtaining the dataset
- `requirements.txt`: Python dependencies

## Dataset and setup

Download [IMDb Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) and save its CSV as `data/imdb_top_1000.csv`. The CSV is deliberately not bundled.

Requires Python 3.10+. From the repository root:

```bash
python -m venv .venv
# Activate your virtual environment, then:
pip install -r requirements.txt
python -m src.train --data data/imdb_top_1000.csv
python -m unittest discover -s tests -v
```

Run the notebook from the repository root in Jupyter/VS Code. Training writes `outputs/metrics.json`, an OLS summary, and charts (distribution, held-out actual vs. predicted, residuals). Generated outputs are gitignored.

## Methodology

1. Load data; clean year, votes, revenue, runtime, and critic scores as numeric variables.
2. Inspect distributions, summary statistics, missingness, and correlations.
3. Median-impute predictors and fit a **full-sample OLS** model for exploratory interpretation.
4. Separately use an **80/20 holdout split** with training-only median imputation and fit a scikit-learn Linear Regression model.
5. Evaluate held-out R², mean absolute error, and root mean squared error; inspect charts and limitations.

## Limitations

- The rounded in-sample statistics were reproduced on the supplied CSV, but the original submitted notebook remains unavailable. Results may change with a different dataset version or preprocessing.
- IMDb Top 1000 is a selected high-rated sample, not a random sample of every movie.
- Variables such as votes, gross revenue, and critic scores can be observed only after release; this is explanatory modeling, not a trustworthy pre-release forecasting system.
- Consider transformation of skewed predictors, tests of model assumptions, multicollinearity checks, cross-validation, and alternative regression models for future work.

## Academic context

Adapted for a personal portfolio from a December 2025 course group project (Group 11). This repository is a **reconstruction**, not a copy of the originally submitted code.
