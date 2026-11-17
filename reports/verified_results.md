# Verified IMDb regression results

**Verified:** November 17, 2026, against the supplied `imdb_top_1000.csv` (1,000 movies; 16 original columns). These are the results of the **reconstructed** portfolio pipeline. The originally submitted class notebook has not been recovered.

## Reproduced historical regression results

The reconstructed model uses the five numeric predictors `Released_Year`, `No_of_Votes`, `Gross`, `Runtime` (minutes), and `Meta_score`; the target is `IMDB_Rating`. Missing features were median-imputed.

| Full-sample OLS metric | Reproduced | Historical rounded value |
|---|---:|---:|
| R² | **0.4434** | 0.443 |
| Adjusted R² | **0.4406** | 0.441 |
| F-statistic | **158.3518** | 158.4 |
| F-test p-value | **8.70e-124** | 8.70e-124 |

**These match the historical statistics to their reported precision.** An in-sample R² of 0.4434 means the model accounts for approximately 44.34% of the variability in observed IMDb ratings *within the fitted data*. It does **not** mean 44.34% classification accuracy or held-out test accuracy.

## Newly verified predictive performance

A separate scikit-learn linear regression was fit using an **80/20 random split**, with `random_state=42` and training-only median imputation:

| Holdout metric | Verified |
|---|---:|
| Training rows | 800 |
| Test rows | 200 |
| Test R² | **0.2877** |
| Test MAE | **0.1700** rating points |
| Test RMSE | **0.2137** rating points |

The test model's predictions differ from observed IMDb ratings by about **0.17 rating points on average** (absolute error).

## Source data and preparation

- Dataset: [IMDb Dataset of Top 1000 Movies and TV Shows (Kaggle)](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows).
- Original columns: 16; target range: 7.6–9.3.
- Missing values after numeric parsing, before imputation: **1** release year, **169** gross values, and **157** Metascores. Votes and runtime had no missing numeric values.
- Commas in `Gross` and the `min` suffix in `Runtime` are removed during numeric parsing.
- Full-sample OLS produces descriptive/in-sample results; the holdout evaluation does **not** use its fitted imputer.

## Reproduce

Download the dataset into `data/imdb_top_1000.csv`, install `requirements.txt`, then run:

```bash
python -m src.train --data data/imdb_top_1000.csv
```

The script writes metrics, a statsmodels OLS summary, and charts to `outputs/`. The dataset is not committed to the repository; obtain it from Kaggle.

## Interpretive cautions

This is an observational study of the **IMDb Top 1000**, a selected collection rather than a representative random sample of movies. Revenue, critic scores, and vote counts generally aren't available before release, so the model should not be presented as a pre-release prediction system. Coefficients do not imply causality. The code is a reconstruction even though its rounded in-sample results match the earlier reported assignment numbers.
