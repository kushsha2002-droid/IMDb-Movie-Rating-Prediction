# Dataset instructions

Download the [IMDb Top 1000 dataset](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), then place its CSV here as `imdb_top_1000.csv`.

The original course project files were unavailable during portfolio reconstruction; the dataset is not bundled. Required columns: `IMDB_Rating`, `Released_Year`, `No_of_Votes`, `Gross`, `Runtime`, `Meta_score`.

Run from the repository root using `python -m src.train --data data/imdb_top_1000.csv`.
