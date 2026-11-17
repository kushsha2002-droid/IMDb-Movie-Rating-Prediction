"""Synthetic-data tests for reconstructed regression workflow, not original metrics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import FEATURES, fit_explanatory_ols, fit_holdout_model, load_and_clean, run


class PipelineSmokeTest(unittest.TestCase):
    def test_coercion_and_end_to_end(self):
        rng = np.random.default_rng(7)
        n = 100
        votes = rng.integers(20000, 500000, n)
        metascores = rng.integers(40, 95, n)
        ratings = 5.4 + 0.025 * metascores + rng.normal(0, 0.25, n)
        df = pd.DataFrame({
            "IMDB_Rating": ratings,
            "Released_Year": rng.integers(1960, 2022, n).astype(str),
            "No_of_Votes": votes,
            "Gross": [f"{v:,}" for v in rng.integers(10000, 200000000, n)],
            "Runtime": [f"{v} min" for v in rng.integers(70, 190, n)],
            "Meta_score": metascores.astype(float),
        })
        df.loc[0, "Gross"] = None
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "imdb_top_1000.csv"
            df.to_csv(csv_path, index=False)
            cleaned = load_and_clean(csv_path)
            self.assertEqual(len(cleaned), n)
            self.assertEqual(list(cleaned[FEATURES].columns), FEATURES)
            self.assertEqual(cleaned["Runtime"].iloc[0], int(df["Runtime"].iloc[0].split()[0]))
            ols = fit_explanatory_ols(cleaned)
            self.assertTrue(np.isfinite(ols.rsquared))
            _, holdout, actual, predicted = fit_holdout_model(cleaned)
            self.assertEqual(len(actual), 20)
            self.assertEqual(len(predicted), 20)
            self.assertIn("test_rmse", holdout)
            result = run(csv_path, Path(directory) / "outputs")
            self.assertEqual(result["sample_rows"], n)
            self.assertTrue((Path(directory) / "outputs" / "metrics.json").exists())
            self.assertTrue((Path(directory) / "outputs" / "residuals.png").exists())


if __name__ == "__main__":
    unittest.main()
