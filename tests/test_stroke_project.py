"""Small deterministic tests. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stroke_project import choose_f2_threshold, metric_dict, model_pipeline, preprocessors
from analysis_utils import exploratory_report, subgroup_error_table


class TestStrokePipeline(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "age": [20, 24, 35, 42, 47, 51, 60, 67, 72, 80, 28, 38],
            "bmi": [22, np.nan, 25, 26, 27, 24, 30, np.nan, 31, 26, 23, 29],
            "gender": ["Male", "Female"] * 6,
            "hypertension": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            "heart_disease": [0] * 12,
            "Residence_type": ["Urban", "Rural"] * 6,
            "smoking_status": ["Unknown", "never smoked"] * 6,
        })
        self.y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0])

    def test_preprocessor_fits_and_predicts(self):
        from sklearn.linear_model import LogisticRegression
        pipe = model_pipeline(self.X, LogisticRegression(max_iter=2000))
        pipe.fit(self.X, self.y)
        unseen = self.X.iloc[:2].copy()
        unseen.loc[0, "gender"] = "Other"
        probabilities = pipe.predict_proba(unseen)[:, 1]
        self.assertEqual(len(probabilities), 2)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_confusion_matrix_counts(self):
        result = metric_dict([0, 0, 1, 1], [0, 1, 0, 1], [.1, .8, .2, .9])
        self.assertEqual([result[key] for key in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertAlmostEqual(result["recall_stroke"], .5)

    def test_threshold_returns_valid_value(self):
        t = choose_f2_threshold(np.array([0, 1, 0, 1]), np.array([.1, .4, .3, .9]))
        self.assertGreaterEqual(t, 0)
        self.assertLessEqual(t, 1)

    def test_subgroups_cover_all_test_rows(self):
        out = subgroup_error_table(self.X, self.y, np.zeros(len(self.y), dtype=int))
        self.assertEqual(int(out[out.feature == "gender"].n.sum()), len(self.y))
        self.assertEqual(int(out[out.feature == "age_band"].n.sum()), len(self.y))
        self.assertIn("small_positive_sample", out.columns)

    def test_eda_flags_unknown_and_missing_bmi(self):
        df = self.X.copy()
        df["id"] = range(len(df))
        df["stroke"] = self.y
        df["avg_glucose_level"] = 100.
        df["ever_married"] = "No"
        df["work_type"] = "Private"
        report = exploratory_report(df)
        self.assertEqual(report["missing_values"]["bmi"], 2)
        self.assertEqual(report["smoking_status_unknown"], 6)


if __name__ == "__main__":
    unittest.main()
