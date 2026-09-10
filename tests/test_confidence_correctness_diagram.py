import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from confidence_correctness_diagram import (
    confidence_correctness_analysis,
    save_confidence_correctness_diagram,
)


class ConfidenceCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.classes = np.array(["A", "B", "C"])
        self.y_true = np.array(["A", "A", "B", "B", "C", "C"])
        self.y_pred = np.array(["A", "B", "B", "B", "C", "A"])
        self.proba = np.array([
            [0.80, 0.10, 0.10],
            [0.30, 0.60, 0.10],
            [0.10, 0.80, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.60, 0.20, 0.20],
        ])

    def test_global_decomposition_sums_to_one(self):
        metrics, _, _ = confidence_correctness_analysis(
            self.y_true, self.y_pred, self.proba, self.classes
        )
        self.assertAlmostEqual(
            metrics["R"] + metrics["O"] + metrics["U"] + metrics["A"],
            1.0,
            places=14,
        )
        self.assertAlmostEqual(metrics["lambda_h"] + metrics["lambda_l"], 1.0, places=14)

    def test_each_class_profile_sums_to_one(self):
        _, class_df, _ = confidence_correctness_analysis(
            self.y_true, self.y_pred, self.proba, self.classes
        )
        profile_sum = class_df[
            [
                "R_alpha_row_norm",
                "O_alpha_row_norm",
                "U_alpha_row_norm",
                "A_alpha_row_norm",
            ]
        ].sum(axis=1)
        np.testing.assert_allclose(profile_sum.to_numpy(), np.ones(len(class_df)), atol=1e-14)

    def test_matrix_identities(self):
        _, _, matrices = confidence_correctness_analysis(
            self.y_true, self.y_pred, self.proba, self.classes
        )
        np.testing.assert_allclose(matrices["C_star"], matrices["H"] + matrices["L"], atol=1e-14)
        np.testing.assert_allclose(matrices["C_norm"], matrices["C_star"] / len(self.y_true), atol=1e-14)

    def test_known_v4_values(self):
        metrics, class_df, _ = confidence_correctness_analysis(
            self.y_true, self.y_pred, self.proba, self.classes
        )
        expected = {
            "R": 0.5,
            "O": 0.2,
            "U": 13.0 / 60.0,
            "A": 1.0 / 12.0,
            "acc_star": 7.0 / 12.0,
            "lambda_h": 0.7,
            "lambda_l": 0.3,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=14, msg=key)

        self.assertEqual(list(class_df.columns), [
            "class", "n_class", "R_alpha", "O_alpha", "U_alpha", "A_alpha",
            "R_alpha_row_norm", "O_alpha_row_norm", "U_alpha_row_norm",
            "A_alpha_row_norm", "hard_sensitivity", "hard_precision",
            "prob_sensitivity", "prob_precision",
        ])

    def test_convenience_function_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagram.png"
            _, class_df, _ = save_confidence_correctness_diagram(
                self.y_true,
                self.proba,
                self.classes,
                path,
                title="test",
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            self.assertIsInstance(class_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
