import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_testing.specification.variable import Input
from causal_testing.utils.validation import CausalValidator
from causal_testing.specification.capabilities import TreatmentSequence
from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator


class TestLogisticRegressionEstimator(unittest.TestCase):
    """Test the logistic regression estimator against the scarf example from
    https://investigate.ai/regression/logistic-regression/.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_ate(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        ate, _ = logistic_regression_estimator.estimate_ate()
        self.assertEqual(round(ate, 4), -0.1987)

    def test_risk_ratio(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        rr, _ = logistic_regression_estimator.estimate_risk_ratio()
        self.assertEqual(round(rr, 4), 0.7664)

    def test_odds_ratio(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, set(), "completed", df)
        odds = logistic_regression_estimator.estimate_unit_odds_ratio()
        self.assertEqual(round(odds, 4), 0.8948)

    def test_ate_adjustment(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator(
            "length_in", 65, 55, {"large_gauge"}, "completed", df
        )
        ate, _ = logistic_regression_estimator.estimate_ate(adjustment_config={"large_gauge": 0})
        self.assertEqual(round(ate, 4), -0.3388)

    def test_ate_invalid_adjustment(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator("length_in", 65, 55, {}, "completed", df)
        with self.assertRaises(ValueError):
            ate, _ = logistic_regression_estimator.estimate_ate(adjustment_config={"large_gauge": 0})

    def test_ate_effect_modifiers(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator(
            "length_in", 65, 55, set(), "completed", df, effect_modifiers={"large_gauge": 0}
        )
        ate, _ = logistic_regression_estimator.estimate_ate()
        self.assertEqual(round(ate, 4), -0.3388)

    def test_ate_effect_modifiers_formula(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator(
            "length_in",
            65,
            55,
            set(),
            "completed",
            df,
            effect_modifiers={"large_gauge": 0},
            formula="completed ~ length_in + large_gauge",
        )
        ate, _ = logistic_regression_estimator.estimate_ate()
        self.assertEqual(round(ate, 4), -0.3388)
