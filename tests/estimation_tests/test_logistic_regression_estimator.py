import unittest
import pandas as pd
from causal_testing.estimation.logistic_regression_estimator import LogisticRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output


class TestLogisticRegressionEstimator(unittest.TestCase):
    """Test the logistic regression estimator against the scarf example from
    https://investigate.ai/regression/logistic-regression/.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_odds_ratio(self):
        df = self.scarf_df.copy()
        logistic_regression_estimator = LogisticRegressionEstimator(
            BaseTestCase(Input("length_in", float), Output("completed", bool)), 65, 55, set(), df
        )
        odds, _ = logistic_regression_estimator.estimate_unit_odds_ratio()
        self.assertEqual(round(odds[0], 4), 0.8948)
