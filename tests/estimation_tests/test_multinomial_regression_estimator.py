import unittest
import pandas as pd
from causal_testing.estimation.multinomial_regression_estimator import MultinomialRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.specification.variable import Input, Output


class TestMultinomialRegressionEstimator(unittest.TestCase):
    """Test the multinomial regression estimator against the scarf example from
    https://investigate.ai/regression/multinomial-regression/.
    (For binary categories, this should behave the same as for logistic regression)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_odds_ratio(self):
        df = self.scarf_df.copy()
        multinomial_regression_estimator = MultinomialRegressionEstimator(
            BaseTestCase(Input("length_in", float), Output("completed", bool)), 65, 55, set(), df
        )
        effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio()
        self.assertEqual(round(effect_estimate.value.iloc[0], 4), 0.8948)

    def test_odds_ratio_category(self):
        df = self.scarf_df.copy()
        multinomial_regression_estimator = MultinomialRegressionEstimator(
            BaseTestCase(Input("length_in", float), Output("color", bool)), 65, 55, set(), df
        )
        effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio()
        print(effect_estimate.value)
        self.assertTrue(effect_estimate.value.round(4).equals, pd.Series({"grey": 1.0072, "orange": 0.9668}))

    def test_odds_ratio_data(self):
        df = self.scarf_df.copy()
        multinomial_regression_estimator = MultinomialRegressionEstimator(
            BaseTestCase(Input("length_in", float), Output("completed", bool)), 65, 55, set()
        )
        multinomial_regression_estimator.df = df
        effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio()
        self.assertEqual(round(effect_estimate.value.iloc[0], 4), 0.8948)
