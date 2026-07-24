import unittest
import pandas as pd
from causal_testing.estimation.multinomial_regression_estimator import MultinomialRegressionEstimator


class TestMultinomialRegressionEstimator(unittest.TestCase):
    """Test the multinomial regression estimator against the scarf example from
    https://investigate.ai/regression/multinomial-regression/.
    (For binary categories, this should behave the same as for logistic regression)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scarf_df = pd.read_csv("tests/resources/data/scarf_data.csv")

    def test_odds_ratio(self):
        multinomial_regression_estimator = MultinomialRegressionEstimator(
            treatment_variable="length_in",
            outcome_variable="completed",
            control_value=65,
            treatment_value=55,
            adjustment_set=set(),
        )
        effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio(self.scarf_df)
        self.assertEqual(round(effect_estimate.value.iloc[0], 4), 0.8948)

    def test_odds_ratio_category(self):
        multinomial_regression_estimator = MultinomialRegressionEstimator(
            treatment_variable="length_in",
            outcome_variable="color",
            control_value=65,
            treatment_value=55,
            adjustment_set=set(),
        )
        effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio(self.scarf_df)
        self.assertTrue(effect_estimate.value.round(4).equals, pd.Series({"grey": 1.0072, "orange": 0.9668}))

    def test_odds_ratio_data(self):
        multinomial_regression_estimator = MultinomialRegressionEstimator(
            treatment_variable="length_in",
            outcome_variable="completed",
            control_value=65,
            treatment_value=55,
            adjustment_set=set(),
        )
        effect_estimate = multinomial_regression_estimator.estimate_unit_odds_ratio(self.scarf_df)
        self.assertEqual(round(effect_estimate.value.iloc[0], 4), 0.8948)
